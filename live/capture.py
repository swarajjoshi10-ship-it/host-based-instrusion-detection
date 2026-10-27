"""
Live system call capture. Every source yields the same events:

    ("S", tid, tgid, nr, comm)   task tid (thread of process tgid) entered system call nr (x86_64 numbering).
                                 tgid / comm are None when the backend does not report them.
    ("X", tid, None, None, None) task tid exited
    ("E", tid, None, None, comm) task tid finished a successful execve and now runs program comm   (strace only)
    ("C", child, parent, is_thread, None)  parent created task child                            (strace only)

Backends
  ebpf    bpftrace program attached to the raw_syscalls:sys_enter tracepoint (in-kernel, needs root).
          Follows a process tree placed in its own cgroup (children inherit it in the kernel), or the whole host.
  strace  ptrace(2): the kernel stops the traced task on every system call and wakes strace. Works without root for a
          command it starts itself. Much slower, because every call costs two extra context switches.
  replay  re-reads a saved strace (-f) or bpftrace output file; used for tests.
"""
import os, pwd, re, shutil, subprocess, tempfile, threading, time
from syscalls import name_to_x64

HERE = os.path.dirname(os.path.abspath(__file__))
BT_SCRIPT = os.path.join(HERE, "ebpf", "syscalls.bt")

# ------------------------------------------------------------------ parsers (shared by live sources and replay)
_STRACE_LINE = re.compile(r"^(\d+)\s+(.*)$")
_STRACE_CALL = re.compile(r"^([a-z_0-9]+)\(")
_RESUMED = re.compile(r"^<\.\.\. ([a-z_0-9]+) resumed>")
_RESULT = re.compile(r"\)\s+= (-?\d+)")
_SPAWN = ("clone", "clone3", "fork", "vfork")

class StraceParser:
    """`strace -f` output -> events. A call that another task interrupts is split into an "<unfinished ...>" line and
    a "<... resumed>" line; it is counted once, at entry, and its result is read from whichever line has it.
    strace prints after the fact, so program names come from execve/clone in the trace, not from /proc."""
    def __init__(self):
        self.open = {}                                          # tid -> (call name, exec path / thread flag) unfinished

    def _done(self, tid, name, info, rest):
        r = _RESULT.search(rest)
        if not r: return []
        val = int(r.group(1))
        if name == "execve" and val == 0: return [("E", tid, None, None, os.path.basename(info)[:15])]
        if name in _SPAWN and val > 0: return [("C", val, tid, bool(info), None)]
        return []

    def __call__(self, line):
        m = _STRACE_LINE.match(line)
        if not m: return []
        tid, rest = int(m.group(1)), m.group(2)
        if rest.startswith("+++"): return [("X", tid, None, None, None)]     # "+++ exited with 0 +++" / killed
        if rest.startswith("---"): return []                                  # signal delivery
        r = _RESUMED.match(rest)
        if r:
            name, info = self.open.pop(tid, (r.group(1), None))
            return self._done(tid, name, info, rest)
        c = _STRACE_CALL.match(rest)
        if not c: return []
        name = c.group(1)
        nr = int(name[10:], 16) if name.startswith("syscall_0x") else name_to_x64(name)
        info = rest.split('"')[1] if name == "execve" and '"' in rest else ("CLONE_THREAD" in rest if name in _SPAWN else None)
        ev = [("S", tid, None, nr, None)]
        if rest.rstrip().endswith("<unfinished ...>"): self.open[tid] = (name, info)
        else: ev += self._done(tid, name, info, rest)
        return ev

def parse_bpftrace(line):
    if line.startswith("S "):
        p = line.rstrip("\n").split(" ", 4)
        if len(p) < 4: return []
        return [("S", int(p[2]), int(p[1]), int(p[3]), p[4] if len(p) > 4 else None)]
    if line.startswith("X "):
        return [("X", int(line.split()[1]), None, None, None)]
    if line.startswith("Lost "):                                              # perf ring buffer overflowed
        return [("LOST", int(line.split()[1]), None, None, None)]
    return []

# ------------------------------------------------------------------ sources
def strace_source(cmd):
    """Start cmd under strace -f and stream its events. strace writes to a FIFO so the traced program keeps its own
    stdout/stderr and does not inherit the pipe (strace opens -o files close-on-exec)."""
    if not shutil.which("strace"): raise SystemExit("strace not found: sudo apt install strace")
    tmp = tempfile.mkdtemp(prefix="hids_"); fifo = os.path.join(tmp, "events")
    os.mkfifo(fifo)
    proc = subprocess.Popen(["strace", "-f", "-q", "-e", "signal=none", "-s", "0", "-o", fifo, "--", *cmd])
    try:
        parse = StraceParser()
        with open(fifo, "r", errors="replace") as fh:
            for line in fh:
                yield from parse(line)
    finally:
        proc.wait(); shutil.rmtree(tmp, ignore_errors=True)

def sudo_user():
    """(uid, gid) of the user who ran sudo, or None when not running under sudo"""
    if os.geteuid() == 0 and "SUDO_UID" in os.environ:
        return int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
    return None

def _gated(cmd, user=None):
    """Start cmd held at a gate (a shell blocked on read) so the tracer can attach before it runs.
    user=(uid, gid): drop root privileges for the program; the monitor needs root, the watched program must not get it."""
    exe = shutil.which(cmd[0]) or cmd[0]                      # absolute path: one execve, no PATH search by the gate
    kw = {}
    if user:
        uid, gid = user
        kw = dict(user=uid, group=gid, extra_groups=os.getgrouplist(pwd.getpwuid(uid).pw_name, gid))
    r, w = os.pipe()
    # the read end keeps its own number in the child (pass_fds); close_fds would undo a dup2 done in preexec_fn
    p = subprocess.Popen(["/bin/sh", "-c", f'read _ <&{r}; exec {r}<&-; exec "$@"', "sh", exe, *cmd[1:]], pass_fds=(r,), **kw)
    os.close(r)
    def release():
        os.write(w, b"go\n"); os.close(w)
    return p, release

EXECVE = 59                                                   # x86_64

def bpftrace_env():
    """bpftrace's per-CPU perf ring buffer defaults to 64 pages (256 KB). When it is full the kernel drops events
    instead of slowing the program, so syscall bursts are lost; 1024 pages (4 MB per CPU) absorbs far more."""
    return dict(os.environ, BPFTRACE_PERF_RB_PAGES=os.environ.get("BPFTRACE_PERF_RB_PAGES", "1024"))

class Cgroup:
    """A cgroup v2 directory used as a container for the watched process tree."""
    ROOT = "/sys/fs/cgroup"

    def __init__(self, name):
        self.path = os.path.join(self.ROOT, name)
        os.mkdir(self.path)
        self.id = os.stat(self.path).st_ino                  # = the id bpftrace's `cgroup` builtin returns

    def _write(self, f, v):
        with open(os.path.join(self.path, f), "w") as fh: fh.write(f"{v}\n")

    def add(self, pid): self._write("cgroup.procs", pid)   # moves the whole process (all its threads)
    def pids(self): return [int(x) for x in open(os.path.join(self.path, "cgroup.procs")).read().split()]
    def frozen(self): return "frozen 1" in open(os.path.join(self.path, "cgroup.events")).read()
    def freeze(self): self._write("cgroup.freeze", 1)      # every task stops, like SIGSTOP but uncatchable as a group
    def kill(self): self._write("cgroup.kill", 1)          # SIGKILL to every task in the cgroup (Linux >= 5.14)
    def remove(self):
        try: os.rmdir(self.path)
        except OSError: pass

def _descendants(pid):
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        for t in os.listdir(f"/proc/{p}/task"):
            try: kids = [int(k) for k in open(f"/proc/{p}/task/{t}/children").read().split()]
            except OSError: kids = []
            out += kids; todo += kids
    return out

class EbpfSource:
    """Events from the bpftrace program. cmd: start and follow this command; pid: follow a running process and its
    descendants; whole_host: every task on the machine.

    The followed tree lives in its own cgroup, so the kernel filters by cgroup id. pid/tid in the events are
    kernel-global; inside a pid namespace (containers, WSL2) they differ from the pids user space sees, which is why
    the response acts on the cgroup (freeze/kill) instead of calling kill() on a pid."""

    def __init__(self, cmd=None, pid=0, whole_host=False, as_root=False):
        self.bt = shutil.which("bpftrace")
        if not self.bt: raise SystemExit("bpftrace not found: sudo apt install bpftrace")
        if os.geteuid() != 0: raise SystemExit("the eBPF backend must run as root (sudo -E python3 live/monitor.py ...)")
        if not os.path.exists(os.path.join(Cgroup.ROOT, "cgroup.procs")):
            raise SystemExit("cgroup v2 is not mounted at /sys/fs/cgroup")
        self.child = self.release = self.jail = self.self_cg = None
        tag = f"hids-{os.getpid()}-{time.monotonic_ns() % 10 ** 6}"
        if whole_host:
            self.home = Cgroup.ROOT + open("/proc/self/cgroup").read().strip().split("::", 1)[1]
            self.self_cg = Cgroup(tag + "-monitor"); self.self_cg.add(os.getpid())   # bpftrace inherits it too
            self.args = (0, self.self_cg.id)
        else:
            self.jail = Cgroup(tag)
            if cmd:
                self.child, self.release = _gated(cmd, None if as_root else sudo_user())
                self.jail.add(self.child.pid)                 # placed in the cgroup before it can run
            else:
                for p in [pid] + _descendants(pid): self.jail.add(p)
            self.args = (self.jail.id, 0)

    def respond(self, action):
        if self.jail is None: return "no response in whole-host mode (no cgroup to act on)"
        if action == "stop":
            self.jail.freeze()
            return f"froze cgroup {self.jail.path} (resume: echo 0 > {self.jail.path}/cgroup.freeze)"
        self.jail.kill()
        return f"killed every process in cgroup {self.jail.path}"

    def _watch(self, tracer):
        """end the stream once the followed tree is gone (or frozen by the response)"""
        while tracer.poll() is None:
            try: done = (self.child is None or self.child.poll() is not None) and not self.jail.pids()
            except OSError: done = True
            if done or self.jail.frozen():
                time.sleep(0.3)                               # let bpftrace print the last events
                tracer.terminate(); return
            time.sleep(0.05)

    def __iter__(self):
        tracer = subprocess.Popen([self.bt, "-B", "line", BT_SCRIPT, *map(str, self.args)], env=bpftrace_env(),
                                  stdout=subprocess.PIPE, text=True, errors="replace")
        ready = False
        gate, gate_tid = self.child is not None, None
        try:
            for line in tracer.stdout:
                if not ready:
                    if line.startswith("Attaching"):      # bpftrace prints this once the probes are loaded
                        ready = True; time.sleep(0.2)
                        if self.jail: threading.Thread(target=self._watch, args=(tracer,), daemon=True).start()
                        if self.release: self.release()
                    continue
                for ev in parse_bpftrace(line):
                    if gate and ev[0] == "S":
                        # the gate shell's own calls (read, close, ...) are not the program's: start at its execve,
                        # exactly where a trace started by strace begins
                        gate_tid = gate_tid or ev[1]
                        if ev[1] == gate_tid:
                            if ev[3] != EXECVE: continue
                            gate = False
                    yield ev
            if not ready:
                raise SystemExit(f"bpftrace exited before attaching its probes (code {tracer.wait()}); see its error above")
        finally:
            tracer.terminate(); tracer.wait()
            frozen = self.jail is not None and self.jail.frozen()
            if self.child is not None:
                if not ready: self.child.kill()             # never let the program run unwatched
                if not frozen: self.child.wait()
            if self.jail is not None and not frozen: self.jail.remove()
            if self.self_cg is not None:
                with open(os.path.join(self.home, "cgroup.procs"), "w") as fh: fh.write(f"{os.getpid()}\n")
                self.self_cg.remove()

def replay_source(path, fmt):
    parse = StraceParser() if fmt == "strace" else parse_bpftrace
    with open(path, errors="replace") as fh:
        for line in fh:
            yield from parse(line)

# ------------------------------------------------------------------ /proc helpers
def proc_info(tid):
    """(tgid, comm) of a live task, read from /proc; (None, '?') once it is gone."""
    try:
        with open(f"/proc/{tid}/status") as fh:
            st = dict(l.split(":", 1) for l in fh if ":" in l)
        return int(st["Tgid"]), st["Name"].strip()
    except (OSError, KeyError, ValueError):
        return None, "?"
