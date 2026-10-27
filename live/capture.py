"""
Live system call capture. Every source yields the same events:

    ("S", tid, tgid, nr, comm)   task tid (thread of process tgid) entered system call nr (x86_64 numbering).
                                 tgid / comm are None when the backend does not report them.
    ("X", tid, None, None, None) task tid exited
    ("E", tid, None, None, comm) task tid finished a successful execve and now runs program comm   (strace only)
    ("C", child, parent, is_thread, None)  parent created task child                            (strace only)

Backends
  ebpf    bpftrace program attached to the raw_syscalls:sys_enter tracepoint (in-kernel, needs root).
          Follows a process tree (forks are tracked inside the kernel) or the whole host.
  strace  ptrace(2): the kernel stops the traced task on every system call and wakes strace. Works without root for a
          command it starts itself. Much slower, because every call costs two extra context switches.
  replay  re-reads a saved strace (-f) or bpftrace output file; used for tests.
"""
import os, re, shutil, subprocess, tempfile, time
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

def _gated(cmd):
    """Start cmd held at a gate (a shell blocked on read) so the tracer can attach to its pid before it runs."""
    r, w = os.pipe()
    # the read end keeps its own number in the child (pass_fds); close_fds would undo a dup2 done in preexec_fn
    p = subprocess.Popen(["sh", "-c", f'read _ <&{r}; exec {r}<&-; exec "$@"', "sh", *cmd], pass_fds=(r,))
    os.close(r)
    def release():
        os.write(w, b"go\n"); os.close(w)
    return p, release

def ebpf_source(cmd=None, pid=0):
    """Stream events from the bpftrace program. cmd: start and follow this command; pid: follow an existing process
    tree; neither: whole host."""
    bt = shutil.which("bpftrace")
    if not bt: raise SystemExit("bpftrace not found: sudo apt install bpftrace")
    if os.geteuid() != 0: raise SystemExit("the eBPF backend must run as root (sudo -E python3 live/monitor.py ...)")
    child, release = (None, None)
    if cmd: child, release = _gated(cmd); pid = child.pid
    tracer = subprocess.Popen([bt, "-B", "line", BT_SCRIPT, str(pid), str(os.getpid())], stdout=subprocess.PIPE, text=True, errors="replace")
    ready = False
    try:
        for line in tracer.stdout:
            if not ready:
                if line.startswith("Attaching"):      # bpftrace prints this once the probes are loaded
                    ready = True; time.sleep(0.2)
                    if release: release()
                continue
            evs = parse_bpftrace(line)
            yield from evs
            if child is not None and evs and evs[0][0] == "X" and evs[0][1] == child.pid:
                break                                 # followed program finished
        if not ready:
            raise SystemExit(f"bpftrace exited before attaching its probes (code {tracer.wait()}); see its error above")
    finally:
        tracer.terminate(); tracer.wait()
        if child is not None:
            if not ready: child.kill()                # never let the program run unwatched
            child.wait()

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
