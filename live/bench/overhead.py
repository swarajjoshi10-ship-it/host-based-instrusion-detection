"""
Monitoring overhead: how much slower do programs run while they are being watched?

Each workload is timed (wall clock, measured inside the traced tree so monitor start-up is excluded) under:
  none            no monitoring
  strace          ptrace capture only, output discarded         (cost of the capture mechanism)
  monitor-strace  full pipeline: strace -> parser -> LSTM -> alerts
  ebpf            bpftrace capture only, output discarded       (--ebpf, root)
  monitor-ebpf    full pipeline with eBPF capture               (--ebpf, root)

  python3 live/bench/overhead.py                      # strace conditions (no root)
  sudo -E python3 live/bench/overhead.py --ebpf       # adds the eBPF conditions
Writes live/results/overhead.csv and prints a table (median of --repeats runs, slowdown relative to none).
"""
import argparse, csv, os, re, shlex, shutil, statistics, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__)); LIVE = os.path.dirname(HERE); ROOT = os.path.dirname(LIVE)
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(LIVE, "models", "host.npz"))
ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--ebpf", action="store_true")
ap.add_argument("--storm", type=int, default=10000, help="iterations of the syscall storm (5 calls each)")
a = ap.parse_args()

tmp = tempfile.mkdtemp(prefix="hids_bench_")
storm = os.path.join(tmp, "storm")
subprocess.run(["gcc", "-O2", "-o", storm, os.path.join(HERE, "storm.c")], check=True)
blob = os.path.join(tmp, "blob")                             # ~32 MB of real binaries for the compression workload
with open(blob, "wb") as out:
    for d in ("/usr/bin", "/usr/lib/x86_64-linux-gnu"):
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p) and not os.path.islink(p) and out.tell() < 32 << 20:
                with open(p, "rb") as fh: out.write(fh.read(1 << 20))
service = os.path.join(LIVE, "demo", "service.sh")
WORKLOADS = {
    "syscall storm": f"{shlex.quote(storm)} {a.storm}",
    "report service x10": f"for i in 1 2 3 4 5 6 7 8 9 10; do bash {shlex.quote(service)} $i; done",
    "gzip 32 MB": f"gzip -1 -c {shlex.quote(blob)} > /dev/null",
}
BT = os.path.join(LIVE, "ebpf", "syscalls.bt")
mon = [sys.executable, os.path.join(LIVE, "monitor.py"), "--model", a.model, "--quiet", "--log", os.path.join(tmp, "alerts.jsonl")]
CONDITIONS = {
    "none": lambda c: c,
    "strace": lambda c: ["strace", "-f", "-qq", "-o", "/dev/null", "--", *c],
    "monitor-strace": lambda c: [*mon, "--backend", "strace", "--", *c],
}
if a.ebpf:
    if os.geteuid() != 0: sys.exit("--ebpf needs root: sudo -E python3 live/bench/overhead.py --ebpf")
    CONDITIONS["ebpf"] = None                                 # handled below: tracer runs beside the workload
    CONDITIONS["monitor-ebpf"] = lambda c: [*mon, "--backend", "ebpf", "--", *c]

def timed(workload, stamp):
    """the workload wrapped so it records its own start and end time"""
    return ["bash", "-c", f'date +%s.%N > {stamp}; {workload}; date +%s.%N >> {stamp}']

def run_once(cond, workload):
    stamp = os.path.join(tmp, "stamp"); cmd = timed(workload, stamp); info = {}
    if cond == "ebpf":
        # gate the workload so bpftrace follows its pid from the first call
        r, w = os.pipe()
        child = subprocess.Popen(["sh", "-c", 'read _ <&3; exec 3<&-; exec "$@"', "sh", *cmd],
                                 preexec_fn=lambda: os.dup2(r, 3), pass_fds=(r,), stdout=subprocess.DEVNULL)
        os.close(r)
        tr = subprocess.Popen(["bpftrace", "-B", "full", BT, str(child.pid), str(os.getpid())], stdout=subprocess.PIPE, text=True)
        tr.stdout.readline(); time.sleep(0.5); os.write(w, b"go\n"); os.close(w)
        child.wait(); tr.terminate(); lost = sum(int(l.split()[1]) for l in tr.stdout if l.startswith("Lost "))
        tr.wait(); info["lost"] = lost
    else:
        p = subprocess.run(CONDITIONS[cond](cmd), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        m = re.search(r"monitor CPU ([\d.]+) s .*peak RSS ([\d.]+) MB", p.stdout)
        if m: info["monitor_cpu_s"], info["monitor_rss_mb"] = float(m.group(1)), float(m.group(2))
        m = re.search(r"\] (\d+) system calls", p.stdout)
        if m: info["syscalls"] = int(m.group(1))
        m = re.search(r"(\d+) events LOST", p.stdout)
        if m: info["lost"] = int(m.group(1))
    t0, t1 = map(float, open(stamp).read().split())
    return t1 - t0, info

rows = []
for wname, wl in WORKLOADS.items():
    base = None
    for cond in CONDITIONS:
        times, infos = [], []
        for _ in range(a.repeats):
            t, info = run_once(cond, wl); times.append(t); infos.append(info)
        med = statistics.median(times)
        if cond == "none": base = med
        row = dict(workload=wname, condition=cond, median_s=round(med, 3), slowdown=round(med / base, 2),
                   min_s=round(min(times), 3), max_s=round(max(times), 3), repeats=a.repeats)
        for k in ("syscalls", "monitor_cpu_s", "monitor_rss_mb", "lost"):
            v = [i[k] for i in infos if k in i]
            if v: row[k] = round(statistics.median(v), 2)
        rows.append(row)
        print(f"{wname:20s} {cond:15s} {med:8.3f} s  x{med / base:6.2f}" + "".join(f"  {k}={row[k]}" for k in ("syscalls", "monitor_cpu_s", "monitor_rss_mb", "lost") if k in row), flush=True)

out = os.path.join(LIVE, "results", "overhead.csv"); os.makedirs(os.path.dirname(out), exist_ok=True)
keys = ["workload", "condition", "median_s", "slowdown", "min_s", "max_s", "repeats", "syscalls", "monitor_cpu_s", "monitor_rss_mb", "lost"]
mode = "a" if os.path.exists(out) and a.ebpf else "w"            # the root run adds its rows to the strace run's file
with open(out, mode, newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    if mode == "w": w.writeheader()
    w.writerows([r for r in rows if mode == "w" or r["condition"].endswith("ebpf")])
shutil.rmtree(tmp, ignore_errors=True)
print(f"\nsaved {os.path.relpath(out, ROOT)}")
