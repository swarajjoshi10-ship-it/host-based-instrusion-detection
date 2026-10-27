"""
Real-time host intrusion monitor.

Captures system calls (eBPF or strace), keeps a sliding window of the last L calls per task (thread), scores every
window with the LSTM next-call model, and raises an alert when a task's running mean score exceeds the threshold
learned from normal traces. Optionally stops or kills the offending process.

  # follow one command (no root needed with strace)
  python3 live/monitor.py --model live/models/host.npz --backend strace -- bash live/demo/service.sh
  # same with eBPF, kernel-side capture
  sudo -E python3 live/monitor.py --model live/models/host.npz --backend ebpf -- bash live/demo/service.sh
  # watch a running process tree / the whole host (eBPF)
  sudo -E python3 live/monitor.py --model live/models/host.npz --backend ebpf --pid 1234
  sudo -E python3 live/monitor.py --model live/models/host.npz --backend ebpf --all
  # record normal traces for training (no model needed)
  python3 live/monitor.py --backend strace --record live/recordings/normal_train -- bash live/demo/service.sh
"""
import argparse, json, os, resource, signal, sys, time
from collections import deque
import capture
from syscalls import convert, name_of


class Task:
    __slots__ = ("tid", "tgid", "comm", "calls", "raw", "n", "nwin", "sum", "worst", "worst_win", "alerted", "stale")
    def __init__(self, tid, L):
        self.tid, self.tgid, self.comm = tid, None, None
        self.calls = deque(maxlen=L); self.raw = []           # model-numbered window / full x86_64 trace (record)
        self.n = self.nwin = 0; self.sum = 0.0; self.worst = -1.0; self.worst_win = None
        self.alerted = False; self.stale = True
    @property
    def mean(self): return self.sum / self.nwin if self.nwin else 0.0

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--backend", choices=["ebpf", "strace", "replay"], default="strace")
    ap.add_argument("--model", help="detector saved by train.py (omit to only --record)")
    ap.add_argument("--threshold", default="live_p99", help="live_p99 | live_p95 (calibrated for this live rule) or a number")
    ap.add_argument("--min_windows", type=int, help="windows a task must produce before it can be flagged while running (default: the model's)")
    ap.add_argument("--batch", type=int, default=16, help="windows scored together: larger = less CPU, later alerts")
    ap.add_argument("--respond", choices=["none", "stop", "kill"], default="none", help="SIGSTOP or SIGKILL a flagged process")
    ap.add_argument("--record", help="folder to write one trace file per task (x86_64 numbers, ADFA-LD format)")
    ap.add_argument("--tag", default="run", help="file name prefix for --record")
    ap.add_argument("--pid", type=int, default=0, help="ebpf: follow this running process and its descendants")
    ap.add_argument("--all", action="store_true", help="ebpf: whole host")
    ap.add_argument("--replay", help="replay a saved strace/bpftrace output file"); ap.add_argument("--replay_format", default="strace")
    ap.add_argument("--log", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "alerts.jsonl"))
    ap.add_argument("--quiet", action="store_true", help="no per-task summary table")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not a.model and not a.record: ap.error("give --model (detect) and/or --record (collect traces)")

    det = None; L = 20; S = 10; thr = None
    if a.model:
        from model import Detector                           # torch import only when detecting
        det = Detector.load(a.model); L, S = det.L, det.S
        thr = det.thresholds[a.threshold] if a.threshold in det.thresholds else float(a.threshold)
        if a.min_windows is None: a.min_windows = det.min_windows
        print(f"[hids] model {a.model}: numbering {det.numbering}, window {L}/{S}, threshold {a.threshold} = {thr:.3f}", flush=True)

    if a.backend == "strace":
        if not cmd: ap.error("strace backend needs a command after --")
        src = capture.strace_source(cmd)
    elif a.backend == "ebpf":
        if not (cmd or a.pid or a.all): ap.error("ebpf backend needs a command, --pid or --all")
        src = capture.ebpf_source(cmd=cmd or None, pid=a.pid)
    else:
        src = capture.replay_source(a.replay, a.replay_format)
    live = a.backend != "replay"

    tasks, done, alerts = {}, [], []
    pending = []                                              # (task, window) waiting to be scored in one batch
    stats = dict(events=0, windows=0, score_s=0.0, lost=0)
    os.makedirs(os.path.dirname(a.log), exist_ok=True); logf = open(a.log, "a")
    if a.record: os.makedirs(a.record, exist_ok=True)

    def ident(t):
        if t.stale and live:
            tg, comm = capture.proc_info(t.tid)
            if comm != "?": t.tgid, t.comm = tg, comm
            t.stale = False

    def alert(t, when):
        t.alerted = True
        top = [name_of(c, det.numbering) for c in (t.worst_win or [])]
        rec = dict(time=time.strftime("%H:%M:%S"), tid=t.tid, pid=t.tgid, comm=t.comm, score=round(t.mean, 3),
                   threshold=round(thr, 3), windows=t.nwin, calls=t.n, when=when, worst_window=top, response=a.respond)
        act = ""
        if when == "running" and a.respond != "none" and t.tgid and t.tgid != os.getpid():
            sig = signal.SIGSTOP if a.respond == "stop" else signal.SIGKILL
            try: os.kill(t.tgid, sig); act = f" -> sent {sig.name} to pid {t.tgid}"
            except OSError as e: act = f" -> {a.respond} failed: {e}"
            rec["response_result"] = act.strip(" ->")
        print(f"\n[ALERT {rec['time']}] {t.comm or '?'} (pid {t.tgid}, tid {t.tid}) mean score {t.mean:.3f} > {thr:.3f} "
              f"after {t.nwin} windows / {t.n} calls ({when}){act}\n          most anomalous window: {' '.join(top)}", flush=True)
        logf.write(json.dumps(rec) + "\n"); logf.flush(); alerts.append(rec)

    def flush():
        if not pending: return
        t0 = time.perf_counter(); sc = det.score_windows([w for _, w in pending]); stats["score_s"] += time.perf_counter() - t0
        stats["windows"] += len(pending)
        for (t, w), s in zip(pending, sc):                     # in arrival order: the rule is checked after every window
            t.nwin += 1; t.sum += float(s)
            if s > t.worst: t.worst, t.worst_win = float(s), w
            if not t.alerted and t.nwin >= a.min_windows and t.mean > thr: alert(t, "running" if t.tid in tasks else "at exit")
        pending.clear()

    def finish(t):
        if det:
            if t.n and t.n < L: pending.append((t, list(t.calls)))   # short task: one partial window, as offline
            flush()
            if not t.alerted and 0 < t.nwin < a.min_windows and t.mean > thr: alert(t, "at exit")
        if a.record and t.raw:
            name = "".join(ch if ch.isalnum() else "_" for ch in (t.comm or "unknown"))
            with open(os.path.join(a.record, f"{a.tag}_{t.tid}_{name}.txt"), "w") as fh: fh.write(" ".join(map(str, t.raw)) + "\n")
        done.append(t)

    t_start = time.perf_counter(); cpu0 = os.times(); last_flush = time.perf_counter()
    try:
        for kind, tid, tgid, nr, comm in src:
            if kind == "LOST": stats["lost"] += tid; continue
            if kind == "X":
                t = tasks.pop(tid, None)
                if t: finish(t)
                continue
            t = tasks.get(tid)
            if t is None: t = tasks[tid] = Task(tid, L)
            if kind == "E":                                   # strace: execve succeeded, new program
                t.comm, t.stale = comm, False; t.tgid = t.tgid or tid; continue
            if kind == "C":                                   # strace: tgid=parent; nr=is_thread. Child inherits name
                p = tasks.get(tgid)
                if p and not t.comm: t.comm = p.comm
                t.tgid = (p.tgid if p else None) if nr else tid; t.stale = False
                continue
            stats["events"] += 1
            if comm: t.comm, t.tgid, t.stale = comm, tgid, False
            elif t.stale: ident(t)
            if a.record: t.raw.append(nr)
            if det:
                t.calls.append(convert([nr], det.numbering)[0]); t.n += 1
                if t.n >= L and (t.n - L) % S == 0: pending.append((t, list(t.calls)))
                if len(pending) >= a.batch or (pending and time.perf_counter() - last_flush > 0.05):
                    flush(); last_flush = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[hids] interrupted")
    for t in list(tasks.values()): finish(t)
    tasks.clear()

    wall = time.perf_counter() - t_start; cpu1 = os.times()
    cpu = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    if det and not a.quiet:
        rows = sorted((t for t in done if t.nwin), key=lambda t: -t.mean)
        print(f"\n{'tid':>7s} {'pid':>7s} {'program':16s} {'calls':>7s} {'windows':>7s} {'score':>7s}  verdict")
        for t in rows[:25]:
            print(f"{t.tid:7d} {str(t.tgid or '?'):>7s} {(t.comm or '?')[:16]:16s} {t.n:7d} {t.nwin:7d} {t.mean:7.3f}  {'ANOMALOUS' if t.alerted else 'normal'}")
        if len(rows) > 25: print(f"... {len(rows) - 25} more tasks")
    print(f"\n[hids] {stats['events']} system calls from {len(done)} tasks in {wall:.2f} s ({stats['events'] / max(wall, 1e-9):.0f}/s)"
          + (f", {stats['lost']} events LOST by the kernel buffer" if stats["lost"] else ""))
    if det:
        print(f"[hids] {stats['windows']} windows scored, {1000 * stats['score_s'] / max(stats['windows'], 1):.3f} ms/window; "
              f"{len(alerts)} alert(s) -> {a.log}")
    print(f"[hids] monitor CPU {cpu:.2f} s ({100 * cpu / max(wall, 1e-9):.1f}% of one core), peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB"
          + (f"; traces written to {a.record}" if a.record else ""), flush=True)
    sys.exit(3 if alerts else 0)

if __name__ == "__main__":
    main()
