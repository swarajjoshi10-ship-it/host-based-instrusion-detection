# Host-Based Intrusion Detection Using System Call Analysis

Operating Systems course project. A host-based intrusion detection system (HIDS) that watches the **system calls**
processes make on a running Linux host, learns what normal behaviour looks like, and flags (and optionally stops)
processes that deviate from it.

The project has two parts:

| Part | Folder | What it does |
|---|---|---|
| **Live monitor** | `live/` | Captures system calls from the kernel (eBPF tracepoint or ptrace), scores them in real time with an LSTM, raises alerts, and can `SIGSTOP`/`SIGKILL` the offending process. Evaluated on a live host with real attack payloads and benchmarked for overhead. |
| Offline study | `phases/`, `results/` | Compares Isolation Forest, One-Class SVM, an autoencoder and an LSTM next-call model on the ADFA-LD and LID-DS 2021 datasets (5 seeds, tuning/test split). The LSTM won and is the engine of the live monitor. |

---

## 1. The operating-system side

### Where the data comes from
Every time a program needs the kernel (open a file, start a process, open a socket), it executes a `syscall`
instruction and the CPU switches from user mode to kernel mode. The sequence of these calls is a precise fingerprint
of what a process does, and an attacker who takes over a process cannot avoid making system calls. The monitor reads
that sequence in two ways:

| Backend | Mechanism | Needs root | Cost |
|---|---|---|---|
| **eBPF** (`live/ebpf/syscalls.bt`) | A small program verified and run *inside the kernel*, attached to the `raw_syscalls:sys_enter` tracepoint. Events go to user space through a per-CPU ring buffer. | yes | Low: no extra context switches; if user space falls behind, the kernel drops events (reported as `LOST`) instead of slowing the workload. |
| **strace** (`ptrace(2)`) | The kernel stops the traced task at every system call entry and exit and wakes the tracer, which reads the registers. | no (for a command it starts) | High: at least two extra context switches per call, and the traced program waits while the tracer works. |

### Following processes correctly
* **fork/clone (eBPF): cgroups, not pids.** Before the watched program runs, the monitor creates a cgroup v2 group
  (`/sys/fs/cgroup/hids-…`) and moves the program into it. Every process and thread it creates inherits the cgroup
  inside the kernel, so the eBPF filter `cgroup == <id>` follows the whole tree with nothing missed between `fork()`
  and the child's first call. With strace, the `clone`/`fork`/`vfork` return value names the child.
* **pid namespaces**: eBPF reports the kernel's *global* pid/tid, but processes inside a container, or inside a
  WSL2 distro (which runs in a nested pid namespace), see different, namespace-local pids. Following "pid 1234" from
  the kernel side therefore silently matches nothing, and `kill(1234)` could hit the wrong process. A cgroup id
  (the inode number of its directory) is the same in every namespace, which is why the eBPF backend uses it.
* **execve**: a task keeps its pid but becomes a different program, so its name is taken from the successful
  `execve` (strace) or from `comm` on every event (eBPF).
* **Threads**: windows are kept per kernel task (thread id); the response targets the thread group (process id).
* **Exit**: `sched:sched_process_exit` / strace's `+++ exited` ends the task, and its final verdict is computed.

### ABI differences: why models do not transfer between systems
ADFA-LD was recorded on 32-bit x86 (Ubuntu 11.04). The same libc call becomes a different system call there:
`mmap` → `mmap2`, `stat` → `stat64`, `getuid` → `getuid32`, and every socket operation goes through one multiplexed
`socketcall(2)` (Linux < 4.3). `live/syscalls.py` translates live x86_64 calls to that ABI using the kernel's own
`unistd_64.h` / `unistd_32.h` tables, so the ADFA-LD model *can* score live data, but see the results.

### Response
With strace, a flagged process is suspended with `SIGSTOP` (resume with `kill -CONT <pid>` after inspection) or
terminated with `SIGKILL`. Both are delivered by the kernel, cannot be caught, and act on the whole thread group.
With eBPF, the response acts on the watched cgroup: `--respond stop` writes `cgroup.freeze` (every task stops and
stays frozen for inspection; `echo 0 > …/cgroup.freeze` resumes) and `--respond kill` writes `cgroup.kill`
(SIGKILL to every task in the group, Linux ≥ 5.14), the same mechanisms container runtimes use.

---

## 2. Live monitor architecture

```
  kernel                                 user space (live/monitor.py, one core, ~28 MB)
 ┌──────────────────────────────┐       ┌───────────────────────────────────────────────────────────────┐
 │ raw_syscalls:sys_enter ──eBPF──ring──▶ parse ─▶ per-task window (last 20 calls, step 10)               │
 │ filter: cgroup id; exit       │ buffer│            │                                                  │
 │        or                     │       │            ▼ batch of windows                                  │
 │ ptrace stops ───────── strace ──FIFO─▶ LSTM next-call model (numpy) ─▶ score = mean −log P(next call)  │
 └──────────────────────────────┘       │            │                                                  │
                                        │            ▼ running mean per task > threshold?                │
                                        │   ALERT (+ most anomalous window) ─▶ alerts.jsonl, SIGSTOP/KILL  │
                                        └───────────────────────────────────────────────────────────────┘
```

* **Model**: LSTM next-call predictor (embedding 32, 64 hidden units, ~30 k parameters, 122 KB). Trained with PyTorch,
  but inference runs in plain numpy (checked against PyTorch to within 1e-6), so the monitor is **28 MB instead of
  340 MB**.
* **Alert rule**: a task is flagged when the running mean of its window scores exceeds the threshold after at least
  5 windows, or at exit for shorter tasks. The threshold is the 99th percentile of the **same statistic** on normal
  calibration runs, so the calibrated false alarm rate matches what the monitor really does.

---

## 3. Results on a live host

Environment: WSL2 Ubuntu 22.04, kernel 6.6.87, x86_64. Every number below comes from `live/results/*.csv`.

### Detection (`live/demo/`)
The demo workload is a small "report service" (bash script that copies, counts, hashes, sorts and compresses files;
about 23 processes per run). Attacks are injected into the same service: **recon** (whoami/id/SUID scan),
**adduser** (adds a uid-0 line to a copy of `/etc/passwd`), **revshell** (connects back to a local listener and runs
its commands) and **exfil** (archives config files and sends them over TCP). Held-out runs use seeds not seen in
training.

Every row: 40 held-out normal runs and 40 attack runs (10 per payload), 99th-percentile threshold (the default),
95th in brackets where it differs.

| Model trained on | Test runs captured with | AUC | Precision | Recall | F1 | False alarm rate |
|---|---|---|---|---|---|---|
| this host, **eBPF** | **eBPF** | 1.000 | 1.000 (0.976) | 1.000 | **1.000** (0.988) | **0.000** (0.025) |
| this host, strace | strace | 1.000 | 0.976 | 1.000 | **0.988** | **0.025** |
| this host, strace | eBPF | 0.983 | 0.541 | 1.000 | 0.702 | **0.850** |
| this host, eBPF | strace | 1.000 | 1.000 (0.784) | 1.000 | 1.000 (0.879) | 0.000 (0.275) |
| ADFA-LD | strace | 0.378 | 0.500 | 1.000 | 0.667 | **1.000** |

* **A model must be trained on the host it protects.** The ADFA-LD model, even with the syscall translation, flags
  *every* normal run: AUC 0.378 is worse than chance. Fourteen years of glibc and kernel changes alter normal
  behaviour more than the attacks do.
* **It must also see the program in the same environment.** Run from an interactive terminal, the same script makes
  extra job-control calls (`getpgrp`, `ioctl`) at startup; under `sudo`, PATH is replaced and bash's command lookup
  changes; started as `/usr/bin/bash` instead of `bash`, bash searches PATH for itself differently; even launching the
  script from another working directory changes its trace. Each of these alone raised false alarms on normal runs.
  The monitor therefore always starts the watched program like a service (own session, no controlling terminal, stdin from `/dev/null`) and keeps the user's PATH.
  After that, 5 normal runs × with/without a terminal raised 0 alerts.
* **Train with the capture mechanism you detect with (observer effect).** ptrace stops the program at every call,
  which fixes the order of events such as a child's exit, bash's `wait4` and the SIGCHLD handler (`rt_sigreturn`):
  in 100 strace-recorded runs `wait4 → rt_sigreturn` appears 2,012 times and the reverse order never. Under eBPF the
  program runs at full speed and other orderings appear, so a strace-trained model raised false alarms on normal
  eBPF-monitored runs even after the launch was matched exactly. Measured (table above): the strace-trained model
  flags **34 of 40** normal eBPF-captured runs, the eBPF-trained model **0 of 40**. Its AUC stays 0.983, so the
  attacks still stand out; what breaks is the threshold, calibrated on strace timing. Under eBPF, 5 of 280 runs
  even contain one extra short-lived bash subshell that never appears when the same seed runs under strace.
  `live/demo/ebpf_experiment.sh` reproduces all of this.
* Live checks with the eBPF model (`ebpf_experiment.sh`, step 4): 2 of 3 fresh normal runs raised no alert; the
  third (seed 5002) scored **0.181 against a 0.181 threshold** on a child-reaping window (`read wait4 wait4
  rt_sigreturn …`). The reverse shell was flagged on `… socket connect fcntl … dup2` and its cgroup killed. The recon
  run was killed on that same borderline pre-payload window, i.e. for the wrong reason; it does not count as a
  detection. Earlier strace live checks: 0 alerts on 10 normal runs, reverse shell flagged on `socket connect … dup2`.
* In the first attempt, thresholds came from 20 calibration runs, so the "99th percentile" was simply the largest
  of 20 values. That gave a false alarm rate of 0.10 on the test runs. A separate set of 100 calibration runs was then
  added. This change was made after seeing test numbers, so it is reported here. The dataset was recorded again after
  the launch change above, with the same result.

### Overhead (`sudo -E python3 live/bench/overhead.py --ebpf`, one run, median of 5 repeats, workload wall time)

| Workload | No monitor | strace only | strace + monitor | eBPF only | **eBPF + monitor** |
|---|---|---|---|---|---|
| Syscall storm (50 k calls in a tight loop) | 0.015 s | 2.92 s (×201) | 3.78 s (×259) | 0.095 s (×6.5) | **0.093 s (×6.4)** |
| Report service ×10 (43 k calls, ~230 processes) | 0.361 s | 2.77 s (×7.7) | 3.64 s (×10.1) | 0.407 s (×1.13) | **0.316 s (×0.88)** |
| gzip 32 MB (1.7 k calls, CPU bound) | 0.466 s | 0.58 s (×1.25) | 0.62 s (×1.32) | 0.494 s (×1.06) | **0.483 s (×1.04)** |

eBPF lost **0** events in every run (with the 4 MB-per-CPU ring buffer) and the monitor scored all of them:
50,552 / 42,737 / 1,685 calls. The monitor process: 28 MB peak RSS, 0.05–0.39 s of CPU per workload,
about 0.1 ms per scored window.

* Overhead grows with the **system call rate**, not with the work done. ptrace stops the program twice per call and
  wakes the tracer each time, so a tight syscall loop runs ~200× slower; the same loop under eBPF pays only for a
  small in-kernel program per call (×6.4, roughly 1.6 µs per call). For ordinary programs eBPF is within run-to-run
  noise: the unmonitored service itself varied from 0.24 to 0.42 s across repeats, so ×0.88 means "no measurable
  cost", not a speed-up.
* With eBPF, the monitor adds nothing measurable on top of capture: the kernel buffers the events and the program never
  waits for the detector. With strace, the program does wait (×201 → ×259).
* All workloads ran as root in this run, in every condition alike. A previous run gave the same picture
  (eBPF + monitor ×6.4 / ×1.45 / ×1.04).

---

## 4. How to run (Linux or WSL2)

```bash
pip3 install --user numpy scikit-learn psutil matplotlib
pip3 install --user torch --index-url https://download.pytorch.org/whl/cpu      # training only
sudo apt install strace gcc bpftrace                                          # bpftrace for the eBPF backend

python3 live/tests/test_live.py                                               # offline tests (parsers, mapping)
bash live/demo/collect.sh                                                     # record normal + attack runs (~3 min)
python3 live/train.py --traces live/recordings/normal_train --calib live/recordings/normal_calib \
        --numbering x86_64 --out live/models/host.npz
python3 live/evaluate.py --model live/models/host.npz --normal live/recordings/normal_test \
        --attack live/recordings/attack --tag strace

# live demo: start the "attacker", then watch a compromised run get killed
python3 live/demo/listener.py &
ATTACK=revshell python3 live/monitor.py --model live/models/host.npz --respond kill -- bash live/demo/service.sh 7

# eBPF backend (root): same command, kernel-side capture; or a running process tree / the whole host
ATTACK=revshell sudo -E env "PATH=$PATH" python3 live/monitor.py --backend ebpf --model live/models/host.npz --respond kill -- bash live/demo/service.sh 7
sudo -E env "PATH=$PATH" python3 live/monitor.py --backend ebpf --model live/models/host.npz --pid <PID>

sudo -E env "PATH=$PATH" bash live/demo/ebpf_experiment.sh   # eBPF dataset, model, cross-backend evaluation, live checks

python3 live/bench/overhead.py                    # strace overhead
sudo -E python3 live/bench/overhead.py --ebpf     # adds the eBPF columns

# ADFA-LD model, for the cross-system comparison
python3 live/train.py --traces Training_Data_Master --calib Validation_Data_Master --numbering i386 --out live/models/adfa.npz
python3 live/evaluate.py --model live/models/adfa.npz --normal live/recordings/normal_test --attack live/recordings/attack --tag host_recordings
```

Recording only (no model) for a new program: `python3 live/monitor.py --record <folder> --tag <run> -- <command>`.

### Offline study
```
python run_phase.py 1 ... 8            # data loading, features, baselines, autoencoder, LSTM, evaluation, LID-DS, report
python phase6b_improve.py --data . --exp FINAL --hidden 64 --layers 1 --L 20 --extra_train 2000 --ens 1 --rule mean --seeds 5
```
Results and exact commands: `results/final/README.md`. Headline: on ADFA-LD the LSTM reaches AUC 0.854 and F1 0.527
at a 5.4% false alarm rate; on LID-DS Bruteforce_CWE-307, F1 0.812 with thresholds learned from normal data only.

---

## 5. Limitations
* **The threshold sits right at the top of normal behaviour.** Normal runs of the demo service are nearly identical,
  so the 99th percentile of 100 calibration runs is only just above typical scores (strace emulation: 0.164 vs 0.165;
  live eBPF: one normal run at 0.181 vs 0.181). Rare but normal timing orderings at native speed are then enough for a
  false alarm, and with `--respond kill` a false alarm kills a healthy run. The principled fix is more normal
  training and calibration runs, so those orderings are learned and the percentile estimate is stable; a fixed safety
  margin was deliberately not added after seeing results.
* The demo service is small and repetitive (held-out normal runs score 0.17–0.27, attack runs 0.30–3.97), so the live detection numbers
  show that the pipeline works end to end; they are not a measure of accuracy on a busy general-purpose server.
  Much of the signal is "a program this service never runs" (e.g. `find`, `id`) plus unusual calls in bash itself
  (`connect`, `dup2`).
* strace slows syscall-heavy programs by up to ×250; it is the fallback, not the intended deployment.
* The model also learns the *environment*: under `sudo`, PATH is replaced by `secure_path`, bash's command lookup
  makes different calls, and a normal run raised 3 false alarms (reproduced without eBPF by running the strace
  monitor with sudo's PATH). Watched programs must run in the environment the normal traces were recorded in.
* With eBPF, the response acts on the whole watched tree (its cgroup), not on the single flagged process; in
  whole-host mode there is no cgroup to act on, so no response is taken. `--pid` moves the existing process tree
  into a new cgroup, which also changes its resource accounting.
* Ubuntu 22.04's bpftrace 0.14 cannot run scripts with `BEGIN`/`END` blocks (stripped binary), so the script has none.
* eBPF drops events instead of slowing the program when bpftrace's per-CPU perf ring buffer is full. With the default
  256 KB buffer, a first benchmark run lost 33,616 of 50,582 calls in the syscall storm, a blind spot an attacker
  could exploit with a burst. The monitor now uses 4 MB per CPU (`BPFTRACE_PERF_RB_PAGES=1024`) and reports every loss.
* The monitor needs root for eBPF, but the watched program runs as the user who invoked `sudo` (`--as_root` to
  override). Running it as root changes its system calls (extra permission checks) and caused false alarms against
  a model trained on user-level runs.
* Whole-host eBPF mode produces every system call on the machine; a single Python consumer cannot keep up on a busy
  host and the kernel will drop events (counted and reported).
* An attacker who knows the model can try to mimic normal call sequences (mimicry attacks); arguments and return
  values are not used.
* On WSL2, bpftrace may need tracefs mounted first: `sudo mount -t tracefs nodev /sys/kernel/tracing`.
  If root's Python cannot import numpy under `sudo -E`, install the packages for root as well.
