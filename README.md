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
* **fork/clone**: the eBPF program listens on `sched:sched_process_fork` and adds every child (process *or* thread) of
  a followed task to an in-kernel map, so nothing is missed between `fork()` and the child's first call.
  With strace, the `clone`/`fork`/`vfork` return value names the child.
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
A flagged process can be suspended with `SIGSTOP` (resume with `kill -CONT <pid>` after inspection) or terminated
with `SIGKILL`. Both are delivered by the kernel, cannot be caught by the process, and act on the whole thread group.

---

## 2. Live monitor architecture

```
  kernel                                 user space (live/monitor.py, one core, ~28 MB)
 ┌──────────────────────────────┐       ┌───────────────────────────────────────────────────────────────┐
 │ raw_syscalls:sys_enter ──eBPF──ring──▶ parse ─▶ per-task window (last 20 calls, step 10)               │
 │ sched_process_fork / exit     │ buffer│            │                                                  │
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

### Detection (strace capture, `live/demo/`)
The demo workload is a small "report service" (bash script that copies, counts, hashes, sorts and compresses files;
about 23 processes per run). Attacks are injected into the same service: **recon** (whoami/id/SUID scan),
**adduser** (adds a uid-0 line to a copy of `/etc/passwd`), **revshell** (connects back to a local listener and runs
its commands) and **exfil** (archives config files and sends them over TCP). Held-out runs use seeds not seen in
training.

| Model | Threshold | Runs (normal / attack) | AUC | Precision | Recall | F1 | False alarm rate |
|---|---|---|---|---|---|---|---|
| Trained on this host | 99th pct (default) | 40 / 40 | 1.000 | 0.976 | 1.000 | **0.988** | **0.025** |
| Trained on this host | 95th pct | 40 / 40 | 1.000 | 0.930 | 1.000 | 0.964 | 0.075 |
| Trained on ADFA-LD | 99th pct | 40 / 40 | 0.437 | 0.500 | 1.000 | 0.667 | **1.000** |

* **A model must be trained on the host it protects.** The ADFA-LD model, even with the syscall translation, flags
  *every* normal run: AUC 0.437 is worse than chance. Fourteen years of glibc and kernel changes alter normal
  behaviour more than the attacks do.
* Live, with `--respond kill`: the reverse shell's bash is flagged on the window containing `connect … dup2 …` and
  killed; in the recon run, bash is killed before the SUID scan starts.
* In the first attempt, thresholds came from 20 calibration runs, so the "99th percentile" was simply the largest
  of 20 values. That gave a false alarm rate of 0.10 on the test runs. A separate set of 100 calibration runs was then
  added. This change was made after seeing test numbers, so it is reported here.

### Overhead (`live/bench/overhead.py`, median of 5 runs, wall time of the workload itself)

| Workload | No monitor | strace only | strace + monitor | eBPF only | eBPF + monitor |
|---|---|---|---|---|---|
| Syscall storm (50 k calls in a tight loop) | 0.031 s | 4.86 s (×157) | 6.57 s (×213) | pending | pending |
| Report service ×10 (43 k calls, ~230 processes) | 0.641 s | 4.60 s (×7.2) | 5.78 s (×9.0) | pending | pending |
| gzip 32 MB (1.7 k calls, CPU bound) | 0.735 s | 0.96 s (×1.31) | 1.01 s (×1.38) | pending | pending |

Monitor process: 28 MB peak RSS, about 0.12–0.19 ms per scored window, 0.15–2.3 s of CPU per workload.
Overhead grows with the **system call rate**, not with the work done: ptrace makes every call pay for context
switches, so syscall-heavy programs suffer most. That is the motivation for the eBPF backend. The eBPF columns need
root; run the command in section 4 to fill them in.

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
ATTACK=revshell sudo -E python3 live/monitor.py --backend ebpf --model live/models/host.npz --respond kill -- bash live/demo/service.sh 7
sudo -E python3 live/monitor.py --backend ebpf --model live/models/host.npz --pid <PID>

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
* The demo service is small and repetitive (held-out normal runs score 0.17–0.27, attack runs 0.30–3.97), so the live detection numbers
  show that the pipeline works end to end; they are not a measure of accuracy on a busy general-purpose server.
  Much of the signal is "a program this service never runs" (e.g. `find`, `id`) plus unusual calls in bash itself
  (`connect`, `dup2`).
* strace slows syscall-heavy programs by up to ×200; it is the fallback, not the intended deployment.
* The eBPF backend is implemented and its output parser is unit-tested, but it has not yet been run in this
  environment (needs root). Its overhead columns are pending.
* Whole-host eBPF mode produces every system call on the machine; a single Python consumer cannot keep up on a busy
  host and the kernel will drop events (counted and reported).
* An attacker who knows the model can try to mimic normal call sequences (mimicry attacks); arguments and return
  values are not used.
* On WSL2, bpftrace may need tracefs mounted first: `sudo mount -t tracefs nodev /sys/kernel/tracing`.
  If root's Python cannot import numpy under `sudo -E`, install the packages for root as well.
