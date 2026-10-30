#!/usr/bin/env bash
# Guided live demo for a presentation: a normal run and attacked runs of the demo service, first with the strace
# backend (no root), then with eBPF (root). Each step prints a banner, the command, runs it, then the points to
# explain, and waits for Enter.
#   bash live/demo/present.sh              full demo (asks for the sudo password once, before the eBPF part)
#   bash live/demo/present.sh --no-ebpf    strace steps only
#   bash live/demo/present.sh --auto       no pauses (rehearsal)
# Run it as your normal user from a terminal in the Ubuntu-22.04 distro, not through sudo or from PowerShell: the
# models learned the program's system calls in this environment (PATH, user, working directory), and a different
# environment changes the trace.
set -u
cd "$(dirname "$0")/../.."
EBPF=1; AUTO=0
for a in "$@"; do
    case "$a" in --no-ebpf) EBPF=0 ;; --auto) AUTO=1 ;; *) echo "unknown option $a"; exit 1 ;; esac
done

B=$'\e[1m'; DIM=$'\e[2m'; CYAN=$'\e[36m'; YEL=$'\e[33m'; GRN=$'\e[32m'; RED=$'\e[31m'; R=$'\e[0m'
STRACE_MODEL=live/models/host.npz; EBPF_MODEL=live/models/host_ebpf.npz
MON="python3 live/monitor.py"
SUDO_MON='sudo -E env "PATH=$PATH" python3 live/monitor.py'   # keep the user's PATH (sudo would replace it)
STEP=0

banner() {
    STEP=$((STEP + 1)); clear 2>/dev/null || true
    printf '%s\n' "${CYAN}${B}══════════════════════════════════════════════════════════════════════${R}"
    printf '%s\n' "${CYAN}${B}  STEP $STEP  $1${R}"
    printf '%s\n\n' "${CYAN}${B}══════════════════════════════════════════════════════════════════════${R}"
}
say()   { printf '%s\n' "${YEL}  ▸ $*${R}"; }
pause() {
    [ "$AUTO" = 1 ] && { echo; return; }
    printf '\n%s' "${DIM}  [Enter] next step   [r] run again   [q] quit ${R}"
    read -r k; case "$k" in q) exit 0 ;; r) return 1 ;; esac; return 0
}
run() {   # prints the command, then runs it
    printf '%s\n\n' "${GRN}\$ $*${R}"
    bash -c "$*"; echo "${DIM}  (exit code $?: 0 = no alert, 3 = alert raised)${R}"
}
step() {  # step "title" "command" "point" ... ; repeats while the presenter types r
    local title=$1 cmd=$2; shift 2
    while :; do
        banner "$title"; run "$cmd"; echo
        for p in "$@"; do say "$p"; done
        pause && break; STEP=$((STEP - 1))
    done
}

# ---------------------------------------------------------------- preflight
banner "Preflight checks"
ok=1
for f in "$STRACE_MODEL" "$EBPF_MODEL"; do [ -f "$f" ] || { echo "${RED}missing $f (train it first, see README)${R}"; ok=0; }; done
command -v strace >/dev/null || { echo "${RED}strace not found: sudo apt install strace${R}"; ok=0; }
[ "$(id -u)" = 0 ] && { echo "${RED}run this as your normal user, not with sudo${R}"; ok=0; }
[ "$ok" = 1 ] || exit 1
python3 live/tests/test_live.py | sed 's/^/  /'
if (exec 3<>/dev/tcp/127.0.0.1/4444) 2>/dev/null; then
    echo "  attacker listener already running on 127.0.0.1:4444"
else
    python3 live/demo/listener.py 4444 > /tmp/hids_listener.log 2>&1 & LPID=$!
    trap 'kill $LPID 2>/dev/null' EXIT; sleep 1
    echo "  started the attacker listener on 127.0.0.1:4444 (log: /tmp/hids_listener.log)"
fi
if [ "$EBPF" = 1 ]; then
    command -v bpftrace >/dev/null || { echo "${RED}bpftrace not found: sudo apt install bpftrace (or use --no-ebpf)${R}"; exit 1; }
    echo "  the eBPF steps need root: enter your password now so the demo is not interrupted later"
    sudo -v || exit 1
    [ -d /sys/kernel/tracing/events ] || sudo mount -t tracefs nodev /sys/kernel/tracing
fi
python3 - <<'EOF'
import json, numpy as np
for m in ("host", "host_ebpf"):
    t = json.loads(str(np.load(f"live/models/{m}.npz", allow_pickle=True)["meta"]))["thresholds"]
    print(f"  model {m + '.npz':14s} alert above {t['live_p99']:.3f}   kill above {1.1 * t['live_max']:.3f}")
EOF
say "Everything is ready. Normal runs use seeds the models never saw in training."
pause

# ---------------------------------------------------------------- strace backend
step "Normal run, strace backend (ptrace, no root needed)" \
    "$MON --model $STRACE_MODEL --respond kill -- bash live/demo/service.sh 5000" \
    "The report service runs ~23 processes: bash, cp, wc, sha256sum, grep, sort, gzip ..." \
    "Every process scores well below the alert threshold (0.170): all 'normal', 0 alerts." \
    "Kill mode is ON, and nothing was killed: no false alarm."

step "Reverse-shell attack, strace backend" \
    "ATTACK=revshell $MON --model $STRACE_MODEL --respond kill -- bash live/demo/service.sh 2001" \
    "The compromised service connects back to the attacker and runs its commands." \
    "ALERT: bash scores above 0.170; RESPONSE: above the kill threshold 0.199, SIGKILL is sent." \
    "Most anomalous window: socket connect ... dup2 = bash wiring its input/output to a network socket." \
    "The shell it spawned is killed too. The model never saw an attack: it only knows normal behaviour."

step "Privilege-persistence attack (adds a uid-0 user), strace backend" \
    "ATTACK=adduser $MON --model $STRACE_MODEL --respond kill -- bash live/demo/service.sh 2002" \
    "The payload writes a root account line into a copy of /etc/passwd." \
    "bash is killed after the passwd copy: unusual file activity for this service."

# ---------------------------------------------------------------- eBPF backend
if [ "$EBPF" = 1 ]; then
    sudo -v
    step "Normal run, eBPF backend (program running inside the kernel)" \
        "$SUDO_MON --backend ebpf --model $EBPF_MODEL --respond kill -- bash live/demo/service.sh 5002" \
        "bpftrace loads live/ebpf/syscalls.bt into the kernel on the raw_syscalls:sys_enter tracepoint." \
        "The service runs in its own cgroup; the kernel filter only emits calls from that cgroup." \
        "The program is never paused (unlike ptrace): 0 lost events, 0 alerts, nothing killed."

    step "Data-exfiltration attack, eBPF backend" \
        "ATTACK=exfil $SUDO_MON --backend ebpf --model $EBPF_MODEL --respond kill -- bash live/demo/service.sh 2003" \
        "tar packs config files, then the data is sent over TCP." \
        "RESPONSE: 'killed every process in cgroup /sys/fs/cgroup/hids-...' = one write to cgroup.kill." \
        "The kernel kills the whole process tree at once; no child can escape by forking."

    step "Reconnaissance attack, eBPF backend" \
        "ATTACK=recon $SUDO_MON --backend ebpf --model $EBPF_MODEL --respond kill -- bash live/demo/service.sh 2004" \
        "whoami, id and a SUID-file scan with find: programs this service never runs." \
        "Detected on the attacker's own commands, then the cgroup is killed."
fi

# ---------------------------------------------------------------- behind the scenes
banner "The kernel program and the alert log"
echo "${B}live/ebpf/syscalls.bt${R} (the whole in-kernel part, comments removed):"; echo
grep -v '^\s*\*\|^/\*\|^\s*$' live/ebpf/syscalls.bt | sed 's/^/    /'
echo; echo "${B}Last alert in live/logs/alerts.jsonl:${R}"; echo
tail -n 1 live/logs/alerts.jsonl | python3 -c '
import json, sys, textwrap
a = json.loads(sys.stdin.read())
w = a.pop("worst_window", [])
for k, v in a.items(): print(f"    {k:12s} {v}")
print("    worst_window " + textwrap.fill(" ".join(w), 90, subsequent_indent=" " * 17))'
echo
say "Two probes: one per system call entry, one per process exit, both filtered by the cgroup id."
say "Every alert is stored as JSON with the process, score, threshold, window and response."
pause

banner "Monitoring overhead (live/results/overhead.csv)"
python3 - <<'EOF'
import csv
rows = list(csv.DictReader(open("live/results/overhead.csv")))
conds = ["strace", "monitor-strace", "ebpf", "monitor-ebpf"]
print(f"  {'workload':22s}" + "".join(f"{c:>16s}" for c in conds))
for w in dict.fromkeys(r["workload"] for r in rows):
    print(f"  {w:22s}" + "".join(f"{'x' + next(r for r in rows if r['workload'] == w and r['condition'] == c)['slowdown']:>16s}" for c in conds))
EOF
echo
say "Overhead grows with the system call rate: ptrace stops the program twice per call (x259 in a tight loop)."
say "eBPF runs a tiny in-kernel program per call: no measurable cost on ordinary programs (x0.88-x1.04)."
pause

banner "Done"
say "Results: 40/40 attacks killed, 0/40 normal runs killed (report, Section VIII)."
say "A model trained on the public ADFA-LD dataset fails on this host (AUC 0.378): train where you protect."
echo
