#!/usr/bin/env bash
# The eBPF half of the live evaluation, in one go (root):
#   sudo -E env "PATH=$PATH" bash live/demo/ebpf_experiment.sh
# 1. records the dataset through the eBPF backend           -> live/recordings_ebpf/
# 2. trains and calibrates a model on it                     -> live/models/host_ebpf.npz
# 3. evaluates eBPF-trained and strace-trained models on eBPF-recorded runs (and the reverse): the observer effect
# 4. live checks with the eBPF model: a normal run, and a reverse shell with --respond kill
# Everything is also appended to live/results/ebpf_experiment.log. Interrupted? Run it again: runs already recorded
# are kept and only the missing ones are recorded (FRESH=1 starts the recordings from scratch).
set -eu
cd "$(dirname "$0")/../.."
[ "$(id -u)" = 0 ] || { echo "run as root: sudo -E env \"PATH=\$PATH\" bash live/demo/ebpf_experiment.sh"; exit 1; }
mkdir -p live/results
exec > >(tee -a live/results/ebpf_experiment.log) 2>&1
M=live/models

echo; echo "== 1. recording through eBPF ($(date "+%F %T"))"
if [ "${FRESH:-0}" = 1 ]; then RESUME=0; else RESUME=1; fi
BACKEND=ebpf RESUME=$RESUME bash live/demo/collect.sh

echo; echo "== 2. training on eBPF recordings ($(date +%T))"
python3 live/train.py --traces live/recordings_ebpf/normal_train --calib live/recordings_ebpf/normal_calib \
        --numbering x86_64 --out $M/host_ebpf.npz 2>&1 | grep -v "held-out"

echo; echo "== 3a. eBPF-trained model, eBPF-recorded test runs"
python3 live/evaluate.py --model $M/host_ebpf.npz --normal live/recordings_ebpf/normal_test --attack live/recordings_ebpf/attack --tag ebpf
echo; echo "== 3b. strace-trained model, eBPF-recorded test runs (observer effect)"
python3 live/evaluate.py --model $M/host.npz --normal live/recordings_ebpf/normal_test --attack live/recordings_ebpf/attack --tag ebpf_recordings
echo; echo "== 3c. eBPF-trained model, strace-recorded test runs (the reverse)"
python3 live/evaluate.py --model $M/host_ebpf.npz --normal live/recordings/normal_test --attack live/recordings/attack --tag strace_recordings

echo; echo "== 4. live checks with the eBPF model ($(date +%T))"
python3 live/demo/listener.py 4444 > /dev/null 2>&1 &
LPID=$!; trap 'kill $LPID 2>/dev/null' EXIT; sleep 1
for s in 7 5000 5002; do
    echo "-- normal run, seed $s"
    python3 live/monitor.py --backend ebpf --quiet --model $M/host_ebpf.npz -- bash live/demo/service.sh $s | grep -E "ALERT|most anomalous|alert\(s\)" || true
done
for p in revshell recon; do
    echo "-- attack $p, --respond kill"
    ATTACK=$p python3 live/monitor.py --backend ebpf --quiet --respond kill --model $M/host_ebpf.npz -- bash live/demo/service.sh 7 \
        | grep -E "ALERT|most anomalous|alert\(s\)" || true
done
echo; echo "done ($(date +%T)); log: live/results/ebpf_experiment.log"
