#!/usr/bin/env bash
# Records the live dataset used by the demo and the evaluation (strace backend, no root needed):
#   live/recordings/normal_train   NTRAIN normal runs  (training)
#   live/recordings/normal_calib   NCAL   normal runs  (threshold calibration)
#   live/recordings/normal_test    NTEST  normal runs  (held out: false alarm rate)
#   live/recordings/attack         NATT runs of every payload (recon, adduser, revshell, exfil)
# Seeds differ between the sets, so no test run repeats a training run.
#   bash live/demo/collect.sh [NTRAIN] [NCAL] [NTEST] [NATT]
set -eu
cd "$(dirname "$0")/../.."
NTRAIN=${1:-100}; NCAL=${2:-100}; NTEST=${3:-40}; NATT=${4:-10}; R=live/recordings
rm -rf "$R"; mkdir -p "$R"
python3 live/demo/listener.py 4444 > /dev/null 2>&1 &
LPID=$!; trap 'kill $LPID 2>/dev/null' EXIT; sleep 1

rec() {  # rec <folder> <tag> <seed> [payload]
    ATTACK=${4:-} python3 live/monitor.py --backend strace --record "$R/$1" --tag "$2" -- bash live/demo/service.sh "$3" > /dev/null 2>&1
}
for i in $(seq 1 "$NTRAIN"); do rec normal_train "n$i" "$i"; done;              echo "normal_train: $NTRAIN runs"
for i in $(seq 1 "$NCAL"); do rec normal_calib "c$i" "$((3000 + i))"; done;     echo "normal_calib: $NCAL runs"
for i in $(seq 1 "$NTEST"); do rec normal_test "t$i" "$((1000 + i))"; done;     echo "normal_test: $NTEST runs"
for p in recon adduser revshell exfil; do
    for i in $(seq 1 "$NATT"); do rec attack "$p$i" "$((2000 + i))" "$p"; done; echo "attack/$p: $NATT runs"
done
echo "trace files: $(find "$R" -name '*.txt' | wc -l)"
