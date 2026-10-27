"""
Offline tests for the live monitor (no root, no tracing): parsers, syscall mapping, alert-rule statistic.
  python3 live/tests/test_live.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capture import StraceParser, parse_bpftrace
from syscalls import X64, I386, x64_to_i386, UNKNOWN
from model import live_stat, windows

def test_strace_parser():
    p = StraceParser(); ev = []
    for line in [
        '100  execve("/usr/bin/bash", ["bash"], 0x7ffd /* 3 vars */) = 0',
        '100  openat(AT_FDCWD, "/etc/passwd", O_RDONLY|O_CLOEXEC) = 3',
        '100  clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|SIGCHLD <unfinished ...>',
        '101  execve("/usr/bin/id", ["id"], 0x5f /* 3 vars */ <unfinished ...>',
        '100  <... clone resumed>, child_tidptr=0x7f) = 101',
        '101  <... execve resumed>)          = 0',
        '102  clone3({flags=CLONE_VM|CLONE_THREAD, ...}, 88) = 103',
        '101  --- SIGCHLD {si_signo=SIGCHLD} ---',
        '101  syscall_0x1c0(0, 0) = -1 ENOSYS (Function not implemented)',
        '101  +++ exited with 0 +++',
    ]: ev += p(line)
    assert ev[0] == ("S", 100, None, X64["execve"], None)
    assert ev[1] == ("E", 100, None, None, "bash")
    assert ev[2] == ("S", 100, None, X64["openat"], None)
    assert ev[3] == ("S", 100, None, X64["clone"], None)               # counted once, at entry
    assert ev[4] == ("S", 101, None, X64["execve"], None)
    assert ev[5] == ("C", 101, 100, False, None)                       # result read from the resumed line
    assert ev[6] == ("E", 101, None, None, "id")
    assert ev[7] == ("S", 102, None, X64["clone3"], None) and ev[8] == ("C", 103, 102, True, None)
    assert ev[9] == ("S", 101, None, 0x1c0, None)
    assert ev[10] == ("X", 101, None, None, None) and len(ev) == 11

def test_bpftrace_parser():
    assert parse_bpftrace("READY\n") == []
    assert parse_bpftrace("S 200 201 257 my prog\n") == [("S", 201, 200, 257, "my prog")]   # comm may contain spaces
    assert parse_bpftrace("X 201\n") == [("X", 201, None, None, None)]
    assert parse_bpftrace("Lost 42 events\n") == [("LOST", 42, None, None, None)]

def test_syscall_mapping():
    assert x64_to_i386(X64["read"]) == I386["read"] == 3
    assert x64_to_i386(X64["mmap"]) == I386["mmap2"] == 192
    assert x64_to_i386(X64["connect"]) == I386["socketcall"] == 102
    assert x64_to_i386(X64["getuid"]) == I386["getuid32"]
    assert x64_to_i386(10 ** 6) == UNKNOWN

def test_live_stat():
    assert abs(live_stat([1, 1, 1, 9], 2) - 3.0) < 1e-9                # running means 1,1,1,3 -> max from window 2
    assert abs(live_stat([9, 1, 1, 1], 2) - 5.0) < 1e-9                # a spike in window 1 is only seen averaged
    assert abs(live_stat([4, 2], 5) - 3.0) < 1e-9                      # short task: final mean (checked at exit)
    assert windows(list(range(25)), 20, 10) == [list(range(20))] and windows([1, 2], 20, 10) == [[1, 2]]

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print(f"ok  {name}")
