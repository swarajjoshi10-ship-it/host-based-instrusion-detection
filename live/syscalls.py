"""
System call numbering for the live monitor.

Internally the monitor always works with x86_64 syscall NUMBERS (what the kernel's raw_syscalls tracepoint reports
on this machine). ADFA-LD was recorded on 32-bit x86 (Ubuntu 11.04, kernel 2.6.38), where the same call has a
different number and sometimes a different name, so a model trained on ADFA-LD needs every live call translated
to the i386 table. Tables come from the kernel UAPI headers (see syscall_tables.json).
"""
import json, os

_T = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "syscall_tables.json")))
X64 = _T["x86_64"]                           # name -> x86_64 number
I386 = _T["i386"]                            # name -> i386 number
X64_NAME = {v: k for k, v in X64.items()}   # x86_64 number -> name
UNKNOWN = 9999                               # never in any vocabulary, so models score it as the "unknown" token

# x86_64 call -> the i386 call that 32-bit glibc of 2011 would have issued for the same libc function.
_ALIAS = {
    # 64-bit file offsets / stat structures got their own calls on i386
    "stat": "stat64", "lstat": "lstat64", "fstat": "fstat64", "newfstatat": "fstatat64",
    "fcntl": "fcntl64", "truncate": "truncate64", "ftruncate": "ftruncate64", "statfs": "statfs64",
    "fstatfs": "fstatfs64", "sendfile": "sendfile64", "lseek": "_llseek", "fadvise64": "fadvise64_64",
    "mmap": "mmap2", "getrlimit": "ugetrlimit", "select": "_newselect",
    # 32-bit uid/gid variants
    **{c: c + "32" for c in ("getuid", "getgid", "geteuid", "getegid", "setuid", "setgid", "setreuid", "setregid",
                             "getgroups", "setgroups", "setresuid", "getresuid", "setresgid", "getresgid",
                             "chown", "fchown", "lchown", "setfsuid", "setfsgid")},
    # before Linux 4.3 every socket operation on i386 went through one multiplexed call, socketcall(2)
    **{c: "socketcall" for c in ("socket", "bind", "connect", "listen", "accept", "accept4", "getsockname",
                                 "getpeername", "socketpair", "sendto", "recvfrom", "shutdown", "setsockopt",
                                 "getsockopt", "sendmsg", "recvmsg")},
}

def name_to_x64(name):
    return X64.get(name, UNKNOWN)

def x64_to_i386(nr):
    name = X64_NAME.get(nr)
    if name is None: return UNKNOWN
    return I386.get(_ALIAS.get(name, name), UNKNOWN)

def convert(seq, numbering):
    """seq of x86_64 numbers -> numbering used by a model ('x86_64' or 'i386')"""
    if numbering == "x86_64": return list(seq)
    if numbering == "i386": return [x64_to_i386(n) for n in seq]
    raise ValueError(numbering)

def name_of(nr, numbering="x86_64"):
    table = X64 if numbering == "x86_64" else I386
    for k, v in table.items():
        if v == nr: return k
    return f"#{nr}"
