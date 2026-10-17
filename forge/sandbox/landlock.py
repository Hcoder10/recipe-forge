"""Landlock via ctypes (Linux x86_64). No root needed.

Only `apply()` touches the calling process; everything else is pure data. Call `apply()` in the
child right before exec (forge.sandbox.enter does this).
"""
from __future__ import annotations

import ctypes
import os
import platform

SYS_CREATE_RULESET = 444
SYS_ADD_RULE = 445
SYS_RESTRICT_SELF = 446
CREATE_RULESET_VERSION = 1 << 0
RULE_PATH_BENEATH = 1
RULE_NET_PORT = 2
PR_SET_NO_NEW_PRIVS = 38

# filesystem rights
EXECUTE = 1 << 0
WRITE_FILE = 1 << 1
READ_FILE = 1 << 2
READ_DIR = 1 << 3
REMOVE_DIR = 1 << 4
REMOVE_FILE = 1 << 5
MAKE_CHAR = 1 << 6
MAKE_DIR = 1 << 7
MAKE_REG = 1 << 8
MAKE_SOCK = 1 << 9
MAKE_FIFO = 1 << 10
MAKE_BLOCK = 1 << 11
MAKE_SYM = 1 << 12
REFER = 1 << 13        # ABI 2
TRUNCATE = 1 << 14     # ABI 3
IOCTL_DEV = 1 << 15    # ABI 5

FILE_ONLY = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE | IOCTL_DEV

# network rights (ABI 4)
NET_BIND_TCP = 1 << 0
NET_CONNECT_TCP = 1 << 1

# scopes (ABI 6)
SCOPE_ABSTRACT_UNIX = 1 << 0
SCOPE_SIGNAL = 1 << 1

RO = EXECUTE | READ_FILE | READ_DIR
RW = (RO | WRITE_FILE | REMOVE_DIR | REMOVE_FILE | MAKE_DIR | MAKE_REG | MAKE_SOCK | MAKE_FIFO
      | MAKE_SYM | REFER | TRUNCATE)
DEV_RW = READ_FILE | WRITE_FILE | IOCTL_DEV


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64),
                ("handled_access_net", ctypes.c_uint64),
                ("scoped", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _NetPort(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("port", ctypes.c_uint64)]


class LandlockUnavailable(RuntimeError):
    pass


def _libc():
    return ctypes.CDLL(None, use_errno=True)


def abi_version() -> int:
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        return 0
    libc = _libc()
    v = libc.syscall(SYS_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(CREATE_RULESET_VERSION))
    return v if v > 0 else 0


def fs_mask(abi: int) -> int:
    m = (1 << 13) - 1
    if abi >= 2:
        m |= REFER
    if abi >= 3:
        m |= TRUNCATE
    if abi >= 5:
        m |= IOCTL_DEV
    return m


def apply(paths: list[tuple[str, int]], bind_ports: list[int] | None, connect_ports: list[int] | None,
          scope: bool = True) -> int:
    """Restrict the calling process (and its future children) for good.

    paths: (path, rights) pairs; missing paths are skipped. bind_ports/connect_ports: None leaves
    that direction unrestricted, a list allows only those TCP ports. Returns the ABI used.
    """
    abi = abi_version()
    if abi < 1:
        raise LandlockUnavailable("Landlock is not available on this kernel")
    libc = _libc()
    handled_fs = fs_mask(abi)
    handled_net = 0
    if abi >= 4:
        if bind_ports is not None:
            handled_net |= NET_BIND_TCP
        if connect_ports is not None:
            handled_net |= NET_CONNECT_TCP
    elif bind_ports is not None or connect_ports is not None:
        raise LandlockUnavailable(f"Landlock ABI {abi} has no network rules")
    scoped = (SCOPE_ABSTRACT_UNIX | SCOPE_SIGNAL) if (scope and abi >= 6) else 0
    attr = _RulesetAttr(handled_fs, handled_net, scoped)
    size = ctypes.sizeof(_RulesetAttr) if abi >= 6 else (16 if abi >= 4 else 8)
    fd = libc.syscall(SYS_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(size), ctypes.c_uint32(0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        for path, rights in paths:
            try:
                pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except FileNotFoundError:
                continue
            try:
                allowed = rights & handled_fs
                if not os.path.isdir(path):
                    allowed &= FILE_ONLY
                if not allowed:
                    continue
                pb = _PathBeneath(allowed, pfd)
                if libc.syscall(SYS_ADD_RULE, fd, ctypes.c_uint32(RULE_PATH_BENEATH), ctypes.byref(pb),
                                ctypes.c_uint32(0)) != 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule {path}")
            finally:
                os.close(pfd)
        for ports, right in ((bind_ports, NET_BIND_TCP), (connect_ports, NET_CONNECT_TCP)):
            if ports is None:
                continue
            for p in sorted(set(ports)):
                np_ = _NetPort(right, p)
                if libc.syscall(SYS_ADD_RULE, fd, ctypes.c_uint32(RULE_NET_PORT), ctypes.byref(np_),
                                ctypes.c_uint32(0)) != 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule port {p}")
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS)")
        if libc.syscall(SYS_RESTRICT_SELF, fd, ctypes.c_uint32(0)) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(fd)
    return abi
