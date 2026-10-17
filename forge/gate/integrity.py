"""Content hashes for everything a gate run depends on."""
from __future__ import annotations

import hashlib
import json
import os


def hash_tree(root: str, skip_dirs: tuple[str, ...] = ("__pycache__",)) -> str:
    """sha256 over relative paths, file bytes and symlink targets (never mtimes)."""
    h = hashlib.sha256()
    root = os.path.realpath(root)
    if os.path.isfile(root):
        with open(root, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    for dp, dns, fns in os.walk(root, followlinks=False):
        dns[:] = sorted(d for d in dns if d not in skip_dirs)
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            if os.path.islink(p):
                h.update(b"L" + rel.encode() + b"\0" + os.readlink(p).encode() + b"\0")
                continue
            if not os.path.isfile(p):
                continue
            h.update(b"F" + rel.encode() + b"\0")
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            h.update(b"\0")
    return h.hexdigest()


def tree_signature(root: str) -> str:
    """Cheap change detector (count, sizes, mtimes). Only used to decide when to re-hash contents."""
    h = hashlib.sha256()
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            h.update(f"{os.path.relpath(p, root)}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def venv_hash(venv: str, cache_path: str) -> str:
    """Content hash of a venv, re-hashed whenever its signature changes."""
    venv = os.path.realpath(venv)
    sig = tree_signature(venv)
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    ent = cache.get(venv)
    if ent and ent.get("sig") == sig:
        return ent["sha256"]
    digest = hash_tree(venv)
    cache[venv] = {"sig": sig, "sha256": digest}
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, cache_path)
    return digest


def canon_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
