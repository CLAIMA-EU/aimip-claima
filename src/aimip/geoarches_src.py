"""Pinned geoarches source for the ArchesWeather models (standard library only).

geoarches (INRIA, BSD-3-Clause) is not pip-installed: it takes its version and its package data (normalization
statistics, constant masks) from git, so a wheel built from the GitHub archive would lack those files. The image build
(container/Dockerfile) and install/install-native.sh run this file as a script instead:

    python geoarches_src.py install <target>     download the pinned archive, check size and SHA-256, unpack the
                                                 `geoarches/` package and `LICENSE` into <target>, check the tree digest
    python geoarches_src.py verify <target>      check the tree digest only

<target> is `/opt/aimip/geoarches` in the container and `<prefix>/geoarches` in native mode; the emulator job finds it
next to the emulator environment, verifies the digest, and puts it first on `sys.path`.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

GEOARCHES_COMMIT = "4d0c77a1cd48f3cbc0dc1b367a20abec495c2196"   # INRIA/geoarches branch `aimip`, 2026-09-18
GEOARCHES_ARCHIVE_URL = f"https://github.com/INRIA/geoarches/archive/{GEOARCHES_COMMIT}.tar.gz"
GEOARCHES_ARCHIVE_SIZE = 76_569_329
GEOARCHES_ARCHIVE_SHA256 = "71fda70b4454639dc1f1d6f75096e1a50a5a829aca3f11a2bb3a72684e5c69a1"
GEOARCHES_TREE_SHA256 = "0bac016742260e3213e6ee1d2e03c93c16182e523c317d0acb709493114f8688"
GEOARCHES_TREE_FILES = 88
_KEEP = ("geoarches/", "LICENSE")


def tree_digest(root: str) -> tuple[str, int]:
    """SHA-256 over sorted `relpath \\0 sha256(file) \\n` records (the same digest as the ArchesClimate-SSP tree).

    Symlinks and other non-regular entries are refused."""
    records = []
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        for name in (*files, *dirs):
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                raise ValueError(f"symlink in the geoarches tree: {full}")
        dirs.sort()
        for name in files:
            full = os.path.join(dirpath, name)
            if not os.path.isfile(full):
                raise ValueError(f"unexpected non-regular entry in the geoarches tree: {full}")
            h = hashlib.sha256()
            with open(full, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            records.append((os.path.relpath(full, root).replace(os.sep, "/"), h.hexdigest()))
    records.sort()
    overall = hashlib.sha256()
    for rel, val in records:
        overall.update(f"{rel}\0{val}\n".encode("utf-8"))
    return overall.hexdigest(), len(records)


def verify(root: str) -> None:
    if not os.path.isdir(root) or os.path.islink(root):
        raise SystemExit(f"geoarches source directory missing: {root}")
    digest, n = tree_digest(root)
    if (digest, n) != (GEOARCHES_TREE_SHA256, GEOARCHES_TREE_FILES):
        pyc = any(d == "__pycache__" for _, dirs, _ in os.walk(root) for d in dirs)
        raise SystemExit(f"geoarches source tree {root} does not match the pinned commit {GEOARCHES_COMMIT[:12]} "
                         f"({n} files, sha256 {digest}; expected {GEOARCHES_TREE_FILES} files, "
                         f"sha256 {GEOARCHES_TREE_SHA256})"
                         + ("; it contains __pycache__ (bytecode written by an import without python -B)" if pyc else ""))


def install(target: str, archive: str | None = None) -> None:
    """Download (or read *archive*), check it, unpack `geoarches/` and `LICENSE` into *target*, verify the tree."""
    if os.path.lexists(target):
        raise SystemExit(f"refusing to overwrite {target}")
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    work = tempfile.mkdtemp(prefix=".geoarches-", dir=parent)
    try:
        tgz = archive or os.path.join(work, "geoarches.tar.gz")
        if archive is None:
            req = urllib.request.Request(GEOARCHES_ARCHIVE_URL, headers={"User-Agent": "aimip-install"})
            with urllib.request.urlopen(req, timeout=300) as resp, open(tgz, "wb") as out:  # noqa: S310 - pinned URL
                shutil.copyfileobj(resp, out, 1 << 20)
        h = hashlib.sha256()
        with open(tgz, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        size = os.path.getsize(tgz)
        if (size, h.hexdigest()) != (GEOARCHES_ARCHIVE_SIZE, GEOARCHES_ARCHIVE_SHA256):
            raise SystemExit(f"geoarches archive mismatch: {size} bytes, sha256 {h.hexdigest()} "
                             f"(expected {GEOARCHES_ARCHIVE_SIZE}, {GEOARCHES_ARCHIVE_SHA256})")
        stage = os.path.join(work, "tree")
        prefix = f"geoarches-{GEOARCHES_COMMIT}/"
        with tarfile.open(tgz, "r:gz") as tf:
            for m in tf.getmembers():
                if m.name == prefix.rstrip("/") and m.isdir():
                    continue
                if not m.name.startswith(prefix):
                    raise SystemExit(f"unexpected archive entry {m.name!r}")
                rel = m.name[len(prefix):]
                if not rel or not (rel.startswith(_KEEP[0]) or rel == _KEEP[1]):
                    continue
                parts = rel.split("/")
                if ".." in parts or rel.startswith("/") or not (m.isfile() or m.isdir()):
                    raise SystemExit(f"refusing archive entry {m.name!r}")
                dst = os.path.join(stage, *parts)
                if m.isdir():
                    os.makedirs(dst, mode=0o755, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(dst), mode=0o755, exist_ok=True)
                src = tf.extractfile(m)
                if src is None:
                    raise SystemExit(f"cannot read archive entry {m.name!r}")
                with src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
                os.chmod(dst, 0o644)
        verify(stage)
        os.replace(stage, target)
        os.chmod(target, 0o755)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def code_dir(python_executable: str | None = None) -> str:
    """`<env parent>/geoarches` for the emulator interpreter (`/opt/aimip/venv-emu/bin/python` -> `/opt/aimip/geoarches`).

    `$AIMIP_GEOARCHES_CODE` overrides it (development only; the job then skips the digest check and says so)."""
    override = os.environ.get("AIMIP_GEOARCHES_CODE")
    if override:
        return override
    exe = python_executable or sys.executable
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(exe))), "geoarches")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "install":
        install(argv[1])
    elif len(argv) == 3 and argv[0] == "install":
        install(argv[1], archive=argv[2])
    elif len(argv) == 2 and argv[0] == "verify":
        verify(argv[1])
    else:
        print(__doc__, file=sys.stderr)
        return 2
    print(f"geoarches {GEOARCHES_COMMIT[:12]} OK: {argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
