"""Trust checks for paths reopened by CLI subprocesses (not a pickle sandbox)."""

import os
from pathlib import Path
import stat


def trusted_ancestry(path: Path, owners: set[int]) -> Path:
    """Check lexical links AND resolved parents; permit protected sticky directories."""
    for entry in (path, *path.parents):
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode) and info.st_uid not in owners:
            raise ValueError(f"Untrusted symlink owner: {entry}")
    resolved = path.resolve(strict=True)
    for directory in resolved.parents:
        info = directory.stat()
        if info.st_uid not in owners:
            raise ValueError(f"Untrusted parent owner: {directory}")
        if info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
            raise ValueError(f"Parent is writable by other users: {directory}")
    return resolved


def trusted_lora_path(value: str) -> str:
    """Allow caller-owned or installed regular checkpoints, excluding shared writes."""
    from aimip.validators import validate_visible_input_path

    original = Path(validate_visible_input_path(value, label="agent LoRA checkpoint"))
    resolved = original.resolve(strict=True)
    owners = {0, os.getuid()}
    installed = False
    for name in ("AIMIP_DATA", "AIMIP_HOME"):
        if os.environ.get(name):
            root = Path(os.environ[name]).resolve(strict=True)
            if resolved.is_relative_to(root):
                installed = True
                owners.add(root.stat().st_uid)  # installation administrator is trusted
    trusted_ancestry(original, owners)
    fd = os.open(original, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("LoRA checkpoint must be a regular file")
        if info.st_uid != os.getuid() and not installed:
            raise ValueError("LoRA checkpoint must be caller-owned or inside AIMIP_DATA/AIMIP_HOME")
        if info.st_mode & 0o022:
            raise ValueError("LoRA checkpoint must not be group/world-writable")
        current = resolved.stat()
        if (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("LoRA checkpoint changed during validation")
    finally:
        os.close(fd)
    # Reopening is safe against other-user swaps because its resolved ancestry and
    # file write permissions were checked. Caller/admin mutation is trusted here.
    return str(resolved)
