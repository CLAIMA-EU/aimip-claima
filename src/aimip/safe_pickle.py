"""Mandatory O_NOFOLLOW SHA-256 verification and restricted deserialization for Phase 2c artifacts.

Every Phase 2c checkpoint or serialized configuration file (`*.pt`, `*.pth`, `*.ckpt`,
`*.checkpoint`, `config.pkl`, `OM4_*.zip`) is opened with `O_NOFOLLOW`, verified for
trusted ownership and absence of group/world write bits on the file and the parent
directory, and streamed through SHA-256 against the pinned digest BEFORE `torch.load`,
`_RestrictedConfigUnpickler`, or `zipfile.ZipFile` reads from that same open descriptor
(`stream.seek(0)`). The cryptographic SHA-256 check on the open descriptor is the primary
integrity gate; `weights_only=True` and closed pickle global allowlists provide a second
defense-in-depth layer.
"""

from __future__ import annotations

import builtins
import collections
import contextlib
from collections.abc import Iterator
import hashlib
import io
import os
import pickle
import re
import shutil
import stat
import types
import uuid
import zipfile
from typing import Any

from .errors import ValidationError
from .validators import makedirs_no_group_other_write, strip_group_other_write, validate_safe_path

_HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def normalize_sha256(expected_sha256: str, *, label: str = "expected SHA-256") -> str:
    """Validate and normalize a 64-character lowercase hex SHA-256 digest."""
    if not isinstance(expected_sha256, str):
        raise ValidationError(f"{label} must be a 64-character hex string.")
    norm = expected_sha256.strip().lower()
    if not _HEX64_RE.match(norm):
        raise ValidationError(
            f"Invalid {label} {expected_sha256!r}: expected 64 lowercase hexadecimal characters."
        )
    return norm


def _trusted_owners_for_path(safe_path: str) -> set[int]:
    """Return trusted UIDs (`root`, current `euid`, plus non-writable `$AIMIP_DATA`/`$AIMIP_HOME` admin)."""
    owners: set[int] = {0, os.geteuid()}
    real_target = os.path.realpath(safe_path)
    for env_key in ("AIMIP_DATA", "AIMIP_HOME", "AIMIP_PLAYGROUND"):
        raw_root = (os.environ.get(env_key) or "").strip()
        if not raw_root or os.path.islink(raw_root) or not os.path.isdir(raw_root):
            continue
        try:
            rst = os.stat(raw_root)
        except OSError:
            continue
        if (rst.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) != 0:
            continue
        real_root = os.path.realpath(raw_root).rstrip("/")
        if real_root and (real_target == real_root or real_target.startswith(real_root + os.sep)):
            owners.add(rst.st_uid)
    return owners


def _verify_directory_permissions(
    dir_path: str,
    *,
    label: str,
    owners: set[int],
    allow_sticky: bool = False,
) -> None:
    """Refuse symlinked, foreign-owned, or group/world-writable directories."""
    try:
        dst = os.lstat(dir_path)
    except OSError as exc:
        raise ValidationError(f"Cannot stat directory {dir_path} for {label}: {exc}") from exc
    if stat.S_ISLNK(dst.st_mode) or not stat.S_ISDIR(dst.st_mode):
        raise ValidationError(f"Refusing symlinked or non-directory path in {label}: {dir_path}")
    if dst.st_uid not in owners:
        raise ValidationError(
            f"Refusing directory with untrusted owner (uid={dst.st_uid}) in {label}: {dir_path}"
        )
    writable_mask = dst.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if writable_mask != 0 and not (allow_sticky and bool(dst.st_mode & stat.S_ISVTX)):
        raise ValidationError(
            f"Refusing group- or world-writable directory in {label}: {dir_path} "
            f"(fix with: chmod go-w {dir_path})"
        )


@contextlib.contextmanager
def open_verified_hash_stream(
    path: str,
    expected_sha256: str,
    *,
    label: str = "serialized artifact",
    chunk_size: int = 8 * 1024 * 1024,
) -> Iterator[Any]:
    """Open *path* with `O_NOFOLLOW`, verify owner/mode and SHA-256 on the open fd, rewind, and yield a binary stream.

    Aborts with `ValidationError` BEFORE yielding if *path* is missing, a symlink,
    non-regular, foreign-owned, group- or world-writable, inside a writable non-sticky
    parent directory, or if its byte content does not match *expected_sha256*.
    """
    safe_path = validate_safe_path(path, label=label)
    want = normalize_sha256(expected_sha256, label=f"{label} SHA-256")

    if os.path.islink(safe_path):
        raise ValidationError(f"Refusing symlinked {label}: {safe_path}")

    owners = _trusted_owners_for_path(safe_path)
    parent_dir = os.path.dirname(safe_path) or "/"
    if parent_dir != "/":
        _verify_directory_permissions(
            parent_dir,
            label=f"{label} parent directory",
            owners=owners,
            allow_sticky=True,
        )

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(safe_path, flags)
    except OSError as exc:
        raise ValidationError(
            f"Cannot open {label} {safe_path} (symlink or missing file): {exc}"
        ) from exc

    stream = None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValidationError(f"Refusing non-regular {label}: {safe_path}")
        if st.st_uid not in owners:
            raise ValidationError(
                f"Refusing {label} with untrusted owner (uid={st.st_uid}): {safe_path}"
            )
        if (st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) != 0:
            raise ValidationError(
                f"Refusing group- or world-writable {label}: {safe_path} "
                f"(fix with: chmod go-w {safe_path})"
            )

        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, chunk_size)
            if not chunk:
                break
            digest.update(chunk)
        got = digest.hexdigest().lower()
        if got != want:
            raise ValidationError(
                f"SHA-256 mismatch before loading {label} ({safe_path}): "
                f"expected {want}, got {got}."
            )

        os.lseek(fd, 0, os.SEEK_SET)
        stream = os.fdopen(fd, "rb")
    except BaseException:
        if stream is not None:
            stream.close()
        else:
            try:
                os.close(fd)
            except OSError:
                pass
        raise

    with stream:
        yield stream


def torch_load_verified(
    path: str,
    expected_sha256: str,
    *,
    map_location: Any = "cpu",
    weights_only: bool = True,
    label: str = "checkpoint",
) -> Any:
    """Verify *path* against *expected_sha256* on an `O_NOFOLLOW` descriptor, then `torch.load` from that stream."""
    import torch

    with open_verified_hash_stream(path, expected_sha256, label=label) as stream:
        return torch.load(stream, map_location=map_location, weights_only=weights_only)


_SAFE_BUILTINS = frozenset(
    {
        "bool",
        "bytearray",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "range",
        "set",
        "slice",
        "str",
        "tuple",
    }
)
_SAFE_COLLECTIONS = frozenset({"OrderedDict", "defaultdict"})
_SAFE_TORCH_ACTIVATIONS = frozenset({"GELU", "Identity", "ReLU", "SiLU", "Tanh"})
_SAFE_TORCH_REBUILD = frozenset({"_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_tensor_v3"})
_SAFE_TORCH_STORAGES = frozenset(
    {
        "BFloat16Storage",
        "BoolStorage",
        "ByteStorage",
        "CharStorage",
        "ComplexDoubleStorage",
        "ComplexFloatStorage",
        "DoubleStorage",
        "FloatStorage",
        "HalfStorage",
        "IntStorage",
        "LongStorage",
        "ShortStorage",
        "TypedStorage",
        "UntypedStorage",
    }
)


def _safe_numpy_dtype(*args: Any, **kwargs: Any) -> Any:
    import numpy as np

    dt = np.dtype(*args, **kwargs)
    if dt.kind == "O" or bool(dt.hasobject):
        raise pickle.UnpicklingError("Refusing object numpy.dtype in restricted pickle.")
    return dt


def _safe_numpy_scalar(dtype: Any, raw_data: Any) -> Any:
    import numpy as np

    if not isinstance(dtype, np.dtype) or dtype.kind == "O" or bool(dtype.hasobject):
        raise pickle.UnpicklingError(f"Refusing non-primitive numpy scalar dtype: {dtype!r}")
    try:
        from numpy._core.multiarray import scalar as np_scalar  # type: ignore[import-not-found]
    except ImportError:
        from numpy.core.multiarray import scalar as np_scalar  # type: ignore[no-redef]
    return np_scalar(dtype, raw_data)


def _make_restricted_storage_pickle_module() -> types.ModuleType:
    """Build a minimal `pickle_module` allowing ONLY `torch.*Storage` and tensor rebuilders."""
    import torch
    import torch._utils as torch_utils

    class _StorageUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if module == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if module == "torch" and name in _SAFE_TORCH_STORAGES:
                return getattr(torch, name)
            if module == "torch.storage" and name in ("TypedStorage", "_TypedStorage", "UntypedStorage"):
                return getattr(torch.storage, name)
            if module == "torch._utils" and name in _SAFE_TORCH_REBUILD:
                return getattr(torch_utils, name)
            raise pickle.UnpicklingError(
                f"Refusing non-whitelisted nested storage global {module}.{name}"
            )

    mod = types.ModuleType("_aimip_restricted_storage_pickle")
    mod.Unpickler = _StorageUnpickler  # type: ignore[attr-defined]
    mod.UnpicklingError = pickle.UnpicklingError  # type: ignore[attr-defined]
    mod.load = lambda fh, **kw: _StorageUnpickler(fh, **kw).load()  # type: ignore[attr-defined]
    return mod


def _make_safe_storage_from_bytes(map_location: Any) -> Any:
    import torch

    target_loc = "cpu" if map_location is None else map_location

    def _load_storage_from_bytes(raw_bytes: bytes) -> Any:
        if not isinstance(raw_bytes, (bytes, bytearray)):
            raise pickle.UnpicklingError("Expected bytes in torch.storage._load_from_bytes")
        try:
            return torch.load(
                io.BytesIO(raw_bytes),
                map_location=target_loc,
                weights_only=True,
            )
        except (pickle.UnpicklingError, RuntimeError) as exc:
            if isinstance(exc, pickle.UnpicklingError):
                raise
            if "legacy" not in str(exc).lower():
                raise
            return torch.load(
                io.BytesIO(raw_bytes),
                map_location=target_loc,
                weights_only=False,
                pickle_module=_make_restricted_storage_pickle_module(),
            )

    return _load_storage_from_bytes


class _RestrictedConfigUnpickler(pickle.Unpickler):
    """Strict whitelist unpickler for SHA-256-verified configuration pickles (e.g. MD-1.5 `config.pkl`)."""

    def __init__(
        self,
        file_obj: Any,
        *,
        map_location: Any = "cpu",
        allow_torch_config: bool = False,
    ) -> None:
        super().__init__(file_obj)
        self._map_location = map_location
        self._allow_torch_config = allow_torch_config

    def find_class(self, module: str, name: str) -> Any:
        if module in ("builtins", "__builtin__") and name in _SAFE_BUILTINS:
            return getattr(builtins, name)
        if module == "collections" and name in _SAFE_COLLECTIONS:
            return getattr(collections, name)
        if module in ("numpy", "numpy._core.multiarray", "numpy.core.multiarray") and name == "dtype":
            return _safe_numpy_dtype
        if module in ("numpy._core.multiarray", "numpy.core.multiarray") and name == "scalar":
            return _safe_numpy_scalar
        if self._allow_torch_config:
            if module == "torch.nn.modules.activation" and name in _SAFE_TORCH_ACTIVATIONS:
                import torch.nn.modules.activation as torch_act

                return getattr(torch_act, name)
            if module == "torch" and name == "device":
                import torch

                target_loc = self._map_location
                if target_loc is None:
                    return torch.device
                return lambda *_args, **_kwargs: torch.device(target_loc)
            if module == "torch._utils" and name in _SAFE_TORCH_REBUILD:
                import torch._utils as torch_utils

                return getattr(torch_utils, name)
            if module == "torch.storage" and name == "_load_from_bytes":
                return _make_safe_storage_from_bytes(self._map_location)
        raise pickle.UnpicklingError(
            f"Refusing non-whitelisted pickle global {module}.{name}"
        )


def pickle_load_verified(
    path: str,
    expected_sha256: str,
    *,
    map_location: Any = "cpu",
    allow_torch_config: bool = False,
    label: str = "pickle configuration",
) -> Any:
    """Verify *path* against *expected_sha256* on an `O_NOFOLLOW` descriptor, then unpickle via strict global whitelist."""
    with open_verified_hash_stream(path, expected_sha256, label=label) as stream:
        try:
            return _RestrictedConfigUnpickler(
                stream,
                map_location=map_location,
                allow_torch_config=allow_torch_config,
            ).load()
        except pickle.UnpicklingError as exc:
            raise ValidationError(
                f"Blocked unsafe or corrupted {label} ({path}): {exc}"
            ) from exc


def compute_sha256sum_tree_digest(
    root_dir: str,
    *,
    label: str = "source tree",
    chunk_size: int = 4 * 1024 * 1024,
) -> tuple[str, int]:
    """Compute `(tree_sha256, file_count)` matching `LC_ALL=C find . -type f | sort | xargs sha256sum | sha256sum`.

    Refuses symlinks, foreign owners, and group- or world-writable directories or files.
    Skips a top-level `.git` directory only; nested `.git` entries are refused.
    """
    safe_root = validate_safe_path(root_dir, label=label)
    if os.path.islink(safe_root) or not os.path.isdir(safe_root):
        raise ValidationError(f"{label} is missing, not a directory, or symlinked: {safe_root}")
    owners = _trusted_owners_for_path(safe_root)
    _verify_directory_permissions(safe_root, label=label, owners=owners, allow_sticky=False)

    entries: list[tuple[str, str]] = []
    for walk_root, dirs, files in os.walk(safe_root, topdown=True, followlinks=False):
        if walk_root == safe_root:
            if ".git" in dirs:
                dirs.remove(".git")
        elif ".git" in dirs or ".git" in files:
            raise ValidationError(
                f"Refusing nested .git entry inside {label}: {os.path.join(walk_root, '.git')}"
            )
        dirs.sort()
        files.sort()
        _verify_directory_permissions(walk_root, label=label, owners=owners, allow_sticky=False)
        for dname in dirs:
            dpath = os.path.join(walk_root, dname)
            _verify_directory_permissions(dpath, label=label, owners=owners, allow_sticky=False)
        for fname in files:
            fpath = os.path.join(walk_root, fname)
            if os.path.islink(fpath):
                raise ValidationError(f"Refusing symlinked file inside {label}: {fpath}")
            rel = os.path.relpath(fpath, safe_root).replace(os.sep, "/")
            entries.append((f"./{rel}", fpath))

    entries.sort(key=lambda item: item[0])
    if not entries:
        raise ValidationError(f"{label} contains no regular files: {safe_root}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    overall = hashlib.sha256()
    for dot_rel, fpath in entries:
        try:
            fd = os.open(fpath, flags)
        except OSError as exc:
            raise ValidationError(f"Cannot inspect file {fpath} in {label}: {exc}") from exc
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise ValidationError(f"Refusing non-regular entry in {label}: {fpath}")
            if st.st_uid not in owners:
                raise ValidationError(
                    f"Refusing file with untrusted owner (uid={st.st_uid}) in {label}: {fpath}"
                )
            if (st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) != 0:
                raise ValidationError(
                    f"Refusing group- or world-writable file in {label}: {fpath} "
                    f"(fix with: chmod go-w {fpath})"
                )
            fhash = hashlib.sha256()
            while True:
                buf = os.read(fd, chunk_size)
                if not buf:
                    break
                fhash.update(buf)
            overall.update(f"{fhash.hexdigest()}  {dot_rel}\n".encode("utf-8"))
        finally:
            os.close(fd)

    return overall.hexdigest(), len(entries)


def compute_tree_digest(
    root_dir: str,
    *,
    label: str = "staged directory",
    chunk_size: int = 4 * 1024 * 1024,
) -> tuple[str, int, int]:
    """Compute deterministic `(tree_sha256, file_count, total_bytes)` over all regular files in *root_dir*.

    Refuses symlinked entries, foreign owners, and group- or world-writable directories or files.
    """
    safe_root = validate_safe_path(root_dir, label=label)
    if os.path.islink(safe_root) or not os.path.isdir(safe_root):
        raise ValidationError(f"{label} is missing, not a directory, or symlinked: {safe_root}")
    owners = _trusted_owners_for_path(safe_root)
    _verify_directory_permissions(safe_root, label=label, owners=owners, allow_sticky=False)

    rel_files: list[tuple[str, str]] = []
    for walk_root, dirs, files in os.walk(safe_root, topdown=True, followlinks=False):
        dirs.sort()
        files.sort()
        _verify_directory_permissions(walk_root, label=label, owners=owners, allow_sticky=False)
        for dname in dirs:
            dpath = os.path.join(walk_root, dname)
            _verify_directory_permissions(dpath, label=label, owners=owners, allow_sticky=False)
        for fname in files:
            fpath = os.path.join(walk_root, fname)
            if os.path.islink(fpath):
                raise ValidationError(f"Refusing symlinked file inside {label}: {fpath}")
            rel = os.path.relpath(fpath, safe_root).replace(os.sep, "/")
            rel_files.append((rel, fpath))

    rel_files.sort(key=lambda item: item[0])
    if not rel_files:
        raise ValidationError(f"{label} contains no regular files: {safe_root}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    tree_hash = hashlib.sha256()
    total_bytes = 0
    for rel, fpath in rel_files:
        try:
            fd = os.open(fpath, flags)
        except OSError as exc:
            raise ValidationError(f"Cannot inspect file {fpath} in {label}: {exc}") from exc
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise ValidationError(f"Refusing non-regular entry in {label}: {fpath}")
            if st.st_uid not in owners:
                raise ValidationError(
                    f"Refusing file with untrusted owner (uid={st.st_uid}) in {label}: {fpath}"
                )
            if (st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) != 0:
                raise ValidationError(
                    f"Refusing group- or world-writable file in {label}: {fpath} "
                    f"(fix with: chmod go-w {fpath})"
                )
            fhash = hashlib.sha256()
            while True:
                buf = os.read(fd, chunk_size)
                if not buf:
                    break
                fhash.update(buf)
            fsize = int(st.st_size)
            total_bytes += fsize
            tree_hash.update(f"{rel}\0{fsize}\0{fhash.hexdigest()}\n".encode("utf-8"))
        finally:
            os.close(fd)

    return tree_hash.hexdigest(), len(rel_files), total_bytes


def extract_verified_zip_store(
    zip_path: str,
    expected_sha256: str,
    dst_dir: str,
    *,
    label: str = "Zarr zip archive",
) -> str:
    """Verify *zip_path* on an `O_NOFOLLOW` descriptor, safely extract into *dst_dir*, and return the Zarr root path."""
    safe_dst = validate_safe_path(dst_dir, label=f"{label} destination")
    parent = os.path.dirname(safe_dst) or "/"
    makedirs_no_group_other_write(parent)
    staging = os.path.join(parent, f".aimip_zip_{uuid.uuid4().hex}")
    makedirs_no_group_other_write(staging)
    real_staging = os.path.realpath(staging)

    try:
        with open_verified_hash_stream(zip_path, expected_sha256, label=label) as stream:
            try:
                zf = zipfile.ZipFile(stream, "r")
            except zipfile.BadZipFile as exc:
                raise ValidationError(f"Corrupted {label} ({zip_path}): {exc}") from exc
            with zf:
                infos = zf.infolist()
                if not infos:
                    raise ValidationError(f"Empty {label}: {zip_path}")
                extracted_files = 0
                for info in infos:
                    raw_name = info.filename
                    if not raw_name or "\\" in raw_name or raw_name.startswith("/"):
                        raise ValidationError(
                            f"Unsafe archive member path {raw_name!r} in {zip_path}"
                        )
                    unix_ftype = (info.external_attr >> 16) & 0o170000
                    if unix_ftype == stat.S_IFLNK:
                        raise ValidationError(
                            f"Refusing symlink archive member {raw_name!r} in {zip_path}"
                        )
                    parts = [p for p in raw_name.split("/") if p]
                    if any(p in (".", "..") for p in parts):
                        raise ValidationError(
                            f"Path traversal member {raw_name!r} in {zip_path}"
                        )
                    if not parts:
                        continue
                    target = os.path.realpath(os.path.join(real_staging, *parts))
                    if target != real_staging and not target.startswith(real_staging + os.sep):
                        raise ValidationError(
                            f"Archive member {raw_name!r} escapes staging directory in {zip_path}"
                        )
                    if info.is_dir() or raw_name.endswith("/"):
                        makedirs_no_group_other_write(target)
                        continue
                    makedirs_no_group_other_write(os.path.dirname(target))
                    with zf.open(info, "r") as src_fh, open(target, "wb") as out_fh:
                        shutil.copyfileobj(src_fh, out_fh, length=1024 * 1024)
                    strip_group_other_write(target)
                    extracted_files += 1
                if extracted_files == 0:
                    raise ValidationError(f"{label} contained no regular files: {zip_path}")

        if os.path.lexists(safe_dst):
            if os.path.islink(safe_dst) or os.path.isfile(safe_dst):
                os.unlink(safe_dst)
            else:
                shutil.rmtree(safe_dst)
        os.replace(staging, safe_dst)
    finally:
        if os.path.lexists(staging):
            shutil.rmtree(staging, ignore_errors=True)

    for marker in (".zgroup", ".zmetadata", "zarr.json"):
        if os.path.isfile(os.path.join(safe_dst, marker)):
            return safe_dst
    subdirs = [
        os.path.join(safe_dst, entry)
        for entry in sorted(os.listdir(safe_dst))
        if os.path.isdir(os.path.join(safe_dst, entry))
        and not os.path.islink(os.path.join(safe_dst, entry))
    ]
    for sub in subdirs:
        for marker in (".zgroup", ".zmetadata", "zarr.json"):
            if os.path.isfile(os.path.join(sub, marker)):
                return sub
    raise ValidationError(
        f"Extracted {label} ({zip_path}) does not contain a Zarr root group (.zgroup / zarr.json)."
    )
