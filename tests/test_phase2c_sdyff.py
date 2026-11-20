"""Offline Phase 2c tests for `spherical-dyffusion` (`family="sdyff"`, zero network, zero GPU).

Covers:
1. Provenance & Apache-2.0 / CC-BY-4.0 contract (`Rose-STL-Lab/spherical-dyffusion@a10b3e7f`,
   `ai2cm/modulus@94f62e1c`, `salv47/spherical-dyffusion@56fcb5d9`, 12 Zenodo `10791087` files
   with `ace_ckpt.tar` excluded, `aimip show` & `aimip cite`).
2. Source archive `LC_ALL=C` `sha256sum` tree verification (`190` files), `ensure_sdyff_code`
   (`file://` archive), and symlink/traversal/byte-tamper refusal.
3. Pre-pickle `O_NOFOLLOW` SHA-256 gate + restricted `pickle_module` (`make_sdyff_restricted_pickle_module()`,
   allowing ONLY `builtins` `dict`/`list`/`int`(`long`), `collections` `OrderedDict`/`defaultdict`,
   `omegaconf` `ContainerMetadata`/`Metadata`/`DictConfig`/`ListConfig`/`AnyNode`, `typing.Any`,
   `torch._utils._rebuild_tensor_v2`, and `torch` `FloatStorage`/`IntStorage`/`DoubleStorage`):
   - `torch.load` call count is `0` on SHA mismatch or symlink!
   - Non-allowlisted globals (`os.system` / `posix.system`, `subprocess.Popen`) are refused even
     when SHA-256 matches!
4. Cycle schedule & YAML renderer: `--start 2021-01-01 --steps 240` (`60` days, `coarsen_factor=4`)
   yields `2` complete cycle months (`0001-01` = 31d/124 steps, `0001-02` = 28d/112 steps),
   `log_to_wandb=False`, `overrides.seed`, and rejects `--ic` / non-2021 / partial-only horizons.
5. End-to-end `services.plan -> prepare -> exec_fme -> run_sdyff -> summarize_run`:
   - Verifies all 34 predicted variables + `total_water_path` on `180 x 360`.
   - Verifies `surface_temperature` is renamed to `surface_temperature_prescribed_ocean` and
     `aimip summary` reports predicted `air_temperature_7` (`K`), never prescribed ocean SST!
   - Verifies nominal cycle labels `["0001-01", "0001-02"]` (never relabelled to `1995..` or `2021..`),
     exclusion of `timestep=0` initial state and incomplete trailing Day 60, and `EXIT_UNSTABLE == 5`.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
from typing import Any

import pytest

from aimip.emu import _job_env, precheck_emu_request
from aimip.emu_catalog import artifacts, registry_entries, required_files
from aimip.errors import ConfigError, ValidationError
from aimip.lora import sha256_file
from aimip.plan_types import RunSpec
from aimip.registry import DATA_PLACEHOLDER, _parse_model_entry, build_default_registry_payload
from aimip.runner import exec_fme
from aimip.safe_pickle import compute_sha256sum_tree_digest
from aimip.sdyff import (
    EXIT_UNSTABLE,
    SDYFF_CALENDAR_NOTE,
    SDYFF_COARSEN_FACTOR,
    SDYFF_CODE_COMMIT,
    SDYFF_CODE_REL_DIR,
    SDYFF_DATA_REL_DIR,
    SDYFF_FORECASTER_CKPT_REL_PATH,
    SDYFF_FORECASTER_CKPT_SHA256,
    SDYFF_FORECASTER_CKPT_SIZE,
    SDYFF_GITHUB_ARCHIVE_SHA256,
    SDYFF_GITHUB_ARCHIVE_SIZE,
    SDYFF_HF_REPO,
    SDYFF_HF_REVISION,
    SDYFF_INTERPOLATOR_CFG_REL_PATH,
    SDYFF_INTERPOLATOR_CFG_SHA256,
    SDYFF_INTERPOLATOR_CFG_SIZE,
    SDYFF_INTERPOLATOR_CKPT_REL_PATH,
    SDYFF_INTERPOLATOR_CKPT_SHA256,
    SDYFF_INTERPOLATOR_CKPT_SIZE,
    SDYFF_MODULUS_ARCHIVE_SHA256,
    SDYFF_MODULUS_ARCHIVE_SIZE,
    SDYFF_MODULUS_COMMIT,
    SDYFF_MODULUS_TREE_FILES,
    SDYFF_MODULUS_TREE_SHA256,
    SDYFF_OFFLINE_WANDB_ENTITY,
    SDYFF_PREDICTED_VARS,
    SDYFF_PRESCRIBED_SURFACE_VAR,
    SDYFF_SUMMARY_TEMP_VAR,
    SDYFF_TREE_FILES,
    SDYFF_TREE_SHA256,
    SDYFF_ZENODO_FILES,
    SDYFF_ZENODO_RECORD,
    extract_and_verify_sdyff_archive,
    inject_sdyff_local_interpolator_overrides,
    install_sdyff_apex_shim,
    install_sdyff_offline_runtime_hooks,
    render_sdyff_runtime_config,
    run_sdyff,
    sdyff_plan_schedule,
    verified_sdyff_loaders,
    verify_sdyff_checkpoint,
    verify_sdyff_code_dir,
)
from aimip.services import Authorization, MemoryApprovalBroker, cite, current_user, plan, prepare, show
from aimip.setup_data import ensure_sdyff_code, load_lock_file
from aimip.status import SUMMARY_TEMP_VARS, summarize_run


class _DisallowedPayload:
    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (subprocess.Popen, (["/bin/true"],))


class _OsSystemPayload:
    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (os.system, ("echo SHOULD_NEVER_RUN",))


def _write_synthetic_sdyff_ckpt(path: Path, *, tensor_val: float = 1.0) -> str:
    """Create a Lightning-style `.ckpt` requiring `OmegaConf` + `defaultdict` + `OrderedDict` globals."""
    from omegaconf import OmegaConf
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    dd: defaultdict[str, list[int]] = defaultdict(list)
    dd["steps"].append(1)
    omega_cfg = OmegaConf.create(
        {
            "seed": 11,
            "out_names": list(SDYFF_PREDICTED_VARS[:4]),
            "nested": {"use_ema": True, "layers": [0, 1, 2]},
        }
    )
    ckpt = {
        "epoch": 31,
        "global_step": 86560,
        "state_dict": OrderedDict(
            [
                ("model.pos_embed", torch.full((1, 4, 4), tensor_val, dtype=torch.float32)),
                ("model.head.weight", torch.ones((34, 4), dtype=torch.float32)),
            ]
        ),
        "hyper_parameters": omega_cfg,
        "callbacks_meta": dd,
    }
    torch.save(ckpt, str(path))
    return sha256_file(str(path))


def _build_synthetic_190_file_tar(archive_path: Path, *, include_symlink: bool = False) -> tuple[int, str, str]:
    """Build a deterministic 190-file `.tar.gz` matching `SDYFF_TREE_FILES == 190`."""
    src_stage = archive_path.parent / f"_stage_{archive_path.stem}"
    top = src_stage / f"spherical-dyffusion-{SDYFF_CODE_COMMIT}"
    (top / "data_statistics").mkdir(parents=True)
    (top / "src").mkdir(parents=True)
    (top / "LICENSE").write_text("Apache License, Version 2.0\n", encoding="utf-8")
    (top / "run_inference.py").write_text("# placeholder\n", encoding="utf-8")
    (top / "data_statistics" / "centering.nc").write_bytes(b"CENTER")
    (top / "data_statistics" / "scaling.nc").write_bytes(b"SCALE")
    # 4 files above + 186 module files = 190 files total.
    for i in range(186):
        (top / "src" / f"mod_{i:03d}.py").write_text(f"VAL = {i}\n", encoding="utf-8")

    tree_sha, n_files = compute_sha256sum_tree_digest(str(top))
    assert n_files == SDYFF_TREE_FILES == 190

    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(top, arcname=top.name)
        if include_symlink:
            sym = tarfile.TarInfo(name=f"{top.name}/bad_link")
            sym.type = tarfile.SYMTYPE
            sym.linkname = "/etc/passwd"
            tf.addfile(sym)

    raw_bytes = archive_path.read_bytes()
    return len(raw_bytes), hashlib.sha256(raw_bytes).hexdigest(), tree_sha


def test_sdyff_provenance_pins_and_card_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify code, `ai2cm/modulus`, HF weight, and 12 Zenodo `10791087` pins + card/citation contract."""
    import aimip

    monkeypatch.setenv("AIMIP_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    assert SDYFF_CODE_COMMIT == "a10b3e7f1ae529193537a128c86b531adb45f23b"
    assert SDYFF_GITHUB_ARCHIVE_SIZE == 284_547
    assert len(SDYFF_GITHUB_ARCHIVE_SHA256) == 64
    assert SDYFF_TREE_FILES == 190
    assert len(SDYFF_TREE_SHA256) == 64

    assert SDYFF_MODULUS_COMMIT == "94f62e1ce2083640829ec12d80b00619c40a47f8"
    assert SDYFF_MODULUS_ARCHIVE_SIZE == 529_142
    assert len(SDYFF_MODULUS_ARCHIVE_SHA256) == 64
    assert SDYFF_MODULUS_TREE_FILES == 219
    assert len(SDYFF_MODULUS_TREE_SHA256) == 64

    assert len(SDYFF_PREDICTED_VARS) == 34
    assert SDYFF_SUMMARY_TEMP_VAR == "air_temperature_7"
    assert SDYFF_SUMMARY_TEMP_VAR in SUMMARY_TEMP_VARS
    assert SDYFF_PRESCRIBED_SURFACE_VAR not in SUMMARY_TEMP_VARS

    reqs = required_files("spherical-dyffusion")
    assert len(reqs) == 15   # 3 HF + 12 Zenodo monthly files
    assert all("ace_ckpt.tar" not in r for r in reqs)

    pkg_dir = Path(aimip.__file__).resolve().parent
    lock_files = load_lock_file(str(pkg_dir / "aimip.lock"))["files"]

    for rel, exp_sz, exp_sha in (
        (SDYFF_FORECASTER_CKPT_REL_PATH, SDYFF_FORECASTER_CKPT_SIZE, SDYFF_FORECASTER_CKPT_SHA256),
        (SDYFF_INTERPOLATOR_CKPT_REL_PATH, SDYFF_INTERPOLATOR_CKPT_SIZE, SDYFF_INTERPOLATOR_CKPT_SHA256),
        (SDYFF_INTERPOLATOR_CFG_REL_PATH, SDYFF_INTERPOLATOR_CFG_SIZE, SDYFF_INTERPOLATOR_CFG_SHA256),
    ):
        le = lock_files[rel]
        assert le["size"] == exp_sz
        assert le["sha256"] == exp_sha
        assert le["source"]["kind"] == "hf"
        assert le["source"]["repo"] == SDYFF_HF_REPO
        assert le["source"]["revision"] == SDYFF_HF_REVISION

    assert len(SDYFF_ZENODO_FILES) == 12
    for fname, zspec in SDYFF_ZENODO_FILES.items():
        rel = f"{SDYFF_DATA_REL_DIR}/{fname}"
        assert rel in reqs
        le = lock_files[rel]
        assert le["size"] == zspec["size"]
        assert le["sha256"] == zspec["sha256"]
        assert le["source"]["kind"] == "zenodo"
        assert le["source"]["record"] == SDYFF_ZENODO_RECORD
        assert le["source"]["md5"] == zspec["md5"]

    emu_sd = [a for a in artifacts() if "spherical-dyffusion" in a["models"]]
    assert len(emu_sd) == 15

    card = show("spherical-dyffusion")
    assert card["family"] == "sdyff"
    assert (
        card["verified_on"] is not None
        and "A100" in card["verified_on"]
        and "283.0611" in card["verified_on"]
        and "1.4.0" in card["verified_on"]
        and "job " not in card["verified_on"]
    )
    assert "Apache-2.0" in card["licence"] and "CC-BY-4.0" in card["licence"]
    assert "air_temperature_7" in card["limitations"]
    assert "surface_temperature_prescribed_ocean" in card["limitations"]
    assert "0001-01" in card["limitations"]
    assert "FusedLayerNorm" in card["limitations"] and "torch.nn.LayerNorm" in card["limitations"]
    assert "instance_norm" in card["limitations"]

    # Verify isolated `venv-sdyff` apex shim installs cleanly and matches `torch.nn.LayerNorm` identically.
    import importlib
    import torch

    fake_venv = tmp_path / "venv-sdyff"
    fake_sp = fake_venv / "lib" / "python3.12" / "site-packages"
    fake_sp.mkdir(parents=True)
    shim_info = install_sdyff_apex_shim(str(fake_venv), site_packages_dir=str(fake_sp))
    assert Path(shim_info["normalization.py"]).is_file()
    monkeypatch.syspath_prepend(str(fake_sp))
    monkeypatch.delitem(sys.modules, "apex", raising=False)
    monkeypatch.delitem(sys.modules, "apex.normalization", raising=False)
    apex_norm = importlib.import_module("apex.normalization")
    assert apex_norm.FusedLayerNorm is torch.nn.LayerNorm
    torch.manual_seed(0)
    ln_ref = torch.nn.LayerNorm((8, 16), eps=1e-6)
    ln_shim = apex_norm.FusedLayerNorm((8, 16), eps=1e-6)
    ln_shim.load_state_dict(ln_ref.state_dict())
    x_in = torch.randn(2, 4, 8, 16)
    assert torch.equal(ln_shim(x_in), ln_ref(x_in))

    cit = cite("spherical-dyffusion")
    assert "2406.14798" in cit["bibtex"] and "NeurIPS 2024" in cit["bibtex"]
    assert any("2406.14798" in b and "NeurIPS 2024" in b for b in cit["entries"])


def test_sdyff_archive_extraction_tree_digest_and_tamper_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 190-file `LC_ALL=C` tree SHA-256 verification and refusal of symlinks / tampered files."""
    import aimip.sdyff as sdyff_mod

    tar_ok = tmp_path / "sd_ok.tar.gz"
    sz, ar_sha, tree_sha = _build_synthetic_190_file_tar(tar_ok)

    dest = tmp_path / "data"
    monkeypatch.setattr(sdyff_mod, "SDYFF_GITHUB_ARCHIVE_SIZE", sz)
    monkeypatch.setattr(sdyff_mod, "SDYFF_GITHUB_ARCHIVE_SHA256", ar_sha)
    monkeypatch.setattr(sdyff_mod, "SDYFF_TREE_SHA256", tree_sha)

    info = ensure_sdyff_code(
        str(dest),
        archive_url=f"file://{tar_ok}",
        expected_archive_sha256=ar_sha,
        expected_archive_size=sz,
        expected_tree_sha256=tree_sha,
        expected_tree_files=190,
    )
    assert info["downloaded"] is True
    assert info["files"] == 190
    assert info["tree_sha256"] == tree_sha

    cached = ensure_sdyff_code(
        str(dest),
        archive_url=f"file://{tar_ok}",
        expected_archive_sha256=ar_sha,
        expected_archive_size=sz,
        expected_tree_sha256=tree_sha,
        expected_tree_files=190,
    )
    assert cached["downloaded"] is False

    # Tampering 1 file inside `code/spherical-dyffusion` causes `verify_sdyff_code_dir` to fail closed!
    mod0 = dest / SDYFF_CODE_REL_DIR / "src" / "mod_000.py"
    orig_bytes = mod0.read_bytes()
    mod0.write_bytes(b"VAL = 999999\n")
    with pytest.raises(ValidationError, match="tree digest/count mismatch"):
        verify_sdyff_code_dir(str(dest), expected_tree_sha256=tree_sha, expected_tree_files=190)
    mod0.write_bytes(orig_bytes)

    # Archive containing a symlink is refused before publishing!
    tar_sym = tmp_path / "sd_sym.tar.gz"
    sz_s, sha_s, _ = _build_synthetic_190_file_tar(tar_sym, include_symlink=True)
    with pytest.raises(ValidationError, match="Refusing unsupported tar entry type"):
        extract_and_verify_sdyff_archive(
            str(tar_sym),
            str(tmp_path / "sym_out"),
            expected_archive_sha256=sha_s,
            expected_archive_size=sz_s,
            expected_tree_sha256=tree_sha,
            expected_tree_files=190,
        )


def test_sdyff_safe_globals_pre_pickle_sha_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `verify_sdyff_checkpoint` loads OmegaConf + `defaultdict` `.ckpt` via restricted `pickle_module` and refuses RCE globals."""
    import pickle
    import torch

    ckpt_p = tmp_path / "forecaster.ckpt"
    good_sha = _write_synthetic_sdyff_ckpt(ckpt_p, tensor_val=3.25)

    # PyTorch's built-in `weights_only=True` unpickler cannot load OmegaConf / `defaultdict` `.ckpt` files.
    with pytest.raises(pickle.UnpicklingError):
        torch.load(str(ckpt_p), map_location="cpu", weights_only=True)

    # `verify_sdyff_checkpoint` succeeds cleanly via `make_sdyff_restricted_pickle_module()` after SHA-256 match!
    meta = verify_sdyff_checkpoint(str(ckpt_p), good_sha)
    assert meta["n_tensors"] == 2
    assert meta["epoch"] == 31

    load_calls = 0
    real_load = torch.load

    def _spy_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal load_calls
        assert kwargs.get("weights_only") is False
        pm = kwargs.get("pickle_module")
        assert pm is not None and pm.__name__ == "_aimip_sdyff_restricted_pickle"
        load_calls += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy_load)

    # 1. Wrong expected SHA -> fails BEFORE `torch.load` (`load_calls == 0`)
    with pytest.raises(ValidationError, match="SHA-256 mismatch before loading"):
        verify_sdyff_checkpoint(str(ckpt_p), "f" * 64)
    assert load_calls == 0

    # 2. 1-byte corruption -> fails BEFORE `torch.load` (`load_calls == 0`)
    raw = ckpt_p.read_bytes()
    ckpt_p.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))
    with pytest.raises(ValidationError, match="SHA-256 mismatch before loading"):
        verify_sdyff_checkpoint(str(ckpt_p), good_sha)
    assert load_calls == 0
    ckpt_p.write_bytes(raw)

    # 3. Symlink -> refused via `O_NOFOLLOW` BEFORE `torch.load` (`load_calls == 0`)
    sym_p = tmp_path / "sym.ckpt"
    sym_p.symlink_to(ckpt_p)
    with pytest.raises(ValidationError):
        verify_sdyff_checkpoint(str(sym_p), good_sha)
    assert load_calls == 0

    # 4. Non-allowlisted `os.system` / `posix.system` global is refused EVEN with matching SHA!
    evil_os = tmp_path / "evil_os.ckpt"
    torch.save({"state_dict": OrderedDict(), "evil": _OsSystemPayload()}, str(evil_os))
    evil_os_sha = sha256_file(str(evil_os))
    with pytest.raises(ValidationError, match=r"(?:os|posix)\.system"):
        verify_sdyff_checkpoint(str(evil_os), evil_os_sha)

    # 5. Non-allowlisted `subprocess.Popen` (`commands.Popen` under protocol-2 `fix_imports`) is refused EVEN with matching SHA!
    evil_p = tmp_path / "evil_popen.ckpt"
    torch.save({"state_dict": OrderedDict(), "evil": _DisallowedPayload()}, str(evil_p))
    evil_sha = sha256_file(str(evil_p))
    with pytest.raises(ValidationError, match=r"(?:subprocess|commands)\.Popen"):
        verify_sdyff_checkpoint(str(evil_p), evil_sha)

    # 6. `verified_sdyff_loaders` loads `torch.load` and `OmegaConf.load` from verified fds and refuses unlisted/tampered paths!
    from omegaconf import OmegaConf

    monkeypatch.setattr(torch, "load", real_load)
    cfg_y = tmp_path / "interpolator.yaml"
    cfg_y.write_text("model: sfno\nembed_dim: 256\n", encoding="utf-8")
    cfg_y_sha = sha256_file(str(cfg_y))
    unlisted_y = tmp_path / "unlisted.yaml"
    unlisted_y.write_text("evil: true\n", encoding="utf-8")

    with verified_sdyff_loaders({str(ckpt_p): good_sha, str(cfg_y): cfg_y_sha}):
        ck_ok = torch.load(str(ckpt_p), map_location="cpu")
        assert ck_ok["epoch"] == 31
        oc_ok = OmegaConf.load(str(cfg_y))
        assert oc_ok["embed_dim"] == 256
        with pytest.raises(ValidationError, match="unverified Spherical DYffusion torch.load path"):
            torch.load(str(evil_p), map_location="cpu")
        with pytest.raises(ValidationError, match="unverified Spherical DYffusion OmegaConf.load path"):
            OmegaConf.load(str(unlisted_y))
        cfg_y.write_text("model: tampered_target\n", encoding="utf-8")
        try:
            with pytest.raises(ValidationError, match="SHA-256 mismatch before loading"):
                OmegaConf.load(str(cfg_y))
        finally:
            cfg_y.write_text("model: sfno\nembed_dim: 256\n", encoding="utf-8")

    # Stream provenance with two harmless distinct checkpoints and two harmless distinct YAML configs:
    # fd-backed streams bound to the verified file succeed; anonymous and mismatched-descriptor streams fail closed!
    ckpt_b = tmp_path / "second_harmless.ckpt"
    torch.save({"epoch": 99, "state_dict": OrderedDict({"w": torch.full((1,), 3.0)})}, str(ckpt_b))
    cfg_b = tmp_path / "second_harmless.yaml"
    cfg_b.write_text("model: sfno\nembed_dim: 512\n", encoding="utf-8")

    class _FakeLocalOpener:
        def __init__(self, claimed_path: str, raw_fh: Any) -> None:
            self.path = claimed_path
            self.f = raw_fh

        def fileno(self) -> int:
            return int(self.f.fileno())

    with verified_sdyff_loaders({str(ckpt_p): good_sha, str(cfg_y): cfg_y_sha}) as seen_sd:
        with open(ckpt_p, "rb") as fh_ck_ok:
            assert torch.load(fh_ck_ok, map_location="cpu")["epoch"] == 31
        assert os.path.realpath(ckpt_p) in seen_sd
        with open(cfg_y, "r", encoding="utf-8") as fh_y_ok:
            assert OmegaConf.load(fh_y_ok)["embed_dim"] == 256
        with pytest.raises(ValidationError, match="unbound anonymous stream"):
            torch.load(io.BytesIO(ckpt_b.read_bytes()), map_location="cpu")
        with pytest.raises(ValidationError, match="unbound anonymous stream"):
            OmegaConf.load(io.StringIO(cfg_b.read_text(encoding="utf-8")))
        with open(ckpt_b, "rb") as fh_ck_b:
            with pytest.raises(ValidationError, match="mismatched file provenance"):
                torch.load(_FakeLocalOpener(str(ckpt_p), fh_ck_b), map_location="cpu")
        with open(cfg_b, "r", encoding="utf-8") as fh_y_b:
            with pytest.raises(ValidationError, match="mismatched file provenance"):
                OmegaConf.load(_FakeLocalOpener(str(cfg_y), fh_y_b))


def test_sdyff_schedule_240_steps_two_months_and_runtime_yaml(tmp_path: Path) -> None:
    """Verify `--steps 240` (`60` days) -> `2` complete cycle months (`0001-01`, `0001-02`), `coarsen_factor=4`, no W&B."""
    import yaml

    sched = sdyff_plan_schedule("2021-01-01T00:00:00", 240)
    assert sched["coarsen_factor"] == SDYFF_COARSEN_FACTOR == 4
    assert sched["n_days"] == 60
    assert sched["start_index"] == 0
    assert [m["label"] for m in sched["complete_months"]] == ["0001-01", "0001-02"]
    assert [m["days"] for m in sched["complete_months"]] == [31, 28]
    assert [m["steps_6h"] for m in sched["complete_months"]] == [124, 112]

    # Multi-year wrap (`2` full 365-day cycle years = `2920` steps -> `0001-01 .. 0002-12`).
    sched_2y = sdyff_plan_schedule("2021-01-01T00:00:00", 2920)
    assert len(sched_2y["complete_months"]) == 24
    assert sched_2y["complete_months"][0]["label"] == "0001-01"
    assert sched_2y["complete_months"][-1]["label"] == "0002-12"
    assert sched_2y["n_repeats"] >= 3

    payload = build_default_registry_payload(root_placeholder=DATA_PLACEHOLDER)
    entry = next(e for e in payload["models"] if e["name"] == "spherical-dyffusion")
    spec_obj = _parse_model_entry(dict(entry), None, str(tmp_path))

    dummy_ic = tmp_path / "ic.nc"
    dummy_ic.write_bytes(b"nc")
    for bad_kw, match in (
        ({"ic_path": str(dummy_ic)}, "not supported"),
        ({"lora_path": str(dummy_ic)}, "not supported"),
        ({"start_iso": "1995-01-01T00:00:00"}, "2021-MM-DDT00:00:00"),
        ({"start_iso": "2021-01-01T06:00:00"}, "2021-MM-DDT00:00:00"),
        ({"steps": 242}, "multiple of 4"),
        ({"steps": 80}, "0 complete calendar months"),
    ):
        kw: dict[str, Any] = {
            "start_iso": "2021-01-01T00:00:00",
            "steps": 240,
            "members": 1,
            "forcing_set_name": None,
            "ic_path": None,
        }
        kw.update(bad_kw)
        with pytest.raises(ConfigError, match=match):
            precheck_emu_request(spec_obj, **kw)

    cfg_path = tmp_path / "sdyff_runtime_config.yaml"
    render_sdyff_runtime_config(
        forecaster_ckpt="/data/models/forecaster.ckpt",
        interpolator_ckpt="/data/models/interpolator.ckpt",
        interpolator_cfg="/data/models/interpolator.yaml",
        stats_dir="/data/code/spherical-dyffusion/data_statistics",
        data_dir="/data/inputs/spherical-dyffusion/fv3gfs-2021",
        experiment_dir=str(tmp_path / "raw"),
        start_iso="2021-01-01T00:00:00",
        steps=240,
        members=3,
        seed=17,
        dst_config_path=str(cfg_path),
    )
    rendered = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert rendered["logging"]["log_to_wandb"] is False
    assert rendered["logging"]["entity"] is None
    assert rendered["data_writer"]["time_coarsen"]["coarsen_factor"] == 4
    assert rendered["n_forward_steps"] == 240
    assert rendered["n_ensemble_members"] == 3
    assert rendered["overrides"]["seed"] == 17
    diff_cfg = rendered["overrides"]["diffusion_config"]
    assert diff_cfg["interpolator_local_checkpoint_path"] == "/data/models/interpolator.ckpt"
    assert diff_cfg["interpolator_local_config_path"] == "/data/models/interpolator.yaml"
    assert diff_cfg["interpolator_wandb_kwargs"]["entity"] == SDYFF_OFFLINE_WANDB_ENTITY
    assert "hf:" not in cfg_path.read_text(encoding="utf-8")

    with pytest.raises(ValidationError, match="Refusing remote URI"):
        render_sdyff_runtime_config(
            forecaster_ckpt="/data/models/forecaster.ckpt",
            interpolator_ckpt="hf:salv47/spherical-dyffusion/interpolator-sfno-best-val_avg_crps.ckpt",
            interpolator_cfg="/data/models/interpolator.yaml",
            stats_dir="/data/code/spherical-dyffusion/data_statistics",
            data_dir="/data/inputs/spherical-dyffusion/fv3gfs-2021",
            experiment_dir=str(tmp_path / "raw"),
            start_iso="2021-01-01T00:00:00",
            steps=240,
            members=1,
            seed=0,
            dst_config_path=str(tmp_path / "bad_hf.yaml"),
        )


def test_sdyff_offline_from_state_injects_local_interpolator_and_blocks_wandb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instantiate from a fake forecaster `.ckpt` lacking local interpolator paths and verify offline injection."""
    import importlib
    import types
    from omegaconf import OmegaConf
    import torch

    env_job = _job_env(str(tmp_path / "job_out"))
    assert env_job["WANDB_MODE"] == "disabled"
    assert env_job["WANDB_DISABLED"] == "true"
    assert env_job["HF_HUB_OFFLINE"] == "1"

    ipol_ckpt = tmp_path / "data" / SDYFF_INTERPOLATOR_CKPT_REL_PATH
    ipol_cfg = tmp_path / "data" / SDYFF_INTERPOLATOR_CFG_REL_PATH
    _write_synthetic_sdyff_ckpt(ipol_ckpt, tensor_val=2.25)
    ipol_cfg.parent.mkdir(parents=True, exist_ok=True)
    ipol_cfg.write_text(
        "logger:\n  wandb:\n    id: 01HER6GMAMHXPA62240XSV4TD4\nmodel:\n  name: sfno\n",
        encoding="utf-8",
    )

    # Poison `wandb` in `sys.modules` so any `wandb.api.default_entity`, `wandb.Api()`, or `wandb.login()`
    # immediately fails with `AssertionError` if any code path touches the network.
    class _ForbiddenWandbApiSingleton:
        @property
        def default_entity(self) -> str:
            raise AssertionError("W&B network access forbidden (wandb.api.default_entity)")

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.run = None  # type: ignore[attr-defined]
    fake_wandb.api = _ForbiddenWandbApiSingleton()  # type: ignore[attr-defined]
    fake_wandb.Api = lambda *a, **kw: (_ for _ in ()).throw(  # type: ignore[attr-defined]
        AssertionError("W&B network access forbidden (wandb.Api)")
    )
    fake_wandb.login = lambda *a, **kw: (_ for _ in ()).throw(  # type: ignore[attr-defined]
        AssertionError("W&B network access forbidden (wandb.login)")
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    # Mirror upstream `Rose-STL-Lab/spherical-dyffusion@a10b3e7f` module layout & call chain:
    # `MultiStepStepper.from_state` -> `update_dict_with_other` -> `DYffusion` ->
    # `reload_checkpoint_from_wandb` -> `wandb_api.get_entity(entity)` -> local config & ckpt load.
    code_dir = tmp_path / "sdyff_code"
    for rel_pkg in (
        "src",
        "src/utilities",
        "src/diffusion",
        "src/ace_inference",
        "src/ace_inference/core",
        "src/ace_inference/inference",
        "src/ace_inference/inference/data_writer",
    ):
        pkg_p = code_dir / rel_pkg
        pkg_p.mkdir(parents=True, exist_ok=True)
        (pkg_p / "__init__.py").write_text("", encoding="utf-8")

    (code_dir / "src/utilities/utils.py").write_text(
        "def update_dict_with_other(d1, other):\n"
        "    diff = []\n"
        "    for k, v in (other or {}).items():\n"
        "        if isinstance(v, dict) and d1.get(k) is not None:\n"
        "            d1[k], sub = update_dict_with_other(d1.get(k, {}), v)\n"
        "            diff += [f'{k}.{x}' for x in sub]\n"
        "        else:\n"
        "            if d1.get(k) != v:\n"
        "                diff.append(f'{k}: {d1.get(k, None)} -> {v}')\n"
        "            d1[k] = v\n"
        "    return d1, diff\n",
        encoding="utf-8",
    )
    (code_dir / "src/utilities/wandb_api.py").write_text(
        "import os, wandb\n"
        "_ENTITY = None\n"
        "PROJECT = 'Spherical-DYffusion'\n"
        "def get_entity(entity=None):\n"
        "    if entity is None:\n"
        "        return _ENTITY or wandb.api.default_entity\n"
        "    return entity\n"
        "def get_project_train(project=None):\n"
        "    return project or PROJECT\n"
        "def get_api(wandb_api=None, timeout=100):\n"
        "    return wandb_api or wandb.Api(timeout=timeout)\n"
        "def restore_model_from_wandb_cloud(run_path, local_checkpoint_path=None, **kw):\n"
        "    entity, project, wandb_id = run_path.split('/')\n"
        "    if isinstance(local_checkpoint_path, str):\n"
        "        if local_checkpoint_path.startswith('hf:'):\n"
        "            raise AssertionError('HF Hub network download forbidden')\n"
        "        return os.path.abspath(local_checkpoint_path)\n"
        "    return get_api().run(run_path)\n",
        encoding="utf-8",
    )
    (code_dir / "src/interface.py").write_text(
        "import os, torch\n"
        "from omegaconf import OmegaConf\n"
        "import src.utilities.wandb_api as wandb_api\n"
        "def reload_checkpoint_from_wandb(run_id, entity=None, project=None, config_path=None, local_checkpoint_path=None, **kw):\n"
        "    entity, project = wandb_api.get_entity(entity), project or wandb_api.get_project_train()\n"
        "    run_path = f'{entity}/{project}/{str(run_id).strip()}'\n"
        "    if config_path is None or str(config_path).startswith('hf:'):\n"
        "        raise AssertionError(f'Expected local config_path, got {config_path!r}')\n"
        "    cfg = OmegaConf.load(os.path.abspath(config_path))\n"
        "    ckpt_path = wandb_api.restore_model_from_wandb_cloud(run_path, local_checkpoint_path=local_checkpoint_path)\n"
        "    assert os.path.isfile(ckpt_path)\n"
        "    assert str(cfg.logger.wandb.id) == str(run_id)\n"
        "    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)\n"
        "    return {'model': {'ckpt_path': ckpt_path, 'config_path': os.path.abspath(config_path), 'epoch': state['epoch']}}\n",
        encoding="utf-8",
    )
    (code_dir / "src/diffusion/dyffusion.py").write_text(
        "from src.interface import reload_checkpoint_from_wandb\n"
        "class DYffusion:\n"
        "    def __init__(self, interpolator=None, interpolator_run_id=None, interpolator_local_checkpoint_path=True,\n"
        "                 interpolator_local_config_path=None, interpolator_wandb_kwargs=None, **kwargs):\n"
        "        wkw = dict(interpolator_wandb_kwargs or {})\n"
        "        self.interpolator = reload_checkpoint_from_wandb(\n"
        "            run_id=interpolator_run_id,\n"
        "            local_checkpoint_path=interpolator_local_checkpoint_path,\n"
        "            config_path=interpolator_local_config_path,\n"
        "            **wkw,\n"
        "        )['model']\n"
        "        self.extra_kwargs = kwargs\n",
        encoding="utf-8",
    )
    (code_dir / "src/ace_inference/core/stepper_multistep.py").write_text(
        "import importlib\n"
        "from omegaconf import OmegaConf\n"
        "from src.utilities.utils import update_dict_with_other\n"
        "class MultiStepStepper:\n"
        "    def __init__(self, diffusion_model, actual_hparams):\n"
        "        self.diffusion_model = diffusion_model\n"
        "        self.actual_hparams = actual_hparams\n"
        "    @classmethod\n"
        "    def from_state(cls, state, load_optimizer=True, overrides=None):\n"
        "        actual_hparams, _ = update_dict_with_other(state['hyper_parameters'], overrides or {})\n"
        "        dcfg = OmegaConf.to_container(actual_hparams['diffusion_config'], resolve=True) \\\n"
        "            if OmegaConf.is_config(actual_hparams['diffusion_config']) else dict(actual_hparams['diffusion_config'])\n"
        "        mod_name, cls_name = dcfg.pop('_target_').rsplit('.', 1)\n"
        "        target_cls = getattr(importlib.import_module(mod_name), cls_name)\n"
        "        return cls(target_cls(**dcfg), actual_hparams)\n",
        encoding="utf-8",
    )
    (code_dir / "src/ace_inference/inference/data_writer/main.py").write_text(
        "class DataWriter:\n"
        "    def __init__(self, path, n_samples, n_timesteps, metadata, coords,\n"
        "                 enable_prediction_netcdfs, enable_video_netcdfs, time_coarsen=None, n_ensemble_members=1):\n"
        "        self.n_samples = n_samples\n"
        "        self.records = []\n"
        "    def append_batch(self, target, prediction, start_timestep, start_sample, batch_times=None):\n"
        "        self.records.append((target, prediction, start_timestep, start_sample))\n",
        encoding="utf-8",
    )

    saved_src_mods = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in list(saved_src_mods):
        sys.modules.pop(k, None)
    monkeypatch.syspath_prepend(str(code_dir))
    try:
        def _make_forecaster_state() -> dict[str, Any]:
            # Exact shape of `forecaster-sfno-best-inference_avg_crps.ckpt`: `struct=True` DictConfig
            # carrying `interpolator_run_id` and LACKING `interpolator_local_checkpoint_path` / `config_path`!
            diff_cfg = OmegaConf.create(
                {
                    "_target_": "src.diffusion.dyffusion.DYffusion",
                    "timesteps": 6,
                    "interpolator": None,
                    "interpolator_run_id": "01HER6GMAMHXPA62240XSV4TD4",
                    "interpolator_checkpoint_path": True,
                    "interpolator_wandb_ckpt_filename": "best-val_avg_16ens_mems_weighted_crps.ckpt",
                }
            )
            OmegaConf.set_struct(diff_cfg, True)
            hparams = OmegaConf.create({"seed": 11, "diffusion_config": diff_cfg})
            OmegaConf.set_struct(hparams, True)
            return {"epoch": 31, "hyper_parameters": hparams}

        stepper_ms = importlib.import_module("src.ace_inference.core.stepper_multistep")
        wandb_api = importlib.import_module("src.utilities.wandb_api")

        # Without our offline hook, `wandb_api.get_entity(None)` hits `wandb.api.default_entity` (network)!
        with pytest.raises(AssertionError, match="W&B network access forbidden"):
            wandb_api.get_entity(None)

        hooks = install_sdyff_offline_runtime_hooks(
            str(code_dir),
            interpolator_ckpt=str(ipol_ckpt),
            interpolator_cfg=str(ipol_cfg),
        )
        assert hooks == {
            "wandb_client": True,
            "wandb_api": True,
            "from_state": True,
            "data_writer": True,
        }
        assert os.environ["WANDB_MODE"] == "disabled"
        assert os.environ["WANDB_DISABLED"] == "true"

        # Case 1: Checkpoint `diffusion_config` lacks local paths AND caller passes `overrides={}` (or `entity=None`).
        state_empty_ov = _make_forecaster_state()
        stepper = stepper_ms.MultiStepStepper.from_state(
            state_empty_ov,
            load_optimizer=False,
            overrides={"diffusion_config": {"interpolator_wandb_kwargs": {"entity": None}}},
        )
        exp_ipol_ckpt = os.path.abspath(str(ipol_ckpt))
        exp_ipol_cfg = os.path.abspath(str(ipol_cfg))
        assert stepper.diffusion_model.interpolator["ckpt_path"] == exp_ipol_ckpt
        assert stepper.diffusion_model.interpolator["config_path"] == exp_ipol_cfg
        assert stepper.diffusion_model.interpolator["epoch"] == 31
        injected = state_empty_ov["hyper_parameters"]["diffusion_config"]
        assert injected["interpolator_local_checkpoint_path"] == exp_ipol_ckpt
        assert injected["interpolator_local_config_path"] == exp_ipol_cfg
        assert injected["interpolator_wandb_kwargs"]["entity"] == SDYFF_OFFLINE_WANDB_ENTITY

        # Direct W&B API calls fail closed offline instead of attempting network login.
        with pytest.raises(RuntimeError, match="disabled in offline AIMIP jobs"):
            wandb_api.get_api()
        with pytest.raises(RuntimeError, match="disabled in offline AIMIP jobs"):
            fake_wandb.login()

        # Remote `hf:` paths are refused by `inject_sdyff_local_interpolator_overrides`.
        with pytest.raises(ValidationError, match="Refusing remote URI"):
            inject_sdyff_local_interpolator_overrides(
                _make_forecaster_state(),
                {},
                interpolator_ckpt="hf:salv47/spherical-dyffusion/interpolator-sfno-best-val_avg_crps.ckpt",
                interpolator_cfg=str(ipol_cfg),
            )

        # Ensemble DataWriter hook: 2 members x 1 IC reshapes 5D `(2, 1, T, H, W)` -> 4D `(2, T, H, W)`.
        dw_mod = importlib.import_module("src.ace_inference.inference.data_writer.main")
        dw = dw_mod.DataWriter(
            path=str(tmp_path / "dw"),
            n_samples=1,
            n_timesteps=9,
            metadata={},
            coords={},
            enable_prediction_netcdfs=True,
            enable_video_netcdfs=False,
            n_ensemble_members=2,
        )
        assert dw.n_samples == 2
        pred_5d = {"air_temperature_7": torch.arange(2 * 1 * 8 * 2 * 3, dtype=torch.float32).reshape(2, 1, 8, 2, 3)}
        targ_4d = {"air_temperature_7": torch.ones((1, 8, 2, 3), dtype=torch.float32)}
        dw.append_batch(targ_4d, pred_5d, start_timestep=0, start_sample=0)
        rec_targ, rec_pred, _, rec_sample = dw.records[0]
        assert rec_sample == 0
        assert rec_targ["air_temperature_7"].shape == (2, 8, 2, 3)
        assert rec_pred["air_temperature_7"].shape == (2, 8, 2, 3)
        assert torch.equal(rec_pred["air_temperature_7"][0], pred_5d["air_temperature_7"][0, 0])
        assert torch.equal(rec_pred["air_temperature_7"][1], pred_5d["air_temperature_7"][1, 0])
    finally:
        for k in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
            sys.modules.pop(k, None)
        sys.modules.update(saved_src_mods)


def test_sdyff_end_to_end_run_34_vars_air_temperature_7_summary_nominal_labels_and_exit_5(
    tmp_path: Path,
    allow_tmp_run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end `plan -> prepare -> exec_fme -> run_sdyff -> summarize_run` + code-tamper & exit-5 gates."""
    import aimip.sdyff as sdyff_mod
    import xarray as xr

    data_dir = tmp_path / "data"
    pg_dir = tmp_path / "pg"
    (pg_dir / "bin").mkdir(parents=True)
    (pg_dir / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    code_dir = data_dir / SDYFF_CODE_REL_DIR
    (code_dir / "data_statistics").mkdir(parents=True)
    (code_dir / "data_statistics" / "centering.nc").write_bytes(b"CENTER_STATS")
    (code_dir / "data_statistics" / "scaling.nc").write_bytes(b"SCALE_STATS")
    for rel_pkg in (
        "src",
        "src/utilities",
        "src/ace_inference",
        "src/ace_inference/core",
        "src/ace_inference/inference",
        "src/ace_inference/inference/data_writer",
    ):
        (code_dir / rel_pkg).mkdir(parents=True, exist_ok=True)
        (code_dir / rel_pkg / "__init__.py").write_text("", encoding="utf-8")
    (code_dir / "src/utilities/wandb_api.py").write_text(
        "_ENTITY = None\n"
        "def get_entity(entity=None):\n"
        "    return entity\n"
        "def get_api():\n"
        "    raise AssertionError('unexpected network call')\n",
        encoding="utf-8",
    )
    (code_dir / "src/ace_inference/core/stepper_multistep.py").write_text(
        "class MultiStepStepper:\n"
        "    @classmethod\n"
        "    def from_state(cls, state, load_optimizer=True, overrides=None):\n"
        "        return cls()\n",
        encoding="utf-8",
    )
    (code_dir / "src/ace_inference/inference/data_writer/main.py").write_text(
        "class DataWriter:\n"
        "    def __init__(self, path, n_samples, n_timesteps, metadata, coords,\n"
        "                 enable_prediction_netcdfs, enable_video_netcdfs, time_coarsen=None, n_ensemble_members=1):\n"
        "        self.n_samples = n_samples\n"
        "    def append_batch(self, target, prediction, start_timestep, start_sample, batch_times=None):\n"
        "        pass\n",
        encoding="utf-8",
    )

    # Synthetic `run_inference.py` verifying offline env, seed, coarsen_factor=4, and writing
    # `autoregressive_predictions.nc` with `timestep=61` (`timestep=0` IC + 60 daily means) across all 34 vars.
    run_inf_src = """from __future__ import annotations
import os, sys
_entered_sentinel = os.environ.get("AIMIP_TEST_SDYFF_ENTERED_SENTINEL", "")
if _entered_sentinel:
    with open(_entered_sentinel, "w", encoding="utf-8") as _sfh:
        _sfh.write("entered\\n")
import netCDF4, numpy as np, torch, yaml

assert os.environ.get("WANDB_MODE") == "disabled"
assert os.environ.get("WANDB_DISABLED") == "true"
assert os.environ.get("HF_HUB_OFFLINE") == "1"

with open(sys.argv[1], encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

# Verify `_SDYFF_BOOTSTRAP`'s `install_sdyff_verified_loaders` loads both `.ckpt` files and Hydra YAML
# from their SHA-256-verified `O_NOFOLLOW` descriptors and refuses unlisted paths!
from aimip.errors import ValidationError
from omegaconf import OmegaConf

skip_load = os.environ.get("AIMIP_TEST_SDYFF_SKIP_LOAD", "")
if skip_load != "forecaster":
    loaded_ckpt = torch.load(cfg["checkpoint_path"], map_location="cpu")
    assert loaded_ckpt["callbacks_meta"]["steps"] == [1]

assert cfg["logging"]["log_to_wandb"] is False
assert cfg["data_writer"]["time_coarsen"]["coarsen_factor"] == 4
diff_ov = cfg["overrides"]["diffusion_config"]
assert os.path.isfile(diff_ov["interpolator_local_checkpoint_path"])
assert os.path.isfile(diff_ov["interpolator_local_config_path"])
if skip_load != "interpolator":
    loaded_ipol = torch.load(diff_ov["interpolator_local_checkpoint_path"], map_location="cpu")
    assert len(loaded_ipol["state_dict"]) == 2
loaded_ipol_cfg = OmegaConf.load(diff_ov["interpolator_local_config_path"])
assert loaded_ipol_cfg["model"] == "sfno"
try:
    torch.load(os.path.join(cfg["validation_loader"]["dataset"]["data_path"], "2021010100.nc"), map_location="cpu")
except ValidationError:
    pass
else:
    raise AssertionError("Expected verified torch.load to refuse unlisted path")
assert diff_ov["interpolator_wandb_kwargs"]["entity"] == "salv47"
n_days = cfg["n_forward_steps"] // 4
n_mem = int(cfg["n_ensemble_members"])
seed_v = int(cfg["overrides"]["seed"])
assert os.environ.get("PL_GLOBAL_SEED") == str(seed_v)

exp_dir = cfg["experiment_dir"]
assert not os.path.exists(exp_dir), f"experiment_dir must not pre-exist before run_inference.py: {exp_dir}"
os.makedirs(exp_dir)
out_nc = os.path.join(exp_dir, "autoregressive_predictions.nc")

pred_vars = [
    "PRESsfc",
    "surface_temperature",
    *[f"air_temperature_{k}" for k in range(8)],
    *[f"specific_total_water_{k}" for k in range(8)],
    *[f"eastward_wind_{k}" for k in range(8)],
    *[f"northward_wind_{k}" for k in range(8)],
    "total_water_path",
]
lat = np.linspace(-89.5, 89.5, 180, dtype=np.float64)
lon = np.linspace(0.5, 359.5, 360, dtype=np.float64)
ta7_override = os.environ.get("AIMIP_TEST_SDYFF_TA7")

with netCDF4.Dataset(out_nc, "w", format="NETCDF4") as ds:
    ds.createDimension("source", 2)
    ds.createDimension("sample", n_mem)
    ds.createDimension("timestep", n_days + 1)
    ds.createDimension("lat", 180)
    ds.createDimension("lon", 360)

    v_src = ds.createVariable("source", str, ("source",))
    v_src[0] = "target"
    v_src[1] = "prediction"
    ds.createVariable("lat", "f8", ("lat",))[:] = lat
    ds.createVariable("lon", "f8", ("lon",))[:] = lon

    for v_i, vname in enumerate(pred_vars):
        v = ds.createVariable(vname, "f4", ("source", "sample", "timestep", "lat", "lon"))
        arr = np.full((2, n_mem, n_days + 1, 180, 360), 10.0 + v_i, dtype=np.float32)
        if vname == "surface_temperature":
            # Deliberately extreme over-ocean prescribed sentinel: MUST NOT appear as summary temperature!
            arr[1, :, :, :, :] = 500.0
        elif vname == "air_temperature_7":
            for m_i in range(n_mem):
                # timestep 0 is IC (999.0 -> must be dropped!); days 1..31 -> 284+m_i; days 32..59 -> 286+m_i;
                # day 60 (incomplete March) -> 888.0 (must be dropped!).
                arr[1, m_i, 0, :, :] = 999.0
                base_jan = float(ta7_override) if ta7_override else (284.0 + m_i)
                base_feb = float(ta7_override) if ta7_override else (286.0 + m_i)
                arr[1, m_i, 1:32, :, :] = base_jan
                arr[1, m_i, 32:60, :, :] = base_feb
                if n_days >= 60:
                    arr[1, m_i, 60:, :, :] = 888.0
        elif vname == "PRESsfc":
            arr[1, :, :, :, :] = 101100.0
        elif vname == "total_water_path":
            arr[1, :, :, :, :] = 24.5
        v[:] = arr
"""
    (code_dir / "run_inference.py").write_text(run_inf_src, encoding="utf-8")
    code_tree_sha, code_tree_files = compute_sha256sum_tree_digest(str(code_dir))
    monkeypatch.setattr(sdyff_mod, "SDYFF_TREE_SHA256", code_tree_sha)
    monkeypatch.setattr(sdyff_mod, "SDYFF_TREE_FILES", code_tree_files)

    lock_files: dict[str, dict[str, Any]] = {}
    f_ckpt_p = data_dir / SDYFF_FORECASTER_CKPT_REL_PATH
    i_ckpt_p = data_dir / SDYFF_INTERPOLATOR_CKPT_REL_PATH
    i_cfg_p = data_dir / SDYFF_INTERPOLATOR_CFG_REL_PATH
    f_sha = _write_synthetic_sdyff_ckpt(f_ckpt_p, tensor_val=1.5)
    i_sha = _write_synthetic_sdyff_ckpt(i_ckpt_p, tensor_val=2.5)
    i_cfg_p.parent.mkdir(parents=True, exist_ok=True)
    i_cfg_p.write_text("model: sfno\n", encoding="utf-8")
    i_cfg_sha = sha256_file(str(i_cfg_p))

    for rel, p, sha in (
        (SDYFF_FORECASTER_CKPT_REL_PATH, f_ckpt_p, f_sha),
        (SDYFF_INTERPOLATOR_CKPT_REL_PATH, i_ckpt_p, i_sha),
        (SDYFF_INTERPOLATOR_CFG_REL_PATH, i_cfg_p, i_cfg_sha),
    ):
        lock_files[rel] = {
            "size": p.stat().st_size,
            "sha256": sha,
            "source": {
                "kind": "hf",
                "repo": SDYFF_HF_REPO,
                "path": p.name,
                "revision": SDYFF_HF_REVISION,
            },
        }

    for fname in sorted(SDYFF_ZENODO_FILES):
        rel = f"{SDYFF_DATA_REL_DIR}/{fname}"
        zp = data_dir / rel
        zp.parent.mkdir(parents=True, exist_ok=True)
        raw_z = f"SYNTH_FV3GFS_{fname}".encode("ascii")
        zp.write_bytes(raw_z)
        lock_files[rel] = {
            "size": len(raw_z),
            "sha256": sha256_file(str(zp)),
            "source": {
                "kind": "zenodo",
                "record": SDYFF_ZENODO_RECORD,
                "file": fname,
                "url": f"https://zenodo.org/records/{SDYFF_ZENODO_RECORD}/files/{fname}?download=1",
                "md5": hashlib.md5(raw_z).hexdigest(),
            },
        }

    sd_entry = next(e for e in registry_entries(str(data_dir)) if e["name"] == "spherical-dyffusion")
    sd_entry = dict(sd_entry, checkpoint_sha256=f_sha)
    reg_file = data_dir / "registry.json"
    reg_file.write_text(
        json.dumps({"schema_version": 3, "generated_by": "test", "models": [sd_entry]}),
        encoding="utf-8",
    )
    lock_file = tmp_path / "sdyff.lock"
    lock_file.write_text(json.dumps({"schema_version": 1, "files": lock_files}), encoding="utf-8")

    runs_dir = allow_tmp_run_root / "sdyff_e2e_runs"
    runs_dir.mkdir()
    site_file = tmp_path / "site.json"
    site_file.write_text(
        json.dumps(
            {
                "name": "levante",
                "scheduler": "slurm",
                "runtime": "native",
                "sbatch": {
                    "partition": "gpu",
                    "account": "acct",
                    "gres": "gpu:a100_80:1",
                    "cpus_per_task": 16,
                    "mem": "64G",
                    "max_time": "08:00:00",
                },
                "run_root": str(runs_dir),
                "extra_ro_binds": [],
                "gpu_check": "nvidia-smi",
            }
        ),
        encoding="utf-8",
    )

    src_root = str(Path(__file__).resolve().parents[1] / "src")
    fake_sdyff_py = tmp_path / "fake_sdyff_py"
    fake_sdyff_py.write_text(
        f"""#!/bin/bash
set -eu
[ "$1 $2 $3" = "-I -m aimip.emu_job" ] || exit 9
export PYTHONPATH={src_root!r}
exec {sys.executable!r} -c "
import json, sys
from aimip.sdyff import run_sdyff
out = sys.argv[1]
with open(out + '/config.yaml', encoding='utf-8') as fh:
    job = json.load(fh)
job['device'] = 'cpu'
job.setdefault('code_tree_sha256', {code_tree_sha!r})
job.setdefault('code_tree_files', {code_tree_files})
run_sdyff(out, job)
" "$5"
""",
        encoding="utf-8",
    )
    fake_sdyff_py.chmod(fake_sdyff_py.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg_dir))
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_REGISTRY", str(reg_file))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_file))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_file))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(runs_dir))
    monkeypatch.setenv("AIMIP_SDYFF_PYTHON", str(fake_sdyff_py))

    spec = RunSpec(
        model="spherical-dyffusion",
        start="2021-01-01",
        steps=240,
        members=2,
        seed=7,
        name="sdyff-qual-offline",
    )
    rp = plan(spec)
    assert rp.requires_staging is False
    assert rp.steps == 240
    assert rp.seed == 7
    assert rp.runtime_identity["coarsen_factor"] == 4
    assert rp.runtime_identity["calendar_note"] == SDYFF_CALENDAR_NOTE

    now = 1_750_000_000.0
    broker = MemoryApprovalBroker(now_fn=lambda: now)
    auth = Authorization(
        action="submit",
        digest=rp.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="sdyff-prep-1",
        allow_download=False,
    )
    broker.store_minted(auth)
    prep = prepare(spec, authorization=auth, require_authorization=False, broker=broker)
    assert prep.manifest["family"] == "sdyff"
    assert prep.manifest["seed"] == 7
    assert prep.manifest["nominal_cycle_months"] == ["0001-01", "0001-02"]

    # Tampering 1 byte in `code/spherical-dyffusion` causes `exec_fme` to fail closed BEFORE running `emu_job`!
    stat_c = code_dir / "data_statistics" / "centering.nc"
    orig_c = stat_c.read_bytes()
    stat_c.write_bytes(b"TAMPERED")
    with pytest.raises(ValidationError, match="Lock verification failed for code/spherical-dyffusion"):
        exec_fme(prep.run_dir)
    stat_c.write_bytes(orig_c)

    # Setting an unapproved `AIMIP_SDYFF_CODE` override at execution time is refused!
    unapproved_sd = tmp_path / "unapproved_sdyff"
    unapproved_sd.mkdir()
    monkeypatch.setenv("AIMIP_SDYFF_CODE", str(unapproved_sd))
    try:
        with pytest.raises(ValidationError, match="Refusing unapproved source override AIMIP_SDYFF_CODE"):
            exec_fme(prep.run_dir)
    finally:
        monkeypatch.delenv("AIMIP_SDYFF_CODE", raising=False)

    done = exec_fme(prep.run_dir)
    assert done["rc"] == 0
    assert done["result"]["months_written"] == {"atmosphere": 2}
    assert done["result"]["months"] == ["0001-01", "0001-02"]
    assert done["result"]["predicted_variables"] == 34

    out_nc = Path(prep.run_dir) / "monthly_mean_predictions.nc"
    with xr.open_dataset(out_nc, engine="netcdf4", decode_times=False) as ds_m:
        assert "surface_temperature" not in ds_m.data_vars
        assert SDYFF_PRESCRIBED_SURFACE_VAR in ds_m.data_vars
        assert SDYFF_SUMMARY_TEMP_VAR in ds_m.data_vars
        assert "PRESsfc" in ds_m.data_vars
        assert "total_water_path" in ds_m.data_vars
        assert len([v for v in ds_m.data_vars if v not in ("valid_time", "counts")]) == 35
        assert ds_m.attrs["calendar_note"] == SDYFF_CALENDAR_NOTE
        assert int(ds_m.attrs["seed"]) == 7
        assert int(ds_m.attrs["coarsen_factor"]) == 4

    summary = summarize_run(prep.run_dir)
    assert list(summary["variables"].keys()) == ["air_temperature_7"]
    assert "surface_temperature" not in summary["variables"]
    ta7_meta = summary["variables"]["air_temperature_7"]
    assert ta7_meta["units"] == "K"
    series = ta7_meta["series"]
    assert [r["month"] for r in series] == ["0001-01", "0001-02"]
    assert [r["counts"] for r in series] == [124, 112]
    # Member 0: 284.0 & 286.0; Member 1: 285.0 & 287.0 -> ensemble means 284.5 & 286.5
    # (proves IC sentinel 999.0 at timestep=0 AND partial-March sentinel 888.0 at day 60 were excluded!).
    assert math.isclose(float(series[0]["ensemble_mean"]), 284.5, rel_tol=1e-5)
    assert math.isclose(float(series[1]["ensemble_mean"]), 286.5, rel_tol=1e-5)

    # If the child skips loading either checkpoint through `install_sdyff_verified_loaders`,
    # `run_sdyff` fails closed and names the missing checkpoint!
    job_base = json.loads((Path(prep.run_dir) / "config.yaml").read_text(encoding="utf-8"))
    monkeypatch.setenv("AIMIP_TEST_SDYFF_SKIP_LOAD", "interpolator")
    try:
        bypass_ipol_dir = runs_dir / "sdyff-bypass-ipol"
        bypass_ipol_dir.mkdir()
        with pytest.raises(
            ValidationError,
            match=r"without loading required checkpoint.*interpolator-sfno-best-val_avg_crps\.ckpt",
        ):
            run_sdyff(
                str(bypass_ipol_dir),
                {
                    **job_base,
                    "out_dir": str(bypass_ipol_dir),
                    "device": "cpu",
                    "code_tree_sha256": code_tree_sha,
                    "code_tree_files": code_tree_files,
                },
            )

        monkeypatch.setenv("AIMIP_TEST_SDYFF_SKIP_LOAD", "forecaster")
        bypass_fcast_dir = runs_dir / "sdyff-bypass-fcast"
        bypass_fcast_dir.mkdir()
        with pytest.raises(
            ValidationError,
            match=r"without loading required checkpoint.*forecaster-sfno-best-inference_avg_crps\.ckpt",
        ):
            run_sdyff(
                str(bypass_fcast_dir),
                {
                    **job_base,
                    "out_dir": str(bypass_fcast_dir),
                    "device": "cpu",
                    "code_tree_sha256": code_tree_sha,
                    "code_tree_files": code_tree_files,
                },
            )
    finally:
        monkeypatch.delenv("AIMIP_TEST_SDYFF_SKIP_LOAD", raising=False)

    # Out-of-band `air_temperature_7` (`400 K`) exits with `EXIT_UNSTABLE == 5`!
    monkeypatch.setenv("AIMIP_TEST_SDYFF_TA7", "400.0")
    unstable_dir = runs_dir / "sdyff-unstable"
    unstable_dir.mkdir()
    job_u = json.loads((Path(prep.run_dir) / "config.yaml").read_text(encoding="utf-8"))
    job_u.update(
        {
            "out_dir": str(unstable_dir),
            "device": "cpu",
            "code_tree_sha256": code_tree_sha,
            "code_tree_files": code_tree_files,
        }
    )
    with pytest.raises(SystemExit) as exc_u:
        run_sdyff(str(unstable_dir), job_u)
    assert exc_u.value.code == EXIT_UNSTABLE == 5
    monkeypatch.delenv("AIMIP_TEST_SDYFF_TA7", raising=False)

    # Missing required offline hook (`src/ace_inference/core/stepper_multistep.py` -> `from_state`)
    # MUST fail closed BEFORE `run_inference.py` starts (`entered_sentinel` stays uncreated)!
    broken_code_dir = tmp_path / "sdyff_missing_hook_code"
    shutil.copytree(code_dir, broken_code_dir)
    (broken_code_dir / "src/ace_inference/core/stepper_multistep.py").unlink()
    broken_sha, broken_files = compute_sha256sum_tree_digest(str(broken_code_dir))
    sentinel_file = tmp_path / "run_inference_entered.sentinel"
    missing_hook_out = runs_dir / "sdyff-missing-hook"
    missing_hook_out.mkdir()
    monkeypatch.setenv("AIMIP_TEST_SDYFF_ENTERED_SENTINEL", str(sentinel_file))
    try:
        with pytest.raises(ValidationError, match=r"required offline runtime hook.*from_state"):
            run_sdyff(
                str(missing_hook_out),
                {
                    **job_base,
                    "out_dir": str(missing_hook_out),
                    "device": "cpu",
                    "code_dir": str(broken_code_dir),
                    "code_tree_sha256": broken_sha,
                    "code_tree_files": broken_files,
                    "verify_code_tree": True,
                },
            )
        assert not sentinel_file.exists()
    finally:
        monkeypatch.delenv("AIMIP_TEST_SDYFF_ENTERED_SENTINEL", raising=False)

    # Approved `AIMIP_SDYFF_CODE` override bound in `prepare` fails closed if edited before `exec_fme`!
    ovr_sdyff_dir = tmp_path / "sdyff_approved_ovr"
    shutil.copytree(code_dir, ovr_sdyff_dir)
    monkeypatch.setenv("AIMIP_SDYFF_CODE", str(ovr_sdyff_dir))
    try:
        spec_ovr = RunSpec(
            model="spherical-dyffusion",
            start="2021-01-01",
            steps=240,
            members=1,
            seed=7,
            name="sdyff-ovr-tamper",
        )
        rp_ovr = plan(spec_ovr)
        auth_ovr = Authorization(
            action="submit",
            digest=rp_ovr.digest,
            user=current_user(),
            issued_at=now,
            expires_at=now + 600,
            nonce="sdyff-ovr-1",
            allow_download=False,
        )
        broker.store_minted(auth_ovr)
        prep_ovr = prepare(spec_ovr, authorization=auth_ovr, require_authorization=False, broker=broker)
        assert prep_ovr.manifest["code_is_override"] is True
        (ovr_sdyff_dir / "src/utilities/wandb_api.py").write_text("# edited after prepare\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="Approved source override tree changed at job time"):
            exec_fme(prep_ovr.run_dir)
    finally:
        monkeypatch.delenv("AIMIP_SDYFF_CODE", raising=False)
