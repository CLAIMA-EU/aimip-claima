"""Offline Phase 2c tests for `cbottle-3d` (`family="cbottle"`, zero network, zero GPU).

Covers:
1. Provenance & Apache-2.0 / CC-BY-4.0 contract (`NVlabs/cBottle@f0f51998`, `nvidia/cbottle@e60ce82b`,
   `PCMDI-AMIP-1-1-9` SST, no `eebd93c8`, no Zenodo `20832634`, no `ema` weights, verbatim `"Authors' limits"`).
2. Generator request guards: rejects `--ic`, `--ocean-ic`, `--lora`, dates outside
   `1940-01-01T00:00:00 .. 2022-12-16T00:00:00`, and non-midnight `daily_synoptic` starts;
   records `(seed, batch_size=4, cadence='daily_synoptic', date_order='chronological')` in `plan`.
3. Pre-pickle `O_NOFOLLOW` SHA-256 gate on `.checkpoint` plain ZIP archives + `weights_only=True`
   verification of inner `net_state.pth` (`torch.load` call count is `0` on SHA mismatch or symlink!).
4. End-to-end `services.plan -> prepare -> run_cbottle -> summarize_run`: bit-identical `t2m`
   monthly means across repeated runs with `(seed=42, batch_size=4)`, distinct means for different
   seeds, `counts` (`31` Jan, `28` Feb), and `EXIT_UNSTABLE == 5` on out-of-band `t2m`.
"""

from __future__ import annotations

from collections import OrderedDict
import io
import json
from pathlib import Path
import types
from typing import Any
import zipfile

import pytest

from aimip.cbottle import (
    CBOTTLE_BATCH_SIZE,
    CBOTTLE_CHECKPOINT_REL_PATHS,
    CBOTTLE_CODE_ARCHIVE_SHA256,
    CBOTTLE_CODE_ARCHIVE_SIZE,
    CBOTTLE_CODE_COMMIT,
    CBOTTLE_CODE_REPO,
    CBOTTLE_DATE_ORDER,
    CBOTTLE_DEFAULT_CADENCE,
    CBOTTLE_FILE_SPECS,
    CBOTTLE_HF_REPO,
    CBOTTLE_HF_REVISION,
    CBOTTLE_OUTPUT_VARS,
    CBOTTLE_SEED_SCHEME,
    CBOTTLE_SST_REL_PATH,
    CBOTTLE_TREE_FILES,
    CBOTTLE_TREE_SHA256,
    CBOTTLE_WINDOW,
    EXIT_UNSTABLE,
    cbottle_batch_seed,
    cbottle_sample_timestamps,
    run_cbottle,
    verified_cbottle_pretrained_loader,
    verify_cbottle_checkpoint_zip,
)
from aimip.emu import precheck_emu_request
from aimip.emu_catalog import artifacts, registry_entries, required_files
from aimip.errors import ConfigError, ValidationError
from aimip.lora import sha256_file
from aimip.plan_types import RunSpec
from aimip.registry import DATA_PLACEHOLDER, _parse_model_entry, build_default_registry_payload
from aimip.services import Authorization, MemoryApprovalBroker, cite, current_user, plan, prepare, show
from aimip.setup_data import load_lock_file
from aimip.status import summarize_run

_CHANNELS_45 = [
    *(f"{v}{p}" for v in ("U", "V", "T", "Z") for p in (1000, 850, 700, 500, 300, 200, 50, 10)),
    "tcwv",
    "cllvi",
    "clivi",
    "tas",
    "uas",
    "vas",
    "rlut",
    "rsut",
    "pres_msl",
    "pr",
    "rsds",
    "sst",
    "sic",
]


def _write_synthetic_cbottle_checkpoint_zip(path: Path, *, seed_val: float = 1.0) -> str:
    """Create a realistic plain-ZIP `.checkpoint` archive with `net_state.pth` + JSON configs."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    state = OrderedDict(
        [
            ("device_buffer", torch.zeros(1, dtype=torch.float32)),
            ("model.pos_embed", torch.full((1, 4, 8), seed_val, dtype=torch.float32)),
        ]
    )
    torch.save(state, buf)
    model_json = {
        "architecture": "unet_hpx64",
        "model_channels": 192,
        "out_channels": 45,
        "calendar_include_legacy_bug": True,
    }
    batch_info = {
        "channels": _CHANNELS_45,
        "time_step": 1,
        "time_unit": "HOUR",
        "scales": [1.0] * 45,
    }
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("net_state.pth", buf.getvalue())
        zf.writestr("model.json", json.dumps(model_json))
        zf.writestr("batch_info.json", json.dumps(batch_info))
        zf.writestr("metadata.json", json.dumps({"version": 1}))
    return sha256_file(str(path))


def _write_synthetic_sst_nc(path: Path) -> str:
    """Create a minimal `amip_midmonth_sst.nc` (`PCMDI-AMIP-1-1-9`, CC-BY-4.0)."""
    import netCDF4
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(path), "w", format="NETCDF4") as ds:
        ds.createDimension("time", 2)
        ds.createDimension("lat", 6)
        ds.createDimension("lon", 12)
        v_t = ds.createVariable("time", "f8", ("time",))
        v_t.units = "days since 1870-01-01"
        v_t.calendar = "gregorian"
        v_t[:] = [45670.5, 45700.0]
        v_lat = ds.createVariable("lat", "f8", ("lat",))
        v_lat[:] = np.linspace(-75.0, 75.0, 6)
        v_lon = ds.createVariable("lon", "f8", ("lon",))
        v_lon[:] = np.linspace(15.0, 345.0, 12)
        v_sst = ds.createVariable("tosbcs", "f4", ("time", "lat", "lon"))
        v_sst.units = "degC"
        v_sst[:] = np.full((2, 6, 12), 18.5, dtype=np.float32)
        ds.source_id = "PCMDI-AMIP-1-1-9"
        ds.license = "CC BY 4.0"
    return sha256_file(str(path))


def test_cbottle_provenance_apache2_pin_and_card_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `NVlabs/cBottle@f0f51998`, `nvidia/cbottle@e60ce82b`, lock pins, and card disclosures."""
    import aimip

    monkeypatch.setenv("AIMIP_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    assert CBOTTLE_CODE_REPO == "NVlabs/cBottle"
    assert CBOTTLE_CODE_COMMIT == "f0f519987589cc0b368ab892a1adfb2fb30a5587"
    assert CBOTTLE_CODE_ARCHIVE_SIZE == 183_593
    assert len(CBOTTLE_CODE_ARCHIVE_SHA256) == 64
    assert CBOTTLE_TREE_FILES == 136
    assert len(CBOTTLE_TREE_SHA256) == 64

    assert CBOTTLE_HF_REPO == "nvidia/cbottle"
    assert CBOTTLE_HF_REVISION == "e60ce82b236b953d79d3ead7f43445a3bb8e37d3"
    assert CBOTTLE_HF_REVISION != "eebd93c85b3cd3a5a8f79c546ed917b0b80438f4"

    reqs = required_files("cbottle-3d")
    assert len(reqs) == 4
    assert all("ema-checkpoint" not in r and "cBottle-3d.zip" not in r for r in reqs)

    pkg_dir = Path(aimip.__file__).resolve().parent
    lock_files = load_lock_file(str(pkg_dir / "aimip.lock"))["files"]
    for hf_name, spec in CBOTTLE_FILE_SPECS.items():
        rel = spec["rel_path"]
        assert rel in reqs
        le = lock_files[rel]
        assert le["size"] == spec["size"]
        assert le["sha256"] == spec["sha256"]
        assert le["source"] == {
            "kind": "hf",
            "repo": CBOTTLE_HF_REPO,
            "path": hf_name,
            "revision": CBOTTLE_HF_REVISION,
        }

    emu_cb = [a for a in artifacts() if "cbottle-3d" in a["models"]]
    assert len(emu_cb) == 4

    card = show("cbottle-3d")
    assert card["family"] == "cbottle"
    assert card["task"] == "generator"
    assert (
        card["verified_on"] is not None
        and "A100" in card["verified_on"]
        and "285.7072" in card["verified_on"]
        and "1.4.0" in card["verified_on"]
        and "job " not in card["verified_on"]
    )
    assert "Apache-2.0" in card["licence"] and "CC-BY-4.0" in card["licence"]
    assert "20832634" not in json.dumps(card)
    assert '"resolve meso-scale motions"' in card["limitations"]
    assert '"no independent data withheld for evaluation"' in card["limitations"]
    assert "Conditional state generator, not a time-stepping simulation" in card["limitations"]
    assert "independent draws" in card["limitations"]
    assert "SHA-256" in card["limitations"] and "batch_index" in card["limitations"]
    assert "2018" in card["limitations"]

    cit = cite("cbottle-3d")
    assert "2505.06474" in cit["bibtex"]
    assert any("2505.06474" in b for b in cit["entries"])


def test_cbottle_rejects_ic_and_dates_outside_window_and_records_sampling_params(
    tmp_path: Path,
    allow_tmp_run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `--ic` rejection, `1940-01-01 .. 2022-12-16` bounds, rotating synoptic hours, and plan binding."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    runs_dir = allow_tmp_run_root / "cbottle_plan_runs"
    runs_dir.mkdir()
    site_path = tmp_path / "site.json"
    site_path.write_text(
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
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_path))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(runs_dir))
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    payload = build_default_registry_payload(root_placeholder=DATA_PLACEHOLDER)
    entry = next(e for e in payload["models"] if e["name"] == "cbottle-3d")
    spec_obj = _parse_model_entry(dict(entry), None, str(data_dir))

    # Synoptic hours rotate 00/06/12/18 UTC across consecutive days.
    ts_list = cbottle_sample_timestamps("1995-01-01T00:00:00", 6)
    assert [dt.strftime("%Y-%m-%dT%H:%M:%S") for dt in ts_list] == [
        "1995-01-01T00:00:00",
        "1995-01-02T06:00:00",
        "1995-01-03T12:00:00",
        "1995-01-04T18:00:00",
        "1995-01-05T00:00:00",
        "1995-01-06T06:00:00",
    ]
    assert CBOTTLE_WINDOW == ("1940-01-01T00:00:00", "2022-12-16T00:00:00")

    dummy_ic = tmp_path / "dummy_ic.nc"
    dummy_ic.write_bytes(b"nc")
    for bad_kw, match in (
        ({"ic_path": str(dummy_ic)}, "does not accept --ic"),
        ({"ocean_ic_path": str(dummy_ic)}, "not supported"),
        ({"lora_path": str(dummy_ic)}, "not supported"),
        ({"start_iso": "1939-12-31T00:00:00"}, "valid date range is 1940-01-01T00:00:00 .. 2022-12-16T00:00:00"),
        ({"start_iso": "2022-12-16T00:00:00", "steps": 1}, "valid date range is 1940-01-01T00:00:00 .. 2022-12-16T00:00:00"),
        ({"start_iso": "1995-01-01T06:00:00"}, "requires --start at 00:00:00 UTC"),
        ({"forcing_set_name": "amip-hist"}, "Unknown --forcing-set"),
    ):
        kw: dict[str, Any] = {
            "start_iso": "1995-01-01T00:00:00",
            "steps": 10,
            "members": 1,
            "forcing_set_name": None,
            "ic_path": None,
        }
        kw.update(bad_kw)
        with pytest.raises(ConfigError, match=match):
            precheck_emu_request(spec_obj, **kw)

    # Boundary last valid day (`2022-12-15T00:00:00` + 1 day -> end `2022-12-16T00:00:00`) is accepted.
    edge_ok = precheck_emu_request(
        spec_obj,
        start_iso="2022-12-15T00:00:00",
        steps=1,
        members=1,
        forcing_set_name=None,
        ic_path=None,
    )
    assert edge_ok["end"] == "2022-12-16T00:00:00"

    resolved = plan(RunSpec(model="cbottle-3d", start="1995-01-01", length="2m", seed=42, name="cb-2m"))
    assert resolved.requires_staging is False
    assert resolved.steps == 59   # 31 (Jan) + 28 (Feb) in 1995
    assert resolved.seed == 42
    assert resolved.runtime_identity["batch_size"] == CBOTTLE_BATCH_SIZE == 4
    assert resolved.runtime_identity["cadence"] == CBOTTLE_DEFAULT_CADENCE == "daily_synoptic"
    assert resolved.runtime_identity["date_order"] == CBOTTLE_DATE_ORDER == "chronological"
    assert resolved.runtime_identity["seed_scheme"] == CBOTTLE_SEED_SCHEME
    assert resolved.canonical_payload["runtime_identity"]["batch_size"] == 4
    assert resolved.canonical_payload["runtime_identity"]["cadence"] == "daily_synoptic"
    assert resolved.canonical_payload["runtime_identity"]["seed_scheme"] == CBOTTLE_SEED_SCHEME


def test_cbottle_zip_checkpoint_pre_pickle_sha_gate_and_symlink_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `.checkpoint` SHA-256 mismatch or symlink aborts BEFORE `torch.load` (`call_count == 0`)."""
    import torch

    ckpt_p = tmp_path / "training-state-000512000.checkpoint"
    good_sha = _write_synthetic_cbottle_checkpoint_zip(ckpt_p, seed_val=2.5)

    meta = verify_cbottle_checkpoint_zip(str(ckpt_p), good_sha)
    assert meta["out_channels"] == 45
    assert meta["n_tensors"] == 2

    load_calls = 0
    real_torch_load = torch.load

    def _spy_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal load_calls
        load_calls += 1
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy_load)

    # 1. Wrong expected SHA-256 -> raises BEFORE `torch.load`
    with pytest.raises(ValidationError, match="SHA-256 mismatch before loading"):
        verify_cbottle_checkpoint_zip(str(ckpt_p), "0" * 64)
    assert load_calls == 0

    # 2. 1-byte corruption in `.checkpoint` -> raises BEFORE `torch.load`
    raw = ckpt_p.read_bytes()
    ckpt_p.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))
    with pytest.raises(ValidationError, match="SHA-256 mismatch before loading"):
        verify_cbottle_checkpoint_zip(str(ckpt_p), good_sha)
    assert load_calls == 0
    ckpt_p.write_bytes(raw)

    # 3. Symlink swap -> refused via `O_NOFOLLOW` BEFORE `torch.load`
    sym_p = tmp_path / "sym.checkpoint"
    sym_p.symlink_to(ckpt_p)
    with pytest.raises(ValidationError):
        verify_cbottle_checkpoint_zip(str(sym_p), good_sha)
    assert load_calls == 0

    # 4. `verified_cbottle_pretrained_loader` reads from the verified descriptor and refuses
    #    unlisted archives, missing `metadata.json`, or `weights_only=False`!
    monkeypatch.setattr(torch, "load", real_torch_load)
    unlisted_ckpt = tmp_path / "unlisted.checkpoint"
    unlisted_ckpt.write_bytes(raw)
    no_meta_buf = io.BytesIO()
    with zipfile.ZipFile(no_meta_buf, "w") as zfw:
        zfw.writestr("model.json", '{"out_channels": 45}')
        zfw.writestr("batch_info.json", json.dumps({"channels": _CHANNELS_45}))
        zfw.writestr("net_state.pth", b"placeholder")
    no_meta_ckpt = tmp_path / "no_metadata.checkpoint"
    no_meta_ckpt.write_bytes(no_meta_buf.getvalue())
    no_meta_sha = sha256_file(str(no_meta_ckpt))

    with verified_cbottle_pretrained_loader(
        {str(ckpt_p): good_sha, str(no_meta_ckpt): no_meta_sha},
        require_all_loaded=False,
    ):
        with zipfile.ZipFile(str(ckpt_p), "r") as zf_ok:
            with zf_ok.open("net_state.pth") as net_fh:
                sd_ok = torch.load(net_fh, map_location="cpu")
            assert len(sd_ok) == 2
            with zf_ok.open("net_state.pth") as net_fh2:
                with pytest.raises(ValidationError, match="weights_only=False"):
                    torch.load(net_fh2, map_location="cpu", weights_only=False)
        with pytest.raises(ValidationError, match="unverified cBottle-3d checkpoint"):
            zipfile.ZipFile(str(unlisted_ckpt), "r")
        with pytest.raises(ValidationError, match=r"metadata\.json"):
            zipfile.ZipFile(str(no_meta_ckpt), "r")

    # Default `require_all_loaded=True` fails loudly if expected archive was not opened via wrapper!
    with pytest.raises(ValidationError, match="bypassed verified ZipFile wrapper"):
        with verified_cbottle_pretrained_loader({str(ckpt_p): good_sha}):
            pass


def test_cbottle_end_to_end_run_reproducibility_summary_and_exit_5(
    tmp_path: Path,
    allow_tmp_run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `plan -> prepare -> run_cbottle -> summarize_run`, bit-identical seed=42 runs, & exit 5."""
    import sys
    import numpy as np
    import xarray as xr

    data_dir = tmp_path / "data"
    pg_dir = tmp_path / "pg"
    (pg_dir / "bin").mkdir(parents=True)
    (pg_dir / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    lock_files: dict[str, dict[str, Any]] = {}
    ckpt_shas: dict[str, str] = {}
    for idx, rel in enumerate(CBOTTLE_CHECKPOINT_REL_PATHS):
        p = data_dir / rel
        sha = _write_synthetic_cbottle_checkpoint_zip(p, seed_val=float(idx + 1))
        ckpt_shas[rel] = sha
        lock_files[rel] = {
            "size": p.stat().st_size,
            "sha256": sha,
            "source": {
                "kind": "hf",
                "repo": CBOTTLE_HF_REPO,
                "path": rel[len("models/cbottle-3d/") :],
                "revision": CBOTTLE_HF_REVISION,
            },
        }

    sst_p = data_dir / CBOTTLE_SST_REL_PATH
    sst_sha = _write_synthetic_sst_nc(sst_p)
    lock_files[CBOTTLE_SST_REL_PATH] = {
        "size": sst_p.stat().st_size,
        "sha256": sst_sha,
        "source": {
            "kind": "hf",
            "repo": CBOTTLE_HF_REPO,
            "path": "amip_midmonth_sst.nc",
            "revision": CBOTTLE_HF_REVISION,
        },
    }

    cb_entry = next(e for e in registry_entries(str(data_dir)) if e["name"] == "cbottle-3d")
    cb_entry = dict(cb_entry, checkpoint_sha256=ckpt_shas[CBOTTLE_CHECKPOINT_REL_PATHS[-1]])
    reg_file = data_dir / "registry.json"
    reg_file.write_text(
        json.dumps({"schema_version": 3, "generated_by": "test", "models": [cb_entry]}),
        encoding="utf-8",
    )
    lock_file = tmp_path / "cbottle.lock"
    lock_file.write_text(json.dumps({"schema_version": 1, "files": lock_files}), encoding="utf-8")

    runs_dir = allow_tmp_run_root / "cbottle_e2e_runs"
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
    fake_emu_py = tmp_path / "fake_emu_py"
    fake_emu_py.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_emu_py.chmod(0o755)

    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg_dir))
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_REGISTRY", str(reg_file))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_file))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_file))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(runs_dir))
    monkeypatch.setenv("AIMIP_EMU_PYTHON", str(fake_emu_py))

    # Install lightweight deterministic `earth2studio.data` + `earth2studio.data.cbottle` modules
    # whose output depends on `(seed, batch_size, batch_dates)` just like upstream CBottle3D.
    moe_calls: list[tuple[list[str], tuple[float, float]]] = []
    t2m_override: dict[str, float | None] = {"val": None}

    class _FakeMoE:
        @classmethod
        def from_pretrained(cls, paths: list[str], sigmas: tuple[float, float]) -> _FakeMoE:
            import torch as _torch

            for cp in paths:
                with zipfile.ZipFile(cp, "r") as zf_in:
                    assert "metadata.json" in zf_in.namelist()
                    with zf_in.open("net_state.pth") as nfh:
                        sd = _torch.load(nfh, map_location="cpu")
                    assert len(sd) == 2
            moe_calls.append((list(paths), tuple(sigmas)))
            return cls()

    recorded_batch_seeds: list[int] = []

    class _FakeCBottle3D:
        VARIABLES = np.array(list(CBOTTLE_OUTPUT_VARS))

        def __init__(
            self,
            core: Any,
            sst_ds: Any,
            *,
            lat_lon: bool = True,
            seed: int = 0,
            batch_size: int = 4,
        ) -> None:
            assert lat_lon is True
            assert "tosbcs" in sst_ds
            self.seed = seed
            self.batch_size = batch_size

        def to(self, _device: str) -> _FakeCBottle3D:
            return self

        def __call__(self, times: list[ Any ], variables: list[str]) -> xr.DataArray:
            assert 1 <= len(times) <= self.batch_size
            # Mirror `earth2studio/data/cbottle.py:218,234`: upstream resets local `i = 0`
            # on every `__call__` and seeds the sampler from `self.seed + i`!
            cur_seed = int(self.seed)
            recorded_batch_seeds.append(cur_seed)
            rng = np.random.default_rng(cur_seed * 1009 + self.batch_size * 37)
            lat = np.linspace(90.0, -90.0, 9, dtype=np.float64)
            lon = np.linspace(0.0, 330.0, 12, dtype=np.float64)
            arr = np.zeros((len(times), len(variables), lat.size, lon.size), dtype=np.float32)
            batch_noise = float(rng.normal(0.0, 0.25))
            for ti, dt in enumerate(times):
                seasonal = 2.0 if dt.month == 1 else 4.0
                for vi, vname in enumerate(variables):
                    if vname == "t2m":
                        base = t2m_override["val"] if t2m_override["val"] is not None else (285.0 + seasonal + batch_noise)
                        arr[ti, vi] = np.float32(base)
                    elif vname == "msl":
                        arr[ti, vi] = np.float32(101325.0)
                    else:
                        arr[ti, vi] = np.float32(1.0 + vi * 0.1)
            return xr.DataArray(
                arr,
                dims=("time", "variable", "lat", "lon"),
                coords={"time": times, "variable": variables, "lat": lat, "lon": lon},
            )

    e2s_pkg = types.ModuleType("earth2studio")
    e2s_data_pkg = types.ModuleType("earth2studio.data")
    e2s_cb_pkg = types.ModuleType("earth2studio.data.cbottle")
    e2s_data_pkg.CBottle3D = _FakeCBottle3D  # type: ignore[attr-defined]
    e2s_cb_pkg.MixtureOfExpertsDenoiser = _FakeMoE  # type: ignore[attr-defined]
    e2s_pkg.data = e2s_data_pkg  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "earth2studio", e2s_pkg)
    monkeypatch.setitem(sys.modules, "earth2studio.data", e2s_data_pkg)
    monkeypatch.setitem(sys.modules, "earth2studio.data.cbottle", e2s_cb_pkg)

    def _prepare_and_run(run_name: str, seed_val: int) -> tuple[str, dict[str, Any]]:
        spec = RunSpec(model="cbottle-3d", start="1995-01-01", length="2m", seed=seed_val, name=run_name)
        rp = plan(spec)
        now = 1_750_000_000.0
        broker = MemoryApprovalBroker(now_fn=lambda: now)
        auth = Authorization(
            action="submit",
            digest=rp.digest,
            user=current_user(),
            issued_at=now,
            expires_at=now + 600,
            nonce=f"cb-{run_name}",
            allow_download=False,
        )
        broker.store_minted(auth)
        prep = prepare(spec, authorization=auth, require_authorization=False, broker=broker)
        assert prep.manifest["batch_size"] == 4
        assert prep.manifest["cadence"] == "daily_synoptic"
        assert prep.manifest["ic_path"] is None
        job = json.loads((Path(prep.run_dir) / "config.yaml").read_text(encoding="utf-8"))
        job["device"] = "cpu"
        res = run_cbottle(prep.run_dir, job)
        return prep.run_dir, res

    dir_a, res_a = _prepare_and_run("cb-s42-a", 42)
    dir_b, res_b = _prepare_and_run("cb-s42-b", 42)
    dir_c, _ = _prepare_and_run("cb-s99", 99)

    assert res_a["months"] == ["1995-01", "1995-02"]
    assert res_a["samples_per_member"] == 59
    assert len(moe_calls) == 3

    sum_a = summarize_run(dir_a)
    sum_b = summarize_run(dir_b)
    sum_c = summarize_run(dir_c)
    rows_a = sum_a["variables"]["t2m"]["series"]
    rows_b = sum_b["variables"]["t2m"]["series"]
    rows_c = sum_c["variables"]["t2m"]["series"]

    assert [r["month"] for r in rows_a] == ["1995-01", "1995-02"]
    assert [r["counts"] for r in rows_a] == [31, 28]
    # Bit-identical monthly means across two independent runs with `(seed=42, batch_size=4)`!
    assert rows_a == rows_b
    # Different seed produces distinct stochastic draws!
    assert rows_a[0]["ensemble_mean"] != rows_c[0]["ensemble_mean"]

    with xr.open_dataset(Path(dir_a) / "monthly_mean_predictions.nc", engine="netcdf4", decode_times=False) as ds_out:
        assert int(ds_out.attrs["seed"]) == 42
        assert ds_out.attrs["seed_scheme"] == CBOTTLE_SEED_SCHEME
        assert int(ds_out.attrs["batch_size"]) == 4
        assert ds_out.attrs["cadence"] == "daily_synoptic"
        assert ds_out.attrs["date_order"] == "chronological"

    # Multi-member, multi-batch (`members=2, steps=10 -> 3 batches/member = 6 calls`) derives 6 distinct seeds
    # and repeats identically for the same `--seed`!
    def _prepare_and_run_multi(run_name: str, seed_val: int) -> tuple[str, dict[str, Any], list[int]]:
        recorded_batch_seeds.clear()
        spec_m = RunSpec(
            model="cbottle-3d",
            start="1995-01-01",
            steps=10,
            members=2,
            seed=seed_val,
            name=run_name,
        )
        rp_m = plan(spec_m)
        now_m = 1_750_000_000.0
        broker_m = MemoryApprovalBroker(now_fn=lambda: now_m)
        auth_m = Authorization(
            action="submit",
            digest=rp_m.digest,
            user=current_user(),
            issued_at=now_m,
            expires_at=now_m + 600,
            nonce=f"cb-{run_name}",
            allow_download=False,
        )
        broker_m.store_minted(auth_m)
        prep_m = prepare(spec_m, authorization=auth_m, require_authorization=False, broker=broker_m)
        expected_grid = [
            [cbottle_batch_seed(seed_val, m_i, b_i) for b_i in range(3)]
            for m_i in range(2)
        ]
        assert prep_m.manifest["seed_scheme"] == CBOTTLE_SEED_SCHEME
        assert prep_m.manifest["batch_seeds"] == expected_grid
        job_m = json.loads((Path(prep_m.run_dir) / "config.yaml").read_text(encoding="utf-8"))
        job_m["device"] = "cpu"
        res_m = run_cbottle(prep_m.run_dir, job_m)
        return prep_m.run_dir, res_m, list(recorded_batch_seeds)

    m_dir_1, m_res_1, m_seeds_1 = _prepare_and_run_multi("cb-multi-42-a", 42)
    m_dir_2, m_res_2, m_seeds_2 = _prepare_and_run_multi("cb-multi-42-b", 42)
    assert len(m_seeds_1) == 6
    assert len(set(m_seeds_1)) == 6
    assert m_seeds_1 == m_seeds_2 == [
        cbottle_batch_seed(42, m_i, b_i) for m_i in range(2) for b_i in range(3)
    ]
    assert m_res_1["seed_scheme"] == CBOTTLE_SEED_SCHEME
    assert m_res_1["batch_seeds"] == m_res_2["batch_seeds"]
    m_sum_1 = summarize_run(m_dir_1)["variables"]["t2m"]["series"]
    m_sum_2 = summarize_run(m_dir_2)["variables"]["t2m"]["series"]
    assert m_sum_1 == m_sum_2
    assert m_sum_1[0]["members"][0] != m_sum_1[0]["members"][1]

    # Corrupting 1 byte in an expert checkpoint aborts BEFORE `MixtureOfExpertsDenoiser.from_pretrained` is called!
    moe_before = len(moe_calls)
    bad_dir = runs_dir / "cb-corrupt"
    bad_dir.mkdir()
    job_ref = json.loads((Path(dir_a) / "config.yaml").read_text(encoding="utf-8"))
    job_bad = {**job_ref, "out_dir": str(bad_dir), "device": "cpu"}
    ckpt0 = Path(job_bad["expert_checkpoints"][0]["path"])
    orig_ckpt0 = ckpt0.read_bytes()
    ckpt0.write_bytes(orig_ckpt0[:-1] + bytes([orig_ckpt0[-1] ^ 0xFF]))
    with pytest.raises(ValidationError, match="SHA-256 mismatch before loading"):
        run_cbottle(str(bad_dir), job_bad)
    assert len(moe_calls) == moe_before
    ckpt0.write_bytes(orig_ckpt0)

    # Bypassing `verified_cbottle_pretrained_loader` inside `from_pretrained` fails loudly naming the checkpoint!
    class _BypassMoE:
        @classmethod
        def from_pretrained(cls, _paths: list[str], _sigmas: tuple[float, float]) -> _BypassMoE:
            return cls()

    e2s_cb_pkg.MixtureOfExpertsDenoiser = _BypassMoE  # type: ignore[attr-defined]
    try:
        bypass_dir = runs_dir / "cb-bypass"
        bypass_dir.mkdir()
        with pytest.raises(
            ValidationError,
            match=r"bypassed verified ZipFile wrapper.*training-state-000512000\.checkpoint",
        ):
            run_cbottle(str(bypass_dir), {**job_ref, "out_dir": str(bypass_dir), "device": "cpu"})
    finally:
        e2s_cb_pkg.MixtureOfExpertsDenoiser = _FakeMoE  # type: ignore[attr-defined]

    # Out-of-band `t2m` (`390 K`) exits with `EXIT_UNSTABLE == 5`!
    t2m_override["val"] = 390.0
    unstable_dir = runs_dir / "cb-unstable"
    unstable_dir.mkdir()
    with pytest.raises(SystemExit) as exc_u:
        run_cbottle(str(unstable_dir), {**job_ref, "out_dir": str(unstable_dir), "device": "cpu"})
    assert exc_u.value.code == EXIT_UNSTABLE == 5
