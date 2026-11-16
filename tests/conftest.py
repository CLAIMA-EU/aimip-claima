"""Shared fixtures and synthetic fme artifact builders for offline pytest runs."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from collections.abc import Sequence
from typing import Any

import cftime
import numpy as np
import pytest
import xarray as xr

import importlib.util
import os

os.umask(0o022)


def pytest_configure(config: Any) -> None:
    os.umask(0o022)


@pytest.fixture(autouse=True)
def _isolate_aimip_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ambient `AIMIP_*` path/site/registry variables from the caller's shell."""
    for key in (
        "AIMIP_HOME",
        "AIMIP_PLAYGROUND",
        "AIMIP_DATA",
        "AIMIP_SITE_CONFIG",
        "AIMIP_DEFAULT_OUT_ROOT",
        "AIMIP_STATE",
        "AIMIP_TTY_PATH",
        "AIMIP_RUN_ROOTS",
        "AIMIP_EXTRA_RO_BINDS",
        "AIMIP_REGISTRY",
        "AIMIP_LOCK",
        "AIMIP_HF_MIRROR",
        "AIMIP_SCHEDULER",
        "AIMIP_IMAGE_SHA",
        "AIMIP_FME_2026_10_OVERLAY",
        "AIMIP_EMU_PYTHON",
        "AIMIP_ARCHES_URL",
    ):
        monkeypatch.delenv(key, raising=False)

# Ensure `aimip` and test helpers are importable regardless of working dir or pytest import mode.
_TESTS_DIR = pathlib.Path(__file__).resolve().parent
_SRC_ROOT = _TESTS_DIR.parent / "src"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
if importlib.util.find_spec("aimip") is None and (_SRC_ROOT / "aimip").is_dir():
    sys.path.insert(0, str(_SRC_ROOT))


def sha256_path(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def parse_cli_json(raw: str) -> Any:
    """Parse a v1 CLI `--json` envelope (`{"aimip": "1", "ok": True, "result": ...}`) and return `result`."""
    doc = json.loads(raw)
    assert isinstance(doc, dict) and doc.get("aimip") == "1", f"Missing v1 envelope: {doc!r}"
    assert doc.get("ok") is True, f"Expected ok=True envelope: {doc!r}"
    assert "result" in doc
    return doc["result"]


import contextlib
import pty
import termios


@contextlib.contextmanager
def pty_responder(*lines: str):
    """Create a real pseudo-TTY pair (`os.isatty == True`), pre-fill human responses on `master_fd`, and yield `(tty_path, read_Transcript_fn)`."""
    master_fd, slave_fd = pty.openpty()
    try:
        attrs = termios.tcgetattr(slave_fd)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
        tty_path = os.ttyname(slave_fd)
        for line in lines:
            os.write(master_fd, (line + "\n").encode("utf-8"))

        def _drain_prompt() -> str:
            os.set_blocking(master_fd, False)
            chunks: list[bytes] = []
            while True:
                try:
                    chunk = os.read(master_fd, 4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                except BlockingIOError:
                    break
            return b"".join(chunks).decode("utf-8", errors="replace")

        yield tty_path, _drain_prompt
    finally:
        for fd in (slave_fd, master_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def write_fake_ic_netcdf(
    path: pathlib.Path,
    timestamps: Sequence[str],
    *,
    calendar: str = "proleptic_gregorian",
    sample_dim: str = "time",
    spatial_dims: tuple[str, str] = ("lat", "lon"),
    var_names: Sequence[str] = ("TMP2m", "surface_temperature"),
    seed: int = 0,
) -> pathlib.Path:
    """Write a tiny fme-compatible IC netCDF (2 variables on a 4x8 grid with configurable dim names)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    y_dim, x_dim = spatial_dims
    n_samples = len(timestamps)
    lats = np.array([-60.0, -20.0, 20.0, 60.0], dtype=np.float32)
    lons = np.linspace(0.0, 315.0, 8, dtype=np.float32)

    dt_objs = np.empty((n_samples,), dtype=object)
    for idx, ts in enumerate(timestamps):
        dt_objs[idx] = cftime.datetime(
            int(ts[0:4]),
            int(ts[5:7]),
            int(ts[8:10]),
            int(ts[11:13]),
            int(ts[14:16]),
            int(ts[17:19]),
            calendar=calendar,
        )

    data_vars: dict[str, Any] = {}
    for v_idx, vname in enumerate(var_names):
        base = 280.0 + 5.0 * v_idx + rng.normal(0.0, 1.0, size=(n_samples, 4, 8))
        data_vars[vname] = xr.DataArray(
            base.astype(np.float32),
            dims=(sample_dim, y_dim, x_dim),
            attrs={"units": "K", "long_name": f"Synthetic {vname}"},
        )

    # Include hybrid vertical scalar coordinates like shipped ACE2 ICs.
    data_vars["ak_0"] = xr.DataArray(np.float32(0.0), dims=(), attrs={"units": "Pa"})
    data_vars["bk_0"] = xr.DataArray(np.float32(1.0), dims=(), attrs={"units": "1"})
    data_vars["time"] = xr.DataArray(dt_objs, dims=(sample_dim,))

    coords = {
        y_dim: xr.DataArray(lats, dims=(y_dim,), attrs={"units": "degrees_north"}),
        x_dim: xr.DataArray(lons, dims=(x_dim,), attrs={"units": "degrees_east"}),
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={"source": "synthetic-aimip-test"},
    )
    ds.to_netcdf(
        str(path),
        engine="netcdf4",
        encoding={
            "time": {
                "units": f"hours since {timestamps[0][:10]} 00:00:00",
                "calendar": calendar,
                "dtype": "float64",
            }
        },
    )
    return path


def write_fake_fme_checkpoint(
    path: pathlib.Path,
    *,
    lora_rank: int = 4,
    lora_alpha: float | None = 8.0,
    spectral_lora_rank: int = 0,
    spectral_lora_alpha: float | None = None,
    wrapped_multicall: bool = False,
) -> pathlib.Path:
    """Write a small PyTorch checkpoint mirroring fme 2026.5.1 `Trainer.save_checkpoint` layout."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(1234)

    module_sd: dict[str, Any] = {
        "module.encoder.conv.base_conv.weight": torch.randn((6, 3, 1, 1), generator=gen),
        "module.encoder.conv.base_conv.bias": torch.randn((6,), generator=gen),
        "module.decoder.weight": torch.randn((2, 6, 1, 1), generator=gen),
    }
    if lora_rank > 0:
        module_sd["module.encoder.conv.lora_down.weight"] = torch.randn(
            (lora_rank, 3, 1, 1), generator=gen
        )
        module_sd["module.encoder.conv.lora_up.weight"] = torch.randn(
            (6, lora_rank, 1, 1), generator=gen
        )
        module_sd["module.blocks.0.mlp.fc1.lora_down.weight"] = torch.randn(
            (lora_rank, 6, 1, 1), generator=gen
        )
        module_sd["module.blocks.0.mlp.fc1.lora_up.weight"] = torch.randn(
            (6, lora_rank, 1, 1), generator=gen
        )

    if spectral_lora_rank > 0:
        module_sd["module.blocks.0.filter.lora_A"] = torch.randn(
            (6, spectral_lora_rank), generator=gen
        )
        module_sd["module.blocks.0.filter.lora_B"] = torch.randn(
            (spectral_lora_rank, 6), generator=gen
        )

    builder_cfg = {
        "type": "SphericalFourierNeuralOperatorNet",
        "config": {
            "embed_dim": 6,
            "num_layers": 1,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "spectral_lora_rank": spectral_lora_rank,
            "spectral_lora_alpha": spectral_lora_alpha,
        },
    }

    single_step_cfg = {
        "type": "single_module",
        "config": {
            "builder": builder_cfg,
            "in_names": ["TMP2m", "surface_temperature"],
            "out_names": ["TMP2m", "surface_temperature"],
        },
    }

    if wrapped_multicall:
        step_cfg_selector = {
            "type": "multi_call",
            "config": {
                "wrapped_step": single_step_cfg,
            },
        }
        step_state: dict[str, Any] = {"wrapped_step": {"module": module_sd}}
    else:
        step_cfg_selector = single_step_cfg
        step_state = {"module": module_sd}

    ckpt = {
        "epoch": 5,
        "num_batches_seen": 100,
        "stepper": {
            "config": {"step": step_cfg_selector},
            "step": step_state,
            "dataset_info_tuple": ("latlon_grid", (4, 8), 6),
        },
    }
    torch.save(ckpt, str(path))
    return path


@pytest.fixture
def allow_tmp_run_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    """Configure `tmp_path/scratch` as the sole default output root for offline tests."""
    import aimip.validators as validators_mod

    scratch_sim = tmp_path / "scratch"
    scratch_sim.mkdir(exist_ok=True)
    monkeypatch.setattr(
        validators_mod,
        "ALLOWED_RUN_ROOTS",
        (f"{scratch_sim}/",),
    )
    return scratch_sim


@pytest.fixture
def playground_env(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Create a complete offline playground + registry v2 (`forcing_sets`) + scratch runs dir."""
    pg_dir = tmp_path / "playground"
    runs_dir = allow_tmp_run_root / "aimip-runs"
    pg_dir.mkdir()
    runs_dir.mkdir()
    (pg_dir / "bin").mkdir()
    wrapper = pg_dir / "bin" / "aimip"
    wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    # 1. Deterministic Gregorian model: ace2-era5 (dims: latitude/longitude, 2 IC files, amip-hist)
    era5_ckpt = pg_dir / "models" / "ace2-era5" / "ckpt.tar"
    era5_ckpt.parent.mkdir(parents=True)
    era5_ckpt.write_bytes(b"fake-ace2-era5-base-checkpoint-bytes")
    era5_sha = sha256_path(era5_ckpt)
    era5_ic_1979 = write_fake_ic_netcdf(
        pg_dir / "models" / "ace2-era5" / "ic" / "ic_1979.nc",
        [
            "1979-01-01T00:00:00",
            "1979-01-02T00:00:00",
            "1980-01-01T00:00:00",
            "1980-06-01T00:00:00",
        ],
        calendar="proleptic_gregorian",
        spatial_dims=("latitude", "longitude"),
        seed=1,
    )
    era5_ic_2001 = write_fake_ic_netcdf(
        pg_dir / "models" / "ace2-era5" / "ic" / "ic_2001.nc",
        [
            "2001-01-01T00:00:00",
            "2001-02-01T00:00:00",
        ],
        calendar="proleptic_gregorian",
        spatial_dims=("latitude", "longitude"),
        seed=11,
    )
    era5_forcing = pg_dir / "inputs" / "ace2-era5" / "amip-hist"
    era5_forcing.mkdir(parents=True)
    (era5_forcing / "forcing_1979.nc").write_bytes(b"x" * 128)
    (era5_forcing / "forcing_2001.nc").write_bytes(b"y" * 256)

    # 2. Stochastic Gregorian model: ace2s-shield-plus (dims: grid_yt/grid_xt, amip-hist + amip-oos)
    ace2s_ckpt = pg_dir / "models" / "ace2s-shield-plus" / "ckpt.tar"
    ace2s_ckpt.parent.mkdir(parents=True)
    ace2s_ckpt.write_bytes(b"fake-ace2s-stochastic-checkpoint-bytes")
    ace2s_sha = sha256_path(ace2s_ckpt)
    ace2s_ic = write_fake_ic_netcdf(
        pg_dir / "models" / "ace2s-shield-plus" / "ic" / "ic_1979.nc",
        ["1979-01-01T00:00:00"],
        calendar="proleptic_gregorian",
        spatial_dims=("grid_yt", "grid_xt"),
        seed=2,
    )
    ace2s_hist = pg_dir / "inputs" / "ace2s-shield-plus" / "amip-hist"
    ace2s_oos = pg_dir / "inputs" / "ace2s-shield-plus" / "amip-oos"
    ace2s_hist.mkdir(parents=True)
    ace2s_oos.mkdir(parents=True)
    (ace2s_hist / "forcing_hist.nc").write_bytes(b"h" * 64)
    (ace2s_oos / "forcing_oos.nc").write_bytes(b"o" * 80)

    # 3. Deterministic noleap model: ace2-eamv3 (dims: lat/lon, amip-hist 1994..2014)
    eam_ckpt = pg_dir / "models" / "ace2-eamv3" / "ckpt.tar"
    eam_ckpt.parent.mkdir(parents=True)
    eam_ckpt.write_bytes(b"fake-ace2-eamv3-noleap-checkpoint")
    eam_sha = sha256_path(eam_ckpt)
    eam_ic = write_fake_ic_netcdf(
        pg_dir / "models" / "ace2-eamv3" / "ic" / "ic_1994.nc",
        ["1994-01-01T00:00:00", "1996-01-01T00:00:00", "1996-02-01T00:00:00"],
        calendar="noleap",
        spatial_dims=("lat", "lon"),
        seed=3,
    )
    eam_forcing = pg_dir / "inputs" / "ace2-eamv3" / "amip-hist"
    eam_forcing.mkdir(parents=True)
    (eam_forcing / "forcing_eamv3.nc").write_bytes(b"w" * 96)

    registry_data = {
        "schema_version": 2,
        "models": [
            {
                "name": "ace2-era5",
                "fme_module": "fme.ace.inference",
                "checkpoint": "${PLAYGROUND}/models/ace2-era5/ckpt.tar",
                "checkpoint_sha256": era5_sha,
                "ic_files": [
                    "${PLAYGROUND}/models/ace2-era5/ic/ic_1979.nc",
                    "${PLAYGROUND}/models/ace2-era5/ic/ic_2001.nc",
                ],
                "forcing_sets": [
                    {
                        "name": "amip-hist",
                        "dir": "${PLAYGROUND}/inputs/ace2-era5/amip-hist",
                        "window": ["1979-01-01T00:00:00", "2014-12-31T18:00:00"],
                    }
                ],
                "timestep_hours": 6,
                "stochastic": False,
                "default_n_ensemble_per_ic": 1,
                "notes": "Synthetic ACE2-ERA5 deterministic test model (latitude/longitude).",
            },
            {
                "name": "ace2s-shield-plus",
                "fme_module": "fme.ace.inference",
                "checkpoint": "${PLAYGROUND}/models/ace2s-shield-plus/ckpt.tar",
                "checkpoint_sha256": ace2s_sha,
                "ic_files": ["${PLAYGROUND}/models/ace2s-shield-plus/ic/ic_1979.nc"],
                "forcing_sets": [
                    {
                        "name": "amip-hist",
                        "dir": "${PLAYGROUND}/inputs/ace2s-shield-plus/amip-hist",
                        "window": ["1979-01-01T00:00:00", "2014-12-31T18:00:00"],
                    },
                    {
                        "name": "amip-oos",
                        "dir": "${PLAYGROUND}/inputs/ace2s-shield-plus/amip-oos",
                        "window": ["2014-01-01T00:00:00", "2024-12-31T18:00:00"],
                    },
                ],
                "timestep_hours": 6,
                "stochastic": True,
                "default_n_ensemble_per_ic": 1,
                "notes": "Synthetic ACE2S stochastic test model (grid_yt/grid_xt, hist + oos).",
            },
            {
                "name": "ace2-eamv3",
                "fme_module": "fme.ace.inference",
                "checkpoint": "${PLAYGROUND}/models/ace2-eamv3/ckpt.tar",
                "checkpoint_sha256": eam_sha,
                "ic_files": ["${PLAYGROUND}/models/ace2-eamv3/ic/ic_1994.nc"],
                "forcing_sets": [
                    {
                        "name": "amip-hist",
                        "dir": "${PLAYGROUND}/inputs/ace2-eamv3/amip-hist",
                        "window": ["1994-01-01T00:00:00", "2014-12-31T18:00:00"],
                    }
                ],
                "timestep_hours": 6,
                "stochastic": False,
                "default_n_ensemble_per_ic": 1,
                "notes": "Synthetic ACE2-EAMv3 noleap calendar model (lat/lon).",
            },
        ],
    }

    reg_path = pg_dir / "registry.json"
    reg_path.write_text(json.dumps(registry_data, indent=2), encoding="utf-8")

    default_out_root = allow_tmp_run_root / "default-runs"
    default_out_root.mkdir(exist_ok=True)

    (pg_dir / "site.json").write_text(
        json.dumps(
            {
                "name": "levante",
                "scheduler": "slurm",
                "runtime": "apptainer",
                "sbatch": {
                    "partition": "gpu",
                    "account": "test_gpu",
                    "gres": "gpu:a100_80:1",
                    "cpus_per_task": 32,
                    "mem": "64G",
                    "max_time": "12:00:00",
                },
                "run_root": str(default_out_root),
                "extra_ro_binds": [],
                "gpu_check": "nvidia-smi",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg_dir))
    monkeypatch.setenv("AIMIP_REGISTRY", str(reg_path))
    monkeypatch.setenv("AIMIP_IMAGE_SHA", "sha256:deadbeef1234")
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(default_out_root))

    return {
        "playground_dir": pg_dir,
        "runs_dir": runs_dir,
        "default_out_root": default_out_root,
        "scratch_sim": allow_tmp_run_root,
        "registry_path": reg_path,
        "era5_ic_1979": era5_ic_1979,
        "era5_ic_2001": era5_ic_2001,
        "era5_ckpt": era5_ckpt,
        "era5_sha": era5_sha,
        "ace2s_ic": ace2s_ic,
        "ace2s_hist": ace2s_hist,
        "ace2s_oos": ace2s_oos,
        "eam_ic": eam_ic,
    }
