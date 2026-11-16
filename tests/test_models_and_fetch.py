"""Offline tests for additional fme models, coupled configurations, and fetch-on-first-use:
- Remaining 5 fme-family models: `ace2-som`, `ace2s`, `hiro-acev2`, `samudrace-cm4`, `samudrace-e3smv3`
- Dual fme runtime (`2026.5.1` base + `2026.10.0` overlay) and strict `fme.__version__` verification in `exec-fme`
- Coupled config builder (`fme.coupled.inference`), paired ocean ICs, `n_repeats`, `ICx3`, coupled spin/collect/restarts
- Fetch-on-first-use (`aimip fetch -m X`, auto-fetch in `aimip run` & `aimip ic spin`, `--no-fetch`, `fcntl.flock` concurrency)
- Coupled `status` and `summary` (`Tat2m` + `ocean.sst`)
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import cftime
import numpy as np
import pytest
import xarray as xr
import yaml

from aimip.cli import main
from aimip.errors import ConfigError, RegistryError, ValidationError
from aimip.lora import sha256_file
from aimip.registry import (
    SAMUDRACE_CM4_IC_SLUGS,
    build_default_registry_payload,
    load_registry,
)
from aimip.runner import (
    compute_coupled_steps_in_memory,
    compute_forcing_n_repeats,
    ensure_model_files_ready,
    prepare_run,
    resolve_model_for_prepare,
)
from aimip.setup_data import (
    CANONICAL_MODELS,
    HF_REVISIONS,
    SetupYearConfig,
    build_artifact_catalog,
    load_lock_file,
)
from conftest import write_fake_ic_netcdf


@pytest.fixture(autouse=True)
def _test_site_config(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> pathlib.Path:
    """Provide a valid test `site.json` (`account='test_gpu'`) since built-in presets require configuring `CHANGE_ME`."""
    site_runs = tmp_path / "site-runs"
    site_runs.mkdir(exist_ok=True)
    site_path = tmp_path / "test-site.json"
    site_path.write_text(
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
                "run_root": str(site_runs),
                "extra_ro_binds": [],
                "gpu_check": "nvidia-smi",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_path))
    return site_path


def _write_icx3_netcdf(
    path: pathlib.Path,
    *,
    timestamp: str = "0425-01-03T12:00:00",
    calendar: str = "noleap",
    var_names: Sequence[str] = ("Tat2m", "ps"),
    with_sample_coord: bool = True,
    n_samples: int = 3,
    seed: int = 10,
) -> pathlib.Path:
    """Write a synthetic 3-state (or restart) coupled IC netCDF with or without an explicit `sample` index coordinate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lats = np.array([-60.0, -20.0, 20.0, 60.0], dtype=np.float32)
    lons = np.linspace(0.0, 315.0, 8, dtype=np.float32)

    dt_val = cftime.datetime(
        int(timestamp[0:4]),
        int(timestamp[5:7]),
        int(timestamp[8:10]),
        int(timestamp[11:13]),
        int(timestamp[14:16]),
        int(timestamp[17:19]),
        calendar=calendar,
    )
    dt_arr = np.array([dt_val] * n_samples, dtype=object)

    data_vars: dict[str, Any] = {
        "time": xr.DataArray(dt_arr, dims=("sample",)),
    }
    for idx, vname in enumerate(var_names):
        arr = 285.0 + 3.0 * idx + rng.normal(0.0, 0.5, size=(n_samples, 4, 8))
        data_vars[vname] = xr.DataArray(
            arr.astype(np.float32),
            dims=("sample", "lat", "lon"),
        )

    coords: dict[str, Any] = {
        "lat": xr.DataArray(lats, dims=("lat",)),
        "lon": xr.DataArray(lons, dims=("lon",)),
    }
    if with_sample_coord:
        coords["sample"] = xr.DataArray(
            np.arange(n_samples, dtype=np.int64), dims=("sample",)
        )

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.to_netcdf(
        str(path),
        engine="netcdf4",
        encoding={
            "time": {
                "units": "hours since 0001-01-01 00:00:00",
                "calendar": calendar,
                "dtype": "float64",
            }
        },
    )
    return path


def _populate_extra_fme_data_tree_and_lock(
    hf_mirror: pathlib.Path,
    yc: SetupYearConfig,
    lock_out_path: pathlib.Path,
) -> dict[str, Any]:
    """Create a realistic offline HF mirror + exact `aimip.lock` for the 5 coupled/scenario fme models (`ace2-som`, `ace2s`, `hiro-acev2`, `samudrace-cm4`, `samudrace-e3smv3`)."""
    # 1. Checkpoints & non-LFS reference YAMLs
    blobs: list[tuple[str, str, bytes]] = [
        ("allenai/ACE2-SOM", "ace2_som_ckpt.tar", b"CKPT_ACE2_SOM_BYTES"),
        ("allenai/ACE2-SOM", "inference_config.yaml", b"# ace2-som ref\nn_forward_steps: 4\n"),
        ("allenai/HiRO-ACE", "ACE2S.ckpt", b"CKPT_ACE2S_BYTES"),
        ("allenai/HiRO-ACEv2", "ACE2S-X-SHiELD.ckpt", b"CKPT_HIRO_ACEV2_V11"),
        (
            "allenai/HiRO-ACEv2",
            "ace2s_inference_config_global.yaml",
            b"# hiro-acev2 ref\nn_forward_steps: 4\n",
        ),
        (
            "allenai/SamudrACE-CM4-piControl",
            "samudrACE_CM4_piControl_ckpt.tar",
            b"CKPT_SAMUDRACE_CM4_V11",
        ),
        (
            "allenai/SamudrACE-CM4-piControl",
            "inference_config.yaml",
            b"# samudrace-cm4 ref\nn_coupled_steps: 4\n",
        ),
        ("allenai/SamudrACE-E3SMv3", "SamudrACE-E3SMv3.tar", b"CKPT_SAMUDRACE_E3SMV3_V11"),
        (
            "allenai/SamudrACE-E3SMv3",
            "inference-config.yaml",
            b"# samudrace-e3smv3 ref\nn_coupled_steps: 4\n",
        ),
    ]
    for repo, rel, data in blobs:
        p = hf_mirror / repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    # 2. `ace2-som` 1xCO2 & 3xCO2 ICs + forcings
    for idx, scen in enumerate(("1xCO2", "3xCO2")):
        write_fake_ic_netcdf(
            hf_mirror / "allenai/ACE2-SOM" / "initial_conditions" / scen / "ic.nc",
            ["2032-01-01T00:00:00"],
            calendar="proleptic_gregorian",
            spatial_dims=("lat", "lon"),
            seed=20 + idx,
        )
        for y in yc.som_years:
            fp = hf_mirror / "allenai/ACE2-SOM" / "forcing_data" / scen / f"forcing_{y}.nc"
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(f"FORCING_SOM_{scen}_{y}".encode("ascii"))

    # 3. `ace2s` & `hiro-acev2` julian ICs (`ic_2014.nc`, `ic_2023.nc`) + forcings
    for m_idx, repo in enumerate(("allenai/HiRO-ACE", "allenai/HiRO-ACEv2")):
        for ic_year in (2014, 2023):
            write_fake_ic_netcdf(
                hf_mirror / repo / "initial_conditions" / f"ic_{ic_year}.nc",
                [f"{ic_year}-01-01T00:00:00"],
                calendar="julian",
                spatial_dims=("latitude", "longitude"),
                seed=30 + 10 * m_idx + (ic_year - 2014),
            )
        for y in yc.hiro_native_years:
            fp = hf_mirror / repo / "forcing_data" / f"forcing_{y}.nc"
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(f"FORCING_{repo}_{y}".encode("ascii"))

    # 4. `samudrace-cm4` coupled atmosphere + ocean ICs (with literal `:` in upstream HF directory names!) + forcings
    for fname in ("forcing_0151.nc", "forcing_0311.nc"):
        fp = hf_mirror / "allenai/SamudrACE-CM4-piControl" / "forcing_data" / fname
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(f"FORCING_CM4_{fname}".encode("ascii"))

    for s_idx, (upstream_ts, _slug) in enumerate(SAMUDRACE_CM4_IC_SLUGS):
        write_fake_ic_netcdf(
            hf_mirror
            / "allenai/SamudrACE-CM4-piControl"
            / "initial_conditions"
            / upstream_ts
            / "atmosphere"
            / "initial_condition.nc",
            [upstream_ts],
            calendar="noleap",
            spatial_dims=("lat", "lon"),
            var_names=("TMP2m", "PRESsfc"),
            seed=50 + 2 * s_idx,
        )
        write_fake_ic_netcdf(
            hf_mirror
            / "allenai/SamudrACE-CM4-piControl"
            / "initial_conditions"
            / upstream_ts
            / "ocean"
            / "initial_condition.nc",
            [upstream_ts],
            calendar="noleap",
            spatial_dims=("lat", "lon"),
            var_names=("sst", "thetao_0"),
            seed=51 + 2 * s_idx,
        )

    # 5. `samudrace-e3smv3` coupled 3-state `ICx3` (with explicit `sample` coordinate `[0, 1, 2]`) + dual 1yr forcings
    _write_icx3_netcdf(
        hf_mirror
        / "allenai/SamudrACE-E3SMv3"
        / "initial_conditions"
        / "SamudrACE-E3SMv3-ICx3-train_atmosphere_ic.nc",
        timestamp="0425-01-03T12:00:00",
        calendar="noleap",
        var_names=("Tat2m", "ps"),
        with_sample_coord=True,
        n_samples=3,
        seed=80,
    )
    _write_icx3_netcdf(
        hf_mirror
        / "allenai/SamudrACE-E3SMv3"
        / "initial_conditions"
        / "SamudrACE-E3SMv3-ICx3-train_ocean_ic.nc",
        timestamp="0425-01-03T12:00:00",
        calendar="noleap",
        var_names=("sst", "zos"),
        with_sample_coord=True,
        n_samples=3,
        seed=81,
    )
    for f_name in ("atmosphere-forcing-1yr.nc", "ocean-forcing-1yr.nc"):
        fp = hf_mirror / "allenai/SamudrACE-E3SMv3" / "forcing_data" / f_name
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(f"FORCING_E3SMV3_{f_name}".encode("ascii"))

    # Build lock file matching `build_artifact_catalog(years=yc)` for the 5 coupled/scenario fme models,
    # keeping `sha256: null` + `pin_after_first_fetch: True` on the 4 non-LFS reference YAMLs!
    extra_fme_models = {"ace2-som", "ace2s", "hiro-acev2", "samudrace-cm4", "samudrace-e3smv3"}
    cat = build_artifact_catalog(years=yc)
    lock_files: dict[str, Any] = {}
    for rel_path, spec in cat.items():
        if not (set(spec.models_needed) & extra_fme_models):
            continue
        assert spec.source_kind == "hf" and spec.hf_repo and spec.hf_path
        mirror_file = hf_mirror / spec.hf_repo / spec.hf_path
        f_bytes = mirror_file.read_bytes()
        import hashlib

        f_sha = hashlib.sha256(f_bytes).hexdigest()
        entry: dict[str, Any] = {
            "size": len(f_bytes),
            "sha256": None if spec.pin_after_first_fetch else f_sha,
            "source": {
                "kind": "hf",
                "repo": spec.hf_repo,
                "path": spec.hf_path,
                "revision": HF_REVISIONS[spec.hf_repo],
            },
        }
        if spec.pin_after_first_fetch:
            entry["pin_after_first_fetch"] = True
        lock_files[rel_path] = entry

    lock_doc = {"schema_version": 1, "files": lock_files}
    lock_out_path.write_text(json.dumps(lock_doc, indent=2), encoding="utf-8")
    return lock_doc


def test_registry_default_and_lock_contract_for_all_models(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify all 14 models in `registry.default.json` and their lock pins in `aimip.lock`."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))

    # Generated payload and packaged `registry.default.json` must match byte-for-byte in structure
    import aimip

    pkg_dir = pathlib.Path(aimip.__file__).resolve().parent
    packaged_reg = json.loads(
        (pkg_dir / "registry.default.json").read_text(encoding="utf-8")
    )
    assert packaged_reg == build_default_registry_payload()

    reg = load_registry(data_dir=str(data_dir))
    assert tuple(reg.keys()) == CANONICAL_MODELS
    assert len(reg) == 14   # 10 fme + 4 DLESyM / ArchesClimate-SSP

    expected_meta = {
        "ace2-era5": ("2026.5.1", "fme.ace.inference", "proleptic_gregorian", 6, False),
        "ace2-shield": ("2026.5.1", "fme.ace.inference", "julian", 6, False),
        "ace2s-shield-plus": ("2026.5.1", "fme.ace.inference", "julian", 6, True),
        "ace2-eamv3": ("2026.5.1", "fme.ace.inference", "noleap", 6, False),
        "hiro-ace": ("2026.5.1", "fme.ace.inference", "julian", 6, True),
        "ace2-som": ("2026.5.1", "fme.ace.inference", "proleptic_gregorian", 6, False),
        "ace2s": ("2026.5.1", "fme.ace.inference", "julian", 6, True),
        "hiro-acev2": ("2026.10.0", "fme.ace.inference", "julian", 6, True),
        "samudrace-cm4": ("2026.5.1", "fme.coupled.inference", "noleap", 120, False),
        "samudrace-e3smv3": ("2026.10.0", "fme.coupled.inference", "noleap", 120, True),
    }
    expected_emu = {   # (family, timestep_hours, stochastic, forcing sets)
        "dlesym-v1.0": ("e2s-dlesym", 6, False, ["era5-initial-state"]),
        "dlesym-v1.1": ("e2s-dlesym", 6, True, ["era5-initial-state"]),
        "dlesym-v0-isccp": ("e2s-dlesym", 6, False, ["era5-initial-state"]),
        "archesclimate-ssp": ("arches", 730, False, ["ssp434", "ssp585"]),
    }

    lock_files = load_lock_file(str(pkg_dir / "aimip.lock"))["files"]
    hex64_re = re.compile(r"^[0-9a-f]{64}$")

    for name, model in reg.items():
        if name in expected_emu:
            fam, step, stoch, sets = expected_emu[name]
            assert (model.family, model.timestep_hours, model.stochastic) == (fam, step, stoch)
            assert model.fme_module == "aimip.emu_job" and model.ic_files == () and not model.is_coupled
            assert [fs.name for fs in model.forcing_sets] == sets
            assert model.checkpoint.endswith((".mdlus", "model.ckpt"))
            for req_rel in model.required_files:
                le = lock_files[req_rel]
                assert hex64_re.match(le["sha256"]) and le["size"] > 0 and le["source"]["kind"] == "hf"
                assert re.match(r"^[0-9a-f]{40}$", le["source"]["revision"]), req_rel
            continue
        exp_ver, exp_mod, exp_cal, exp_step, exp_stoch = expected_meta[name]
        assert model.family == "fme"
        assert model.fme_version == exp_ver
        assert model.fme_module == exp_mod
        assert model.calendar == exp_cal
        assert model.timestep_hours == exp_step
        assert model.stochastic is exp_stoch
        assert len(model.required_files) > 0
        if model.is_coupled:
            assert len(model.ocean_ic_files) == len(model.ic_files)
            assert len(model.ocean_ic_files) >= 1
        else:
            assert model.ocean_ic_files == ()

        for req_rel in model.required_files:
            assert req_rel in lock_files, f"{name} required_file {req_rel} missing from aimip.lock"
            le = lock_files[req_rel]
            assert hex64_re.match(le["sha256"]), f"Unpinned sha256 on {name} required_file {req_rel}"
            if le["source"]["kind"] != "derived":
                assert isinstance(le["size"], int) and le["size"] > 0


def test_ace2_som_scenarios_1xco2_3xco2_and_2026_5_1_config(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`ace2-som` pairs both forcing directory AND scenario-specific IC for `1xCO2` vs `3xCO2`."""
    yc = SetupYearConfig(som_years=(2032, 2033, 2034, 2035), hiro_native_years=(2014, 2023))
    hf_mirror = tmp_path / "hf_som"
    lock_p = tmp_path / "fme_som.lock"
    _populate_extra_fme_data_tree_and_lock(hf_mirror, yc, lock_p)

    data_dir = tmp_path / "data_som"
    runs_dir = allow_tmp_run_root / "som_runs"
    runs_dir.mkdir()
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_p))
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    assert main(["fetch", "-m", "ace2-som", "--json"]) == 0
    capsys.readouterr()

    # 1. Default scenario (`1xCO2`)
    out_1x = runs_dir / "som_1x"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-som",
                "--start",
                "2032-01-01T00:00:00",
                "--steps",
                "8",
                "--out",
                str(out_1x),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    res_1x = json.loads(capsys.readouterr().out)
    assert res_1x["forcing_set"]["name"] == "1xCO2"
    assert res_1x["ic_path"] == str(data_dir / "inputs/ace2-som/ic/1xCO2/ic.nc")
    cfg_1x = yaml.safe_load((out_1x / "config.yaml").read_text(encoding="utf-8"))
    assert "seed" not in cfg_1x  # fme 2026.5.1 strict dacite rejects `seed`
    assert cfg_1x["forcing_loader"]["dataset"]["data_path"] == str(
        data_dir / "inputs/ace2-som/1xCO2"
    )

    # 2. `--forcing-set 3xco2` (case-insensitive) switches BOTH forcing dir and scenario IC!
    out_3x = runs_dir / "som_3x"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-som",
                "--start",
                "2032-01-01T00:00:00",
                "--steps",
                "8",
                "--forcing-set",
                "3xco2",
                "--out",
                str(out_3x),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    res_3x = json.loads(capsys.readouterr().out)
    assert res_3x["forcing_set"]["name"] == "3xCO2"
    assert res_3x["ic_path"] == str(data_dir / "inputs/ace2-som/ic/3xCO2/ic.nc")
    cfg_3x = yaml.safe_load((out_3x / "config.yaml").read_text(encoding="utf-8"))
    assert cfg_3x["forcing_loader"]["dataset"]["data_path"] == str(
        data_dir / "inputs/ace2-som/3xCO2"
    )


def test_ace2s_and_hiro_acev2_julian_window_stochastic_members_and_2026_10_seed(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`ace2s` (`2026.5.1`) vs `hiro-acev2` (`2026.10.0`) julian calendar, `seed` config emission, and `--members`."""
    yc = SetupYearConfig(som_years=(2032,), hiro_native_years=tuple(range(2014, 2024)))
    hf_mirror = tmp_path / "hf_hiro_v2"
    lock_p = tmp_path / "fme_hiro.lock"
    _populate_extra_fme_data_tree_and_lock(hf_mirror, yc, lock_p)

    data_dir = tmp_path / "data_hiro"
    runs_dir = allow_tmp_run_root / "hiro_runs"
    runs_dir.mkdir()
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_p))
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    assert main(["fetch", "-m", "ace2s,hiro-acev2", "--json"]) == 0
    capsys.readouterr()

    # 1. `ace2s` (`2026.5.1`, julian): no `seed` in `config.yaml`, `ic_2023.nc` selected for 2023 start
    out_ace2s = runs_dir / "ace2s_2023"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2s",
                "--start",
                "2023-01-01T00:00:00",
                "--steps",
                "12",
                "--members",
                "3",
                "--seed",
                "77",
                "--out",
                str(out_ace2s),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    cfg_ace2s = yaml.safe_load((out_ace2s / "config.yaml").read_text(encoding="utf-8"))
    assert "seed" not in cfg_ace2s
    assert cfg_ace2s["initial_condition"]["path"] == str(out_ace2s / "ic_members.nc")
    assert "start_indices" not in cfg_ace2s["initial_condition"]

    # 2. `hiro-acev2` (`2026.10.0`, julian): `seed` IS emitted in `config.yaml` (`fme_new_ace_inference.py:268`)
    out_v2 = runs_dir / "hiro_acev2_2014"
    assert (
        main(
            [
                "run",
                "-m",
                "hiro-acev2",
                "--start",
                "2014-01-01T00:00:00",
                "--steps",
                "16",
                "--members",
                "2",
                "--seed",
                "99",
                "--out",
                str(out_v2),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    cfg_v2 = yaml.safe_load((out_v2 / "config.yaml").read_text(encoding="utf-8"))
    assert cfg_v2["seed"] == 99
    assert cfg_v2["checkpoint_path"] == str(data_dir / "models/hiro-acev2/ACE2S-X-SHiELD.ckpt")
    assert cfg_v2["initial_condition"]["path"] == str(out_v2 / "ic_members.nc")
    with xr.open_dataset(str(out_v2 / "ic_members.nc"), engine="netcdf4", decode_times=False) as ds_rep:
        assert ds_rep.sizes["sample"] == 2


def test_samudrace_cm4_coupled_config_n_repeats_ic_list_spin_collect_and_continuation(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`samudrace-cm4` coupled config, `n_repeats`, `ic list`, `ic spin`/`collect`, and continuation."""
    # Direct unit checks on `coupled_steps_in_memory` & `n_repeats`
    assert compute_coupled_steps_in_memory(24) == 6
    assert compute_coupled_steps_in_memory(72) == 6
    assert compute_coupled_steps_in_memory(73) == 1
    assert compute_coupled_steps_in_memory(730) == 5
    assert compute_coupled_steps_in_memory(1460) == 5

    assert (
        compute_forcing_n_repeats(
            end_iso="0311-12-27T00:00:00",
            base_start_iso="0311-01-01T00:00:00",
            repeat_period_hours=365 * 24,
            calendar="noleap",
        )
        == 1
    )
    assert (
        compute_forcing_n_repeats(
            end_iso="0331-01-01T00:00:00",
            base_start_iso="0311-01-01T00:00:00",
            repeat_period_hours=365 * 24,
            calendar="noleap",
        )
        == 21
    )

    yc = SetupYearConfig(som_years=(2032,), hiro_native_years=(2014,))
    hf_mirror = tmp_path / "hf_cm4"
    lock_p = tmp_path / "fme_cm4.lock"
    _populate_extra_fme_data_tree_and_lock(hf_mirror, yc, lock_p)

    data_dir = tmp_path / "data_cm4"
    runs_dir = allow_tmp_run_root / "cm4_runs"
    runs_dir.mkdir()
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_p))
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    assert main(["fetch", "-m", "samudrace-cm4", "--json"]) == 0
    capsys.readouterr()

    # Verify colon-in-HF paths were staged into colon-free `$AIMIP_DATA` paths!
    staged_atmos = (
        data_dir / "inputs/samudrace-cm4/ic/0311-01-01T000000/atmosphere/initial_condition.nc"
    )
    staged_ocean = (
        data_dir / "inputs/samudrace-cm4/ic/0311-01-01T000000/ocean/initial_condition.nc"
    )
    assert staged_atmos.is_file() and ":" not in str(staged_atmos)
    assert staged_ocean.is_file() and ":" not in str(staged_ocean)

    # 1. `aimip ic list -m samudrace-cm4 --json` exposes both `ic_files` and `ocean_ic_files`
    assert main(["ic", "list", "-m", "samudrace-cm4", "--json"]) == 0
    ic_list_out = json.loads(capsys.readouterr().out)
    assert ic_list_out["model"] == "samudrace-cm4"
    assert len(ic_list_out["ic_files"]) == 6
    assert len(ic_list_out["ocean_ic_files"]) == 6

    # 2. 20-year run (`--steps 1460` from `0311-01-01T00:00:00`) writes expected coupled `config.yaml`
    out_20y = runs_dir / "cm4_20y"
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-cm4",
                "--start",
                "0311-01-01T00:00:00",
                "--steps",
                "1460",
                "--out",
                str(out_20y),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    cfg_20y = yaml.safe_load((out_20y / "config.yaml").read_text(encoding="utf-8"))
    assert "seed" not in cfg_20y
    assert cfg_20y["n_coupled_steps"] == 1460
    assert cfg_20y["coupled_steps_in_memory"] == 5
    assert cfg_20y["forcing_loader"]["atmosphere"]["dataset"]["n_repeats"] == 21
    assert cfg_20y["forcing_loader"]["num_data_workers"] == 1
    assert cfg_20y["initial_condition"]["atmosphere"]["path"] == str(staged_atmos)
    assert cfg_20y["initial_condition"]["ocean"]["path"] == str(staged_ocean)
    assert cfg_20y["data_writer"] == {
        "ocean": {"save_prediction_files": False, "save_monthly_files": True},
        "atmosphere": {"save_prediction_files": False, "save_monthly_files": True},
    }

    # 3. Later IC (`0313-01-01T00:00:00`, `--steps 72`) emits `n_repeats >= 3` for the 2-year offset
    out_0313 = runs_dir / "cm4_0313"
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-cm4",
                "--start",
                "0313-01-01T00:00:00",
                "--steps",
                "72",
                "--out",
                str(out_0313),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    cfg_0313 = yaml.safe_load((out_0313 / "config.yaml").read_text(encoding="utf-8"))
    assert cfg_0313["forcing_loader"]["atmosphere"]["dataset"]["n_repeats"] >= 3

    # 4. `--forcing-set picontrol-0151` from `0151-01-06T00:00:00`
    out_0151 = runs_dir / "cm4_0151"
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-cm4",
                "--start",
                "0151-01-06T00:00:00",
                "--steps",
                "12",
                "--forcing-set",
                "picontrol-0151",
                "--out",
                str(out_0151),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    cfg_0151 = yaml.safe_load((out_0151 / "config.yaml").read_text(encoding="utf-8"))
    assert cfg_0151["forcing_loader"]["atmosphere"]["dataset"]["file_pattern"] == "forcing_0151.nc"
    assert "0151-01-06T000000" in cfg_0151["initial_condition"]["ocean"]["path"]

    # 5. Coupled `ic spin` + `collect` + continuation `--ic <spun.nc>`
    spin_dir = runs_dir / "cm4_spin_dir"
    spun_atmos_out = runs_dir / "spun_0311_jan31.nc"
    assert (
        main(
            [
                "ic",
                "spin",
                "-m",
                "samudrace-cm4",
                "--from",
                "0311-01-01T00:00:00",
                "--to",
                "0311-01-31T00:00:00",
                "--out",
                str(spin_dir),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    spin_res = json.loads(capsys.readouterr().out)
    assert spin_res["is_coupled"] is True
    assert spin_res["members"][0]["steps"] == 6  # 30 days / 5 days = 6 coupled steps

    # Simulate coupled `fme.coupled.inference` restart outputs (`0/atmosphere/restart.nc` + `0/ocean/restart.nc`)
    write_fake_ic_netcdf(
        spin_dir / "0" / "atmosphere" / "restart.nc",
        ["0311-01-31T00:00:00"],
        calendar="noleap",
        sample_dim="sample",
        var_names=("TMP2m", "PRESsfc"),
        seed=90,
    )
    write_fake_ic_netcdf(
        spin_dir / "0" / "ocean" / "restart.nc",
        ["0311-01-31T00:00:00"],
        calendar="noleap",
        sample_dim="sample",
        var_names=("sst", "thetao_0"),
        seed=91,
    )

    assert main(["collect", str(spin_dir), "--out", str(spun_atmos_out), "--json"]) == 0
    col_res = json.loads(capsys.readouterr().out)
    assert col_res["path"] == str(spun_atmos_out)
    spun_ocean_out = runs_dir / "spun_0311_jan31_ocean.nc"
    assert spun_atmos_out.is_file()
    assert spun_ocean_out.is_file()

    # Continue run from `--ic <spun_atmos_out>` (auto-discovers `<stem>_ocean.nc`!)
    out_cont = runs_dir / "cm4_cont"
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-cm4",
                "--start",
                "0311-01-31T00:00:00",
                "--steps",
                "6",
                "--ic",
                str(spun_atmos_out),
                "--out",
                str(out_cont),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    cont_res = json.loads(capsys.readouterr().out)
    assert cont_res["ic_path"] == str(spun_atmos_out)
    assert cont_res["ocean_ic_path"] == str(spun_ocean_out)
    manifest_cont = json.loads((out_cont / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert manifest_cont["ocean_ic_sha256"] == sha256_file(str(spun_ocean_out))


def test_samudrace_e3smv3_coupled_2026_10_icx3_members_restarts_and_1yr_window_guard(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`samudrace-e3smv3` (`2026.10.0`): `ICx3` ensemble selection/replication, restart `sample`-coord guard, and 1yr cap."""
    yc = SetupYearConfig(som_years=(2032,), hiro_native_years=(2014,))
    hf_mirror = tmp_path / "hf_e3sm"
    lock_p = tmp_path / "fme_e3sm.lock"
    _populate_extra_fme_data_tree_and_lock(hf_mirror, yc, lock_p)

    data_dir = tmp_path / "data_e3sm"
    runs_dir = allow_tmp_run_root / "e3sm_runs"
    runs_dir.mkdir()
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_p))
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    assert main(["fetch", "-m", "samudrace-e3smv3", "--json"]) == 0
    capsys.readouterr()

    # 1. `--members 1`: selects index `[0]` via `start_indices`, emits `seed` (`2026.10.0`), dual forcing loaders
    out_m1 = runs_dir / "e3sm_m1"
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "0425-01-03T12:00:00",
                "--steps",
                "72",
                "--members",
                "1",
                "--seed",
                "123",
                "--out",
                str(out_m1),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    cfg_m1 = yaml.safe_load((out_m1 / "config.yaml").read_text(encoding="utf-8"))
    assert cfg_m1["seed"] == 123
    assert cfg_m1["initial_condition"]["start_indices"] == {
        "first": 0,
        "n_initial_conditions": 1,
    }
    assert (
        cfg_m1["forcing_loader"]["atmosphere"]["dataset"]["file_pattern"]
        == "atmosphere-forcing-1yr.nc"
    )
    assert cfg_m1["forcing_loader"]["num_data_workers"] == 1
    assert (
        cfg_m1["forcing_loader"]["ocean"]["dataset"]["file_pattern"]
        == "ocean-forcing-1yr.nc"
    )
    assert "n_repeats" not in cfg_m1["forcing_loader"]["atmosphere"]["dataset"]

    # 2. `--members 3`: selects all 3 `ICx3` states directly (`first: 0, n_initial_conditions: 3`, no replication)
    out_m3 = runs_dir / "e3sm_m3"
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "0425-01-03T12:00:00",
                "--steps",
                "6",
                "--members",
                "3",
                "--out",
                str(out_m3),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    cfg_m3 = yaml.safe_load((out_m3 / "config.yaml").read_text(encoding="utf-8"))
    assert cfg_m3["initial_condition"]["start_indices"] == {
        "first": 0,
        "n_initial_conditions": 3,
    }
    assert not (out_m3 / "ic_members_atmosphere.nc").exists()

    # 3. `--members 6`: replicates the 3 `ICx3` states 2x (`sample=6`) into paired `ic_members_{atmosphere,ocean}.nc`
    out_m6 = runs_dir / "e3sm_m6"
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "0425-01-03T12:00:00",
                "--steps",
                "6",
                "--members",
                "6",
                "--out",
                str(out_m6),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    cfg_m6 = yaml.safe_load((out_m6 / "config.yaml").read_text(encoding="utf-8"))
    assert "start_indices" not in cfg_m6["initial_condition"]
    assert cfg_m6["initial_condition"]["atmosphere"]["path"] == str(
        out_m6 / "ic_members_atmosphere.nc"
    )
    assert cfg_m6["initial_condition"]["ocean"]["path"] == str(
        out_m6 / "ic_members_ocean.nc"
    )
    with xr.open_dataset(
        str(out_m6 / "ic_members_ocean.nc"), engine="netcdf4", decode_times=False
    ) as ds_m6:
        assert ds_m6.sizes["sample"] == 6

    # 4. `--members 4` (when > 3 on `ICx3`): let's verify clean handling / refusal
    res_m4_rc = main(
        [
            "run",
            "-m",
            "samudrace-e3smv3",
            "--start",
            "0425-01-03T12:00:00",
            "--steps",
            "6",
            "--members",
            "4",
            "--out",
            str(runs_dir / "e3sm_m4_check"),
            "--dry-run",
        ]
    )
    capsys.readouterr()
    assert res_m4_rc in (0, 2)

    # 5. Restart continuation via `--ic <prev_run_dir>` (`atmosphere/restart.nc` + `ocean/restart.nc` WITHOUT `sample` coord):
    # `fme_new_coupled_inference.py:121-126` forbids `start_indices` when ocean IC has no `sample` coord!
    prev_run = runs_dir / "e3sm_prev_seg"
    _write_icx3_netcdf(
        prev_run / "atmosphere" / "restart.nc",
        timestamp="0425-02-02T12:00:00",
        calendar="noleap",
        var_names=("Tat2m", "ps"),
        with_sample_coord=False,
        n_samples=1,
        seed=95,
    )
    _write_icx3_netcdf(
        prev_run / "ocean" / "restart.nc",
        timestamp="0425-02-02T12:00:00",
        calendar="noleap",
        var_names=("sst", "zos"),
        with_sample_coord=False,
        n_samples=1,
        seed=96,
    )
    out_cont = runs_dir / "e3sm_cont_seg"
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "0425-02-02T12:00:00",
                "--steps",
                "6",
                "--ic",
                str(prev_run),
                "--seed",
                "55",
                "--out",
                str(out_cont),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    cfg_cont = yaml.safe_load((out_cont / "config.yaml").read_text(encoding="utf-8"))
    assert "start_indices" not in cfg_cont["initial_condition"]
    assert cfg_cont["initial_condition"]["atmosphere"]["path"] == str(
        prev_run / "atmosphere" / "restart.nc"
    )
    assert cfg_cont["initial_condition"]["ocean"]["path"] == str(
        prev_run / "ocean" / "restart.nc"
    )

    # 6. Exceeding 1-year window (`--steps 73` from `0425-01-03T12:00:00`) refuses cleanly (`no n_repeats` in `2026.10.0`)
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "0425-01-03T12:00:00",
                "--steps",
                "73",
                "--out",
                str(runs_dir / "e3sm_over_1yr"),
                "--dry-run",
            ]
        )
        == 2
    )
    assert "0426-01-03T12:00:00" in capsys.readouterr().err


def _install_fake_fme_package(
    site_root: pathlib.Path,
    *,
    version: str,
) -> pathlib.Path:
    """Create a synthetic `fme` package (`2026.5.1` or `2026.10.0`) supporting both `fme.ace.inference` and `fme.coupled.inference`."""
    pkg = site_root / "fme"
    (pkg / "core").mkdir(parents=True, exist_ok=True)
    (pkg / "ace" / "inference").mkdir(parents=True, exist_ok=True)
    (pkg / "coupled" / "inference").mkdir(parents=True, exist_ok=True)

    (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    (pkg / "core" / "__init__.py").write_text(
        "LAST_SEED = None\n"
        "def set_seed(s):\n"
        "    global LAST_SEED\n"
        "    LAST_SEED = s\n",
        encoding="utf-8",
    )
    (pkg / "ace" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "ace" / "inference" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "coupled" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "coupled" / "inference" / "__init__.py").write_text("", encoding="utf-8")

    ace_main = (
        "import json, pathlib, sys, yaml\n"
        "import fme, fme.core\n"
        "cfg = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())\n"
        "out = pathlib.Path(cfg['experiment_dir'])\n"
        "(out / 'fme_runtime_trace.json').write_text(json.dumps({\n"
        "    'version': fme.__version__,\n"
        "    'file': str(pathlib.Path(fme.__file__).resolve()),\n"
        "    'core_seed': fme.core.LAST_SEED,\n"
        "    'cfg_seed': cfg.get('seed'),\n"
        "}))\n"
        "(out / 'monthly_mean_predictions.nc').write_bytes(b'ACE_MONTHLY')\n"
        "(out / 'restart.nc').write_bytes(b'ACE_RESTART')\n"
    )
    (pkg / "ace" / "inference" / "__main__.py").write_text(ace_main, encoding="utf-8")

    coupled_main = (
        "import json, pathlib, sys, yaml\n"
        "import fme, fme.core\n"
        "cfg = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())\n"
        "out = pathlib.Path(cfg['experiment_dir'])\n"
        "(out / 'atmosphere').mkdir(parents=True, exist_ok=True)\n"
        "(out / 'ocean').mkdir(parents=True, exist_ok=True)\n"
        "(out / 'fme_runtime_trace.json').write_text(json.dumps({\n"
        "    'version': fme.__version__,\n"
        "    'file': str(pathlib.Path(fme.__file__).resolve()),\n"
        "    'core_seed': fme.core.LAST_SEED,\n"
        "    'cfg_seed': cfg.get('seed'),\n"
        "}))\n"
        "(out / 'atmosphere' / 'monthly_mean_predictions.nc').write_bytes(b'CPL_ATMOS_MONTHLY')\n"
        "(out / 'ocean' / 'monthly_mean_predictions.nc').write_bytes(b'CPL_OCEAN_MONTHLY')\n"
        "(out / 'atmosphere' / 'restart.nc').write_bytes(b'CPL_ATMOS_RESTART')\n"
        "(out / 'ocean' / 'restart.nc').write_bytes(b'CPL_OCEAN_RESTART')\n"
    )
    (pkg / "coupled" / "inference" / "__main__.py").write_text(
        coupled_main, encoding="utf-8"
    )
    return site_root


def test_exec_fme_overlay_2026_10_strict_version_check_and_ocean_ic_tamper_guard(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`exec-fme` activates `$AIMIP_FME_2026_10_OVERLAY` only for `2026.10.0`, verifies `fme.__version__`, and checks `ocean_ic_sha256`."""
    base_site = _install_fake_fme_package(tmp_path / "base_fme_2026_5_1", version="2026.5.1")
    overlay_site = _install_fake_fme_package(
        tmp_path / "overlay_fme_2026_10_0", version="2026.10.0"
    )
    monkeypatch.syspath_prepend(str(base_site))
    monkeypatch.setenv("AIMIP_FME_2026_10_OVERLAY", str(overlay_site))

    yc = SetupYearConfig(som_years=(2032,), hiro_native_years=(2014,))
    hf_mirror = tmp_path / "hf_exec"
    lock_p = tmp_path / "fme_exec.lock"
    _populate_extra_fme_data_tree_and_lock(hf_mirror, yc, lock_p)

    data_dir = tmp_path / "data_exec"
    runs_dir = allow_tmp_run_root / "exec_runs"
    runs_dir.mkdir()
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_p))
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    assert main(["fetch", "-m", "samudrace-e3smv3,samudrace-cm4", "--json"]) == 0
    capsys.readouterr()

    # 1. `samudrace-e3smv3` (`2026.10.0`) imports overlay `2026.10.0` and records `fme_version_imported`
    out_e3sm = runs_dir / "exec_e3sm"
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "0425-01-03T12:00:00",
                "--steps",
                "6",
                "--seed",
                "321",
                "--out",
                str(out_e3sm),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["exec-fme", str(out_e3sm)]) == 0
    capsys.readouterr()
    trace_e3sm = json.loads(
        (out_e3sm / "fme_runtime_trace.json").read_text(encoding="utf-8")
    )
    assert trace_e3sm["version"] == "2026.10.0"
    assert str(overlay_site.resolve()) in trace_e3sm["file"]
    assert trace_e3sm["cfg_seed"] == 321
    manifest_e3sm = json.loads(
        (out_e3sm / "aimip-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_e3sm["rc"] == 0
    assert "finished" in manifest_e3sm
    assert manifest_e3sm["fme_version_imported"] == "2026.10.0"

    # 2. Immediately run `samudrace-cm4` (`2026.5.1`) in the SAME Python process -> overlay was cleaned up, base `2026.5.1` runs!
    out_cm4 = runs_dir / "exec_cm4"
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-cm4",
                "--start",
                "0311-01-01T00:00:00",
                "--steps",
                "6",
                "--seed",
                "456",
                "--out",
                str(out_cm4),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["exec-fme", str(out_cm4)]) == 0
    capsys.readouterr()
    trace_cm4 = json.loads((out_cm4 / "fme_runtime_trace.json").read_text(encoding="utf-8"))
    assert trace_cm4["version"] == "2026.5.1"
    assert str(base_site.resolve()) in trace_cm4["file"]
    assert trace_cm4["core_seed"] == 456
    manifest_cm4 = json.loads((out_cm4 / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert manifest_cm4["rc"] == 0
    assert manifest_cm4["fme_version_imported"] == "2026.5.1"

    # 3. Wrong overlay version (`2026.5.1` imported when `2026.10.0` required) fails BEFORE `runpy`!
    out_mismatch = runs_dir / "exec_ver_mismatch"
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "0425-01-03T12:00:00",
                "--steps",
                "6",
                "--out",
                str(out_mismatch),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    monkeypatch.setenv("AIMIP_FME_2026_10_OVERLAY", str(base_site))
    assert main(["exec-fme", str(out_mismatch)]) == 2
    err_ver = capsys.readouterr().err
    assert "requires fme==2026.10.0, but imported fme=='2026.5.1'" in err_ver
    assert not (out_mismatch / "fme_runtime_trace.json").exists()
    manifest_mis = json.loads(
        (out_mismatch / "aimip-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_mis["rc"] == 1
    assert "fme==2026.10.0" in manifest_mis["error"]

    # 4. `--ocean-ic` enforces container-visible root policy AND `exec-fme` enforces `ocean_ic_sha256`!
    monkeypatch.setenv("AIMIP_FME_2026_10_OVERLAY", str(overlay_site))
    out_tamper = runs_dir / "exec_ocean_tamper"
    outside_ocean = tmp_path / "outside" / "outside_ocean.nc"
    _write_icx3_netcdf(
        outside_ocean,
        timestamp="0425-01-03T12:00:00",
        calendar="noleap",
        var_names=("sst", "zos"),
        with_sample_coord=True,
        n_samples=3,
        seed=88,
    )
    custom_atmos = (
        data_dir / "inputs/samudrace-e3smv3/ic/SamudrACE-E3SMv3-ICx3-train_atmosphere_ic.nc"
    )
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "0425-01-03T12:00:00",
                "--steps",
                "6",
                "--ic",
                str(custom_atmos),
                "--ocean-ic",
                str(outside_ocean),
                "--out",
                str(out_tamper),
                "--dry-run",
            ]
        )
        == 2
    )
    assert "path will not be visible inside the container" in capsys.readouterr().err

    custom_ocean = runs_dir / "custom_ocean.nc"
    _write_icx3_netcdf(
        custom_ocean,
        timestamp="0425-01-03T12:00:00",
        calendar="noleap",
        var_names=("sst", "zos"),
        with_sample_coord=True,
        n_samples=3,
        seed=88,
    )
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "0425-01-03T12:00:00",
                "--steps",
                "6",
                "--ic",
                str(custom_atmos),
                "--ocean-ic",
                str(custom_ocean),
                "--out",
                str(out_tamper),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    custom_ocean.write_bytes(b"CORRUPTED_OCEAN_IC_BYTES")
    assert main(["exec-fme", str(out_tamper)]) == 2
    assert "Ocean initial-condition SHA-256 mismatch" in capsys.readouterr().err


def test_fetch_on_first_use_cli_run_spin_no_fetch_and_concurrent_flock(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify `aimip fetch -m X`, auto-fetch at prepare time in `run`/`ic spin`, `--no-fetch`, and `fcntl.flock`."""
    yc = SetupYearConfig(som_years=(2032, 2033), hiro_native_years=(2014, 2015))
    hf_mirror = tmp_path / "hf_fetch"
    lock_p = tmp_path / "fme_fetch.lock"
    _populate_extra_fme_data_tree_and_lock(hf_mirror, yc, lock_p)

    data_dir = tmp_path / "data_fetch_first_use"
    runs_dir = allow_tmp_run_root / "fetch_runs"
    runs_dir.mkdir()
    pg_dir = tmp_path / "pg_wrapper"
    (pg_dir / "bin").mkdir(parents=True)
    (pg_dir / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg_dir))
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_p))
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    # Note: monkeypatch `SetupYearConfig` in `setup_data` so CLI `fetch` uses our compact `yc`
    import aimip.setup_data as setup_data_mod

    monkeypatch.setattr(setup_data_mod, "SetupYearConfig", lambda: yc)

    # 1. `aimip fetch -m ace2-som --json` downloads ONLY `ace2-som` (8 files with `som_years=(2032,2033)`),
    # prints progress + byte sizes to `stderr`, keeps `stdout` valid JSON, and computes `pin_after_first_fetch` sha256!
    assert main(["fetch", "-m", "ace2-som", "--json"]) == 0
    captured = capsys.readouterr()
    fetch_out = json.loads(captured.out)
    assert fetch_out["models"] == ["ace2-som"]
    assert fetch_out["downloaded_files"] == 8
    assert fetch_out["verified_models"] == ["ace2-som"]
    assert "models/ace2-som/inference_config.yaml" in fetch_out["pin_after_first_fetch"]
    assert (
        len(
            fetch_out["pin_after_first_fetch"]["models/ace2-som/inference_config.yaml"][
                "sha256"
            ]
        )
        == 64
    )
    assert "downloading 8 missing file(s)" in captured.err
    assert "models/ace2-som/ace2_som_ckpt.tar" in captured.err
    assert not (data_dir / "models/samudrace-e3smv3/SamudrACE-E3SMv3.tar").exists()
    assert list(load_registry(data_dir=str(data_dir)).keys()) == ["ace2-som"]

    # 2. Invalid `--start` on an unfetched model MUST fail BEFORE downloading anything!
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "NOT-AN-ISO-DATE",
                "--steps",
                "2",
                "--out",
                str(runs_dir / "bad_start_no_dl"),
                "--dry-run",
            ]
        )
        == 2
    )
    capsys.readouterr()
    assert not (data_dir / "models/samudrace-e3smv3/SamudrACE-E3SMv3.tar").exists()

    # 3. `--no-fetch` refuses cleanly when `samudrace-e3smv3` files are missing
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "0425-01-03T12:00:00",
                "--steps",
                "2",
                "--out",
                str(runs_dir / "no_fetch_refuse"),
                "--no-fetch",
                "--dry-run",
            ]
        )
        == 2
    )
    err_nf = capsys.readouterr().err
    assert "--no-fetch" in err_nf
    assert not (data_dir / "models/samudrace-e3smv3/SamudrACE-E3SMv3.tar").exists()

    # 3b. Read-only `$AIMIP_DATA` (`AIMIP_DATA_MODE=ro`) refuses auto-fetch cleanly.
    monkeypatch.setenv("AIMIP_DATA_MODE", "ro")
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "0425-01-03T12:00:00",
                "--steps",
                "2",
                "--out",
                str(runs_dir / "ro_data_refuse"),
                "--dry-run",
            ]
        )
        == 2
    )
    err_ro = capsys.readouterr().err
    assert (
        "model samudrace-e3smv3 is not installed here: "
        "ask the install owner to run `aimip fetch -m samudrace-e3smv3`"
    ) in err_ro
    assert not (data_dir / "models/samudrace-e3smv3/SamudrACE-E3SMv3.tar").exists()
    monkeypatch.setenv("AIMIP_DATA_MODE", "rw")

    # 4. Auto-fetch on `aimip run ... --print-sbatch-path`: downloads to `stderr`, emits ONLY single-line sbatch path on `stdout`!
    out_auto_run = runs_dir / "auto_fetched_e3sm"
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "0425-01-03T12:00:00",
                "--steps",
                "6",
                "--out",
                str(out_auto_run),
                "--print-sbatch-path",
            ]
        )
        == 0
    )
    auto_cap = capsys.readouterr()
    assert auto_cap.out.strip() == str(out_auto_run / "job.sbatch")
    assert "\n" not in auto_cap.out.strip()
    assert "downloading 6 missing file(s)" in auto_cap.err
    assert (data_dir / "models/samudrace-e3smv3/SamudrACE-E3SMv3.tar").is_file()
    assert list(load_registry(data_dir=str(data_dir)).keys()) == [
        "ace2-som",
        "samudrace-e3smv3",
    ]

    # 5. Auto-fetch on `aimip ic spin` (and `--no-fetch` refusal first) for `samudrace-cm4`
    spun_target_dir = runs_dir / "cm4_auto_spin_dir"
    assert (
        main(
            [
                "ic",
                "spin",
                "-m",
                "samudrace-cm4",
                "--from",
                "0311-01-01T00:00:00",
                "--to",
                "0311-01-11T00:00:00",
                "--out",
                str(spun_target_dir),
                "--no-fetch",
                "--dry-run",
            ]
        )
        == 2
    )
    capsys.readouterr()
    assert not (data_dir / "models/samudrace-cm4/samudrACE_CM4_piControl_ckpt.tar").exists()

    assert (
        main(
            [
                "ic",
                "spin",
                "-m",
                "samudrace-cm4",
                "--from",
                "0311-01-01T00:00:00",
                "--to",
                "0311-01-11T00:00:00",
                "--out",
                str(spun_target_dir),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    spin_cap = capsys.readouterr()
    spin_json = json.loads(spin_cap.out)
    assert spin_json["members"][0]["steps"] == 2
    assert "samudrace-cm4" in spin_cap.err
    assert list(load_registry(data_dir=str(data_dir)).keys()) == [
        "ace2-som",
        "samudrace-cm4",
        "samudrace-e3smv3",
    ]

    # 6. Concurrency safety (`fcntl.flock` on `<dest>/.aimip-fetch.lock`):
    # Two parallel subprocesses fetch `samudrace-e3smv3` (using default production catalog years —
    # `samudrace-e3smv3` has 6 fixed files regardless of year config!) into the SAME brand-new `--dest`.
    concurrent_dest = tmp_path / "concurrent_fetch_dest"
    import aimip

    src_root = pathlib.Path(aimip.__file__).resolve().parent.parent
    sub_env = dict(os.environ)
    sub_env["PYTHONPATH"] = (
        f"{src_root}:{sub_env['PYTHONPATH']}" if sub_env.get("PYTHONPATH") else str(src_root)
    )
    sub_env["AIMIP_HF_MIRROR"] = f"file://{hf_mirror}"
    sub_env["AIMIP_LOCK"] = str(lock_p)
    sub_env.pop("AIMIP_REGISTRY", None)

    worker_code = (
        "import sys; from aimip.cli import main; "
        f"sys.exit(main(['fetch', '-m', 'samudrace-e3smv3', '--dest', {str(concurrent_dest)!r}, '--json']))"
    )
    p1 = subprocess.Popen(
        [sys.executable, "-c", worker_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=sub_env,
    )
    p2 = subprocess.Popen(
        [sys.executable, "-c", worker_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=sub_env,
    )
    out1, err1 = p1.communicate(timeout=30)
    out2, err2 = p2.communicate(timeout=30)
    assert p1.returncode == 0, err1
    assert p2.returncode == 0, err2
    assert (concurrent_dest / ".aimip-fetch.lock").is_file()
    r1 = json.loads(out1)
    r2 = json.loads(out2)
    assert r1["verified_models"] == ["samudrace-e3smv3"]
    assert r2["verified_models"] == ["samudrace-e3smv3"]
    # Second lock holder sees files already staged by the first!
    assert sorted([r1["downloaded_files"], r2["downloaded_files"]]) == [0, 6]
    assert list(load_registry(data_dir=str(concurrent_dest)).keys()) == [
        "samudrace-e3smv3"
    ]


def test_coupled_status_and_summary_atmosphere_tat2m_and_ocean_sst(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Coupled `status` and `summary` inspect `atmosphere/` (`Tat2m`) and `ocean/` (`ocean.sst`)."""
    yc = SetupYearConfig(som_years=(2032,), hiro_native_years=(2014,))
    hf_mirror = tmp_path / "hf_sum"
    lock_p = tmp_path / "fme_sum.lock"
    _populate_extra_fme_data_tree_and_lock(hf_mirror, yc, lock_p)

    data_dir = tmp_path / "data_sum"
    runs_dir = allow_tmp_run_root / "sum_runs"
    runs_dir.mkdir()
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_p))
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    assert main(["fetch", "-m", "samudrace-e3smv3", "--json"]) == 0
    capsys.readouterr()

    run_dir = runs_dir / "coupled_sum_e3sm"
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-e3smv3",
                "--start",
                "0425-01-03T12:00:00",
                "--steps",
                "12",
                "--out",
                str(run_dir),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()

    # Mark manifest completed & write synthetic coupled `atmosphere/` (`Tat2m`) + `ocean/` (`sst` with land NaNs)
    manifest_p = run_dir / "aimip-manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    manifest["finished"] = "2025-01-01T00:00:00Z"
    manifest["rc"] = 0
    manifest["fme_version_imported"] = "2026.10.0"
    manifest_p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lats = np.array([-60.0, -20.0, 20.0, 60.0], dtype=np.float32)
    lons = np.linspace(0.0, 315.0, 8, dtype=np.float32)
    (run_dir / "atmosphere").mkdir(parents=True)
    (run_dir / "ocean").mkdir(parents=True)

    xr.Dataset(
        {
            "Tat2m": (
                ("sample", "time", "lat", "lon"),
                np.full((1, 2, 4, 8), 287.5, dtype=np.float32),
            )
        },
        coords={"lat": lats, "lon": lons},
    ).to_netcdf(str(run_dir / "atmosphere" / "monthly_mean_predictions.nc"), engine="netcdf4")

    sst_arr = np.full((1, 2, 4, 8), 291.25, dtype=np.float32)
    sst_arr[:, :, 0, :2] = np.nan  # masked land cells in ocean grid
    xr.Dataset(
        {"sst": (("sample", "time", "lat", "lon"), sst_arr)},
        coords={"lat": lats, "lon": lons},
    ).to_netcdf(str(run_dir / "ocean" / "monthly_mean_predictions.nc"), engine="netcdf4")

    (run_dir / "atmosphere" / "restart.nc").write_bytes(b"ATMOS_RESTART")
    (run_dir / "ocean" / "restart.nc").write_bytes(b"OCEAN_RESTART")

    assert main(["status", str(run_dir), "--json"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["state"] == "completed"
    assert st["artifacts"]["atmosphere/monthly_mean_predictions.nc"] is True
    assert st["artifacts"]["ocean/monthly_mean_predictions.nc"] is True
    assert st["artifacts"]["atmosphere/restart.nc"] is True
    assert st["artifacts"]["ocean/restart.nc"] is True

    assert main(["summary", str(run_dir), "--json"]) == 0
    sm = json.loads(capsys.readouterr().out)
    assert sm["ocean_monthly_file"] == str(run_dir / "ocean/monthly_mean_predictions.nc")
    assert sm["variables"]["Tat2m"]["series"][0]["ensemble_mean"] == pytest.approx(
        287.5, rel=1e-5
    )
    assert sm["variables"]["ocean.sst"]["series"][0]["ensemble_mean"] == pytest.approx(
        291.25, rel=1e-5
    )


def test_autofetch_preserves_explicit_registry_env_registry_and_caller_model(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Auto-fetch preserves `--registry`, `$AIMIP_REGISTRY`, and direct `prepare_run(model=...)`."""
    yc = SetupYearConfig(som_years=(2032, 2033), hiro_native_years=(2014, 2015))
    hf_mirror = tmp_path / "hf_reg"
    lock_p = tmp_path / "fme_reg.lock"
    _populate_extra_fme_data_tree_and_lock(hf_mirror, yc, lock_p)

    import aimip.setup_data as setup_data_mod

    monkeypatch.setattr(setup_data_mod, "SetupYearConfig", lambda: yc)

    pg_dir = tmp_path / "pg_reg"
    (pg_dir / "bin").mkdir(parents=True)
    (pg_dir / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runs_dir = allow_tmp_run_root / "reg_runs"
    runs_dir.mkdir()

    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg_dir))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_p))
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    # Seed a `${DATA}`-templated registry JSON with valid mirror checksums, then write customized `--registry` & `$AIMIP_REGISTRY`
    seed_data = tmp_path / "seed_data"
    monkeypatch.setenv("AIMIP_DATA", str(seed_data))
    assert main(["fetch", "-m", "ace2-som", "--json"]) == 0
    capsys.readouterr()
    base_reg_payload = json.loads((seed_data / "registry.json").read_text(encoding="utf-8"))

    cli_reg_payload = json.loads(json.dumps(base_reg_payload))
    cli_reg_payload["models"][0]["timestep_hours"] = 12
    cli_reg_payload["models"][0]["notes"] = "custom-cli-registry-12h"
    custom_cli_reg = tmp_path / "custom_cli_registry.json"
    custom_cli_reg.write_text(json.dumps(cli_reg_payload, indent=2), encoding="utf-8")

    env_reg_payload = json.loads(json.dumps(base_reg_payload))
    env_reg_payload["models"][0]["timestep_hours"] = 18
    env_reg_payload["models"][0]["notes"] = "custom-env-registry-18h"
    custom_env_reg = tmp_path / "custom_env_registry.json"
    custom_env_reg.write_text(json.dumps(env_reg_payload, indent=2), encoding="utf-8")

    # 1. Explicit `--registry` survives auto-fetch (even when `$AIMIP_REGISTRY` is ALSO set and `<data>/registry.json` is written!)
    data_cli_1 = tmp_path / "empty_data_cli_1"
    monkeypatch.setenv("AIMIP_DATA", str(data_cli_1))
    monkeypatch.setenv("AIMIP_REGISTRY", str(custom_env_reg))  # competing env registry (18h)
    resolved_cli = resolve_model_for_prepare(
        "ace2-som",
        registry_path=str(custom_cli_reg),
        playground_dir=str(pg_dir),
        data_dir=str(data_cli_1),
        auto_fetch=True,
    )
    capsys.readouterr()
    assert resolved_cli.timestep_hours == 12
    assert resolved_cli.notes == "custom-cli-registry-12h"
    # Data registry written by `fetch_model` has canonical 6h, proving reload did NOT clobber `--registry`!
    assert (
        load_registry(
            registry_path=str(data_cli_1 / "registry.json"),
            playground_dir=str(pg_dir),
            data_dir=str(data_cli_1),
        )["ace2-som"].timestep_hours
        == 6
    )

    data_cli_2 = tmp_path / "empty_data_cli_2"
    out_cli_run = runs_dir / "cli_reg_autofetch_run"
    monkeypatch.setenv("AIMIP_DATA", str(data_cli_2))
    assert (
        main(
            [
                "run",
                "--registry",
                str(custom_cli_reg),
                "-m",
                "ace2-som",
                "--start",
                "2032-01-01T00:00:00",
                "--steps",
                "4",
                "--out",
                str(out_cli_run),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest_cli = json.loads((out_cli_run / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert manifest_cli["registry_entry"]["timestep_hours"] == 12
    assert manifest_cli["end_time"] == "2032-01-03T00:00:00"  # 4 * 12h = 48h (not 4 * 6h = 24h)

    # 2. `$AIMIP_REGISTRY` survives auto-fetch when `--registry` is omitted
    data_env_1 = tmp_path / "empty_data_env_1"
    monkeypatch.setenv("AIMIP_DATA", str(data_env_1))
    monkeypatch.setenv("AIMIP_REGISTRY", str(custom_env_reg))
    resolved_env = resolve_model_for_prepare(
        "ace2-som",
        registry_path=None,
        playground_dir=str(pg_dir),
        data_dir=str(data_env_1),
        auto_fetch=True,
    )
    capsys.readouterr()
    assert resolved_env.timestep_hours == 18
    assert resolved_env.notes == "custom-env-registry-18h"

    data_env_2 = tmp_path / "empty_data_env_2"
    out_env_run = runs_dir / "env_reg_autofetch_run"
    monkeypatch.setenv("AIMIP_DATA", str(data_env_2))
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-som",
                "--start",
                "2032-01-01T00:00:00",
                "--steps",
                "4",
                "--out",
                str(out_env_run),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest_env = json.loads((out_env_run / "aimip-manifest.json").read_text(encoding="utf-8"))
    assert manifest_env["registry_entry"]["timestep_hours"] == 18
    assert manifest_env["end_time"] == "2032-01-04T00:00:00"  # 4 * 18h = 72h

    # 3. Direct `prepare_run` with a caller-supplied `ModelSpec` + missing files keeps the caller's model intact
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)
    data_direct = tmp_path / "empty_data_direct"
    monkeypatch.setenv("AIMIP_DATA", str(data_direct))
    caller_base = load_registry(
        registry_path=str(custom_cli_reg),
        playground_dir=str(pg_dir),
        data_dir=str(data_direct),
    )["ace2-som"]
    caller_spec = dataclasses.replace(
        caller_base,
        timestep_hours=24,
        notes="direct-caller-supplied-spec",
    )
    assert not (data_direct / "models/ace2-som/ace2_som_ckpt.tar").exists()

    res_direct = prepare_run(
        model=caller_spec,
        start_iso="2032-01-01T00:00:00",
        steps=3,
        out_dir=str(runs_dir / "direct_caller_run"),
        playground_dir=str(pg_dir),
        data_dir=str(data_direct),
        mode="dry-run",
        auto_fetch=True,
    )
    capsys.readouterr()
    assert (data_direct / "models/ace2-som/ace2_som_ckpt.tar").is_file()
    assert res_direct["registry_entry"]["timestep_hours"] == 24
    assert res_direct["end_time"] == "2032-01-04T00:00:00"  # 3 * 24h = 72h

    # 4. If a custom spec requires an extra file not produced by `fetch_model`, `ensure_model_files_ready` fails clearly
    data_extra = tmp_path / "empty_data_extra"
    monkeypatch.setenv("AIMIP_DATA", str(data_extra))
    bad_spec = dataclasses.replace(
        caller_spec,
        required_files=(*caller_spec.required_files, "models/ace2-som/unfetched_custom.bin"),
    )
    with pytest.raises(ConfigError, match="files this model entry needs are still missing"):
        ensure_model_files_ready(
            bad_spec,
            playground_dir=str(pg_dir),
            data_dir=str(data_extra),
            auto_fetch=True,
        )
    capsys.readouterr()


def test_first_command_autofetch_on_empty_data_dir_reloads_data_registry_and_cm4_colon_free_ics(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """First-ever `aimip run -m samudrace-cm4` on an empty `$AIMIP_DATA` (no `--registry`/`$AIMIP_REGISTRY`) reloads `<data>/registry.json` with colon-free ICs and lock SHAs."""
    yc = SetupYearConfig(som_years=(2032, 2033), hiro_native_years=(2014, 2015))
    hf_mirror = tmp_path / "hf_cm4_empty"
    lock_p = tmp_path / "fme_cm4_empty.lock"
    _populate_extra_fme_data_tree_and_lock(hf_mirror, yc, lock_p)

    import aimip.setup_data as setup_data_mod

    monkeypatch.setattr(setup_data_mod, "SetupYearConfig", lambda: yc)

    data_cm4 = tmp_path / "brand_new_empty_data_cm4"
    pg_dir = tmp_path / "pg_cm4_empty"
    (pg_dir / "bin").mkdir(parents=True)
    (pg_dir / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    out_cm4 = allow_tmp_run_root / "cm4_first_ever_run"

    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg_dir))
    monkeypatch.setenv("AIMIP_DATA", str(data_cm4))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_p))
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    assert not data_cm4.exists()
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-cm4",
                "--start",
                "0311-01-01T00:00:00",
                "--steps",
                "6",
                "--out",
                str(out_cm4),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    cfg = yaml.safe_load((out_cm4 / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["initial_condition"]["atmosphere"]["path"] == str(
        data_cm4 / "inputs/samudrace-cm4/ic/0311-01-01T000000/atmosphere/initial_condition.nc"
    )
    assert cfg["initial_condition"]["ocean"]["path"] == str(
        data_cm4 / "inputs/samudrace-cm4/ic/0311-01-01T000000/ocean/initial_condition.nc"
    )
    assert ":" not in cfg["initial_condition"]["atmosphere"]["path"]
    assert ":" not in cfg["initial_condition"]["ocean"]["path"]


def test_ace2_som_ic_continuation_inherits_scenario_or_refuses_orphan_ic(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--ic` without `--forcing-set` inherits `3xCO2` from `aimip-manifest.json`, refuses orphan ICs on overlapping scenario sets, lets explicit `--forcing-set` win, and leaves single-set / non-overlapping models unaffected."""
    yc = SetupYearConfig(som_years=(2032, 2033), hiro_native_years=(2014, 2015))
    hf_mirror = tmp_path / "hf_som"
    lock_p = tmp_path / "fme_som.lock"
    _populate_extra_fme_data_tree_and_lock(hf_mirror, yc, lock_p)

    import aimip.setup_data as setup_data_mod

    monkeypatch.setattr(setup_data_mod, "SetupYearConfig", lambda: yc)

    data_dir = tmp_path / "data_som"
    pg_dir = tmp_path / "pg_som"
    (pg_dir / "bin").mkdir(parents=True)
    (pg_dir / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg_dir))
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_p))
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    assert main(["fetch", "-m", "ace2-som,ace2s,samudrace-cm4", "--json"]) == 0
    capsys.readouterr()

    # 1. Prepare a 3xCO2 run and write its synthetic `restart.nc` at 2032-01-02T00:00:00.
    som_3x_dir = allow_tmp_run_root / "som_3xco2_seg1"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-som",
                "--start",
                "2032-01-01T00:00:00",
                "--steps",
                "4",
                "--forcing-set",
                "3xCO2",
                "--out",
                str(som_3x_dir),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    restart_3x = write_fake_ic_netcdf(
        som_3x_dir / "restart.nc",
        ["2032-01-02T00:00:00"],
        calendar="proleptic_gregorian",
        spatial_dims=("lat", "lon"),
        var_names=("TMP2m", "PRATEsfc", "surface_temperature"),
        seed=701,
    )

    # 1a. Continue with `--ic <run>/restart.nc` and NO `--forcing-set` -> inherits `3xCO2`.
    som_cont_dir = allow_tmp_run_root / "som_3xco2_seg2"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-som",
                "--start",
                "2032-01-02T00:00:00",
                "--steps",
                "4",
                "--ic",
                str(restart_3x),
                "--out",
                str(som_cont_dir),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    captured_cont = capsys.readouterr()
    assert "--ic comes from a 3xCO2 run" in captured_cont.err
    cont_manifest = json.loads(captured_cont.out)
    assert cont_manifest["forcing_set"]["name"] == "3xCO2"
    cont_cfg = yaml.safe_load((som_cont_dir / "config.yaml").read_text(encoding="utf-8"))
    assert cont_cfg["forcing_loader"]["dataset"]["data_path"].endswith("/inputs/ace2-som/3xCO2")

    # 1b. Symlink `--ic` pointing into `som_3x_dir/restart.nc` also resolves `realpath` and inherits `3xCO2`.
    links_dir = allow_tmp_run_root / "links"
    links_dir.mkdir()
    link_restart = links_dir / "latest_restart.nc"
    link_restart.symlink_to(restart_3x)
    som_link_cont = allow_tmp_run_root / "som_3xco2_seg2_via_link"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-som",
                "--start",
                "2032-01-02T00:00:00",
                "--steps",
                "4",
                "--ic",
                str(link_restart),
                "--out",
                str(som_link_cont),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    link_manifest = json.loads(capsys.readouterr().out)
    assert link_manifest["forcing_set"]["name"] == "3xCO2"

    # 2. Orphan IC file with no `aimip-manifest.json` and NO `--forcing-set` -> ConfigError (`rc=2`).
    orphan_ic = write_fake_ic_netcdf(
        allow_tmp_run_root / "orphan_som" / "custom_ic.nc",
        ["2032-01-02T00:00:00"],
        calendar="proleptic_gregorian",
        spatial_dims=("lat", "lon"),
        var_names=("TMP2m", "PRATEsfc", "surface_temperature"),
        seed=702,
    )
    rc_orphan = main(
        [
            "run",
            "-m",
            "ace2-som",
            "--start",
            "2032-01-02T00:00:00",
            "--steps",
            "4",
            "--ic",
            str(orphan_ic),
            "--out",
            str(allow_tmp_run_root / "som_orphan_fail"),
            "--dry-run",
        ]
    )
    assert rc_orphan == 2
    err_orphan = capsys.readouterr().err
    assert "1xCO2, 3xCO2 all cover this window" in err_orphan
    assert "Pass --forcing-set" in err_orphan

    # 3. Explicit `--forcing-set 1xCO2` still wins even when `--ic` comes from a `3xCO2` run.
    som_explicit_dir = allow_tmp_run_root / "som_explicit_1xco2"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-som",
                "--start",
                "2032-01-02T00:00:00",
                "--steps",
                "4",
                "--ic",
                str(restart_3x),
                "--forcing-set",
                "1xCO2",
                "--out",
                str(som_explicit_dir),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    explicit_manifest = json.loads(capsys.readouterr().out)
    assert explicit_manifest["forcing_set"]["name"] == "1xCO2"

    # 4. Single-set (`ace2s`) and non-overlapping scenario-set (`samudrace-cm4`) models are unaffected by orphan `--ic`.
    orphan_ace2s_ic = write_fake_ic_netcdf(
        allow_tmp_run_root / "orphan_ace2s" / "ic.nc",
        ["2014-01-02T00:00:00"],
        calendar="julian",
        spatial_dims=("grid_yt", "grid_xt"),
        var_names=("TMP2m", "PRATEsfc", "surface_temperature"),
        seed=703,
    )
    assert (
        main(
            [
                "run",
                "-m",
                "ace2s",
                "--start",
                "2014-01-02T00:00:00",
                "--steps",
                "4",
                "--seed",
                "0",
                "--ic",
                str(orphan_ace2s_ic),
                "--out",
                str(allow_tmp_run_root / "ace2s_orphan_ok"),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()

    orphan_cm4_atmos = write_fake_ic_netcdf(
        allow_tmp_run_root / "orphan_cm4" / "atmosphere" / "restart.nc",
        ["0311-01-06T00:00:00"],
        calendar="noleap",
        spatial_dims=("lat", "lon"),
        var_names=("TMP2m", "surface_temperature"),
        seed=704,
    )
    write_fake_ic_netcdf(
        allow_tmp_run_root / "orphan_cm4" / "ocean" / "restart.nc",
        ["0311-01-06T00:00:00"],
        calendar="noleap",
        spatial_dims=("yh", "xh"),
        var_names=("sst", "thetao_0"),
        seed=705,
    )
    assert (
        main(
            [
                "run",
                "-m",
                "samudrace-cm4",
                "--start",
                "0311-01-06T00:00:00",
                "--steps",
                "6",
                "--ic",
                str(orphan_cm4_atmos.parent.parent),
                "--out",
                str(allow_tmp_run_root / "cm4_orphan_ok"),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
