"""Tests for `aimip site init`, `${DATA}` registry resolution, deterministic builders, and `aimip setup` / `aimip.lock`."""

from __future__ import annotations

import calendar as py_calendar
import datetime
import json
import pathlib
import re
import shutil
import time
from typing import Any

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aimip.builders import (
    build_co2_table,
    build_eamv3_forcing,
    build_eamv3_ic,
    build_hiro_forcing,
    build_hiro_ic,
    build_oos_forcing,
)
from aimip.cli import main
from aimip.errors import ConfigError, ValidationError
from aimip.registry import (
    SAMUDRACE_CM4_IC_SLUGS,
    build_default_registry_payload,
    load_registry,
)
from aimip.setup_data import (
    CANONICAL_MODELS,
    REVISION_TBD,
    SHA256_TBD,
    ZENODO_AIMIP_FILENAME,
    ZENODO_AIMIP_RECORD,
    ZENODO_AIMIP_REL_PATH,
    SetupYearConfig,
    build_artifact_catalog,
    download_hf_artifact,
    download_zenodo_artifact,
    load_lock_file,
    md5_file,
    run_setup,
)


def _create_synthetic_upstream_mirrors(
    hf_mirror: pathlib.Path,
    zenodo_mirror: pathlib.Path,
    yc: SetupYearConfig,
) -> None:
    """Populate tiny realistic HF and Zenodo directory mirrors for all registered models."""
    lats = np.array([-67.5, -22.5, 22.5, 67.5], dtype=np.float64)
    lons = np.linspace(22.5, 337.5, 8, dtype=np.float64)

    # 1. Checkpoints for classic ACE models
    ckpt_map = [
        ("allenai/ACE2-ERA5", "ace2_era5_ckpt.tar", b"FAKE_ERA5_CKPT_BYTES"),
        ("allenai/ACE2-SHiELD", "ace2_shield_ckpt.tar", b"FAKE_SHIELD_CKPT_BYTES"),
        ("allenai/ACE2S-SHiELD-plus", "ace2s_shield_plus_ckpt.tar", b"FAKE_SHIELDPLUS_CKPT_BYTES"),
        ("allenai/ACE2-EAMv3", "ace2_EAMv3_ckpt.tar", b"FAKE_EAMV3_CKPT_BYTES"),
        ("allenai/HiRO-ACE", "ACE2S.ckpt", b"FAKE_HIRO_CKPT_BYTES"),
    ]
    for repo, rel, content in ckpt_map:
        p = hf_mirror / repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)

    # 2. Direct HF ICs (`ace2-era5` and `ACE2S-SHiELD-plus`)
    for ic_name, year in (("ic_1979.nc", 1979), ("ic_2001.nc", 2001)):
        p = hf_mirror / "allenai/ACE2-ERA5" / "initial_conditions" / ic_name
        p.parent.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(
            {
                "TMP2m": (
                    ("time", "latitude", "longitude"),
                    np.full((1, 4, 8), 288.0, dtype=np.float32),
                )
            },
            coords={
                "time": [np.datetime64(f"{year}-01-01T00:00:00", "ns")],
                "latitude": lats,
                "longitude": lons,
            },
        )
        ds.to_netcdf(
            str(p),
            engine="netcdf4",
            encoding={
                "time": {
                    "units": "hours since 1979-01-01T00:00:00",
                    "calendar": "proleptic_gregorian",
                    "dtype": "float64",
                }
            },
        )

    shield_ic_path = (
        hf_mirror / "allenai/ACE2S-SHiELD-plus" / "initial_conditions" / "AMIP" / "ic.nc"
    )
    shield_ic_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "TMP2m": (
                ("time", "grid_yt", "grid_xt"),
                np.full((1, 4, 8), 289.0, dtype=np.float32),
            )
        },
        coords={
            "time": [np.datetime64("1979-01-01T00:00:00", "ns")],
            "grid_yt": lats,
            "grid_xt": lons,
        },
    ).to_netcdf(
        str(shield_ic_path),
        engine="netcdf4",
        encoding={
            "time": {
                "units": "hours since 1979-01-01T00:00:00",
                "calendar": "proleptic_gregorian",
                "dtype": "float64",
            }
        },
    )

    # Helper for 6-hourly calendar timestamps over a short window (4 noleap steps + optional Feb 29)
    def _short_year_times(y: int) -> np.ndarray:
        base = [
            f"{y:04d}-01-01T00:00:00",
            f"{y:04d}-01-01T06:00:00",
            f"{y:04d}-02-28T18:00:00",
        ]
        if py_calendar.isleap(y):
            base.append(f"{y:04d}-02-29T00:00:00")
        base.append(f"{y:04d}-03-01T00:00:00")
        return np.array(base, dtype="datetime64[ns]")

    # Land mask with mixed ocean/land/coastal cells
    shield_land = np.zeros((4, 8), dtype=np.float32)
    shield_land[1, 2:5] = 1.0
    shield_land[2, 3] = 0.7

    hiro_land = shield_land.copy()
    # Make cell (2, 3) water in HiRO (< 0.5) while >= 0.5 in SHiELD to exercise v2 coastal SST refill!
    hiro_land[2, 3] = 0.2

    # 3. ERA5 yearly forcing (`allenai/ACE2-ERA5/forcing_data/forcing_YYYY.nc`)
    for y in yc.era5_hist_years:
        t_arr = _short_year_times(y)
        nt = len(t_arr)
        p = hf_mirror / "allenai/ACE2-ERA5" / "forcing_data" / f"forcing_{y}.nc"
        p.parent.mkdir(parents=True, exist_ok=True)
        co2_val = np.float32(3.50e-4 + (y - 1990) * 2.0e-6)
        ds = xr.Dataset(
            {
                "surface_temperature": (
                    ("time", "latitude", "longitude"),
                    np.full((nt, 4, 8), 295.0 + 0.1 * (y - 1994), dtype=np.float32),
                ),
                "sea_ice_fraction": (
                    ("time", "latitude", "longitude"),
                    np.full((nt, 4, 8), 0.15, dtype=np.float32),
                ),
                "land_fraction": (
                    ("time", "latitude", "longitude"),
                    np.broadcast_to(shield_land, (nt, 4, 8)).copy(),
                ),
                "DSWRFtoa": (
                    ("time", "latitude", "longitude"),
                    np.full((nt, 4, 8), 340.0, dtype=np.float32),
                ),
                "global_mean_co2": (
                    ("time",),
                    np.full((nt,), co2_val, dtype=np.float32),
                ),
            },
            coords={"time": t_arr, "latitude": lats, "longitude": lons},
        )
        ds.to_netcdf(
            str(p),
            engine="netcdf4",
            encoding={
                "time": {
                    "units": "hours since 1979-01-01T00:00:00",
                    "calendar": "proleptic_gregorian",
                    "dtype": "float64",
                }
            },
        )

    # Also write continuous 6-hourly CO2-bearing ERA5 files for `co2_y0..co2_last_upstream`
    # when those years aren't already in `era5_hist_years` (or overwrite with full 6h CO2 if needed).
    # For fast tests we keep `co2_y0..co2_y1` to 2 years or pass a `co2_year_reader`.

    # 4. SHiELD+ AMIP yearly forcing (`allenai/ACE2S-SHiELD-plus/forcing_data/AMIP/forcing_YYYY.nc`)
    for y in yc.shield_hist_years:
        t_arr = _short_year_times(y)
        nt = len(t_arr)
        p = (
            hf_mirror
            / "allenai/ACE2S-SHiELD-plus"
            / "forcing_data"
            / "AMIP"
            / f"forcing_{y}.nc"
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        co2_val = np.float32(3.50e-4 + (y - 1990) * 2.0e-6)
        ice = np.minimum(0.1, 1.0 - shield_land)
        ds = xr.Dataset(
            {
                "surface_temperature": (
                    ("time", "grid_yt", "grid_xt"),
                    np.full((nt, 4, 8), 296.0 + 0.1 * (y - 1994), dtype=np.float32),
                ),
                "sea_ice_fraction": (
                    ("time", "grid_yt", "grid_xt"),
                    np.broadcast_to(ice.astype(np.float32), (nt, 4, 8)).copy(),
                ),
                "land_fraction": (
                    ("time", "grid_yt", "grid_xt"),
                    np.broadcast_to(shield_land, (nt, 4, 8)).copy(),
                ),
                "ocean_fraction": (
                    ("time", "grid_yt", "grid_xt"),
                    np.broadcast_to((1.0 - shield_land - ice).astype(np.float32), (nt, 4, 8)).copy(),
                ),
                "DSWRFtoa": (
                    ("time", "grid_yt", "grid_xt"),
                    np.full((nt, 4, 8), 341.0, dtype=np.float32),
                ),
                "global_mean_co2": (
                    ("time",),
                    np.full((nt,), co2_val, dtype=np.float32),
                ),
                "ak_0": np.float32(0.0),
                "bk_0": np.float32(1.0),
            },
            coords={"time": t_arr, "grid_yt": lats, "grid_xt": lons},
        )
        ds.to_netcdf(
            str(p),
            engine="netcdf4",
            encoding={
                "time": {
                    "units": "hours since 1979-01-01T00:00:00",
                    "calendar": "proleptic_gregorian",
                    "dtype": "float64",
                }
            },
        )

    # 5. Raw EAMv3 1971 IC and 1971 forcing (`4` noleap steps matching `_short_year_times` minus Feb 29)
    eam_ic_raw = hf_mirror / "allenai/ACE2-EAMv3" / "initial_conditions" / "1971010100.nc"
    eam_ic_raw.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "THT": (("time", "lat", "lon"), np.full((1, 4, 8), 287.5, dtype=np.float32)),
            "ak_0": np.float32(10.0),
            "bk_0": np.float32(0.95),
        },
        coords={
            "time": [cftime.DatetimeNoLeap(1971, 1, 1)],
            "lat": lats,
            "lon": lons,
        },
    ).to_netcdf(
        str(eam_ic_raw),
        engine="netcdf4",
        encoding={
            "time": {
                "units": "hours since 1970-01-01T00:00:00",
                "calendar": "noleap",
                "dtype": "float64",
            }
        },
    )

    eam_f_raw = hf_mirror / "allenai/ACE2-EAMv3" / "forcing_data" / "1971.nc"
    eam_f_raw.parent.mkdir(parents=True, exist_ok=True)
    t_1971 = [
        cftime.DatetimeNoLeap(1971, 1, 1) + datetime.timedelta(hours=6 * i)
        for i in range(4)
    ]
    xr.Dataset(
        {
            "LANDFRAC": (
                ("time", "lat", "lon"),
                np.broadcast_to(shield_land, (4, 4, 8)).copy(),
            ),
            "SOLIN": (
                ("time", "lat", "lon"),
                np.full((4, 4, 8), 338.0, dtype=np.float32),
            ),
            "PHIS": (("lat", "lon"), np.full((4, 8), 120.0, dtype=np.float32)),
            "ak_0": np.float32(10.0),
            "bk_0": np.float32(0.95),
        },
        coords={"time": t_1971, "lat": lats, "lon": lons},
    ).to_netcdf(
        str(eam_f_raw),
        engine="netcdf4",
        encoding={
            "time": {
                "units": "hours since 1970-01-01T00:00:00",
                "calendar": "noleap",
                "dtype": "float64",
            }
        },
    )

    # 6. Raw HiRO-ACE `ic_2014.nc` and `forcing_2014.nc`
    hiro_ic_raw = hf_mirror / "allenai/HiRO-ACE" / "initial_conditions" / "ic_2014.nc"
    hiro_ic_raw.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "TMP2m": (
                ("time", "latitude", "longitude"),
                np.full((1, 4, 8), 290.5, dtype=np.float32),
            )
        },
        coords={
            "time": [np.datetime64("2014-01-01T00:00:00", "ns")],
            "latitude": lats,
            "longitude": lons,
        },
    ).to_netcdf(
        str(hiro_ic_raw),
        engine="netcdf4",
        encoding={
            "time": {
                "units": "hours since 1979-01-01T00:00:00",
                "calendar": "proleptic_gregorian",
                "dtype": "float64",
            }
        },
    )

    hiro_f_raw = hf_mirror / "allenai/HiRO-ACE" / "forcing_data" / "forcing_2014.nc"
    hiro_f_raw.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "land_fraction": (
                ("time", "latitude", "longitude"),
                hiro_land[None, ...],
            ),
            "HGTsfc": (
                ("time", "latitude", "longitude"),
                np.full((1, 4, 8), 55.0, dtype=np.float32),
            ),
            "ak_0": (("time",), np.array([0.0], dtype=np.float32)),
            "bk_0": (("time",), np.array([1.0], dtype=np.float32)),
        },
        coords={
            "time": [np.datetime64("2014-01-01T00:00:00", "ns")],
            "latitude": lats,
            "longitude": lons,
        },
    ).to_netcdf(
        str(hiro_f_raw),
        engine="netcdf4",
        encoding={
            "time": {
                "units": "hours since 1979-01-01T00:00:00",
                "calendar": "proleptic_gregorian",
                "dtype": "float64",
            }
        },
    )

    # 7. Zenodo 17065758 monthly SST/SIC (`8x16` finer source grid with NaNs over land + seam)
    zen_path = zenodo_mirror / ZENODO_AIMIP_RECORD / ZENODO_AIMIP_FILENAME
    zen_path.parent.mkdir(parents=True, exist_ok=True)
    lat_s = np.linspace(-78.75, 78.75, 8, dtype=np.float64)
    lon_s = np.linspace(11.25, 348.75, 16, dtype=np.float64)
    months = pd.date_range(yc.oos_anchor_t0, yc.oos_anchor_t1, freq="MS")
    nm = len(months)
    sst_s = np.full((nm, 8, 16), 298.5, dtype=np.float32)
    sic_s = np.full((nm, 8, 16), 0.05, dtype=np.float32)
    # Inject NaNs on a target cell patch near the longitude boundary to exercise `_fill_nearest`
    sst_s[:, 2:4, 0:2] = np.nan
    sic_s[:, 2:4, 0:2] = np.nan
    xr.Dataset(
        {
            "sea_surface_temperature": (("time", "latitude", "longitude"), sst_s),
            "sea_ice_cover": (("time", "latitude", "longitude"), sic_s),
        },
        coords={"time": months.values, "latitude": lat_s, "longitude": lon_s},
    ).to_netcdf(str(zen_path), engine="netcdf4")

    # 8. Additional fme-family models (`ace2-som`, `ace2s`, `hiro-acev2`, `samudrace-cm4`, `samudrace-e3smv3`)
    extra_fme_static_blobs = [
        ("allenai/ACE2-SOM", "ace2_som_ckpt.tar", b"FAKE_ACE2_SOM_CKPT_BYTES"),
        ("allenai/ACE2-SOM", "inference_config.yaml", b"n_forward_steps: 4\n"),
        ("allenai/ACE2-SOM", "initial_conditions/1xCO2/ic.nc", b"FAKE_SOM_IC_1XCO2"),
        ("allenai/ACE2-SOM", "initial_conditions/3xCO2/ic.nc", b"FAKE_SOM_IC_3XCO2"),
        ("allenai/HiRO-ACE", "initial_conditions/ic_2023.nc", b"FAKE_ACE2S_IC_2023"),
        ("allenai/HiRO-ACEv2", "ACE2S-X-SHiELD.ckpt", b"FAKE_HIRO_ACEV2_CKPT_BYTES"),
        ("allenai/HiRO-ACEv2", "ace2s_inference_config_global.yaml", b"n_forward_steps: 4\n"),
        ("allenai/HiRO-ACEv2", "initial_conditions/ic_2014.nc", b"FAKE_HIRO_V2_IC_2014"),
        ("allenai/HiRO-ACEv2", "initial_conditions/ic_2023.nc", b"FAKE_HIRO_V2_IC_2023"),
        (
            "allenai/SamudrACE-CM4-piControl",
            "samudrACE_CM4_piControl_ckpt.tar",
            b"FAKE_SAMUDRACE_CM4_CKPT_BYTES",
        ),
        ("allenai/SamudrACE-CM4-piControl", "inference_config.yaml", b"n_coupled_steps: 4\n"),
        ("allenai/SamudrACE-CM4-piControl", "forcing_data/forcing_0151.nc", b"FAKE_CM4_F_0151"),
        ("allenai/SamudrACE-CM4-piControl", "forcing_data/forcing_0311.nc", b"FAKE_CM4_F_0311"),
        ("allenai/SamudrACE-E3SMv3", "SamudrACE-E3SMv3.tar", b"FAKE_SAMUDRACE_E3SMV3_CKPT_BYTES"),
        ("allenai/SamudrACE-E3SMv3", "inference-config.yaml", b"n_coupled_steps: 4\n"),
        (
            "allenai/SamudrACE-E3SMv3",
            "initial_conditions/SamudrACE-E3SMv3-ICx3-train_atmosphere_ic.nc",
            b"FAKE_E3SMV3_ATMOS_ICX3",
        ),
        (
            "allenai/SamudrACE-E3SMv3",
            "initial_conditions/SamudrACE-E3SMv3-ICx3-train_ocean_ic.nc",
            b"FAKE_E3SMV3_OCEAN_ICX3",
        ),
        (
            "allenai/SamudrACE-E3SMv3",
            "forcing_data/atmosphere-forcing-1yr.nc",
            b"FAKE_E3SMV3_ATMOS_FORCING",
        ),
        (
            "allenai/SamudrACE-E3SMv3",
            "forcing_data/ocean-forcing-1yr.nc",
            b"FAKE_E3SMV3_OCEAN_FORCING",
        ),
    ]
    for repo, rel, content in extra_fme_static_blobs:
        p = hf_mirror / repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)

    for scen in ("1xCO2", "3xCO2"):
        for y in yc.som_years:
            p = hf_mirror / "allenai/ACE2-SOM" / "forcing_data" / scen / f"forcing_{y}.nc"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f"FAKE_SOM_{scen}_{y}".encode("ascii"))

    for y in yc.hiro_native_years:
        if y != 2014:
            p_ace2s = hf_mirror / "allenai/HiRO-ACE" / "forcing_data" / f"forcing_{y}.nc"
            p_ace2s.parent.mkdir(parents=True, exist_ok=True)
            p_ace2s.write_bytes(f"FAKE_ACE2S_FORCING_{y}".encode("ascii"))
        p_v2 = hf_mirror / "allenai/HiRO-ACEv2" / "forcing_data" / f"forcing_{y}.nc"
        p_v2.parent.mkdir(parents=True, exist_ok=True)
        p_v2.write_bytes(f"FAKE_HIRO_V2_FORCING_{y}".encode("ascii"))

    # Upstream SamudrACE-CM4 IC directories have literal `:` in ISO timestamps (`0311-01-01T00:00:00`)
    for upstream_ts, _local_slug in SAMUDRACE_CM4_IC_SLUGS:
        for realm in ("atmosphere", "ocean"):
            p_cm4 = (
                hf_mirror
                / "allenai/SamudrACE-CM4-piControl"
                / "initial_conditions"
                / upstream_ts
                / realm
                / "initial_condition.nc"
            )
            p_cm4.parent.mkdir(parents=True, exist_ok=True)
            p_cm4.write_bytes(f"FAKE_CM4_IC_{upstream_ts}_{realm}".encode("ascii"))

    # DLESyM / ArchesClimate-SSP pinned files (model and dataset repos alike, mirrored by repo id)
    from aimip.emu_catalog import artifacts as _emu_artifacts

    for a in _emu_artifacts():
        if a.get("source_kind") == "github_archive":
            continue
        if a.get("source_kind") == "zenodo":
            p_emu = zenodo_mirror / a["record"] / a["path"]
        else:
            p_emu = hf_mirror / a["repo"] / a["path"]
        p_emu.parent.mkdir(parents=True, exist_ok=True)
        p_emu.write_bytes(f"FAKE_EMU {a['repo']} {a['path']}".encode("ascii"))


def _synthetic_co2_reader(year: int) -> xr.DataArray:
    """Return a synthetic 6-hourly 1D `global_mean_co2` DataArray for *year*."""
    t = np.arange(
        np.datetime64(f"{year:04d}-01-01T00", "ns"),
        np.datetime64(f"{year + 1:04d}-01-01T00", "ns"),
        np.timedelta64(6, "h"),
    ).astype("datetime64[ns]")
    vals = np.linspace(
        3.95e-4 + (year - 2013) * 2.2e-6,
        3.95e-4 + (year - 2012) * 2.2e-6,
        t.size,
        endpoint=False,
        dtype=np.float32,
    )
    return xr.DataArray(vals, dims=("time",), coords={"time": t}, name="global_mean_co2")


def test_site_init_presets_overrides_and_sbatch_rendering(
    playground_env: dict[str, Any],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`aimip site init` presets + `site.json` rendering into `job.sbatch`."""
    pg_dir: pathlib.Path = playground_env["playground_dir"]
    runs_dir: pathlib.Path = playground_env["runs_dir"]
    aimip_home = tmp_path / "aimip_home"
    aimip_home.mkdir()
    monkeypatch.setenv("AIMIP_HOME", str(aimip_home))

    # 1. `generic-slurm` preset with custom partition, empty account, custom gres, and 02:00:00 cap.
    assert (
        main(
            [
                "site",
                "init",
                "--preset",
                "generic-slurm",
                "--partition",
                "a100_short",
                "--account",
                "",
                "--gres",
                "gpu:1",
                "--cpus-per-task",
                "8",
                "--mem",
                "32G",
                "--max-time",
                "02:00:00",
                "--run-root",
                str(runs_dir),
                "--json",
            ]
        )
        == 0
    )
    from conftest import parse_cli_json

    init_res = parse_cli_json(capsys.readouterr().out)
    assert init_res["path"] == str(aimip_home / "site.json")
    assert init_res["site"]["sbatch"] == {
        "partition": "a100_short",
        "account": "",
        "gres": "gpu:1",
        "cpus_per_task": 8,
        "mem": "32G",
        "max_time": "02:00:00",
    }

    # Verify `aimip run` picks up `$AIMIP_HOME/site.json` automatically with NO `--partition` flags!
    out_slurm = runs_dir / "custom_site_slurm"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "50000",  # Long run so walltime hits site.json `max_time` cap (02:00:00)
                "--out",
                str(out_slurm),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()

    sbatch_lines = (out_slurm / "job.sbatch").read_text(encoding="utf-8").splitlines()
    assert sbatch_lines[1] == "# aimip-generated v1"
    assert "#SBATCH --partition=a100_short" in sbatch_lines
    assert "#SBATCH --gres=gpu:1" in sbatch_lines
    assert "#SBATCH --cpus-per-task=8" in sbatch_lines
    assert "#SBATCH --hint=nomultithread" in sbatch_lines
    assert "#SBATCH --mem=32G" in sbatch_lines
    assert "#SBATCH --time=02:00:00" in sbatch_lines
    # Empty account MUST be omitted so the submit gate `#SBATCH` regex passes.
    assert not any(ln.startswith("#SBATCH --account") for ln in sbatch_lines)

    sbatch_guard = re.compile(
        r"^#SBATCH --((job-name|partition|account|gres|cpus-per-task|mem|time|output|error)=[A-Za-z0-9._/:%+-]+"
        r"|hint=nomultithread)$"
    )
    for ln in sbatch_lines:
        if ln.startswith("#SBATCH"):
            assert sbatch_guard.match(ln), f"Invalid #SBATCH directive for wrapper: {ln}"

    # 2. `local` preset (`scheduler == "none"`) omits `#SBATCH` lines while keeping `# aimip-generated v1`.
    assert (
        main(
            [
                "site",
                "init",
                "--preset",
                "local",
                "--run-root",
                str(runs_dir),
                "--overwrite",
            ]
        )
        == 0
    )
    capsys.readouterr()

    out_local = runs_dir / "custom_site_local"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--out",
                str(out_local),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    local_lines = (out_local / "job.sbatch").read_text(encoding="utf-8").splitlines()
    assert local_lines[:2] == ["#!/bin/bash", "# aimip-generated v1"]
    assert not any(ln.startswith("#SBATCH") for ln in local_lines)
    assert [ln for ln in local_lines if ln.strip() and not ln.startswith("#")] == [
        "set -euo pipefail",
        f'"{pg_dir}/bin/aimip" exec-fme "{out_local}"',
    ]

    # 3. Unedited `CHANGE_ME` in `site.json` refuses cleanly unless overridden on CLI.
    (aimip_home / "site.json").write_text(
        json.dumps(
            {
                "name": "unconfigured",
                "scheduler": "slurm",
                "runtime": "apptainer",
                "sbatch": {
                    "partition": "CHANGE_ME",
                    "account": "",
                    "gres": "gpu:1",
                    "cpus_per_task": 8,
                    "mem": "32G",
                    "max_time": "04:00:00",
                },
                "run_root": str(runs_dir),
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--out",
                str(runs_dir / "change_me_fail"),
                "--dry-run",
            ]
        )
        == 2
    )
    assert "CHANGE_ME" in capsys.readouterr().err


    # 4. With `AIMIP_SCHEDULER=none`, even a `CHANGE_ME` SLURM `site.json` does not refuse.
    monkeypatch.setenv("AIMIP_SCHEDULER", "none")
    out_sched_none = runs_dir / "sched_none_override"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--out",
                str(out_sched_none),
                "--dry-run",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (out_sched_none / "job.sbatch").is_file()

    # 5. Built-in `levante` and `generic-slurm` presets match `install/presets/*.json` (`CHANGE_ME` account),
    #    and unconfigured fallback (`load_site_config()` with no `site.json`) refuses rather than emitting a fake account.
    from aimip.site_config import SiteConfig, load_site_config

    monkeypatch.delenv("AIMIP_SCHEDULER", raising=False)
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    lev_json = json.loads((repo_root / "install" / "presets" / "levante.json").read_text(encoding="utf-8"))
    gen_json = json.loads((repo_root / "install" / "presets" / "generic-slurm.json").read_text(encoding="utf-8"))
    assert SiteConfig.preset("levante").account == lev_json["sbatch"]["account"] == "CHANGE_ME"
    assert SiteConfig.preset("generic-slurm").partition == gen_json["sbatch"]["partition"] == "CHANGE_ME"
    assert SiteConfig.preset("generic-slurm").account == gen_json["sbatch"]["account"] == "CHANGE_ME"

    assert main(["site", "init", "--preset", "levante", "--overwrite"]) == 2
    assert "CHANGE_ME" in capsys.readouterr().err

    (aimip_home / "site.json").unlink()
    if (pg_dir / "site.json").exists():
        (pg_dir / "site.json").unlink()
    monkeypatch.delenv("AIMIP_SITE_CONFIG", raising=False)
    assert load_site_config().account == "CHANGE_ME"
    assert (
        main(
            [
                "run",
                "-m",
                "ace2-era5",
                "--start",
                "1979-01-01T00:00:00",
                "--steps",
                "4",
                "--out",
                str(runs_dir / "unconfigured_levante_fallback"),
                "--dry-run",
            ]
        )
        == 2
    )
    assert "CHANGE_ME" in capsys.readouterr().err


def test_registry_data_placeholder_and_aimip_data_resolution(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packaged `registry.default.json` + `${DATA}` expansion when `AIMIP_REGISTRY` is unset."""
    empty_ro_data = tmp_path / "ro_data_without_registry_json"
    empty_ro_data.mkdir()

    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)
    monkeypatch.delenv("AIMIP_PLAYGROUND", raising=False)
    monkeypatch.delenv("AIMIP_HOME", raising=False)
    monkeypatch.setenv("AIMIP_DATA", str(empty_ro_data))

    # 1. With no `data/registry.json` on disk, falls back to packaged `aimip/registry.default.json`
    default_catalog = load_registry()
    assert list(default_catalog.keys()) == list(CANONICAL_MODELS)
    assert (
        default_catalog["ace2-era5"].checkpoint
        == f"{empty_ro_data}/models/ace2-era5/ace2_era5_ckpt.tar"
    )
    assert default_catalog["ace2-shield"].ic_files == (
        f"{empty_ro_data}/models/ace2s-shield-plus/ic.nc",
    )
    assert default_catalog["ace2s-shield-plus"].ic_files == (
        f"{empty_ro_data}/models/ace2s-shield-plus/ic.nc",
    )
    assert (
        default_catalog["hiro-ace"].forcing_sets[1].dir
        == f"{empty_ro_data}/inputs/hiro-ace/amip-oos"
    )

    # 2. Custom `$AIMIP_DATA/registry.json` override still works when present on disk
    data_dir = tmp_path / "portable_data"
    data_dir.mkdir()
    reg_payload = build_default_registry_payload(selected_models=("ace2-era5", "hiro-ace"))
    (data_dir / "registry.json").write_text(json.dumps(reg_payload), encoding="utf-8")
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))

    catalog = load_registry()
    assert set(catalog.keys()) == {"ace2-era5", "hiro-ace"}
    assert catalog["ace2-era5"].checkpoint == f"{data_dir}/models/ace2-era5/ace2_era5_ckpt.tar"


def test_all_derived_builders_are_byte_for_byte_deterministic(
    tmp_path: pathlib.Path,
) -> None:
    """Running every derived builder twice produces identical bytes."""
    yc = SetupYearConfig(
        era5_hist_years=(1994, 2012, 2014),
        shield_hist_years=(1994, 2012, 2014),
        derived_hist_years=(1994, 2012, 2014),
        oos_years=(2015, 2016),
        oos_bridge_year=2014,
        oos_ref_nonleap_year=2014,
        oos_ref_leap_year=2012,
        oos_anchor_t0="2014-12-01",
        oos_anchor_t1="2017-01-01",
        co2_y0=2013,
        co2_y1=2016,
        co2_last_upstream=2015,
        co2_validate_reference_stats=False,
        oos_expected_nonleap_steps=4,
        oos_expected_leap_steps=5,
        som_years=(2032, 2033),
        hiro_native_years=(2014, 2015),
    )
    hf_mirror = tmp_path / "hf_src"
    zen_mirror = tmp_path / "zen_src"
    _create_synthetic_upstream_mirrors(hf_mirror, zen_mirror, yc)

    def _run_all_builders_once(out_root: pathlib.Path) -> dict[str, bytes]:
        co2_p = build_co2_table(
            str(out_root / "co2.nc"),
            y0=yc.co2_y0,
            y1=yc.co2_y1,
            last_upstream=yc.co2_last_upstream,
            year_reader=_synthetic_co2_reader,
            validate_reference_stats=False,
            force=True,
        )
        hiro_ic_p = build_hiro_ic(
            str(hf_mirror / "allenai/HiRO-ACE/initial_conditions/ic_2014.nc"),
            str(out_root / "hiro_ic_1994.nc"),
            force=True,
        )
        hiro_f_list = build_hiro_forcing(
            raw_hiro_forcing_2014_path=str(
                hf_mirror / "allenai/HiRO-ACE/forcing_data/forcing_2014.nc"
            ),
            shield_amip_hist_dir=str(
                hf_mirror / "allenai/ACE2S-SHiELD-plus/forcing_data/AMIP"
            ),
            out_dir=str(out_root / "hiro_hist"),
            years=(1994, 2012, 2014),
            force=True,
        )
        eam_ic_p = build_eamv3_ic(
            str(hf_mirror / "allenai/ACE2-EAMv3/initial_conditions/1971010100.nc"),
            str(out_root / "eam_ic_1994.nc"),
            force=True,
        )
        eam_f_list = build_eamv3_forcing(
            raw_eamv3_1971_path=str(hf_mirror / "allenai/ACE2-EAMv3/forcing_data/1971.nc"),
            era5_amip_hist_dir=str(hf_mirror / "allenai/ACE2-ERA5/forcing_data"),
            out_dir=str(out_root / "eam_hist"),
            years=(1994, 2014),
            force=True,
        )
        shield_oos_list = build_oos_forcing(
            "shieldplus",
            hist_forcing_dir=str(
                hf_mirror / "allenai/ACE2S-SHiELD-plus/forcing_data/AMIP"
            ),
            aimip_monthly_nc_path=str(
                zen_mirror / ZENODO_AIMIP_RECORD / ZENODO_AIMIP_FILENAME
            ),
            co2_nc_path=co2_p,
            out_dir=str(out_root / "shield_oos"),
            years=(2015, 2016),
            bridge_year=2014,
            anchor_t0="2014-12-01",
            anchor_t1="2017-01-01",
            expected_nonleap_steps=4,
            expected_leap_steps=5,
            force=True,
        )
        hiro_oos_list = build_oos_forcing(
            "hiro",
            hist_forcing_dir=str(out_root / "hiro_hist"),
            aimip_monthly_nc_path=str(
                zen_mirror / ZENODO_AIMIP_RECORD / ZENODO_AIMIP_FILENAME
            ),
            co2_nc_path=co2_p,
            out_dir=str(out_root / "hiro_oos"),
            years=(2015, 2016),
            bridge_year=2014,
            anchor_t0="2014-12-01",
            anchor_t1="2017-01-01",
            expected_nonleap_steps=4,
            expected_leap_steps=5,
            force=True,
        )
        all_paths = [
            co2_p,
            hiro_ic_p,
            *hiro_f_list,
            eam_ic_p,
            *eam_f_list,
            *shield_oos_list,
            *hiro_oos_list,
        ]
        return {
            str(pathlib.Path(p).relative_to(out_root)): pathlib.Path(p).read_bytes()
            for p in all_paths
        }

    bytes_run1 = _run_all_builders_once(tmp_path / "build_run1")
    time.sleep(0.05)  # Advance wallclock to prove zero timestamp sensitivity
    bytes_run2 = _run_all_builders_once(tmp_path / "build_run2")

    assert set(bytes_run1.keys()) == set(bytes_run2.keys())
    for rel_key in sorted(bytes_run1.keys()):
        assert bytes_run1[rel_key] == bytes_run2[rel_key], (
            f"Builder output {rel_key} is not byte-identical across runs!"
        )

    # Verify NaN-free OOS output and absence of dynamic timestamp attributes
    with xr.open_dataset(
        str(tmp_path / "build_run1" / "shield_oos" / "forcing_2016.nc"),
        engine="netcdf4",
    ) as ds_oos:
        assert np.isfinite(ds_oos["surface_temperature"].values).all()
        assert np.isfinite(ds_oos["sea_ice_fraction"].values).all()
        assert "date_created" not in ds_oos.attrs
        assert "history" not in ds_oos.attrs


def test_setup_end_to_end_file_mirrors_write_lock_and_verify_pass_fail(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end `setup` via `file://` mirrors + lock pass/fail."""
    yc = SetupYearConfig(
        era5_hist_years=(1994, 2012, 2014),
        shield_hist_years=(1994, 2012, 2014),
        derived_hist_years=(1994, 2012, 2014),
        oos_years=(2015, 2016),
        oos_bridge_year=2014,
        oos_ref_nonleap_year=2014,
        oos_ref_leap_year=2012,
        oos_anchor_t0="2014-12-01",
        oos_anchor_t1="2017-01-01",
        co2_y0=2013,
        co2_y1=2016,
        co2_last_upstream=2015,
        co2_validate_reference_stats=False,
        oos_expected_nonleap_steps=4,
        oos_expected_leap_steps=5,
        som_years=(2032, 2033),
        hiro_native_years=(2014, 2015),
    )
    hf_mirror = tmp_path / "hf_mirror"
    zen_mirror = tmp_path / "zen_mirror"
    _create_synthetic_upstream_mirrors(hf_mirror, zen_mirror, yc)

    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")
    monkeypatch.setenv("AIMIP_ZENODO_MIRROR", f"file://{zen_mirror}")

    # Step 1: Bootstrap initial unpinned lock, build staging tree, and generate golden `aimip.lock` via `--write-lock`.
    bootstrap_dest = tmp_path / "bootstrap_data"
    bootstrap_lock = tmp_path / "bootstrap.lock"
    cat = build_artifact_catalog(years=yc)
    bootstrap_lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {
                    rel: {
                        "size": None,
                        "sha256": SHA256_TBD,
                        "source": spec.default_source_dict(),
                    }
                    for rel, spec in cat.items()
                },
            }
        ),
        encoding="utf-8",
    )

    def _synthetic_arches_fetcher(dest: str, **_kw: object) -> dict[str, object]:
        code_dir = pathlib.Path(dest) / "code" / "ArchesClimate-SSP"
        code_dir.mkdir(parents=True, exist_ok=True)
        return {"code_dir": str(code_dir), "downloaded": False}

    def _synthetic_camulator_fetcher(dest: str, **_kw: object) -> dict[str, object]:
        code_dir = pathlib.Path(dest) / "code" / "miles-credit-camulator"
        code_dir.mkdir(parents=True, exist_ok=True)
        return {"code_dir": str(code_dir), "downloaded": False}

    def _synthetic_sdyff_fetcher(dest: str, **_kw: object) -> dict[str, object]:
        code_dir = pathlib.Path(dest) / "code" / "spherical-dyffusion"
        code_dir.mkdir(parents=True, exist_ok=True)
        return {"code_dir": str(code_dir), "downloaded": False}

    def _synthetic_md15_fetcher(dest: str, **_kw: object) -> dict[str, object]:
        from aimip.md15 import MD15_CONFIG_REL_PATH, MD15_STATS_REL_PATH, MD15_WEIGHTS_REL_PATH

        dest_p = pathlib.Path(dest)
        code_dir = dest_p / "code" / "monthly-diffusion"
        code_dir.mkdir(parents=True, exist_ok=True)
        for rel_m in (MD15_CONFIG_REL_PATH, MD15_WEIGHTS_REL_PATH, MD15_STATS_REL_PATH):
            pm = dest_p / rel_m
            pm.parent.mkdir(parents=True, exist_ok=True)
            if not pm.is_file():
                pm.write_bytes(f"FAKE_MD15 {rel_m}".encode("ascii"))
        return {"code_dir": str(code_dir), "downloaded": False}

    run_setup(
        models="all",
        dest_dir=str(bootstrap_dest),
        lock_path=str(bootstrap_lock),
        strict_unpinned=False,
        years=yc,
        arches_code_fetcher=_synthetic_arches_fetcher,
        md15_code_fetcher=_synthetic_md15_fetcher,
        camulator_code_fetcher=_synthetic_camulator_fetcher,
        sdyff_code_fetcher=_synthetic_sdyff_fetcher,
        co2_year_reader=_synthetic_co2_reader,
    )

    golden_lock_path = tmp_path / "golden_aimip.lock"
    wl_res = run_setup(
        models="all",
        dest_dir=str(bootstrap_dest),
        lock_path=str(golden_lock_path),
        write_lock=True,
        years=yc,
    )
    assert wl_res["mode"] == "write-lock"
    assert wl_res["hashed_files"] == len(cat)

    golden_lock = load_lock_file(str(golden_lock_path))
    for rel, entry in golden_lock["files"].items():
        assert isinstance(entry["size"], int) and entry["size"] > 0, rel
        assert entry["sha256"] != SHA256_TBD and len(entry["sha256"]) == 64, rel

    # Step 2: Fresh install from zero into `fresh_dest` verified strictly against `golden_aimip.lock`!
    fresh_dest = tmp_path / "fresh_data"
    setup_res = run_setup(
        models="all",
        dest_dir=str(fresh_dest),
        lock_path=str(golden_lock_path),
        years=yc,
        arches_code_fetcher=_synthetic_arches_fetcher,
        md15_code_fetcher=_synthetic_md15_fetcher,
        camulator_code_fetcher=_synthetic_camulator_fetcher,
        sdyff_code_fetcher=_synthetic_sdyff_fetcher,
        co2_year_reader=_synthetic_co2_reader,
    )
    assert setup_res["mode"] == "full"
    assert setup_res["verified_files"] == len(cat)
    assert pathlib.Path(setup_res["registry_path"]).is_file()

    # Generated `<dest>/registry.json` loads all canonical models cleanly with `${DATA}` expansion!
    loaded_reg = load_registry(data_dir=str(fresh_dest))
    assert tuple(loaded_reg.keys()) == CANONICAL_MODELS

    # Step 3: Idempotent re-run skips all already-verified downloads & builds!
    rerun_res = run_setup(
        models="all",
        dest_dir=str(fresh_dest),
        lock_path=str(golden_lock_path),
        years=yc,
        arches_code_fetcher=_synthetic_arches_fetcher,
        md15_code_fetcher=_synthetic_md15_fetcher,
        camulator_code_fetcher=_synthetic_camulator_fetcher,
        sdyff_code_fetcher=_synthetic_sdyff_fetcher,
        co2_year_reader=_synthetic_co2_reader,
    )
    assert rerun_res["downloaded_files"] == 0
    assert rerun_res["built_files"] == 0
    assert rerun_res["skipped_existing_files"] > 0

    # Step 4: `verify_only=True` passes on intact tree.
    ver_res = run_setup(
        models="all",
        dest_dir=str(fresh_dest),
        lock_path=str(golden_lock_path),
        verify_only=True,
        years=yc,
        arches_code_fetcher=_synthetic_arches_fetcher,
        md15_code_fetcher=_synthetic_md15_fetcher,
        camulator_code_fetcher=_synthetic_camulator_fetcher,
        sdyff_code_fetcher=_synthetic_sdyff_fetcher,
    )
    assert ver_res["verified_files"] == len(cat)

    # Step 5: Flip 1 byte in a derived output -> `verify_only` fails naming the exact file!
    target_corrupt_rel = "inputs/hiro-ace/amip-oos/forcing_2015.nc"
    corrupt_abs = fresh_dest / target_corrupt_rel
    orig_bytes = corrupt_abs.read_bytes()
    corrupt_abs.write_bytes(orig_bytes[:-1] + bytes([orig_bytes[-1] ^ 0xFF]))
    with pytest.raises(ValidationError, match=re.escape(target_corrupt_rel)):
        run_setup(
            models="all",
            dest_dir=str(fresh_dest),
            lock_path=str(golden_lock_path),
            verify_only=True,
            years=yc,
            arches_code_fetcher=_synthetic_arches_fetcher,
            md15_code_fetcher=_synthetic_md15_fetcher,
            camulator_code_fetcher=_synthetic_camulator_fetcher,
            sdyff_code_fetcher=_synthetic_sdyff_fetcher,
        )
    corrupt_abs.write_bytes(orig_bytes)

    # Step 6: Corrupt Zenodo MD5 in lock -> `verify_only` fails naming the Zenodo file!
    bad_md5_lock = json.loads(golden_lock_path.read_text(encoding="utf-8"))
    bad_md5_lock["files"][ZENODO_AIMIP_REL_PATH]["source"]["md5"] = "0" * 32
    bad_md5_lock_path = tmp_path / "bad_md5.lock"
    bad_md5_lock_path.write_text(json.dumps(bad_md5_lock), encoding="utf-8")
    with pytest.raises(ValidationError, match=re.escape(ZENODO_AIMIP_REL_PATH)):
        run_setup(
            models="all",
            dest_dir=str(fresh_dest),
            lock_path=str(bad_md5_lock_path),
            verify_only=True,
            years=yc,
            arches_code_fetcher=_synthetic_arches_fetcher,
            md15_code_fetcher=_synthetic_md15_fetcher,
            camulator_code_fetcher=_synthetic_camulator_fetcher,
            sdyff_code_fetcher=_synthetic_sdyff_fetcher,
        )

    # Step 7: Wrong derived hash in lock -> fresh `setup` rebuilds then raises naming the derived file!
    bad_derived_rel = "inputs/ace2-eamv3/ic/ic_1994.nc"
    bad_derived_lock = json.loads(golden_lock_path.read_text(encoding="utf-8"))
    bad_derived_lock["files"][bad_derived_rel]["sha256"] = "f" * 64
    bad_derived_lock_path = tmp_path / "bad_derived.lock"
    bad_derived_lock_path.write_text(json.dumps(bad_derived_lock), encoding="utf-8")
    with pytest.raises(ValidationError, match=re.escape(bad_derived_rel)):
        run_setup(
            models="ace2-eamv3",
            dest_dir=str(tmp_path / "fresh_bad_derived"),
            lock_path=str(bad_derived_lock_path),
            years=yc,
        )


def test_cli_setup_ace2_era5_only_download_write_lock_and_verify(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test CLI `aimip setup --models ace2-era5` with all 39 production catalog entries via `file://` mirror."""
    hf_mirror = tmp_path / "cli_hf_mirror"
    repo_dir = hf_mirror / "allenai/ACE2-ERA5"
    (repo_dir / "initial_conditions").mkdir(parents=True)
    (repo_dir / "forcing_data").mkdir(parents=True)

    (repo_dir / "ace2_era5_ckpt.tar").write_bytes(b"CLI_CKPT")
    (repo_dir / "initial_conditions" / "ic_1979.nc").write_bytes(b"CLI_IC_1979")
    (repo_dir / "initial_conditions" / "ic_2001.nc").write_bytes(b"CLI_IC_2001")
    for y in range(1979, 2015):
        (repo_dir / "forcing_data" / f"forcing_{y}.nc").write_bytes(
            f"CLI_FORCING_{y}".encode("ascii")
        )

    # Seed tree in `stage_dir` from mirror, then run `aimip setup --write-lock` via CLI!
    stage_dir = tmp_path / "stage_era5"
    (stage_dir / "models" / "ace2-era5").mkdir(parents=True)
    (stage_dir / "inputs" / "ace2-era5" / "ic").mkdir(parents=True)
    (stage_dir / "inputs" / "ace2-era5" / "amip-hist").mkdir(parents=True)
    shutil.copyfile(
        repo_dir / "ace2_era5_ckpt.tar",
        stage_dir / "models" / "ace2-era5" / "ace2_era5_ckpt.tar",
    )
    for ic_n in ("ic_1979.nc", "ic_2001.nc"):
        shutil.copyfile(
            repo_dir / "initial_conditions" / ic_n,
            stage_dir / "inputs" / "ace2-era5" / "ic" / ic_n,
        )
    for y in range(1979, 2015):
        shutil.copyfile(
            repo_dir / "forcing_data" / f"forcing_{y}.nc",
            stage_dir / "inputs" / "ace2-era5" / "amip-hist" / f"forcing_{y}.nc",
        )

    custom_lock = tmp_path / "era5_cli.lock"
    assert (
        main(
            [
                "setup",
                "--models",
                "ace2-era5",
                "--dest",
                str(stage_dir),
                "--lock",
                str(custom_lock),
                "--write-lock",
                "--json",
            ]
        )
        == 0
    )
    from conftest import parse_cli_json

    wl_out = parse_cli_json(capsys.readouterr().out)
    assert wl_out["hashed_files"] == 39

    # Install into a brand-new `$AIMIP_DATA` directory via `aimip setup --models ace2-era5`!
    dest_data = tmp_path / "installed_era5_data"
    monkeypatch.setenv("AIMIP_DATA", str(dest_data))
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")

    assert (
        main(
            [
                "setup",
                "--models",
                "ace2-era5",
                "--lock",
                str(custom_lock),
                "--json",
            ]
        )
        == 0
    )
    install_out = parse_cli_json(capsys.readouterr().out)
    assert install_out["downloaded_files"] == 39
    assert install_out["verified_files"] == 39

    # `--verify-only` passes via CLI
    assert (
        main(
            [
                "setup",
                "--models",
                "ace2-era5",
                "--lock",
                str(custom_lock),
                "--verify-only",
            ]
        )
        == 0
    )
    capsys.readouterr()

    # Corrupt `forcing_1999.nc` -> CLI returns exit code 2 and names `inputs/ace2-era5/amip-hist/forcing_1999.nc`
    (dest_data / "inputs" / "ace2-era5" / "amip-hist" / "forcing_1999.nc").write_bytes(
        b"TAMPERED"
    )
    assert (
        main(
            [
                "setup",
                "--models",
                "ace2-era5",
                "--lock",
                str(custom_lock),
                "--verify-only",
            ]
        )
        == 2
    )
    err_msg = capsys.readouterr().err
    assert "inputs/ace2-era5/amip-hist/forcing_1999.nc" in err_msg


def test_shipped_aimip_lock_covers_all_models_with_pinned_revisions_and_output_shas(
    tmp_path: pathlib.Path,
) -> None:
    """Verify the shipped `aimip.lock` covers all canonical models with 64-hex sha256 and size pins across all entries."""
    import aimip
    from aimip.emu_catalog import (
        artifacts as emu_artifacts,
        on_demand_artifacts as emu_on_demand_artifacts,
    )
    from aimip.setup_data import CO2_TABLE_REL_PATH, MD5_TBD, verify_file_against_lock

    pkg_dir = pathlib.Path(aimip.__file__).resolve().parent
    lock_file = pkg_dir / "aimip.lock"
    raw_lock_text = lock_file.read_text(encoding="utf-8")
    assert SHA256_TBD not in raw_lock_text
    assert MD5_TBD not in raw_lock_text
    assert REVISION_TBD not in raw_lock_text

    shipped_lock = load_lock_file(str(lock_file))
    files = shipped_lock["files"]
    catalog = build_artifact_catalog()
    n_emu = len(emu_artifacts())
    on_demand_rels = {a["rel_path"] for a in emu_on_demand_artifacts()}
    assert n_emu >= 63 and len(on_demand_rels) >= 68

    extra_aux = {
        "models/hiro-acev2/HiROv2.ckpt",
        "models/samudrace-e3smv3/SamudrACE-E3SMv3-atmosphere.tar",
        "models/samudrace-e3smv3/SamudrACE-E3SMv3-ocean.tar",
        *on_demand_rels,
    }
    assert set(files.keys()) == set(catalog.keys()) | extra_aux
    assert len(catalog) == 212 + n_emu  # 212 fme + default emu_sources.json artifacts
    assert len(files) == 215 + n_emu + len(on_demand_rels)    # + 3 companion checkpoints + on-demand emu assets

    hex64_re = re.compile(r"^[0-9a-f]{64}$")
    hex40_re = re.compile(r"^[0-9a-f]{40}$")
    hex32_re = re.compile(r"^[0-9a-f]{32}$")

    output_count = 0
    raw_count = 0
    for rel_path, spec in catalog.items():
        entry = files[rel_path]
        src = entry["source"]
        assert not entry.get("pin_after_first_fetch"), f"Unpinned pin_after_first_fetch on {rel_path}"
        assert hex64_re.match(entry["sha256"]), f"Unpinned sha256 on {rel_path}"
        if spec.source_kind == "hf":
            assert src["revision"] != REVISION_TBD
            assert hex40_re.match(src["revision"]), f"Unpinned HF revision on {rel_path}"
            assert isinstance(entry["size"], int) and entry["size"] > 0
        elif spec.source_kind == "zenodo":
            assert isinstance(entry["size"], int) and entry["size"] > 0
            if rel_path == ZENODO_AIMIP_REL_PATH:
                assert entry["size"] == 4622266486
            assert hex32_re.match(src["md5"]), f"Unpinned Zenodo md5 on {rel_path}"
        elif spec.source_kind == "github_archive":
            assert hex40_re.match(src["revision"]), f"Unpinned archive revision on {rel_path}"
            assert hex64_re.match(src["archive_sha256"]), f"Unpinned archive sha256 on {rel_path}"
            assert isinstance(entry["size"], int) and entry["size"] > 0
        elif spec.source_kind == "derived":
            for in_ref, in_sha in src["inputs"].items():
                assert hex64_re.match(in_sha), f"Unpinned derived input sha on {rel_path} -> {in_ref}"
        if spec.role == "output":
            output_count += 1
        elif spec.role == "raw":
            raw_count += 1
            assert isinstance(entry["size"], int) and entry["size"] > 0

    for aux_rel in extra_aux:
        aux_entry = files[aux_rel]
        assert hex64_re.match(aux_entry["sha256"]), aux_rel
        assert isinstance(aux_entry["size"], int) and aux_entry["size"] > 0
        assert hex40_re.match(aux_entry["source"]["revision"]), aux_rel

    assert output_count == 206 + n_emu   # 206 fme + emu_sources.json outputs
    assert raw_count == 6
    assert files[ZENODO_AIMIP_REL_PATH]["sha256"] == (
        "a52caf3722bc85a58075118fa06f0c6cce5ace279458edda0a2c9d68ec179fb7"
    )
    assert files[ZENODO_AIMIP_REL_PATH]["source"]["md5"] == "11cf22013db1e1d3494e67216c8de268"
    assert files[CO2_TABLE_REL_PATH]["size"] == 815845
    assert files[CO2_TABLE_REL_PATH]["sha256"] == (
        "e2c65b9edd216604a7ffada29fb4f3aa6cb24b2228f3428c1b0a0c29018b2896"
    )

    # Strict verification must also reject unpinned `role="raw"` entries (`SHA256_TBD` or `MD5_TBD`)
    dummy_raw = tmp_path / "raw" / "co2_global_mean.nc"
    dummy_raw.parent.mkdir(parents=True, exist_ok=True)
    dummy_raw.write_bytes(b"NONEMPTY")
    with pytest.raises(ValidationError, match="unpinned checksum"):
        verify_file_against_lock(
            str(tmp_path),
            "raw/co2_global_mean.nc",
            {"size": None, "sha256": SHA256_TBD, "source": {"kind": "derived"}},
            role="raw",
            strict_unpinned=True,
        )


def test_hf_hub_download_delegation_and_zenodo_http_range_resume(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `hf_hub_download` revision/HF_HOME delegation and Zenodo HTTP `Range` resume."""
    import io
    import sys
    import types
    import urllib.request

    monkeypatch.delenv("AIMIP_HF_MIRROR", raising=False)
    monkeypatch.delenv("AIMIP_ZENODO_MIRROR", raising=False)

    # 1. Monkeypatched `huggingface_hub.hf_hub_download`
    hf_calls: list[dict[str, Any]] = []
    cached_src = tmp_path / "hf_blob.tar"
    cached_src.write_bytes(b"FAKE_CKPT_PAYLOAD_BYTES")

    def _fake_hf_hub_download(
        *,
        repo_id: str,
        filename: str,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> str:
        hf_calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
                "cache_dir": cache_dir,
            }
        )
        return str(cached_src)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=_fake_hf_hub_download),
    )

    hf_home = tmp_path / "data" / ".hf"
    hf_dst = tmp_path / "data" / "models" / "ace2-era5" / "ace2_era5_ckpt.tar"
    pinned_rev = "116ef6196452e7e5aa2b01340eb745de87451b13"
    download_hf_artifact(
        repo="allenai/ACE2-ERA5",
        path_in_repo="ace2_era5_ckpt.tar",
        revision=pinned_rev,
        dst_path=str(hf_dst),
        hf_home=str(hf_home),
    )
    assert hf_dst.read_bytes() == b"FAKE_CKPT_PAYLOAD_BYTES"
    assert hf_calls == [
        {
            "repo_id": "allenai/ACE2-ERA5",
            "filename": "ace2_era5_ckpt.tar",
            "revision": pinned_rev,
            "cache_dir": str(hf_home / "hub"),
        }
    ]

    # 2. Zenodo HTTP `Range` resume (`206 Partial Content`) + MD5 verification
    full_payload = b"HEADER_ZENODO_AIMIP_MONTHLY_PAYLOAD_END"
    zen_dst = tmp_path / "data" / ZENODO_AIMIP_REL_PATH
    zen_dst.parent.mkdir(parents=True, exist_ok=True)
    part_file = pathlib.Path(f"{zen_dst}.part")
    part_file.write_bytes(full_payload[:6])  # partial prefix ("HEADER")

    seen_headers: list[dict[str, str]] = []

    class _Fake206Response(io.BytesIO):
        status = 206

        def __enter__(self) -> "_Fake206Response":
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def _fake_urlopen(req: urllib.request.Request, timeout: float = 120.0) -> _Fake206Response:
        seen_headers.append(dict(req.header_items()))
        return _Fake206Response(full_payload[6:])

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    full_tmp = tmp_path / "expected_zen.bin"
    full_tmp.write_bytes(full_payload)
    expected_md5 = md5_file(str(full_tmp))

    download_zenodo_artifact(
        record=ZENODO_AIMIP_RECORD,
        filename=ZENODO_AIMIP_FILENAME,
        url="https://zenodo.org/records/17065758/files/test.nc?download=1",
        dst_path=str(zen_dst),
        expected_size=len(full_payload),
        expected_md5=expected_md5,
    )
    assert zen_dst.read_bytes() == full_payload
    assert not part_file.exists()
    assert seen_headers[0].get("Range") == "bytes=6-"

    # Wrong MD5 must raise ValidationError
    with pytest.raises(ValidationError, match="MD5 mismatch"):
        download_zenodo_artifact(
            record=ZENODO_AIMIP_RECORD,
            filename=ZENODO_AIMIP_FILENAME,
            url="https://zenodo.org/records/17065758/files/test.nc?download=1",
            dst_path=str(tmp_path / "data" / "bad_md5.nc"),
            expected_size=len(full_payload[6:]),
            expected_md5="0" * 32,
        )


def test_hf_hub_download_relative_symlink_into_blobs_stages_real_file(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`hf_hub_download` returns a relative symlink `../../blobs/<sha>`.

    Linux `linkat` without `AT_SYMLINK_FOLLOW` hard-links the symlink itself if `realpath` is omitted,
    leaving a dangling link at `<dest>/models/...`. Verify staged files are regular files (never symlinks)
    for both `download_hf_artifact` and `builders._link_or_copy_atomic`, and dangling links raise cleanly.
    """
    import os
    import sys
    import types

    from aimip.builders import _link_or_copy_atomic

    monkeypatch.delenv("AIMIP_HF_MIRROR", raising=False)

    hf_home = tmp_path / "data" / ".hf"
    repo_cache = hf_home / "hub" / "models--allenai--ACE2-ERA5"
    blobs_dir = repo_cache / "blobs"
    pinned_rev = "f25123c52d7faa8f3533816a311cf3353ee83daa"
    snap_dir = repo_cache / "snapshots" / pinned_rev
    (snap_dir / "initial_conditions").mkdir(parents=True)
    blobs_dir.mkdir(parents=True)

    # Create real blob + relative symlink `../../blobs/<sha>` exactly as `huggingface_hub` does
    blob_ckpt = blobs_dir / "0e2522df5a8db087f18bcebea6e88277cea7c3b71892b868c427aca27673edc2"
    blob_ckpt.write_bytes(b"REAL_HF_BLOB_CKPT_BYTES")
    snap_ckpt = snap_dir / "ace2_era5_ckpt.tar"
    os.symlink(f"../../blobs/{blob_ckpt.name}", snap_ckpt)
    assert snap_ckpt.is_symlink()

    # Nested path (`../../../blobs/<sha>`)
    blob_ic = blobs_dir / "a4e8c13f281bcefd759a18fd5aac21b7af3524fa19ed9b5b53e22f32598e8793"
    blob_ic.write_bytes(b"REAL_HF_BLOB_IC_1979_BYTES")
    snap_ic = snap_dir / "initial_conditions" / "ic_1979.nc"
    os.symlink(f"../../../blobs/{blob_ic.name}", snap_ic)
    assert snap_ic.is_symlink()

    def _fake_hf_hub_download_symlink(
        *,
        repo_id: str,
        filename: str,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> str:
        _ = (repo_id, revision, cache_dir)
        return str(snap_dir / filename)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=_fake_hf_hub_download_symlink),
    )

    dst_ckpt = tmp_path / "data" / "models" / "ace2-era5" / "ace2_era5_ckpt.tar"
    download_hf_artifact(
        repo="allenai/ACE2-ERA5",
        path_in_repo="ace2_era5_ckpt.tar",
        revision=pinned_rev,
        dst_path=str(dst_ckpt),
        hf_home=str(hf_home),
    )
    assert not dst_ckpt.is_symlink(), "Staged checkpoint must be a regular file, not a symlink"
    assert dst_ckpt.is_file()
    assert dst_ckpt.read_bytes() == b"REAL_HF_BLOB_CKPT_BYTES"
    assert os.stat(dst_ckpt).st_ino == os.stat(blob_ckpt).st_ino

    dst_ic = tmp_path / "data" / "inputs" / "ace2-era5" / "ic" / "ic_1979.nc"
    download_hf_artifact(
        repo="allenai/ACE2-ERA5",
        path_in_repo="initial_conditions/ic_1979.nc",
        revision=pinned_rev,
        dst_path=str(dst_ic),
        hf_home=str(hf_home),
    )
    assert not dst_ic.is_symlink(), "Staged IC must be a regular file, not a symlink"
    assert dst_ic.is_file()
    assert dst_ic.read_bytes() == b"REAL_HF_BLOB_IC_1979_BYTES"

    # Also verify `builders._link_or_copy_atomic` on a relative symlink resolves to the real file
    bridge_dst = tmp_path / "data" / "inputs" / "shield" / "amip-oos" / "forcing_2014.nc"
    _link_or_copy_atomic(str(snap_ic), str(bridge_dst))
    assert not bridge_dst.is_symlink()
    assert bridge_dst.is_file()
    assert bridge_dst.read_bytes() == b"REAL_HF_BLOB_IC_1979_BYTES"

    # Dangling relative symlink must fail fast with `ConfigError`
    dangling_snap = snap_dir / "dangling.tar"
    os.symlink("../../blobs/nonexistent_blob_sha", dangling_snap)
    with pytest.raises(ConfigError, match="not a regular file"):
        download_hf_artifact(
            repo="allenai/ACE2-ERA5",
            path_in_repo="dangling.tar",
            revision=pinned_rev,
            dst_path=str(tmp_path / "data" / "models" / "ace2-era5" / "dangling.tar"),
            hf_home=str(hf_home),
        )


def test_oos_fill_nearest_preserves_nonperiodic_seam_property() -> None:
    """Verify `_fill_nearest` uses a non-periodic 2D distance transform across the longitude seam.

    For cell `(row=1, col=0)` with NaN at `col=0,1`, valid `283.0` at `col=2` (distance 2 in-domain),
    and valid `285.0` at `col=7` (distance 1 across a 0/360 wrap), the non-periodic
    `distance_transform_edt` picks `col=2` (`283.0`), NOT `col=7` (`285.0`).
    """
    from aimip.builders import _fill_nearest

    g = np.full((3, 8), 290.0, dtype=np.float64)
    g[1, :] = np.nan
    g[0, :] = np.nan
    g[2, :] = np.nan
    g[1, 2] = 283.0
    g[1, 7] = 285.0

    filled = _fill_nearest(g)
    assert np.isclose(filled[1, 0], 283.0)
    assert np.isclose(filled[1, 6], 285.0)


def test_builder_step_guard_hiro_1994_mask_anchor_and_strict_scipy(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify OOS full-year step assertion, HiRO 1994 mask anchor, and required scipy dependency."""
    import sys

    from aimip.builders import _nearest_valid_indices_2d

    yc = SetupYearConfig(
        era5_hist_years=(1994, 2012, 2014),
        shield_hist_years=(1994, 2012, 2014),
        derived_hist_years=(1994, 2012, 2014),
        oos_years=(2015, 2016),
        oos_bridge_year=2014,
        oos_ref_nonleap_year=2014,
        oos_ref_leap_year=2012,
        oos_anchor_t0="2014-12-01",
        oos_anchor_t1="2017-01-01",
        co2_y0=2013,
        co2_y1=2016,
        co2_last_upstream=2015,
        co2_validate_reference_stats=False,
        som_years=(2032, 2033),
        hiro_native_years=(2014, 2015),
    )
    hf_mirror = tmp_path / "hf_guard"
    zen_mirror = tmp_path / "zen_guard"
    _create_synthetic_upstream_mirrors(hf_mirror, zen_mirror, yc)
    co2_p = build_co2_table(
        str(tmp_path / "co2_guard.nc"),
        y0=2013,
        y1=2016,
        last_upstream=2015,
        year_reader=_synthetic_co2_reader,
        validate_reference_stats=False,
        force=True,
    )

    # 1. Default `expected_nonleap_steps=1460` rejects truncated 4-step reference forcing.
    with pytest.raises(ValidationError, match="has 4 timesteps, expected 1460"):
        build_oos_forcing(
            "shieldplus",
            hist_forcing_dir=str(hf_mirror / "allenai/ACE2S-SHiELD-plus/forcing_data/AMIP"),
            aimip_monthly_nc_path=str(zen_mirror / ZENODO_AIMIP_RECORD / ZENODO_AIMIP_FILENAME),
            co2_nc_path=co2_p,
            out_dir=str(tmp_path / "oos_guard_fail"),
            years=(2015,),
            bridge_year=None,
            anchor_t0="2014-12-01",
            anchor_t1="2017-01-01",
            force=True,
        )

    # 2. Partial `years=[2014]` rebuild in `build_hiro_forcing` still anchors coastal SST mask on `forcing_1994.nc`.
    full_dir = tmp_path / "hiro_full"
    partial_dir = tmp_path / "hiro_partial_2014"
    build_hiro_forcing(
        raw_hiro_forcing_2014_path=str(hf_mirror / "allenai/HiRO-ACE/forcing_data/forcing_2014.nc"),
        shield_amip_hist_dir=str(hf_mirror / "allenai/ACE2S-SHiELD-plus/forcing_data/AMIP"),
        out_dir=str(full_dir),
        years=(1994, 2014),
        mask_reference_year=1994,
        force=True,
    )
    build_hiro_forcing(
        raw_hiro_forcing_2014_path=str(hf_mirror / "allenai/HiRO-ACE/forcing_data/forcing_2014.nc"),
        shield_amip_hist_dir=str(hf_mirror / "allenai/ACE2S-SHiELD-plus/forcing_data/AMIP"),
        out_dir=str(partial_dir),
        years=(2014,),
        mask_reference_year=1994,
        force=True,
    )
    assert (full_dir / "forcing_2014.nc").read_bytes() == (partial_dir / "forcing_2014.nc").read_bytes()

    # 3. Blocking `scipy.ndimage` raises `ConfigError` rather than falling back to pure NumPy.
    monkeypatch.setitem(sys.modules, "scipy.ndimage", None)
    with pytest.raises(ConfigError, match=r"distance_transform_edt.*is required"):
        _nearest_valid_indices_2d(np.array([[True, False], [True, True]]))


def test_setup_persists_verified_models_on_derived_mismatch_and_opt_in_accept_derived(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verified models persist in `<dest>/registry.json`; `--accept-derived` is opt-in for derived outputs only."""
    yc = SetupYearConfig(
        era5_hist_years=(1994, 2012, 2014),
        shield_hist_years=(1994, 2012, 2014),
        derived_hist_years=(1994, 2012, 2014),
        oos_years=(2015, 2016),
        oos_bridge_year=2014,
        oos_ref_nonleap_year=2014,
        oos_ref_leap_year=2012,
        oos_anchor_t0="2014-12-01",
        oos_anchor_t1="2017-01-01",
        co2_y0=2013,
        co2_y1=2016,
        co2_last_upstream=2015,
        co2_validate_reference_stats=False,
        oos_expected_nonleap_steps=4,
        oos_expected_leap_steps=5,
        som_years=(2032, 2033),
        hiro_native_years=(2014, 2015),
    )
    hf_mirror = tmp_path / "hf_ovr"
    zen_mirror = tmp_path / "zen_ovr"
    _create_synthetic_upstream_mirrors(hf_mirror, zen_mirror, yc)
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")
    monkeypatch.setenv("AIMIP_ZENODO_MIRROR", f"file://{zen_mirror}")

    cat = build_artifact_catalog(years=yc)
    boot_dest = tmp_path / "boot_ovr"
    boot_lock = tmp_path / "boot_ovr.lock"
    boot_lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {
                    rel: {"size": None, "sha256": SHA256_TBD, "source": spec.default_source_dict()}
                    for rel, spec in cat.items()
                },
            }
        ),
        encoding="utf-8",
    )
    run_setup(
        models=("ace2-era5", "ace2-eamv3"),
        dest_dir=str(boot_dest),
        lock_path=str(boot_lock),
        strict_unpinned=False,
        years=yc,
    )
    golden_lock_p = tmp_path / "golden_ovr.lock"
    run_setup(
        models=("ace2-era5", "ace2-eamv3"),
        dest_dir=str(boot_dest),
        lock_path=str(golden_lock_p),
        write_lock=True,
        years=yc,
    )

    # Alter derived hash for `inputs/ace2-eamv3/ic/ic_1994.nc` while `ace2-era5` hashes remain exact.
    derived_mismatch_lock = json.loads(golden_lock_p.read_text(encoding="utf-8"))
    derived_mismatch_lock["files"]["inputs/ace2-eamv3/ic/ic_1994.nc"]["sha256"] = "a" * 64
    mismatch_lock_p = tmp_path / "mismatch_ovr.lock"
    mismatch_lock_p.write_text(json.dumps(derived_mismatch_lock), encoding="utf-8")

    # 1. Default strict verification: raises `ValidationError` naming `ace2-eamv3`,
    # BUT `<dest>/registry.json` is written with verified `ace2-era5`!
    strict_dest = tmp_path / "strict_dest_ovr"
    with pytest.raises(ValidationError) as exc_info:
        run_setup(
            models=("ace2-era5", "ace2-eamv3"),
            dest_dir=str(strict_dest),
            lock_path=str(mismatch_lock_p),
            years=yc,
        )
    err_text = str(exc_info.value)
    assert "ace2-eamv3" in err_text
    assert "inputs/ace2-eamv3/ic/ic_1994.nc" in err_text
    assert "Verified models (ace2-era5)" in err_text

    partial_reg = load_registry(data_dir=str(strict_dest))
    assert list(partial_reg.keys()) == ["ace2-era5"]

    # 2. Opt-in `accept_derived=True`: accepts derived mismatch, writes `<dest>/derived-overrides.json`,
    # and registers BOTH `ace2-era5` and `ace2-eamv3`!
    optin_dest = tmp_path / "optin_dest_ovr"
    optin_res = run_setup(
        models=("ace2-era5", "ace2-eamv3"),
        dest_dir=str(optin_dest),
        lock_path=str(mismatch_lock_p),
        accept_derived=True,
        years=yc,
    )
    assert optin_res["verified_models"] == ["ace2-era5", "ace2-eamv3"]
    assert len(optin_res["derived_overrides"]) == 1
    ovr_file = pathlib.Path(optin_res["derived_overrides_path"])
    assert ovr_file.is_file()
    ovr_doc = json.loads(ovr_file.read_text(encoding="utf-8"))
    assert ovr_doc["overrides"][0]["rel_path"] == "inputs/ace2-eamv3/ic/ic_1994.nc"
    assert ovr_doc["overrides"][0]["expected_sha256"] == "a" * 64
    assert list(load_registry(data_dir=str(optin_dest)).keys()) == ["ace2-era5", "ace2-eamv3"]

    # 3. Downloaded HF checkpoint mismatch MUST STILL fail even under `accept_derived=True`!
    bad_hf_lock = json.loads(golden_lock_p.read_text(encoding="utf-8"))
    bad_hf_lock["files"]["models/ace2-era5/ace2_era5_ckpt.tar"]["sha256"] = "b" * 64
    bad_hf_lock_p = tmp_path / "bad_hf_ovr.lock"
    bad_hf_lock_p.write_text(json.dumps(bad_hf_lock), encoding="utf-8")
    with pytest.raises(ValidationError, match="models/ace2-era5/ace2_era5_ckpt.tar"):
        run_setup(
            models=("ace2-era5",),
            dest_dir=str(tmp_path / "bad_hf_dest"),
            lock_path=str(bad_hf_lock_p),
            accept_derived=True,
            years=yc,
        )

    # 4. When `boot_dest` already has verified `ace2-era5` and `ace2-eamv3`,
    # a strict failed-only rerun for ONLY `ace2-eamv3` (`verified_models == []`) MUST evict
    # `ace2-eamv3` from `<boot_dest>/registry.json` while preserving `ace2-era5`.
    assert list(load_registry(data_dir=str(boot_dest)).keys()) == ["ace2-era5", "ace2-eamv3"]
    with pytest.raises(ValidationError, match="ace2-eamv3"):
        run_setup(
            models=("ace2-eamv3",),
            dest_dir=str(boot_dest),
            lock_path=str(mismatch_lock_p),
            years=yc,
        )
    assert list(load_registry(data_dir=str(boot_dest)).keys()) == ["ace2-era5"]

    # 5. Strict retry after `--accept-derived` on `optin_dest`
    # drops `ace2-eamv3` from `registry.json`, preserves `ace2-era5`, and clears the stale override.
    with pytest.raises(ValidationError, match="ace2-eamv3"):
        run_setup(
            models=("ace2-eamv3",),
            dest_dir=str(optin_dest),
            lock_path=str(mismatch_lock_p),
            accept_derived=False,
            years=yc,
        )
    assert list(load_registry(data_dir=str(optin_dest)).keys()) == ["ace2-era5"]
    ovr_after_retry = json.loads(ovr_file.read_text(encoding="utf-8"))
    assert ovr_after_retry["overrides"] == []

    # 6. When EVERY published model is invalidated, `<dest>/registry.json`
    # stays on disk with `"models": []` so `$AIMIP_DATA` does NOT fall back to `registry.default.json`.
    single_dest = tmp_path / "single_eamv3_only"
    run_setup(
        models=("ace2-eamv3",),
        dest_dir=str(single_dest),
        lock_path=str(mismatch_lock_p),
        accept_derived=True,
        years=yc,
    )
    assert list(load_registry(data_dir=str(single_dest)).keys()) == ["ace2-eamv3"]

    with pytest.raises(ValidationError, match="ace2-eamv3"):
        run_setup(
            models=("ace2-eamv3",),
            dest_dir=str(single_dest),
            lock_path=str(mismatch_lock_p),
            accept_derived=False,
            years=yc,
        )
    empty_reg_path = single_dest / "registry.json"
    assert empty_reg_path.is_file()
    assert json.loads(empty_reg_path.read_text(encoding="utf-8"))["models"] == []

    monkeypatch.setenv("AIMIP_DATA", str(single_dest))
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)
    assert load_registry() == {}

    # Step-1 download verification failure ALSO evicts a previously published model (`ace2-era5` in `boot_dest`)!
    with pytest.raises(ValidationError, match="models/ace2-era5/ace2_era5_ckpt.tar"):
        run_setup(
            models=("ace2-era5",),
            dest_dir=str(boot_dest),
            lock_path=str(bad_hf_lock_p),
            years=yc,
        )
    assert load_registry(data_dir=str(boot_dest)) == {}
