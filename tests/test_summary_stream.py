"""Tests for month-by-month single-slice streaming in `aimip.status.summarize_run`."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import pytest
import xarray as xr

from conftest import parse_cli_json
from aimip import status as status_mod
from aimip.cli import main
from aimip.emu_job import MonthlyWriter
from aimip.errors import ValidationError


def test_summary_streams_full_aimip_555_month_quarter_degree_run_within_budget(
    allow_tmp_run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A full 555-month 0.25-deg (`1 x 555 x 721 x 1440`, 2.30 GB uncompressed) run summarizes slice by slice."""
    assert status_mod.MAX_SUMMARY_VAR_BYTES == 512 * 1024 * 1024

    n_months = 555  # 1978-10 through 2024-12 (full AIMIP rollout)
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float64)
    lon = np.linspace(0.0, 359.75, 1440, dtype=np.float64)
    whole_var_bytes = 1 * n_months * lat.size * lon.size * 4
    single_slice_bytes = 1 * lat.size * lon.size * 4
    assert whole_var_bytes == 2_304_892_800
    assert whole_var_bytes > status_mod.MAX_SUMMARY_VAR_BYTES
    assert single_slice_bytes == 4_152_960
    assert single_slice_bytes < status_mod.MAX_SUMMARY_VAR_BYTES

    run_dir = allow_tmp_run_root / "dlesym_aimip_full"
    nc_path = run_dir / "atmosphere" / "monthly_mean_predictions.nc"
    writer = MonthlyWriter(
        str(nc_path),
        ["t2m"],
        lat,
        lon,
        1,
        extra={"t2m": {"units": "K", "long_name": "2m temperature"}},
    )

    cos_lat = np.maximum(0.0, np.cos(np.deg2rad(lat)))
    cos_lat_sum = float(np.sum(cos_lat))
    cos_lat_f32 = cos_lat.astype(np.float32)
    buf = np.empty((lat.size, lon.size), dtype=np.float32)
    start_month = np.datetime64("1978-10", "M")
    expected_means: list[float] = []

    for m_idx in range(n_months):
        profile_f32 = (
            np.float32(275.0 + 0.01 * m_idx) + np.float32(20.0) * cos_lat_f32
        ).astype(np.float32)
        buf[:] = profile_f32[:, None]
        writer.write(0, start_month + m_idx, {"t2m": buf}, 120)
        expected_means.append(
            float(np.sum(profile_f32.astype(np.float64) * cos_lat) / cos_lat_sum)
        )
    writer.close()

    # Sanity check: equator-weighted profile differs by ~3 K from unweighted mean.
    unweighted_m0 = float(np.mean((np.float32(275.0) + np.float32(20.0) * cos_lat_f32).astype(np.float64)))
    assert abs(expected_means[0] - unweighted_m0) > 2.5

    orig_isel = xr.DataArray.isel
    t2m_slices: list[tuple[Any, tuple[int, ...]]] = []

    def _recording_isel(self: xr.DataArray, *args: Any, **kwargs: Any) -> xr.DataArray:
        out = orig_isel(self, *args, **kwargs)
        if self.name == "t2m" and self.ndim == 4:
            idx_map = dict(args[0]) if args and isinstance(args[0], dict) else {}
            idx_map.update(kwargs)
            t2m_slices.append((idx_map.get("time"), tuple(int(s) for s in out.shape)))
        return out

    monkeypatch.setattr(xr.DataArray, "isel", _recording_isel)

    summary = status_mod.summarize_run(str(run_dir))
    monkeypatch.setattr(xr.DataArray, "isel", orig_isel)
    assert [idx for idx, _ in t2m_slices] == list(range(n_months))
    assert all(
        isinstance(idx, int) and shape == (1, 721, 1440)
        for idx, shape in t2m_slices
    )

    assert summary["n_samples"] == 1
    assert summary["n_months"] == n_months
    assert summary["spatial_dims"] == ["lat", "lon"]
    t2m_doc = summary["variables"]["t2m"]
    assert t2m_doc["units"] == "K"
    series = t2m_doc["series"]
    assert len(series) == n_months
    assert series[0]["month"] == "1978-10"
    assert series[-1]["month"] == "2024-12"
    for m_idx, row in enumerate(series):
        assert row["month_index"] == m_idx
        assert row["counts"] == 120
        assert row["ensemble_mean"] is not None
        assert math.isclose(
            row["ensemble_mean"], expected_means[m_idx], rel_tol=0.0, abs_tol=1e-5
        )
        assert row["members"] == [row["ensemble_mean"]]

    assert main(["summary", str(run_dir), "--json"]) == 0
    cli_res = parse_cli_json(capsys.readouterr().out)
    assert cli_res["n_months"] == n_months
    assert math.isclose(
        cli_res["variables"]["t2m"]["series"][-1]["ensemble_mean"],
        expected_means[-1],
        rel_tol=0.0,
        abs_tol=1e-5,
    )

    # Exact single-slice boundary: `single_slice_bytes` passes, `single_slice_bytes - 1` rejects up front.
    exact_fit = status_mod._summarize_monthly_dataset(
        str(nc_path), ("t2m",), max_var_bytes=single_slice_bytes
    )
    assert exact_fit["n_months"] == n_months
    with pytest.raises(ValidationError, match="exceeding diagnostic budget"):
        status_mod._summarize_monthly_dataset(
            str(nc_path), ("t2m",), max_var_bytes=single_slice_bytes - 1
        )


def test_summary_rejects_single_slice_exceeding_default_budget(
    allow_tmp_run_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A single monthly slice exceeding the 512 MiB budget is rejected up front without reading chunks."""
    assert status_mod.MAX_SUMMARY_VAR_BYTES == 536_870_912

    run_dir = allow_tmp_run_root / "oversized_single_slice"
    run_dir.mkdir(parents=True)
    nc_path = run_dir / "monthly_mean_predictions.nc"

    # 64 * 1441 * 2880 * 4 = 1,062,420,480 bytes per monthly slice (> 536,870,912 bytes).
    # Chunked NetCDF4 allocates zero data blocks when unwritten, keeping fixture creation instantaneous.
    with netCDF4.Dataset(str(nc_path), "w", format="NETCDF4") as ds:
        ds.createDimension("sample", 64)
        ds.createDimension("time", 1)
        ds.createDimension("lat", 1441)
        ds.createDimension("lon", 2880)
        v_lat = ds.createVariable("lat", "f8", ("lat",))
        v_lat[:] = np.linspace(90.0, -90.0, 1441)
        v_lat.units = "degrees_north"
        v_lon = ds.createVariable("lon", "f8", ("lon",))
        v_lon[:] = np.linspace(0.0, 359.875, 2880)
        v_lon.units = "degrees_east"
        v_t = ds.createVariable("time", "f8", ("time",))
        v_t[:] = [0.0]
        v_t.units = "days since 1970-01-01"
        v_t.calendar = "proleptic_gregorian"
        ds.createVariable(
            "t2m",
            "f4",
            ("sample", "time", "lat", "lon"),
            zlib=True,
            complevel=1,
            chunksizes=(1, 1, 1441, 2880),
        )

    with pytest.raises(
        ValidationError,
        match=r"1062420480 uncompressed bytes, exceeding diagnostic budget \(536870912 bytes\)",
    ):
        status_mod.summarize_run(str(run_dir))

    assert main(["summary", str(run_dir)]) != 0
    assert "exceeding diagnostic budget" in capsys.readouterr().err


def test_summary_streams_coupled_masked_ocean_and_skips_unwritten_months(
    allow_tmp_run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Month-by-month streaming handles partial land NaNs and skips slices where all members have `counts <= 0`."""
    run_dir = allow_tmp_run_root / "coupled_stream"
    lat = np.array([-60.0, 0.0, 60.0], dtype=np.float64)
    lon = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
    months = (
        np.datetime64("1979-01", "M"),
        np.datetime64("1979-02", "M"),
        np.datetime64("1979-03", "M"),
    )

    atmos_w = MonthlyWriter(
        str(run_dir / "atmosphere" / "monthly_mean_predictions.nc"),
        ["t2m"],
        lat,
        lon,
        2,
        extra={"t2m": {"units": "K"}},
    )
    ocean_w = MonthlyWriter(
        str(run_dir / "ocean" / "monthly_mean_predictions.nc"),
        ["sst"],
        lat,
        lon,
        2,
        extra={"sst": {"units": "degC"}},
    )

    t2m_m0 = np.broadcast_to(np.array([[250.0], [300.0], [250.0]], dtype=np.float32), (3, 4)).copy()
    t2m_m1 = (t2m_m0 + np.float32(6.0)).astype(np.float32)
    sst_m0 = np.broadcast_to(np.array([[2.0], [26.0], [np.nan]], dtype=np.float32), (3, 4)).copy()
    sst_m1 = np.broadcast_to(np.array([[4.0], [28.0], [np.nan]], dtype=np.float32), (3, 4)).copy()

    # Month 0: both members active; Month 1: only member 0 active; Month 2: both members `counts = 0`.
    atmos_w.write(0, months[0], {"t2m": t2m_m0}, 124)
    atmos_w.write(1, months[0], {"t2m": t2m_m1}, 124)
    atmos_w.write(0, months[1], {"t2m": t2m_m0 + np.float32(2.0)}, 112)
    atmos_w.write(1, months[1], {"t2m": np.zeros((3, 4), dtype=np.float32)}, 0)
    atmos_w.write(0, months[2], {"t2m": np.zeros((3, 4), dtype=np.float32)}, 0)
    atmos_w.write(1, months[2], {"t2m": np.zeros((3, 4), dtype=np.float32)}, 0)
    atmos_w.close()

    ocean_w.write(0, months[0], {"sst": sst_m0}, 124)
    ocean_w.write(1, months[0], {"sst": sst_m1}, 124)
    ocean_w.write(0, months[1], {"sst": sst_m0 + np.float32(1.5)}, 112)
    ocean_w.write(1, months[1], {"sst": np.full((3, 4), np.nan, dtype=np.float32)}, 0)
    ocean_w.write(0, months[2], {"sst": np.full((3, 4), np.nan, dtype=np.float32)}, 0)
    ocean_w.write(1, months[2], {"sst": np.full((3, 4), np.nan, dtype=np.float32)}, 0)
    ocean_w.close()

    orig_isel = xr.DataArray.isel
    t2m_loaded_months: list[Any] = []

    def _recording_isel(self: xr.DataArray, *args: Any, **kwargs: Any) -> xr.DataArray:
        out = orig_isel(self, *args, **kwargs)
        if self.name == "t2m" and self.ndim == 4:
            idx_map = dict(args[0]) if args and isinstance(args[0], dict) else {}
            idx_map.update(kwargs)
            t2m_loaded_months.append(idx_map.get("time"))
        return out

    monkeypatch.setattr(xr.DataArray, "isel", _recording_isel)

    res = status_mod.summarize_run(str(run_dir))
    # Month 2 (`counts == 0` for all members) is not read from disk.
    assert t2m_loaded_months == [0, 1]

    t2m_rows = res["variables"]["t2m"]["series"]
    assert [r["month"] for r in t2m_rows] == ["1979-01", "1979-02", "1979-03"]
    assert math.isclose(t2m_rows[0]["members"][0], 275.0, abs_tol=1e-6)
    assert math.isclose(t2m_rows[0]["members"][1], 281.0, abs_tol=1e-6)
    assert math.isclose(t2m_rows[0]["ensemble_mean"], 278.0, abs_tol=1e-6)
    assert math.isclose(t2m_rows[1]["members"][0], 277.0, abs_tol=1e-6)
    assert t2m_rows[1]["members"][1] is None
    assert math.isclose(t2m_rows[1]["ensemble_mean"], 277.0, abs_tol=1e-6)
    assert t2m_rows[2]["counts"] == 0
    assert t2m_rows[2]["members"] == [None, None]
    assert t2m_rows[2]["ensemble_mean"] is None

    sst_rows = res["variables"]["ocean.sst"]["series"]
    # cos(-60) = 0.5, cos(0) = 1.0, cos(60) masked out -> (0.5*2 + 1.0*26)/1.5 = 18.0
    assert math.isclose(sst_rows[0]["members"][0], 18.0, abs_tol=1e-6)
    assert math.isclose(sst_rows[0]["members"][1], 20.0, abs_tol=1e-6)
    assert math.isclose(sst_rows[0]["ensemble_mean"], 19.0, abs_tol=1e-6)
    assert math.isclose(sst_rows[1]["members"][0], 19.5, abs_tol=1e-6)
    assert sst_rows[1]["members"][1] is None
    assert sst_rows[2]["ensemble_mean"] is None
