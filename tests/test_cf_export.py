"""`aimip export-cf`: CF copies of the monthly means (time first, dates from valid_time, one file per member)."""
from __future__ import annotations

import os
from pathlib import Path

import cftime
import numpy as np
import pytest
import xarray as xr

from aimip.cf_export import GRAVITY, export_cf
from aimip.errors import ValidationError


def _ace_like(path: Path, *, n_member: int = 2, n_time: int = 3) -> None:
    """A monthly file in the ACE layout: (sample, time, lat, lon), time = month counter, dates in valid_time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lat = np.linspace(-89.5, 89.5, 4)
    lon = np.linspace(0.5, 359.5, 8)
    shape = (n_member, n_time, lat.size, lon.size)
    base = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    dates = [cftime.DatetimeProlepticGregorian(2001, 1 + t, 1) for t in range(n_time)]
    vt = np.array([dates] * n_member, dtype=object)
    ds = xr.Dataset(
        {
            "TMP2m": (("sample", "time", "lat", "lon"), 280.0 + base, {"units": "K", "long_name": "2m temperature"}),
            "PRATEsfc": (("sample", "time", "lat", "lon"), base * 1e-6, {"units": "kg/m**2/s"}),
            "ULWRFtoa": (("sample", "time", "lat", "lon"), 200.0 + base, {"units": "W/m**2"}),
            "h500": (("sample", "time", "lat", "lon"), 5500.0 + base, {"units": "m"}),
            "Q2m": (("sample", "time", "lat", "lon"), base * 1e-5, {"units": "furlongs"}),  # wrong units: keep name
            "eastward_wind_1": (("sample", "time", "lat", "lon"), base, {"units": "m/s"}),
            "counts": (("sample", "time"), np.full((n_member, n_time), 124, dtype="int32")),
            "valid_time": (("sample", "time"), vt),
        },
        coords={"lat": lat, "lon": lon, "time": np.arange(n_time),
                "init_time": ("sample", [cftime.DatetimeProlepticGregorian(2001, 1, 1)] * n_member)},
    )
    ds["valid_time"].encoding = {"units": "days since 1970-01-01", "calendar": "proleptic_gregorian"}
    ds["init_time"].encoding = {"units": "days since 1970-01-01", "calendar": "proleptic_gregorian"}
    ds["TMP2m"].encoding["coordinates"] = "init_time valid_time"
    ds.to_netcdf(path)


def test_export_cf_time_first_dates_members_and_cmip_names(allow_tmp_run_root: Path) -> None:
    run = allow_tmp_run_root / "cfrun"
    _ace_like(run / "monthly_mean_predictions.nc")
    res = export_cf(str(run))
    assert [f["member"] for f in res["files"]] == [0, 1]
    assert all(f["time_from"] == "valid_time" for f in res["files"])
    out = xr.open_dataset(run / "cf" / "cfrun_monthly_m001.nc", use_cftime=True)
    for v in out.data_vars:
        if "time" in out[v].dims:
            assert out[v].dims[0] == "time", v
    assert "sample" not in out.dims and "valid_time" not in out.variables
    assert [(t.year, t.month) for t in out["time"].values] == [(2001, 1), (2001, 2), (2001, 3)]
    assert out.attrs["Conventions"] == "CF-1.8" and int(out.attrs["aimip_member"]) == 1
    assert out.attrs["aimip_names"] == "cmip" and out.attrs["aimip_source_file"] == "monthly_mean_predictions.nc"
    assert out["lat"].attrs["standard_name"] == "latitude" and out["lon"].attrs["units"] == "degrees_east"
    # exact equivalents renamed, with the model name kept; values unchanged
    assert out["tas"].attrs["original_name"] == "TMP2m" and out["tas"].attrs["units"] == "K"
    assert out["pr"].attrs["standard_name"] == "precipitation_flux"
    assert out["rlut"].attrs["units"] == "W m-2" and out["zg500"].attrs["original_name"] == "h500"
    src = xr.open_dataset(run / "monthly_mean_predictions.nc", decode_times=False)
    np.testing.assert_array_equal(out["tas"].values, src["TMP2m"].isel(sample=1).values)
    # unknown units and model-layer fields keep the model name
    assert "Q2m" in out.data_vars and "huss" not in out.data_vars
    notes = res["files"][1]["notes"]
    assert any(n.startswith("Q2m: kept its model name") for n in notes)
    assert any(n.startswith("renamed TMP2m -> tas") for n in notes)
    assert any(n.startswith("time axis taken from valid_time (calendar proleptic_gregorian)") for n in notes)
    assert "renamed PRATEsfc -> pr" in out.attrs["aimip_postprocessing"]
    assert not any(n.startswith("CONVERTED") for n in notes)  # nothing but names and unit spellings changed
    assert "eastward_wind_1" in out.data_vars
    # no auxiliary coordinates left for cdo to trip over (the writers name valid_time in `coordinates`)
    raw = xr.open_dataset(run / "cf" / "cfrun_monthly_m001.nc", decode_coords=False, use_cftime=True)
    assert all("coordinates" not in raw[v].attrs for v in raw.variables)
    assert "init_time" in raw.data_vars and "init_time" not in raw.coords
    # the run's own file is untouched
    assert src["TMP2m"].dims == ("sample", "time", "lat", "lon")


def test_export_cf_native_names_members_components_and_refusals(allow_tmp_run_root: Path) -> None:
    run = allow_tmp_run_root / "cfrun2"
    _ace_like(run / "atmosphere" / "monthly_mean_predictions.nc", n_member=3)
    res = export_cf(str(run), members=[2], names="native")
    assert len(res["files"]) == 1 and res["files"][0]["component"] == "atmosphere"
    out = xr.open_dataset(run / "cf" / "cfrun2_atmosphere_m002.nc", use_cftime=True)
    assert out.attrs["aimip_names"] == "native"
    assert out.attrs["aimip_source_file"] == "atmosphere/monthly_mean_predictions.nc"
    assert "TMP2m" in out.data_vars and "tas" not in out.data_vars
    with pytest.raises(ValidationError):
        export_cf(str(run), members=[3])
    with pytest.raises(ValidationError):
        export_cf(str(run), names="cf")
    empty = allow_tmp_run_root / "nofiles"
    empty.mkdir()
    with pytest.raises(ValidationError):
        export_cf(str(empty))


def test_export_cf_geopotential_converted_to_height(allow_tmp_run_root: Path) -> None:
    run = allow_tmp_run_root / "cfrun3"
    path = run / "monthly_mean_predictions.nc"
    path.parent.mkdir(parents=True)
    xr.Dataset(
        {"z500": (("sample", "time", "lat", "lon"), np.full((1, 2, 2, 2), 55000.0), {"units": "m**2 s**-2"})},
        coords={"lat": [-45.0, 45.0], "lon": [0.0, 180.0],
                "time": [cftime.DatetimeProlepticGregorian(2001, m, 1) for m in (1, 2)]},
    ).to_netcdf(path)
    export_cf(str(run))
    out = xr.open_dataset(run / "cf" / "cfrun3_monthly_m000.nc", use_cftime=True)
    np.testing.assert_allclose(out["zg500"].values, 55000.0 / GRAVITY)
    assert out["zg500"].attrs["units"] == "m" and out["zg500"].attrs["original_name"] == "z500"
    # the conversion is announced and recorded, in the result and in the file
    res = export_cf(str(run))
    assert any(n.startswith("CONVERTED z500") and "9.80665" in n for n in res["files"][0]["notes"])
    out = xr.open_dataset(run / "cf" / "cfrun3_monthly_m000.nc", use_cftime=True)
    assert "CONVERTED z500" in out.attrs["aimip_postprocessing"] and "9.80665" in out["zg500"].attrs["aimip_conversion"]


def test_export_cf_r5_r6_unit_spellings_unitless_dlesym_drop_z250_and_long_name(allow_tmp_run_root: Path) -> None:
    run = allow_tmp_run_root / "cm4_and_dlesym"
    atmos = run / "atmosphere" / "monthly_mean_predictions.nc"
    atmos.parent.mkdir(parents=True)
    xr.Dataset(
        {
            "TMP2m": (("sample", "time", "lat", "lon"), np.full((1, 2, 2, 2), 287.0, dtype="float32"), {"units": "deg_k"}),
            "ULWRFtoa": (("sample", "time", "lat", "lon"), np.full((1, 2, 2, 2), 240.0, dtype="float32"), {"units": "watts/m2"}),
            "z500": (("sample", "time", "lat", "lon"), np.full((1, 2, 2, 2), 55000.0, dtype="float32"),
                     {"units": "m**2 s**-2", "long_name": "500 hPa geopotential"}),
            "z250": (("sample", "time", "lat", "lon"), np.full((1, 2, 2, 2), 102000.0, dtype="float32"),
                     {"units": "m**2 s**-2", "long_name": "250 hPa geopotential"}),
            "tcwv": (("sample", "time", "lat", "lon"), np.full((1, 2, 2, 2), 25.0, dtype="float32")),  # DLESyM: no units
        },
        coords={"lat": [-45.0, 45.0], "lon": [0.0, 180.0],
                "time": [cftime.DatetimeNoLeap(311, m, 15) for m in (1, 2)]},
    ).to_netcdf(atmos)

    res = export_cf(str(run))
    f0 = res["files"][0]
    out = xr.open_dataset(f0["file"], use_cftime=True)
    # R6: SamudrACE-CM4 "deg_k" and "watts/m2" are recognised
    assert "tas" in out.data_vars and out["tas"].attrs["original_units"] == "deg_k"
    assert "rlut" in out.data_vars and out["rlut"].attrs["original_units"] == "watts/m2"
    # C1: converted geopotential updates long_name and keeps original_long_name
    assert out["zg500"].attrs["long_name"] == "geopotential height at 500 hPa"
    assert out["zg500"].attrs["original_long_name"] == "500 hPa geopotential"
    # Small: z250 is not renamed to non-CMIP zg250
    assert "z250" in out.data_vars and "zg250" not in out.data_vars
    # R5: unitless variable keeps its name with an explicit "no units in the file" note
    assert "tcwv" in out.data_vars and "prw" not in out.data_vars
    assert any("tcwv: kept its model name and values (no units in the file: not renamed to prw)" in n for n in f0["notes"])


def test_export_cf_preserves_source_time_attrs_and_handles_no_valid_time_or_missing_vt(
    allow_tmp_run_root: Path,
) -> None:
    # 1. Spherical DYffusion nominal cycle long_name and 0001-01-01 epoch are preserved on the new time axis.
    sdyff_run = allow_tmp_run_root / "sdyff_nominal"
    sdyff_nc = sdyff_run / "monthly_mean_predictions.nc"
    sdyff_nc.parent.mkdir(parents=True)
    vt_nom = np.array([[cftime.DatetimeNoLeap(1, 1, 1), cftime.DatetimeNoLeap(1, 2, 1)]], dtype=object)
    ds_nom = xr.Dataset(
        {
            "PRESsfc": (("sample", "time", "lat", "lon"), np.full((1, 2, 2, 2), 101325.0, dtype="float32"), {"units": "Pa"}),
            "valid_time": (("sample", "time"), vt_nom),
        },
        coords={
            "lat": [-45.0, 45.0],
            "lon": [0.0, 180.0],
            "time": ("time", [0, 31], {"long_name": "nominal cycle month start (not a historical calendar year)"}),
        },
    )
    ds_nom["valid_time"].encoding = {"units": "days since 0001-01-01 00:00:00", "calendar": "noleap"}
    ds_nom.to_netcdf(sdyff_nc)

    res_nom = export_cf(str(sdyff_run))
    out_nom = xr.open_dataset(res_nom["files"][0]["file"], use_cftime=True)
    assert out_nom["time"].attrs["long_name"] == "nominal cycle month start (not a historical calendar year)"
    ref_nom = cftime.num2date(0, out_nom["time"].encoding["units"], calendar=out_nom["time"].encoding["calendar"])
    assert (ref_nom.year, ref_nom.month, ref_nom.day, ref_nom.hour, ref_nom.minute, ref_nom.second) == (1, 1, 1, 0, 0, 0)
    assert [(t.year, t.month) for t in out_nom["time"].values] == [(1, 1), (1, 2)]

    # 2. Old run without valid_time and integer step-counter time -> keeps time as written + "no dates" note, nothing invented.
    old_run = allow_tmp_run_root / "old_nodates"
    old_nc = old_run / "monthly_mean_predictions.nc"
    old_nc.parent.mkdir(parents=True)
    xr.Dataset(
        {"TMP2m": (("sample", "time", "lat", "lon"), np.full((1, 2, 2, 2), 288.0, dtype="float32"), {"units": "K"})},
        coords={"lat": [-45.0, 45.0], "lon": [0.0, 180.0], "time": [0, 1]},
    ).to_netcdf(old_nc)
    res_old = export_cf(str(old_run))
    assert res_old["files"][0]["time_from"] == "time"
    assert any("no dates" in n for n in res_old["files"][0]["notes"])
    out_old = xr.open_dataset(res_old["files"][0]["file"], decode_times=False)
    np.testing.assert_array_equal(out_old["time"].values, [0, 1])

    # 3. Incomplete member with missing valid_time is refused cleanly with ValidationError.
    bad_vt_run = allow_tmp_run_root / "bad_vt"
    bad_vt_nc = bad_vt_run / "monthly_mean_predictions.nc"
    bad_vt_nc.parent.mkdir(parents=True)
    ds_bad = xr.Dataset(
        {
            "TMP2m": (("sample", "time", "lat", "lon"), np.full((2, 2, 2, 2), 288.0, dtype="float32"), {"units": "K"}),
            "valid_time": (("sample", "time"), np.array([[0.0, 31.0], [0.0, np.nan]], dtype="float64"),
                           {"units": "days since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian"}),
        },
        coords={"lat": [-45.0, 45.0], "lon": [0.0, 180.0], "time": [0, 1]},
    )
    ds_bad.to_netcdf(bad_vt_nc)
    with pytest.raises(ValidationError, match="missing values"):
        export_cf(str(bad_vt_run))


def test_export_cf_reads_outside_run_root_writes_only_under_run_root_and_s3_labels(
    allow_tmp_run_root: Path,
) -> None:
    import hashlib
    from aimip.cli import main

    ext_run = allow_tmp_run_root.parent / "external_aqua_run"
    src_nc = ext_run / "monthly_mean_predictions.nc"
    _ace_like(src_nc, n_member=1, n_time=2)
    sha_before = hashlib.sha256(src_nc.read_bytes()).hexdigest()

    # Without --to, writing into <ext_run>/cf outside the run root is refused with --to guidance.
    with pytest.raises(ValidationError, match="--to"):
        export_cf(str(ext_run))

    # --to outside the run root is refused with the '--to' label (not '--out').
    bad_to = allow_tmp_run_root.parent / "outside_cf"
    with pytest.raises(ValidationError, match="--to") as exc_to:
        export_cf(str(ext_run), out_dir=str(bad_to))
    assert "--out" not in str(exc_to.value)

    # Reading from external read-only source into --to under the caller's run root succeeds; source SHA-256 unchanged.
    dst_cf = allow_tmp_run_root / "aqua_cf"
    assert main(["export-cf", str(ext_run), "--to", str(dst_cf)]) == 0
    assert hashlib.sha256(src_nc.read_bytes()).hexdigest() == sha_before
    assert not (ext_run / "cf").exists()
    res = export_cf(str(ext_run), out_dir=str(dst_cf))
    assert res["files"][0]["component"] == "monthly"
    assert hashlib.sha256(src_nc.read_bytes()).hexdigest() == sha_before


def test_export_cf_wrapper_s7_options_and_ro_external_source_mount(tmp_path: Path) -> None:
    import hashlib
    import subprocess

    repo_root = Path(__file__).resolve().parent.parent
    tmpl = (repo_root / "container" / "aimip-wrapper.in").read_text(encoding="utf-8")
    pfx = tmp_path / "wpfx"
    (pfx / "data").mkdir(parents=True)
    sif = pfx / "aimip.sif"
    sif.write_bytes(b"fake-sif-image")
    (pfx / "image.sha256").write_text(hashlib.sha256(b"fake-sif-image").hexdigest() + "\n", encoding="utf-8")
    run_root = tmp_path / "wruns"
    run_root.mkdir(mode=0o700)
    ext_src = tmp_path / "ext_aqua"
    ext_src.mkdir(mode=0o500)
    extra_out = tmp_path / "extra_out"
    extra_out.mkdir(mode=0o700)

    argv_log = tmp_path / "rt_argv.txt"
    fake_rt = tmp_path / "fake_apptainer"
    fake_rt.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$@" > "{argv_log}"\nexit 0\n',
        encoding="utf-8",
    )
    fake_rt.chmod(0o755)
    wrapper = tmp_path / "aimip"
    rendered = (
        tmpl.replace("@PREFIX@", str(pfx))
        .replace("@RUNTIME@", "apptainer")
        .replace("@RUNTIME_BIN@", str(fake_rt))
        .replace("@IMAGE@", str(sif))
        .replace("@SCHEDULER@", "none")
        .replace("@RUN_ROOT@", str(run_root))
        .replace("@EXTRA_RO@", "")
        .replace("@UNSQUASH@", "0")
    )
    wrapper.write_text(rendered, encoding="utf-8")
    wrapper.chmod(0o755)

    try:
        r = subprocess.run(
            [
                str(wrapper),
                "export-cf",
                "--names",
                "native",
                "--members",
                "0",
                str(ext_src),
                "--to",
                str(extra_out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert r.returncode == 0, r.stderr
        argv_lines = argv_log.read_text(encoding="utf-8").splitlines()
        assert f"{ext_src}:{ext_src}:ro" in argv_lines
        assert f"{ext_src}:{ext_src}:rw" not in argv_lines
        assert f"{extra_out}:{extra_out}:rw" in argv_lines
    finally:
        ext_src.chmod(0o700)


def test_export_cf_s9_exports_monthly_levels_and_dlesym_monthly_precip(allow_tmp_run_root: Path) -> None:
    run = allow_tmp_run_root / "aw_and_dlesym_aux"
    (run / "atmosphere").mkdir(parents=True)
    dates = [cftime.DatetimeProlepticGregorian(1979, m, 1) for m in (1, 2)]
    vt = np.array([dates, dates], dtype=object)

    # ArchesWeather surface + 13-level monthly_levels.nc (sample, time, level, lat, lon)
    _ace_like(run / "atmosphere" / "monthly_mean_predictions.nc", n_member=2, n_time=2)
    ds_lev = xr.Dataset(
        {
            "zg": (("sample", "time", "level", "lat", "lon"), np.full((2, 2, 3, 2, 2), 5500.0, dtype="float32"), {"units": "m"}),
            "ta": (("sample", "time", "level", "lat", "lon"), np.full((2, 2, 3, 2, 2), 260.0, dtype="float32"), {"units": "K"}),
            "valid_time": (("sample", "time"), vt),
            "counts": (("sample", "time"), np.full((2, 2), 31, dtype="int32")),
        },
        coords={"level": ("level", [500.0, 850.0, 1000.0], {"units": "hPa"}), "lat": [-45.0, 45.0], "lon": [0.0, 180.0], "time": [0, 1]},
    )
    ds_lev["valid_time"].encoding = {"units": "days since 1970-01-01", "calendar": "proleptic_gregorian"}
    ds_lev.to_netcdf(run / "atmosphere" / "monthly_levels.nc")

    # DLESyM v1.1 atmosphere/monthly_precip.nc
    ds_pr = xr.Dataset(
        {
            "tp06": (("sample", "time", "lat", "lon"), np.full((2, 2, 2, 2), 0.001, dtype="float32"), {"units": "m per 6 h"}),
            "valid_time": (("sample", "time"), vt),
            "counts": (("sample", "time"), np.full((2, 2), 124, dtype="int32")),
        },
        coords={"lat": [-45.0, 45.0], "lon": [0.0, 180.0], "time": [0, 1]},
    )
    ds_pr["valid_time"].encoding = {"units": "days since 1970-01-01", "calendar": "proleptic_gregorian"}
    ds_pr.to_netcdf(run / "atmosphere" / "monthly_precip.nc")

    res = export_cf(str(run), members=[1])
    by_comp = {f["component"]: f for f in res["files"]}
    assert set(by_comp) == {"atmosphere", "atmosphere_levels", "atmosphere_precip"}
    out_lev = xr.open_dataset(by_comp["atmosphere_levels"]["file"], use_cftime=True)
    assert out_lev["zg"].dims == ("time", "level", "lat", "lon")
    assert [(t.year, t.month) for t in out_lev["time"].values] == [(1979, 1), (1979, 2)]
    out_pr = xr.open_dataset(by_comp["atmosphere_precip"]["file"], use_cftime=True)
    assert out_pr["tp06"].dims == ("time", "lat", "lon")


def test_export_cf_s4_s5_run_and_model_filename_and_untrusted_model_slug_guard(allow_tmp_run_root: Path) -> None:
    import json

    run = allow_tmp_run_root / "first-year"
    _ace_like(run / "monthly_mean_predictions.nc", n_member=1, n_time=2)
    (run / "aimip-manifest.json").write_text(
        json.dumps({"registry_entry": {"name": "ace2-era5"}, "finished": "2026-11-20T00:00:00Z", "rc": 0}),
        encoding="utf-8",
    )
    res = export_cf(str(run))
    assert res["model"] == "ace2-era5"
    expect = run / "cf" / "first-year_ace2-era5_monthly_m000.nc"
    assert Path(res["files"][0]["file"]) == expect and expect.is_file()

    # Path traversal in an untrusted manifest model slug is ignored rather than escaping out_dir.
    bad = allow_tmp_run_root / "untrusted-meta"
    _ace_like(bad / "monthly_mean_predictions.nc", n_member=1, n_time=1)
    (bad / "aimip-manifest.json").write_text(json.dumps({"model": "../../evil"}), encoding="utf-8")
    res_bad = export_cf(str(bad))
    assert res_bad["model"] is None
    assert Path(res_bad["files"][0]["file"]) == bad / "cf" / "untrusted-meta_monthly_m000.nc"


def test_export_cf_s1_uses_synchronous_dask_scheduler_on_to_netcdf(
    allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dask = pytest.importorskip("dask")
    run = allow_tmp_run_root / "sync_dask_run"
    _ace_like(run / "monthly_mean_predictions.nc", n_member=1, n_time=2)

    seen_schedulers: list[str | None] = []
    orig_to_netcdf = xr.Dataset.to_netcdf

    def _spy_to_netcdf(self: xr.Dataset, *args: object, **kwargs: object) -> object:
        seen_schedulers.append(dask.config.get("scheduler", None))
        return orig_to_netcdf(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _spy_to_netcdf)
    export_cf(str(run))
    assert seen_schedulers == ["synchronous"]


def test_export_cf_r2_s6_s8_prewrite_members_check_clean_errors_and_incomplete_cleanup(
    allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = allow_tmp_run_root / "err_guard_run"
    _ace_like(run / "monthly_mean_predictions.nc", n_member=2, n_time=2)

    # S6: --members=[0, 9] fails BEFORE creating <run>/cf or writing member 0.
    with pytest.raises(ValidationError, match="Member 9"):
        export_cf(str(run), members=[0, 9])
    assert not (run / "cf").exists()

    # R2: <run>/cf existing as a regular file raises ValidationError naming the path.
    (run / "cf").write_text("not a dir", encoding="utf-8")
    with pytest.raises(ValidationError, match="not a directory"):
        export_cf(str(run))
    (run / "cf").unlink()

    # S8 / Codex TASK-421 item 2: a file or symlink at the old predictable <target>.incomplete name, or a hard link to
    # another file, is never opened, truncated or removed: the writer uses its own exclusive temporary file.
    (run / "cf").mkdir()
    victim = allow_tmp_run_root / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    inc_link = run / "cf" / "err_guard_run_monthly_m000.nc.incomplete"
    inc_link.symlink_to(victim)
    hard = allow_tmp_run_root / "hard.txt"
    hard.write_text("keep me too", encoding="utf-8")
    other = run / "cf" / "other.incomplete"
    os.link(hard, other)
    export_cf(str(run), members=[0])
    assert victim.read_text(encoding="utf-8") == "keep me" and inc_link.is_symlink()
    assert hard.read_text(encoding="utf-8") == "keep me too" and other.exists()
    assert not [p for p in (run / "cf").iterdir() if p.name.startswith(".") and p.name.endswith(".incomplete")]
    inc_link.unlink()
    other.unlink()

    # R2: OSError during to_netcdf raises ValidationError naming the file and removes .incomplete.
    orig_to_netcdf = xr.Dataset.to_netcdf

    def _boom_to_netcdf(self: xr.Dataset, path: str, *a: object, **kw: object) -> None:
        Path(path).write_bytes(b"partial-bytes")
        raise OSError("disk quota exceeded")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _boom_to_netcdf)
    with pytest.raises(ValidationError, match="disk quota exceeded") as exc_io:
        export_cf(str(run), members=[0])
    assert "err_guard_run_monthly_m000.nc" in str(exc_io.value)
    assert not [p for p in (run / "cf").iterdir() if p.name.endswith(".incomplete")]  # its own tmp removed

    # R2: corrupted NetCDF source raises ValidationError naming the file instead of raw traceback.
    monkeypatch.setattr(xr.Dataset, "to_netcdf", orig_to_netcdf)
    corrupt = allow_tmp_run_root / "corrupt_run"
    (corrupt / "monthly_mean_predictions.nc").parent.mkdir(parents=True)
    (corrupt / "monthly_mean_predictions.nc").write_bytes(b"not-a-netcdf-file")
    with pytest.raises(ValidationError) as exc_bad:
        export_cf(str(corrupt))
    assert "monthly_mean_predictions.nc" in str(exc_bad.value)


def test_export_cf_r4_s2_and_owner_terminal_warnings_text_and_json(
    allow_tmp_run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import hashlib
    import json
    import os
    from aimip.cli import main

    # 1. R4: symlinked --to output directory is refused; --to pointing at the run dir itself does NOT overwrite
    #    monthly_mean_predictions.nc (SHA-256 unchanged); second export overwrites the first cleanly.
    run = allow_tmp_run_root / "r4_run"
    src_nc = run / "monthly_mean_predictions.nc"
    _ace_like(src_nc, n_member=1, n_time=3)
    src_sha = hashlib.sha256(src_nc.read_bytes()).hexdigest()

    real_dst = allow_tmp_run_root / "real_dst"
    real_dst.mkdir()
    sym_dst = allow_tmp_run_root / "sym_dst"
    sym_dst.symlink_to(real_dst)
    with pytest.raises(ValidationError, match="symlink"):
        export_cf(str(run), out_dir=str(sym_dst))

    res_same = export_cf(str(run), out_dir=str(run))
    assert hashlib.sha256(src_nc.read_bytes()).hexdigest() == src_sha
    cf_in_run = Path(res_same["files"][0]["file"])
    assert cf_in_run.parent == run and cf_in_run.name == "r4_run_monthly_m000.nc"
    assert res_same["files"][0]["months"] == 3
    assert res_same["files"][0]["first_date"] == "2001-01-01"
    assert res_same["files"][0]["last_date"] == "2001-03-01"

    # Second export with names="native" overwrites the earlier copy cleanly.
    export_cf(str(run), out_dir=str(run), names="native")
    out2 = xr.open_dataset(cf_in_run, use_cftime=True)
    assert out2.attrs["aimip_names"] == "native"
    assert "TMP2m" in out2.data_vars and "tas" not in out2.data_vars
    assert hashlib.sha256(src_nc.read_bytes()).hexdigest() == src_sha

    # 2. Active run is refused; failed/incomplete run warns loudly on stderr; every component's CONVERTED /
    #    kept-name / no-dates note prints as WARNING: on stderr in BOTH text and --json mode, while JSON stdout stays pure.
    active_run = allow_tmp_run_root / "active_run"
    _ace_like(active_run / "monthly_mean_predictions.nc", n_member=1, n_time=1)
    (active_run / "aimip-manifest.json").write_text(json.dumps({"registry_entry": {"name": "ace2-era5"}}), encoding="utf-8")
    (active_run / "local.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="running"):
        export_cf(str(active_run))

    multi = allow_tmp_run_root / "multi_warn_run"
    (multi / "atmosphere").mkdir(parents=True)
    (multi / "ocean").mkdir(parents=True)
    (multi / "aimip-manifest.json").write_text(
        json.dumps({"registry_entry": {"name": "dlesym-v1.1"}, "finished": "2026-11-20T00:00:00Z", "rc": 5}),
        encoding="utf-8",
    )
    xr.Dataset(
        {
            "z500": (("sample", "time", "lat", "lon"), np.full((1, 2, 2, 2), 55000.0, dtype="float32"), {"units": "m**2 s**-2"}),
            "Q2m": (("sample", "time", "lat", "lon"), np.full((1, 2, 2, 2), 0.01, dtype="float32"), {"units": "furlongs"}),
        },
        coords={"lat": [-45.0, 45.0], "lon": [0.0, 180.0],
                "time": [cftime.DatetimeProlepticGregorian(2010, m, 1) for m in (6, 7)]},
    ).to_netcdf(multi / "atmosphere" / "monthly_mean_predictions.nc")
    xr.Dataset(
        {"sst": (("sample", "time", "lat", "lon"), np.full((1, 2, 2, 2), 290.0, dtype="float32"), {"units": "K"})},
        coords={"lat": [-45.0, 45.0], "lon": [0.0, 180.0], "time": [0, 1]},  # ocean without valid_time/dates
    ).to_netcdf(multi / "ocean" / "monthly_mean_predictions.nc")

    capsys.readouterr()
    assert main(["export-cf", str(multi)]) == 0
    captured_txt = capsys.readouterr()
    assert "2 months (2010-06-01..2010-07-01)" in captured_txt.out
    assert "2 months (no dates)" in captured_txt.out
    assert "WARNING: run status is failed (rc=5)" in captured_txt.err
    assert "WARNING: atmosphere: CONVERTED z500" in captured_txt.err
    assert "WARNING: atmosphere: Q2m: kept its model name" in captured_txt.err
    assert "WARNING: ocean: time has no dates" in captured_txt.err

    assert main(["export-cf", str(multi), "--json"]) == 0
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ok"] is True
    assert payload["result"]["run_state"] == "failed (rc=5)"
    assert "WARNING: run status is failed (rc=5)" in captured_json.err
    assert "WARNING: atmosphere: CONVERTED z500" in captured_json.err
    assert "WARNING: atmosphere: Q2m: kept its model name" in captured_json.err
    assert "WARNING: ocean: time has no dates" in captured_json.err

    # Invalid --members syntax on CLI fails cleanly (rc 2, no Python traceback).
    assert main(["export-cf", str(multi), "--members", "abc"]) == 2
    capsys.readouterr()


def test_export_cf_r1_s10_docs_and_completion() -> None:
    from aimip.completion import render_completion

    repo_root = Path(__file__).resolve().parent.parent
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    ug = (repo_root / "docs" / "user-guide.md").read_text(encoding="utf-8")
    assert "aimip export-cf" in readme and "aimip export-cmor" in readme
    assert "### CF copies for `cdo` and other tools (`aimip export-cf`)" in ug
    assert "aimip_postprocessing" in ug and "original_name" in ug and "DLESyM" in ug
    assert "Calendar julian unsupported" in ug
    assert "/path/to/shared/run" in ug and "/work/shared/ext-run" not in ug
    for sh in ("bash", "zsh"):
        comp = render_completion(sh)
        assert "export-cf" in comp and "--names" in comp and "--allow-unfinished" in comp
        assert "export-cmor" in comp and "--keep-first-month" in comp


def test_export_cf_allow_unfinished_on_killed_run(
    allow_tmp_run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json
    import os
    from aimip.cli import main

    killed = allow_tmp_run_root / "killed_local"
    _ace_like(killed / "monthly_mean_predictions.nc", n_member=1, n_time=2)
    (killed / "aimip-manifest.json").write_text(
        json.dumps({"registry_entry": {"name": "ace2-era5"}, "started_utc": "2026-11-20T00:00:00Z"}),
        encoding="utf-8",
    )
    (killed / "local.pid").write_text("99999999\n", encoding="utf-8")  # stale PID (> Linux PID_MAX_LIMIT)

    with pytest.raises(ValidationError) as exc_info:
        export_cf(str(killed))
    msg = str(exc_info.value)
    assert f"aimip status {killed.resolve()} refreshes the state" in msg
    assert "--allow-unfinished" in msg

    capsys.readouterr()
    assert main(["export-cf", str(killed), "--allow-unfinished"]) == 0
    err = capsys.readouterr().err
    assert "WARNING: run status is unfinished (not completed)" in err
    out = xr.open_dataset(killed / "cf" / "killed_local_ace2-era5_monthly_m000.nc", use_cftime=True)
    assert "run status is unfinished (not completed)" in out.attrs["aimip_postprocessing"]

    # An actively running PID is still refused even with --allow-unfinished.
    (killed / "local.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="still running"):
        export_cf(str(killed), allow_unfinished=True)

