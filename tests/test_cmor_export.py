"""`aimip export-cmor`: CMIP6-style Amon files, only exact mappings, nothing invented, every change stated."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cftime
import numpy as np
import pytest
import xarray as xr

from aimip.cf_export import GRAVITY
from aimip.cmor_export import _AMON, export_cmor
from aimip.errors import ValidationError

KW = {"source_id": "ACE2-ERA5", "experiment_id": "amip", "grid_label": "gn", "institution_id": "MY-INST",
      "model": "ace2-era5"}
FULL6H = ("full month = 4 model states per day (time step 6 h, current registry entry of ace2-era5 (the run record has "
          "no time step))")


@pytest.fixture(autouse=True)
def _packaged_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model time steps from the packaged registry (the test environment may resolve another registry)."""
    import json
    import types

    import aimip.registry as registry_mod

    packaged = json.loads((Path(registry_mod.__file__).parent / "registry.default.json").read_text())
    steps = {m["name"]: m["timestep_hours"] for m in packaged["models"]}
    monkeypatch.setattr(registry_mod, "get_model", lambda name, **_: types.SimpleNamespace(timestep_hours=steps[name]))


def _run(path: Path, *, months: list[tuple[int, int]] | None = None, counts: list[int] | None = None,
         cal: str = "proleptic_gregorian", with_dates: bool = True, extra: dict | None = None,
         attrs: dict | None = None) -> None:
    """ACE layout (sample, time, lat, lon); default: Oct 1978 partial (60 of 124 states), Nov and Dec full."""
    path.parent.mkdir(parents=True, exist_ok=True)
    months = months or [(1978, 10), (1978, 11), (1978, 12)]
    counts = counts or [60, 120, 124]
    nt = len(months)
    lat, lon = np.array([-45.0, 45.0]), np.array([0.0, 180.0])
    shape = (2, nt, 2, 2)
    base = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    data = {
        "TMP2m": (("sample", "time", "lat", "lon"), 280.0 + base, {"units": "K"}),
        "LHTFLsfc": (("sample", "time", "lat", "lon"), 80.0 + base, {"units": "W/m**2"}),
        "z500": (("sample", "time", "lat", "lon"), np.full(shape, 55000.0), {"units": "m**2 s**-2"}),
        "z1000": (("sample", "time", "lat", "lon"), np.full(shape, 1000.0), {"units": "m**2 s**-2"}),
        "Q2m": (("sample", "time", "lat", "lon"), base, {"units": "furlongs"}),
        "eastward_wind_1": (("sample", "time", "lat", "lon"), base, {"units": "m/s"}),
        "counts": (("sample", "time"), np.array([counts] * 2, dtype="int32")),
    }
    data.update(extra or {})
    if with_dates:
        dates = [cftime.datetime(y, m, 15, calendar=cal) for y, m in months]
        data["valid_time"] = (("sample", "time"), np.array([dates] * 2, dtype=object))
    ds = xr.Dataset(data, coords={"lat": lat, "lon": lon, "time": np.arange(nt)})
    ds.attrs.update(attrs or {})
    if with_dates:
        ds["valid_time"].encoding = {"units": "days since 1970-01-01", "calendar": cal}
    ds.to_netcdf(path)


def _one(res: dict, var: str, member: int = 0) -> xr.Dataset:
    f = next(Path(f["file"]) for f in res["files"] if f["variable"] == var and f["member"] == member)
    return xr.open_dataset(f, use_cftime=True)


def test_export_cmor_amon_files_drs_partial_month_and_conversion(allow_tmp_run_root: Path) -> None:
    run = allow_tmp_run_root / "cmorrun"
    src = run / "monthly_mean_predictions.nc"
    _run(src)
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    res = export_cmor(str(run), out_dir=str(run / "cmor"), **KW)
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before  # the run's own file is never touched
    # partial October left out and announced; Nov (120 = 30 days x 4) and Dec (124) are full
    assert res["notes"] == [FULL6H, "LEFT OUT 1978-10: partial month (60 of 124 model states)"]
    names = sorted(Path(f["file"]).name for f in res["files"] if f["member"] == 0)
    assert names == ["hfls_Amon_ACE2-ERA5_amip_r1i1p1f1_gn_197811-197812.nc",
                     "tas_Amon_ACE2-ERA5_amip_r1i1p1f1_gn_197811-197812.nc",
                     "zg_Amon_ACE2-ERA5_amip_r1i1p1f1_gn_197811-197812.nc"]
    tas = _one(res, "tas", member=1)
    assert Path(next(f["file"] for f in res["files"] if f["variable"] == "tas" and f["member"] == 1)).parent.parts[-8:] \
        == ("MY-INST", "ACE2-ERA5", "amip", "r2i1p1f1", "Amon", "tas", "gn", Path(res["files"][0]["file"]).parent.name)
    assert tas.attrs["variable_id"] == "tas" and tas.attrs["variant_label"] == "r2i1p1f1"
    assert tas.attrs["Conventions"] == "CF-1.7" and "PrePARE" in tas.attrs["aimip_missing"]
    assert float(tas["height"]) == 2.0 and tas["tas"].attrs["cell_methods"] == "area: time: mean"
    assert tas["tas"].dims == ("time", "lat", "lon")
    raw = xr.open_dataset(src, decode_times=False)
    np.testing.assert_allclose(tas["tas"].values, raw["TMP2m"].isel(sample=1, time=[1, 2]).values)
    # time is the middle of the month, bounds the calendar month, Dec -> Jan crossing
    b, t = tas["time_bnds"].values, tas["time"].values
    assert (b[0, 0].month, b[0, 0].day) == (11, 1) and (b[1, 1].year, b[1, 1].month, b[1, 1].day) == (1979, 1, 1)
    assert (t[0].month, t[0].day, t[0].hour) == (11, 16, 0) and (t[1].month, t[1].day, t[1].hour) == (12, 16, 12)
    raw_t = xr.open_dataset(next(f["file"] for f in res["files"] if f["variable"] == "tas"), decode_times=False)
    assert raw_t["time"].dtype == np.float64
    # CMIP encoding; `coordinates` only on the variable (cdo: "Inconsistent variable definition" on time_bnds)
    import netCDF4
    with netCDF4.Dataset(next(f["file"] for f in res["files"] if f["variable"] == "tas")) as nc:
        assert nc["tas"].getncattr("_FillValue") == np.float32(1e20) and nc["tas"].getncattr("coordinates") == "height"
        for v in ("time", "time_bnds", "lat", "lon", "height"):
            assert "_FillValue" not in nc[v].ncattrs() and "coordinates" not in nc[v].ncattrs(), v
    assert "LEFT OUT 1978-10" in tas.attrs["aimip_postprocessing"]
    hfls = _one(res, "hfls")
    assert hfls["hfls"].attrs["positive"] == "up" and hfls["hfls"].attrs["units"] == "W m-2"
    # zg500 and zg1000 in ONE zg file, plev descending, values / g, conversion stated
    zg = _one(res, "zg")
    assert zg["zg"].dims == ("time", "plev", "lat", "lon") and list(zg["plev"].values) == [100000.0, 50000.0]
    np.testing.assert_allclose(zg["zg"].sel(plev=50000.0).values, 55000.0 / GRAVITY, rtol=1e-6)
    np.testing.assert_allclose(zg["zg"].sel(plev=100000.0).values, 1000.0 / GRAVITY, rtol=1e-6)
    post = zg.attrs["aimip_postprocessing"]
    assert "CONVERTED z500" in post and "CONVERTED z1000" in post and "9.80665" in post
    assert "2 of the 19 plev19 levels" in post
    # nothing invented: unknown units and model-layer fields are not written, the reason is returned
    assert "Q2m" in res["skipped"] and "furlongs" in res["skipped"]["Q2m"]
    assert not any(f["from"] in ("Q2m", "eastward_wind_1") for f in res["files"])


def test_export_cmor_attributes_match_the_amon_table(allow_tmp_run_root: Path) -> None:
    run = allow_tmp_run_root / "attrs"
    extra = {n: (("sample", "time", "lat", "lon"), np.ones((2, 3, 2, 2)), {"units": u}) for n, u in (
        ("Q2m", "kg/kg"), ("UGRD10m", "m/s"), ("VGRD10m", "m/s"), ("PRESsfc", "Pa"), ("surface_temperature", "K"),
        ("PRATEsfc", "kg/m**2/s"), ("SHTFLsfc", "W/m**2"), ("DLWRFsfc", "W/m**2"), ("ULWRFsfc", "W/m**2"),
        ("DSWRFsfc", "W/m**2"), ("USWRFsfc", "W/m**2"), ("ULWRFtoa", "W/m**2"), ("USWRFtoa", "W/m**2"),
        ("DSWRFtoa", "W/m**2"), ("tcwv", "kg/m**2"), ("psl", "Pa"))}
    _run(run / "monthly_mean_predictions.nc", counts=[124, 120, 124], extra=extra)
    res = export_cmor(str(run), out_dir=str(run / "cmor"), members=[0], **KW)
    assert res["notes"] == [FULL6H]
    assert {f["variable"] for f in res["files"]} == set(_AMON) | {"zg"}
    for var, (long_name, cell_methods, positive, height) in _AMON.items():
        a = _one(res, var)[var].attrs
        assert (a["long_name"], a["cell_methods"]) == (long_name, cell_methods), var
        assert a.get("positive", "") == positive, var
        if height is not None:
            assert float(_one(res, var)["height"]) == height, var


def test_export_cmor_calendars_year_one_and_same_name(allow_tmp_run_root: Path) -> None:
    run = allow_tmp_run_root / "noleap"
    _run(run / "monthly_mean_predictions.nc", months=[(1, 1), (1, 2), (1, 3)], counts=[1, 1, 1], cal="noleap",
         extra={"tas": (("sample", "time", "lat", "lon"), np.ones((2, 3, 2, 2)), {"units": "K"})},
         attrs={"monthly_representation": "single monthly model state per step"})
    with pytest.raises(ValidationError, match="Two variables"):  # TMP2m and tas both map to tas: refuse to choose
        export_cmor(str(run), out_dir=str(run / "cmor"), **{**KW, "model": "md-1.5"})
    run2 = allow_tmp_run_root / "samename"
    _run(run2 / "monthly_mean_predictions.nc", months=[(1, 1), (1, 2), (1, 3)], counts=[1, 1, 1], cal="noleap",
         attrs={"monthly_representation": "single monthly model state per step (not an average of sub-monthly steps)"})
    ds = xr.open_dataset(run2 / "monthly_mean_predictions.nc").load()
    ds = ds.rename({"TMP2m": "tas"})  # a model that already writes CMIP names (ArchesClimate-SSP)
    (run2 / "monthly_mean_predictions.nc").unlink()
    ds.to_netcdf(run2 / "monthly_mean_predictions.nc")
    res = export_cmor(str(run2), out_dir=str(run2 / "cmor"), **{**KW, "model": None})  # the file states the step
    assert res["notes"][0].startswith("monthly-step model (file attribute monthly_representation")
    tas = _one(res, "tas")
    assert tas.attrs["aimip_postprocessing"].count("from model variable tas") == 1
    assert tas["time_bnds"].values[1, 1].month == 3 and tas["time"].values[0].calendar == "noleap"
    assert Path(next(f["file"] for f in res["files"] if f["variable"] == "tas")).name.endswith("_000101-000103.nc")


def test_export_cmor_refuses_guessing(allow_tmp_run_root: Path) -> None:
    run = allow_tmp_run_root / "nodates"
    _run(run / "monthly_mean_predictions.nc", with_dates=False)
    with pytest.raises(ValidationError, match="no valid_time"):
        export_cmor(str(run), out_dir=str(run / "cmor"), **KW)
    assert not any((run / "cmor").rglob("*.nc")) if (run / "cmor").exists() else True
    gap = allow_tmp_run_root / "gap"
    _run(gap / "monthly_mean_predictions.nc", months=[(2001, 1), (2001, 2), (2001, 4)], counts=[124, 112, 120])
    with pytest.raises(ValidationError, match="consecutive months"):
        export_cmor(str(gap), out_dir=str(gap / "cmor"), **KW)
    mid = allow_tmp_run_root / "mid"
    _run(mid / "monthly_mean_predictions.nc", months=[(2001, 1), (2001, 2), (2001, 3)], counts=[124, 50, 124])
    with pytest.raises(ValidationError, match="partial month inside"):
        export_cmor(str(mid), out_dir=str(mid / "cmor"), **KW)
    ok = allow_tmp_run_root / "ok"
    _run(ok / "monthly_mean_predictions.nc")
    for bad in ("../x", "a b", ""):
        with pytest.raises(ValidationError):
            export_cmor(str(ok), out_dir=str(ok / "cmor"), **{**KW, "source_id": bad})
    with pytest.raises(ValidationError):
        export_cmor(str(ok), out_dir=str(ok / "cmor"), **{**KW, "grid_label": "native"})
    with pytest.raises(ValidationError):
        export_cmor(str(ok), out_dir=str(ok / "cmor"), members=[2], **KW)
    with pytest.raises(ValidationError):
        export_cmor(str(ok), out_dir="/etc/aimip-cmor-test", **KW)


def test_export_cmor_ocean_component_not_written_as_amon(allow_tmp_run_root: Path) -> None:
    run = allow_tmp_run_root / "coupled"
    _run(run / "atmosphere" / "monthly_mean_predictions.nc")
    _run(run / "ocean" / "monthly_mean_predictions.nc")
    res = export_cmor(str(run), out_dir=str(run / "cmor"), **KW)
    assert all(f["component"] == "atmosphere" for f in res["files"])
    assert "ocean" in next(iter(k for k in res["skipped"] if k.startswith("ocean")))


def test_export_cmor_cli_prints_warnings_to_terminal(allow_tmp_run_root: Path,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    from aimip.cli import main

    run = allow_tmp_run_root / "clirun"
    _run(run / "monthly_mean_predictions.nc")
    args = ["export-cmor", str(run), "--source-id", "ACE2-ERA5", "--experiment-id", "amip", "--grid-label", "gn",
            "--institution-id", "MY-INST", "--model", "ace2-era5"]
    for extra in ([], ["--json"]):
        assert main([*args, *extra]) == 0
        err = capsys.readouterr().err
        assert "WARNING: zg: CONVERTED z500 geopotential" in err and "WARNING: zg: CONVERTED z1000" in err
        assert "WARNING: LEFT OUT 1978-10: partial month" in err
        assert "WARNING: Q2m not written (units 'furlongs'" in err
        assert "not validated by the CMOR library" in err
        assert "WARNING: not written, no exact CMIP6 Amon equivalent: eastward_wind_1" in err
    try:  # --experiment-id and --grid-label are required (argparse exits, or main returns its code)
        rc = main(["export-cmor", str(run), "--source-id", "ACE2-ERA5"])
    except SystemExit as exc:
        rc = exc.code
    assert rc not in (0, None)


def test_export_cmor_short_run_without_a_confirmed_full_month(allow_tmp_run_root: Path) -> None:
    run = allow_tmp_run_root / "eightsteps"
    _run(run / "monthly_mean_predictions.nc", months=[(1979, 1)], counts=[8])  # the documented 8-step first test
    res = export_cmor(str(run), out_dir=str(run / "cmor"), **KW)
    assert res["files"] == [] and "member 0" in res["skipped"]
    assert "LEFT OUT 1979-01: partial month (8 of 124 model states)" in res["notes"]
    two = allow_tmp_run_root / "twopartial"
    _run(two / "monthly_mean_predictions.nc", months=[(1979, 1), (1979, 2)], counts=[60, 50])
    assert export_cmor(str(two), out_dir=str(two / "cmor"), **KW)["files"] == []


def test_export_cmor_counts_of_one_need_a_stated_monthly_step(allow_tmp_run_root: Path) -> None:
    one = allow_tmp_run_root / "onestep"  # one step of a 6-hourly model also has counts = 1
    _run(one / "monthly_mean_predictions.nc", months=[(1979, 1)], counts=[1])
    res = export_cmor(str(one), out_dir=str(one / "cmor"), **KW)
    assert res["files"] == [] and "LEFT OUT 1979-01: partial month (1 of 124 model states)" in res["notes"]
    md = allow_tmp_run_root / "md15"
    rep = "single monthly model state per step (not an average of sub-monthly steps)"
    tas = (("sample", "time", "lat", "lon"), np.ones((2, 1, 2, 2)),
           {"units": "K", "cell_methods": "time: point (single monthly latent-diffusion state)"})
    _run(md / "monthly_mean_predictions.nc", months=[(1979, 1)], counts=[1], attrs={"monthly_representation": rep},
         extra={"TMP2m": tas})
    res = export_cmor(str(md), out_dir=str(md / "cmor"), members=[0], **{**KW, "model": None})
    out = _one(res, "tas")
    assert out["tas"].attrs["cell_methods"] == "time: point (single monthly latent-diffusion state)"
    assert "not a time mean" in out.attrs["aimip_postprocessing"]
    # without a stated step nothing is written
    none = allow_tmp_run_root / "nostep"
    _run(none / "monthly_mean_predictions.nc", months=[(1979, 1)], counts=[1])
    res = export_cmor(str(none), out_dir=str(none / "cmor"), **{**KW, "model": None})
    assert res["files"] == [] and res["notes"][0].startswith("model time step unknown")
    # the model recorded in the run gives the registry step (ArchesClimate-SSP, MD-1.5: 730 h)
    reg = allow_tmp_run_root / "regmonthly"
    _run(reg / "monthly_mean_predictions.nc", months=[(2015, 3)], counts=[1])
    (reg / "aimip-manifest.json").write_text('{"model": "archesclimate-ssp"}')
    res = export_cmor(str(reg), out_dir=str(reg / "cmor"), members=[0], **{**KW, "model": None})
    monthly = ("monthly-step model (current registry entry of archesclimate-ssp (the run record has no time step)): "
               "one model state per month")
    assert res["files"] and monthly in res["notes"]
    assert res["files"][0]["file"].count("/") and _one(res, "tas").attrs["aimip_model"] == "archesclimate-ssp"


def test_export_cmor_keep_first_month_only_when_asked_and_only_the_initial_state_missing(allow_tmp_run_root: Path) -> None:
    run = allow_tmp_run_root / "first"
    _run(run / "monthly_mean_predictions.nc", months=[(2001, 1), (2001, 2), (2001, 3)], counts=[123, 112, 1])
    res = export_cmor(str(run), out_dir=str(run / "cmor"), members=[0], **KW)
    assert "LEFT OUT 2001-01: partial month (123 of 124 model states)" in res["notes"]
    assert Path(res["files"][0]["file"]).name.endswith("_200102-200102.nc")
    res = export_cmor(str(run), out_dir=str(run / "cmor2"), members=[0], keep_first_month=True, **KW)
    kept = "KEPT 2001-01 with 123 of 124 model states (--keep-first-month); which state is missing is not verified"
    assert any(n.startswith(kept) for n in res["notes"]) and "LEFT OUT 2001-03" in " ".join(res["notes"])
    tas = _one(res, "tas")
    assert Path(next(f["file"] for f in res["files"] if f["variable"] == "tas")).name.endswith("_200101-200102.nc")
    assert kept in tas.attrs["aimip_postprocessing"]
    two = allow_tmp_run_root / "first2"  # two states missing: still left out
    _run(two / "monthly_mean_predictions.nc", months=[(2001, 1), (2001, 2)], counts=[122, 112])
    res = export_cmor(str(two), out_dir=str(two / "cmor"), members=[0], keep_first_month=True, **KW)
    assert "LEFT OUT 2001-01: partial month (122 of 124 model states)" in res["notes"]


def test_export_cmor_carbon_review_cases(allow_tmp_run_root: Path, tmp_path: Path) -> None:
    import os

    # SHiELD / HiRO layout: grid_yt / grid_xt become lat / lon (same values), labelled as degrees
    run = allow_tmp_run_root / "shield"
    _run(run / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    ds = xr.open_dataset(run / "monthly_mean_predictions.nc").load().rename({"lat": "grid_yt", "lon": "grid_xt"})
    (run / "monthly_mean_predictions.nc").unlink()
    ds.to_netcdf(run / "monthly_mean_predictions.nc")
    res = export_cmor(str(run), out_dir=str(run / "cmor"), members=[0], **KW)
    tas = _one(res, "tas")
    assert tas["tas"].dims == ("time", "lat", "lon") and tas["lat"].attrs["units"] == "degrees_north"
    assert "grid_yt -> lat" in tas.attrs["aimip_postprocessing"]
    # a run outside the run root is read (never written); output only under the run root
    outside = tmp_path / "shared-copy"
    _run(outside / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    before = sorted(p.name for p in outside.rglob("*"))
    res = export_cmor(str(outside), out_dir=str(allow_tmp_run_root / "from-shared"), members=[0], **KW)
    assert res["files"] and sorted(p.name for p in outside.rglob("*")) == before
    with pytest.raises(ValidationError):  # default <run>/cmor is outside the run root
        export_cmor(str(outside), out_dir=str(outside / "cmor"), **KW)
    # --to the run directory itself, and a symlinked --to, are refused
    with pytest.raises(ValidationError, match="run directory itself"):
        export_cmor(str(run), out_dir=str(run), **KW)
    os.symlink(run / "cmor", allow_tmp_run_root / "link")
    with pytest.raises(ValidationError, match="symlinked"):
        export_cmor(str(run), out_dir=str(allow_tmp_run_root / "link"), **KW)
    # counts larger than the model step allows: nothing written
    big = allow_tmp_run_root / "big"
    _run(big / "monthly_mean_predictions.nc", counts=[124, 240, 124])
    res = export_cmor(str(big), out_dir=str(big / "cmor"), **KW)
    assert res["files"] == [] and "do not match the model step" in res["notes"][0]
    with pytest.raises(ValidationError, match="empty"):
        export_cmor(str(run), out_dir=str(run / "cmor"), members=[], **KW)


def test_export_cmor_pre_1850_epoch(allow_tmp_run_root: Path) -> None:
    run = allow_tmp_run_root / "year1"
    rep = "single monthly model state per step (not an average of sub-monthly steps)"
    _run(run / "monthly_mean_predictions.nc", months=[(1, 1), (1, 2)], counts=[1, 1], cal="noleap",
         attrs={"monthly_representation": rep})
    res = export_cmor(str(run), out_dir=str(run / "cmor"), members=[0], **{**KW, "model": None})
    raw = xr.open_dataset(next(f["file"] for f in res["files"] if f["variable"] == "tas"), decode_times=False)
    assert raw["time"].attrs["units"].startswith("days since 0001-01-01") and float(raw["time"][0]) >= 0


def test_export_cmor_fable_x1_x3(allow_tmp_run_root: Path) -> None:
    # X1 SamudrACE: the registry step (120 h) is the ocean coupling; the atmosphere file is 6-hourly
    run = allow_tmp_run_root / "samudrace"
    _run(run / "atmosphere" / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    _run(run / "ocean" / "monthly_mean_predictions.nc", counts=[6, 6, 6])
    res = export_cmor(str(run), out_dir=str(run / "cmor"), members=[0], **{**KW, "model": "samudrace-cm4"})
    assert {f["component"] for f in res["files"]} == {"atmosphere"} and len({f["variable"] for f in res["files"]}) == 3
    assert any("atmosphere step of samudrace-cm4" in n for n in res["notes"])
    # X2 a one-month run with full-1 states is not kept: the missing state may be the last one
    one = allow_tmp_run_root / "onemonth"
    _run(one / "monthly_mean_predictions.nc", months=[(1978, 10)], counts=[123])
    res = export_cmor(str(one), out_dir=str(one / "cmor"), keep_first_month=True, **KW)
    assert res["files"] == [] and "LEFT OUT 1978-10: partial month (123 of 124 model states)" in res["notes"]
    # X3 the source time long_name (Spherical DYffusion's nominal cycle) is carried into the note
    sd = allow_tmp_run_root / "sdyff"
    _run(sd / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    ds = xr.open_dataset(sd / "monthly_mean_predictions.nc").load()
    ds["time"].attrs["long_name"] = "nominal cycle month, not a historical calendar year"
    (sd / "monthly_mean_predictions.nc").unlink()
    ds.to_netcdf(sd / "monthly_mean_predictions.nc")
    res = export_cmor(str(sd), out_dir=str(sd / "cmor"), members=[0], **KW)
    assert "source time long_name: nominal cycle month" in _one(res, "tas").attrs["aimip_postprocessing"]


def test_registry_step_without_data_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """The step is read from the registry file when the full loader needs $AIMIP_DATA (found on Levante)."""
    import aimip.registry as registry_mod
    from aimip.cmor_export import _registry_step
    from aimip.errors import RegistryError

    def _no_data(name: str, **_: object) -> object:
        raise RegistryError("contains ${DATA}, but neither $AIMIP_DATA nor $AIMIP_PLAYGROUND is set")

    monkeypatch.setattr(registry_mod, "get_model", _no_data)
    monkeypatch.setenv("AIMIP_REGISTRY", str(Path(registry_mod.__file__).parent / "registry.default.json"))
    assert _registry_step("ace2-era5") == 6 and _registry_step("md-1.5") == 730
    assert _registry_step("no-such-model") is None


def test_export_cmor_allow_unfinished_on_killed_run(
    allow_tmp_run_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json
    from aimip.cli import main

    killed = allow_tmp_run_root / "killed_cmor"
    _run(killed / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    (killed / "aimip-manifest.json").write_text(
        json.dumps({"model": "ace2-era5", "started_utc": "2026-11-20T00:00:00Z"}),
        encoding="utf-8",
    )
    (killed / "local.pid").write_text("99999999\n", encoding="utf-8")

    with pytest.raises(ValidationError) as exc_info:
        export_cmor(str(killed), out_dir=str(killed / "cmor"), **KW)
    msg = str(exc_info.value)
    assert f"aimip status {killed.resolve()} refreshes the state" in msg
    assert "--allow-unfinished" in msg and "export-cmor" in msg

    capsys.readouterr()
    assert main([
        "export-cmor", str(killed), "--source-id", "ACE2-ERA5", "--experiment-id", "amip",
        "--grid-label", "gn", "--institution-id", "MY-INST", "--allow-unfinished",
    ]) == 0
    err = capsys.readouterr().err
    assert "WARNING: run status is unfinished (not completed)" in err
    res = export_cmor(str(killed), out_dir=str(killed / "cmor2"), members=[0], allow_unfinished=True, **KW)
    assert "run status is unfinished (not completed)" in _one(res, "tas").attrs["aimip_postprocessing"]


def test_export_cmor_codex_final_review(allow_tmp_run_root: Path, tmp_path: Path) -> None:
    import json
    import os

    # 1: a symlink inside the DRS path (here the institution directory) is refused; nothing written through it
    run = allow_tmp_run_root / "drs"
    _run(run / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    elsewhere = tmp_path / "other-project"
    elsewhere.mkdir()
    (run / "cmor").mkdir()
    os.symlink(elsewhere, run / "cmor" / "MY-INST")
    with pytest.raises(ValidationError, match="is a symlink or not a directory"):
        export_cmor(str(run), out_dir=str(run / "cmor"), members=[0], **KW)
    assert not any(elsewhere.rglob("*"))
    (run / "cmor" / "MY-INST").unlink()
    # 2: an existing file at the old predictable temporary name is never touched
    res = export_cmor(str(run), out_dir=str(run / "cmor"), members=[0], **KW)
    target = Path(next(f["file"] for f in res["files"] if f["variable"] == "tas"))
    old_tmp = Path(str(target) + ".incomplete")
    old_tmp.write_text("someone else's", encoding="utf-8")
    export_cmor(str(run), out_dir=str(run / "cmor"), members=[0], **KW)
    assert old_tmp.read_text(encoding="utf-8") == "someone else's"
    # 3: coordinates in radians or degrees_west are refused, not relabelled
    for units in ("radians", "degrees_west"):
        bad = allow_tmp_run_root / f"coord-{units}"
        _run(bad / "monthly_mean_predictions.nc", counts=[124, 120, 124])
        ds = xr.open_dataset(bad / "monthly_mean_predictions.nc").load()
        ds["lon"].attrs["units"] = units
        (bad / "monthly_mean_predictions.nc").unlink()
        ds.to_netcdf(bad / "monthly_mean_predictions.nc")
        with pytest.raises(ValidationError, match="not plain degrees"):
            export_cmor(str(bad), out_dir=str(bad / "cmor"), members=[0], **KW)
    # 4: float64 values are cast to the float32 CMIP6 asks for, with the largest change stated
    f64 = allow_tmp_run_root / "f64"
    _run(f64 / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    ds = xr.open_dataset(f64 / "monthly_mean_predictions.nc").load()
    ds["TMP2m"] = ds["TMP2m"].astype("float64") + 1e-9
    (f64 / "monthly_mean_predictions.nc").unlink()
    ds.to_netcdf(f64 / "monthly_mean_predictions.nc")
    res = export_cmor(str(f64), out_dir=str(f64 / "cmor"), members=[0], **KW)
    tas = _one(res, "tas")
    assert tas["tas"].dtype == np.float32 and "CAST TMP2m from float64 to float32" in tas.attrs["aimip_postprocessing"]
    np.testing.assert_allclose(tas["tas"].values, ds["TMP2m"].isel(sample=0).values, rtol=1e-6)
    huge = allow_tmp_run_root / "huge"
    _run(huge / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    ds = xr.open_dataset(huge / "monthly_mean_predictions.nc").load()
    ds["TMP2m"] = ds["TMP2m"].astype("float64") * 1e300
    (huge / "monthly_mean_predictions.nc").unlink()
    ds.to_netcdf(huge / "monthly_mean_predictions.nc")
    with pytest.raises(ValidationError, match="beyond the float32 range"):
        export_cmor(str(huge), out_dir=str(huge / "cmor"), members=[0], **KW)
    # 5: the step recorded in the run wins over today's registry; a contradicting --model is refused
    rec = allow_tmp_run_root / "recorded"
    _run(rec / "monthly_mean_predictions.nc", counts=[248, 240, 248])  # a 3-hourly run
    (rec / "aimip-manifest.json").write_text(json.dumps(
        {"registry_entry": {"name": "ace2-era5", "timestep_hours": 3}, "finished": "x", "rc": 0}))
    res = export_cmor(str(rec), out_dir=str(rec / "cmor"), members=[0], **{**KW, "model": None})
    assert res["files"] and "full month = 8 model states per day (time step 3 h, registry entry of ace2-era5 " \
        "recorded in the run)" in res["notes"]
    with pytest.raises(ValidationError, match="contradicts the run record"):
        export_cmor(str(rec), out_dir=str(rec / "cmor"), members=[0], **{**KW, "model": "ace2-shield"})
    # 6: counts that are not whole numbers write nothing; the kept first month does not claim which state is missing
    frac = allow_tmp_run_root / "frac"
    _run(frac / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    ds = xr.open_dataset(frac / "monthly_mean_predictions.nc").load()
    ds["counts"] = ds["counts"].astype("float64") + 0.5
    (frac / "monthly_mean_predictions.nc").unlink()
    ds.to_netcdf(frac / "monthly_mean_predictions.nc")
    res = export_cmor(str(frac), out_dir=str(frac / "cmor"), **KW)
    assert res["files"] == [] and res["notes"][0].startswith("counts are not one whole")
    first = allow_tmp_run_root / "first-unverified"
    _run(first / "monthly_mean_predictions.nc", months=[(2001, 1), (2001, 2)], counts=[123, 112])
    res = export_cmor(str(first), out_dir=str(first / "cmor"), members=[0], keep_first_month=True, **KW)
    assert any("which state is missing is not verified" in n for n in res["notes"])
    # 7: a run with both a top-level and an atmosphere monthly file is refused before any write
    both = allow_tmp_run_root / "both"
    _run(both / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    _run(both / "atmosphere" / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    with pytest.raises(ValidationError, match="both monthly_mean_predictions.nc and atmosphere"):
        export_cmor(str(both), out_dir=str(both / "cmor"), **KW)
    assert not (both / "cmor").exists()


def test_export_cmor_codex_recheck(allow_tmp_run_root: Path) -> None:
    import json

    # 4: the rounding of a float64 geopotential is reported in metres (the written units)
    run = allow_tmp_run_root / "z64"
    _run(run / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    ds = xr.open_dataset(run / "monthly_mean_predictions.nc").load()
    ds["z500"] = ds["z500"].astype("float64") + 0.123456789
    (run / "monthly_mean_predictions.nc").unlink()
    ds.to_netcdf(run / "monthly_mean_predictions.nc")
    res = export_cmor(str(run), out_dir=str(run / "cmor"), members=[0], **KW)
    post = _one(res, "zg").attrs["aimip_postprocessing"]
    assert "CAST z500 from float64 to float32" in post and " m (relative" in post
    # 5: a --model that contradicts a plan that records only its top-level model is refused
    plan = allow_tmp_run_root / "planonly"
    _run(plan / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    (plan / "plan.json").write_text(json.dumps({"model": "ace2-shield"}))
    with pytest.raises(ValidationError, match="contradicts the run record"):
        export_cmor(str(plan), out_dir=str(plan / "cmor"), **KW)
    # 6: counts indexed by something other than the months are refused
    lat = allow_tmp_run_root / "countslat"
    _run(lat / "monthly_mean_predictions.nc", months=[(1978, 11), (1978, 12)], counts=[120, 124])
    ds = xr.open_dataset(lat / "monthly_mean_predictions.nc").load()
    ds["counts"] = (("sample", "lat"), np.array([[120, 124]] * 2, dtype="int32"))
    (lat / "monthly_mean_predictions.nc").unlink()
    ds.to_netcdf(lat / "monthly_mean_predictions.nc")
    with pytest.raises(ValidationError, match="counts must be one value per month"):
        export_cmor(str(lat), out_dir=str(lat / "cmor"), **KW)


def test_export_cmor_all_recorded_identities(allow_tmp_run_root: Path) -> None:
    import json

    run = allow_tmp_run_root / "mixed"
    _run(run / "monthly_mean_predictions.nc", counts=[124, 120, 124])
    (run / "aimip-manifest.json").write_text(json.dumps(
        {"registry_entry": {"name": "ace2-era5", "timestep_hours": 6}, "finished": "x", "rc": 0}))
    (run / "plan.json").write_text(json.dumps({"model": "ace2-shield"}))
    with pytest.raises(ValidationError, match="contradicts the run record"):  # the manifest matches, the plan does not
        export_cmor(str(run), out_dir=str(run / "cmor"), **KW)
    with pytest.raises(ValidationError, match="records disagree"):
        export_cmor(str(run), out_dir=str(run / "cmor"), **{**KW, "model": None})


def test_write_pinned_refuses_a_foreign_staging_directory_and_deletes_nothing(tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    import aimip.cf_export as cf

    root = tmp_path / "out"
    root.mkdir()
    (root / "f.nc").write_text("existing result", encoding="utf-8")
    real_getuid = os.getuid
    monkeypatch.setattr(cf.os, "getuid", lambda: real_getuid() + 1)  # the staging directory looks foreign
    with pytest.raises(ValidationError, match="not private"):
        cf._write_pinned(lambda p: Path(p).write_text("new", encoding="utf-8"), str(root), [], "f.nc")
    assert (root / "f.nc").read_text(encoding="utf-8") == "existing result"
    monkeypatch.setattr(cf.os, "getuid", real_getuid)
    if os.path.isdir("/proc/self/fd"):
        cf._write_pinned(lambda p: Path(p).write_text("new", encoding="utf-8"), str(root), [], "f.nc")
        assert (root / "f.nc").read_text(encoding="utf-8") == "new"
        assert not [p for p in root.iterdir() if p.name.endswith(".incomplete") and any(p.iterdir())]


def test_write_pinned_removes_its_staging_directory_only_where_entries_are_protected(tmp_path: Path) -> None:
    import os

    import aimip.cf_export as cf

    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("Linux only")
    mine = tmp_path / "mine"
    mine.mkdir(mode=0o755)
    cf._write_pinned(lambda p: Path(p).write_text("x", encoding="utf-8"), str(mine), [], "a.nc")
    assert sorted(p.name for p in mine.iterdir()) == ["a.nc"]
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o775)  # group-writable, not sticky: another user could swap the entry, so it is left (empty)
    cf._write_pinned(lambda p: Path(p).write_text("x", encoding="utf-8"), str(shared), [], "a.nc")
    left = [p for p in shared.iterdir() if p.name != "a.nc"]
    assert (shared / "a.nc").read_text(encoding="utf-8") == "x"
    assert all(p.name.endswith(".incomplete") and p.is_dir() and not any(p.iterdir()) for p in left)
