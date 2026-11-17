"""md-1.5 (Monthly Diffusion v0.9): pinned files, request rules, job binding, the job's interpreter, and the
forcing regridding (no torch: the job interpreter is faked, the regridder is numpy)."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from aimip.emu_catalog import (AIMIP_SST_REL, MD_CONFIG, MD_IC, MD_INPUT_FILES, MD_STATS, MD_WEIGHTS, MD_ZIP_MEMBERS,
                               MD_ZIP_REL, registry_entries, required_files)
from aimip.errors import ConfigError, ValidationError

from conftest import sha256_path
from test_emu import _prepare, emu_env  # noqa: F401  (fixture)


@pytest.fixture
def md_env(emu_env, monkeypatch):
    monkeypatch.setenv("AIMIP_MD_PYTHON", str(pathlib.Path(emu_env["pg"]).parent / "fakepy"))
    return emu_env


def test_catalog_and_lock_pin_code_checkpoint_inputs_and_forcing():
    from aimip import setup_data

    rels = required_files("md-1.5")
    assert set(MD_ZIP_MEMBERS) | set(MD_INPUT_FILES) | {AIMIP_SST_REL} == set(rels)
    assert MD_WEIGHTS in rels and MD_CONFIG in rels and "models/md-1.5/code/md/cvae.py" in rels
    lock = json.loads((pathlib.Path(setup_data.__file__).parent / "aimip.lock").read_text())["files"]
    assert lock[MD_ZIP_REL]["source"]["kind"] == "zenodo" and lock[MD_ZIP_REL]["source"]["record"] == "21430292"
    for rel, (_member, size, sha) in MD_ZIP_MEMBERS.items():
        assert lock[rel]["sha256"] == sha and lock[rel]["size"] == size
        assert lock[rel]["source"] == {"kind": "derived", "builder": "extract_zip_member",
                                       "inputs": {MD_ZIP_REL: lock[MD_ZIP_REL]["sha256"]}}
    for rel, (fname, size, md5, sha) in MD_INPUT_FILES.items():
        assert lock[rel]["sha256"] == sha and lock[rel]["source"]["file"] == fname and lock[rel]["source"]["md5"] == md5
    cat = setup_data.build_artifact_catalog()
    assert cat[MD_ZIP_REL].role == "raw" and cat[MD_WEIGHTS].builder == "extract_zip_member"
    assert "md-1.5" in cat[AIMIP_SST_REL].models_needed
    entry = next(e for e in registry_entries("${DATA}") if e["name"] == "md-1.5")
    assert entry["family"] == "md" and entry["stochastic"] and entry["checkpoint"].endswith(MD_WEIGHTS)


def test_request_rules(md_env):
    from aimip.emu import precheck_emu_request

    m = md_env["specs"]["md-1.5"]
    ok = precheck_emu_request(m, start_iso="1978-10-01T00:00:00", steps=554, members=5, forcing_set_name=None,
                              ic_path=None)
    assert ok["end"] == "2024-12-01T00:00:00" and ok["members"] == 5 and ok["scenario"] == "aimip"
    assert precheck_emu_request(m, start_iso="1978-10-01T00:00:00", steps=3, members=None, forcing_set_name="aimip",
                                ic_path=None)["end"] == "1979-01-01T00:00:00"
    for kw, msg in (
        (dict(start_iso="1979-01-01T00:00:00", steps=3), "October 1978 only"),
        (dict(start_iso="1978-10-01T00:00:00", steps=555), "at most 554 monthly steps"),
        (dict(start_iso="1978-10-01T00:00:00", steps=3, forcing_set_name="ssp585"), "Unknown --forcing-set"),
        (dict(start_iso="1978-10-01T00:00:00", steps=3, ic_path="/tmp/x"), "--ic is not supported"),
    ):
        args = dict(members=None, forcing_set_name=None, ic_path=None)
        args.update(kw)
        with pytest.raises(ConfigError, match=msg):
            precheck_emu_request(m, **args)


def test_horizon_by_length_counts_months():
    from aimip.time_utils import resolve_horizon

    assert resolve_horizon(start_iso="1978-10-01T00:00:00", steps=None, length="1y", end_iso=None, timestep_hours=730,
                           calendar="proleptic_gregorian", family="md", model_name="md-1.5") == (12, "1979-10-01T00:00:00")


def test_prepare_binds_inputs_and_runs_in_the_main_environment(md_env):
    from aimip.runner import exec_fme

    out = md_env["runs"] / "md"
    m = _prepare(md_env, "md-1.5", out, start_iso="1978-10-01T00:00:00", steps=12, members=2, seed=4)
    data = md_env["data"]
    job = json.loads((out / "config.yaml").read_text())
    assert job["code_dir"] == str(data / "models/md-1.5/code") and job["ic_file"] == str(data / MD_IC)
    assert job["stats_file"] == str(data / MD_STATS) and job["sst_forcing"] == str(data / AIMIP_SST_REL)
    assert m["ic_path"] == str(data / MD_IC) and m["ic_sha256"] == sha256_path(data / MD_IC)
    assert m["runtime_python"].endswith("fakepy") and m["member_seeds"] == [4, 5] and m["end_time"] == "1979-10-01T00:00:00"
    assert exec_fme(str(out))["rc"] == 0

    for name, key, value in (("m1", "ic_file", "/elsewhere/ic.nc"), ("m2", "code_dir", "/elsewhere/code"),
                             ("m3", "stats_file", str(data / MD_IC))):
        o = md_env["runs"] / name
        _prepare(md_env, "md-1.5", o, start_iso="1978-10-01T00:00:00", steps=2)
        j = json.loads((o / "config.yaml").read_text())
        j[key] = value
        (o / "config.yaml").write_text(json.dumps(j))
        with pytest.raises(ValidationError):
            exec_fme(str(o))
    o = md_env["runs"] / "m4"
    _prepare(md_env, "md-1.5", o, start_iso="1978-10-01T00:00:00", steps=2)
    (data / MD_STATS).write_text("{}")
    with pytest.raises(ValidationError, match="SHA-256 mismatch at job time"):
        exec_fme(str(o))


def _tiny_forcing(path, months=3, hole_month=None):
    import netCDF4

    lat, lon = np.linspace(90, -90, 721), np.arange(1440) * 0.25
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("time", months)
        d.createDimension("latitude", 721)
        d.createDimension("longitude", 1440)
        for n, v in (("latitude", lat), ("longitude", lon)):
            d.createVariable(n, "f4", (n,))[:] = v
        t = d.createVariable("time", "f8", ("time",))
        t.units, t.calendar = "days since 1978-10-01", "proleptic_gregorian"
        t[:] = [0, 31, 61][:months]
        land = (lat[:, None] > 0) & (lon[None, :] < 90)          # one land block
        lsm = d.createVariable("land_sea_mask", "f4", ("latitude", "longitude"))
        lsm[:] = land.astype("f4")
        for name, value in (("sea_surface_temperature", 290.0), ("sea_ice_cover", 0.0)):
            v = d.createVariable(name, "f4", ("time", "latitude", "longitude"), fill_value=np.nan)
            for i in range(months):
                f = np.where(land, np.nan, value).astype("f4")
                if i == hole_month:
                    f[400, 1000] = np.nan                            # one missing ocean cell in one month
                v[i] = f


def test_forcing_regridding_is_conservative_and_keeps_nans_local(tmp_path):
    from aimip.emu_job import MDForcing, _md_edges, _md_overlap

    e = _md_edges(np.linspace(-90, 90, 121), (-90, 90))
    assert e[0] == -90 and e[-1] == 90 and np.allclose(np.diff(e)[1:-1], 1.5)
    w = _md_overlap(_md_edges(np.arange(0, 360, 1.5), (0, 360)), _md_edges(np.arange(1440) * 0.25, (0, 360)), lambda x: x)
    assert np.allclose(w.sum(axis=1)[1:-1], 1.5)      # interior cells covered; the authors' clipped edges at 0/360
    _tiny_forcing(tmp_path / "f.nc", hole_month=1)
    f = MDForcing(str(tmp_path / "f.nc"))
    x = f.fields(["1978-10", "1978-11"])
    sst0, sst1 = x[0, 0], x[1, 0]
    ocean = np.isfinite(sst0)
    assert np.allclose(sst0[ocean], 290.0) and 0 < (~ocean).sum() < 0.2 * ocean.size
    assert np.isnan(sst1).sum() - np.isnan(sst0).sum() in (1, 2, 4)   # only the cells overlapping the hole
    assert np.all((x[0, 2] >= 0) & (x[0, 2] <= 1)) and np.isfinite(x[0, 2]).all()
