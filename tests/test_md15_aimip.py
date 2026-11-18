"""md-1.5-aimip: MD-1.5 as run for AIMIP Phase 1 (MD-1p5.pth with the architecture its tensors imply, rebuilt 1985-2014
statistics and October 1978 state, conservatively regridded forcing). Pinned files, request rules, the job's binding
of the variant, and the forcing regridding (no torch, no network: the emulator interpreter is faked)."""
from __future__ import annotations

import json
import pathlib

import netCDF4
import numpy as np
import pytest

from aimip.emu_catalog import AIMIP_SST_REL, registry_entries, required_files
from aimip.errors import ConfigError, ValidationError

from aimip.safe_pickle import compute_sha256sum_tree_digest
from test_emu import _prepare, emu_env  # noqa: F401  (fixture)

IC_NAME = "md15_era5_ic_1978-10-01_1p5.nc"
STATS_NAME = "md15_variable_statistics_198501-201412.json"


def test_catalog_pins_the_aimip_checkpoint_and_inputs():
    from aimip import md15, setup_data

    rels = required_files("md-1.5-aimip")
    assert sorted(rels) == sorted([
        AIMIP_SST_REL,
        md15.MD15_AIMIP_CONFIG_REL_PATH,
        md15.MD15_AIMIP_WEIGHTS_REL_PATH,
        f"inputs/md-1.5-aimip/{IC_NAME}",
        f"inputs/md-1.5-aimip/{STATS_NAME}",
    ])
    assert not set(rels) & {md15.MD15_WEIGHTS_REL_PATH, md15.MD15_STATS_REL_PATH}   # not md-1.5's files
    lock = json.loads((pathlib.Path(setup_data.__file__).parent / "aimip.lock").read_text())["files"]
    latest = lock[md15.MD15_WEIGHTS_REL_PATH]["source"]
    for rel, sha in ((md15.MD15_AIMIP_CONFIG_REL_PATH, md15.MD15_AIMIP_CONFIG_SHA256),
                     (md15.MD15_AIMIP_WEIGHTS_REL_PATH, md15.MD15_AIMIP_WEIGHTS_SHA256)):
        src = lock[rel]["source"]   # from the same pinned archive as md-1.5
        assert lock[rel]["sha256"] == sha and src["kind"] == "github_archive"
        assert (src["archive_sha256"], src["revision"]) == (latest["archive_sha256"], latest["revision"])
        assert src["path"] == "models/MD-1p5.pth/" + rel.rsplit("/", 1)[1]
    for name in (IC_NAME, STATS_NAME):
        src = lock[f"inputs/md-1.5-aimip/{name}"]["source"]
        assert src["kind"] == "zenodo" and src["record"] == md15.MD15_AIMIP_INPUTS_RECORD and len(src["md5"]) == 32
        assert src["url"] == f"https://zenodo.org/records/{md15.MD15_AIMIP_INPUTS_RECORD}/files/{name}?download=1"
    entry = next(e for e in registry_entries("${DATA}") if e["name"] == "md-1.5-aimip")
    assert entry["family"] == "md" and entry["stochastic"] and entry["checkpoint_sha256"] == md15.MD15_AIMIP_WEIGHTS_SHA256
    assert entry["ic_files"] == [f"${{DATA}}/inputs/md-1.5-aimip/{IC_NAME}"]
    spec = setup_data.build_artifact_catalog()[AIMIP_SST_REL]
    assert "md-1.5-aimip" in spec.models_needed and "md-1.5-aimip" in spec.models_output

    v, plain = md15.md15_variant("md-1.5-aimip"), md15.md15_variant("md-1.5")
    assert v["arch"] == md15.MD15_AIMIP_ARCH and v["forcing_regrid"] == "conservative"
    assert v["published_starts"] == ("1978-10-01T00:00:00",) and v["stats_rel"].endswith(STATS_NAME)
    assert plain["arch"] is None and plain["forcing_regrid"] == "coincident" and plain["stats_rel"] == md15.MD15_STATS_REL_PATH


@pytest.fixture
def md_env(emu_env, monkeypatch):
    """emu_env plus a 37-file stand-in for the pinned `code/monthly-diffusion` tree."""
    code_dir = emu_env["data"] / "code" / "monthly-diffusion" / "md"
    code_dir.mkdir(parents=True)
    for i in range(37):
        (code_dir / f"f_{i:02d}.py").write_text(f"# {i}\n")
    tree_sha, n = compute_sha256sum_tree_digest(str(code_dir.parent))
    assert n == 37
    monkeypatch.setattr("aimip.md15.MD15_TREE_SHA256", tree_sha)
    return emu_env


def test_request_rules(md_env):
    from aimip.emu import precheck_emu_request

    m = md_env["specs"]["md-1.5-aimip"]
    ok = precheck_emu_request(m, start_iso="1978-10-01T00:00:00", steps=555, members=5, forcing_set_name=None,
                              ic_path=None)
    assert ok["end"] == "2024-12-01T00:00:00" and ok["members"] == 5 and ok["scenario"] == "aimip-era5-monthly"
    for kw, msg in (
        (dict(start_iso="1979-01-01T00:00:00"), "no published prebuilt initial state"),   # md-1.5's other presets
        (dict(steps=557), "at most 556 monthly steps"),
        (dict(start_iso="1978-10-15T00:00:00"), "YYYY-MM-01T00:00:00"),
    ):
        args = dict(start_iso="1978-10-01T00:00:00", steps=12, members=1, forcing_set_name=None, ic_path=None)
        args.update(kw)
        with pytest.raises(ConfigError, match=msg):
            precheck_emu_request(m, **args)


def test_prepare_binds_the_variant(md_env):
    from aimip import md15
    from aimip.runner import exec_fme

    data = md_env["data"]
    out = md_env["runs"] / "ma"
    m = _prepare(md_env, "md-1.5-aimip", out, start_iso="1978-10-01T00:00:00", steps=12, members=2)
    job = json.loads((out / "config.yaml").read_text())
    assert job["arch"] == md15.MD15_AIMIP_ARCH and job["forcing_regrid"] == "conservative"
    assert job["stats_file"] == str(data / md15.MD15_AIMIP_STATS_REL_PATH)
    assert job["ckpt_dir"] == str(data / "models/md-1.5/MD-1p5.pth")
    assert job["ic_file"] == str(data / "inputs/md-1.5-aimip" / IC_NAME) == m["ic_path"]
    assert exec_fme(str(out))["rc"] == 0

    plain = md_env["runs"] / "mp"
    own_ic = md_env["runs"] / "own_ic.nc"
    own_ic.write_bytes(b"an initial state")
    _prepare(md_env, "md-1.5", plain, start_iso="1978-10-01T00:00:00", steps=12, members=2, ic_path=str(own_ic))
    pj = json.loads((plain / "config.yaml").read_text())
    assert pj["arch"] is None and pj["forcing_regrid"] == "coincident" and pj["stats_file"].endswith("197901-201412.json")

    # a job file that drops the architecture, switches the regridding or swaps in md-1.5's statistics is refused
    for name, change in (
        ("t1", lambda j: j.update(arch=None)),
        ("t2", lambda j: j.update(arch=dict(md15.MD15_AIMIP_ARCH, decoder_rank=128))),
        ("t3", lambda j: j.update(forcing_regrid="coincident")),
        ("t4", lambda j: j.update(stats_file=str(data / md15.MD15_STATS_REL_PATH),
                                  stats_sha256=pj["stats_sha256"])),
    ):
        o = md_env["runs"] / name
        _prepare(md_env, "md-1.5-aimip", o, start_iso="1978-10-01T00:00:00", steps=12, members=2)
        j = json.loads((o / "config.yaml").read_text())
        change(j)
        (o / "config.yaml").write_text(json.dumps(j))
        with pytest.raises(ValidationError, match="Integrity check failed"):
            exec_fme(str(o))


def _write_forcing(path: pathlib.Path, months: list[str], sst: np.ndarray, sic: np.ndarray, lsm: np.ndarray,
                   lat: np.ndarray, lon: np.ndarray) -> None:
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("time", len(months))
        d.createDimension("latitude", lat.size)
        d.createDimension("longitude", lon.size)
        t = d.createVariable("time", "i8", ("time",))
        t.units, t.calendar = "days since 1970-01-01", "proleptic_gregorian"
        t[:] = [(np.datetime64(mo + "-01") - np.datetime64("1970-01-01")).astype(int) for mo in months]
        d.createVariable("latitude", "f8", ("latitude",))[:] = lat
        d.createVariable("longitude", "f8", ("longitude",))[:] = lon
        for name, arr in (("sea_surface_temperature", sst), ("sea_ice_cover", sic)):
            d.createVariable(name, "f4", ("time", "latitude", "longitude"), fill_value=np.float32(np.nan))[:] = arr
        d.createVariable("land_sea_mask", "f4", ("latitude", "longitude"))[:] = lsm


def test_conservative_forcing_regrid(tmp_path):
    from aimip import md15

    lat = np.linspace(90, -90, 361)             # 0.5 deg, north to south like the AIMIP file
    lon = np.arange(720) * 0.5
    months = ["1985-01", "1985-02", "1985-03"]
    land = (np.abs(lat[:, None] - 0.0) <= 10.0) & (lon[None, :] >= 100.0) & (lon[None, :] < 130.0)
    sst = np.where(land, np.nan, 290.0 + 0.0 * lon[None, :] + 10.0 * np.cos(np.deg2rad(lat))[:, None])
    sst = np.repeat(sst[None].astype("f4"), 3, axis=0)
    sst[1] += 1.0
    sst[2, 180, 300] = np.nan                   # a hole inside the ocean in the third month only
    sic = np.where(land, np.nan, np.clip((np.abs(lat) - 60.0) / 30.0, 0.0, 1.0)[:, None] * np.ones((1, 720)))
    sic = np.repeat(sic[None].astype("f4"), 3, axis=0)
    path = tmp_path / "forcing.nc"
    _write_forcing(path, months, sst, sic, land.astype("f4"), lat, lon)

    with netCDF4.Dataset(path) as ds:
        c = md15.conservative_aimip_forcing_1p5(ds, "latitude", "longitude")
    assert c["sst"].shape == (3, 121, 240) and np.allclose(c["target_lat"], np.linspace(-90, 90, 121))
    out_lat = c["target_lat"]
    # fully land destination cells are NaN; a cell partly over land averages the ocean part only
    j0, i0 = int(np.argmin(np.abs(out_lat - 0.0))), int(round(115 / 1.5))
    assert np.isnan(c["sst"][0, j0, i0])
    j_coast = j0 + int(round(10.5 / 1.5))       # 10.5 N: half of the 1.5-deg cell is land
    assert c["sst"][0, j_coast, i0] == pytest.approx(290.0 + 10.0 * np.cos(np.deg2rad(10.5)), abs=0.02)
    # monthly offsets carry over; the ocean hole blanks only the destination cells it overlaps
    ocean = np.isfinite(c["sst"][0])
    assert np.allclose((c["sst"][1] - c["sst"][0])[ocean], 1.0, atol=1e-4)
    hole = np.isfinite(c["sst"][0]) & ~np.isfinite(c["sst"][2])
    assert 1 <= hole.sum() <= 4
    assert np.allclose(c["sst"][2][~hole & ocean], c["sst"][0][~hole & ocean])
    # land fraction is conservative too, with no NaN
    assert np.isfinite(c["lsm"]).all() and c["lsm"][0, j0, i0] == pytest.approx(1.0)

    f = md15.subsample_and_normalize_aimip_forcing(str(path), start_iso="1985-01-01", steps=3,
                                                   train_period=("1985-01-01", "1985-03-01"), regrid="conservative")
    assert f["forcings"].shape == (3, 3, 121, 240) and np.isfinite(f["forcings"]).all()
    assert f["dates"] == ["1985-01-01", "1985-02-01", "1985-03-01"]
    g = md15.subsample_and_normalize_aimip_forcing(str(path), start_iso="1985-01-01", steps=3,
                                                   train_period=("1985-01-01", "1985-03-01"))
    assert not np.allclose(f["forcings"][:, 2], g["forcings"][:, 2])   # land fraction vs point land-sea mask
    with pytest.raises(ValidationError, match="Unknown MD-1.5 forcing regridding"):
        md15.subsample_and_normalize_aimip_forcing(str(path), start_iso="1985-01-01", steps=3, regrid="bilinear")
