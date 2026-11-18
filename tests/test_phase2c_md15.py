"""Offline CPU test suite for Phase 2c `md-1.5` (`src/aimip/md15.py`) and restricted unpickling.

Covers:
1. Restricted whitelist unpickler (`pickle_load_verified(..., allow_torch_config=True)`):
   - Deserializes a realistic `config.pkl` payload (`nn.GELU`, `torch.device('cuda')` remapped to CPU,
     embedded `statics` tensor via `_rebuild_tensor_v2` + `torch.storage._load_from_bytes`,
     `OrderedDict`, and 48 `numpy.str_` `variable_names` matching `MD15_CONFIG_VARIABLE_NAMES`).
   - Blocks `os.system` at BOTH the outer pickle level and inside nested `torch.storage._load_from_bytes`,
     blocks `numpy.dtype('O')`, and blocks 1-byte tamper / symlink swaps BEFORE unpickling.
2. `compute_sha256sum_tree_digest` (`LC_ALL=C` `sha256sum` `./<rel>` format) and safe archive extraction
   (`extract_and_verify_md15_archive`, `verify_md15_code_dir`).
3. Coincident `0.25 deg -> 1.5 deg` (`121 x 240`) subsampling (`max|diff| == 0.0` vs periodic bilinear,
   identical land `NaN` masks, fail-closed `ValidationError` on non-coincident grids, and exact
   `1985-01 .. 2014-12` `SSTKSFC` / `CISFC` / `LSMSFC` normalization).
4. Multi-layout initial-condition loading (`da(time, varlev, lat, lon)`, CF/ERA5 4D pressure-level, and
   `restart.nc`), `resolve_horizon(family='md')`, batched stochastic `run_md15`, all 37 prognostic channels
   with `counts = 1` + `monthly_representation`, `summarize_run`, `restart.nc` continuation, and `EXIT_UNSTABLE == 5`.
"""

from __future__ import annotations

import collections
import datetime
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
import tarfile
from typing import Any

import numpy as np
import pytest

from aimip.errors import ValidationError
from aimip.ic import inspect_ic_file
from aimip.md15 import (
    EXIT_UNSTABLE,
    MD15_COMMIT,
    MD15_CONDITIONING_VARS,
    MD15_CONFIG_VARIABLE_NAMES,
    MD15_LAT_COUNT,
    MD15_LON_COUNT,
    MD15_MONTHLY_REPRESENTATION,
    MD15_OUTPUT_CHANNELS,
    MD15_PLEVELS_HPA,
    MD15_PROGNOSTIC_VARS,
    MD15_STATIC_VARS,
    extract_and_verify_md15_archive,
    find_coincident_1p5_indices,
    load_md15_cvae,
    load_md15_initial_state,
    run_md15,
    subsample_and_normalize_aimip_forcing,
    verify_md15_code_dir,
)
from aimip.safe_pickle import compute_sha256sum_tree_digest, pickle_load_verified
from aimip.status import summarize_run
from aimip.time_utils import resolve_horizon

from conftest import sha256_path


class _UnsupportedMd15WeightsObj:
    """Harmless custom Python object that `load_md15_cvae` (`weights_only=True`) must refuse."""


def _periodic_bilinear_1p5(src_field_asc: np.ndarray, src_lat_asc: np.ndarray, src_lon_asc: np.ndarray) -> np.ndarray:
    """Reference pure-numpy periodic bilinear interpolation onto `(-90..90, 0..358.5)` (`121 x 240`)."""
    target_lat = np.linspace(-90.0, 90.0, MD15_LAT_COUNT, dtype=np.float64)
    target_lon = np.arange(0.0, 360.0, 1.5, dtype=np.float64)

    lon_ext = np.concatenate([src_lon_asc, [360.0]])
    if src_field_asc.ndim == 2:
        fld_ext = np.concatenate([src_field_asc, src_field_asc[:, :1]], axis=-1)
    else:
        fld_ext = np.concatenate([src_field_asc, src_field_asc[:, :, :1]], axis=-1)

    i1 = np.clip(np.searchsorted(src_lat_asc, target_lat, side="right"), 1, len(src_lat_asc) - 1)
    i0 = i1 - 1
    wy = (target_lat - src_lat_asc[i0]) / (src_lat_asc[i1] - src_lat_asc[i0])

    j1 = np.clip(np.searchsorted(lon_ext, target_lon, side="right"), 1, len(lon_ext) - 1)
    j0 = j1 - 1
    wx = (target_lon - lon_ext[j0]) / (lon_ext[j1] - lon_ext[j0])

    wy2 = wy[:, np.newaxis]
    wx2 = wx[np.newaxis, :]
    if fld_ext.ndim == 2:
        f00 = fld_ext[i0[:, None], j0[None, :]]
        f01 = fld_ext[i0[:, None], j1[None, :]]
        f10 = fld_ext[i1[:, None], j0[None, :]]
        f11 = fld_ext[i1[:, None], j1[None, :]]
    else:
        f00 = fld_ext[:, i0[:, None], j0[None, :]]
        f01 = fld_ext[:, i0[:, None], j1[None, :]]
        f10 = fld_ext[:, i1[:, None], j0[None, :]]
        f11 = fld_ext[:, i1[:, None], j1[None, :]]
        wy2 = wy2[np.newaxis, :, :]
        wx2 = wx2[np.newaxis, :, :]

    # Exact weight-zero short-circuit at coincident grid nodes so `0.0 * NaN` does not contaminate wet nodes.
    top = np.where(wx2 == 0.0, f00, np.where(wx2 == 1.0, f01, (1.0 - wx2) * f00 + wx2 * f01))
    bot = np.where(wx2 == 0.0, f10, np.where(wx2 == 1.0, f11, (1.0 - wx2) * f10 + wx2 * f11))
    out = np.where(wy2 == 0.0, top, np.where(wy2 == 1.0, bot, (1.0 - wy2) * top + wy2 * bot))
    return out.astype(np.float32)


def _default_stats_dict() -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {
        "SPSFC": {"mu": 96375.0, "sigma": 9590.0},
        "SKTSFC": {"mu": 278.0, "sigma": 22.0},
        "VAR_2TSFC": {"mu": 277.6, "sigma": 21.5},
        "VAR_2DSFC": {"mu": 273.1, "sigma": 20.9},
        "VAR_10USFC": {"mu": 0.0, "sigma": 3.8},
        "VAR_10VSFC": {"mu": 0.2, "sigma": 2.4},
        "SSTKSFC": {"mu": 286.9, "sigma": 11.4},
        "CISFC": {"mu": -5.915, "sigma": 2.966},
        "MTPRSFC": {"mu": 0.0045, "sigma": 0.0025},
        "MSLSFC": {"mu": 100960.0, "sigma": 1168.5},
        "z500.0": {"mu": 54004.0, "sigma": 3267.5},
    }
    for p in MD15_PLEVELS_HPA:
        stats[f"t{float(p)}"] = {"mu": 255.0, "sigma": 14.0}
        stats[f"q{float(p)}"] = {"mu": 0.03, "sigma": 0.01}
        stats[f"u{float(p)}"] = {"mu": 5.0, "sigma": 8.0}
        stats[f"v{float(p)}"] = {"mu": 0.0, "sigma": 3.0}
    return stats


def test_restricted_unpickler_whitelists_md15_config_and_refuses_rce_globals(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import os as os_mod
    import torch.nn as nn

    assert len(MD15_CONFIG_VARIABLE_NAMES) == 48
    assert len(set(MD15_CONFIG_VARIABLE_NAMES)) == 48
    assert len(MD15_PROGNOSTIC_VARS) == 37
    assert MD15_CONDITIONING_VARS == (
        "SSTKSFC",
        "CISFC",
        "nanmaskSFC",
        "time0SFC",
        "time1SFC",
        "time2SFC",
    )
    assert MD15_STATIC_VARS == ("SDORSFC", "ANORSFC", "ISORSFC", "SLORSFC", "LSMSFC")
    assert MD15_PROGNOSTIC_VARS + MD15_CONDITIONING_VARS + MD15_STATIC_VARS == MD15_CONFIG_VARIABLE_NAMES
    assert len(MD15_OUTPUT_CHANNELS) == 37

    # 1. Realistic MD-1.5 `config.pkl` structure (`GELU`, `torch.device('cuda')`, 3D `statics` `(5, H, W)`, `OrderedDict`, numpy `U` scalars)
    np_var_scalars = list(np.asarray(MD15_CONFIG_VARIABLE_NAMES, dtype="<U10"))
    statics_t = torch.arange(20, dtype=torch.float32).reshape(5, 2, 2)
    cfg_payload = {
        "forcing_dim": 3,
        "data_dimensions": (37, 2, 2),
        "n_statics": 5,
        "activation": nn.GELU,
        "device": torch.device("cuda"),
        "statics": statics_t,
        "meta": collections.OrderedDict([("rank", 32)]),
        "variable_names": np_var_scalars,
        "loading": True,
    }
    cfg_file = tmp_path / "config.pkl"
    cfg_bytes = pickle.dumps(cfg_payload)
    cfg_file.write_bytes(cfg_bytes)
    cfg_sha = hashlib.sha256(cfg_bytes).hexdigest()

    # Without `allow_torch_config=True`, `torch.nn.modules.activation.GELU` is refused even when SHA-256 matches!
    with pytest.raises(ValidationError, match="Refusing non-whitelisted pickle global"):
        pickle_load_verified(str(cfg_file), cfg_sha, allow_torch_config=False)

    # With `allow_torch_config=True`, loads cleanly and remaps `torch.device('cuda')` -> `torch.device('cpu')`!
    loaded = pickle_load_verified(str(cfg_file), cfg_sha, map_location="cpu", allow_torch_config=True)
    assert loaded["activation"] is nn.GELU
    assert loaded["device"] == torch.device("cpu")
    assert torch.equal(loaded["statics"], statics_t)
    assert loaded["statics"].device == torch.device("cpu")
    assert tuple(str(v) for v in loaded["variable_names"]) == MD15_CONFIG_VARIABLE_NAMES

    # 2. Valid SHA-256 outer RCE payload (`os.system`) MUST be refused!
    class _OuterExploit:
        def __reduce__(self):
            return (os_mod.system, ("echo SHOULD_NEVER_RUN",))

    evil_outer = tmp_path / "evil_outer.pkl"
    evil_outer_bytes = pickle.dumps({"bad": _OuterExploit()})
    evil_outer.write_bytes(evil_outer_bytes)
    evil_outer_sha = hashlib.sha256(evil_outer_bytes).hexdigest()
    with pytest.raises(ValidationError, match=r"(?:os|posix)\.system"):
        pickle_load_verified(str(evil_outer), evil_outer_sha, allow_torch_config=True)

    # 3. Valid SHA-256 nested `torch.storage._load_from_bytes(evil_bytes)` MUST ALSO be refused!
    nested_evil_buf = io.BytesIO()
    torch.save({"bad": _OuterExploit()}, nested_evil_buf)

    class _NestedStorageExploit:
        def __reduce__(self):
            return (torch.storage._load_from_bytes, (nested_evil_buf.getvalue(),))

    evil_nested = tmp_path / "evil_nested.pkl"
    evil_nested_bytes = pickle.dumps({"statics": _NestedStorageExploit()})
    evil_nested.write_bytes(evil_nested_bytes)
    evil_nested_sha = hashlib.sha256(evil_nested_bytes).hexdigest()
    with pytest.raises((ValidationError, pickle.UnpicklingError, RuntimeError)):
        pickle_load_verified(str(evil_nested), evil_nested_sha, allow_torch_config=True)

    # 4. Object `numpy.dtype('O')` and structured `numpy.dtype([('a', 'O')])` (`kind == 'V'`, `hasobject == True`) are refused!
    obj_dt_file = tmp_path / "obj_dtype.pkl"
    obj_dt_bytes = pickle.dumps(np.dtype("O"))
    obj_dt_file.write_bytes(obj_dt_bytes)
    obj_dt_sha = hashlib.sha256(obj_dt_bytes).hexdigest()
    with pytest.raises(ValidationError, match="object numpy.dtype"):
        pickle_load_verified(str(obj_dt_file), obj_dt_sha, allow_torch_config=True)

    struct_obj_file = tmp_path / "struct_obj_dtype.pkl"
    struct_obj_bytes = pickle.dumps(np.dtype([("a", "O"), ("b", "f4")]))
    struct_obj_file.write_bytes(struct_obj_bytes)
    struct_obj_sha = hashlib.sha256(struct_obj_bytes).hexdigest()
    with pytest.raises(ValidationError, match="object numpy.dtype"):
        pickle_load_verified(str(struct_obj_file), struct_obj_sha, allow_torch_config=True)

    nested_struct_obj_file = tmp_path / "nested_struct_obj_dtype.pkl"
    nested_struct_obj_bytes = pickle.dumps(np.dtype([("outer", [("inner", "O"), ("x", "f4")])]))
    nested_struct_obj_file.write_bytes(nested_struct_obj_bytes)
    nested_struct_obj_sha = hashlib.sha256(nested_struct_obj_bytes).hexdigest()
    with pytest.raises(ValidationError, match="object numpy.dtype"):
        pickle_load_verified(str(nested_struct_obj_file), nested_struct_obj_sha, allow_torch_config=True)


def test_md15_sha256sum_tree_digest_and_archive_extraction(tmp_path: Path) -> None:
    src_tree = tmp_path / "monthly-diffusion-src"
    (src_tree / "md").mkdir(parents=True)
    (src_tree / "models" / "MD-1p5-latest.pth").mkdir(parents=True)
    (src_tree / "scripts").mkdir(parents=True)

    (src_tree / "LICENSE").write_text("MIT License\nCopyright (c) 2026 Kyle Hall\n", encoding="utf-8")
    (src_tree / "md" / "__init__.py").write_text("__version__ = '0.9'\n", encoding="utf-8")
    cfg_bytes = b"synthetic-config"
    wt_bytes = b"synthetic-weights"
    st_bytes = b'{"VAR_2TSFC": {"mu": 277.5, "sigma": 21.5}}\n'
    (src_tree / "models" / "MD-1p5-latest.pth" / "config.pkl").write_bytes(cfg_bytes)
    (src_tree / "models" / "MD-1p5-latest.pth" / "weights.pt").write_bytes(wt_bytes)
    (src_tree / "scripts" / "variable_statistics_197901-201412.json").write_bytes(st_bytes)

    tree_sha, n_files = compute_sha256sum_tree_digest(str(src_tree))
    assert n_files == 5
    # Manually verify exact `LC_ALL=C` sorted `<sha256>  ./<rel>\n` formula
    rels = sorted(
        f.relative_to(src_tree).as_posix()
        for f in src_tree.rglob("*")
        if f.is_file()
    )
    manual_lines = "".join(
        f"{hashlib.sha256((src_tree / r).read_bytes()).hexdigest()}  ./{r}\n" for r in rels
    )
    assert tree_sha == hashlib.sha256(manual_lines.encode("utf-8")).hexdigest()

    tar_path = tmp_path / "md15.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for r in rels:
            tf.add(src_tree / r, arcname=f"monthly-diffusion-{MD15_COMMIT}/{r}")
    tar_sha = sha256_path(tar_path)
    tar_size = tar_path.stat().st_size

    data_root = tmp_path / "data"
    code_dst = data_root / "code" / "monthly-diffusion"
    res = extract_and_verify_md15_archive(
        str(tar_path),
        str(code_dst),
        model_target_dir=None,
        expected_archive_sha256=tar_sha,
        expected_archive_size=tar_size,
        expected_tree_sha256=tree_sha,
        expected_tree_files=5,
    )
    assert res == {"code_dir": str(code_dst), "tree_sha256": tree_sha, "files": 5}
    verify_md15_code_dir(str(data_root), expected_tree_sha256=tree_sha, expected_tree_files=5)

    # Mutating 1 byte in extracted `md/__init__.py` invalidates `verify_md15_code_dir`
    (code_dst / "md" / "__init__.py").write_text("__version__ = 'tampered'\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="Lock verification failed for code/monthly-diffusion"):
        verify_md15_code_dir(str(data_root), expected_tree_sha256=tree_sha, expected_tree_files=5)


def test_md15_coincident_0p25_subsampling_matches_periodic_bilinear_and_fails_closed(
    tmp_path: Path,
) -> None:
    netCDF4 = pytest.importorskip("netCDF4")

    # 1. Non-coincident grids (missing poles or 0.4 deg step) MUST fail closed!
    with pytest.raises(ValidationError, match="does not coincide"):
        find_coincident_1p5_indices(np.linspace(-89.25, 89.25, 120), np.arange(0.0, 360.0, 1.5))
    with pytest.raises(ValidationError, match="does not coincide"):
        find_coincident_1p5_indices(np.linspace(-90.0, 90.0, 451), np.arange(0.0, 360.0, 0.4))

    # 2. Real Zenodo 17065758 geometry: descending `latitude` `90 -> -90` (`721`), `longitude` `0 -> 359.75` (`1440`)
    lat_desc = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon_025 = np.arange(0.0, 360.0, 0.25, dtype=np.float32)
    dates = ["1985-01-01", "1995-07-01", "2014-12-01", "2015-01-01"]
    epo = datetime.date(1970, 1, 1)
    day_nums = np.asarray(
        [(datetime.date(int(d[:4]), int(d[5:7]), 1) - epo).days for d in dates], dtype=np.int64
    )

    rng = np.random.default_rng(123)
    lsm_desc = (rng.uniform(0.0, 1.0, size=(721, 1440)) > 0.7).astype(np.float32)
    land_mask = lsm_desc > 0.5
    sst_desc = (285.0 + 12.0 * np.cos(np.deg2rad(lat_desc))[:, None] + rng.normal(0.0, 0.5, size=(4, 721, 1440))).astype(np.float32)
    sic_desc = np.clip(0.3 * (np.abs(lat_desc)[:, None] > 65.0) + rng.uniform(0.0, 0.05, size=(4, 721, 1440)), 0.0, 1.0).astype(np.float32)
    sst_desc[:, land_mask] = np.nan
    sic_desc[:, land_mask] = np.nan

    f_nc = tmp_path / "ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc"
    with netCDF4.Dataset(str(f_nc), "w", format="NETCDF4") as ds:
        ds.createDimension("time", len(dates))
        ds.createDimension("latitude", 721)
        ds.createDimension("longitude", 1440)
        vt = ds.createVariable("time", "i8", ("time",))
        vt.units = "days since 1970-01-01 00:00:00"
        vt.calendar = "proleptic_gregorian"
        vt[:] = day_nums
        vlat = ds.createVariable("latitude", "f4", ("latitude",))
        vlat[:] = lat_desc
        vlon = ds.createVariable("longitude", "f4", ("longitude",))
        vlon[:] = lon_025
        vsst = ds.createVariable("sea_surface_temperature", "f4", ("time", "latitude", "longitude"))
        vsst[:] = sst_desc
        vsic = ds.createVariable("sea_ice_cover", "f4", ("time", "latitude", "longitude"))
        vsic[:] = sic_desc
        vlsm = ds.createVariable("land_sea_mask", "f4", ("latitude", "longitude"))
        vlsm[:] = lsm_desc

    # Compare coincident subsampling with reference periodic bilinear on ascending latitude
    lat_asc = lat_desc[::-1].astype(np.float64)
    sst_asc = sst_desc[:, ::-1, :]
    bilin_sst = _periodic_bilinear_1p5(sst_asc, lat_asc, lon_025.astype(np.float64))
    sub_sst = sst_asc[:, ::6, ::6]
    assert np.array_equal(np.isnan(bilin_sst), np.isnan(sub_sst))
    assert float(np.nanmax(np.abs(bilin_sst - sub_sst))) == 0.0

    norm_res = subsample_and_normalize_aimip_forcing(
        str(f_nc),
        start_iso="2014-12-01T00:00:00",
        steps=2,
    )
    assert norm_res["forcings"].shape == (2, 3, 121, 240)
    assert np.isfinite(norm_res["forcings"]).all()
    assert norm_res["dates"] == ["2014-12-01", "2015-01-01"]
    assert norm_res["months"].flatten().tolist() == [12.0, 1.0]
    assert norm_res["land_nan_cells_per_step"] == int(np.sum(land_mask[::-1, :][::6, ::6]))
    # Over land cells, standardized `SSTKSFC` is filled with exact `0.0`
    land_1p5 = land_mask[::-1, :][::6, ::6]
    assert np.all(norm_res["forcings"][:, 0, land_1p5] == 0.0)


def test_md15_horizon_ic_layouts_rollout_restart_continuation_and_unstable_exit(
    tmp_path: Path,
    allow_tmp_run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    netCDF4 = pytest.importorskip("netCDF4")
    import torch.nn as nn
    import aimip.md15 as md15_mod

    # 1. `resolve_horizon` monthly calendar math for `family="md"`
    assert resolve_horizon(start_iso="1978-10-01T00:00:00", steps=12, family="md", model_name="md-1.5") == (
        12,
        "1979-09-01T00:00:00",
    )
    assert resolve_horizon(start_iso="1978-10-01T00:00:00", length="10y", family="md", model_name="md-1.5") == (
        120,
        "1988-09-01T00:00:00",
    )
    assert resolve_horizon(
        start_iso="1978-10-01T00:00:00",
        end_iso="2025-01-01T00:00:00",
        family="md",
        model_name="md-1.5",
    ) == (556, "2025-01-01T00:00:00")
    with pytest.raises(ValidationError, match="whole calendar monthly steps"):
        resolve_horizon(start_iso="1978-10-01T00:00:00", length="6h", family="md", model_name="md-1.5")

    # 2. Build synthetic verified `md` package + `config.pkl` + `weights.pt` + `stats.json` + forcing + 2 IC layouts
    code_dir = tmp_path / "code" / "monthly-diffusion"
    (code_dir / "md").mkdir(parents=True)
    (code_dir / "md" / "__init__.py").write_text(
        """import math
import torch
import torch.nn as nn

class _Seasonality(nn.Module):
    def forward(self, month):
        b = month.shape[0]
        return torch.zeros(b, 3, 121, 240, device=month.device, dtype=torch.float32)

class _Encoder(nn.Module):
    def __init__(self, n_statics=5, statics=None, varnames=None):
        super().__init__()
        self.n_statics = n_statics
        self.varnames = list(varnames) if varnames is not None else []
        assert len(self.varnames) == 37 + 6 + n_statics
        self.proj = nn.Conv2d(37 + 6 + n_statics, 4, kernel_size=1)
        self.statics = self.embed_conditions(statics) if statics is not None else [None, None]
    def embed_conditions(self, c):
        if c.ndim == 3:
            return [c, c]
        return [c, c[:, :2, :, :]]
    def forward(self, x, c):
        s0 = self.statics[0].unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x_cat = torch.cat([x, c, s0], dim=1)
        assert x_cat.shape[1] == len(self.varnames) == 48
        z = self.proj(x_cat)
        return z, torch.zeros_like(z), self.embed_conditions(c)

class _Predictor(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.latent_mu = nn.Parameter(torch.zeros(1)).to(device)
        self.latent_log_sigma = nn.Parameter(torch.zeros(1)).to(device)
    def forward(self, z, cond):
        noise = 0.05 * torch.randn_like(z)
        return z * self.scale + noise

class _Decoder(nn.Module):
    def __init__(self, n_statics=5, statics=None):
        super().__init__()
        self.statics = statics
        self.out = nn.Conv2d(4 + n_statics, 37, kernel_size=1)
    def forward(self, z, cs):
        s_dec = self.statics[::-1][0].unsqueeze(0).expand(z.shape[0], -1, -1, -1)
        return self.out(torch.cat([z, s_dec], dim=1))

class CVAE(nn.Module):
    def __init__(self, **cfg):
        super().__init__()
        self.device = cfg["device"]
        self.n_statics = int(cfg.get("n_statics", 5))
        self.statics = cfg["statics"].to(self.device)
        self.variable_names = cfg["variable_names"]
        self.seasonality_embedding = _Seasonality()
        self.encoder = _Encoder(n_statics=self.n_statics, statics=self.statics, varnames=self.variable_names)
        self.predictor = _Predictor(self.device)
        self.decoder = _Decoder(n_statics=self.n_statics, statics=self.encoder.statics)
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(md15_mod, "ensure_torch_harmonics", lambda: None)

    ckpt_dir = tmp_path / "MD-1p5-latest.pth"
    ckpt_dir.mkdir()
    cfg_obj = {
        "forcing_dim": 3,
        "data_dimensions": (37, 121, 240),
        "n_statics": 5,
        "activation": nn.GELU,
        "device": torch.device("cuda"),
        "statics": torch.zeros(5, 121, 240),
        "variable_names": list(np.asarray(MD15_CONFIG_VARIABLE_NAMES, dtype="<U10")),
    }
    cfg_path = ckpt_dir / "config.pkl"
    cfg_path.write_bytes(pickle.dumps(cfg_obj))
    cfg_sha = sha256_path(cfg_path)

    # Instantiate once on CPU to produce a checkpoint WITHOUT `latent_mu`/`latent_log_sigma` in state_dict
    md_pkg = md15_mod.load_verified_md_package(str(code_dir), verify_tree=False)
    ref_model = md_pkg.CVAE(**{**cfg_obj, "device": torch.device("cpu")})
    for k in ("latent_mu", "latent_log_sigma"):
        val = ref_model.predictor._parameters.pop(k).detach()
        setattr(ref_model.predictor, k, val)
    for p in ref_model.parameters():
        nn.init.zeros_(p)
    nn.init.ones_(ref_model.predictor.scale)
    wt_path = ckpt_dir / "weights.pt"
    torch.save(ref_model.state_dict(), wt_path)
    wt_sha = sha256_path(wt_path)

    # Confirm `load_md15_cvae` verifies both hashes, checks 48 `variable_names` + 3D `statics` `(5, 121, 240)`,
    # converts `latent_mu`/`latent_log_sigma`, and loads `strict=True`
    loaded_cvae, loaded_cfg = load_md15_cvae(
        str(ckpt_dir),
        config_sha256=cfg_sha,
        weights_sha256=wt_sha,
        device=torch.device("cpu"),
        md_module=md_pkg,
    )
    assert tuple(str(v) for v in loaded_cfg["variable_names"]) == MD15_CONFIG_VARIABLE_NAMES
    assert tuple(loaded_cfg["statics"].shape) == (5, 121, 240)
    assert "latent_mu" not in loaded_cvae.predictor._parameters

    # Truncated 43-name `variable_names` or mismatched `statics` shape MUST fail closed in `load_md15_cvae`!
    bad_ckpt_dir = tmp_path / "bad_ckpt"
    bad_ckpt_dir.mkdir()
    (bad_ckpt_dir / "weights.pt").write_bytes(wt_path.read_bytes())
    trunc_cfg = {**cfg_obj, "variable_names": list(MD15_CONFIG_VARIABLE_NAMES[:43])}
    (bad_ckpt_dir / "config.pkl").write_bytes(pickle.dumps(trunc_cfg))
    with pytest.raises(ValidationError, match="variable_names mismatch"):
        load_md15_cvae(
            str(bad_ckpt_dir),
            config_sha256=sha256_path(bad_ckpt_dir / "config.pkl"),
            weights_sha256=wt_sha,
            device=torch.device("cpu"),
            md_module=md_pkg,
        )

    bad_statics_cfg = {**cfg_obj, "statics": torch.zeros(1, 5, 121, 240)}
    (bad_ckpt_dir / "config.pkl").write_bytes(pickle.dumps(bad_statics_cfg))
    with pytest.raises(ValidationError, match="statics` shape mismatch"):
        load_md15_cvae(
            str(bad_ckpt_dir),
            config_sha256=sha256_path(bad_ckpt_dir / "config.pkl"),
            weights_sha256=wt_sha,
            device=torch.device("cpu"),
            md_module=md_pkg,
        )

    # `weights.pt` carrying a harmless unsupported Python object is refused by `load_md15_cvae`
    # (`weights_only=True`, no pickle fallback) even when its SHA-256 matches!
    obj_ckpt_dir = tmp_path / "obj_weights_ckpt"
    obj_ckpt_dir.mkdir()
    (obj_ckpt_dir / "config.pkl").write_bytes(cfg_path.read_bytes())
    torch.save({**ref_model.state_dict(), "extra_obj": _UnsupportedMd15WeightsObj()}, obj_ckpt_dir / "weights.pt")
    with pytest.raises(pickle.UnpicklingError):
        load_md15_cvae(
            str(obj_ckpt_dir),
            config_sha256=cfg_sha,
            weights_sha256=sha256_path(obj_ckpt_dir / "weights.pt"),
            device=torch.device("cpu"),
            md_module=md_pkg,
        )

    stats = _default_stats_dict()
    stats_path = tmp_path / "variable_statistics_197901-201412.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    stats_sha = sha256_path(stats_path)

    # 1.5 deg forcing NetCDF covering `1985-01`, `2014-12`, `2015-01`, `2015-02`, `2015-03`
    lat_1p5 = np.linspace(-90.0, 90.0, 121, dtype=np.float32)
    lon_1p5 = np.arange(0.0, 360.0, 1.5, dtype=np.float32)
    f_dates = ["1985-01-01", "2014-12-01", "2015-01-01", "2015-02-01", "2015-03-01"]
    epo = datetime.date(1970, 1, 1)
    f_days = np.asarray([(datetime.date(int(d[:4]), int(d[5:7]), 1) - epo).days for d in f_dates], dtype=np.int64)
    forcing_nc = tmp_path / "forcing_1p5.nc"
    with netCDF4.Dataset(str(forcing_nc), "w", format="NETCDF4") as ds:
        ds.createDimension("time", len(f_dates))
        ds.createDimension("latitude", 121)
        ds.createDimension("longitude", 240)
        vt = ds.createVariable("time", "i8", ("time",))
        vt.units = "days since 1970-01-01 00:00:00"
        vt.calendar = "proleptic_gregorian"
        vt[:] = f_days
        ds.createVariable("latitude", "f4", ("latitude",))[:] = lat_1p5
        ds.createVariable("longitude", "f4", ("longitude",))[:] = lon_1p5
        sst_v = ds.createVariable("sea_surface_temperature", "f4", ("time", "latitude", "longitude"))
        sst_arr = np.broadcast_to((280.0 + 10.0 * np.cos(np.deg2rad(lat_1p5)))[None, :, None], (len(f_dates), 121, 240)).copy()
        sst_arr[:, 0, :] = np.nan
        sst_v[:] = sst_arr
        sic_v = ds.createVariable("sea_ice_cover", "f4", ("time", "latitude", "longitude"))
        sic_arr = np.broadcast_to((0.2 * (np.abs(lat_1p5) > 70.0))[None, :, None], (len(f_dates), 121, 240)).astype(np.float32).copy()
        sic_arr[:, 0, :] = np.nan
        sic_v[:] = sic_arr
        lsm_v = ds.createVariable("land_sea_mask", "f4", ("latitude", "longitude"))
        lsm_arr = np.zeros((121, 240), dtype=np.float32)
        lsm_arr[0, :] = 1.0
        lsm_v[:] = lsm_arr

    # Layout A: 4D flat `da(time, varlev, lat, lon)` IC at `2015-01-01`
    flat_ic = tmp_path / "ic_2015-01-01_flat.nc"
    with netCDF4.Dataset(str(flat_ic), "w", format="NETCDF4") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("varlev", len(MD15_PROGNOSTIC_VARS))
        ds.createDimension("lat", 121)
        ds.createDimension("lon", 240)
        vt = ds.createVariable("time", "i8", ("time",))
        vt.units = "days since 1970-01-01 00:00:00"
        vt.calendar = "proleptic_gregorian"
        vt[:] = [(datetime.date(2015, 1, 1) - epo).days]
        ds.createVariable("lat", "f4", ("lat",))[:] = lat_1p5
        ds.createVariable("lon", "f4", ("lon",))[:] = lon_1p5
        vvl = ds.createVariable("varlev", str, ("varlev",))
        for idx_v, v_str in enumerate(MD15_PROGNOSTIC_VARS):
            vvl[idx_v] = v_str
        vda = ds.createVariable("da", "f4", ("time", "varlev", "lat", "lon"))
        arr_ic = np.zeros((1, len(MD15_PROGNOSTIC_VARS), 121, 240), dtype=np.float32)
        for c_i, ch_key in enumerate(MD15_PROGNOSTIC_VARS):
            mu_val = float(stats[ch_key]["mu"])
            arr_ic[0, c_i, :, :] = (mu_val * mu_val) if ch_key in ("MTPRSFC", *(f"q{float(p)}" for p in MD15_PLEVELS_HPA)) else mu_val
        vda[:] = arr_ic

    # `inspect_ic_file` recognizes 4D `da(time, varlev, lat, lon)` ICs!
    ic_info = inspect_ic_file(str(flat_ic))
    assert ic_info.timestamps == ("2015-01-01T00:00:00",)
    assert ic_info.grid_shape == (121, 240)
    assert len(ic_info.variable_names) == 37

    # Layout B: CF/ERA5 4D pressure-level IC (`t2m`, ..., `temperature(time, level, lat, lon)`)
    cf_ic = tmp_path / "ic_2015-01-01_cf.nc"
    with netCDF4.Dataset(str(cf_ic), "w", format="NETCDF4") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("level", len(MD15_PLEVELS_HPA))
        ds.createDimension("latitude", 121)
        ds.createDimension("longitude", 240)
        vt = ds.createVariable("time", "i8", ("time",))
        vt.units = "days since 1970-01-01 00:00:00"
        vt.calendar = "proleptic_gregorian"
        vt[:] = [(datetime.date(2015, 1, 1) - epo).days]
        ds.createVariable("level", "i4", ("level",))[:] = np.asarray(MD15_PLEVELS_HPA, dtype=np.int32)
        ds.createVariable("latitude", "f4", ("latitude",))[:] = lat_1p5[::-1]  # descending latitude!
        ds.createVariable("longitude", "f4", ("longitude",))[:] = lon_1p5
        for sfc_n, md_k in (
            ("sp", "SPSFC"),
            ("skt", "SKTSFC"),
            ("t2m", "VAR_2TSFC"),
            ("d2m", "VAR_2DSFC"),
            ("u10", "VAR_10USFC"),
            ("v10", "VAR_10VSFC"),
            ("mtpr", "MTPRSFC"),
            ("msl", "MSLSFC"),
            ("z500", "z500.0"),
        ):
            val = float(stats[md_k]["mu"]) ** 2 if md_k == "MTPRSFC" else float(stats[md_k]["mu"])
            ds.createVariable(sfc_n, "f4", ("time", "latitude", "longitude"))[:] = np.full((1, 121, 240), val, dtype=np.float32)
        for pl_n, pfx in (
            ("temperature", "t"),
            ("specific_humidity", "q"),
            ("u_component_of_wind", "u"),
            ("v_component_of_wind", "v"),
        ):
            pl_arr = np.zeros((1, len(MD15_PLEVELS_HPA), 121, 240), dtype=np.float32)
            for l_i, p_hpa in enumerate(MD15_PLEVELS_HPA):
                mk = f"{pfx}{float(p_hpa)}"
                pl_arr[0, l_i] = float(stats[mk]["mu"]) ** 2 if pfx == "q" else float(stats[mk]["mu"])
            ds.createVariable(pl_n, "f4", ("time", "level", "latitude", "longitude"))[:] = pl_arr

    norm_from_flat, _ = load_md15_initial_state(str(flat_ic), start_iso="2015-01-01T00:00:00", members=2, stats_dict=stats)
    norm_from_cf, _ = load_md15_initial_state(str(cf_ic), start_iso="2015-01-01T00:00:00", members=2, stats_dict=stats)
    assert norm_from_flat.shape == (2, 37, 121, 240)
    assert np.allclose(norm_from_flat, 0.0, atol=1e-5)
    assert np.allclose(norm_from_cf, norm_from_flat, atol=1e-5)

    # Layout C: 2D `|S1` char-array `varlev(39, 10)` + `da(varlev, time, lat, lon)` (exact xarray `era5-flat-1p5x1p5.nc` layout!)
    char_ic = tmp_path / "era5-flat-1p5x1p5.nc"
    all_39_vars = (*MD15_PROGNOSTIC_VARS, "SSTKSFC", "CISFC")
    with netCDF4.Dataset(str(char_ic), "w", format="NETCDF4") as ds:
        ds.createDimension("varlev", len(all_39_vars))
        ds.createDimension("string10", 10)
        ds.createDimension("time", 1)
        ds.createDimension("lat", 121)
        ds.createDimension("lon", 240)
        vt = ds.createVariable("time", "i8", ("time",))
        vt.units = "days since 1978-10-01 00:00:00"
        vt.calendar = "proleptic_gregorian"
        vt[:] = [0]
        ds.createVariable("lat", "f4", ("lat",))[:] = lat_1p5
        ds.createVariable("lon", "f4", ("lon",))[:] = lon_1p5
        vvl_c = ds.createVariable("varlev", "S1", ("varlev", "string10"))
        vvl_c[:] = netCDF4.stringtochar(np.asarray(all_39_vars, dtype="S10"))
        vda_c = ds.createVariable("da", "f4", ("varlev", "time", "lat", "lon"))
        arr_39 = np.zeros((len(all_39_vars), 1, 121, 240), dtype=np.float32)
        for c_i, ch_key in enumerate(MD15_PROGNOSTIC_VARS):
            mu_val = float(stats[ch_key]["mu"])
            arr_39[c_i, 0, :, :] = (
                (mu_val * mu_val)
                if ch_key in ("MTPRSFC", *(f"q{float(p)}" for p in MD15_PLEVELS_HPA))
                else mu_val
            )
        arr_39[37, 0, :, :] = 285.0
        arr_39[38, 0, :, :] = 0.05
        vda_c[:] = arr_39

    norm_from_char, char_meta = load_md15_initial_state(
        str(char_ic), start_iso="1978-10-01T00:00:00", members=2, stats_dict=stats
    )
    assert norm_from_char.shape == (2, 37, 121, 240)
    assert np.allclose(norm_from_char, 0.0, atol=1e-5)
    assert char_meta == {"ic_samples_loaded": 1, "start": "1978-10-01"}

    # 2b. Multi-year NetCDF archives (`time=6`) MUST slice selected samples & coincident grid directly
    #     without ever reading the full time axis (`:` or `...`), on BOTH flat `da` and 3D/4D CF layouts!
    multi_dates = [
        "2014-10-01",
        "2014-11-01",
        "2014-12-01",
        "2015-01-01",
        "2015-02-01",
        "2015-03-01",
    ]
    multi_days = np.asarray(
        [(datetime.date(int(d[:4]), int(d[5:7]), 1) - epo).days for d in multi_dates],
        dtype=np.int64,
    )
    multi_flat_ic = tmp_path / "multi_year_flat.nc"
    with netCDF4.Dataset(str(multi_flat_ic), "w", format="NETCDF4") as ds:
        ds.createDimension("varlev", len(all_39_vars))
        ds.createDimension("string10", 10)
        ds.createDimension("time", len(multi_dates))
        ds.createDimension("lat", 121)
        ds.createDimension("lon", 240)
        vt_m = ds.createVariable("time", "i8", ("time",))
        vt_m.units = "days since 1970-01-01 00:00:00"
        vt_m.calendar = "proleptic_gregorian"
        vt_m[:] = multi_days
        ds.createVariable("lat", "f4", ("lat",))[:] = lat_1p5
        ds.createVariable("lon", "f4", ("lon",))[:] = lon_1p5
        ds.createVariable("varlev", "S1", ("varlev", "string10"))[:] = netCDF4.stringtochar(
            np.asarray(all_39_vars, dtype="S10")
        )
        arr_m = np.full((len(all_39_vars), len(multi_dates), 121, 240), 999.0, dtype=np.float32)
        arr_m[:37, 3, :, :] = arr_39[:37, 0, :, :]  # Only index 3 (`2015-01-01`) has valid state
        ds.createVariable("da", "f4", ("varlev", "time", "lat", "lon"))[:] = arr_m

    lat_075_desc = np.linspace(90.0, -90.0, 241, dtype=np.float32)
    lon_075 = np.arange(0.0, 360.0, 0.75, dtype=np.float32)
    multi_cf_ic = tmp_path / "multi_year_cf_075.nc"
    with netCDF4.Dataset(str(multi_cf_ic), "w", format="NETCDF4") as ds:
        ds.createDimension("time", len(multi_dates))
        ds.createDimension("level", len(MD15_PLEVELS_HPA))
        ds.createDimension("latitude", 241)
        ds.createDimension("longitude", 480)
        vt_c = ds.createVariable("time", "i8", ("time",))
        vt_c.units = "days since 1970-01-01 00:00:00"
        vt_c.calendar = "proleptic_gregorian"
        vt_c[:] = multi_days
        ds.createVariable("level", "i4", ("level",))[:] = np.asarray(MD15_PLEVELS_HPA, dtype=np.int32)
        ds.createVariable("latitude", "f4", ("latitude",))[:] = lat_075_desc
        ds.createVariable("longitude", "f4", ("longitude",))[:] = lon_075
        for sfc_n, md_k in (
            ("sp", "SPSFC"),
            ("skt", "SKTSFC"),
            ("t2m", "VAR_2TSFC"),
            ("d2m", "VAR_2DSFC"),
            ("u10", "VAR_10USFC"),
            ("v10", "VAR_10VSFC"),
            ("mtpr", "MTPRSFC"),
            ("msl", "MSLSFC"),
            ("z500", "z500.0"),
        ):
            val = float(stats[md_k]["mu"]) ** 2 if md_k == "MTPRSFC" else float(stats[md_k]["mu"])
            sfc_arr = np.full((len(multi_dates), 241, 480), -999.0, dtype=np.float32)
            sfc_arr[3] = val
            ds.createVariable(sfc_n, "f4", ("time", "latitude", "longitude"))[:] = sfc_arr
        for pl_n, pfx in (
            ("temperature", "t"),
            ("specific_humidity", "q"),
            ("u_component_of_wind", "u"),
            ("v_component_of_wind", "v"),
        ):
            pl_arr = np.full((len(multi_dates), len(MD15_PLEVELS_HPA), 241, 480), -999.0, dtype=np.float32)
            for l_i, p_hpa in enumerate(MD15_PLEVELS_HPA):
                mk = f"{pfx}{float(p_hpa)}"
                pl_arr[3, l_i] = float(stats[mk]["mu"]) ** 2 if pfx == "q" else float(stats[mk]["mu"])
            ds.createVariable(pl_n, "f4", ("time", "level", "latitude", "longitude"))[:] = pl_arr

    real_nc_dataset = netCDF4.Dataset

    class _NoFullTimeReadVar:
        def __init__(self, name: str, var: Any) -> None:
            self._name = name
            self._var = var

        def __getattr__(self, attr: str) -> Any:
            return getattr(self._var, attr)

        def __getitem__(self, key: Any) -> Any:
            if key is Ellipsis:
                raise AssertionError(f"Full-archive Ellipsis read attempted on {self._name}")
            key_tup = key if isinstance(key, tuple) else (key,)
            if any(k is Ellipsis for k in key_tup):
                raise AssertionError(f"Ellipsis read attempted on {self._name}: {key!r}")
            t_ax = self._var.dimensions.index("time")
            t_total = int(self._var.shape[t_ax])
            t_idx = key_tup[t_ax] if t_ax < len(key_tup) else slice(None)
            if isinstance(t_idx, slice):
                n_req = len(range(*t_idx.indices(t_total)))
                if n_req >= t_total:
                    raise AssertionError(
                        f"Full time axis read ({n_req}/{t_total}) attempted on {self._name}: {key!r}"
                    )
            return self._var[key]

    class _NoFullTimeReadDataset:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._ds = real_nc_dataset(*args, **kwargs)
            self.variables = {
                vname: (
                    _NoFullTimeReadVar(vname, vobj)
                    if vobj.ndim >= 3 and "time" in vobj.dimensions and int(vobj.shape[vobj.dimensions.index("time")]) > 2
                    else vobj
                )
                for vname, vobj in self._ds.variables.items()
            }

        def __getattr__(self, attr: str) -> Any:
            return getattr(self._ds, attr)

        def __enter__(self) -> _NoFullTimeReadDataset:
            self._ds.__enter__()
            return self

        def __exit__(self, *exc: Any) -> Any:
            return self._ds.__exit__(*exc)

    monkeypatch.setattr(netCDF4, "Dataset", _NoFullTimeReadDataset)
    try:
        sliced_flat, sf_meta = load_md15_initial_state(
            str(multi_flat_ic), start_iso="2015-01-01T00:00:00", members=1, stats_dict=stats
        )
        sliced_cf, sc_meta = load_md15_initial_state(
            str(multi_cf_ic), start_iso="2015-01-01T00:00:00", members=1, stats_dict=stats
        )
    finally:
        monkeypatch.setattr(netCDF4, "Dataset", real_nc_dataset)
    assert sf_meta == {"ic_samples_loaded": 1, "start": "2015-01-01"}
    assert sc_meta == {"ic_samples_loaded": 1, "start": "2015-01-01"}
    assert np.allclose(sliced_flat, 0.0, atol=1e-5)
    assert np.allclose(sliced_cf, 0.0, atol=1e-5)

    # 2c. Multi-state sample-indexed IC (`N == 3` distinct states at `2015-01-01`):
    #     - Refuses `members=2` and `members=4` instead of silently duplicating state 0!
    #     - Preserves all 3 distinct states when `members=3`!
    restart_3m = tmp_path / "restart_3m.nc"
    t2m_distinct_vals = (270.0, 280.0, 290.0)
    with netCDF4.Dataset(str(restart_3m), "w", format="NETCDF4") as ds:
        ds.createDimension("sample", 3)
        ds.createDimension("lat", 121)
        ds.createDimension("lon", 240)
        vt_r = ds.createVariable("time", str, ("sample",))
        for s_i in range(3):
            vt_r[s_i] = "2015-01-01T00:00:00"
        ds.createVariable("lat", "f4", ("lat",))[:] = lat_1p5
        ds.createVariable("lon", "f4", ("lon",))[:] = lon_1p5
        for md_k, _out_name, _u, _ln in MD15_OUTPUT_CHANNELS:
            if md_k == "VAR_2TSFC":
                arr_s = np.stack(
                    [np.full((121, 240), v_k, dtype=np.float32) for v_k in t2m_distinct_vals],
                    axis=0,
                )
            else:
                base_mu = (
                    float(stats[md_k]["mu"]) ** 2
                    if md_k in ("MTPRSFC", *(f"q{float(p)}" for p in MD15_PLEVELS_HPA))
                    else float(stats[md_k]["mu"])
                )
                arr_s = np.full((3, 121, 240), base_mu, dtype=np.float32)
            ds.createVariable(md_k, "f4", ("sample", "lat", "lon"))[:] = arr_s

    with pytest.raises(ValidationError, match=r"contains 3 states.*--members 2"):
        load_md15_initial_state(str(restart_3m), start_iso="2015-01-01T00:00:00", members=2, stats_dict=stats)
    with pytest.raises(ValidationError, match=r"contains 3 states.*--members 4"):
        load_md15_initial_state(str(restart_3m), start_iso="2015-01-01T00:00:00", members=4, stats_dict=stats)

    norm_3m, meta_3m = load_md15_initial_state(
        str(restart_3m), start_iso="2015-01-01T00:00:00", members=3, stats_dict=stats
    )
    assert meta_3m == {"ic_samples_loaded": 3, "start": "2015-01-01"}
    t2m_ch = MD15_PROGNOSTIC_VARS.index("VAR_2TSFC")
    t2m_mu = float(stats["VAR_2TSFC"]["mu"])
    t2m_sig = float(stats["VAR_2TSFC"]["sigma"])
    for m_i, raw_v in enumerate(t2m_distinct_vals):
        assert np.allclose(norm_3m[m_i, t2m_ch], (raw_v - t2m_mu) / t2m_sig, atol=1e-5)

    # 3. End-to-end 3-month, 2-member rollout + `summarize_run` + `restart.nc` continuation!
    # Give decoder `out.weight` a small non-zero weight so distinct diffusion latents decode to distinct members
    with torch.no_grad():
        ref_model.decoder.out.weight.fill_(0.1)
    torch.save(ref_model.state_dict(), wt_path)
    wt_sha = sha256_path(wt_path)

    run1_dir = allow_tmp_run_root / "md15_run1"
    run1_dir.mkdir()
    job1 = {
        "family": "md",
        "model": "md-1.5",
        "out_dir": str(run1_dir),
        "code_dir": str(code_dir),
        "verify_code_tree": False,
        "ckpt_dir": str(ckpt_dir),
        "config_sha256": cfg_sha,
        "weights_sha256": wt_sha,
        "stats_file": str(stats_path),
        "stats_sha256": stats_sha,
        "forcing_file": str(forcing_nc),
        "ic_file": str(flat_ic),
        "ic_sha256": sha256_path(flat_ic),
        "start": "2015-01-01T00:00:00",
        "steps": 3,
        "members": 2,
        "seed": 17,
        "device": "cpu",
    }
    res1 = run_md15(str(run1_dir), job1)
    assert res1["months_written"] == {"atmosphere": 3}
    assert res1["first_month"] == "2015-01"
    assert res1["last_month"] == "2015-03"
    assert res1["in_training_or_val_period"] is False
    assert res1["spinup_warning"] is not None

    monthly_nc = run1_dir / "monthly_mean_predictions.nc"
    restart_nc = run1_dir / "restart.nc"
    assert monthly_nc.is_file() and restart_nc.is_file()
    with netCDF4.Dataset(str(monthly_nc), "r") as ds:
        assert ds.monthly_representation == MD15_MONTHLY_REPRESENTATION
        assert np.all(np.asarray(ds.variables["counts"][:]) == 1)
        for _md_k, out_name, units, _ln in MD15_OUTPUT_CHANNELS:
            assert out_name in ds.variables
            assert ds.variables[out_name].units == units
        t2m_arr = np.asarray(ds.variables["TMP2m"][:])  # (2, 3, 121, 240)
        # Step 0 is deterministic autoencoder reconstruction; step >= 1 diverges stochastically across members!
        assert np.allclose(t2m_arr[0, 0], t2m_arr[1, 0], atol=1e-6)
        assert not np.allclose(t2m_arr[0, 1], t2m_arr[1, 1], atol=1e-6)

    summary = summarize_run(str(run1_dir))
    assert "TMP2m" in summary["variables"]
    assert "surface_temperature" in summary["variables"]
    assert summary["variables"]["TMP2m"]["units"] == "K"
    assert [row["month"] for row in summary["variables"]["TMP2m"]["series"]] == ["2015-01", "2015-02", "2015-03"]
    assert all(row["counts"] == 1 for row in summary["variables"]["TMP2m"]["series"])

    # `restart.nc` passes `inspect_ic_file` and drives a 1-step continuation run at `2015-03-01`!
    r_info = inspect_ic_file(str(restart_nc))
    assert r_info.n_samples == 2
    assert r_info.timestamps == ("2015-03-01T00:00:00", "2015-03-01T00:00:00")
    assert len(r_info.variable_names) == 37

    run2_dir = allow_tmp_run_root / "md15_run2_cont"
    run2_dir.mkdir()
    res2 = run_md15(
        str(run2_dir),
        {
            **job1,
            "out_dir": str(run2_dir),
            "ic_file": str(restart_nc),
            "ic_sha256": sha256_path(restart_nc),
            "start": "2015-03-01T00:00:00",
            "steps": 1,
        },
    )
    assert res2["ic_samples_loaded"] == 2
    assert res2["first_month"] == "2015-03"

    # 4. Out-of-band `TMP2m` (~450 K) exits with `EXIT_UNSTABLE == 5`!
    with torch.no_grad():
        ref_model.decoder.out.bias.fill_(8.0)  # 8 * 21.5 + 277.6 = 449.6 K > 330 K!
    torch.save(ref_model.state_dict(), wt_path)
    wt_sha_bad = sha256_path(wt_path)
    unstable_dir = allow_tmp_run_root / "md15_unstable"
    unstable_dir.mkdir()
    with pytest.raises(SystemExit) as exc_info:
        run_md15(str(unstable_dir), {**job1, "out_dir": str(unstable_dir), "weights_sha256": wt_sha_bad})
    assert exc_info.value.code == EXIT_UNSTABLE == 5


def test_md15_emu_request_validation_and_prepare_exec_manifest_integrity(
    tmp_path: Path,
    allow_tmp_run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stat
    from aimip import services
    from aimip.emu import precheck_emu_request
    from aimip.errors import ConfigError
    from aimip.families import get_family_adapter
    from aimip.md15 import (
        MD15_COMMIT,
        MD15_CONFIG_REL_PATH,
        MD15_FORCING_WINDOW,
        MD15_PUBLISHED_STARTS,
        MD15_STATS_REL_PATH,
        MD15_WEIGHTS_REL_PATH,
    )
    from aimip.plan_types import Authorization, RunSpec
    from aimip.registry import _parse_model_entry
    from aimip.runner import exec_fme
    from aimip.services.approval import MemoryApprovalBroker, current_user

    data_dir = tmp_path / "data"
    pg_dir = tmp_path / "pg"
    (pg_dir / "bin").mkdir(parents=True)
    (pg_dir / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runs_dir = allow_tmp_run_root / "md15_svc_runs"
    runs_dir.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)

    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg_dir))
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(runs_dir))
    monkeypatch.setenv("AIMIP_STATE", str(state_dir))

    # Stage synthetic pinned MD-1.5 files + 37-file `code/monthly-diffusion` tree
    code_dir = data_dir / "code" / "monthly-diffusion"
    (code_dir / "md").mkdir(parents=True)
    for idx in range(37):
        (code_dir / "md" / f"f_{idx:02d}.py").write_text(f"# file {idx} ({MD15_COMMIT})\n", encoding="utf-8")
    real_tree_sha, real_tree_files = compute_sha256sum_tree_digest(str(code_dir))
    assert real_tree_files == 37
    monkeypatch.setattr("aimip.md15.MD15_TREE_SHA256", real_tree_sha)

    ic_rels = [f"inputs/md-1.5/ic/ic_{ts[:10]}.nc" for ts in MD15_PUBLISHED_STARTS]
    forcing_rel = "raw/zenodo-17065758/ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc"
    all_rels = [MD15_CONFIG_REL_PATH, MD15_WEIGHTS_REL_PATH, MD15_STATS_REL_PATH, *ic_rels, forcing_rel]

    lock_files: dict[str, dict] = {}
    for rel in all_rels:
        p = data_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"md15-pinned-{rel}".encode("utf-8"))
        lock_files[rel] = {
            "size": p.stat().st_size,
            "sha256": sha256_path(p),
            "source": {"kind": "url", "url": f"https://example.org/{Path(rel).name}"},
        }

    md_entry = {
        "name": "md-1.5",
        "family": "md",
        "fme_module": "aimip.emu_job",
        "calendar": "proleptic_gregorian",
        "checkpoint": f"{data_dir}/{MD15_WEIGHTS_REL_PATH}",
        "checkpoint_sha256": lock_files[MD15_WEIGHTS_REL_PATH]["sha256"],
        "ic_files": [f"{data_dir}/{r}" for r in ic_rels],
        "forcing_sets": [
            {
                "name": "amip-hist",
                "dir": f"{data_dir}/raw/zenodo-17065758",
                "window": list(MD15_FORCING_WINDOW),
            }
        ],
        "required_files": all_rels,
        "timestep_hours": 730,
        "stochastic": True,
        "default_n_ensemble_per_ic": 1,
        "notes": "Monthly Diffusion v0.9 at 1.5 deg (Hall & Molina, 2026).",
    }
    spec_obj = _parse_model_entry(md_entry, str(pg_dir), str(data_dir))

    # 1. Published starts accepted; unpublished start without `--ic` refused; custom `--ic` inside run root accepted!
    for pub_ts in MD15_PUBLISHED_STARTS:
        ok = precheck_emu_request(
            spec_obj,
            start_iso=pub_ts,
            steps=12,
            members=3,
            forcing_set_name=None,
            ic_path=None,
        )
        assert ok["start"] == pub_ts and ok["members"] == 3 and ok["scenario"] == "amip-hist"

    with pytest.raises(ConfigError, match="no published prebuilt initial state"):
        precheck_emu_request(
            spec_obj,
            start_iso="1980-01-01T00:00:00",
            steps=12,
            members=1,
            forcing_set_name=None,
            ic_path=None,
        )

    custom_ic = runs_dir / "custom_restart.nc"
    custom_ic.write_bytes(b"custom-ic-bytes")
    ok_custom = precheck_emu_request(
        spec_obj,
        start_iso="1980-01-01T00:00:00",
        steps=12,
        members=2,
        forcing_set_name="amip-hist",
        ic_path=str(custom_ic),
    )
    assert ok_custom["end"] == "1980-12-01T00:00:00"

    for bad_kw, match in (
        ({"start_iso": "1978-10-15T00:00:00"}, "YYYY-MM-01T00:00:00"),
        ({"start_iso": "1978-10-01T00:00:00", "steps": 557}, "at most 556 monthly steps"),
        ({"ocean_ic_path": str(custom_ic)}, "does not support --ocean-ic"),
    ):
        base_kw = {
            "start_iso": "1978-10-01T00:00:00",
            "steps": 12,
            "members": 1,
            "forcing_set_name": None,
            "ic_path": None,
        }
        base_kw.update(bad_kw)
        with pytest.raises(ConfigError, match=match):
            precheck_emu_request(spec_obj, **base_kw)

    # 2. `services.plan` -> `services.prepare` -> `exec_fme` verifies `code/monthly-diffusion` tree digest at job time!
    reg_file = data_dir / "registry.json"
    reg_file.write_text(json.dumps({"schema_version": 3, "generated_by": "test", "models": [md_entry]}), encoding="utf-8")
    lock_file = tmp_path / "md15.lock"
    lock_file.write_text(json.dumps({"schema_version": 1, "files": lock_files}), encoding="utf-8")
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
    fake_emu_py = tmp_path / "fake_md15_py"
    fake_emu_py.write_text(
        """#!/bin/bash
set -eu
[ "$1 $2 $3 $4" = "-I -m aimip.emu_job run" ] || exit 9
out="$5"
printf '{"months_written": {"atmosphere": 12}, "monthly_representation": "single monthly model state per step"}\\n' > "$out/emu-result.json"
""",
        encoding="utf-8",
    )
    fake_emu_py.chmod(fake_emu_py.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setenv("AIMIP_REGISTRY", str(reg_file))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_file))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_file))
    monkeypatch.setenv("AIMIP_EMU_PYTHON", str(fake_emu_py))

    run_spec = RunSpec(model="md-1.5", start="1978-10-01", length="1y", members=5, seed=42, name="md15-svc")
    resolved_plan = services.plan(run_spec)
    assert resolved_plan.requires_staging is False
    assert resolved_plan.required_approval_action == "submit"
    assert resolved_plan.steps == 12
    assert resolved_plan.end_time == "1979-09-01T00:00:00"

    now = 1_750_000_000.0
    broker = MemoryApprovalBroker(now_fn=lambda: now)
    auth = Authorization(
        action="submit",
        digest=resolved_plan.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="md15-prep-1",
        allow_download=False,
    )
    broker.store_minted(auth)
    prepared = services.prepare(run_spec, authorization=auth, require_authorization=False, broker=broker)
    assert prepared.manifest["family"] == "md"
    assert prepared.manifest["seed"] == 42

    # Tampering 1 byte in `code/monthly-diffusion` causes `exec_fme` to fail closed BEFORE running `emu_job`!
    f0 = code_dir / "md" / "f_00.py"
    orig_f0 = f0.read_bytes()
    f0.write_bytes(b"# tampered\n")
    with pytest.raises(ValidationError, match="Lock verification failed for code/monthly-diffusion"):
        exec_fme(prepared.run_dir)
    f0.write_bytes(orig_f0)

    # Setting an unapproved `AIMIP_MD15_CODE` override at `exec_fme` time is refused!
    unapproved_md_code = tmp_path / "unapproved_md15_code"
    unapproved_md_code.mkdir()
    monkeypatch.setenv("AIMIP_MD15_CODE", str(unapproved_md_code))
    try:
        with pytest.raises(ValidationError, match="Refusing unapproved source override AIMIP_MD15_CODE"):
            exec_fme(prepared.run_dir)
    finally:
        monkeypatch.delenv("AIMIP_MD15_CODE", raising=False)

    # Approved `AIMIP_MD15_CODE` override bound at `prepare` fails closed if edited before `exec_fme`!
    ovr_md_code = tmp_path / "approved_md15_ovr"
    shutil.copytree(code_dir, ovr_md_code)
    monkeypatch.setenv("AIMIP_MD15_CODE", str(ovr_md_code))
    try:
        spec_ovr = RunSpec(model="md-1.5", start="1978-10-01", length="1y", members=2, seed=42, name="md15-ovr-svc")
        rp_ovr = services.plan(spec_ovr)
        auth_ovr = Authorization(
            action="submit",
            digest=rp_ovr.digest,
            user=current_user(),
            issued_at=now,
            expires_at=now + 600,
            nonce="md15-prep-ovr",
            allow_download=False,
        )
        broker.store_minted(auth_ovr)
        prep_ovr = services.prepare(spec_ovr, authorization=auth_ovr, require_authorization=False, broker=broker)
        assert prep_ovr.manifest["code_is_override"] is True
        (ovr_md_code / "md" / "f_00.py").write_bytes(b"# edited after prepare\n")
        with pytest.raises(ValidationError, match="Approved source override tree changed at job time"):
            exec_fme(prep_ovr.run_dir)
    finally:
        monkeypatch.delenv("AIMIP_MD15_CODE", raising=False)

    # Restoring `code/monthly-diffusion` succeeds cleanly!
    done = exec_fme(prepared.run_dir)
    assert done["rc"] == 0
    assert done["result"]["months_written"] == {"atmosphere": 12}
    assert get_family_adapter("md").family_id == "md"


def test_md15_registered_in_default_catalog_card_and_cite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aimip import services

    empty_data = tmp_path / "empty_md15_data"
    empty_data.mkdir()
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)
    monkeypatch.delenv("AIMIP_LOCK", raising=False)
    monkeypatch.setenv("AIMIP_DATA", str(empty_data))

    md_models = services.models(family="md")
    assert [m["name"] for m in md_models] == ["md-1.5"]
    emu_models = services.models(family="emu")
    assert "md-1.5" in {m["name"] for m in emu_models}

    card = services.show("md-1.5", include_examples=True)
    assert card["family"] == "md"
    assert card["task"] == "amip"
    assert card["status"] == "runs"
    assert (
        card["verified_on"] is not None
        and "A100" in card["verified_on"]
        and "287.1284" in card["verified_on"]
        and "1.4.0" in card["verified_on"]
        and "job " not in card["verified_on"]
    )
    assert card["stochastic"] is True
    assert card["total_files"] == 5  # 4 locked files + code/monthly-diffusion
    assert "at least a decade" in card["limitations"]
    assert "encoder and decoder" in card["limitations"]
    assert "T=15" in card["limitations"]
    assert "counts=1" in card["limitations"]
    assert "no initial state" in card["limitations"]
    assert [fs["name"] for fs in card["forcing_sets"]] == ["aimip-era5-monthly"]

    cite_doc = services.cite("md-1.5")
    assert cite_doc["confirmed"] is True
    assert "2604.13481" in cite_doc["bibtex"]
    assert "Hall, Kyle J. C." in cite_doc["bibtex"]
    assert "Molina, Maria J." in cite_doc["bibtex"]

