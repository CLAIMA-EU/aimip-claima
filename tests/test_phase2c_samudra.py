"""Offline CPU test suite for Phase 2c `samudra-om4` and shared `safe_pickle` verification.

Covers:
1. Apache-2.0 provenance (`m2lines/Samudra@0c2a1bd8be6260fe02507fbb697ef50be8835fe6`), `THIRD_PARTY.md`,
   and contract-driven `show` / `cite` / `models` metadata.
2. `safe_pickle` O_NOFOLLOW pre-load SHA-256 gate (`open_verified_hash_stream`, `pickle_load_verified`,
   `torch_load_verified`, `extract_verified_zip_store`, `compute_tree_digest`) blocking 1-byte tamper
   and symlink swaps BEFORE `pickle.load` or `torch.load` is ever invoked.
3. Samudra request contract: even `--steps >= 2`, `--members == 1` (`seed1..seed5` explanation),
   `12:00:00` Julian timestamps, window bounds, and `--ic` / `aimip ic` refusal.
4. Julian `(year, month, day)` label preservation (zero 13-day shift), leap-year 6-day step tolerance,
   and exact overlap-day monthly aggregation (`w_d / days_in_month`).
5. Licensed `UNet` 178-tensor `state_dict` verification, offline `stage_samudra` + `run_samudra`,
   `degC` ocean monthly output, and ocean-only `summarize_run`.
6. Two-step OSN window staging (`requires_staging == True`), `prepare` vs `submit` token separation,
   and staged Zarr byte-tamper refusal in both `services.submit` and `exec_fme`.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import stat
import zipfile

import numpy as np
import pytest

from aimip import services
from aimip.cli import main
from aimip.emu import precheck_emu_request
from aimip.emu_catalog import registry_entries
from aimip.errors import ApprovalError, ConfigError, ValidationError
from aimip.plan_types import Authorization, RunSpec
from aimip.registry import _parse_model_entry
from aimip.safe_pickle import (
    _RestrictedConfigUnpickler,
    compute_sha256sum_tree_digest,
    compute_tree_digest,
    extract_verified_zip_store,
    pickle_load_verified,
    torch_load_verified,
)
from aimip.samudra import (
    SAMUDRA_BOUNDARY_VARS,
    SAMUDRA_BYTES_PER_STEP,
    SAMUDRA_CH_WIDTH,
    SAMUDRA_CODE_LICENCE,
    SAMUDRA_DEPTH_METERS,
    SAMUDRA_LICENSED_COMMIT,
    SAMUDRA_N_OUT,
    SAMUDRA_STATE_VARS,
    aggregate_samudra_5day_to_monthly,
    build_samudra_unet,
    check_samudra_window,
    julian_date_parts,
    julian_state_overlap_months,
    run_samudra,
    stage_samudra,
)
from aimip.services.approval import MemoryApprovalBroker, current_user
from aimip.status import summarize_run

from conftest import sha256_path


class _UnsupportedSamudraCheckpointObj:
    """Harmless custom Python object that `weights_only=True` must refuse to unpickle."""


def _zip_directory_bytes(src_dir: Path, *, prefix: str = "") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(p for p in src_dir.rglob("*") if p.is_file()):
            rel = fpath.relative_to(src_dir).as_posix()
            arcname = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
            zf.writestr(arcname, fpath.read_bytes())
    return buf.getvalue()


def test_samudra_apache2_provenance_and_card_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIMIP_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    repo_root = Path(__file__).resolve().parents[1]
    samudra_src = (repo_root / "src" / "aimip" / "samudra.py").read_text(encoding="utf-8")
    third_party = (repo_root / "THIRD_PARTY.md").read_text(encoding="utf-8")

    assert SAMUDRA_LICENSED_COMMIT == "0c2a1bd8be6260fe02507fbb697ef50be8835fe6"
    assert SAMUDRA_CODE_LICENCE == "Apache-2.0"
    assert "Copyright 2025 Surya Dheeshjith" in samudra_src
    assert "Apache License, Version 2.0" in samudra_src
    assert SAMUDRA_LICENSED_COMMIT in samudra_src
    assert "6f4e8ab4" not in samudra_src

    assert "m2lines/Samudra" in third_party
    assert SAMUDRA_LICENSED_COMMIT in third_party
    assert "Apache-2.0" in third_party

    card = services.show("samudra-om4", include_examples=True)
    assert card["family"] == "samudra"
    assert card["task"] == "ocean"
    assert card["status"] == "runs"
    assert (
        card["verified_on"] is not None
        and "A100" in card["verified_on"]
        and "18.1524" in card["verified_on"]
        and "1.4.0" in card["verified_on"]
        and "job " not in card["verified_on"]
    )
    assert "Apache-2.0" in card["licence"] and "MIT" in card["licence"]
    assert '"struggles to capture trends"' in card["limitations"]
    assert "AIMIP:" in card["limitations"]
    assert "compare with OM4" in card["limitations"]
    assert "2014-09-30" in card["limitations"]
    assert "degC" in card["limitations"]
    assert "cos(latitude)" in card["limitations"]
    assert "Dec 28 -> Jan 3" in card["limitations"] and "1/31" in card["limitations"]

    cite_doc = services.cite("samudra-om4")
    assert cite_doc["confirmed"] is True
    assert "10.1029/2024GL114318" in cite_doc["bibtex"]
    assert "2412.03795" in cite_doc["bibtex"]

    ocean_models = services.models(task="ocean")
    assert [m["name"] for m in ocean_models] == ["samudra-om4"]
    samudra_fam = services.models(family="samudra")
    assert [m["name"] for m in samudra_fam] == ["samudra-om4"]
    emu_fam = services.models(family="emu")
    assert "samudra-om4" in {m["name"] for m in emu_fam}


def test_safe_pickle_pre_load_sha256_gate_blocks_tamper_and_symlink_before_unpickling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. `pickle_load_verified`: spy on `_RestrictedConfigUnpickler.load` proves 1 call on valid load
    #    and 0 calls on 1-byte tamper, symlink swap, or group-writable file/directory!
    pkl_file = tmp_path / "config.pkl"
    payload = {"channels": 37, "grid": [121, 240]}
    pkl_bytes = pickle.dumps(payload)
    pkl_file.write_bytes(pkl_bytes)
    good_pkl_sha = hashlib.sha256(pkl_bytes).hexdigest()

    unpickler_calls = 0
    real_unpickler_load = _RestrictedConfigUnpickler.load

    def _spy_unpickler_load(self):
        nonlocal unpickler_calls
        unpickler_calls += 1
        return real_unpickler_load(self)

    monkeypatch.setattr(_RestrictedConfigUnpickler, "load", _spy_unpickler_load)

    assert pickle_load_verified(str(pkl_file), good_pkl_sha) == payload
    assert unpickler_calls == 1
    unpickler_calls = 0

    tampered_pkl = tmp_path / "tampered.pkl"
    tampered_pkl.write_bytes(pkl_bytes[:-1] + bytes([pkl_bytes[-1] ^ 0xFF]))
    with pytest.raises(ValidationError, match="SHA-256 mismatch before loading"):
        pickle_load_verified(str(tampered_pkl), good_pkl_sha)
    assert unpickler_calls == 0

    sym_pkl = tmp_path / "symlink_config.pkl"
    sym_pkl.symlink_to(pkl_file)
    with pytest.raises(ValidationError, match="symlink"):
        pickle_load_verified(str(sym_pkl), good_pkl_sha)
    assert unpickler_calls == 0

    gw_pkl = tmp_path / "group_writable.pkl"
    gw_pkl.write_bytes(pkl_bytes)
    gw_pkl.chmod(0o664)
    with pytest.raises(ValidationError, match="group- or world-writable"):
        pickle_load_verified(str(gw_pkl), good_pkl_sha)
    assert unpickler_calls == 0
    gw_pkl.chmod(0o644)

    gw_dir = tmp_path / "gw_parent"
    gw_dir.mkdir(mode=0o755)
    gw_child = gw_dir / "config.pkl"
    gw_child.write_bytes(pkl_bytes)
    gw_dir.chmod(0o775)
    try:
        with pytest.raises(ValidationError, match="group- or world-writable directory"):
            pickle_load_verified(str(gw_child), good_pkl_sha)
        assert unpickler_calls == 0
    finally:
        gw_dir.chmod(0o755)

    # Tree digest checks: top-level `.git` skipped, nested `.git` refused, group-writable subdir refused.
    tree_dir = tmp_path / "tree_perm_check"
    (tree_dir / ".git").mkdir(parents=True)
    (tree_dir / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (tree_dir / "sub").mkdir()
    (tree_dir / "sub" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    _, n_tree_files = compute_sha256sum_tree_digest(str(tree_dir))
    assert n_tree_files == 1

    (tree_dir / "sub").chmod(0o775)
    try:
        with pytest.raises(ValidationError, match="group- or world-writable directory"):
            compute_sha256sum_tree_digest(str(tree_dir))
        with pytest.raises(ValidationError, match="group- or world-writable directory"):
            compute_tree_digest(str(tree_dir / "sub"))
    finally:
        (tree_dir / "sub").chmod(0o755)

    (tree_dir / "sub" / ".git").mkdir()
    (tree_dir / "sub" / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    with pytest.raises(ValidationError, match=r"nested \.git"):
        compute_sha256sum_tree_digest(str(tree_dir))

    # 2. `extract_verified_zip_store`: verifies zip hash, handles nested Zarr root, refuses tamper & zip-slip
    zarr_dir = tmp_path / "src_zarr"
    zarr_dir.mkdir()
    (zarr_dir / ".zgroup").write_text('{"zarr_format": 2}\n', encoding="utf-8")
    (zarr_dir / ".zmetadata").write_text('{"metadata": {}}\n', encoding="utf-8")
    zip_bytes = _zip_directory_bytes(zarr_dir, prefix="OM4_means")
    zip_path = tmp_path / "OM4_means.zip"
    zip_path.write_bytes(zip_bytes)
    zip_sha = hashlib.sha256(zip_bytes).hexdigest()

    extracted_root = extract_verified_zip_store(str(zip_path), zip_sha, str(tmp_path / "ext_ok"))
    assert Path(extracted_root).name == "OM4_means"
    assert (Path(extracted_root) / ".zgroup").is_file()

    with pytest.raises(ValidationError, match="SHA-256 mismatch"):
        extract_verified_zip_store(str(zip_path), "0" * 64, str(tmp_path / "ext_bad_sha"))

    slip_buf = io.BytesIO()
    with zipfile.ZipFile(slip_buf, "w") as zf:
        zf.writestr("../escape.txt", "bad")
    slip_zip = tmp_path / "slip.zip"
    slip_zip.write_bytes(slip_buf.getvalue())
    slip_sha = hashlib.sha256(slip_buf.getvalue()).hexdigest()
    with pytest.raises(ValidationError, match="traversal"):
        extract_verified_zip_store(str(slip_zip), slip_sha, str(tmp_path / "ext_slip"))

    # 3. `torch_load_verified`: spy proves `torch.load` call count is 0 on mismatch or symlink
    torch = pytest.importorskip("torch")
    ckpt_file = tmp_path / "weights.pt"
    torch.save({"model": {"w": torch.ones(2, 2)}, "step": 5}, ckpt_file)
    ckpt_sha = sha256_path(ckpt_file)

    loaded = torch_load_verified(str(ckpt_file), ckpt_sha, weights_only=True)
    assert torch.equal(loaded["model"]["w"], torch.ones(2, 2))

    torch_calls = 0
    real_torch_load = torch.load

    def _spy_torch_load(*args, **kwargs):
        nonlocal torch_calls
        torch_calls += 1
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _spy_torch_load)

    bad_ckpt = tmp_path / "bad_weights.pt"
    raw_pt = ckpt_file.read_bytes()
    bad_ckpt.write_bytes(raw_pt[:16] + bytes([raw_pt[16] ^ 0x01]) + raw_pt[17:])
    with pytest.raises(ValidationError, match="SHA-256 mismatch before loading"):
        torch_load_verified(str(bad_ckpt), ckpt_sha)
    assert torch_calls == 0

    sym_ckpt = tmp_path / "sym_weights.pt"
    sym_ckpt.symlink_to(ckpt_file)
    with pytest.raises(ValidationError, match="symlink"):
        torch_load_verified(str(sym_ckpt), ckpt_sha)
    assert torch_calls == 0


def test_samudra_request_validation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from aimip.agent.tools import AgentTools

    monkeypatch.setenv("AIMIP_DATA", str(tmp_path))
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    raw_entry = next(e for e in registry_entries(str(tmp_path)) if e["name"] == "samudra-om4")
    spec = _parse_model_entry(raw_entry, str(tmp_path / "pg"), str(tmp_path))

    ok = precheck_emu_request(
        spec,
        start_iso="2014-10-10T12:00:00",
        steps=14,
        members=None,
        forcing_set_name=None,
        ic_path=None,
    )
    assert ok == {
        "start": "2014-10-10T12:00:00",
        "steps": 14,
        "members": 1,
        "scenario": "om4-jra55do-replay",
        "end": "2014-12-19T12:00:00",
    }

    for bad_kwargs, match in (
        ({"steps": 1}, "even integer >= 2"),
        ({"steps": 15}, "even integer >= 2"),
        ({"members": 2}, "seed1..seed5"),
        ({"start_iso": "2014-10-10T00:00:00"}, "12:00:00"),
        ({"start_iso": "1975-01-03T12:00:00"}, "outside the OM4 replay window"),
        ({"start_iso": "2022-12-24T12:00:00", "steps": 2}, "exceeds the OM4 replay window end"),
        ({"ic_path": "/scratch/u/restart.nc"}, "does not support custom --ic"),
        ({"forcing_set_name": "ssp585"}, "accepts no custom forcing"),
    ):
        call_args = {
            "start_iso": "2014-10-10T12:00:00",
            "steps": 14,
            "members": None,
            "forcing_set_name": None,
            "ic_path": None,
        }
        call_args.update(bad_kwargs)
        with pytest.raises(ConfigError, match=match):
            precheck_emu_request(spec, **call_args)

    assert main(["ic", "list", "--model", "samudra-om4"]) != 0
    assert "OM4 ocean window" in capsys.readouterr().err

    with pytest.raises(ValueError, match="OM4 ocean window"):
        AgentTools().list_ics("samudra-om4")


def test_samudra_julian_labels_leap_cadence_and_overlap_day_monthly_aggregation() -> None:
    cftime = pytest.importorskip("cftime")

    # 1. Direct `(year, month, day)` extraction without 13-day Julian->Gregorian day-number shift
    t_start = cftime.DatetimeJulian(2014, 10, 10, 12, 0, 0)
    assert julian_date_parts(t_start) == (2014, 10, 10)
    assert julian_state_overlap_months(2014, 10, 30) == [("2014-10", 4), ("2014-11", 1)]
    assert julian_state_overlap_months(2014, 11, 29) == [("2014-11", 4), ("2014-12", 1)]

    # 2. 14 output 5-day states after 2014-10-10T12:00:00 -> Oct (19d), Nov (30d complete), Dec (21d)
    import datetime as dt_mod

    out_times = [t_start + dt_mod.timedelta(days=5 * (k + 1)) for k in range(14)]
    vals = np.zeros((14, 2, 2), dtype=np.float32)
    for k, ts in enumerate(out_times):
        y, m, d = julian_date_parts(ts)
        if (y, m, d) == (2014, 10, 30):
            vals[k] = 10.0  # 1 day in Nov
        elif (y, m) == (2014, 11) and d in (4, 9, 14, 19, 24):
            vals[k] = 20.0  # 25 days in Nov
        elif (y, m, d) == (2014, 11, 29):
            vals[k] = 35.0  # 4 days in Nov
        else:
            vals[k] = 5.0

    all_months = aggregate_samudra_5day_to_monthly(
        out_times, {"thetao_0": vals}, prev_time=t_start, complete_months_only=False
    )
    assert [(r["month"], r["covered_days"], r["days_in_month"]) for r in all_months] == [
        ("2014-10", 19, 31),
        ("2014-11", 30, 30),
        ("2014-12", 21, 31),
    ]

    complete_only = aggregate_samudra_5day_to_monthly(
        out_times, {"thetao_0": vals}, prev_time=t_start, complete_months_only=True
    )
    assert len(complete_only) == 1
    nov = complete_only[0]
    assert nov["month"] == "2014-11" and nov["covered_days"] == 30
    expected_nov = (1.0 * 10.0 + 25.0 * 20.0 + 4.0 * 35.0) / 30.0
    assert np.allclose(nov["fields"]["thetao_0"], expected_nov, atol=1e-5)

    # 3a. Real OSN Feb-Mar 2016 sequence has uniform 5-day steps (`2016-02-27 -> 2016-03-03`) and covers 29 + 31 days.
    feb_prev = cftime.DatetimeJulian(2016, 1, 28, 12, 0, 0)
    feb_mar_5d = [
        cftime.DatetimeJulian(2016, 2, 2, 12, 0, 0),
        cftime.DatetimeJulian(2016, 2, 7, 12, 0, 0),
        cftime.DatetimeJulian(2016, 2, 12, 12, 0, 0),
        cftime.DatetimeJulian(2016, 2, 17, 12, 0, 0),
        cftime.DatetimeJulian(2016, 2, 22, 12, 0, 0),
        cftime.DatetimeJulian(2016, 2, 27, 12, 0, 0),
        cftime.DatetimeJulian(2016, 3, 3, 12, 0, 0),
        cftime.DatetimeJulian(2016, 3, 8, 12, 0, 0),
        cftime.DatetimeJulian(2016, 3, 13, 12, 0, 0),
        cftime.DatetimeJulian(2016, 3, 18, 12, 0, 0),
        cftime.DatetimeJulian(2016, 3, 23, 12, 0, 0),
        cftime.DatetimeJulian(2016, 3, 28, 12, 0, 0),
        cftime.DatetimeJulian(2016, 4, 2, 12, 0, 0),
    ]
    feb_mar_rows = aggregate_samudra_5day_to_monthly(
        feb_mar_5d,
        {"thetao_0": np.full((len(feb_mar_5d), 2, 2), 14.5, dtype=np.float32)},
        prev_time=feb_prev,
        complete_months_only=True,
    )
    assert [(r["month"], r["covered_days"]) for r in feb_mar_rows] == [("2016-02", 29), ("2016-03", 31)]

    # 3b. Real OSN leap-year-end 6-day step (`2016-12-28 12:00` -> `2017-01-03 12:00`):
    #     passes `check_samudra_window`, gives Dec 31 weight `1/31` from the `2017-01-03` state,
    #     and covers all 31 days of Dec 2016 and Jan 2017.
    assert julian_state_overlap_months(2017, 1, 3, span_days=6) == [("2016-12", 1), ("2017-01", 5)]
    leap_end_prev = cftime.DatetimeJulian(2016, 11, 28, 12, 0, 0)
    leap_dec_jan = [
        cftime.DatetimeJulian(2016, 12, 3, 12, 0, 0),
        cftime.DatetimeJulian(2016, 12, 8, 12, 0, 0),
        cftime.DatetimeJulian(2016, 12, 13, 12, 0, 0),
        cftime.DatetimeJulian(2016, 12, 18, 12, 0, 0),
        cftime.DatetimeJulian(2016, 12, 23, 12, 0, 0),
        cftime.DatetimeJulian(2016, 12, 28, 12, 0, 0),
        cftime.DatetimeJulian(2017, 1, 3, 12, 0, 0),   # Real OSN 6-day leap-year-end step!
        cftime.DatetimeJulian(2017, 1, 8, 12, 0, 0),
        cftime.DatetimeJulian(2017, 1, 13, 12, 0, 0),
        cftime.DatetimeJulian(2017, 1, 18, 12, 0, 0),
        cftime.DatetimeJulian(2017, 1, 23, 12, 0, 0),
        cftime.DatetimeJulian(2017, 1, 28, 12, 0, 0),
        cftime.DatetimeJulian(2017, 2, 2, 12, 0, 0),
    ]
    win_times = [cftime.DatetimeJulian(2016, 11, 23, 12, 0, 0), leap_end_prev, *leap_dec_jan[:12]]
    offsets = [(t - win_times[0]).days + (t - win_times[0]).seconds / 86400.0 for t in win_times]
    check_samudra_window(
        {"start": "2016-11-28T12:00:00", "calendar": "julian"},
        [str(t) for t in win_times],
        offsets,
        "2016-11-28T12:00:00",
        12,
    )
    dec_jan_vals = np.full((len(leap_dec_jan), 2, 2), 10.0, dtype=np.float32)
    dec_jan_vals[6] = 41.0  # 2017-01-03 state contributes 1/31 to 2016-12 -> (30*10 + 1*41)/31 == 11.0
    leap_rows = aggregate_samudra_5day_to_monthly(
        leap_dec_jan,
        {"thetao_0": dec_jan_vals},
        prev_time=leap_end_prev,
        complete_months_only=True,
    )
    assert [(r["month"], r["covered_days"]) for r in leap_rows] == [("2016-12", 31), ("2017-01", 31)]
    assert np.allclose(leap_rows[0]["fields"]["thetao_0"], 11.0, atol=1e-5)


def _build_synthetic_om4_source(root_dir: Path) -> tuple[Path, Path, Path]:
    """Create a tiny offline Zarr v2 OM4 store (16 Julian 5-day steps, 16x32 grid) + means/stds zips."""
    import datetime as dt_mod
    import cftime
    import xarray as xr

    y_len, x_len, n_lev = 16, 32, len(SAMUDRA_DEPTH_METERS)
    t0 = cftime.DatetimeJulian(2014, 10, 5, 12, 0, 0)
    times = [t0 + dt_mod.timedelta(days=5 * k) for k in range(16)]  # 2 inputs + 14 targets

    lat_1d = np.linspace(-84.375, 84.375, y_len, dtype=np.float32)
    lon_1d = np.linspace(5.625, 354.375, x_len, dtype=np.float32)
    lat_2d, lon_2d = np.meshgrid(lat_1d, lon_1d, indexing="ij")

    wetmask = np.ones((n_lev, y_len, x_len), dtype=np.float32)
    wetmask[:, 0, :] = 0.0  # Polar land row -> must become NaN in output

    data_vars: dict[str, xr.DataArray] = {
        "wetmask": xr.DataArray(wetmask, dims=("lev", "y", "x")),
    }
    means_vars: dict[str, xr.DataArray] = {}
    stds_vars: dict[str, xr.DataArray] = {}

    for vname in (*SAMUDRA_STATE_VARS, *SAMUDRA_BOUNDARY_VARS):
        base = 15.0 if vname.startswith("thetao") else (34.5 if vname.startswith("so") else 0.1)
        arr = np.full((len(times), y_len, x_len), base, dtype=np.float32)
        data_vars[vname] = xr.DataArray(arr, dims=("time", "y", "x"))
        means_vars[vname] = xr.DataArray(np.float32(base))
        stds_vars[vname] = xr.DataArray(np.float32(1.0))

    ds = xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "lev": np.asarray(SAMUDRA_DEPTH_METERS, dtype=np.float32),
            "y": lat_1d,
            "x": lon_1d,
            "lat": (("y", "x"), lat_2d),
            "lon": (("y", "x"), lon_2d),
        },
    )
    ds["time"].encoding = {"units": "days since 1975-01-03 12:00:00", "calendar": "julian", "dtype": "int64"}
    om4_store = root_dir / "osn" / "OM4"
    om4_store.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(str(om4_store), zarr_format=2, consolidated=True)

    means_tmp = root_dir / "means_zarr"
    stds_tmp = root_dir / "stds_zarr"
    xr.Dataset(means_vars).to_zarr(str(means_tmp), zarr_format=2, consolidated=True)
    xr.Dataset(stds_vars).to_zarr(str(stds_tmp), zarr_format=2, consolidated=True)

    means_zip = root_dir / "OM4_means.zip"
    stds_zip = root_dir / "OM4_stds.zip"
    means_zip.write_bytes(_zip_directory_bytes(means_tmp, prefix="OM4_means"))
    stds_zip.write_bytes(_zip_directory_bytes(stds_tmp, prefix="OM4_stds"))
    return root_dir / "osn", means_zip, stds_zip


def test_samudra_unet_178_state_dict_keys_and_end_to_end_stage_run_and_summary(
    tmp_path: Path,
    allow_tmp_run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    import aimip.samudra as samudra_mod
    import netCDF4

    # 1. Verify full-width `UNet` from `0c2a1bd8` matches the cluster-inspected 178-tensor checkpoint layout
    #    on the native 180x360 OM4 grid (360 -> 180 -> 90 -> 45 -> 22 bottleneck width > dilation pad 8).
    wet_full = torch.ones(SAMUDRA_N_OUT, 180, 360)
    wet_full[:, 0, :] = 0.0
    full_unet = build_samudra_unet(wet_full).eval()
    sd = full_unet.state_dict()
    assert len(sd) == 178
    assert tuple(sd["layers.0.skip_module.weight"].shape) == (200, 158, 1, 1)
    assert tuple(sd["layers.0.convblock.0.weight"].shape) == (632, 158, 3, 3)
    assert tuple(sd["layers.17.weight"].shape) == (154, 200, 3, 3)
    with torch.inference_mode():
        out_y = full_unet(torch.zeros(1, SAMUDRA_CH_WIDTH[0], 180, 360))
    assert tuple(out_y.shape) == (1, SAMUDRA_N_OUT, 180, 360)
    assert float(out_y[:, :, 0, :].abs().max()) == 0.0

    # 2. End-to-end offline `stage_samudra` + `run_samudra` + `summarize_run` (compact width for fast CPU CI)
    osn_dir, means_zip, stds_zip = _build_synthetic_om4_source(tmp_path / "src")
    compact_width = (158, 16, 16)
    compact_dil = (1, 2)
    monkeypatch.setattr(
        samudra_mod,
        "build_samudra_unet",
        lambda wet: build_samudra_unet(wet, ch_width=compact_width, dilation=compact_dil),
    )
    compact_model = samudra_mod.build_samudra_unet(torch.ones(SAMUDRA_N_OUT, 16, 32))
    for p in compact_model.parameters():
        torch.nn.init.zeros_(p)
    ckpt_path = tmp_path / "samudra_seed1.pt"
    torch.save({"model": compact_model.state_dict(), "epoch": 1}, ckpt_path)
    ckpt_sha = sha256_path(ckpt_path)

    run_dir = allow_tmp_run_root / "samudra_e2e"
    run_dir.mkdir()
    job = {
        "family": "samudra",
        "model": "samudra-om4",
        "out_dir": str(run_dir),
        "window_dir": str(run_dir / "window"),
        "osn_source": str(osn_dir),
        "start": "2014-10-10T12:00:00",
        "steps": 14,
        "device": "cpu",
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "means_zip": str(means_zip),
        "means_zip_sha256": sha256_path(means_zip),
        "stds_zip": str(stds_zip),
        "stds_zip_sha256": sha256_path(stds_zip),
    }

    facts = stage_samudra(str(run_dir), job)
    assert facts["steps"] == 14 and facts["input_steps"] == 2 and facts["calendar"] == "julian"
    res = run_samudra(str(run_dir), job)
    assert res["months_written"] == {"ocean": 1}
    assert res["first_month"] == "2014-11"
    assert res["in_training_period"] is False

    ocean_nc = run_dir / "ocean" / "monthly_mean_predictions.nc"
    thetao_nc = run_dir / "ocean" / "monthly_thetao.nc"
    assert ocean_nc.is_file() and thetao_nc.is_file()
    with netCDF4.Dataset(str(ocean_nc), "r") as ds:
        assert ds.variables["thetao_0"].units == "degC"
        assert "Dec 28 -> Jan 3" in ds.leap_year_note and "1/31" in ds.leap_year_note
        assert int(ds.variables["counts"][0, 0]) == 30
        arr0 = np.asarray(ds.variables["thetao_0"][0, 0])
        assert np.isnan(arr0[0, :]).all()
        assert np.allclose(arr0[1:, :], 15.0, atol=1e-4)

    summary = summarize_run(str(run_dir))
    assert "ocean.thetao_0" in summary["variables"]
    t0_meta = summary["variables"]["ocean.thetao_0"]
    assert t0_meta["units"] == "degC"
    assert len(t0_meta["series"]) == 1
    assert t0_meta["series"][0]["month"] == "2014-11"
    assert t0_meta["series"][0]["counts"] == 30
    assert abs(float(t0_meta["series"][0]["ensemble_mean"]) - 15.0) < 1e-4

    # 3. Corrupting 1 byte in checkpoint causes `run_samudra` to fail BEFORE `torch.load`
    bad_run = allow_tmp_run_root / "samudra_bad_ckpt"
    bad_run.mkdir()
    stage_samudra(str(bad_run), {**job, "out_dir": str(bad_run), "window_dir": str(bad_run / "window")})
    raw_bytes = ckpt_path.read_bytes()
    ckpt_path.write_bytes(raw_bytes[:-1] + bytes([raw_bytes[-1] ^ 0xFF]))
    with pytest.raises(ValidationError, match="SHA-256 mismatch before loading"):
        run_samudra(str(bad_run), {**job, "out_dir": str(bad_run), "window_dir": str(bad_run / "window")})

    # 4. A checkpoint carrying a harmless unsupported Python object is refused by `run_samudra` (`weights_only=True`, no pickle fallback) even when its SHA-256 matches!
    obj_ckpt = tmp_path / "samudra_obj.pt"
    torch.save({"model": compact_model.state_dict(), "extra": _UnsupportedSamudraCheckpointObj()}, str(obj_ckpt))
    obj_ckpt_sha = sha256_path(obj_ckpt)
    obj_run = allow_tmp_run_root / "samudra_obj_ckpt"
    obj_run.mkdir()
    stage_samudra(str(obj_run), {**job, "out_dir": str(obj_run), "window_dir": str(obj_run / "window")})
    with pytest.raises(pickle.UnpicklingError):
        run_samudra(
            str(obj_run),
            {
                **job,
                "out_dir": str(obj_run),
                "window_dir": str(obj_run / "window"),
                "checkpoint": str(obj_ckpt),
                "checkpoint_sha256": obj_ckpt_sha,
            },
        )


def test_samudra_two_step_staging_approval_and_zarr_tamper_refusal(
    tmp_path: Path,
    allow_tmp_run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aimip.runner import exec_fme
    from aimip.services import core

    data_dir = tmp_path / "data"
    pg_dir = tmp_path / "pg"
    (pg_dir / "bin").mkdir(parents=True)
    (pg_dir / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    runs_dir = allow_tmp_run_root / "samudra_runs"
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

    # Fake emulator python stages a deterministic Zarr v2 window tree on `stage` and writes emu-result.json on `run`
    fake_emu_py = tmp_path / "fake_emu_py"
    fake_emu_py.write_text(
        """#!/bin/bash
set -eu
[ "$1 $2 $3" = "-I -m aimip.emu_job" ] || exit 9
mode="$4"; out="$5"
case "$mode" in
  stage)
    mkdir -p "$out/window/OM4.zarr/thetao_lev_2_5"
    printf '{"zarr_format": 2}\\n' > "$out/window/OM4.zarr/.zgroup"
    printf 'chunk-bytes-v1' > "$out/window/OM4.zarr/thetao_lev_2_5/0.0.0"
    printf '{"start": "2014-10-10T12:00:00", "steps": 14, "input_steps": 2, "calendar": "julian"}\\n' > "$out/window/window.json"
    ;;
  run)
    printf '{"months_written": {"ocean": 1}}\\n' > "$out/emu-result.json"
    ;;
  *) exit 7 ;;
esac
""",
        encoding="utf-8",
    )
    fake_emu_py.chmod(fake_emu_py.stat().st_mode | stat.S_IXUSR)

    samudra_entry = next(e for e in registry_entries(str(data_dir)) if e["name"] == "samudra-om4")
    lock_files: dict[str, dict] = {}
    for rel in samudra_entry["required_files"]:
        p = data_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"pinned-{rel}".encode("utf-8"))
        lock_files[rel] = {
            "size": p.stat().st_size,
            "sha256": sha256_path(p),
            "source": {"kind": "hf", "repo": "M2LInES/Samudra", "path": Path(rel).name, "revision": "a" * 40},
        }
    ckpt_sha = sha256_path(Path(samudra_entry["checkpoint"]))
    reg_file = data_dir / "registry.json"
    reg_file.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "generated_by": "test",
                "models": [{**samudra_entry, "checkpoint_sha256": ckpt_sha}],
            }
        ),
        encoding="utf-8",
    )
    lock_path = tmp_path / "samudra.lock"
    lock_path.write_text(json.dumps({"schema_version": 1, "files": lock_files}), encoding="utf-8")

    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg_dir))
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_REGISTRY", str(reg_file))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_path))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_path))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(runs_dir))
    monkeypatch.setenv("AIMIP_STATE", str(state_dir))
    monkeypatch.setenv("AIMIP_EMU_PYTHON", str(fake_emu_py))

    spec = RunSpec(model="samudra-om4", start="2014-10-10T12:00:00", steps=14, name="sam-2step")
    pre_plan = services.plan(spec)
    assert pre_plan.requires_staging is True
    assert pre_plan.required_approval_action == "prepare"
    assert pre_plan.missing_bytes == (14 + 2) * SAMUDRA_BYTES_PER_STEP
    card_text = services.render_plan_card(pre_plan, is_tty=False)
    assert "OSN (m2lines-pubs/Samudra)" in card_text

    now = 1_750_000_000.0
    broker = MemoryApprovalBroker(now_fn=lambda: now)
    prep_auth = Authorization(
        action="prepare",
        digest=pre_plan.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="samudra-prep-1",
        allow_download=True,
    )
    broker.store_minted(prep_auth)
    prepared = services.prepare(spec, authorization=prep_auth, require_authorization=True, broker=broker)
    assert prepared.plan.requires_staging is False
    assert prepared.plan.required_approval_action == "submit"
    assert prepared.plan.digest != pre_plan.digest

    win_sha, _, _ = compute_tree_digest(str(Path(prepared.run_dir) / "window"))
    assert prepared.manifest["window_tree_sha256"] == win_sha

    # Prepare token can NEVER submit
    forged_prep = Authorization(
        action="prepare",
        digest=prepared.plan.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="samudra-forged-prep",
        allow_download=False,
    )
    broker.store_minted(forged_prep)
    with pytest.raises(ApprovalError, match="expected 'submit'"):
        services.submit(prepared, authorization=forged_prep, require_authorization=True, broker=broker)

    # Mutating 1 byte inside staged `<out>/window/OM4.zarr` invalidates `plan.json` for `submit` AND `exec_fme`!
    chunk_file = Path(prepared.run_dir) / "window" / "OM4.zarr" / "thetao_lev_2_5" / "0.0.0"
    orig_chunk = chunk_file.read_bytes()
    chunk_file.write_bytes(b"tampered-zarr-chunk")

    sub_auth = Authorization(
        action="submit",
        digest=prepared.plan.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="samudra-sub-1",
        allow_download=False,
    )
    broker.store_minted(sub_auth)
    with pytest.raises(ApprovalError, match="Plan changed, approve again"):
        services.submit(prepared, authorization=sub_auth, require_authorization=True, broker=broker)
    with pytest.raises(ValidationError, match="Staged Samudra OM4 window changed at job time"):
        exec_fme(prepared.run_dir)

    # Restoring the exact staged chunk allows `exec_fme` and `submit` with a fresh submit token to succeed!
    chunk_file.write_bytes(orig_chunk)
    sub_auth_ok = Authorization(
        action="submit",
        digest=prepared.plan.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="samudra-sub-2",
        allow_download=False,
    )
    broker.store_minted(sub_auth_ok)
    submitted: list[str] = []
    monkeypatch.setattr(core.shutil, "which", lambda name: "/fake/" + name)
    monkeypatch.setattr(
        core,
        "submit_run",
        lambda rd, **kw: submitted.append(rd) or {"slurm_job_id": "987654", "scheduler": "slurm"},
    )
    attempt = services.submit(prepared, authorization=sub_auth_ok, require_authorization=True, broker=broker)
    assert submitted == [prepared.run_dir]
    assert attempt.attempt_id == "987654"

    done_manifest = exec_fme(prepared.run_dir)
    assert done_manifest["rc"] == 0
    assert done_manifest["result"] == {"months_written": {"ocean": 1}}
