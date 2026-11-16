"""Tests for `forcing_sets` selection, cross-set split guidance, `ic spin`, and `aimip collect`."""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml
from xarray.coding.times import CFDatetimeCoder

from aimip.cli import main
from aimip.errors import ConfigError, ValidationError
from aimip.ic import collect_spin, inspect_ic_file, prepare_spin
from aimip.registry import ForcingSet, get_model
from aimip.time_utils import check_within_window, compute_spin_steps, select_forcing_set

from conftest import write_fake_ic_netcdf


def test_window_check_and_forcing_set_selection_and_cross_set_refusal() -> None:
    window = ("1980-01-01T00:00:00", "1981-01-01T00:00:00")
    assert (
        check_within_window(
            "1980-01-01T00:00:00", 1464, 6, window, calendar="proleptic_gregorian"
        )
        == "1981-01-01T00:00:00"
    )
    with pytest.raises(ConfigError, match="maximum steps from 1980-01-01T00:00:00 is 1464"):
        check_within_window(
            "1980-01-01T00:00:00", 1465, 6, window, calendar="proleptic_gregorian"
        )
    with pytest.raises(ConfigError, match="maximum steps from 1980-01-01T00:00:00 is 1460"):
        check_within_window("1980-01-01T00:00:00", 1464, 6, window, calendar="noleap")

    sets = [
        ForcingSet(
            name="amip-hist",
            dir="/opt/aimip/inputs/shield/amip-hist",
            window=("1979-01-01T00:00:00", "2014-12-31T18:00:00"),
        ),
        ForcingSet(
            name="amip-oos",
            dir="/opt/aimip/inputs/shield/amip-oos",
            window=("2014-01-01T00:00:00", "2024-12-31T18:00:00"),
        ),
    ]

    # 1) Pure amip-hist run picks amip-hist.
    fs_hist, end_hist = select_forcing_set(
        "1979-01-01T00:00:00", 1460, 6, sets, model_name="ace2-shield"
    )
    assert fs_hist.name == "amip-hist"
    assert end_hist == "1980-01-01T00:00:00"

    # 2) Pure amip-oos run picks amip-oos.
    fs_oos, end_oos = select_forcing_set(
        "2014-01-01T00:00:00", 2920, 6, sets, model_name="ace2-shield"
    )
    assert fs_oos.name == "amip-oos"
    assert end_oos == "2016-01-01T00:00:00"

    # 3) Run crossing from 2013-01-01 (amip-hist) to 2016-01-01 (amip-oos) is refused
    # with explicit instructions to split the run and continue from `restart.nc`.
    with pytest.raises(ConfigError) as exc_info:
        select_forcing_set(
            "2013-01-01T00:00:00", 4380, 6, sets, model_name="ace2-shield"
        )
    msg = str(exc_info.value)
    assert "crosses forcing sets (amip-hist -> amip-oos" in msg
    assert "restart.nc" in msg
    assert "--ic <seg1-dir>/restart.nc" in msg


def test_spin_step_arithmetic_across_leap_century_and_noleap_calendars() -> None:
    assert (
        compute_spin_steps(
            "1979-01-01T00:00:00",
            "1980-01-01T00:00:00",
            6,
            calendar="proleptic_gregorian",
        )
        == 1460
    )
    assert (
        compute_spin_steps(
            "1980-01-01T00:00:00",
            "1981-01-01T00:00:00",
            6,
            calendar="proleptic_gregorian",
        )
        == 1464
    )
    assert (
        compute_spin_steps(
            "2000-01-01T00:00:00",
            "2001-01-01T00:00:00",
            6,
            calendar="proleptic_gregorian",
        )
        == 1464
    )
    assert (
        compute_spin_steps(
            "1996-01-01T00:00:00",
            "1997-01-01T00:00:00",
            6,
            calendar="noleap",
        )
        == 1460
    )
    with pytest.raises(ValidationError, match="noleap"):
        compute_spin_steps(
            "1996-02-29T00:00:00",
            "1996-03-01T00:00:00",
            6,
            calendar="noleap",
        )
    with pytest.raises(ConfigError, match="strictly earlier"):
        compute_spin_steps("1980-06-01T00:00:00", "1980-06-01T00:00:00", 6)
    with pytest.raises(ConfigError, match="not an exact multiple"):
        compute_spin_steps("1980-01-01T00:00:00", "1980-01-01T03:00:00", 6)


def test_ic_list_ic_spin_and_collect_heterogeneous_dims(
    playground_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_dir: pathlib.Path = playground_env["runs_dir"]
    pg_dir: pathlib.Path = playground_env["playground_dir"]

    # 1. `aimip ic list -m ace2-era5` reads BOTH `ic_1979.nc` and `ic_2001.nc` (`latitude/longitude`).
    assert main(["ic", "list", "-m", "ace2-era5"]) == 0
    list_out = capsys.readouterr().out
    assert "1979-01-01T00:00:00" in list_out
    assert "2001-02-01T00:00:00" in list_out
    assert "latitude/longitude" in list_out

    # 2. `aimip ic spin` across leap boundary (1979-01-01, 1979-01-02, 1980-01-01 -> 1980-03-01).
    spin_dir = runs_dir / "spin_19800301"
    rc = main(
        [
            "ic",
            "spin",
            "-m",
            "ace2-era5",
            "--from",
            "1979-01-01T00:00:00,1979-01-02T00:00:00,1980-01-01T00:00:00",
            "--to",
            "1980-03-01T00:00:00",
            "--out",
            str(spin_dir),
            "--submit",
        ]
    )
    assert rc == 0
    spin_stdout = capsys.readouterr().out
    assert f"Next:     aimip submit {spin_dir}" in spin_stdout
    assert f"aimip collect {spin_dir}" in spin_stdout
    assert "--collect" in spin_stdout

    spin_manifest = json.loads((spin_dir / "spin-manifest.json").read_text(encoding="utf-8"))
    assert spin_manifest["member_count"] == 3
    assert spin_manifest["sbatch_path"] == f"{spin_dir}/job.sbatch"
    assert [m["steps"] for m in spin_manifest["members"]] == [1700, 1696, 240]

    wrapper_guard_re = re.compile(
        rf'^(#|$|set -euo pipefail$|"{re.escape(str(pg_dir))}/bin/aimip" exec-fme "/[A-Za-z0-9._/+-]+"$)'
    )
    top_sbatch_lines = (spin_dir / "job.sbatch").read_text(encoding="utf-8").splitlines()
    assert top_sbatch_lines[1] == "# aimip-generated v1"
    assert [ln for ln in top_sbatch_lines if not wrapper_guard_re.match(ln)] == []
    top_body = [ln for ln in top_sbatch_lines if ln.strip() and not ln.startswith("#")]
    assert top_body == [
        "set -euo pipefail",
        f'"{pg_dir}/bin/aimip" exec-fme "{spin_dir}/0"',
        f'"{pg_dir}/bin/aimip" exec-fme "{spin_dir}/1"',
        f'"{pg_dir}/bin/aimip" exec-fme "{spin_dir}/2"',
    ]

    for idx, expected_steps in enumerate([1700, 1696, 240]):
        m_dir = spin_dir / str(idx)
        cfg = yaml.safe_load((m_dir / "config.yaml").read_text(encoding="utf-8"))
        assert cfg["n_forward_steps"] == expected_steps
        sbatch_lines = (m_dir / "job.sbatch").read_text(encoding="utf-8").splitlines()
        assert sbatch_lines[1] == "# aimip-generated v1"
        assert [ln for ln in sbatch_lines if not wrapper_guard_re.match(ln)] == []
        body = [ln for ln in sbatch_lines if ln.strip() and not ln.startswith("#")]
        assert body == [
            "set -euo pipefail",
            f'"{pg_dir}/bin/aimip" exec-fme "{m_dir}"',
        ]

        # Note fme writes `restart.nc` with `lat`/`lon` even though `ace2-era5` shipped IC
        # uses `latitude`/`longitude` (`LatLonCoordinates.dims == ["lat", "lon"]`).
        write_fake_ic_netcdf(
            m_dir / "restart.nc",
            ["1980-03-01T00:00:00"],
            calendar="proleptic_gregorian",
            sample_dim="sample",
            spatial_dims=("lat", "lon"),
            seed=100 + idx,
        )

    # Verify `aimip status <spin_dir>` reports completed spin status cleanly.
    (spin_dir / "slurm.jobid").write_text("999001\n", encoding="utf-8")
    assert main(["status", str(spin_dir), "--json"]) == 0
    spin_status = json.loads(capsys.readouterr().out)
    assert spin_status["state"] == "completed"
    assert spin_status["spin_members_completed"] == 3
    assert spin_status["slurm_job_id"] == "999001"

    # 3. `aimip collect <spin_dir> --out <out_ic>` merges restarts and preserves shipped IC
    # horizontal dim names (`latitude`, `longitude`) with zero orphan coordinates!
    out_ic = runs_dir / "spun_19800301.nc"
    rc_collect = main(["collect", str(spin_dir), "--out", str(out_ic)])
    assert rc_collect == 0
    capsys.readouterr()

    # Also verify `aimip ic spin --collect --spin-dir <spin_dir> --out <out_ic2>`.
    out_ic_alt = runs_dir / "spun_19800301_alt.nc"
    assert (
        main(
            [
                "ic",
                "spin",
                "--collect",
                "--spin-dir",
                str(spin_dir),
                "--out",
                str(out_ic_alt),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert inspect_ic_file(str(out_ic_alt)).n_samples == 3

    info = inspect_ic_file(str(out_ic))
    assert info.n_samples == 3
    assert info.sample_dim == "sample"
    assert info.spatial_dims == ("latitude", "longitude")
    assert info.grid_shape == (4, 8)
    assert info.timestamps == (
        "1980-03-01T00:00:00",
        "1980-03-01T00:00:00",
        "1980-03-01T00:00:00",
    )

    with xr.open_dataset(
        str(out_ic),
        engine="netcdf4",
        decode_times=CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
    ) as merged_ds:
        assert merged_ds["TMP2m"].dims == ("sample", "latitude", "longitude")
        assert merged_ds["TMP2m"].shape == (3, 4, 8)
        assert "lat" not in merged_ds.dims
        assert "lon" not in merged_ds.dims
        assert merged_ds["time"].dims == ("sample",)
        assert merged_ds["time"].shape == (3,)
        assert not np.allclose(
            merged_ds["TMP2m"].values[0], merged_ds["TMP2m"].values[1]
        )
        assert "ak_0" in merged_ds

    # Refuses to overwrite existing collected IC file.
    with pytest.raises(ConfigError, match="Refusing to overwrite"):
        collect_spin(str(spin_dir), str(out_ic))

    # 4. Verify exact fme `InitialConditionConfig` + `get_initial_condition` + `get_forcing_data`
    # duplicate-timestamp invariants on the merged netCDF.
    with xr.open_dataset(
        str(out_ic),
        engine="netcdf4",
        decode_times=CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
    ) as fme_ds:
        sub = fme_ds.isel({fme_ds.time.dims[0]: slice(None, None)})
        initial_times = xr.DataArray(data=sub.time.values[:, None], dims=["sample", "time"])
        assert initial_times.shape == (3, 1)
        forcing_times = xr.CFTimeIndex([sub.time.values[0]])
        resolved_indices = [forcing_times.get_loc(t) for t in initial_times.values[:, 0]]
        assert resolved_indices == [0, 0, 0]


def test_ic_spin_noleap_model_and_multi_file_era5_ic(playground_env: dict[str, Any]) -> None:
    runs_dir: pathlib.Path = playground_env["runs_dir"]
    eam_spec = get_model("ace2-eamv3")
    spin_dir = runs_dir / "eam_spin"

    manifest = prepare_spin(
        model=eam_spec,
        from_times=["1996-01-01T00:00:00", "1996-02-01T00:00:00"],
        to_time="1996-03-01T00:00:00",
        spin_dir=str(spin_dir),
    )
    assert manifest["calendar"] == "noleap"
    assert [m["steps"] for m in manifest["members"]] == [236, 112]

    # Also verify ace2-era5 spin can combine one timestamp from ic_1979.nc and one from ic_2001.nc!
    era5_spec = get_model("ace2-era5")
    multi_spin = prepare_spin(
        model=era5_spec,
        from_times=["1980-06-01T00:00:00", "2001-01-01T00:00:00"],
        to_time="2001-03-01T00:00:00",
        spin_dir=str(runs_dir / "era5_two_ic_files"),
    )
    assert multi_spin["members"][0]["shipped_ic"].endswith("ic_1979.nc")
    assert multi_spin["members"][1]["shipped_ic"].endswith("ic_2001.nc")


def test_ic_spin_rejects_unshipped_from_timestamp(playground_env: dict[str, Any]) -> None:
    runs_dir: pathlib.Path = playground_env["runs_dir"]
    era5_spec = get_model("ace2-era5")
    with pytest.raises(ConfigError, match="not found in shipped IC"):
        prepare_spin(
            model=era5_spec,
            from_times=["1979-05-15T00:00:00"],
            to_time="1980-01-01T00:00:00",
            spin_dir=str(runs_dir / "bad_spin"),
        )


def test_deterministic_duplicate_from_refused_and_single_ic_members_message(
    playground_env: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Refuse duplicate `--from` on deterministic models and report single-IC `--members` limits clearly."""
    runs_dir: pathlib.Path = playground_env["runs_dir"]
    pg_dir: pathlib.Path = playground_env["playground_dir"]
    era5_spec = get_model("ace2-era5")
    stoch_spec = get_model("ace2s-shield-plus")

    # 1. Duplicate normalized `--from` timestamps (`1979-01-01` and `1979-01-01T00:00:00`) refused on deterministic model.
    with pytest.raises(ConfigError, match="duplicate --from timestamp"):
        prepare_spin(
            model=era5_spec,
            from_times=["1979-01-01", "1979-01-01T00:00:00"],
            to_time="1980-01-01T00:00:00",
            spin_dir=str(runs_dir / "dup_det_spin"),
        )

    # 2. Stochastic model ALLOWS duplicate `--from` timestamps because each member gets `seed + 1000 * idx`.
    stoch_spin = prepare_spin(
        model=stoch_spec,
        from_times=["1979-01-01", "1979-01-01T00:00:00"],
        to_time="1979-01-03T00:00:00",
        spin_dir=str(runs_dir / "dup_stoch_spin"),
        seed=10,
    )
    assert [m["seed"] for m in stoch_spin["members"]] == [10, 1010]

    # 3. Single-IC deterministic model (like shipped `ace2-shield` or `ace2-eamv3` with 1 timestamp)
    # states plainly that `ic spin` from shipped ICs cannot create K distinct members!
    single_ic_file = write_fake_ic_netcdf(
        pg_dir / "models" / "single_det_ic.nc",
        ["1994-01-01T00:00:00"],
        calendar="noleap",
        spatial_dims=("lat", "lon"),
    )
    reg_doc = json.loads(pathlib.Path(playground_env["registry_path"]).read_text(encoding="utf-8"))
    single_reg = pg_dir / "single_ic_registry.json"
    for m in reg_doc["models"]:
        if m["name"] == "ace2-eamv3":
            m["ic_files"] = [str(single_ic_file)]
    single_reg.write_text(json.dumps(reg_doc), encoding="utf-8")

    rc = main(
        [
            "run",
            "--registry",
            str(single_reg),
            "-m",
            "ace2-eamv3",
            "--start",
            "1994-01-01T00:00:00",
            "--steps",
            "4",
            "--members",
            "2",
            "--out",
            str(runs_dir / "single_ic_det_fail"),
            "--dry-run",
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "ships only 1 initial-condition state" in err
    assert "cannot produce 2 distinct members" in err


def test_coupled_collect_spin_missing_or_bad_ocean_restart_leaves_no_partial_output(
    playground_env: dict[str, Any],
) -> None:
    """Coupled `collect_spin` refuses missing or mismatched ocean restarts without leaving half a pair on disk."""
    runs_dir: pathlib.Path = playground_env["runs_dir"]
    spin_dir = runs_dir / "coupled_spin_partial"
    (spin_dir / "0").mkdir(parents=True)
    (spin_dir / "1").mkdir(parents=True)
    (spin_dir / "spin-manifest.json").write_text(
        json.dumps(
            {
                "is_coupled": True,
                "target_time": "0311-01-31T00:00:00",
                "calendar": "noleap",
                "members": [
                    {"experiment_dir": str(spin_dir / "0")},
                    {"experiment_dir": str(spin_dir / "1")},
                ],
            }
        ),
        encoding="utf-8",
    )

    for idx, seed in ((0, 11), (1, 12)):
        write_fake_ic_netcdf(
            spin_dir / str(idx) / "atmosphere" / "restart.nc",
            ["0311-01-31T00:00:00"],
            calendar="noleap",
            sample_dim="sample",
            var_names=("TMP2m", "PRESsfc"),
            seed=seed,
        )
    write_fake_ic_netcdf(
        spin_dir / "0" / "ocean" / "restart.nc",
        ["0311-01-31T00:00:00"],
        calendar="noleap",
        sample_dim="sample",
        var_names=("sst", "thetao_0"),
        seed=21,
    )

    out_atmos = runs_dir / "coupled_collected.nc"
    out_ocean = runs_dir / "coupled_collected_ocean.nc"
    missing_ocean = spin_dir / "1" / "ocean" / "restart.nc"

    # 1. Missing member 1 ocean/restart.nc is caught before writing any output file.
    with pytest.raises(ConfigError, match="Missing coupled spin restart") as exc_info:
        collect_spin(str(spin_dir), str(out_atmos))
    assert str(missing_ocean) in str(exc_info.value)
    assert not out_atmos.exists()
    assert not out_ocean.exists()

    # 2. If member 1 ocean/restart.nc exists on disk but fails ocean merge validation after
    #    atmosphere merge wrote out_atmos, collect_spin cleans up out_atmos before re-raising.
    write_fake_ic_netcdf(
        missing_ocean,
        ["0311-02-05T00:00:00"],
        calendar="noleap",
        sample_dim="sample",
        var_names=("sst", "thetao_0"),
        seed=22,
    )
    with pytest.raises(ConfigError, match="0311-01-31T00:00:00"):
        collect_spin(str(spin_dir), str(out_atmos))
    assert not out_atmos.exists()
    assert not out_ocean.exists()

    # 3. After fixing member 1 ocean/restart.nc, retry succeeds cleanly.
    missing_ocean.unlink()
    write_fake_ic_netcdf(
        missing_ocean,
        ["0311-01-31T00:00:00"],
        calendar="noleap",
        sample_dim="sample",
        var_names=("sst", "thetao_0"),
        seed=22,
    )
    info = collect_spin(str(spin_dir), str(out_atmos))
    assert info.n_samples == 2
    assert out_atmos.is_file()
    assert out_ocean.is_file()
