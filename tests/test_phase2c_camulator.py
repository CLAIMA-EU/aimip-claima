"""Offline unit and end-to-end integration tests for Phase 2c `camulator` (`family="camulator"`)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import tarfile
from typing import Any

import netCDF4
import numpy as np
import pytest
import torch
import yaml

from aimip.camulator import (
    CAMULATOR_BRANCH_FORCING_NAME,
    CAMULATOR_BRANCH_FORCING_REL_PATH,
    CAMULATOR_CKPT_REL_PATH,
    CAMULATOR_CKPT_SHA256,
    CAMULATOR_CODE_COMMIT,
    CAMULATOR_CODE_REL_DIR,
    CAMULATOR_CONFIG_REL_PATH,
    CAMULATOR_CYCLIC_FORCING_NAME,
    CAMULATOR_CYCLIC_FORCING_REL_PATH,
    CAMULATOR_HF_REVISION,
    CAMULATOR_IC_SHAPE,
    CAMULATOR_MEAN_REL_PATH,
    CAMULATOR_METADATA_REL_PATH,
    CAMULATOR_MIN_OCEAN_SST_K,
    CAMULATOR_PUBLISHED_STARTS,
    CAMULATOR_STATICS_B_REL_PATH,
    CAMULATOR_STATICS_BE21_REL_PATH,
    CAMULATOR_STATICS_FE21_REL_PATH,
    CAMULATOR_STD_REL_PATH,
    EXIT_UNSTABLE,
    assert_no_symlinks_under,
    camulator_forcing_rel_path,
    camulator_ic_rel_path,
    consolidate_camulator_monthly_outputs,
    extract_and_verify_camulator_archive,
    noleap_expected_6h_counts,
    render_camulator_runtime_config,
    run_camulator,
    verified_camulator_torch_load,
    verify_camulator_code_dir,
)
from aimip.emu_catalog import artifacts, on_demand_artifacts, required_files
from aimip.errors import ConfigError, ValidationError
from aimip.plan_types import RunSpec
from aimip.runner import exec_fme
from aimip.safe_pickle import compute_sha256sum_tree_digest
from aimip.services import Authorization, MemoryApprovalBroker, cite, current_user, plan, prepare, show
from aimip.setup_data import ensure_camulator_selected_assets, load_lock_file
from aimip.status import summarize_run


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sample_inference_config_dict() -> dict[str, Any]:
    """Minimal faithful structure of upstream `willychap/camulator/inference_config.yaml`."""
    return {
        "save_loc": "./assets/",
        "data": {
            "variables": ["U", "V", "T", "Qtot"],
            "surface_variables": ["PS", "TREFHT"],
            "dynamic_forcing_variables": ["SOLIN", "SST", "ICEFRAC", "co2vmr_3d"],
            "diagnostic_variables": [
                "PRECT",
                "TS",
                "CLDHGH",
                "CLDLOW",
                "CLDMED",
                "TAUX",
                "TAUY",
                "U10",
                "QFLX",
                "FSDS_J",
                "FLDS_J",
                "FSUS",
                "FLUS",
                "SHFLX",
                "LHFLX",
                "FSUTOA",
                "FLUT",
            ],
            "forcing_variables": [],
            "static_variables": ["z_norm", "LANDM_COSLAT"],
            "save_loc_static": "./assets/statics_b_credit_runs_f32_02.nc",
            "save_loc_physics": "./assets/statics_b_credit_runs_f32_02.nc",
            "mean_path": "./assets/mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc",
            "std_path": "./assets/std_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc",
            "scaler_type": "std_new",
            "static_first": True,
        },
        "model": {
            "type": "crossformer",
            "post_conf": {
                "activate": True,
                "tracer_fixer": {
                    "activate": True,
                    "denorm": True,
                    "mean_path": "./assets/mean.nc",
                    "std_path": "./assets/std.nc",
                },
                "global_mass_fixer": {
                    "activate": True,
                    "denorm": True,
                    "save_loc_physics": "./assets/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc",
                    "mean_path": "./assets/mean.nc",
                    "std_path": "./assets/std.nc",
                },
                "global_water_fixer": {
                    "activate": True,
                    "denorm": True,
                    "save_loc_physics": "./assets/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc",
                    "mean_path": "./assets/mean.nc",
                    "std_path": "./assets/std.nc",
                },
                "global_energy_fixer_updown": {
                    "activate": True,
                    "denorm": True,
                    "save_loc_physics": "./assets/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc",
                    "mean_path": "./assets/mean.nc",
                    "std_path": "./assets/std.nc",
                },
            },
        },
        "postprocessing": {
            "wind_artifact_filter": {
                "activate": True,
            }
        },
        "loss": {
            "latitude_weights": "./assets/f.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc",
        },
        "predict": {
            "forecasts": {
                "type": "custom",
                "start_year": 1981,
                "start_month": 1,
                "start_day": 1,
                "start_hours": [0],
                "duration": 1,
                "days": 365,
            },
            "start_datetime": "1981-01-01 00:00:00",
            "metadata": "./camulator_metadata.yaml",
            "save_forecast": "./output/",
            "forcing_file": "./assets/b.e21.CREDIT_climate_cyclic_1yr_f32coords.nc",
            "init_cond_fast_climate": "./assets/init_camulator_condition_tensor_1981-01-01T00Z.pth",
            "timesteps_fast_climate": 1460,
        },
    }


def _sample_metadata_dict() -> dict[str, Any]:
    return {
        "U": {"units": "m s**-1", "long_name": "Zonal wind"},
        "V": {"units": "m s**-1", "long_name": "Meridional wind"},
        "T": {"units": "K", "long_name": "Temperature"},
        "Qtot": {"units": "kg kg**-1", "long_name": "Specific total water (vapor + condensate)"},
        "PS": {"units": "Pa", "long_name": "Surface pressure"},
        "TREFHT": {"units": "K", "long_name": "Reference height (2 m) temperature"},
        "TS": {"units": "K", "long_name": "Surface temperature (radiative)"},
        "PRECT": {
            "units": "m per 6hr step",
            "long_name": "Total precipitation, liquid water equivalent accumulated over the 6-hour step",
        },
    }


def test_camulator_provenance_pins_card_and_69_published_starts(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify all 69 published start tensors, selective on-demand catalog, and verbatim card disclosures."""
    import aimip

    monkeypatch.setenv("AIMIP_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    assert len(CAMULATOR_PUBLISHED_STARTS) == 69
    assert CAMULATOR_PUBLISHED_STARTS[0] == "1980-01-01T00:00:00"
    assert CAMULATOR_PUBLISHED_STARTS[1] == "1981-01-01T00:00:00"
    assert CAMULATOR_PUBLISHED_STARTS[2] == "1981-07-01T00:00:00"
    assert CAMULATOR_PUBLISHED_STARTS[-1] == "2014-07-01T00:00:00"
    assert CAMULATOR_MIN_OCEAN_SST_K == 273.0

    # Default fetch pulls ONLY 11 files (~6.3 GB), while 68 files (`branch_1980_2014.nc` + 67 ICs) are on-demand.
    req_rels = required_files("camulator")
    assert len(req_rels) == 11
    assert CAMULATOR_CKPT_REL_PATH in req_rels
    assert CAMULATOR_CYCLIC_FORCING_REL_PATH in req_rels
    assert CAMULATOR_BRANCH_FORCING_REL_PATH not in req_rels

    on_demand_cam = [a for a in on_demand_artifacts() if "camulator" in a["models"]]
    assert len(on_demand_cam) == 68
    default_cam = [a for a in artifacts() if "camulator" in a["models"]]
    assert len(default_cam) == 11

    pkg_dir = pathlib.Path(aimip.__file__).resolve().parent
    lock_files = load_lock_file(str(pkg_dir / "aimip.lock"))["files"]
    assert lock_files[CAMULATOR_CKPT_REL_PATH]["sha256"] == CAMULATOR_CKPT_SHA256
    for ts in CAMULATOR_PUBLISHED_STARTS:
        ic_rel = camulator_ic_rel_path(ts)
        entry = lock_files[ic_rel]
        assert entry["size"] == 30_082_527
        assert len(entry["sha256"]) == 64
        assert entry["source"]["revision"] == CAMULATOR_HF_REVISION

    card = show("camulator", include_examples=True)
    assert card["family"] == "camulator"
    assert (
        card["verified_on"] is not None
        and "A100" in card["verified_on"]
        and "285.8759" in card["verified_on"]
        and "1.4.0" in card["verified_on"]
        and "job " not in card["verified_on"]
    )
    assert card["calendar"] == "noleap"
    assert card["timestep_hours"] == 6
    assert card["stochastic"] is False
    assert [fs["name"] for fs in card["forcing_sets"]] == [
        CAMULATOR_CYCLIC_FORCING_NAME,
        CAMULATOR_BRANCH_FORCING_NAME,
    ]
    assert "Apache-2.0" in card["licence"]
    limits = card["limitations"]
    # Verbatim quoted authors' limits before "AIMIP:":
    assert '"CAMulator is a research tool.' in limits
    assert "not a substitute for an operational forecast or a full Earth-system model" in limits
    assert '"running without it is a train/inference mismatch"' in limits
    assert '"it makes global-mean precipitation ~2% wetter"' in limits
    assert '"Later is not better: the latest checkpoints are measurably colder"' in limits
    assert "AIMIP:" in limits
    assert "CAM6 / ERA5-scaled" in limits
    assert "never amip-hist" in limits

    cites = cite("camulator")
    assert "2504.06007" in cites["bibtex"]
    assert "2504.06007" in cites["entries"][0]
    assert CAMULATOR_CODE_COMMIT[:8] in card["licence"]


def test_camulator_request_validation_noleap_horizon_and_on_demand_plan(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate 69-start enforcement, honest forcing names, 35-year `noleap` bound, and one-step on-demand plan."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    runs_dir = allow_tmp_run_root / "cam_plan_runs"
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
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_path))
    monkeypatch.setenv("AIMIP_DEFAULT_OUT_ROOT", str(runs_dir))
    monkeypatch.delenv("AIMIP_REGISTRY", raising=False)

    # 1. Rejects non-published start timestamp (1980-07-01 does not exist upstream; only 1980-01-01 then 1981..2014).
    with pytest.raises(ConfigError, match="69 published initial states"):
        plan(RunSpec(model="camulator", start="1980-07-01T00:00:00", steps=4))

    # 2. Rejects `--members > 1`, `--ic`, and `--forcing-set amip-hist`.
    with pytest.raises(ConfigError, match="deterministic"):
        plan(RunSpec(model="camulator", start="1981-01-01T00:00:00", steps=4, members=2))

    fake_ic = tmp_path / "custom.nc"
    fake_ic.write_bytes(b"NC")
    with pytest.raises(ConfigError, match="--ic is not supported"):
        plan(RunSpec(model="camulator", start="1981-01-01T00:00:00", steps=4, ic=str(fake_ic)))

    with pytest.raises(ConfigError, match="amip-hist"):
        plan(
            RunSpec(
                model="camulator",
                start="1981-01-01T00:00:00",
                steps=4,
                forcing_set="amip-hist",
            )
        )

    # 3. `noleap` 35-year evaluation run (`51,100` 6-hourly steps from `1980-01-01`) succeeds on `branch-1980-2014`,
    # while `51,101` steps is refused!
    p_35y = plan(
        RunSpec(
            model="camulator",
            start="1980-01-01T00:00:00",
            length="35y",
            forcing_set=CAMULATOR_BRANCH_FORCING_NAME,
            out=str(allow_tmp_run_root / "cam_35y"),
        )
    )
    assert p_35y.steps == 51_100
    assert p_35y.end_time == "2015-01-01T00:00:00"
    assert p_35y.requires_staging is False  # All assets have known lock SHA-256s -> one-step approval!

    with pytest.raises(ConfigError, match="exceeds forcing set"):
        plan(
            RunSpec(
                model="camulator",
                start="1980-01-01T00:00:00",
                steps=51_101,
                forcing_set=CAMULATOR_BRANCH_FORCING_NAME,
            )
        )

    # 4. On-demand 1994-01-01 IC + branch forcing are accurately accounted in `plan.assets` with pinned SHA-256s.
    p_1994 = plan(
        RunSpec(
            model="camulator",
            start="1994-01-01T00:00:00",
            length="1y",
            forcing_set=CAMULATOR_BRANCH_FORCING_NAME,
            out=str(allow_tmp_run_root / "cam_1994"),
        )
    )
    assert p_1994.steps == 1460
    by_role = {a.role: a for a in p_1994.assets if a.role in ("ic", "forcing", "source_tree")}
    assert by_role["ic"].rel_path == "inputs/camulator/initial_conditions/init_camulator_condition_tensor_1994-01-01T00Z.pth"
    assert by_role["ic"].sha256 == "ebba6db09c918cae43c4d8e6b913cc8b1f00aaf9134dbf32714619bcee689186"
    assert by_role["forcing"].rel_path == CAMULATOR_BRANCH_FORCING_REL_PATH
    assert by_role["forcing"].sha256 == "0d3b71131cb44a9092d6babf884077c35a1d062f75258fe333693cc4c973767e"
    assert by_role["source_tree"].rel_path == CAMULATOR_CODE_REL_DIR


def test_camulator_runtime_config_zero_symlinks_and_active_conservation_fixers(
    tmp_path: pathlib.Path,
) -> None:
    """Verify `render_camulator_runtime_config` binds all 3 statics files, enforces fixers, and bans symlinks."""
    base_cfg = tmp_path / "inference_config.yaml"
    base_bytes = yaml.safe_dump(_sample_inference_config_dict(), sort_keys=False).encode("utf-8")
    base_cfg.write_bytes(base_bytes)
    base_sha = _sha256_bytes(base_bytes)

    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    raw_dir = run_dir / "_raw_pred"
    raw_dir.mkdir()
    dst_cfg = run_dir / "camulator_runtime_config.yaml"

    rendered = render_camulator_runtime_config(
        str(base_cfg),
        base_sha,
        checkpoint_dir="/abs/models/camulator",
        mean_nc="/abs/models/camulator/normalization/mean.nc",
        std_nc="/abs/models/camulator/normalization/std.nc",
        statics_b_nc="/abs/models/camulator/normalization/statics_b_credit_runs_f32_02.nc",
        statics_be21_nc="/abs/models/camulator/normalization/b.e21.statics.nc",
        statics_fe21_nc="/abs/models/camulator/normalization/f.e21.statics.nc",
        metadata_yaml="/abs/models/camulator/metadata/camulator_metadata.yaml",
        forcing_nc="/abs/inputs/camulator/forcing_data/cyclic.nc",
        ic_pth="/abs/inputs/camulator/initial_conditions/init_1981-01-01T00Z.pth",
        start_iso="1981-01-01T00:00:00",
        steps=236,
        raw_save_dir=str(raw_dir),
        dst_config_path=str(dst_cfg),
    )

    assert rendered["save_loc"] == "/abs/models/camulator/"
    assert rendered["data"]["save_loc_static"] == "/abs/models/camulator/normalization/statics_b_credit_runs_f32_02.nc"
    assert rendered["data"]["save_loc_physics"] == "/abs/models/camulator/normalization/statics_b_credit_runs_f32_02.nc"
    assert (
        rendered["model"]["post_conf"]["global_energy_fixer_updown"]["save_loc_physics"]
        == "/abs/models/camulator/normalization/b.e21.statics.nc"
    )
    assert rendered["loss"]["latitude_weights"] == "/abs/models/camulator/normalization/f.e21.statics.nc"
    assert rendered["predict"]["timesteps_fast_climate"] == 236
    assert rendered["predict"]["forcing_file"] == "/abs/inputs/camulator/forcing_data/cyclic.nc"
    assert rendered["predict"]["init_cond_fast_climate"] == "/abs/inputs/camulator/initial_conditions/init_1981-01-01T00Z.pth"
    assert "./assets" not in dst_cfg.read_text(encoding="utf-8")

    assert_no_symlinks_under(str(run_dir))

    # Introducing any symlink inside `run_dir` fails closed immediately.
    rogue_link = run_dir / "assets_link"
    rogue_link.symlink_to(tmp_path)
    with pytest.raises(ValidationError, match="symlink"):
        assert_no_symlinks_under(str(run_dir))
    rogue_link.unlink()

    # Disabling `global_energy_fixer_updown` is refused as a train/inference mismatch.
    bad_dict = _sample_inference_config_dict()
    bad_dict["model"]["post_conf"]["global_energy_fixer_updown"]["activate"] = False
    bad_bytes = yaml.safe_dump(bad_dict).encode("utf-8")
    bad_cfg = tmp_path / "bad_inference_config.yaml"
    bad_cfg.write_bytes(bad_bytes)
    with pytest.raises(ValidationError, match="global_energy_fixer_updown"):
        render_camulator_runtime_config(
            str(bad_cfg),
            _sha256_bytes(bad_bytes),
            checkpoint_dir="/abs/models/camulator",
            mean_nc="/abs/mean.nc",
            std_nc="/abs/std.nc",
            statics_b_nc="/abs/sb.nc",
            statics_be21_nc="/abs/sbe.nc",
            statics_fe21_nc="/abs/sfe.nc",
            metadata_yaml="/abs/meta.yaml",
            forcing_nc="/abs/f.nc",
            ic_pth="/abs/ic.pth",
            start_iso="1981-01-01T00:00:00",
            steps=4,
            raw_save_dir=str(raw_dir),
            dst_config_path=str(dst_cfg),
        )


def test_camulator_archive_tree_verification_and_on_demand_fetch(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `extract_and_verify_camulator_archive` tree verification (incl. in-tree relative symlinks) and HF fetch."""
    # 1. Build a synthetic 3-file tar.gz archive + 1 valid in-tree relative symlink (`credit/applications/... -> ../../applications/...`).
    src_stage = tmp_path / "src_stage"
    (src_stage / "applications").mkdir(parents=True)
    (src_stage / "climate").mkdir(parents=True)
    (src_stage / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    (src_stage / "applications" / "calc_global_solar.py").write_text("# solar\n", encoding="utf-8")
    (src_stage / "climate" / "Quick_Climate.py").write_text("print('ok')\n", encoding="utf-8")
    exp_tree_sha, exp_files = compute_sha256sum_tree_digest(str(src_stage))
    assert exp_files == 3

    tar_path = tmp_path / "camulator_src.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for rel in ("LICENSE", "applications/calc_global_solar.py", "climate/Quick_Climate.py"):
            full = src_stage / rel
            ti = tarfile.TarInfo(name=f"miles-credit-{CAMULATOR_CODE_COMMIT}/{rel}")
            raw = full.read_bytes()
            ti.size = len(raw)
            tf.addfile(ti, io.BytesIO(raw))
        sym_ok = tarfile.TarInfo(
            name=f"miles-credit-{CAMULATOR_CODE_COMMIT}/credit/applications/calc_global_solar.py"
        )
        sym_ok.type = tarfile.SYMTYPE
        sym_ok.linkname = "../../applications/calc_global_solar.py"
        tf.addfile(sym_ok)

    tar_bytes = tar_path.read_bytes()
    tar_sha = _sha256_bytes(tar_bytes)
    data_root = tmp_path / "data"
    target_code = data_root / CAMULATOR_CODE_REL_DIR

    res = extract_and_verify_camulator_archive(
        str(tar_path),
        str(target_code),
        expected_archive_sha256=tar_sha,
        expected_archive_size=len(tar_bytes),
        expected_tree_sha256=exp_tree_sha,
        expected_tree_files=exp_files,
    )
    assert res["tree_sha256"] == exp_tree_sha
    assert res["files"] == 3
    assert_no_symlinks_under(str(target_code))
    assert not (target_code / "credit" / "applications" / "calc_global_solar.py").exists()
    verify_camulator_code_dir(
        str(data_root),
        expected_tree_sha256=exp_tree_sha,
        expected_tree_files=exp_files,
    )

    # Escaping symlink (`../../../etc/passwd`) and dangling symlink (`../../applications/missing.py`) fail closed!
    for bad_name, bad_target, match_re in (
        ("escape.tar.gz", "../../../etc/passwd", "Refusing escaping symlink"),
        ("dangling.tar.gz", "../../applications/missing.py", "missing regular-file target"),
    ):
        bad_tar = tmp_path / bad_name
        with tarfile.open(bad_tar, "w:gz") as tf:
            ti = tarfile.TarInfo(name=f"miles-credit-{CAMULATOR_CODE_COMMIT}/LICENSE")
            raw = b"Apache-2.0\n"
            ti.size = len(raw)
            tf.addfile(ti, io.BytesIO(raw))
            sym_bad = tarfile.TarInfo(
                name=f"miles-credit-{CAMULATOR_CODE_COMMIT}/credit/applications/bad.py"
            )
            sym_bad.type = tarfile.SYMTYPE
            sym_bad.linkname = bad_target
            tf.addfile(sym_bad)
        bad_bytes = bad_tar.read_bytes()
        with pytest.raises(ValidationError, match=match_re):
            extract_and_verify_camulator_archive(
                str(bad_tar),
                str(tmp_path / f"out_{bad_name}"),
                expected_archive_sha256=_sha256_bytes(bad_bytes),
                expected_archive_size=len(bad_bytes),
                expected_tree_sha256=exp_tree_sha,
                expected_tree_files=1,
            )

    # Flip 1 byte in extracted `Quick_Climate.py` -> `verify_camulator_code_dir` fails closed!
    (target_code / "climate" / "Quick_Climate.py").write_text("print('tampered')\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="Lock verification failed"):
        verify_camulator_code_dir(
            str(data_root),
            expected_tree_sha256=exp_tree_sha,
            expected_tree_files=exp_files,
        )

    # 2. On-demand fetch & lock verification via `AIMIP_HF_MIRROR`.
    hf_mirror = tmp_path / "hf_mirror"
    ic_1994_rel = camulator_ic_rel_path("1994-01-01T00:00:00")
    branch_rel = camulator_forcing_rel_path(CAMULATOR_BRANCH_FORCING_NAME)
    ic_bytes = b"SYNTHETIC_IC_1994_TENSOR"
    branch_bytes = b"SYNTHETIC_BRANCH_FORCING_1980_2014"

    mirror_ic = hf_mirror / "willychap/camulator" / "initial_conditions/init_camulator_condition_tensor_1994-01-01T00Z.pth"
    mirror_br = hf_mirror / "willychap/camulator" / "forcing_data/b.e21.CREDIT_climate_branch_1980_2014.nc"
    mirror_ic.parent.mkdir(parents=True, exist_ok=True)
    mirror_br.parent.mkdir(parents=True, exist_ok=True)
    mirror_ic.write_bytes(ic_bytes)
    mirror_br.write_bytes(branch_bytes)

    custom_lock = tmp_path / "cam_ondemand.lock"
    custom_lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {
                    ic_1994_rel: {
                        "size": len(ic_bytes),
                        "sha256": _sha256_bytes(ic_bytes),
                        "source": {
                            "kind": "hf",
                            "repo": "willychap/camulator",
                            "path": "initial_conditions/init_camulator_condition_tensor_1994-01-01T00Z.pth",
                            "revision": CAMULATOR_HF_REVISION,
                        },
                    },
                    branch_rel: {
                        "size": len(branch_bytes),
                        "sha256": _sha256_bytes(branch_bytes),
                        "source": {
                            "kind": "hf",
                            "repo": "willychap/camulator",
                            "path": "forcing_data/b.e21.CREDIT_climate_branch_1980_2014.nc",
                            "revision": CAMULATOR_HF_REVISION,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AIMIP_HF_MIRROR", f"file://{hf_mirror}")

    # `auto_fetch=False` refuses before downloading missing on-demand files.
    with pytest.raises(ConfigError, match="not yet staged"):
        ensure_camulator_selected_assets(
            str(data_root),
            start_iso="1994-01-01T00:00:00",
            scenario=CAMULATOR_BRANCH_FORCING_NAME,
            auto_fetch=False,
            lock_path=str(custom_lock),
        )

    fetched = ensure_camulator_selected_assets(
        str(data_root),
        start_iso="1994-01-01T00:00:00",
        scenario=CAMULATOR_BRANCH_FORCING_NAME,
        auto_fetch=True,
        lock_path=str(custom_lock),
    )
    assert fetched["ic"]["sha256"] == _sha256_bytes(ic_bytes)
    assert fetched["forcing"]["sha256"] == _sha256_bytes(branch_bytes)


def test_camulator_spawn_end_to_end_run_summary_tamper_gate_and_unstable_exit_5(
    tmp_path: pathlib.Path,
    allow_tmp_run_root: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end `run_camulator` under `spawn`, `services.plan -> prepare -> exec_fme`, pre-pickle SHA gate, & exit 5."""
    import stat

    # Verify exact 6-hourly noleap step counts across Jan + Feb (`31*4 = 124`, `28*4 = 112`).
    counts_jf = noleap_expected_6h_counts("1981-01-01T00:00:00", 236)
    assert counts_jf == {"1981-01": 124, "1981-02": 112}

    data_dir = tmp_path / "data"
    code_dir = data_dir / CAMULATOR_CODE_REL_DIR
    (code_dir / "climate").mkdir(parents=True)

    # Synthetic `Quick_Climate.py` verifying `spawn` start method, reading rendered config, and writing 2 months.
    quick_climate_src = '''from __future__ import annotations
import argparse, multiprocessing as mp, os
import netCDF4, numpy as np, torch, yaml
from aimip.errors import ValidationError

assert mp.get_start_method() == "spawn", f"Expected spawn, got {mp.get_start_method()}"
p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--model_name", required=True)
p.add_argument("--device", required=True)
p.add_argument("--save_append", required=True)
p.add_argument("--monthly_mean", action="store_true")
args = p.parse_args()
assert args.monthly_mean

with open(args.config, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

ckpt_file = os.path.join(cfg["save_loc"], args.model_name)
skip_mode = os.environ.get("AIMIP_TEST_CAMULATOR_SKIP_LOAD", "")
if skip_mode != "ckpt":
    ckpt_state = torch.load(ckpt_file, map_location="cpu")
    assert ckpt_state["epoch"] == 69
if skip_mode != "ic":
    ic_state = torch.load(cfg["predict"]["init_cond_fast_climate"], map_location="cpu")
    assert tuple(ic_state.shape) == (1, 136, 1, 192, 288)
try:
    torch.load(cfg["predict"]["forcing_file"], map_location="cpu")
except ValidationError:
    pass
else:
    raise AssertionError("Expected child verified torch.load to refuse unlisted path")
assert os.path.isfile(cfg["predict"]["forcing_file"])
assert os.path.isfile(cfg["predict"]["metadata"])

out_sub = os.path.join(cfg["predict"]["save_forecast"], args.save_append, "1981-01-01T00Z")
os.makedirs(out_sub, exist_ok=True)
lat = np.linspace(-90.0, 90.0, 192, dtype=np.float32)
lon = np.linspace(0.0, 358.75, 288, dtype=np.float32)
lev = np.linspace(3.6, 992.6, 32, dtype=np.float32)

trefht_base = float(os.environ.get("AIMIP_TEST_CAMULATOR_TREFHT", "287.5"))
for idx, ym in enumerate(("198101", "198102")):
    fn = os.path.join(out_sub, f"pred_1981-01-01T00Z_{ym}.nc")
    with netCDF4.Dataset(fn, "w", format="NETCDF4") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("level", 32)
        ds.createDimension("latitude", 192)
        ds.createDimension("longitude", 288)
        ds.createVariable("latitude", "f4", ("latitude",))[:] = lat
        ds.createVariable("longitude", "f4", ("longitude",))[:] = lon
        ds.createVariable("level", "f4", ("level",))[:] = lev
        ds.createVariable("TREFHT", "f4", ("time", "latitude", "longitude"))[:] = trefht_base + idx
        ds.createVariable("TS", "f4", ("time", "latitude", "longitude"))[:] = trefht_base + 1.0 + idx
        ds.createVariable("PS", "f4", ("time", "latitude", "longitude"))[:] = 101325.0
        ds.createVariable("PRECT", "f4", ("time", "latitude", "longitude"))[:] = 5.0e-4
        for ua in ("U", "V", "T", "Qtot"):
            ds.createVariable(ua, "f4", ("time", "level", "latitude", "longitude"))[:] = 1.0
'''
    (code_dir / "climate" / "Quick_Climate.py").write_text(quick_climate_src, encoding="utf-8")
    tree_sha_ok, tree_files_ok = compute_sha256sum_tree_digest(str(code_dir))

    # Stage synthetic 11 required files + custom `aimip.lock` and registry.
    ckpt_buf = io.BytesIO()
    torch.save({"epoch": 69, "model_state_dict": {"w": torch.ones(2, dtype=torch.float32)}}, ckpt_buf)
    ckpt_bytes = ckpt_buf.getvalue()

    ic_buf = io.BytesIO()
    torch.save(torch.zeros(CAMULATOR_IC_SHAPE, dtype=torch.float32), ic_buf)
    ic_bytes = ic_buf.getvalue()

    cfg_bytes = yaml.safe_dump(_sample_inference_config_dict(), sort_keys=False).encode("utf-8")
    meta_bytes = yaml.safe_dump(_sample_metadata_dict(), sort_keys=False).encode("utf-8")

    staged_blobs: dict[str, bytes] = {
        CAMULATOR_CKPT_REL_PATH: ckpt_bytes,
        CAMULATOR_CONFIG_REL_PATH: cfg_bytes,
        CAMULATOR_METADATA_REL_PATH: meta_bytes,
        CAMULATOR_MEAN_REL_PATH: b"FAKE_MEAN_NC",
        CAMULATOR_STD_REL_PATH: b"FAKE_STD_NC",
        CAMULATOR_STATICS_B_REL_PATH: b"FAKE_STATICS_B_NC",
        CAMULATOR_STATICS_BE21_REL_PATH: b"FAKE_STATICS_BE21_NC",
        CAMULATOR_STATICS_FE21_REL_PATH: b"FAKE_STATICS_FE21_NC",
        CAMULATOR_CYCLIC_FORCING_REL_PATH: b"FAKE_CYCLIC_FORCING_NC",
        camulator_ic_rel_path("1980-01-01T00:00:00"): ic_bytes,
        camulator_ic_rel_path("1981-01-01T00:00:00"): ic_bytes,
    }
    lock_entries: dict[str, Any] = {}
    for rel, raw in staged_blobs.items():
        p = data_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        lock_entries[rel] = {
            "size": len(raw),
            "sha256": _sha256_bytes(raw),
            "source": {
                "kind": "hf",
                "repo": "willychap/camulator",
                "path": rel.split("/", 2)[-1],
                "revision": CAMULATOR_HF_REVISION,
            },
        }

    # 1. Direct `run_camulator` execution under `spawn` -> consolidates CF-1.8 NetCDFs -> `summarize_run`!
    direct_out = allow_tmp_run_root / "camulator_direct_run"
    direct_out.mkdir()
    job_dict: dict[str, Any] = {
        "family": "camulator",
        "model": "camulator",
        "device": "cpu",
        "code_dir": str(code_dir),
        "verify_code_tree": False,
        "checkpoint": str(data_dir / CAMULATOR_CKPT_REL_PATH),
        "checkpoint_sha256": _sha256_bytes(ckpt_bytes),
        "base_config_file": str(data_dir / CAMULATOR_CONFIG_REL_PATH),
        "base_config_sha256": _sha256_bytes(cfg_bytes),
        "metadata_file": str(data_dir / CAMULATOR_METADATA_REL_PATH),
        "metadata_sha256": _sha256_bytes(meta_bytes),
        "mean_file": str(data_dir / CAMULATOR_MEAN_REL_PATH),
        "mean_sha256": _sha256_bytes(b"FAKE_MEAN_NC"),
        "std_file": str(data_dir / CAMULATOR_STD_REL_PATH),
        "std_sha256": _sha256_bytes(b"FAKE_STD_NC"),
        "statics_b_file": str(data_dir / CAMULATOR_STATICS_B_REL_PATH),
        "statics_b_sha256": _sha256_bytes(b"FAKE_STATICS_B_NC"),
        "statics_be21_file": str(data_dir / CAMULATOR_STATICS_BE21_REL_PATH),
        "statics_be21_sha256": _sha256_bytes(b"FAKE_STATICS_BE21_NC"),
        "statics_fe21_file": str(data_dir / CAMULATOR_STATICS_FE21_REL_PATH),
        "statics_fe21_sha256": _sha256_bytes(b"FAKE_STATICS_FE21_NC"),
        "forcing_file": str(data_dir / CAMULATOR_CYCLIC_FORCING_REL_PATH),
        "forcing_sha256": _sha256_bytes(b"FAKE_CYCLIC_FORCING_NC"),
        "ic_file": str(data_dir / camulator_ic_rel_path("1981-01-01T00:00:00")),
        "ic_sha256": _sha256_bytes(ic_bytes),
        "start": "1981-01-01T00:00:00",
        "steps": 236,
        "scenario": CAMULATOR_CYCLIC_FORCING_NAME,
    }
    direct_res = run_camulator(str(direct_out), job_dict)
    assert direct_res["months_written"] == {"atmosphere": 2}
    assert direct_res["first_month"] == "1981-01"
    assert direct_res["last_month"] == "1981-02"
    assert_no_symlinks_under(str(direct_out))

    with netCDF4.Dataset(direct_out / "monthly_mean_predictions.nc", "r") as mds:
        assert mds.Conventions == "CF-1.8"
        assert mds.variables["valid_time"].calendar == "noleap"
        assert mds.variables["counts"][:].tolist() == [[124, 112]]
        assert mds.variables["TREFHT"].units == "K"
        assert mds.variables["PRECT"].units == "m per 6hr step"

    with netCDF4.Dataset(direct_out / "monthly_levels.nc", "r") as lds:
        assert lds.variables["U"].shape == (2, 32, 192, 288)
        assert lds.variables["time"].calendar == "noleap"

    summary = summarize_run(str(direct_out))
    trefht_rows = summary["variables"]["TREFHT"]["series"]
    assert [r["month"] for r in trefht_rows] == ["1981-01", "1981-02"]
    assert pytest.approx(trefht_rows[0]["members"][0], abs=1e-4) == 287.5
    assert pytest.approx(trefht_rows[1]["members"][0], abs=1e-4) == 288.5

    # Pre-pickle SHA-256 gate inside `run_camulator` AND child `verified_camulator_torch_load`!
    ckpt_disk_p = data_dir / CAMULATOR_CKPT_REL_PATH
    unlisted_pt = tmp_path / "unlisted.pt"
    torch.save({"x": torch.ones(1)}, unlisted_pt)
    with verified_camulator_torch_load({str(ckpt_disk_p): _sha256_bytes(ckpt_bytes)}):
        loaded_ck = torch.load(str(ckpt_disk_p), map_location="cpu")
        assert loaded_ck["epoch"] == 69
        with pytest.raises(ValidationError, match="unverified CAMulator torch.load path"):
            torch.load(str(unlisted_pt), map_location="cpu")
        with pytest.raises(ValidationError, match="weights_only=False"):
            torch.load(str(ckpt_disk_p), map_location="cpu", weights_only=False)
        ckpt_disk_p.write_bytes(ckpt_bytes[:-1] + bytes([ckpt_bytes[-1] ^ 0xFF]))
        try:
            with pytest.raises(ValidationError, match="SHA-256 mismatch before loading"):
                torch.load(str(ckpt_disk_p), map_location="cpu")
        finally:
            ckpt_disk_p.write_bytes(ckpt_bytes)

    with pytest.raises(ValidationError, match="SHA-256 mismatch"):
        run_camulator(str(allow_tmp_run_root / "bad_ckpt_sha"), {**job_dict, "checkpoint_sha256": "0" * 64})
    with pytest.raises(ValidationError, match="SHA-256 mismatch"):
        run_camulator(str(allow_tmp_run_root / "bad_ic_sha"), {**job_dict, "ic_sha256": "0" * 64})

    # If the child bypasses `torch.load` for either the checkpoint or the initial-state `.pth`,
    # `run_camulator` fails closed and names the missing file!
    monkeypatch.setenv("AIMIP_TEST_CAMULATOR_SKIP_LOAD", "ckpt")
    try:
        skip_ckpt_out = allow_tmp_run_root / "camulator_bypass_ckpt"
        skip_ckpt_out.mkdir()
        with pytest.raises(
            ValidationError,
            match=r"without loading required artifact.*checkpoint\.pt00069\.pt",
        ):
            run_camulator(str(skip_ckpt_out), job_dict)

        monkeypatch.setenv("AIMIP_TEST_CAMULATOR_SKIP_LOAD", "ic")
        skip_ic_out = allow_tmp_run_root / "camulator_bypass_ic"
        skip_ic_out.mkdir()
        with pytest.raises(
            ValidationError,
            match=r"without loading required artifact.*init_camulator_condition_tensor_1981-01-01T00Z\.pth",
        ):
            run_camulator(str(skip_ic_out), job_dict)
    finally:
        monkeypatch.delenv("AIMIP_TEST_CAMULATOR_SKIP_LOAD", raising=False)

    # Repeated `verify_code_tree=True` runs with `Model_State.py` leave zero `.pyc` pollution; degraded `Model_State` fails closed!
    ms_path = code_dir / "climate" / "Model_State.py"
    ms_path.write_text(
        "CREDIT_AVAILABLE = True\nPOSTBLOCK_AVAILABLE = True\nWINDPP_AVAILABLE = True\n",
        encoding="utf-8",
    )
    ms_tree_sha, ms_tree_files = compute_sha256sum_tree_digest(str(code_dir))
    for rep in (1, 2):
        rep_out = allow_tmp_run_root / f"camulator_verified_tree_{rep}"
        rep_out.mkdir()
        run_camulator(
            str(rep_out),
            {
                **job_dict,
                "verify_code_tree": True,
                "code_tree_sha256": ms_tree_sha,
                "code_tree_files": ms_tree_files,
            },
        )
        assert compute_sha256sum_tree_digest(str(code_dir)) == (ms_tree_sha, ms_tree_files)

    ms_path.write_text(
        "CREDIT_AVAILABLE = True\nPOSTBLOCK_AVAILABLE = False\nWINDPP_AVAILABLE = True\n",
        encoding="utf-8",
    )
    bad_ms_sha, bad_ms_files = compute_sha256sum_tree_digest(str(code_dir))
    degraded_out = allow_tmp_run_root / "camulator_degraded_ms"
    degraded_out.mkdir()
    with pytest.raises(SystemExit):
        run_camulator(
            str(degraded_out),
            {
                **job_dict,
                "verify_code_tree": True,
                "code_tree_sha256": bad_ms_sha,
                "code_tree_files": bad_ms_files,
            },
        )
    ms_path.unlink()

    # 2. Service-layer `plan -> prepare -> exec_fme` dispatches via `AIMIP_CAMULATOR_PYTHON` and enforces manifest SHA gates!
    lock_path = tmp_path / "camulator_test.lock"
    lock_path.write_text(json.dumps({"schema_version": 1, "files": lock_entries}), encoding="utf-8")

    reg_path = tmp_path / "camulator_registry.json"
    reg_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "name": "camulator",
                        "family": "camulator",
                        "fme_module": "aimip.emu_job",
                        "calendar": "noleap",
                        "checkpoint": "${DATA}/" + CAMULATOR_CKPT_REL_PATH,
                        "checkpoint_sha256": _sha256_bytes(ckpt_bytes),
                        "ic_files": [],
                        "forcing_sets": [
                            {
                                "name": CAMULATOR_CYCLIC_FORCING_NAME,
                                "dir": "${DATA}/inputs/camulator/forcing_data",
                                "window": ["1980-01-01T00:00:00", "2100-12-31T18:00:00"],
                            }
                        ],
                        "required_files": list(staged_blobs.keys()),
                        "timestep_hours": 6,
                        "stochastic": False,
                        "default_n_ensemble_per_ic": 1,
                        "notes": "Synthetic CAMulator test entry.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    runs_dir = allow_tmp_run_root / "cam_svc_runs"
    runs_dir.mkdir()
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
    pg_dir = tmp_path / "pg"
    (pg_dir / "bin").mkdir(parents=True)
    (pg_dir / "bin" / "aimip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (pg_dir / "bin" / "aimip").chmod(0o755)

    fake_cam_py = tmp_path / "fake_camulator_py"
    fake_cam_py.write_text(
        """#!/bin/bash
set -eu
[ "$1 $2 $3 $4" = "-I -m aimip.emu_job run" ] || exit 9
out="$5"
printf '{"months_written": {"atmosphere": 2}}\\n' > "$out/emu-result.json"
""",
        encoding="utf-8",
    )
    fake_cam_py.chmod(fake_cam_py.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setenv("AIMIP_PLAYGROUND", str(pg_dir))
    monkeypatch.setenv("AIMIP_DATA", str(data_dir))
    monkeypatch.setenv("AIMIP_LOCK", str(lock_path))
    monkeypatch.setenv("AIMIP_REGISTRY", str(reg_path))
    monkeypatch.setenv("AIMIP_SITE_CONFIG", str(site_file))
    monkeypatch.setenv("AIMIP_CAMULATOR_CODE", str(code_dir))
    monkeypatch.setenv("AIMIP_CAMULATOR_PYTHON", str(fake_cam_py))
    monkeypatch.setenv("AIMIP_STATE", str(tmp_path / "state"))

    spec_svc = RunSpec(
        model="camulator",
        start="1981-01-01T00:00:00",
        steps=236,
        name="cam-svc-1",
    )
    resolved = plan(spec_svc)
    now = 1_750_000_000.0
    broker = MemoryApprovalBroker(now_fn=lambda: now)
    auth = Authorization(
        action="submit",
        digest=resolved.digest,
        user=current_user(),
        issued_at=now,
        expires_at=now + 600,
        nonce="cam-prep-1",
        allow_download=False,
    )
    broker.store_minted(auth)
    prep = prepare(spec_svc, authorization=auth, require_authorization=False, broker=broker)
    assert prep.manifest["family"] == "camulator"
    assert prep.manifest["runtime_python"] == str(fake_cam_py)

    # Corrupting `init_camulator_condition_tensor_1981-01-01T00Z.pth` AFTER prepare fails closed in `exec_fme`!
    ic_disk = data_dir / camulator_ic_rel_path("1981-01-01T00:00:00")
    orig_ic_bytes = ic_disk.read_bytes()
    ic_disk.write_bytes(orig_ic_bytes[:-1] + bytes([orig_ic_bytes[-1] ^ 0xFF]))
    try:
        with pytest.raises(ValidationError, match="SHA-256 mismatch at job time"):
            exec_fme(prep.run_dir)
    finally:
        ic_disk.write_bytes(orig_ic_bytes)

    # Editing 1 file inside approved `AIMIP_CAMULATOR_CODE` AFTER prepare fails closed in `exec_fme`!
    qc_script = code_dir / "climate" / "Quick_Climate.py"
    orig_qc_bytes = qc_script.read_bytes()
    qc_script.write_bytes(orig_qc_bytes + b"\n# post-prepare edit\n")
    try:
        with pytest.raises(ValidationError, match="Approved source override tree changed at job time"):
            exec_fme(prep.run_dir)
    finally:
        qc_script.write_bytes(orig_qc_bytes)

    # Pointing `AIMIP_CAMULATOR_CODE` to a different directory at `exec_fme` time also fails closed!
    other_cam_code = tmp_path / "other_cam_code"
    other_cam_code.mkdir()
    monkeypatch.setenv("AIMIP_CAMULATOR_CODE", str(other_cam_code))
    try:
        with pytest.raises(ValidationError, match="differs from approved plan directory"):
            exec_fme(prep.run_dir)
    finally:
        monkeypatch.setenv("AIMIP_CAMULATOR_CODE", str(code_dir))

    fin_manifest = exec_fme(prep.run_dir)
    assert fin_manifest["rc"] == 0
    assert fin_manifest["result"]["months_written"] == {"atmosphere": 2}
    assert tree_files_ok == 1 and len(tree_sha_ok) == 64

    # 3. Out-of-band `TREFHT` (`500 K`) in `consolidate_camulator_monthly_outputs` exits with `EXIT_UNSTABLE == 5`!
    raw_bad = tmp_path / "raw_unstable"
    raw_bad.mkdir()
    bad_nc = raw_bad / "pred_1981-01-01T00Z_198101.nc"
    with netCDF4.Dataset(bad_nc, "w", format="NETCDF4") as bds:
        bds.createDimension("time", 1)
        bds.createDimension("latitude", 4)
        bds.createDimension("longitude", 8)
        bds.createVariable("latitude", "f4", ("latitude",))[:] = np.linspace(-60, 60, 4, dtype=np.float32)
        bds.createVariable("longitude", "f4", ("longitude",))[:] = np.linspace(0, 315, 8, dtype=np.float32)
        bds.createVariable("TREFHT", "f4", ("time", "latitude", "longitude"))[:] = 500.0
    with pytest.raises(SystemExit) as exc_u:
        consolidate_camulator_monthly_outputs(
            str(raw_bad),
            metadata_yaml_path=str(data_dir / CAMULATOR_METADATA_REL_PATH),
            metadata_sha256=_sha256_bytes(meta_bytes),
            start_iso="1981-01-01T00:00:00",
            steps=124,
            scenario=CAMULATOR_CYCLIC_FORCING_NAME,
            out_dir=str(tmp_path / "out_bad"),
        )
    assert exc_u.value.code == EXIT_UNSTABLE == 5
