"""Pinned files and registry entries of the DLESyM / ArchesClimate-SSP / ArchesWeather models (stdlib only).

`emu_sources.json` lists every file at a pinned Hugging Face revision with its size and sha256 (LFS oid, or the hash
of the downloaded git file).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

EMU_MODELS: tuple[str, ...] = (
    "dlesym-v1.0",
    "dlesym-v1.1",
    "dlesym-v0-isccp",
    "dlesym-aimip",
    "archesclimate-ssp",
    "archesweather",
    "archesweathergen",
    "samudra-om4",
    "md-1.5",
    "md-1.5-aimip",
    "camulator",
    "cbottle-3d",
    "spherical-dyffusion",
)
DLESYM_IC_WINDOW: tuple[str, str] = ("1940-01-04T00:00:00", "2023-12-31T18:00:00")   # inputs reach back 48 h; v1.1 ttr03 needs
# trailing ERA5 accumulations that ERA5 lacks at its very start, leaving one day of margin.
# DLESyM AIMIP version (Cresswell-Clay et al., DLESyM_aimip-2026): the dlesym-v0-isccp checkpoints with the SST
# prescribed from the AIMIP Phase 1 forcing (Zenodo 17065758, monthly, stamped at the first of each month; the authors'
# "ground-truth" ocean interpolates linearly in time). The inputs reach back 48 h, so the first start is 1978-10-03T00
# (the authors' members start 1978-10-03 .. 07); every 96-h step must end by 2025-01-01T00, the last stamp.
AIMIP_SST_REL = "raw/zenodo-17065758/ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc"
AIMIP_SST_SHA256 = "a52caf3722bc85a58075118fa06f0c6cce5ace279458edda0a2c9d68ec179fb7"
DLESYM_AIMIP_WINDOW: tuple[str, str] = ("1978-10-03T00:00:00", "2024-12-28T00:00:00")
DLESYM_AIMIP_FORCING_END = "2025-01-01T00:00:00"
ARCHES_WINDOW: tuple[str, str] = ("2015-01-01T00:00:00", "2100-11-01T00:00:00")   # outputs March 2015 .. November 2100
SAMUDRA_WINDOW: tuple[str, str] = ("1975-01-08T12:00:00", "2022-12-29T12:00:00")   # OM4 5-day Julian states (time[1] .. time[-1])
MD15_WINDOW: tuple[str, str] = ("1978-10-01T00:00:00", "2025-01-01T00:00:00")   # 556 month-start-centred AIMIP forcing points
# md-1.5-aimip (MD-1.5 as run for AIMIP Phase 1): the initial state of October 1978 and the 1985-2014 statistics
# rebuilt from ERA5 (tools/md15), pinned from a Zenodo record (`zenodo:md-1.5-aimip-inputs` in emu_sources.json).
MD15_AIMIP_INPUTS_DIR = "inputs/md-1.5-aimip"
CAMULATOR_CYCLIC_WINDOW: tuple[str, str] = ("1980-01-01T00:00:00", "2100-12-31T18:00:00")
CAMULATOR_BRANCH_WINDOW: tuple[str, str] = ("1980-01-01T00:00:00", "2014-12-31T18:00:00")
CBOTTLE_WINDOW: tuple[str, str] = ("1940-01-01T00:00:00", "2022-12-16T00:00:00")
SDYFF_WINDOW: tuple[str, str] = ("2021-01-01T00:00:00", "2120-12-31T18:00:00")

# ArchesWeather / ArchesWeatherGen, AIMIP versions (INRIA; Singh et al. 2026, arXiv:2605.29976): Zenodo record 22795233
# (CC-BY-4.0). Daily steps on the 1-degree grid from the ERA5 daily-mean state of 1978-09-30 (previous state 1978-09-29,
# both in era5_1978_00h_sliced.nc), forced by monthly ERA5 SST and sea-ice cover through December 2024.
AW_MODELS: tuple[str, ...] = ("archesweather", "archesweathergen")
AW_RECORD = "22795233"
AW_START = "1978-09-30T00:00:00"
AW_LAST_DAY = "2024-12-31T00:00:00"
AW_MAX_STEPS = 16894                    # 1978-10-01 .. 2024-12-31
AW_DIR = "models/archesweather"
AW_INPUTS = "inputs/archesweather"
AW_ZIP_REL = f"raw/zenodo-{AW_RECORD}/model_checkpoints.zip"
AW_ZIP_PREFIX = "Downloads/model_checkpoints/"
AW_SEEDS: tuple[str, ...] = tuple(f"aimip-archesweather-m-seed{i}" for i in range(4))
AW_GEN = "aimip-archesweathergen"
AW_CKPT = "checkpoint.ckpt"           # local name; aimip paths may not contain "=" (zip member below)
AW_CKPT_MEMBER = "checkpoints/checkpoint_global_step=300000.ckpt"
AW_IC = f"{AW_INPUTS}/era5_1978_00h_sliced.nc"
AW_FORCING = f"{AW_INPUTS}/ERA5-0.25deg-monthly-mean-forcing-1978-2024_regridded_conservative_2025_11_03.nc"
AW_FORCING_STATS = f"{AW_DIR}/stats/ERA5-0.25deg-monthly-mean-forcing-1978-2013_regridded_conservative_norm_stats.nc"
AW_NORM_STATS = f"{AW_DIR}/stats/daily_averaged_aimip_norm_stats.nc"
AW_RESIDUAL_STATS = f"{AW_DIR}/stats/residual_std_aimip_213852624.nc"
AW_RESTART = "restart.nc"                 # written by every run, read by `aimip run --ic <run-dir>` (continuation)
AW_RESTART_FORMAT = "aimip-geoarches-restart-1"
_AW_BOTH = AW_MODELS
_AW_GEN_ONLY = ("archesweathergen",)
# rel_path -> (Zenodo file, size, md5, sha256, models that need it)
AW_ZENODO_FILES: dict[str, tuple[str, int, str, str, tuple[str, ...]]] = {
    AW_ZIP_REL: ("model_checkpoints.zip", 1745144084, "563f37cb198c134874de30e2241c9340",
                 "db57d2bbbb1784a568f941dd7a3e437672cedd59cbaba10a313d21025a4d229d", _AW_BOTH),
    AW_IC: ("era5_1978_00h_sliced.nc", 65782694, "de4402df63c1c00ef49b5491ddc7da4e",
            "ddad475ae0232ea36bc2859cb1c6075fe41e34d9ca4a703b5572d63d1a3d4e79", _AW_BOTH),
    AW_FORCING: ("ERA5-0.25deg-monthly-mean-forcing-1978-2024_regridded_conservative_2025_11_03.nc", 290114662,
                 "889a71d433d0eb84f65ed1f8eddc3e3c", "9be1dd017ce86c2a4889d93983f8d14c40e16a5c6ab7fc5a08d549820707ff29",
                 _AW_BOTH),
    AW_FORCING_STATS: ("ERA5-0.25deg-monthly-mean-forcing-1978-2013_regridded_conservative_norm_stats.nc", 8276,
                       "453a0916f51872549a544d5a67d20522",
                       "8f8da1ec9f41e120fa4ee3215086f28daa46f28940890673ec5e9c66a65636dc", _AW_BOTH),
    AW_NORM_STATS: ("daily_averaged_aimip_norm_stats.nc", 19500, "6d7eaaaddb340445bbff40cdd11ae9c0",
                    "cf46fd86d8b4db347de56c9ab42a90dca787d8e89ed89733a7744e6437a74acc", _AW_BOTH),
    AW_RESIDUAL_STATS: ("residual_std_aimip_213852624.nc", 18784, "280aadf94cd1c3f3be1aa1e0bb46594f",
                        "8897cd7385c435f00e0d1b7a82abc5ea96722246a185d7b7707cc30bbd6fc280", _AW_GEN_ONLY),
}
# rel_path -> (member of model_checkpoints.zip, size, sha256, models): extracted unchanged by `aimip fetch`
AW_ZIP_MEMBERS: dict[str, tuple[str, int, str, tuple[str, ...]]] = {}
for _name, _cfg_sha, _ckpt_size, _ckpt_sha, _models in (
    (AW_SEEDS[0], "21ab309dd844f03b0d4089b24ad9fbfd5618390b39c2dd35ae442c8af9402862", 339700097,
     "e9e51af0a0b5ec95e3c5b5135b7e3eadda1fb9732c10ea4657c9c8d7d780e73f", _AW_BOTH),
    (AW_SEEDS[1], "f6535f5caed4f18a3ab329ecd58aee9928c411fb8ce0716f77073595e25bc9fe", 339700097,
     "9a29c0422280ac63a54f63b52b2f3160925da372cf0c60c056fcac5a710d575d", _AW_BOTH),
    (AW_SEEDS[2], "2c71f5debf5c7f6db4df64312796cb1b3d835b0f2023ee7b7ea4a9d3e0962c5e", 339700097,
     "063f0e3bf995fb19ea4a2693fc838c56b9a1c06121f034b8ca1fcb782ad5ce03", _AW_BOTH),
    (AW_SEEDS[3], "ef4ab0f5fb5772cd65508d01de938385e41cc3091e769f695691d17d56b0242e", 339700097,
     "53379ca5a87c31b728b113dcd29e92fe79403ca150a61a58203fea4796ad1d43", _AW_BOTH),
    (AW_GEN, "b709dfafd82b08f64fcae92e6eae15e9598d463c940b6deff33e5abdbda51ca3", 521690517,
     "7d790f2072e6ac7cb9fa65004406ca570229acdccb68c2a2ada9c4f607edcbb3", _AW_GEN_ONLY),
):
    AW_ZIP_MEMBERS[f"{AW_DIR}/{_name}/config.yaml"] = (f"{AW_ZIP_PREFIX}{_name}/config.yaml",
                                                       6989 if _name == AW_GEN else 7126, _cfg_sha, _models)
    AW_ZIP_MEMBERS[f"{AW_DIR}/{_name}/{AW_CKPT}"] = (f"{AW_ZIP_PREFIX}{_name}/{AW_CKPT_MEMBER}", _ckpt_size, _ckpt_sha,
                                                     _models)

_SOURCES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emu_sources.json")


def aw_package_digest(shas: Any) -> str:
    """SHA-256 over the sorted SHA-256 values of a run's pinned ArchesWeather files (independent of paths and order).

    A restart records it; a continuation must pin the same checkpoints, statistics, initial state and forcing."""
    h = hashlib.sha256()
    for value in sorted(str(x).lower() for x in shas):
        h.update(value.encode("ascii") + b"\n")
    return h.hexdigest()


def _load_sources() -> dict[str, Any]:
    with open(_SOURCES_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def _place(repo: str, path: str) -> tuple[str, tuple[str, ...]] | None:
    """Local relative path under $AIMIP_DATA and the models that need the file."""
    if repo == "nvidia/dlesym-v1-era5":
        if path.startswith("checkpoints/v1.0/"):
            models: tuple[str, ...] = ("dlesym-v1.0",)
        elif path.startswith("checkpoints/v1.1/"):
            models = ("dlesym-v1.1",)
        else:
            models = ("dlesym-v1.0", "dlesym-v1.1")
        return f"models/dlesym-v1/{path}", models
    if repo == "nvidia/dlesym-v0-isccp-era5":
        return f"models/dlesym-v0-isccp/{path}", ("dlesym-v0-isccp", "dlesym-aimip")
    if repo == "gclyne/ArchesClimate":
        return f"models/archesclimate-ssp/{path}", ("archesclimate-ssp",)
    if repo == "gclyne/ArchesClimate-data":
        if path.startswith("ipsl_scenarios/"):
            return f"inputs/archesclimate-ssp/memmap_filled_in/{path[len('ipsl_scenarios/'):]}", ("archesclimate-ssp",)
        return f"models/archesclimate-ssp/{path}", ("archesclimate-ssp",)
    if repo in ("M2LInES/Samudra", "M2LInES/Samudra-OM4"):
        return f"models/samudra-om4/{path}", ("samudra-om4",)
    if repo == "willychap/camulator":
        if path.startswith(("forcing_data/", "initial_conditions/")):
            return f"inputs/camulator/{path}", ("camulator",)
        return f"models/camulator/{path}", ("camulator",)
    if repo == "nvidia/cbottle":
        return f"models/cbottle-3d/{path}", ("cbottle-3d",)
    if repo == "salv47/spherical-dyffusion":
        return f"models/spherical-dyffusion/{path}", ("spherical-dyffusion",)
    if repo == "zenodo:10791087":
        return f"inputs/spherical-dyffusion/fv3gfs-2021/{path}", ("spherical-dyffusion",)
    if repo == "github:kjhall01/monthly-diffusion":
        if path.startswith("models/MD-1p5.pth/"):
            return f"models/md-1.5/{path[len('models/'):]}", ("md-1.5-aimip",)
        if path.startswith("models/"):
            return f"models/md-1.5/{path[len('models/'):]}", ("md-1.5",)
        if path.startswith("scripts/"):
            return f"models/md-1.5/{path[len('scripts/'):]}", ("md-1.5",)
        return f"models/md-1.5/{path}", ("md-1.5",)
    if repo == "zenodo:md-1.5-aimip-inputs":
        return f"{MD15_AIMIP_INPUTS_DIR}/{path}", ("md-1.5-aimip",)
    return None


def all_artifacts() -> list[dict[str, Any]]:
    """[{rel_path, source_kind, repo/record, path, revision/url/md5, size, sha256, models, on_demand}] in a stable order."""
    out: list[dict[str, Any]] = []
    for repo, info in _load_sources().items():
        src_kind = str(info.get("source_kind", "hf"))
        for path in sorted(info["files"]):
            placed = _place(repo, path)
            if placed is None:
                continue
            rel, models = placed
            f = info["files"][path]
            item: dict[str, Any] = {
                "rel_path": rel,
                "source_kind": src_kind,
                "repo": repo,
                "repo_type": info.get("repo_type", "model"),
                "path": path,
                "revision": info.get("revision", ""),
                "size": int(f["size"]),
                "sha256": str(f["sha256"]),
                "models": models,
                "on_demand": bool(f.get("on_demand", False)),
            }
            if src_kind == "zenodo":
                item["record"] = str(info["record"])
                item["md5"] = str(f["md5"])
                item["url"] = str(f["url"])
            elif src_kind == "github_archive":
                item["archive_url"] = str(info["archive_url"])
                item["archive_sha256"] = str(info["archive_sha256"])
            out.append(item)
    return out


def artifacts() -> list[dict[str, Any]]:
    """Default-fetched artifacts (`on_demand == False`) in a stable order."""
    return [a for a in all_artifacts() if not a["on_demand"]]


def on_demand_artifacts() -> list[dict[str, Any]]:
    """Pinned on-demand artifacts (e.g. non-default CAMulator ICs and 10.46 GB transient branch forcing)."""
    return [a for a in all_artifacts() if a["on_demand"]]


def required_files(model: str) -> list[str]:
    """Files the job loads (for ArchesWeather: the extracted checkpoints and configs, not the zip they come from)."""
    if model in AW_MODELS:
        rels = [r for r, v in AW_ZIP_MEMBERS.items() if model in v[3]]
        rels += [r for r, v in AW_ZENODO_FILES.items() if model in v[4] and r != AW_ZIP_REL]
        return sorted(rels)
    rels = [a["rel_path"] for a in artifacts() if model in a["models"]]
    if model == "dlesym-aimip":
        rels.append(AIMIP_SST_REL)
    elif model in ("md-1.5", "md-1.5-aimip"):
        return [
            AIMIP_SST_REL,
            *rels,
        ]
    return rels


def _sha(rel: str) -> str:
    if rel in AW_ZIP_MEMBERS:
        return AW_ZIP_MEMBERS[rel][2]
    if rel in AW_ZENODO_FILES:
        return AW_ZENODO_FILES[rel][3]
    if rel == AIMIP_SST_REL:
        return AIMIP_SST_SHA256
    for a in all_artifacts():
        if a["rel_path"] == rel:
            return a["sha256"]
    raise KeyError(rel)


def registry_entries(p: str) -> list[dict[str, Any]]:
    """Registry entries with `${DATA}`-style prefix *p*."""
    s2s = ("The authors built the V1 checkpoints for subseasonal-to-seasonal forecasts (weeks to months); multi-year runs are "
           "outside what they validated. ")
    dlesym_notes = {
        "dlesym-v1.0": "DLESyM v1.0 (NVIDIA/UW, Earth2Studio): the prior ERA5 checkpoint of the S2S model, deprecated "
                       "upstream; aimip loads checkpoint pair 0 of 4. Coupled atmosphere-ocean on HEALPix-64, "
                       "output on 0.25 deg lat-lon, deterministic. " + s2s + "Multi-year rollouts can become unstable; "
                       "aimip stops a run at the first non-finite or out-of-band month. Starts "
                       "from ERA5 at --start (ARCO-ERA5, fetched at prepare time), not forced; --steps counts 6-h steps.",
        "dlesym-v1.1": "DLESyM v1.1 (Earth2Studio), AI Weather Quest checkpoint with conditional layer norm: samples noise "
                       "every step, so --members gives distinct members (seed + k). " + s2s + "aimip stops a run at the "
                       "first non-finite or out-of-band month. Diagnostic 6-h precipitation tp6 (monthly_precip.nc). "
                       "Starts from ERA5 at --start; --steps counts 6-h steps.",
        "dlesym-v0-isccp": "DLESyMv0-ISCCP (Earth2Studio): the climate checkpoint Earth2Studio documents for 100-1000-year "
                           "rollouts (Cresswell-Clay et al. 2024). It takes ISCCP-like OLR; with use_ttr the ERA5 OLR of "
                           "the initial state is converted by the authors' day-of-year moment matching, then the model "
                           "uses its own OLR. Deterministic; starts from ERA5 at --start; --steps counts 6-h steps.",
        "dlesym-aimip": "DLESyM, AIMIP Phase 1 version (UW; Cresswell-Clay et al. 2025, DLESyM_aimip-2026): the "
                        "dlesym-v0-isccp checkpoints with the SST prescribed from the AIMIP forcing (Zenodo 17065758) "
                        "instead of predicted, linear in time between the monthly stamps, as the authors' "
                        "ground-truth ocean. Deterministic; the authors' five members start from ERA5 at "
                        "1978-10-03 .. 07 00 UTC (one run per start); --steps counts 6-h steps, through 2024-12-31. "
                        "Trained on 1983-2016, so it saw part of the AIMIP evaluation period.",
    }
    out: list[dict[str, Any]] = []
    for name, pkg, ckpt_rel, stochastic in (
        ("dlesym-v1.0", "models/dlesym-v1", "models/dlesym-v1/checkpoints/v1.0/atmos/atmos_model_0.mdlus", False),
        ("dlesym-v1.1", "models/dlesym-v1", "models/dlesym-v1/checkpoints/v1.1/atmos/atmos_model_0.mdlus", True),
        ("dlesym-v0-isccp", "models/dlesym-v0-isccp", "models/dlesym-v0-isccp/atmos_model_0.mdlus", False),
        ("dlesym-aimip", "models/dlesym-v0-isccp", "models/dlesym-v0-isccp/atmos_model_0.mdlus", False),
    ):
        forcing_set = ({"name": "aimip", "dir": f"{p}/{os.path.dirname(AIMIP_SST_REL)}", "window": list(DLESYM_AIMIP_WINDOW)}
                       if name == "dlesym-aimip" else
                       {"name": "era5-initial-state", "dir": f"{p}/{pkg}", "window": list(DLESYM_IC_WINDOW)})
        out.append({
            "name": name, "family": "e2s-dlesym", "fme_module": "aimip.emu_job", "calendar": "proleptic_gregorian",
            "checkpoint": f"{p}/{ckpt_rel}", "checkpoint_sha256": _sha(ckpt_rel), "ic_files": [],
            "forcing_sets": [forcing_set],
            "required_files": required_files(name), "timestep_hours": 6, "stochastic": stochastic,
            "default_n_ensemble_per_ic": 1, "notes": dlesym_notes[name],
        })
    ackpt = "models/archesclimate-ssp/archesclimate-ssp-deterministic/model.ckpt"
    out.append({
        "name": "archesclimate-ssp", "family": "arches", "fme_module": "aimip.emu_job", "calendar": "proleptic_gregorian",
        "checkpoint": f"{p}/{ackpt}", "checkpoint_sha256": _sha(ackpt), "ic_files": [],
        "forcing_sets": [
            {"name": s, "dir": f"{p}/inputs/archesclimate-ssp/memmap_filled_in/r1i1p1f1_{s}_interpolation.memmap",
             "window": list(ARCHES_WINDOW)}
            for s in ("ssp434", "ssp585")
        ],
        "required_files": required_files("archesclimate-ssp"), "timestep_hours": 730, "stochastic": False,
        "default_n_ensemble_per_ic": 1,
        "notes": "ArchesClimate-SSP deterministic (INRIA/Mila, Clyne et al. 2026): monthly emulator of IPSL-CM6A-LR "
                 "on a 144 x 144 grid (1.25 deg lat x 2.5 deg lon), atmosphere + upper-ocean temperature, driven by "
                 "the scenario's forcing. Starts from the "
                 "scenario's 2015 state (--start 2015-01-01T00:00:00; first output March 2015); --steps counts months "
                 "(at most 1029); --forcing-set ssp434 (default, the authors' validation scenario) or ssp585 (test).",
    })
    aw_common = ("1-degree grid, 6 surface variables and 6 variables on 13 pressure levels as daily means, forced by "
                 "monthly ERA5 SST and sea-ice cover (Zenodo 22795233). Starts from the ERA5 daily-mean state of "
                 f"1978-09-30 only (--start {AW_START}; first output day 1978-10-01, like the authors' AIMIP member "
                 f"r2); --steps counts days (at most {AW_MAX_STEPS}, to 2024-12-31). aimip's rollout follows the paper "
                 "(Singh et al. 2026, arXiv:2605.29976, section 2.3); the authors' rollout script is not published.")
    for name, ckpt_rel, stochastic, lead in (
        ("archesweather", f"{AW_DIR}/{AW_SEEDS[0]}/{AW_CKPT}", False,
         "ArchesWeather-Mx4, AIMIP version (INRIA): the four ArchesWeather-M networks (seeds 0-3) averaged at every "
         "daily step, as in the authors' AIMIP runs; deterministic. "),
        ("archesweathergen", f"{AW_DIR}/{AW_GEN}/{AW_CKPT}", True,
         "ArchesWeatherGen, AIMIP version (INRIA): a flow-matching model that samples the next state around the "
         "ArchesWeather-Mx4 prediction (25 sampler steps per day); stochastic, --members gives distinct noise "
         "seeds (seed + k). "),
    ):
        out.append({
            "name": name, "family": "geoarches", "fme_module": "aimip.emu_job", "calendar": "proleptic_gregorian",
            "checkpoint": f"{p}/{ckpt_rel}", "checkpoint_sha256": _sha(ckpt_rel), "ic_files": [],
            "forcing_sets": [{"name": "aimip", "dir": f"{p}/{AW_INPUTS}", "window": [AW_START, AW_LAST_DAY]}],
            "required_files": required_files(name), "timestep_hours": 24, "stochastic": stochastic,
            "default_n_ensemble_per_ic": 1, "notes": lead + aw_common,
        })
    sckpt = "models/samudra-om4/samudra_thermo_dynamic_seed1.pt"
    out.append({
        "name": "samudra-om4", "family": "samudra", "fme_module": "aimip.emu_job", "calendar": "julian",
        "checkpoint": f"{p}/{sckpt}", "checkpoint_sha256": _sha(sckpt), "ic_files": [],
        "forcing_sets": [
            {"name": "om4-jra55do-replay", "dir": f"{p}/models/samudra-om4",
             "window": list(SAMUDRA_WINDOW)}
        ],
        "required_files": required_files("samudra-om4"), "timestep_hours": 120, "stochastic": False,
        "default_n_ensemble_per_ic": 1,
        "notes": "Samudra thermo-dynamic global ocean emulator of GFDL OM4 (M2LInES, Dheeshjith et al. 2025, "
                 "seed1 checkpoint): 1 deg grid (180 x 360, 19 fixed depths 2.5-6000 m), 5-day mean step (120 h, "
                 "even --steps >= 2), julian calendar. Replays a window of the OM4 JRA55-do OMIP-2 store "
                 "(staged from OSN s3://m2lines-pubs/Samudra at prepare time); independent test period starts "
                 "2014-09-30T12:00:00 (earlier dates are inside training/validation).",
    })
    mdckpt = "models/md-1.5/MD-1p5-latest.pth/weights.pt"
    out.append({
        "name": "md-1.5", "family": "md", "fme_module": "aimip.emu_job", "calendar": "proleptic_gregorian",
        "checkpoint": f"{p}/{mdckpt}", "checkpoint_sha256": _sha(mdckpt), "ic_files": [],
        "forcing_sets": [
            {"name": "aimip-era5-monthly", "dir": f"{p}/raw/zenodo-17065758",
             "window": list(MD15_WINDOW)},
        ],
        "required_files": required_files("md-1.5"), "timestep_hours": 730, "stochastic": True,
        "default_n_ensemble_per_ic": 1,
        "notes": "Monthly Diffusion v0.9 at 1.5 deg (UMD, Hall & Molina 2026, arXiv:2604.13481; "
                 "kjhall01/monthly-diffusion@b077d3bc MIT, MD-1p5-latest.pth): stochastic latent-diffusion "
                 "atmosphere emulator on 121 x 240 regular lat-lon (37 prognostic channels, 250 reverse DDPM "
                 "steps/month, single monthly state per step with counts=1). Forced by AIMIP Phase 1 monthly "
                 "SST/SIC (Zenodo 17065758, CC-BY-4.0, 556 month-start-centred points 1978-10-01 .. 2025-01-01; "
                 "training 1985-2014, independent test 2015-2022; requires >= 10-year spin-up). Upstream publishes "
                 "no initial state: pass --ic <era5_monthly_state.nc> or stage inputs/md-1.5/ic/ic_<YYYY-MM-DD>.nc.",
    })
    mdackpt = "models/md-1.5/MD-1p5.pth/weights.pt"
    out.append({
        "name": "md-1.5-aimip", "family": "md", "fme_module": "aimip.emu_job", "calendar": "proleptic_gregorian",
        "checkpoint": f"{p}/{mdackpt}", "checkpoint_sha256": _sha(mdackpt),
        "ic_files": [f"{p}/{MD15_AIMIP_INPUTS_DIR}/md15_era5_ic_1978-10-01_1p5.nc"],
        "forcing_sets": [
            {"name": "aimip-era5-monthly", "dir": f"{p}/raw/zenodo-17065758",
             "window": list(MD15_WINDOW)},
        ],
        "required_files": required_files("md-1.5-aimip"), "timestep_hours": 730, "stochastic": True,
        "default_n_ensemble_per_ic": 1,
        "notes": "MD-1.5 as run for AIMIP Phase 1 (Hall & Molina 2026, arXiv:2604.13481; kjhall01/monthly-diffusion"
                 "@b077d3bc MIT): MD-1p5.pth, the checkpoint the authors' run_aimip_runs.py loads, built with the "
                 "settings its tensors imply (the published code cannot read its config); 1985-2014 normalization "
                 "statistics and the ERA5 state of October 1978 rebuilt from ERA5 (not published; tools/md15); AIMIP "
                 "forcing regridded conservatively as in the authors' notebooks. --start 1978-10-01 (the AIMIP start); "
                 "--steps counts output months including the decoded initial state (555 = through December 2024); "
                 "members are batched draws of the diffusion noise, as in the authors' forced_run.",
    })
    cckpt = "models/camulator/checkpoint.pt00069.pt"
    out.append({
        "name": "camulator", "family": "camulator", "fme_module": "aimip.emu_job", "calendar": "noleap",
        "checkpoint": f"{p}/{cckpt}", "checkpoint_sha256": _sha(cckpt), "ic_files": [],
        "forcing_sets": [
            {"name": "camulator-cyclic-1yr", "dir": f"{p}/inputs/camulator/forcing_data",
             "window": list(CAMULATOR_CYCLIC_WINDOW)},
            {"name": "camulator-branch-1980-2014", "dir": f"{p}/inputs/camulator/forcing_data",
             "window": list(CAMULATOR_BRANCH_WINDOW)},
        ],
        "required_files": required_files("camulator"), "timestep_hours": 6, "stochastic": False,
        "default_n_ensemble_per_ic": 1,
        "notes": "CAMulator deterministic 6-hourly 1 deg 32-level atmosphere emulator (NCAR MILES/CREDIT, "
                 "Chapman et al. 2025, checkpoint.pt00069.pt). Emulates CAM6 / ERA5-scaled climate target "
                 "(1980-2014; 4 dynamic forcings SOLIN, SST, ICEFRAC, co2vmr_3d — differs from the 2-forcing "
                 "1979-2010 model in arXiv:2504.06007). Starts from any of 69 published noleap initial states "
                 "(1980-01-01, Jan 1 & Jul 1 of 1981-2014); --forcing-set camulator-cyclic-1yr (default) or "
                 "camulator-branch-1980-2014 (1980-2014 b.e21 forcing, SST source undocumented upstream). "
                 "Runs in isolated venv-camulator (torch 2.4.1, torch_harmonics 0.7.2, numpy<2).",
    })
    cbckpt = "models/cbottle-3d/cBottle-3d/training-state-009856000.checkpoint"
    out.append({
        "name": "cbottle-3d", "family": "cbottle", "fme_module": "aimip.emu_job", "calendar": "proleptic_gregorian",
        "checkpoint": f"{p}/{cbckpt}", "checkpoint_sha256": _sha(cbckpt), "ic_files": [],
        "forcing_sets": [
            {"name": "pcmdi-amip-1-1-9-sst", "dir": f"{p}/models/cbottle-3d",
             "window": list(CBOTTLE_WINDOW)},
        ],
        "required_files": required_files("cbottle-3d"), "timestep_hours": 24, "stochastic": True,
        "default_n_ensemble_per_ic": 1,
        "notes": "cBottle-3d conditional atmospheric state generator (NVIDIA, Brenowitz et al. 2025, "
                 "NVlabs/cBottle@f0f51998 via Earth2Studio CBottle3D; nvidia/cbottle@e60ce82b Apache-2.0): "
                 "not a time-stepping simulation — adjacent days are independent diffusion draws conditioned "
                 "only on (day_of_year, hour_of_day) and monthly mid-point SST (PCMDI-AMIP-1-1-9 tosbcs, "
                 "1940-01-01 .. 2022-12-16; dates >= 2018 are outside the 1980-2017 ERA5 training range; "
                 "no initial state, CO2, or sea-ice input). Default daily_synoptic cadence (00/06/12/18 UTC, "
                 "batch_size=4) writes monthly means on 0.25 deg lat-lon.",
    })
    sdckpt = "models/spherical-dyffusion/forecaster-sfno-best-inference_avg_crps.ckpt"
    out.append({
        "name": "spherical-dyffusion", "family": "sdyff", "fme_module": "aimip.emu_job", "calendar": "noleap",
        "checkpoint": f"{p}/{sdckpt}", "checkpoint_sha256": _sha(sdckpt), "ic_files": [],
        "forcing_sets": [
            {"name": "fv3gfs-repeating-climsst-1deg", "dir": f"{p}/inputs/spherical-dyffusion/fv3gfs-2021",
             "window": list(SDYFF_WINDOW)},
        ],
        "required_files": required_files("spherical-dyffusion"), "timestep_hours": 6, "stochastic": True,
        "default_n_ensemble_per_ic": 1,
        "notes": "Spherical DYffusion probabilistic 6-hourly 1 deg (180 x 360) atmosphere emulator of FV3GFS "
                 "(Rose-STL-Lab, Ruhling Cachay et al. NeurIPS 2024; salv47/spherical-dyffusion@56fcb5d9 Apache-2.0, "
                 "Zenodo 10791087 CC-BY-4.0). Emulates a repeating 365-day annual SST cycle (labelled 2021 in the "
                 "reference files; output months use nominal cycle indices 0001-01.. with exact daily coarsening "
                 "coarsen_factor=4 over complete calendar months). Predicts 34 fields; surface_temperature is "
                 "prescribed over ocean and stored as surface_temperature_prescribed_ocean, while predicted "
                 "lowest-layer air_temperature_7 (K) is summarized. Runs in isolated venv-sdyff (apex "
                 "FusedLayerNorm replaced by torch.nn.LayerNorm, numerically equivalent up to float rounding; "
                 "published checkpoints use normalization_layer='instance_norm' / torch.nn.InstanceNorm2d).",
    })
    return out
