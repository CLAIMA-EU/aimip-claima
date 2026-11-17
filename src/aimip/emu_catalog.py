"""Pinned files and registry entries of the DLESyM / ArchesClimate-SSP / ArchesWeather models (stdlib only).

`emu_sources.json` lists every file at a pinned Hugging Face revision with its size and sha256 (LFS oid, or the hash
of the downloaded git file).
"""
from __future__ import annotations

import json
import os
from typing import Any

EMU_MODELS: tuple[str, ...] = ("dlesym-v1.0", "dlesym-v1.1", "dlesym-v0-isccp", "archesclimate-ssp", "archesweather",
                               "archesweathergen")
DLESYM_IC_WINDOW: tuple[str, str] = ("1940-01-04T00:00:00", "2023-12-31T18:00:00")   # inputs reach back 48 h; v1.1 ttr03 needs
# trailing ERA5 accumulations that ERA5 lacks at its very start, leaving one day of margin.
ARCHES_WINDOW: tuple[str, str] = ("2015-01-01T00:00:00", "2100-11-01T00:00:00")   # outputs March 2015 .. November 2100

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
        return f"models/dlesym-v0-isccp/{path}", ("dlesym-v0-isccp",)
    if repo == "gclyne/ArchesClimate":
        return f"models/archesclimate-ssp/{path}", ("archesclimate-ssp",)
    if repo == "gclyne/ArchesClimate-data":
        if path.startswith("ipsl_scenarios/"):
            return f"inputs/archesclimate-ssp/memmap_filled_in/{path[len('ipsl_scenarios/'):]}", ("archesclimate-ssp",)
        return f"models/archesclimate-ssp/{path}", ("archesclimate-ssp",)
    return None


def artifacts() -> list[dict[str, Any]]:
    """[{rel_path, repo, repo_type, path, revision, size, sha256, models}] in a stable order."""
    out: list[dict[str, Any]] = []
    for repo, info in _load_sources().items():
        for path in sorted(info["files"]):
            placed = _place(repo, path)
            if placed is None:
                continue
            rel, models = placed
            f = info["files"][path]
            out.append({"rel_path": rel, "repo": repo, "repo_type": info.get("repo_type", "model"), "path": path,
                        "revision": info["revision"], "size": int(f["size"]), "sha256": str(f["sha256"]),
                        "models": models})
    return out


def required_files(model: str) -> list[str]:
    """Files the job loads (for ArchesWeather: the extracted checkpoints and configs, not the zip they come from)."""
    if model in AW_MODELS:
        rels = [r for r, v in AW_ZIP_MEMBERS.items() if model in v[3]]
        rels += [r for r, v in AW_ZENODO_FILES.items() if model in v[4] and r != AW_ZIP_REL]
        return sorted(rels)
    return [a["rel_path"] for a in artifacts() if model in a["models"]]


def _sha(rel: str) -> str:
    if rel in AW_ZIP_MEMBERS:
        return AW_ZIP_MEMBERS[rel][2]
    if rel in AW_ZENODO_FILES:
        return AW_ZENODO_FILES[rel][3]
    for a in artifacts():
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
    }
    out: list[dict[str, Any]] = []
    for name, pkg, ckpt_rel, stochastic in (
        ("dlesym-v1.0", "models/dlesym-v1", "models/dlesym-v1/checkpoints/v1.0/atmos/atmos_model_0.mdlus", False),
        ("dlesym-v1.1", "models/dlesym-v1", "models/dlesym-v1/checkpoints/v1.1/atmos/atmos_model_0.mdlus", True),
        ("dlesym-v0-isccp", "models/dlesym-v0-isccp", "models/dlesym-v0-isccp/atmos_model_0.mdlus", False),
    ):
        out.append({
            "name": name, "family": "e2s-dlesym", "fme_module": "aimip.emu_job", "calendar": "proleptic_gregorian",
            "checkpoint": f"{p}/{ckpt_rel}", "checkpoint_sha256": _sha(ckpt_rel), "ic_files": [],
            "forcing_sets": [{"name": "era5-initial-state", "dir": f"{p}/{pkg}",
                              "window": list(DLESYM_IC_WINDOW)}],
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
    return out
