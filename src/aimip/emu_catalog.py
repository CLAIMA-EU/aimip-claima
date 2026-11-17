"""Pinned files and registry entries of the DLESyM / ArchesClimate-SSP / ArchesWeather models (stdlib only).

`emu_sources.json` lists every file at a pinned Hugging Face revision with its size and sha256 (LFS oid, or the hash
of the downloaded git file).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

EMU_MODELS: tuple[str, ...] = ("dlesym-v1.0", "dlesym-v1.1", "dlesym-v0-isccp", "dlesym-aimip", "archesclimate-ssp",
                               "archesweather", "archesweathergen", "md-1.5")
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

# MD-1.5 (Monthly Diffusion v0.9, UMD PARETO; Hall and Molina 2026, arXiv:2604.13481; AIMIP Phase 1 entry): code and
# checkpoint from Zenodo 21430292 (MIT). The AIMIP run used the checkpoint MD-1p5.pth, whose config.pkl was written by
# earlier, unpublished code; the job builds the published CVAE with the settings its tensors imply (emu_job.MD_ARCH).
# Monthly steps on a 1.5-degree grid (121 x 240, poles included) from the ERA5 monthly mean of October 1978, forced by
# the AIMIP SST and sea ice (Zenodo 17065758, regridded by the job) and a land-sea mask. The initial state and the
# 1985-2014 normalization statistics (the training period) are rebuilt from ERA5 (tools/md15) and published separately.
MD_MODELS: tuple[str, ...] = ("md-1.5",)
MD_RECORD = "21430292"
MD_START = "1978-10-01T00:00:00"
MD_MAX_STEPS = 554                       # monthly steps after October 1978, to December 2024
MD_DIR = "models/md-1.5"
MD_INPUTS = "inputs/md-1.5"
MD_ZIP_FILE = "kjhall01/monthly-diffusion-MDv0.9-1.5-licensed.zip"     # Zenodo file key
MD_ZIP_REL = f"raw/zenodo-{MD_RECORD}/monthly-diffusion-MDv0.9-1.5-licensed.zip"
MD_ZIP_SIZE, MD_ZIP_MD5 = 63775851, "3c568f95e6151a0704e84ae7a8d2cd5c"
MD_ZIP_SHA256 = "73f2d88e43ceb6f3e9a2a164ce3649ebaef0c9abbe7a359cf4ad2a805476ed58"
MD_ZIP_PREFIX = "kjhall01-monthly-diffusion-b077d3b/"
MD_CODE = f"{MD_DIR}/code"               # the published `md` package, imported by the job without writing bytecode
MD_WEIGHTS = f"{MD_DIR}/MD-1p5/weights.pt"
MD_CONFIG = f"{MD_DIR}/MD-1p5/config.pkl"
# rel_path -> (zip member, size, sha256)
MD_ZIP_MEMBERS: dict[str, tuple[str, int, str]] = {
    f"{MD_CODE}/md/{name}": (f"{MD_ZIP_PREFIX}md/{name}", size, sha) for name, size, sha in (
        ("__init__.py", 587, "3c9ad85f5487bfa92f9b6f04b3d6e823f38f9c6ee8fd74f6be6608cb384ec699"),
        ("blocks.py", 8898, "818b05c5ddd904160d86aa2f7465dfcf725d40f9187c9490135191497ddb75e8"),
        ("cvae.py", 22997, "d633e79d7ba427915b7e28f30370a08f4573374462f9a93cc5eff674d7dee5f4"),
        ("data_utils.py", 4868, "f3f028e630f1debaa511e02b79be315ecc95af121f847351b2ae9da6744738fe"),
        ("decoder.py", 2634, "d60f2060b90e5e4b6df5b286215be42c15c85548896f0f3534c96cbb3b8aac98"),
        ("diffusion.py", 8919, "10ea2a022bb9cb10edfa475b82742f6d0b665cdba15552753851d687c08a0654"),
        ("encoder.py", 3562, "7b7922277ee3d7ae1f70a50d58c69842d45e30d689fd90f9be6f2f4f50ec5313"),
        ("positional.py", 4236, "1d9a0ba6e41689f6d991eb1861eaa1fe6a3a76eba50d4ff9a4fd24ca258b4f41"),
        ("prep_data.py", 2711, "e2f784ff4f18731c0529785c1a155342f686adee2ff9156f9c05579526da4e35"),
        ("preprocessing.py", 11313, "a45f149c2f614c09df21344ce09a63825b420f29d7962b618d3a4185528d3903"),
        ("saveload.py", 1482, "16b1f26241ae11bf992208d9f143381c50d4a880adb7434d668cbb5ef0e9a0e5"),
        ("spatial_conditional_norm.py", 2256, "62557b7fd9789447d27fe674b50ae9d433256e23c86eca174522eabf72848e95"),
        ("spectral_conv.py", 7275, "b2eb642eca58a8eea6f506db181ae610c7ae72b0fc7f468630314ba7ca0551df"),
        ("spectral_fc.py", 2434, "ed76af0bde88a5a196f2f0ad01109df3450eba698454b8a082c20ad1e4efe51b"),
    )
}
MD_ZIP_MEMBERS[MD_WEIGHTS] = (f"{MD_ZIP_PREFIX}models/MD-1p5.pth/weights.pt", 28998624,
                              "7a217274bcb83d2d262412ccf21358d45dac4d4bac6fdb37c42da7f7339e2da2")
MD_ZIP_MEMBERS[MD_CONFIG] = (f"{MD_ZIP_PREFIX}models/MD-1p5.pth/config.pkl", 582489,
                             "c652426f3f425221b9105db42e34848a6f51e7da552c3c2362f0476e8328d29b")
MD_ZIP_MEMBERS[f"{MD_DIR}/LICENSE"] = (f"{MD_ZIP_PREFIX}LICENSE", 1066,
                                       "36f9042913f5f3f042a98961c649a16ef795d10da00ca71b595b936ef7e9960f")
# The rebuilt inputs. PENDING: the record that will hold them (rel_path -> (file, size, md5, sha256)).
MD_INPUTS_RECORD = "PENDING"
MD_IC = f"{MD_INPUTS}/md15_era5_ic_197810_1p5.nc"
MD_STATS = f"{MD_INPUTS}/md15_variable_statistics_198501-201412.json"
MD_INPUT_FILES: dict[str, tuple[str, int, str, str]] = {
    MD_IC: ("md15_era5_ic_197810_1p5.nc", 6772776, "8c52ebcfa73646ecb88f3732c7b9783d",
            "cb38ff2fb434735b815c72adfe6151fcdf1e406effe95b884604629c1abc03da"),
    MD_STATS: ("md15_variable_statistics_198501-201412.json", 3651, "d8b17c0dff9f6a79b98ad1425900fd16",
               "007d471b0187fb63ee2a8e1ed0ae8486eb5761e8958bc1d1d0d2933602a367d6"),
}

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
    if model in MD_MODELS:
        return sorted(list(MD_ZIP_MEMBERS) + list(MD_INPUT_FILES) + [AIMIP_SST_REL])
    rels = [a["rel_path"] for a in artifacts() if model in a["models"]]
    if model == "dlesym-aimip":
        rels.append(AIMIP_SST_REL)
    return rels


def _sha(rel: str) -> str:
    if rel in AW_ZIP_MEMBERS:
        return AW_ZIP_MEMBERS[rel][2]
    if rel in AW_ZENODO_FILES:
        return AW_ZENODO_FILES[rel][3]
    if rel == AIMIP_SST_REL:
        return AIMIP_SST_SHA256
    if rel in MD_ZIP_MEMBERS:
        return MD_ZIP_MEMBERS[rel][2]
    if rel in MD_INPUT_FILES:
        return MD_INPUT_FILES[rel][3]
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
    out.append({
        "name": "md-1.5", "family": "md", "fme_module": "aimip.emu_job", "calendar": "proleptic_gregorian",
        "checkpoint": f"{p}/{MD_WEIGHTS}", "checkpoint_sha256": _sha(MD_WEIGHTS), "ic_files": [],
        "forcing_sets": [{"name": "aimip", "dir": f"{p}/{os.path.dirname(AIMIP_SST_REL)}",
                          "window": [MD_START, "2024-12-01T00:00:00"]}],
        "required_files": required_files("md-1.5"), "timestep_hours": 730, "stochastic": True,
        "default_n_ensemble_per_ic": 1,
        "notes": "MD-1.5 (Monthly Diffusion v0.9, UMD PARETO; Hall and Molina 2026), the AIMIP Phase 1 entry: latent "
                 "diffusion in a conditional VAE with spherical-Fourier blocks, one step per month on a 1.5-degree "
                 "grid (37 channels: 8 surface fields, T/q/u/v on 7 pressure levels, z500), forced by the AIMIP SST "
                 "and sea ice. Starts from the ERA5 monthly mean of October 1978 only (--start 1978-10-01T00:00:00; "
                 "the first output month is the decoded initial state, as in the authors' submission); --steps "
                 f"counts months (at most {MD_MAX_STEPS}, to December 2024); --members gives noise seeds (seed + k). "
                 "The published code cannot load the checkpoint's own config; aimip reads the architecture from its "
                 "tensors (global-mean 2 m temperature 1979-2024 about 0.035 K below the authors' submission).",
    })
    return out
