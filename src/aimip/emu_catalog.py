"""Pinned files and registry entries of the DLESyM / ArchesClimate-SSP models (stdlib only).

`emu_sources.json` lists every file at a pinned Hugging Face revision with its size and sha256 (LFS oid, or the hash
of the downloaded git file).
"""
from __future__ import annotations

import json
import os
from typing import Any

EMU_MODELS: tuple[str, ...] = ("dlesym-v1.0", "dlesym-v1.1", "dlesym-v0-isccp", "archesclimate-ssp")
DLESYM_IC_WINDOW: tuple[str, str] = ("1940-01-04T00:00:00", "2023-12-31T18:00:00")   # inputs reach back 48 h; v1.1 ttr03 needs
# trailing ERA5 accumulations that ERA5 lacks at its very start, leaving one day of margin.
ARCHES_WINDOW: tuple[str, str] = ("2015-01-01T00:00:00", "2100-11-01T00:00:00")   # outputs March 2015 .. November 2100

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
    return [a["rel_path"] for a in artifacts() if model in a["models"]]


def _sha(rel: str) -> str:
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
    return out
