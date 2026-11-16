"""Static model metadata cards, status tiers, examples, and verified BibTeX citations for the 14 canonical models.

Cards are declarative data only — never executable instructions or dynamic imports.
Citations are included ONLY when verified against DOI or arXiv records; otherwise `bibtex` is empty
and `upstream_url` provides the authoritative upstream repository or model card URL.
"""

from __future__ import annotations

from typing import Any

TIER_DEFINITIONS: dict[str, str] = {
    "certified": (
        "Verified end-to-end in AIMIP CI and cluster validation runs (`fme` 2026.5.1 / 2026.10.0 "
        "on 1x NVIDIA A100-SXM4-80GB, 2026-11); artifacts pinned by SHA-256 in `aimip.lock`."
    ),
    "runs": (
        "Executes through the optional emulator environment (`e2s-dlesym` / `arches` on 1x NVIDIA "
        "A100-SXM4-80GB, 2026-11); operational or licensing caveats are stated in `limitations`."
    ),
    "experimental": (
        "Custom or unverified registry entry; not validated end-to-end in this repository."
    ),
}

ERA5_BIBTEX = """@article{hersbach2020era5,
  title   = {The {ERA5} global reanalysis},
  author  = {Hersbach, Hans and Bell, Bill and Berrisford, Paul and others},
  journal = {Quarterly Journal of the Royal Meteorological Society},
  volume  = {146},
  number  = {730},
  pages   = {1999--2049},
  year    = {2020},
  doi     = {10.1002/qj.3803}
}"""

ACE2_BIBTEX = """@article{wattmeyer2024ace2,
  title         = {{ACE2}: Accurately learning subseasonal to decadal atmospheric variability and forced responses},
  author        = {Watt-Meyer, Oliver and Henn, Brian and McGibbon, Jeremy and Clark, Spencer K. and Kwa, Anna and Perkins, W. Andre and Wu, Elynn and Harris, Lucas and Bretherton, Christopher S.},
  journal       = {arXiv preprint arXiv:2411.11268},
  eprint        = {2411.11268},
  archivePrefix = {arXiv},
  year          = {2024},
  url           = {https://arxiv.org/abs/2411.11268}
}"""

ACE2_SOM_BIBTEX = """@article{clark2024ace2som,
  title         = {{ACE2-SOM}: Coupling an {ML} atmospheric emulator to a slab ocean and learning the sensitivity of climate to changed {CO2}},
  author        = {Clark, Spencer K. and Watt-Meyer, Oliver and Kwa, Anna and McGibbon, Jeremy and Henn, Brian and Perkins, W. Andre and Wu, Elynn and Harris, Lucas and Bretherton, Christopher S.},
  journal       = {arXiv preprint arXiv:2412.04418},
  eprint        = {2412.04418},
  archivePrefix = {arXiv},
  year          = {2024},
  url           = {https://arxiv.org/abs/2412.04418}
}"""

SAMUDRACE_BIBTEX = """@article{duncan2025samudrace,
  title         = {{SamudrACE}: Fast and Accurate Coupled Climate Modeling with {3D} Ocean and Atmosphere Emulators},
  author        = {Duncan, James and Wu, Elynn and Dheeshjith, Surya and Subel, Adam and Arcomano, Troy and Clark, Spencer K. and Henn, Brian and Kwa, Anna and McGibbon, Jeremy and Perkins, W. Andre and Gregory, William and Fernandez-Granda, Carlos and Busecke, Julius and Watt-Meyer, Oliver and Hurlin, William J. and Adcroft, Alistair and Zanna, Laure and Bretherton, Christopher S.},
  journal       = {arXiv preprint arXiv:2509.12490},
  eprint        = {2509.12490},
  archivePrefix = {arXiv},
  year          = {2025},
  url           = {https://arxiv.org/abs/2509.12490}
}"""

DLESYM_BIBTEX = """@article{cresswellclay2025dlesym,
  title   = {A deep learning Earth system model for efficient simulation of the observed climate},
  author  = {Cresswell-Clay, Nathaniel and Liu, Bowen and Durran, Dale and Liu, Zihui and Espinosa, Zachary I. and Moreno, Raul and Karlbauer, Matthias},
  journal = {AGU Advances},
  volume  = {6},
  pages   = {e2025AV001706},
  year    = {2025},
  doi     = {10.1029/2025AV001706},
  eprint  = {2409.16247}
}"""

FME_COST_NOTE = (
    "~0.064 GPU-h per model year per member (estimated from ACE2-ERA5 1x A100-80GB baseline)"
)
DLESYM_COST_NOTE = (
    "~40 s per model year per member (DLESyM 1x A100-80GB baseline)"
)
ARCHES_COST_NOTE = (
    "~2 s per simulated month (ArchesClimate-SSP 1x A100-80GB baseline)"
)
PARENT_ACE2_CITATION_NOTE = (
    "Reference is for the parent ACE2 base architecture (arXiv:2411.11268); "
    "no separate published paper for this checkpoint was verified — also cite the upstream model card URL."
)

MODEL_CARDS: dict[str, dict[str, Any]] = {
    "ace2-era5": {
        "title": "ACE2-ERA5 (Ai2)",
        "task": "amip",
        "status": "certified",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/ACE2-ERA5",
        "grid": "180 x 360 Gaussian (1 deg)",
        "outputs": ["TMP2m", "surface_temperature", "PRATE", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": "Deterministic AMIP atmosphere forced by prescribed ERA5 SST, sea ice, and CO2 (1979-01-01..2014-12-31T18:00:00).",
        "examples": [
            "aimip plan ace2-era5 --start 1979-01-01 --length 1y --name era5-1y",
            "aimip run ace2-era5 --start 2001-01-01 --length 5y --name era5-2001",
        ],
        "bibtex": [ACE2_BIBTEX, ERA5_BIBTEX],
    },
    "ace2-shield": {
        "title": "ACE2-SHiELD (Ai2)",
        "task": "amip",
        "status": "certified",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/ACE2-SHiELD",
        "data_urls": ["https://zenodo.org/records/17065758"],
        "grid": "180 x 360 Gaussian (1 deg)",
        "outputs": ["TMP2m", "surface_temperature", "PRATE", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": "Deterministic AMIP emulator trained on GFDL SHiELD; amip-hist (1979-2014) and amip-oos (2014-2024) must be run as separate segments (pass --ic <hist-run>/restart.nc for amip-oos).",
        "examples": [
            "aimip plan ace2-shield --start 1979-01-01 --length 2y --name shield-demo",
            "aimip run ace2-shield --start 1979-01-01 --end 2014-12-31T18:00:00 --forcing-set amip-hist --name shield-hist",
        ],
        "citation_note": PARENT_ACE2_CITATION_NOTE,
        "bibtex": [ACE2_BIBTEX],
    },
    "ace2s-shield-plus": {
        "title": "ACE2S-SHiELD-plus (Ai2)",
        "task": "amip",
        "status": "certified",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/ACE2S-SHiELD-plus",
        "data_urls": ["https://zenodo.org/records/17065758"],
        "grid": "180 x 360 Gaussian (1 deg)",
        "outputs": ["TMP2m", "surface_temperature", "PRATE", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": "Stochastic AMIP emulator; ensemble members share the start state and diverge via internal stochastic noise.",
        "examples": [
            "aimip plan ace2s-shield-plus --start 1979-01-01 --length 1y --members 4 --name shieldp-ens",
            "aimip run ace2s-shield-plus --start 1979-01-01 --steps 40 --members 2 --name shieldp-short",
        ],
        "citation_note": PARENT_ACE2_CITATION_NOTE,
        "bibtex": [ACE2_BIBTEX],
    },
    "ace2-eamv3": {
        "title": "ACE2-EAMv3 (Ai2)",
        "task": "amip",
        "status": "certified",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/ACE2-EAMv3",
        "grid": "180 x 360 equirectangular (1 deg, noleap)",
        "outputs": ["TREFHT", "TS", "PRECT", "PS"],
        "cost_note": FME_COST_NOTE,
        "limitations": "Deterministic AMIP emulator trained on E3SMv3 EAM; 365-day noleap calendar, window 1994-2014.",
        "examples": [
            "aimip plan ace2-eamv3 --start 1994-01-01 --length 20y --name eamv3-hist",
            "aimip run ace2-eamv3 --start 1994-01-01 --length 1y --name eamv3-1y",
        ],
        "citation_note": PARENT_ACE2_CITATION_NOTE,
        "bibtex": [ACE2_BIBTEX],
    },
    "hiro-ace": {
        "title": "HiRO-ACE (Ai2, 1994-2024)",
        "task": "amip",
        "status": "certified",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/HiRO-ACE",
        "data_urls": ["https://zenodo.org/records/17065758"],
        "grid": "180 x 360 Gaussian (1 deg, julian)",
        "outputs": ["TMP2m", "surface_temperature", "PRATE", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": "Stochastic ACE2S global stage with derived 1994-2014 historical and 2014-2024 out-of-sample forcing; cite upstream model card for HiRO-ACE specifics.",
        "examples": [
            "aimip plan hiro-ace --start 1994-01-01 --length 5y --members 3 --name hiro-94",
            "aimip run hiro-ace --start 1994-01-01 --length 1y --members 2 --name hiro-demo",
        ],
        "citation_note": PARENT_ACE2_CITATION_NOTE,
        "bibtex": [ACE2_BIBTEX],
    },
    "ace2-som": {
        "title": "ACE2-SOM (Ai2 Slab Ocean)",
        "task": "slab-ocean",
        "status": "certified",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/ACE2-SOM",
        "grid": "180 x 360 Gaussian (1 deg)",
        "outputs": ["TMP2m", "surface_temperature", "PRATE", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": "Slab-ocean equilibrium response under 1xCO2 or 3xCO2 forcing (2032-01-01T00:00:00..2035-12-31T18:00:00); specify --forcing-set 1xCO2 or 3xCO2.",
        "examples": [
            "aimip plan ace2-som --start 2032-01-01 --end 2035-12-31T18:00:00 --forcing-set 1xCO2 --name som-1x",
            "aimip run ace2-som --start 2032-01-01 --length 3y --forcing-set 3xCO2 --name som-3x",
        ],
        "bibtex": [ACE2_SOM_BIBTEX, ACE2_BIBTEX],
    },
    "ace2s": {
        "title": "ACE2S (Ai2, 2014-2023)",
        "task": "amip",
        "status": "certified",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/HiRO-ACE",
        "grid": "180 x 360 Gaussian (1 deg, julian)",
        "outputs": ["TMP2m", "surface_temperature", "PRATE", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": "Stochastic global atmosphere emulator with native upstream 2014-2023 AMIP forcing.",
        "examples": [
            "aimip plan ace2s --start 2014-01-01 --length 5y --members 4 --name ace2s-ens",
            "aimip run ace2s --start 2014-01-01 --length 1y --members 2 --name ace2s-1y",
        ],
        "citation_note": PARENT_ACE2_CITATION_NOTE,
        "bibtex": [ACE2_BIBTEX],
    },
    "hiro-acev2": {
        "title": "HiRO-ACEv2 (Ai2, fme 2026.10.0)",
        "task": "amip",
        "status": "certified",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/HiRO-ACEv2",
        "grid": "180 x 360 Gaussian (1 deg, julian)",
        "outputs": ["TMP2m", "surface_temperature", "PRATE", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": "Uses fme 2026.10.0 overlay; stochastic global stage over 2014-2023 AMIP window; cite upstream model card for HiRO-ACEv2 specifics.",
        "examples": [
            "aimip plan hiro-acev2 --start 2014-01-01 --length 2y --members 3 --name hiro2-demo",
            "aimip run hiro-acev2 --start 2014-01-01 --steps 20 --members 2 --name hiro2-short",
        ],
        "citation_note": PARENT_ACE2_CITATION_NOTE,
        "bibtex": [ACE2_BIBTEX],
    },
    "samudrace-cm4": {
        "title": "SamudrACE-CM4 piControl (Ai2 Coupled)",
        "task": "coupled",
        "status": "certified",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/SamudrACE-CM4-piControl",
        "grid": "180 x 360 coupled atmosphere-ocean (5-day coupling step, noleap)",
        "outputs": ["atmosphere/TMP2m", "ocean/sst", "ocean/thetao_0"],
        "cost_note": FME_COST_NOTE,
        "limitations": "Coupled 5-day (120h) step on noleap calendar; two 100-year pre-industrial windows (0151-0250 and 0311-0410).",
        "examples": [
            "aimip plan samudrace-cm4 --start 0311-01-01 --length 10y --name cm4-10y",
            "aimip run samudrace-cm4 --start 0311-01-01 --length 30d --name cm4-30d",
        ],
        "bibtex": [SAMUDRACE_BIBTEX],
    },
    "samudrace-e3smv3": {
        "title": "SamudrACE-E3SMv3 (Ai2 Coupled, fme 2026.10.0)",
        "task": "coupled",
        "status": "certified",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/SamudrACE-E3SMv3",
        "grid": "180 x 360 coupled atmosphere-ocean (5-day coupling step, noleap)",
        "outputs": ["atmosphere/Tat2m", "ocean/temperatureCoarsened_0"],
        "cost_note": FME_COST_NOTE,
        "limitations": "Stochastic coupled model (fme 2026.10.0 overlay) with 3-state shipped IC bundle over 0425-01-03T12:00:00..0425-12-29T12:00:00 (max 72 5-day steps).",
        "examples": [
            "aimip plan samudrace-e3smv3 --start 0425-01-03T12:00:00 --end 0425-12-29T12:00:00 --members 3 --name e3sm-full",
            "aimip run samudrace-e3smv3 --start 0425-01-03T12:00:00 --length 30d --name e3sm-30d",
        ],
        "bibtex": [SAMUDRACE_BIBTEX],
    },
    "dlesym-v1.0": {
        "title": "DLESyM v1.0 (NVIDIA / UW, Earth2Studio)",
        "task": "coupled",
        "status": "runs",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Apache-2.0",
        "upstream_url": "https://github.com/NVIDIA/earth2studio",
        "grid": "HEALPix-64 native -> 0.25 deg lat-lon output",
        "outputs": ["atmosphere/t2m", "ocean/sst"],
        "cost_note": DLESYM_COST_NOTE,
        "limitations": "Built upstream for subseasonal-to-seasonal forecasts; multi-year rollouts are outside author validation and stop cleanly at the first unstable month. Initial ERA5 state is fetched at prepare time.",
        "examples": [
            "aimip plan dlesym-v1.0 --start 1995-01-01 --length 1y --name dlesym10-1y",
            "aimip run dlesym-v1.0 --start 1995-01-01 --length 30d --name dlesym10-30d",
        ],
        "bibtex": [DLESYM_BIBTEX, ERA5_BIBTEX],
    },
    "dlesym-v1.1": {
        "title": "DLESyM v1.1 Stochastic (NVIDIA / UW, Earth2Studio)",
        "task": "coupled",
        "status": "runs",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Apache-2.0",
        "upstream_url": "https://github.com/NVIDIA/earth2studio",
        "grid": "HEALPix-64 native -> 0.25 deg lat-lon output",
        "outputs": ["atmosphere/t2m", "ocean/sst", "precip/tp6", "precip/tp06"],
        "cost_note": DLESYM_COST_NOTE,
        "limitations": "Stochastic S2S coupled checkpoint with diagnostic 6h precipitation (tp6/tp06); multi-year rollouts stop cleanly at the first unstable month. Initial ERA5 state is fetched at prepare time.",
        "examples": [
            "aimip plan dlesym-v1.1 --start 1995-01-01 --length 1y --members 3 --name dlesym11-ens",
            "aimip run dlesym-v1.1 --start 2010-06-01 --length 30d --members 2 --name dlesym11-30d",
        ],
        "bibtex": [DLESYM_BIBTEX, ERA5_BIBTEX],
    },
    "dlesym-v0-isccp": {
        "title": "DLESyM v0-ISCCP Climate Checkpoint (NVIDIA / UW)",
        "task": "coupled",
        "status": "runs",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Apache-2.0",
        "upstream_url": "https://github.com/AtmosSci-DLESM/DLESyM",
        "grid": "HEALPix-64 native -> 0.25 deg lat-lon output",
        "outputs": ["atmosphere/t2m", "ocean/sst"],
        "cost_note": DLESYM_COST_NOTE,
        "limitations": "Documented upstream for centennial coupled rollouts using day-of-year OLR moment matching on the initial ERA5 state.",
        "examples": [
            "aimip plan dlesym-v0-isccp --start 1995-01-01 --length 10y --name dlesym0-10y",
            "aimip run dlesym-v0-isccp --start 1995-01-01 --length 1y --name dlesym0-1y",
        ],
        "bibtex": [DLESYM_BIBTEX, ERA5_BIBTEX],
    },
    "archesclimate-ssp": {
        "title": "ArchesClimate-SSP Deterministic (INRIA / Mila)",
        "task": "scenario",
        "status": "runs",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2026-11)",
        "licence": "Source fetched on first use (no upstream licence file; see THIRD_PARTY.md)",
        "upstream_url": "https://github.com/grahamclyne/ArchesClimate-SSP",
        "data_urls": ["https://huggingface.co/datasets/gclyne/ArchesClimate-data"],
        "grid": "144 x 144 (1.25 deg lat x 2.5 deg lon)",
        "outputs": ["atmosphere/tas", "atmosphere/psl", "atmosphere/pr", "ocean/thetao_0", "ocean/thetao"],
        "cost_note": ARCHES_COST_NOTE,
        "limitations": "Monthly IPSL-CM6A-LR emulator starting at 2015-01-01T00:00:00 (first output 2015-03, max 1029 months); choose --forcing-set ssp434 or ssp585.",
        "examples": [
            "aimip plan archesclimate-ssp --start 2015-01-01 --length 10y --forcing-set ssp585 --name arches-585",
            "aimip run archesclimate-ssp --start 2015-01-01 --length 12m --forcing-set ssp434 --name arches-1y",
        ],
        "bibtex": [],
    },
}


def card_for_model(model_name: str) -> dict[str, Any]:
    """Return a copy of the curated card metadata for *model_name* (or honest `experimental` defaults for custom models)."""
    base = MODEL_CARDS.get(model_name)
    if base is not None:
        status = str(base["status"])
        return {
            **base,
            "status_definition": TIER_DEFINITIONS.get(status, ""),
            "outputs": list(base["outputs"]),
            "examples": list(base["examples"]),
            "bibtex": list(base["bibtex"]),
            "data_urls": list(base.get("data_urls") or ()),
        }
    return {
        "title": model_name,
        "task": "amip",
        "status": "experimental",
        "status_definition": TIER_DEFINITIONS["experimental"],
        "verified_on": None,
        "licence": "not recorded",
        "upstream_url": None,
        "data_urls": [],
        "grid": "not recorded (custom registry entry)",
        "outputs": [],
        "cost_note": "not recorded",
        "limitations": "Custom registry entry; status, licence, and literature references are not recorded in this repository.",
        "examples": [],
        "citation_note": None,
        "bibtex": [],
    }
