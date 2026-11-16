"""Static model metadata cards, status tiers, examples, and verified BibTeX citations for the 14 canonical models.

Cards are declarative data only — never executable instructions or dynamic imports.
Citations are included ONLY when verified against DOI or arXiv records; otherwise `bibtex` is empty
and `upstream_url` provides the authoritative upstream repository or model card URL.
"""

from __future__ import annotations

from typing import Any

TIER_DEFINITIONS: dict[str, str] = {
    "certified": (
        "Software path verified on cluster GPU hardware across both container and native installs "
        "with bit-identical outputs (`docs/installation.md`); artifacts pinned by SHA-256 in `aimip.lock`. "
        "Describes software reproducibility in this repository, not scientific validation."
    ),
    "runs": (
        "Software path wired and pinned by SHA-256 in `aimip.lock` (`fme` or optional `venv-emu` environment); "
        "either cluster-tested (`verified_on` set) or sharing an adapter with a verified sibling (`verified_on` unset). "
        "Describes software execution only, not scientific validation."
    ),
    "experimental": (
        "Out-of-training-period extension or custom/unverified registry entry; "
        "treat outputs as exploratory experiments and review `limitations`."
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

HIRO_ACE_BIBTEX = """@article{perkins2025hiroace,
  title         = {{HiRO-ACE}: Fast and skillful {AI} emulation and downscaling trained on a 3 km global storm-resolving model},
  author        = {Perkins, W. Andre and Kwa, Anna and McGibbon, Jeremy and Arcomano, Troy and Clark, Spencer K. and Watt-Meyer, Oliver and Bretherton, Christopher S. and Harris, Lucas},
  journal       = {arXiv preprint arXiv:2512.18224},
  eprint        = {2512.18224},
  archivePrefix = {arXiv},
  year          = {2025},
  url           = {https://arxiv.org/abs/2512.18224}
}"""

HIRO_ACEV2_BIBTEX = """@article{kwa2026hiroacev2,
  title         = {Extreme events and statistics in an {AI} emulator of a global storm-resolving model},
  author        = {Kwa, Anna and Perkins, W. Andre and Arcomano, Troy and Brenowitz, Noah D. and Bretherton, Christopher S. and Clark, Spencer K. and Harris, Lucas and McGibbon, Jeremy and Pritchard, Michael S. and Watt-Meyer, Oliver},
  journal       = {arXiv preprint arXiv:2610.17693},
  eprint        = {2610.17693},
  archivePrefix = {arXiv},
  year          = {2026},
  url           = {https://arxiv.org/abs/2610.17693}
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

SAMUDRACE_BIBTEX = """@article{duncan2026samudrace,
  title         = {{SamudrACE}: Fast and Accurate Coupled Climate Modeling with {3D} Ocean and Atmosphere Emulators},
  author        = {Duncan, James and Wu, Elynn and Dheeshjith, Surya and Subel, Adam and Arcomano, Troy and Clark, Spencer K. and Henn, Brian and Kwa, Anna and McGibbon, Jeremy and Perkins, W. Andre and Gregory, William and Fernandez-Granda, Carlos and Busecke, Julius and Watt-Meyer, Oliver and Hurlin, William J. and Adcroft, Alistair and Zanna, Laure and Bretherton, Christopher S.},
  journal       = {Geophysical Research Letters},
  year          = {2026},
  doi           = {10.1029/2025GL119340},
  eprint        = {2509.12490},
  archivePrefix = {arXiv},
  url           = {https://doi.org/10.1029/2025GL119340}
}"""

SAMUDRACE_E3SMV3_BIBTEX = """@article{wu2026samudrace_e3smv3,
  title         = {Stochastic Emulation of a Fully Coupled Preindustrial {E3SMv3} Simulation},
  author        = {Wu, Elynn and Duncan, James P. C. and Arcomano, Troy and McGibbon, Jeremy and Watt-Meyer, Oliver and Bretherton, Christopher S. and Mahfouz, Naser and Tebaldi, Claudia and others},
  journal       = {arXiv preprint arXiv:2608.10277},
  eprint        = {2608.10277},
  archivePrefix = {arXiv},
  year          = {2026},
  url           = {https://arxiv.org/abs/2608.10277}
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
HIRO_ACEV2_CITATION_NOTE = (
    "References cover HiRO-ACEv2 (arXiv:2610.17693), the HiRO-ACE framework (arXiv:2512.18224), "
    "and the parent ACE2 architecture (arXiv:2411.11268); also cite the upstream HiRO-ACEv2 model card URL."
)

MODEL_CARDS: dict[str, dict[str, Any]] = {
    "ace2-era5": {
        "title": "ACE2-ERA5 (Ai2)",
        "task": "amip",
        "status": "certified",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (8 steps, container == native bit-identical, 1.3.0/1.3.1)",
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/ACE2-ERA5",
        "grid": "180 x 360 Gaussian (1 deg)",
        "outputs": ["TMP2m", "surface_temperature", "PRATEsfc", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": (
            'Authors\' limits (allenai/ACE2-ERA5): "not expected to generalize accurately for large '
            'perturbations of certain inputs (e.g. doubling of CO2)"; separate sensitivities to SST and CO2 '
            'are "not entirely realistic"; "ACE models are research tools and should not be used for '
            'operational climate predictions." AIMIP: forcing window 1979-2014.'
        ),
        "examples": [
            "aimip plan ace2-era5 --start 1979-01-01 --length 1y --name era5-1y",
            "aimip run ace2-era5 --start 2001-01-01 --length 5y --name era5-2001",
        ],
        "bibtex": [ACE2_BIBTEX, ERA5_BIBTEX],
    },
    "ace2-shield": {
        "title": "ACE2-SHiELD (Ai2)",
        "task": "amip",
        "status": "runs",
        "verified_on": None,
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/ACE2-SHiELD",
        "data_urls": ["https://zenodo.org/records/17065758"],
        "grid": "180 x 360 Gaussian (1 deg)",
        "outputs": ["TMP2m", "surface_temperature", "PRATEsfc", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": (
            'Authors\' limits (allenai/ACE2-SHiELD): "ACE models are research tools and should not be used '
            'for operational climate predictions"; "The behavior of ACE2-SHiELD is similar to that of '
            'ACE2-ERA5." AIMIP: upstream ships no IC (uses the 1979-01-01 IC from ACE2S-SHiELD-plus); run '
            'amip-hist (1979-2014) and amip-oos (2014-2024, derived AIMIP SST) as separate segments '
            '(--ic <hist-run>/restart.nc); not cluster-compared across modes in 1.3.x.'
        ),
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
        "status": "runs",
        "verified_on": None,
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/ACE2S-SHiELD-plus",
        "data_urls": ["https://zenodo.org/records/17065758"],
        "grid": "180 x 360 Gaussian (1 deg)",
        "outputs": ["TMP2m", "surface_temperature", "PRATEsfc", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": (
            'Authors\' limits (allenai/ACE2S-SHiELD-plus): emulates independent SST/CO2 perturbations, '
            'but "inherits the biases relative to observations" of SHiELD, "generalization ability to '
            'unseen sea ice coverage is limited", "abrupt regime shifts in stratospheric temperature and '
            'moisture can occur in inference runs", and "ACE models are research tools and should not be '
            'used for operational climate predictions." AIMIP: single shipped IC (1979-01-01); amip-oos '
            '(2014-2024) uses derived AIMIP SST/SIC forcing; not cluster-compared across modes in 1.3.x.'
        ),
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
        "status": "runs",
        "verified_on": None,
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/ACE2-EAMv3",
        "grid": "180 x 360 equirectangular (1 deg, noleap)",
        "outputs": ["TREFHT", "TS", "PRECT", "PS"],
        "cost_note": FME_COST_NOTE,
        "limitations": (
            'Authors\' limits (allenai/ACE2-EAMv3): "trained on output from the E3SMv3\'s atmosphere model '
            '(i.e. EAMv3) forced with historical sea surface temperature (AMIP) from 1970-2020"; "ACE '
            'models are research tools and should not be used for operational climate predictions." AIMIP: '
            'our 1994-2014 window lies inside the 1970-2020 training years; checkpoint takes no CO2 input; '
            'shipped ic_1994.nc is an E3SM 1971-01-01 state relabelled to 1994-01-01 on a 365-day noleap '
            'calendar, with derived ERA5 SST/SIC forcing (1994-2014) — discard the first simulated year as '
            'spin-up; not cluster-compared in 1.3.x.'
        ),
        "examples": [
            "aimip plan ace2-eamv3 --start 1994-01-01 --length 20y --name eamv3-hist",
            "aimip run ace2-eamv3 --start 1994-01-01 --length 1y --name eamv3-1y",
        ],
        "citation_note": PARENT_ACE2_CITATION_NOTE,
        "bibtex": [ACE2_BIBTEX],
    },
    "hiro-ace": {
        "title": "HiRO-ACE (Ai2, experimental 1994-2014 extension)",
        "task": "amip",
        "status": "experimental",
        "verified_on": None,
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/HiRO-ACE",
        "data_urls": ["https://zenodo.org/records/17065758"],
        "grid": "180 x 360 Gaussian (1 deg, julian)",
        "outputs": ["TMP2m", "surface_temperature", "PRATEsfc", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": (
            'Authors\' limits (allenai/HiRO-ACE, arXiv:2512.18224): ACE2S was "trained only on 9 years of '
            'X-SHiELD (2014-2022)", is "not expected to generalize outside of the limited forcing '
            'conditions used during training", and is "not suitable for weather forecasting." AIMIP: '
            '2014-2022 (within amip-oos 2014-2024) is the training period, whereas 1994-2014 (amip-hist) is '
            'AIMIP\'s own out-of-training-period extension using derived SHiELD+ forcing and a 2014-01-01 '
            'state relabelled to 1994-01-01 (discard the first simulated year as spin-up; prefer `ace2s` or '
            '`hiro-acev2` for native 2014-2023 runs).'
        ),
        "examples": [
            "aimip plan hiro-ace --start 1994-01-01 --length 5y --members 3 --name hiro-94",
            "aimip run hiro-ace --start 1994-01-01 --length 1y --members 2 --name hiro-demo",
        ],
        "bibtex": [HIRO_ACE_BIBTEX, ACE2_BIBTEX],
    },
    "ace2-som": {
        "title": "ACE2-SOM (Ai2 Slab Ocean)",
        "task": "slab-ocean",
        "status": "runs",
        "verified_on": None,
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/ACE2-SOM",
        "grid": "180 x 360 Gaussian (1 deg)",
        "outputs": ["TMP2m", "surface_temperature", "PRATEsfc", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": (
            'Authors\' limits (allenai/ACE2-SOM, arXiv:2412.04418): "Sea-ice coverage, ocean heat '
            'transport, and ocean mixed layer depth are prescribed based on a present-day climatologies and '
            'therefore do not respond to changes in CO2"; "Abrupt regime shifts in stratospheric '
            'temperature and moisture can occur in inference runs with changing CO2 concentration"; '
            '"Adjustment to an abrupt change in CO2 concentration occurs too quickly, violating energy '
            'conservation"; "ACE models are research tools and should not be used for operational climate '
            'predictions." AIMIP: nominal years 2032-2035 are equilibrium labels, not calendar dates; '
            'specify --forcing-set 1xCO2 or 3xCO2; not cluster-compared in 1.3.x.'
        ),
        "examples": [
            "aimip plan ace2-som --start 2032-01-01 --end 2035-12-31T18:00:00 --forcing-set 1xCO2 --name som-1x",
            "aimip run ace2-som --start 2032-01-01 --length 3y --forcing-set 3xCO2 --name som-3x",
        ],
        "bibtex": [ACE2_SOM_BIBTEX, ACE2_BIBTEX],
    },
    "ace2s": {
        "title": "ACE2S (Ai2, 2014-2023)",
        "task": "amip",
        "status": "runs",
        "verified_on": None,
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/HiRO-ACE",
        "grid": "180 x 360 Gaussian (1 deg, julian)",
        "outputs": ["TMP2m", "surface_temperature", "PRATEsfc", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": (
            'Authors\' limits (allenai/HiRO-ACE, arXiv:2512.18224): ACE2S was "trained only on 9 years of '
            'X-SHiELD (2014-2022)", is "not expected to generalize outside of the limited forcing '
            'conditions used during training", and is "not suitable for weather forecasting." AIMIP: '
            'upstream native forcing window 2014-2023 covers the training period (plus 2023); not '
            'cluster-compared across modes in 1.3.x.'
        ),
        "examples": [
            "aimip plan ace2s --start 2014-01-01 --length 5y --members 4 --name ace2s-ens",
            "aimip run ace2s --start 2014-01-01 --length 1y --members 2 --name ace2s-1y",
        ],
        "bibtex": [HIRO_ACE_BIBTEX, ACE2_BIBTEX],
    },
    "hiro-acev2": {
        "title": "HiRO-ACEv2 (Ai2, fme 2026.10.0)",
        "task": "amip",
        "status": "certified",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2 members, container == native bit-identical, 1.3.0)",
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/HiRO-ACEv2",
        "grid": "180 x 360 Gaussian (1 deg, julian)",
        "outputs": ["TMP2m", "surface_temperature", "PRATEsfc", "PRESsfc"],
        "cost_note": FME_COST_NOTE,
        "limitations": (
            'Authors\' limits (allenai/HiRO-ACEv2, arXiv:2610.17693): global ACE2S-X-SHiELD stage is '
            '"trained only on 9 years of X-SHiELD (2014-2022)", "not expected to generalize outside of the '
            'limited forcing conditions used during training", and "not suitable for weather forecasting." '
            'AIMIP: upstream 2014-2023 window covers the training period (3 km HiROv2 downscaler stage not '
            'run here; uses fme 2026.10.0).'
        ),
        "examples": [
            "aimip plan hiro-acev2 --start 2014-01-01 --length 2y --members 3 --name hiro2-demo",
            "aimip run hiro-acev2 --start 2014-01-01 --steps 20 --members 2 --name hiro2-short",
        ],
        "citation_note": HIRO_ACEV2_CITATION_NOTE,
        "bibtex": [HIRO_ACEV2_BIBTEX, HIRO_ACE_BIBTEX, ACE2_BIBTEX],
    },
    "samudrace-cm4": {
        "title": "SamudrACE-CM4 piControl (Ai2 Coupled)",
        "task": "coupled",
        "status": "certified",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (6 coupled steps, cudnn.benchmark=False, container == native bit-identical, 1.3.1)",
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/SamudrACE-CM4-piControl",
        "grid": "180 x 360 coupled atmosphere-ocean (5-day coupling step, noleap)",
        "outputs": ["atmosphere/TMP2m", "ocean/sst", "ocean/thetao_0"],
        "cost_note": FME_COST_NOTE,
        "limitations": (
            'Authors\' limits (allenai/SamudrACE-CM4-piControl, doi:10.1029/2025GL119340): "trained on '
            'simulation data from GFDL\'s Climate Model v4 with preindustrial control forcing conditions"; '
            '"ACE models are research tools and should not be used for operational climate predictions." '
            'AIMIP: unforced pre-industrial control (~1 K colder than 1995-2014; no historical GHG/aerosol '
            'trend); 5-day (120h) coupled step on noleap calendar; cuDNN autotuning is disabled for '
            'bit-reproducible coupled rollouts.'
        ),
        "examples": [
            "aimip plan samudrace-cm4 --start 0311-01-01 --length 10y --name cm4-10y",
            "aimip run samudrace-cm4 --start 0311-01-01 --length 30d --name cm4-30d",
        ],
        "bibtex": [SAMUDRACE_BIBTEX],
    },
    "samudrace-e3smv3": {
        "title": "SamudrACE-E3SMv3 (Ai2 Coupled, fme 2026.10.0)",
        "task": "coupled",
        "status": "runs",
        "verified_on": None,
        "licence": "Apache-2.0",
        "upstream_url": "https://huggingface.co/allenai/SamudrACE-E3SMv3",
        "grid": "180 x 360 coupled atmosphere-ocean (5-day coupling step, noleap)",
        "outputs": ["atmosphere/Tat2m", "ocean/temperatureCoarsened_0"],
        "cost_note": FME_COST_NOTE,
        "limitations": (
            'Authors\' limits (allenai/SamudrACE-E3SMv3): emulator of E3SMv3 "with preindustrial control '
            'forcing conditions"; "ACE models are research tools and should not be used for operational '
            'climate predictions." AIMIP: runs in fme 2026.10.0 overlay; upstream ships only 1 year of '
            'forcing over 0425-01-03T12:00:00..0425-12-29T12:00:00 (at most 72 5-day steps; cannot be '
            'continued); not cluster-tested in 1.3.x (`docs/installation.md`).'
        ),
        "examples": [
            "aimip plan samudrace-e3smv3 --start 0425-01-03T12:00:00 --end 0425-12-29T12:00:00 --members 3 --name e3sm-full",
            "aimip run samudrace-e3smv3 --start 0425-01-03T12:00:00 --length 30d --name e3sm-30d",
        ],
        "bibtex": [SAMUDRACE_E3SMV3_BIBTEX, SAMUDRACE_BIBTEX],
    },
    "dlesym-v1.0": {
        "title": "DLESyM v1.0 (NVIDIA / UW, Earth2Studio)",
        "task": "coupled",
        "status": "runs",
        "verified_on": None,
        "licence": "Apache-2.0",
        "upstream_url": "https://github.com/NVIDIA/earth2studio",
        "grid": "HEALPix-64 native -> 0.25 deg lat-lon output",
        "outputs": ["atmosphere/t2m", "ocean/sst"],
        "cost_note": DLESYM_COST_NOTE,
        "limitations": (
            'Authors\' limits (nvidia/dlesym-v1-era5): "checkpoints used in the ECMWF AI Weather Quest S2S '
            'competition", trained on "ERA5 data for the period 1980-2015." AIMIP: built for '
            'subseasonal-to-seasonal forecasts (weeks to months); multi-year rollouts are outside author '
            'validation and stop with exit code 5 at the first non-finite or out-of-band month; `ocean/sst` '
            'has values over land (mask land before use); initial ERA5 state is fetched at prepare time; '
            'not cluster-compared in 1.3.x.'
        ),
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
        "verified_on": "1x NVIDIA A100-SXM4-80GB (2 members, 62 days, container == native bit-identical, 1.3.0)",
        "licence": "Apache-2.0",
        "upstream_url": "https://github.com/NVIDIA/earth2studio",
        "grid": "HEALPix-64 native -> 0.25 deg lat-lon output",
        "outputs": ["atmosphere/t2m", "ocean/sst", "atmosphere/monthly_precip.nc (tp06)"],
        "cost_note": DLESYM_COST_NOTE,
        "limitations": (
            'Authors\' limits (nvidia/dlesym-v1-era5): "checkpoints used in the ECMWF AI Weather Quest S2S '
            'competition" and "trained with CRPS loss and use sampled random noise to produce ensemble '
            'variability." AIMIP: built for subseasonal-to-seasonal forecasts (weeks to months); multi-year '
            'rollouts are outside author validation and stop with exit code 5 at the first unstable month; '
            '`ocean/sst` has values over land (mask land); `tp06` precipitation uses epsilon=1e-8 inferred '
            'from the authors\' pipeline; initial ERA5 state is fetched at prepare time.'
        ),
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
        "verified_on": None,
        "licence": "Apache-2.0",
        "upstream_url": "https://github.com/AtmosSci-DLESM/DLESyM",
        "grid": "HEALPix-64 native -> 0.25 deg lat-lon output",
        "outputs": ["atmosphere/t2m", "ocean/sst"],
        "cost_note": DLESYM_COST_NOTE,
        "limitations": (
            "AIMIP: unforced present-day climate checkpoint (Cresswell-Clay et al., 2025, "
            "doi:10.1029/2025AV001706; insolation seasonal cycle only, no changing GHG/aerosol forcing), "
            "using day-of-year OLR moment matching on the initial ERA5 state; stops with exit code 5 if a "
            "month becomes non-finite or out of band; `ocean/sst` has values over land (mask land before "
            "use); not cluster-compared in 1.3.x."
        ),
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
        "verified_on": "1x NVIDIA A100-SXM4-80GB (24 months, container == native bit-identical, 1.3.0)",
        "licence": "Source fetched on first use (no upstream licence file; see THIRD_PARTY.md)",
        "upstream_url": "https://github.com/grahamclyne/ArchesClimate-SSP",
        "data_urls": ["https://huggingface.co/datasets/gclyne/ArchesClimate-data"],
        "grid": "144 x 144 (1.25 deg lat x 2.5 deg lon)",
        "outputs": ["atmosphere/tas", "atmosphere/psl", "atmosphere/pr", "ocean/thetao_0", "ocean/thetao"],
        "cost_note": ARCHES_COST_NOTE,
        "limitations": (
            "AIMIP: emulates one CMIP6 GCM (IPSL-CM6A-LR r1i1p1f1) under SSP scenario forcing "
            "(--forcing-set ssp434 or ssp585), not observed weather or multi-model spread; always starts at "
            "2015-01-01T00:00:00 (first output 2015-03, max 1029 months). Upstream code has no licence file "
            "and is fetched on first use."
        ),
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
