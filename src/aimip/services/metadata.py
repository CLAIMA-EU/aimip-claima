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

ARCHESWEATHER_BIBTEX = """@article{singh2026archesweather,
  title   = {Evaluating Skill and Stability of {ArchesWeather} and {ArchesWeatherGen} under Multi-Decadal Climate Simulations},
  author  = {Singh, Renu and Brunstein, Robert and Jost, Antonia and Rackow, Thomas and Monteleoni, Claire and Hasson, Yana and Lessig, Christian and Couairon, Guillaume},
  journal = {arXiv preprint},
  year    = {2026},
  eprint  = {2605.29976}
}"""

AIMIP_BIBTEX = """@article{henn2026aimip,
  title   = {{AIMIP} Phase 1: Systematic Evaluations of {AI} Weather and Climate Models},
  author  = {Henn, Brian and Bretherton, Christopher S. and Koldunov, Nikolay and Lessig, Christian and Molina, Maria J. and Arcomano, Troy and Watt-Meyer, Oliver and Couairon, Guillaume and Singh, Renu and Brunstein, Robert and Hasson, Yana and Jost, Antonia and Brenowitz, Noah and Manshausen, Peter and Cresswell-Clay, Nathaniel and Durran, Dale and Hall, Kyle Joseph Chen and Yuval, Janni and Kochkov, Dmitrii and Hoyer, Stephan and Lopez-Gomez, Ignacio},
  journal = {arXiv preprint},
  year    = {2026},
  eprint  = {2605.06944}
}"""

SAMUDRA_BIBTEX = """@article{dheeshjith2025samudra,
  title         = {Samudra: An {AI} Global Ocean Emulator for Climate},
  author        = {Dheeshjith, Surya and Subel, Adam and Adcroft, Alistair and Busecke, Julius and Fernandez-Granda, Carlos and Gupta, Shubham and Zanna, Laure},
  journal       = {Geophysical Research Letters},
  volume        = {52},
  number        = {10},
  year          = {2025},
  doi           = {10.1029/2024GL114318},
  eprint        = {2412.03795},
  archivePrefix = {arXiv},
  url           = {https://doi.org/10.1029/2024GL114318}
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
ARCHESWEATHER_COST_NOTE = (
    "~1.1 s per simulated day for ArchesWeather (four networks), ~4.6 s per simulated day per member for "
    "ArchesWeatherGen (1x A100-80GB); 1978-10 .. 2024-12 is ~5 h and ~21 h per member"
)
SAMUDRA_COST_NOTE = (
    "~18 s per simulated year on 1x A100-80GB (plus ~1.5 GB/yr OSN OM4 window staging before the job)"
)
MD15_BIBTEX = """@article{hall2026md15,
  title         = {Monthly Diffusion v0.9: A Latent Diffusion Model for the First {AI-MIP}},
  author        = {Hall, Kyle J. C. and Molina, Maria J.},
  journal       = {arXiv preprint arXiv:2604.13481},
  eprint        = {2604.13481},
  archivePrefix = {arXiv},
  year          = {2026},
  doi           = {10.48550/arXiv.2604.13481},
  url           = {https://arxiv.org/abs/2604.13481}
}"""
MD15_COST_NOTE = (
    "~20 min for a 46.25-year, 50-member batched rollout on 1x A100-80GB (~15-30 s for 1 year, Hall & Molina, 2026)"
)
CAMULATOR_BIBTEX = """@article{chapman2025camulator,
  title   = {CAMulator: Fast Emulation of the Community Atmosphere Model},
  author  = {Chapman, William E. and Schreck, John S. and Sha, Yingkai and Gagne II, David John and Kimpara, Dhamma and Zanna, Laure and Mayer, Kirsten J. and Berner, Judith},
  journal = {arXiv preprint arXiv:2504.06007},
  year    = {2025},
  doi     = {10.48550/arXiv.2504.06007},
  url     = {https://arxiv.org/abs/2504.06007}
}"""
CAMULATOR_COST_NOTE = (
    "~3 min per simulated year on 1x A100-80GB (~480 simulated years / day, Chapman et al., 2025)"
)
CBOTTLE_BIBTEX = """@article{brenowitz2025cbottle,
  title   = {Climate in a Bottle: Towards a Generative Foundation Model for the Kilometer-Scale Global Atmosphere},
  author  = {Brenowitz, Noah D. and Ge, Tao and Subramaniam, Akshay and Manshausen, Peter and Gupta, Aayush and Hall, David M. and Mardani, Morteza and Vahdat, Arash and Kashinath, Karthik and Pritchard, Michael S.},
  journal = {arXiv preprint arXiv:2505.06474},
  year    = {2025},
  doi     = {10.48550/arXiv.2505.06474},
  url     = {https://arxiv.org/abs/2505.06474}
}"""
CBOTTLE_COST_NOTE = (
    "~2.5 min per sampled year at daily_synoptic cadence (batch_size=4) on 1x A100-80GB"
)
SDYFF_BIBTEX = """@inproceedings{ruhlingcachay2024sphericaldyffusion,
  title     = {Probabilistic Emulation of a Global Climate Model with Spherical {DYffusion}},
  author    = {R{\\"u}hling Cachay, Salva and Henn, Brian and Watt-Meyer, Oliver and Bretherton, Christopher S. and Yu, Rose},
  booktitle = {Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year      = {2024},
  doi       = {10.48550/arXiv.2406.14798},
  eprint    = {2406.14798},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2406.14798}
}"""
SDYFF_COST_NOTE = (
    "~3 min per simulated 365-day cycle year on 1x A100-80GB"
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
    "dlesym-aimip": {
        "title": "DLESyM, AIMIP Phase 1 version (UW / NVIDIA)",
        "task": "amip",
        "status": "runs",
        "verified_on": "1x NVIDIA GH200 (aarch64 image; 1978-10-03 .. 2024-12-28 against the authors' submission)",
        "licence": "Apache-2.0",
        "upstream_url": "https://github.com/nathanielcresswellclay/DLESyM_aimip-2026",
        "grid": "HEALPix-64 native -> 0.25 deg lat-lon output",
        "outputs": ["atmosphere/t2m", "ocean/sst"],
        "cost_note": DLESYM_COST_NOTE,
        "limitations": (
            "AIMIP Phase 1 setup of Cresswell-Clay et al. (2025): the dlesym-v0-isccp checkpoints with the SST "
            "prescribed from the AIMIP forcing (Zenodo 17065758, linear in time between the monthly stamps) instead of "
            "predicted; the authors' five members start from ERA5 at 1978-10-03 .. 07 00 UTC, one run each. The "
            "initial ERA5 OLR is converted by Earth2Studio's day-of-year moment matching (the authors describe an "
            "affine fit) and land SST is filled by the model's own interpolation, so runs match the authors' to "
            "about 0.1 K in 2-m temperature for the first days, then diverge as weather does; over 1979-2024 the "
            "global-mean 2-m temperature (287.32 K) and its trend (0.17 K per decade) match the authors' members "
            "within 0.01 K and 0.005 K per decade. Trained on 1983-2016 "
            "(part of the AIMIP evaluation period). No +2 K / +4 K experiments yet; `ocean/sst` is the prescribed "
            "SST with values over land."
        ),
        "examples": [
            "aimip plan dlesym-aimip --start 1978-10-03 --steps 67552 --name dlesym-aimip-r1",
            "aimip run dlesym-aimip --start 1978-10-03 --length 1y --name dlesym-aimip-1y",
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
    "archesweather": {
        "title": "ArchesWeather-Mx4, AIMIP version (INRIA)",
        "task": "amip",
        "status": "runs",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (container, 62 days via aimip run, also as three continued "
                       "segments, bit-identical; 457 days compared with the authors' AIMIP submission, 1.4.0.dev0)",
        "licence": "Checkpoints and inputs CC-BY-4.0 (Zenodo 22795233); geoarches code BSD-3-Clause",
        "upstream_url": "https://github.com/INRIA/geoarches",
        "data_urls": ["https://zenodo.org/records/22795233"],
        "grid": "1 deg lat-lon (181 x 360), 13 pressure levels, daily-mean states",
        "outputs": ["atmosphere/tas", "atmosphere/psl", "atmosphere/uas", "atmosphere/vas", "atmosphere/sst",
                    "atmosphere/sic", "atmosphere/monthly_levels.nc (zg, ua, va, ta, hus, wap)"],
        "cost_note": ARCHESWEATHER_COST_NOTE,
        "limitations": (
            "AIMIP: the four ArchesWeather-M networks averaged at every daily step, as in the authors' AIMIP runs; "
            "trained on ERA5 1979-2013 (validation 2014, test 2015-2024; most of the AIMIP period is in-sample). "
            "A run starts from the ERA5 daily-mean state of 1978-09-30 (the authors' member r2; their other "
            "members start on 1978-09-29 and 1978-10-01..03) or continues an earlier run from its restart.nc "
            "(--ic <run directory>, same result as one longer run), forced by monthly ERA5 SST and sea-ice cover "
            "through 2024-12-31. Outputs and restart.nc are written when the run ends, so a run must fit in its "
            "job. The rollout follows arXiv:2605.29976 section 2.3; the authors' rollout script is not published, "
            "and aimip's run is not bit-identical to their submission (GPU vs TPU). The longest run "
            "tested in aimip is 457 days (a full 46-year run has not yet been tested)."
        ),
        "examples": [
            "aimip plan archesweather --start 1978-09-30 --length 1y --name aw-1y",
            "aimip run archesweather --start 1978-09-30 --steps 16894 --name aw-aimip",
        ],
        "bibtex": [ARCHESWEATHER_BIBTEX, AIMIP_BIBTEX, ERA5_BIBTEX],
    },
    "archesweathergen": {
        "title": "ArchesWeatherGen, AIMIP version (INRIA)",
        "task": "amip",
        "status": "runs",
        "verified_on": "1x NVIDIA A100-SXM4-80GB (container, 2 members x 31 days via aimip run, also as two "
                       "continued segments, bit-identical; October 1978 spread "
                       "compared with the authors' AIMIP submission, 1.4.0.dev0)",
        "licence": "Checkpoints and inputs CC-BY-4.0 (Zenodo 22795233); geoarches code BSD-3-Clause",
        "upstream_url": "https://github.com/INRIA/geoarches",
        "data_urls": ["https://zenodo.org/records/22795233"],
        "grid": "1 deg lat-lon (181 x 360), 13 pressure levels, daily-mean states",
        "outputs": ["atmosphere/tas", "atmosphere/psl", "atmosphere/uas", "atmosphere/vas", "atmosphere/sst",
                    "atmosphere/sic", "atmosphere/monthly_levels.nc (zg, ua, va, ta, hus, wap)"],
        "cost_note": ARCHESWEATHER_COST_NOTE,
        "limitations": (
            "AIMIP: flow-matching samples (25 sampler steps per day) around the ArchesWeather-Mx4 prediction; "
            "trained on ERA5 1979-2013 (validation 2014, test 2015-2024; most of the AIMIP period is in-sample). "
            "Members differ by the noise seed (step i of member k uses seed + k + 1000 i; keep --seed values of "
            "separate runs less than 1000 apart so they do not reuse daily noise). Starts from the ERA5 daily-mean "
            "state of 1978-09-30 (or continues an earlier run), forced by monthly ERA5 SST and sea-ice cover "
            "through 2024-12-31. The generative checkpoint also stores one deterministic network; like geoarches' "
            "own loader with the published config, aimip uses the four ArchesWeather-M networks instead. About "
            "28 minutes per member-year: long runs go in segments that each fit in one job (outputs and restart.nc "
            "are written when a run ends), each continuing the previous one's restart.nc (--ic <run directory>, "
            "same members and seed); the segments give the same result as one run, and member k of --seed s equals "
            "a one-member run with --seed s+k."
        ),
        "examples": [
            "aimip plan archesweathergen --start 1978-09-30 --length 1y --members 2 --name awgen-1y",
            "aimip run archesweathergen --start 1978-09-30 --length 30d --members 2 --name awgen-30d",
        ],
        "bibtex": [ARCHESWEATHER_BIBTEX, AIMIP_BIBTEX, ERA5_BIBTEX],
    },
    "samudra-om4": {
        "title": "Samudra-OM4 Global Ocean Emulator (M2LInES, thermo-dynamic seed1)",
        "task": "ocean",
        "status": "runs",
        "verified_on": (
            "1x NVIDIA A100-SXM4-80GB (14 steps, 2014-11 thetao_0 = 18.1524 degC, native mode, 1.4.0)"
        ),
        "licence": "Code: Apache-2.0 (m2lines/Samudra@0c2a1bd8); weights & data: MIT",
        "upstream_url": "https://github.com/m2lines/Samudra",
        "data_urls": [
            "https://huggingface.co/M2LInES/Samudra",
            "https://huggingface.co/datasets/M2LInES/Samudra-OM4",
            "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4",
        ],
        "grid": "180 x 360 ocean grid (1 deg), 19 fixed depth levels (2.5 m - 6000 m)",
        "outputs": [
            "ocean/thetao_0 (degC at 2.5 m)",
            "ocean/so_0 (psu at 2.5 m)",
            "ocean/zos (m)",
            "ocean/monthly_thetao.nc (19 depths)",
        ],
        "cost_note": SAMUDRA_COST_NOTE,
        "limitations": (
            'Authors\' limits (Dheeshjith et al., 2025, doi:10.1029/2024GL114318): '
            '"struggles to capture trends" under changed heat-flux forcing (warming trends are '
            '20-50% too weak across most depths in the 8-year test; do not use for warming trends '
            'or ocean heat uptake). AIMIP: Emulates GFDL OM4 (1975-2022 JRA55-do OMIP-2 simulation), '
            'compare with OM4, not with observations; every run replays a window of the OM4 store '
            'driven by OM4\'s own surface fluxes (tauuo, tauvo, hfds, hfds_anomalies; not a forecast '
            'and no custom forcing); starts before 2014-09-30T12:00:00 lie inside training/validation '
            '(1975-01-03..2014-09-20); thetao_0 is 2.5 m potential temperature in degC (not skin SST), '
            'and `aimip summary` computes a cos(latitude)-weighted mean over wet cells. The OSN OM4.zarr '
            'store has 73 timestamps per year (Jan 3 .. Dec 28) with no time bounds or averaging attribute '
            '("5-day mean" is the paper\'s statement); each leap year in 1976..2020 ends with a 6-day '
            'Dec 28 -> Jan 3 step, so Dec 31 of a leap year is covered by no 5-day state and gets weight '
            '1/31 from the Jan 3 state.'
        ),
        "examples": [
            "aimip plan samudra-om4 --start 2014-10-10T12:00:00 --steps 74 --name samudra-1y",
            "aimip run samudra-om4 --start 2014-10-10T12:00:00 --length 70d --name samudra-70d",
        ],
        "bibtex": [SAMUDRA_BIBTEX],
    },
    "md-1.5": {
        "title": "Monthly Diffusion v0.9 at 1.5 deg (UMD, MD-1p5-latest.pth)",
        "task": "amip",
        "status": "runs",
        "verified_on": (
            "1x NVIDIA A100-SXM4-80GB (12 monthly steps x 2 members, "
            "1978-10 TMP2m = 287.1284 K -> 1979-01 284.5882 K -> 1979-07 288.6166 K, native mode, 1.4.0)"
        ),
        "licence": (
            "Code & weights: MIT (kjhall01/monthly-diffusion@b077d3bc, Zenodo 21430292); "
            "AIMIP forcing: CC-BY-4.0 (Zenodo 17065758); ERA5 initial/static fields: CC-BY-4.0 (CDS >= 2025-07-02)"
        ),
        "upstream_url": "https://github.com/kjhall01/monthly-diffusion",
        "data_urls": [
            "https://doi.org/10.5281/zenodo.21430292",
            "https://doi.org/10.5281/zenodo.17065758",
        ],
        "grid": "121 x 240 regular lat-lon (1.5 deg, lat -90..90, lon 0..358.5), 7 pressure levels (1000-50 hPa)",
        "outputs": [
            "TMP2m (K, 2 m air temperature)",
            "surface_temperature (K, skin temperature)",
            "sp (Pa), msl (Pa), d2m (K), u10, v10, mtpr (kg m-2 s-1), z500 (m2 s-2)",
            "t, q, u, v on 1000/850/700/500/250/100/50 hPa (37 prognostic channels total)",
        ],
        "cost_note": MD15_COST_NOTE,
        "limitations": (
            'Authors\' limits (Hall & Molina, 2026, arXiv:2604.13481): initial v0.9 release trained on ~360 '
            'ERA5 monthly samples (1985-01..2014-12; validation 1979-1984, test 2015-2022); spin-up of "at least '
            'a decade may be necessary" because the QBO has a large bias in the first ~5 years (the first years '
            'after start are spin-up, not climate results); meridional temperature gradient is too strong, ENSO '
            'teleconnections and NAO variability are weaker than ERA5, the ITCZ is drier, and a small-scale wave '
            'pattern appears across fields; uniform +2 K SST gave ~0.8 K global response (warming response is not '
            'validated). AIMIP: The first output month (step 0) is the initial state passed through the encoder '
            'and decoder (md/cvae.py:475-481), not a forecast step. Each subsequent step runs T=15 reverse DDPM '
            'steps (code default in md/diffusion.py:106; config.pkl omits T) and outputs a single monthly model '
            'state (counts=1, not an average of sub-monthly steps; no daily values or extremes); AIMIP Phase 1 '
            'forcing (Zenodo 17065758, 556 points 1978-10-01..2025-01-01) is centred on the start of each month '
            'with exact upstream index pairing preserved; upstream publishes no initial state (pass `--ic <path.nc>` '
            'or stage `$AIMIP_DATA/inputs/md-1.5/ic/ic_<YYYY-MM-DD>.nc`; only 1978-10-01 was used by the authors).'
        ),
        "examples": [
            "aimip plan md-1.5 --start 1978-10-01 --length 10y --members 4 --name md15-spin",
            "aimip run md-1.5 --start 1978-10-01 --steps 12 --members 2 --name md15-1y",
        ],
        "bibtex": [MD15_BIBTEX, ERA5_BIBTEX],
    },
    "md-1.5-aimip": {
        "title": "Monthly Diffusion v0.9 at 1.5 deg, AIMIP Phase 1 version (UMD, MD-1p5.pth)",
        "task": "amip",
        "status": "runs",
        "verified_on": (
            "1x NVIDIA GH200 (aarch64 image; 5 members x 555 months, 1978-10 .. 2024-12, in 2 min, against the "
            "authors' submission)"
        ),
        "licence": (
            "Code & weights: MIT (kjhall01/monthly-diffusion@b077d3bc, Zenodo 21430292); "
            "AIMIP forcing: CC-BY-4.0 (Zenodo 17065758); rebuilt ERA5 initial state and statistics: CC-BY-4.0"
        ),
        "upstream_url": "https://github.com/kjhall01/monthly-diffusion",
        "data_urls": [
            "https://doi.org/10.5281/zenodo.21430292",
            "https://doi.org/10.5281/zenodo.17065758",
        ],
        "grid": "121 x 240 regular lat-lon (1.5 deg, lat -90..90, lon 0..358.5), 7 pressure levels (1000-50 hPa)",
        "outputs": [
            "TMP2m (K, 2 m air temperature)",
            "surface_temperature (K, skin temperature)",
            "sp (Pa), msl (Pa), d2m (K), u10, v10, mtpr (kg m-2 s-1), z500 (m2 s-2)",
            "t, q, u, v on 1000/850/700/500/250/100/50 hPa (37 prognostic channels total)",
        ],
        "cost_note": MD15_COST_NOTE,
        "limitations": (
            "A reconstruction of the authors' AIMIP Phase 1 run (Hall & Molina, 2026, arXiv:2604.13481), not a copy: "
            "`MD-1p5.pth` was saved by earlier, unpublished code, so the published classes are built with the settings "
            "its tensors imply (every tensor loads); the 1985-2014 statistics and the ERA5 state of October 1978 are "
            "rebuilt from ERA5 (DKRZ pool) with the authors' data pipeline. The job follows the authors' code path "
            "exactly (the decoded first month matches their `forced_run` on the same inputs to 0.001 K), but the "
            "decoded October 1978 is 0.43 K warmer in global-mean 2-m temperature than the authors' (287.44 vs "
            "287.01 K). Five members over 1979-2024: global-mean 2-m temperature 287.28-287.31 K (authors: "
            "287.32-287.33 K), trend 0.067-0.072 K per decade (0.058-0.069), seasonal cycle about 0.6 K larger "
            "(January-April up to 0.45 K colder, July-September up to 0.4 K warmer). Authors' limits: trained on "
            "1985-2014 (the AIMIP evaluation period overlaps training); QBO spin-up bias in the first ~5 years; "
            "meridional temperature gradient too strong, weak ENSO teleconnections and NAO variability. Step 0 is "
            "the encoder-decoder reconstruction of the initial state; each later step is one monthly state "
            "(counts = 1). Members are batched noise draws (as the authors' `forced_run`); CUDA and CPU draw "
            "different noise for the same seed. No +2 K / +4 K experiments yet."
        ),
        "examples": [
            "aimip plan md-1.5-aimip --start 1978-10-01 --steps 555 --members 5 --name md15-aimip",
            "aimip run md-1.5-aimip --start 1978-10-01 --steps 12 --members 2 --name md15-aimip-1y",
        ],
        "bibtex": [MD15_BIBTEX, ERA5_BIBTEX],
    },
    "camulator": {
        "title": "CAMulator Atmosphere Emulator of CAM6 / ERA5-scaled (NCAR MILES/CREDIT, ckpt 69)",
        "task": "amip",
        "status": "runs",
        "verified_on": (
            "1x NVIDIA A100-SXM4-80GB (236 steps, "
            "1981-01 TREFHT = 285.8759 K, 1981-02 TREFHT = 286.0487 K, native mode, 1.4.0)"
        ),
        "licence": "Apache-2.0 (WillyChap/miles-credit@d0eb9add & willychap/camulator@01afc272)",
        "upstream_url": "https://huggingface.co/willychap/camulator",
        "data_urls": [
            "https://github.com/WillyChap/miles-credit/tree/d0eb9add439f09bd25883f4393296f3266b5876d/climate",
            "https://huggingface.co/willychap/camulator",
        ],
        "grid": "192 x 288 CESM f09 regular lat-lon (1 deg), 32 hybrid sigma-pressure levels",
        "outputs": [
            "TREFHT (K, 2 m air temperature)",
            "PRECT (m per 6hr step; mm/day = value * 4000)",
            "TS (K)",
            "PS (Pa)",
            "U10 (m s**-1)",
            "monthly_levels.nc (32-level U, V, T, Qtot)",
        ],
        "cost_note": CAMULATOR_COST_NOTE,
        "citation_note": (
            "arXiv:2504.06007 describes a 2-forcing (SOLIN, SST) CAM6 AMIP model (1979-2010); "
            "the published checkpoint.pt00069.pt takes 4 forcings (SOLIN, SST, ICEFRAC, co2vmr_3d) "
            "trained on CAM6 / ERA5-scaled (1980-2014). Cite both the preprint and willychap/camulator."
        ),
        "limitations": (
            'Authors\' limits (model card willychap/camulator@01afc272): '
            '"CAMulator is a research tool. It emulates a specific CAM6 configuration and is not a '
            'substitute for an operational forecast or a full Earth-system model." For the global energy '
            'fixer, "running without it is a train/inference mismatch" and "it makes global-mean '
            'precipitation ~2% wetter"; on checkpoints, "Later is not better: the latest checkpoints '
            'are measurably colder". '
            'AIMIP: Emulates CAM6 / ERA5-scaled climate target (1980-2014); compare with that target, '
            'not directly with observations. Published checkpoint.pt00069.pt (4 dynamic forcings, 1980-2014) '
            'is not the 2-forcing 1979-2010 model in arXiv:2504.06007 (which reports muted ENSO/PNA/NAO '
            'responses, ~2x dry days vs CAM6, and high-latitude winter cold bias under +2 K / +4 K); '
            'heavy-rain tails are underestimated (PRECT p99.99 global-mean bias -10.53 mm/day on ckpt 69). '
            'Upstream b.e21 forcing files do not state whether SST comes from observations or a coupled run, '
            'so they are named camulator-cyclic-1yr and camulator-branch-1980-2014 (never amip-hist); '
            'starts are restricted to the 69 published noleap initial states.'
        ),
        "examples": [
            "aimip plan camulator --start 1981-01-01T00:00:00 --length 1y --name cam-1y",
            "aimip run camulator --start 1980-01-01T00:00:00 --steps 1460 --forcing-set camulator-branch-1980-2014 --name cam-1980",
        ],
        "bibtex": [CAMULATOR_BIBTEX],
    },
    "cbottle-3d": {
        "title": "cBottle-3d Conditional Atmospheric State Generator (NVIDIA / Earth2Studio)",
        "task": "generator",
        "status": "runs",
        "verified_on": (
            "1x NVIDIA A100-SXM4-80GB (31 daily_synoptic steps x 2 members, "
            "2020-01 t2m = 285.7072 K, native mode, 1.4.0)"
        ),
        "licence": (
            "Code & weights: Apache-2.0 (NVlabs/cBottle@f0f51998, NVIDIA/earth2studio@8fa7acb, "
            "nvidia/cbottle@e60ce82b); SST conditioning: CC-BY-4.0 (PCMDI-AMIP-1-1-9)"
        ),
        "upstream_url": "https://github.com/NVlabs/cBottle",
        "data_urls": [
            "https://huggingface.co/nvidia/cbottle/tree/e60ce82b236b953d79d3ead7f43445a3bb8e37d3",
        ],
        "grid": "HEALPix-64 native -> 0.25 deg regular lat-lon output (721 x 1440)",
        "outputs": [
            "t2m (K)",
            "msl (Pa)",
            "tcwv (kg m**-2)",
            "u10m (m s**-1)",
            "v10m (m s**-1)",
            "z500 (m**2 s**-2)",
            "t850 (K)",
            "u850 (m s**-1)",
            "v850 (m s**-1)",
            "tpf (kg m**-2 s**-1)",
        ],
        "cost_note": CBOTTLE_COST_NOTE,
        "limitations": (
            'Authors\' limits (model card nvidia/cbottle@e60ce82b): cannot "resolve meso-scale motions" '
            'at this resolution, and for ICON "no independent data withheld for evaluation". '
            "AIMIP: Conditional state generator, not a time-stepping simulation: states on adjacent days "
            "are independent draws conditioned only on (day_of_year, hour_of_day) and monthly mid-point "
            "SST (tosbcs, no CO2 or sea-ice input; no run is a hindcast). Valid for climatological means "
            "and distributions, NOT for weather sequences, persistence, or year-to-year variance of "
            "monthly means compared with autoregressive models. Dates from 2018 onward lie outside the "
            "1980-2017 ERA5 training years (valid SST window 1940-01-01 .. 2022-12-16); each 4-date "
            "batch (batch_size=4, chronological date_order) derives a distinct diffusion seed via "
            'SHA-256("aimip:cbottle-3d:v1:{seed}:{member_index}:{batch_index}") mod (2^31 - 1), '
            "recorded in plan.json, aimip-manifest.json, and emu-result.json."
        ),
        "examples": [
            "aimip plan cbottle-3d --start 1995-01-01 --length 2m --name cbottle-2m",
            "aimip run cbottle-3d --start 1995-01-01 --length 1y --members 2 --name cbottle-1y",
        ],
        "bibtex": [CBOTTLE_BIBTEX],
    },
    "spherical-dyffusion": {
        "title": "Spherical DYffusion Probabilistic Emulator of 1 deg FV3GFS (Rose-STL-Lab)",
        "task": "amip",
        "status": "runs",
        "verified_on": (
            "1x NVIDIA A100-SXM4-80GB (240 steps x 2 members, "
            "0001-01 air_temperature_7 = 283.0611 K, 0001-02 air_temperature_7 = 283.2453 K, native mode, 1.4.0)"
        ),
        "licence": (
            "Code & weights: Apache-2.0 (Rose-STL-Lab/spherical-dyffusion@a10b3e7f, "
            "ai2cm/modulus@94f62e1c, salv47/spherical-dyffusion@56fcb5d9); "
            "reference data: CC-BY-4.0 (Zenodo 10791087)"
        ),
        "upstream_url": "https://github.com/Rose-STL-Lab/spherical-dyffusion",
        "data_urls": [
            "https://huggingface.co/salv47/spherical-dyffusion",
            "https://doi.org/10.5281/zenodo.10791087",
        ],
        "grid": "180 x 360 regular lat-lon (1 deg), 8 atmospheric layers",
        "outputs": [
            "air_temperature_7 (K, predicted lowest-layer air temperature summarized by aimip)",
            "PRESsfc (Pa)",
            "total_water_path (kg m**-2)",
            "air_temperature_0..7, specific_total_water_0..7, eastward_wind_0..7, northward_wind_0..7",
            "surface_temperature_prescribed_ocean (K, overwritten over ocean by prescriber)",
        ],
        "cost_note": SDYFF_COST_NOTE,
        "limitations": (
            "AIMIP: Emulates a 1 deg FV3GFS atmosphere reference simulation (Ruhling Cachay et al., "
            "NeurIPS 2024; ACE dataset arXiv:2310.02074) with a fixed, repeating 365-day annual SST "
            "cycle (labelled 2021 in Zenodo 10791087; compare with FV3GFS, not observations — no "
            "year-to-year forcing, no historical or scenario years; output months use nominal cycle "
            "indices 0001-01..). Over ocean (ocean_fraction == 1) surface_temperature is overwritten "
            "with the prescribed SST input, so it is stored as surface_temperature_prescribed_ocean and "
            "`aimip summary` reports predicted lowest-model-layer air_temperature_7 (K). Uses exact "
            "daily coarsening (coarsen_factor=4) over complete 365-day calendar months only; intended "
            "for time-mean climate and ensemble spread against the FV3GFS reference. In venv-sdyff, "
            "apex FusedLayerNorm is replaced by torch.nn.LayerNorm (numerically equivalent up to float "
            "rounding) to satisfy modulus.models.sfno.sfnonet's import; sfnonet already instantiates "
            "torch.nn.LayerNorm for layer_norm, and the published checkpoints use "
            "normalization_layer='instance_norm' (torch.nn.InstanceNorm2d)."
        ),
        "examples": [
            "aimip plan spherical-dyffusion --start 2021-01-01 --steps 240 --name sdyff-2m",
            "aimip run spherical-dyffusion --start 2021-01-01 --length 1y --members 2 --name sdyff-1y",
        ],
        "bibtex": [SDYFF_BIBTEX],
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
