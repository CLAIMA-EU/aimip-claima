# AI climate models not yet in aimip-claima

Weights status checked in October 2026; worth re-checking before starting on one.

## Weights public, ready to add

| model | group | what it is | weights | notes | paper |
|---|---|---|---|---|---|
| NeuralGCM | Google Research | hybrid atmosphere (differentiable dynamics + learned physics), 2.8° / 1.4° / 0.7°, deterministic and stochastic; precipitation-trained 2.8° version stable for 20+ years | `gs://neuralgcm/models/` (6 checkpoints, CC-BY-SA-4.0) | JAX, needs its own environment in the image. The AIMIP-specific surface heads and the 1° HRD checkpoints are not public yet | [arXiv:2311.07222](https://arxiv.org/abs/2311.07222), [arXiv:2412.11973](https://arxiv.org/abs/2412.11973) |
| LUCIE-3D | Penn State, Argonne | lightweight SFNO atmosphere emulator on ERA5 T30, 8 levels; 40-year runs | Zenodo [17032361](https://zenodo.org/records/17032361) (3 checkpoints + T30 test data, CC-BY-4.0) | check that the normalisation files and land-sea mask are inside the test-data tarball | [arXiv:2509.02061](https://arxiv.org/abs/2509.02061) |
| Samudra 2 | NYU / M2LInES | ocean emulator, multiple resolutions, full depth | HF [M2LInES/Samudra2](https://huggingface.co/M2LInES/Samudra2) (CC-BY-4.0) | successor of Samudra-OM4, which is already in aimip-claima | [arXiv:2606.02610](https://arxiv.org/abs/2606.02610) |
| ORCA-DL | Shanghai AI Lab | ocean emulator, 1°, 16 depths, monthly step, seasonal to decadal | HF [JayKuo/ORCA-DL-data](https://huggingface.co/datasets/JayKuo/ORCA-DL-data) (8 seeds + normalisation stats) | `predict.py` public; ~12 GB GPU memory | [arXiv:2405.15412](https://arxiv.org/abs/2405.15412) |
| FloeNet | NOAA GFDL / Princeton, Ai2 | mass-conserving sea-ice emulator (GNN), 6-hourly | HF [M2LInES/FloeNet-OM4](https://huggingface.co/M2LInES/FloeNet-OM4) (checkpoint + 3 initial states, CC-BY-4.0) | its 6-hourly CM4/OM4 forcing stores are not public, so it can only load, not run, until they are | [arXiv:2603.12449](https://arxiv.org/abs/2603.12449) |

## Hybrid ML + physics (weights public, but the physical host model must be built)

| model | group | what it is | weights / code | notes |
|---|---|---|---|---|
| ACE2-NEMO | University of Oxford | ACE2-ERA5 atmosphere coupled to the NEMO ocean | `allenai/ACE2-ERA5` + Zenodo 19187005 + GitHub `bobbyantonio/ace2_nemo_coupler` | NEMO v3.6 + LIM3 + OASIS3-MCT build needs the EC-Earth 4 source (registration) |
| ClimSim-trained hybrid E3SM-MMF | LEAP (Columbia), NVIDIA, UC Irvine | NN convection inside E3SM-MMF | Zenodo 12797810 / 14704956 | Fortran E3SM-MMF build with pytorch-fortran |
| MOM6/SIS2 ML parameterisations | NYU / M2LInES, NOAA GFDL | ML closures in the GFDL OM4 ocean / sea-ice model | GitHub `m2lines/SIS2`, `m2lines/ANN-momentum-mesoscale`, HF `M2LInES/gz21-ocean-momentum` | compile OM4 (MOM6 + SIS2) with the weight readers; JRA55-do forcing |

## No public weights yet (re-check from time to time)

| model | group | status |
|---|---|---|
| ACE2.1-ERA5 (AIMIP version) | Ai2 | only AIMIP configs on Zenodo 19831256; the base ACE2-ERA5 is already in aimip-claima |
| M-ACE2S (Masked-ACE2S) | Ai2 | configs only (`ai2cm/m-ace2s-paper`) |
| DLESyM-Ocean | University of Washington | paper only (arXiv:2608.11545) |
| ArchesClimate (decadal hindcasts) | INRIA, LOCEAN-IPSL | no checkpoints; ArchesClimate-SSP is already in aimip-claima |
| Field-Space Autoencoder | DKRZ and collaborators | paper only (arXiv:2601.15102) |
| HClimRep-Ocean | AWI, Forschungszentrum Jülich | no public checkpoints yet |

## How to add one

Each model in aimip-claima is a registry entry (`src/aimip/registry.default.json`), its files pinned with SHA-256 in `src/aimip/aimip.lock`, and offline tests; the steps are in [development.md](development.md#adding-a-model), and [models.md](models.md) shows how the existing models are described. Open a PR: GitHub runs the CPU tests, and the maintainers run a GPU check on their cluster before merging.
