# Third-Party Software and Licences

This file lists the third-party software installed by the `aimip` container image (`container/Dockerfile`) and native installer (`install/install-native.sh`), together with components fetched at runtime on first use.

## Installed in the Container Image and Native Environments

Every Python package with its version and declared licence, generated from the built image: [THIRD_PARTY-PACKAGES.md](THIRD_PARTY-PACKAGES.md).

| Component | Source | Licence |
| --- | --- | --- |
| Base OS and Python 3.12 (`python:3.12.11-bookworm`) | Docker Official Images / Debian / Python Software Foundation | PSF-2.0 and Debian component licences |
| `fme` (`2026.5.1` and `2026.10.0` overlay) | Allen Institute for AI (`ai2cm/ace`) | Apache-2.0 |
| `earth2studio` (`8fa7acb28b082e83a0718e8d2ed137fd375a11fe`, optional emulator env) | NVIDIA (`NVIDIA/earth2studio`) | Apache-2.0 |
| `nvidia-physicsnemo` (`1dacf8d691bb0ab987e8832debacab2dbe0043bb`, optional emulator env) | NVIDIA (`NVIDIA/physicsnemo`) | Apache-2.0 |
| `earth2grid` (`11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3`, incl. `healpixpad_cuda` kernel, optional emulator env) | NVIDIA (`NVlabs/earth2grid`) | Apache-2.0 |
| `torch` (`2.14.0`) and `torchvision` (`0.29.0`) | PyTorch Contributors | BSD-3-Clause |
| NVIDIA CUDA Python wheels (`nvidia-*`: runtime, cuBLAS, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NCCL, NVJITLink and others pulled by `torch`) | NVIDIA (`download.pytorch.org/whl/cu130` / PyPI) | NVIDIA proprietary licences (each wheel ships its licence; installed as PyTorch's CUDA runtime, as in PyTorch's own images) |
| Scientific, agent, and MCP Python packages (`numpy`, `scipy`, `xarray`, `netCDF4`, `h5netcdf`, `cftime`, `pandas`, `pyyaml`, `huggingface_hub`, `torch-harmonics`, `zarrs`, `langgraph`, `langchain-core`, `langchain-google-genai`, `google-genai`, `mcp`, etc.) | PyPI (`container/constraints.txt`, `container/constraints-emu.txt`, `container/requirements-agent.txt`) | Open-source licences, mostly BSD, MIT (`mcp` is MIT), Apache-2.0 and PSF; a few are MPL-2.0 (e.g. `certifi`, `tqdm`). Exact versions are recorded in `/opt/aimip/installed-inventory*.txt` |

Note: NVIDIA compiler wheels (`nvidia-cuda-nvcc`, `nvidia-cuda-crt`, `nvidia-cuda-cccl`, `nvidia-nvvm`) are used in a temporary build directory only while compiling `earth2grid` and are removed before the image layer completes.

## Downloaded on First Use (Not Redistributed)

The container image and repository do **not** redistribute model checkpoints, forcing datasets, initial conditions, or unlicenced third-party model repositories:

- **`ArchesClimate-SSP`** (`https://github.com/grahamclyne/ArchesClimate-SSP`, commit `06ce82dca2414db4f1b3d6940b61c5cd723f4620`): upstream repository carries no licence file (all rights reserved by its authors). Its source archive is **not** baked into the container image or repository; instead, `aimip setup` / `aimip fetch -m archesclimate-ssp` (or `aimip run -m archesclimate-ssp` on first use) downloads the pinned source archive directly from GitHub into `$AIMIP_DATA/code/ArchesClimate-SSP` and verifies its SHA-256 / extracted tree digest before execution.
- **Model checkpoints, initial conditions, and forcing datasets**: downloaded directly by the user from Hugging Face Hub, Zenodo, and Google Cloud (`ARCO-ERA5`) into `$AIMIP_DATA` and verified against `src/aimip/aimip.lock` under each upstream dataset or model author's terms.
