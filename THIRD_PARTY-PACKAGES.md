# Python packages in the aimip 1.3.0 image and their declared licences

Generated from the built image with `container/list_licences.py` (package metadata: `License-Expression`, else the
licence classifiers, else the `License` field). The native installer installs the same pinned versions. Declared
metadata can be incomplete; `UNKNOWN` means the package declares none (`google-crc32c` is Apache-2.0 upstream;
`cuda-toolkit` is an NVIDIA meta-package). The `nvidia-*` wheels and most `cuda-*` wheels are NVIDIA proprietary software (`cuda-pathfinder` and `cuda-bindings` declare Apache-2.0),
installed as PyTorch's CUDA runtime under NVIDIA's licence terms shipped in each wheel.

The table lists declared package licences only. Some wheels bundle native libraries under their own licences:
`imageio-ffmpeg` ships an FFmpeg binary (GPL/LGPL), `soundfile` ships libsndfile (LGPL-2.1), `shapely` ships GEOS
(LGPL-2.1); each wheel carries its own licence files.

## Main environment `/opt/aimip/venv` (fme, aimip, agent)

| package | version | declared licence |
|---|---|---|
| aimip | 1.3.0 | MIT |
| aiobotocore | 3.9.1 | Apache-2.0 |
| aiohappyeyeballs | 2.7.1 | Python Software Foundation License |
| aiohttp | 3.14.3 | Apache-2.0 AND MIT |
| aioitertools | 0.13.0 | MIT |
| aiosignal | 1.4.0 | Apache Software License |
| annotated-types | 0.8.0 | MIT |
| antlr4-python3-runtime | 4.9.3 | BSD |
| anyio | 4.15.1 | MIT |
| attrs | 26.1.0 | MIT |
| bokeh | 3.10.0 | BSD-3-Clause |
| botocore | 1.43.75 | Apache-2.0 |
| Cartopy | 0.25.0 | BSD-3-Clause |
| certifi | 2026.7.22 | Mozilla Public License 2.0 (MPL 2.0) |
| cffi | 2.1.1 | MIT-0 |
| cftime | 1.6.5 | MIT |
| charset-normalizer | 3.5.1 | MIT |
| click | 8.5.0 | BSD-3-Clause |
| contourpy | 1.3.3 | BSD License |
| cryptography | 50.0.1 | Apache-2.0 OR BSD-3-Clause |
| cuda-bindings | 13.3.1 | LicenseRef-NVIDIA-SOFTWARE-LICENSE |
| cuda-pathfinder | 1.8.1 | Apache-2.0 |
| cuda-toolkit | 13.0.3.0 | UNKNOWN |
| cycler | 0.12.1 | BSD License |
| dacite | 1.9.2 | MIT License |
| decorator | 5.3.1 | BSD-2-Clause |
| distro | 1.9.0 | Apache Software License |
| donfig | 0.8.1.post1 | MIT License |
| einops | 0.8.2 | MIT License |
| filelock | 3.32.3 | MIT |
| filetype | 1.2.0 | MIT License |
| fme | 2026.5.1 | Apache Software License |
| fonttools | 4.64.0 | MIT |
| frozenlist | 1.8.0 | Apache-2.0 |
| fsspec | 2026.7.0 | BSD-3-Clause |
| gcsfs | 2026.8.0 | BSD-3-Clause |
| google-api-core | 2.36.0 | Apache Software License |
| google-auth | 2.57.1 | Apache Software License |
| google-auth-oauthlib | 1.4.1 | Apache Software License |
| google-cloud-core | 2.7.0 | Apache Software License |
| google-cloud-storage | 3.13.1 | Apache Software License |
| google-cloud-storage-control | 1.15.0 | Apache Software License |
| google-crc32c | 1.8.0 | UNKNOWN |
| google-genai | 2.23.0 | Apache-2.0 |
| google-resumable-media | 2.10.2 | Apache Software License |
| googleapis-common-protos | 1.75.3 | Apache-2.0 |
| grpc-google-iam-v1 | 0.14.5 | Apache Software License |
| grpcio | 1.83.1 | Apache-2.0 |
| grpcio-status | 1.83.1 | Apache-2.0 |
| h11 | 0.16.0 | MIT License |
| h5netcdf | 1.8.1 | BSD License |
| h5py | 3.16.0 | BSD-3-Clause |
| hf-xet | 1.6.0 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpcore2 | 2.16.0 | BSD-3-Clause |
| httpx | 0.28.1 | BSD License |
| httpx2 | 2.16.0 | BSD-3-Clause |
| huggingface_hub | 1.30.0 | Apache Software License |
| idna | 3.19 | BSD-3-Clause |
| ImageIO | 2.37.4 | BSD-2-Clause |
| imageio-ffmpeg | 0.6.0 | BSD License |
| iniconfig | 2.3.1 | MIT |
| Jinja2 | 3.1.6 | BSD License |
| jmespath | 1.1.0 | MIT License |
| jsonpatch | 1.47 | BSD-3-Clause |
| jsonpointer | 3.12 | BSD-3-Clause |
| kiwisolver | 1.5.1 | BSD License |
| langchain-core | 1.6.3 | MIT License |
| langchain-google-genai | 4.4.0 | MIT |
| langchain-protocol | 0.0.21 | MIT License |
| langgraph | 1.2.11 | MIT |
| langgraph-checkpoint | 4.3.2 | MIT |
| langgraph-prebuilt | 1.1.2 | MIT |
| langgraph-sdk | 0.4.14 | MIT |
| langsmith | 0.14.17 | MIT License |
| MarkupSafe | 3.0.3 | BSD-3-Clause |
| matplotlib | 3.11.1 | Python Software Foundation License |
| moviepy | 2.2.1 | MIT License |
| mpmath | 1.3.0 | BSD License |
| multidict | 6.7.1 | Apache License 2.0 |
| narwhals | 2.25.0 | MIT |
| netCDF4 | 1.7.2 | MIT License |
| networkx | 3.6.1 | BSD-3-Clause |
| numcodecs | 0.16.5 | MIT |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| nvidia-cublas | 13.1.1.3 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cuda-cupti | 13.0.85 | Other/Proprietary License |
| nvidia-cuda-nvrtc | 13.0.88 | Other/Proprietary License |
| nvidia-cuda-runtime | 13.0.96 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cudnn-cu13 | 9.24.0.43 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cufft | 12.0.0.61 | Other/Proprietary License |
| nvidia-cufile | 1.15.1.6 | Other/Proprietary License |
| nvidia-curand | 10.4.0.35 | Other/Proprietary License |
| nvidia-cusolver | 12.0.4.66 | Other/Proprietary License |
| nvidia-cusparse | 12.6.3.3 | Other/Proprietary License |
| nvidia-cusparselt-cu13 | 0.8.1 | NVIDIA Proprietary Software |
| nvidia-nccl-cu13 | 2.30.7 | LicenseRef-NVIDIA-Proprietary |
| nvidia-nvjitlink | 13.4.92 | LicenseRef-NVIDIA-Proprietary |
| nvidia-nvshmem-cu13 | 3.4.5 | LicenseRef-NVIDIA-Proprietary |
| nvidia-nvtx | 13.0.85 | Other/Proprietary License |
| oauthlib | 3.3.1 | BSD-3-Clause |
| omegaconf | 2.3.1 | BSD License |
| opentelemetry-api | 1.44.0 | Apache-2.0 |
| orjson | 3.12.0 | MPL-2.0 AND (Apache-2.0 OR MIT) |
| ormsgpack | 1.13.1 | Apache-2.0 OR MIT |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pandas | 3.0.5 | BSD License |
| pillow | 11.3.0 | MIT-CMU |
| pip | 25.2 | MIT |
| platformdirs | 4.11.7 | MIT |
| plotly | 7.0.0 | MIT |
| pluggy | 1.7.0 | MIT License |
| proglog | 0.1.12 | MIT |
| propcache | 0.5.2 | Apache Software License |
| proto-plus | 1.28.4 | Apache Software License |
| protobuf | 7.36.1 | 3-Clause BSD License |
| pyasn1 | 0.6.4 | BSD-2-Clause |
| pyasn1_modules | 0.4.2 | BSD License |
| pycparser | 3.0 | BSD-3-Clause |
| pydantic | 2.13.5 | MIT |
| pydantic_core | 2.46.5 | MIT |
| Pygments | 2.21.0 | BSD-2-Clause |
| pyparsing | 3.3.2 | MIT |
| pyproj | 3.8.0 | MIT |
| pyshp | 3.1.6 | MIT |
| pytest | 9.2.0 | MIT |
| python-dateutil | 2.9.0.post0 | BSD License; Apache Software License |
| python-dotenv | 1.2.3 | BSD-3-Clause |
| PyYAML | 6.0.3 | MIT License |
| rdkit | 2026.3.6 | BSD-3-Clause |
| requests | 2.34.2 | Apache Software License |
| requests-oauthlib | 2.0.0 | BSD License |
| requests-toolbelt | 1.0.0 | Apache Software License |
| s3fs | 2026.7.0 | BSD License |
| scipy | 1.18.1 | BSD License |
| sentry-sdk | 2.68.1 | MIT |
| setuptools | 84.0.0 | MIT |
| shapely | 2.1.2 | BSD License |
| six | 1.17.0 | MIT License |
| sniffio | 1.3.1 | MIT License; Apache Software License |
| soundfile | 0.14.0 | BSD License |
| sympy | 1.14.0 | BSD License |
| tenacity | 9.1.4 | Apache Software License |
| tensorly | 0.9.0 | BSD License |
| tensorly-torch | 0.5.0 | BSD License |
| torch | 2.14.0+cu130 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| torch_harmonics | 0.8.0 | BSD License |
| tornado | 6.5.8 | Apache Software License |
| tqdm | 4.70.0 | MPL-2.0 AND MIT |
| triton | 3.8.0 | MIT |
| truststore | 0.10.5 | MIT |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| urllib3 | 2.7.0 | MIT |
| uuid_utils | 0.19.0 | BSD-3-Clause |
| wandb | 0.29.0 | MIT License |
| websockets | 16.1.1 | BSD-3-Clause |
| wrapt | 2.4.0 | BSD-2-Clause |
| xarray | 2026.2.0 | Apache-2.0 |
| xxhash | 4.0.1 | BSD-2-Clause |
| xyzservices | 2026.9.1 | BSD-3-Clause |
| yarl | 1.24.5 | Apache-2.0 |
| zarr | 3.3.0 | MIT |
| zstandard | 0.25.0 | BSD-3-Clause |

## fme 2026.10.0 overlay `/opt/aimip/overlay-fme-2026.10.0`

| package | version | declared licence |
|---|---|---|
| fme | 2026.10.0 | Apache Software License |
| torch_harmonics | 0.8.0 | BSD License |
| zarrs | 0.2.3 | MIT |

## Emulator environment `/opt/aimip/venv-emu` (DLESyM, ArchesClimate-SSP)

| package | version | declared licence |
|---|---|---|
| aimip | 1.3.0 | MIT |
| aiobotocore | 3.10.0 | Apache-2.0 |
| aiohappyeyeballs | 2.7.1 | Python Software Foundation License |
| aiohttp | 3.14.5 | Apache-2.0 AND MIT |
| aioitertools | 0.13.0 | MIT |
| aiosignal | 1.4.0 | Apache Software License |
| antlr4-python3-runtime | 4.9.3 | BSD |
| anyio | 4.16.0 | MIT |
| attrs | 26.1.0 | MIT |
| bokeh | 3.10.2 | BSD-3-Clause |
| botocore | 1.43.117 | Apache-2.0 |
| certifi | 2026.10.23 | Mozilla Public License 2.0 (MPL 2.0) |
| cffi | 2.1.1 | MIT-0 |
| cftime | 1.6.5 | MIT |
| charset-normalizer | 3.5.2 | MIT |
| click | 8.5.1 | BSD-3-Clause |
| cloudpickle | 3.1.2 | BSD License |
| contourpy | 1.4.0 | BSD-3-Clause |
| cryptography | 50.0.2 | Apache-2.0 OR BSD-3-Clause |
| cuda-bindings | 13.4.4 | Apache-2.0 |
| cuda-pathfinder | 1.8.4 | Apache-2.0 |
| cuda-toolkit | 13.0.3.0 | UNKNOWN |
| cycler | 0.12.1 | BSD License |
| dask | 2026.8.0 | BSD-3-Clause |
| decorator | 5.3.1 | BSD-2-Clause |
| diffusers | 0.43.0 | Apache Software License |
| distributed | 2026.8.0 | BSD-3-Clause |
| donfig | 0.8.1.post1 | MIT License |
| earth2grid | 2025.11.1+torch214 | Apache-2.0 |
| earth2studio | 0.19.0a0 | Apache-2.0 |
| einops | 0.8.2 | MIT License |
| fastcore | 2.2.41 | Apache-2.0 |
| filelock | 4.2.4 | MIT |
| fonttools | 4.68.0 | MIT |
| frozenlist | 1.8.0 | Apache-2.0 |
| fsspec | 2026.7.0 | BSD-3-Clause |
| gcsfs | 2026.8.0 | BSD-3-Clause |
| gitdb | 4.0.12 | BSD License |
| GitPython | 3.2.1 | BSD-3-Clause |
| google-api-core | 2.44.0 | Apache Software License |
| google-auth | 2.62.2 | Apache Software License |
| google-auth-oauthlib | 1.5.0 | Apache Software License |
| google-cloud-core | 2.8.0 | Apache Software License |
| google-cloud-storage | 3.17.1 | Apache Software License |
| google-cloud-storage-control | 1.17.0 | Apache Software License |
| google-crc32c | 1.9.0 | Apache-2.0 |
| google-resumable-media | 2.11.0 | Apache Software License |
| googleapis-common-protos | 1.76.0 | Apache-2.0 |
| grpc-google-iam-v1 | 0.14.6 | Apache Software License |
| grpcio | 1.85.1 | Apache-2.0 |
| grpcio-status | 1.85.1 | Apache-2.0 |
| h11 | 0.16.0 | MIT License |
| h5netcdf | 1.8.1 | BSD License |
| h5py | 3.16.0 | BSD-3-Clause |
| hf-xet | 1.8.0 | Apache-2.0 |
| httpcore2 | 2.16.0 | BSD-3-Clause |
| httpx2 | 2.16.0 | BSD-3-Clause |
| huggingface_hub | 2.4.0 | Apache Software License |
| hydra-core | 1.3.7 | MIT License |
| idna | 3.20 | BSD-3-Clause |
| importlib_metadata | 10.2.0 | Apache-2.0 |
| jaxtyping | 0.3.11 | MIT License |
| Jinja2 | 3.1.6 | BSD License |
| jmespath | 1.1.0 | MIT License |
| kiwisolver | 1.5.1 | BSD License |
| lazy-importer | 0.1.0 | Apache-2.0 |
| lightning | 2.6.6 | Apache Software License |
| lightning-utilities | 0.15.3 | Apache-2.0 |
| locket | 1.0.0 | BSD License |
| loguru | 0.8.0 | MIT License |
| lovely-numpy | 0.2.24 | MIT |
| lovely-tensors | 0.1.22 | MIT |
| lz4 | 4.4.5 | BSD License |
| markdown-it-py | 4.3.0 | MIT License |
| MarkupSafe | 3.0.3 | BSD-3-Clause |
| matplotlib | 3.11.2 | Python Software Foundation License |
| mdurl | 0.1.2 | MIT License |
| mpmath | 1.3.0 | BSD License |
| msgpack | 1.2.4 | Apache-2.0 |
| multidict | 7.2.1 | Apache-2.0 |
| narwhals | 2.28.0 | MIT |
| netCDF4 | 1.7.2 | MIT License |
| networkx | 3.7 | BSD-3-Clause |
| numcodecs | 0.17.1 | MIT |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| nvidia-cublas | 13.1.1.3 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cuda-cupti | 13.0.85 | Other/Proprietary License |
| nvidia-cuda-nvrtc | 13.0.88 | Other/Proprietary License |
| nvidia-cuda-runtime | 13.0.96 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cudnn-cu13 | 9.24.0.43 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cufft | 12.0.0.61 | Other/Proprietary License |
| nvidia-cufile | 1.15.1.6 | Other/Proprietary License |
| nvidia-curand | 10.4.0.35 | Other/Proprietary License |
| nvidia-cusolver | 12.0.4.66 | Other/Proprietary License |
| nvidia-cusparse | 12.6.3.3 | Other/Proprietary License |
| nvidia-cusparselt-cu13 | 0.8.1 | NVIDIA Proprietary Software |
| nvidia-nccl-cu13 | 2.30.7 | LicenseRef-NVIDIA-Proprietary |
| nvidia-nvjitlink | 13.4.92 | LicenseRef-NVIDIA-Proprietary |
| nvidia-nvshmem-cu13 | 3.4.5 | LicenseRef-NVIDIA-Proprietary |
| nvidia-nvtx | 13.0.85 | Other/Proprietary License |
| nvidia-physicsnemo | 2.3.0a0 | Apache-2.0 |
| nvtx | 0.2.17 | Apache-2.0 WITH LLVM-exception |
| oauthlib | 4.0.1 | BSD-3-Clause |
| obspec | 0.1.0 | MIT |
| obstore | 0.11.1 | MIT License |
| omegaconf | 2.3.1 | BSD License |
| opentelemetry-api | 1.45.2 | Apache-2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pandas | 3.0.5 | BSD License |
| partd | 1.4.2 | BSD |
| pillow | 13.0.0 | MIT-CMU |
| pip | 25.2 | MIT |
| propcache | 0.5.4 | Apache-2.0 |
| proto-plus | 1.29.0 | Apache Software License |
| protobuf | 7.36.2 | 3-Clause BSD License |
| psutil | 7.2.2 | BSD-3-Clause |
| pyarrow | 25.0.1 | Apache-2.0 |
| pyasn1 | 0.6.5 | BSD-2-Clause |
| pyasn1_modules | 0.4.2 | BSD License |
| pycparser | 3.12 | BSD-3-Clause |
| Pygments | 2.21.0 | BSD-2-Clause |
| pygrib | 2.1.8 | MIT AND (Apache-2.0 OR BSD-2-Clause) |
| pyparsing | 3.3.3 | MIT |
| pyproj | 3.8.0 | MIT |
| python-dateutil | 2.9.0.post0 | BSD License; Apache Software License |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| pytorch-lightning | 2.6.6 | Apache Software License |
| pyvers | 0.2.3 | MIT License |
| PyYAML | 6.0.3 | MIT License |
| regex | 2026.11.9 | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | Apache Software License |
| requests-oauthlib | 2.0.0 | BSD License |
| rich | 15.0.0 | MIT License |
| s3fs | 2026.7.0 | BSD License |
| safetensors | 0.8.0 | Apache Software License |
| scipy | 1.18.1 | BSD License |
| setuptools | 78.1.0 | MIT License |
| six | 1.17.0 | MIT License |
| smmap | 5.0.3 | BSD License |
| sortedcontainers | 2.4.0 | Apache Software License |
| sympy | 1.14.0 | BSD License |
| tblib | 3.2.2 | BSD-2-Clause |
| tensordict | 0.15.2 | MIT |
| termcolor | 3.3.0 | MIT |
| timm | 1.0.30 | Apache Software License |
| toolz | 1.2.0 | BSD-3-Clause |
| torch | 2.14.0+cu130 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| torchmetrics | 1.9.0 | Apache Software License |
| torchvision | 0.29.0+cu130 | BSD |
| tornado | 6.6.1 | Apache Software License |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| treelib | 1.8.0 | Apache Software License |
| triton | 3.8.0 | MIT |
| truststore | 0.10.5 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| urllib3 | 2.8.0 | MIT |
| wadler_lindig | 0.1.8 | Apache Software License |
| warp-lang | 1.19.0 | Apache-2.0 |
| wrapt | 2.5.2 | BSD-2-Clause |
| xarray | 2026.2.0 | Apache-2.0 |
| xyzservices | 2026.9.1 | BSD-3-Clause |
| yarl | 1.25.1 | Apache-2.0 |
| zarr | 3.3.0 | MIT |
| zict | 3.0.0 | BSD License |
| zipp | 4.1.2 | MIT |
