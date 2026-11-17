# AIMIP container image recipe. container/gen_apptainer_def.py generates apptainer.def from this file so Docker and
# Apptainer builds cannot drift.
# Base: python:3.12.11-bookworm. Build-time pytest runs under umask 022 because Apptainer %post executes with umask 000
# and aimip refuses world-writable run directories.
# fme 2026.10.0 (HiRO-ACEv2, SamudrACE-E3SMv3) is installed in /opt/aimip/overlay-fme-2026.10.0 and placed first on
# sys.path by exec-fme only for models that declare fme_2026_10, keeping fme 2026.5.1 as the default environment.
# Default BASE_IMAGE is the linux/amd64 manifest digest of python:3.12.11-bookworm; on aarch64 hosts
# install-container.sh overrides BASE_IMAGE=python:3.12.11-bookworm.
ARG BASE_IMAGE=python@sha256:1b48a1b13ec678be8d84974541cc007d4367aaf01f985231daad1a940106efc0
FROM ${BASE_IMAGE}
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ENV PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 PATH=/opt/aimip/venv/bin:$PATH
COPY container/constraints.txt container/requirements-agent.txt container/constraints-emu.txt /opt/aimip/
RUN python3 -m venv /opt/aimip/venv \
 && pip install pip==25.2 \
 && pip install --index-url ${TORCH_INDEX} "torch==2.14.0" \
 && pip install -c /opt/aimip/constraints.txt "fme==2026.5.1" netCDF4 h5netcdf pyyaml pytest huggingface_hub scipy pandas \
 && pip install -c /opt/aimip/constraints.txt -r /opt/aimip/requirements-agent.txt \
 && pip install --no-deps --target /opt/aimip/overlay-fme-2026.10.0 "fme==2026.10.0" "torch_harmonics==0.8.0" "zarrs==0.2.3"
# Second environment for DLESyM (Earth2Studio 8fa7acb, physicsnemo 1dacf8d, earth2grid 11dcf1b) and
# ArchesClimate-SSP Python dependencies (ArchesClimate-SSP source itself is downloaded on first use into $AIMIP_DATA/code/).
# earth2grid is built with its CUDA HEALPix padding kernel (healpixpad_cuda) using nvcc 13.0 from NVIDIA's wheels in a
# temporary directory because the pure-PyTorch fallback changes DLESyM v1.1 results.
RUN python3 -m venv /opt/aimip/venv-emu \
 && /opt/aimip/venv-emu/bin/pip install pip==25.2 \
 && /opt/aimip/venv-emu/bin/pip install --index-url ${TORCH_INDEX} "torch==2.14.0" "torchvision==0.29.0" \
 && /opt/aimip/venv-emu/bin/pip install -c /opt/aimip/constraints-emu.txt "earth2studio @ https://github.com/NVIDIA/earth2studio/archive/8fa7acb28b082e83a0718e8d2ed137fd375a11fe.tar.gz#sha256=db814396dff3fe021d9af70bf10a6f80577694416eeb8b70b8b504685df4ff20" "nvidia-physicsnemo @ https://github.com/NVIDIA/physicsnemo/archive/1dacf8d691bb0ab987e8832debacab2dbe0043bb.tar.gz#sha256=ee6a06672258bbb7f20a4554725a5f22b9c504d67f57b4f49561d9cc577f09ae" omegaconf hydra-core lightning tensordict timm diffusers lovely-tensors lovely-numpy matplotlib torchmetrics einops cftime netCDF4 pyyaml torch_harmonics \
 && /opt/aimip/venv-emu/bin/pip install --target /opt/aimip/cuda-build "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-crt==13.0.88" "nvidia-cuda-cccl==13.0.85" "nvidia-cuda-runtime==13.0.96" "nvidia-nvvm==13.0.88" \
 && ln -sfn lib /opt/aimip/cuda-build/nvidia/cu13/lib64 && ln -sfn libcudart.so.13 /opt/aimip/cuda-build/nvidia/cu13/lib/libcudart.so \
 && CUDA_HOME=/opt/aimip/cuda-build/nvidia/cu13 PATH=/opt/aimip/cuda-build/nvidia/cu13/bin:$PATH CPATH=/opt/aimip/venv-emu/lib/python3.12/site-packages/nvidia/cu13/include TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX" MAX_JOBS=4 /opt/aimip/venv-emu/bin/pip install --no-build-isolation "earth2grid @ https://github.com/NVlabs/earth2grid/archive/11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3.tar.gz#sha256=6ae3c5b020e1c9623a77debca1ef10f6a5127cffd8a084db8c965381f8928581" \
 && /opt/aimip/venv-emu/bin/pip install --no-deps "cbottle @ https://github.com/NVlabs/cBottle/archive/f0f519987589cc0b368ab892a1adfb2fb30a5587.tar.gz#sha256=904e2e2e563b0c035f961e235d4081c4800fc4b1ab2767a507016ae2f2144fc1" \
 && rm -rf /opt/aimip/cuda-build \
 && /opt/aimip/venv-emu/bin/pip freeze --all > /opt/aimip/installed-inventory-emu.txt
COPY pyproject.toml LICENSE THIRD_PARTY.md THIRD_PARTY-PACKAGES.md README.md /opt/aimip/src/
COPY src /opt/aimip/src/src
COPY tests /opt/aimip/src/tests
COPY docs /opt/aimip/src/docs
COPY container /opt/aimip/src/container
COPY install /opt/aimip/src/install
RUN pip install --no-deps /opt/aimip/src \
 && /opt/aimip/venv-emu/bin/pip install --no-deps /opt/aimip/src \
 && pip freeze --all > /opt/aimip/installed-inventory.txt \
 && (cd /opt/aimip/src && find . -type f ! -path "*/__pycache__/*" -print0 | sort -z | xargs -0 sha256sum) > /opt/aimip/src.sha256 \
 && chmod go-w /opt && chmod -R go-w /opt/aimip \
 && cd /opt/aimip/src && umask 022 && TMPDIR=/tmp HOME=/tmp python -m pytest -q -p no:cacheprovider --basetemp=/opt/aimip/.pytest-tmp tests \
 && rm -rf /opt/aimip/.pytest-tmp \
 && python -c "import torch, fme, aimip; from fme.core import set_seed; import fme.ace.inference; print('AIMIP_IMAGE_OK', torch.__version__, fme.__version__, aimip.__version__)" \
 && PYTHONPATH=/opt/aimip/overlay-fme-2026.10.0 python -c "import fme, fme.coupled.inference; assert fme.__version__ == '2026.10.0', fme.__version__; print('AIMIP_OVERLAY_OK', fme.__version__)"
RUN /opt/aimip/venv-emu/bin/python -I -c "import torch, earth2studio, physicsnemo, earth2grid, healpixpad_cuda, cbottle, aimip.emu_job; from earth2studio.models.px import DLESyMLatLon; from earth2studio.models.px.dlesym_v0_isccp_era5 import DLESyMv0_ISCCP_ERA5LatLon; from earth2studio.data import ARCO_ERA5, CBottle3D; print('AIMIP_EMU_OK', torch.__version__, earth2studio.__version__, physicsnemo.__version__)"
ENTRYPOINT ["python", "-B", "-m", "aimip"]
