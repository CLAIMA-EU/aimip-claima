#!/bin/bash
# AIMIP native installer (no container). Creates virtual environments, writes the site profile and the `aimip` command.
#   ./install/install-native.sh --prefix DIR [--site levante|generic-slurm|local]
#       [--partition P] [--account A|none] [--gres G|none] [--run-root R]
#       [--with-emu] [--with-camulator] [--with-sdyff] [--python python3.12] [--dry-run]
# Afterwards:  DIR/bin/aimip setup --models all     (downloads models + data, verifies aimip.lock)
set -u
umask 022
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
VERSION=$(tr -d '[:space:]' < "$REPO/container/VERSION")
PREFIX=""; SITE=generic-slurm; PART=""; ACC=""; GRES=""; RUNROOT=""
WITH_EMU=0; WITH_CAMULATOR=0; WITH_SDYFF=0; PY_REQ=""; DRY_RUN=0
die() { echo "install-native: $*" >&2; exit 2; }
need_val() { [ "$1" -ge 2 ] && [ -n "${2:-}" ] || die "option $3 requires a value"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) need_val $# "${2:-}" "$1"; PREFIX=$2; shift 2;;
    --site) need_val $# "${2:-}" "$1"; SITE=$2; shift 2;;
    --partition) need_val $# "${2:-}" "$1"; PART=$2; shift 2;;
    --account) need_val $# "${2:-}" "$1"; ACC=$2; shift 2;;
    --gres) need_val $# "${2:-}" "$1"; GRES=$2; shift 2;;
    --run-root) need_val $# "${2:-}" "$1"; RUNROOT=$2; shift 2;;
    --with-emu) WITH_EMU=1; shift;;
    --with-camulator) WITH_CAMULATOR=1; shift;;
    --with-sdyff) WITH_SDYFF=1; shift;;
    --python) need_val $# "${2:-}" "$1"; PY_REQ=$2; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) sed -n 2,6p "$0"; exit 0;;
    *) die "unknown option $1";;
  esac
done
[ -n "$PREFIX" ] || die "--prefix is required"
case "$PREFIX" in /*) ;; *) PREFIX=$(pwd)/$PREFIX;; esac
case "$PREFIX" in *[!A-Za-z0-9._/+-]*|*..*) die "prefix must be a plain absolute path";; esac
[ -f "$HERE/presets/$SITE.json" ] || die "no preset $SITE (have: $(ls "$HERE/presets" | sed 's/\.json$//' | tr '\n' ' '))"

# ---- Python >= 3.12 check
if [ -n "$PY_REQ" ]; then
  if [ -x "$PY_REQ" ]; then PY_BIN=$PY_REQ; else PY_BIN=$(command -v "$PY_REQ" 2>/dev/null || true); fi
  [ -n "$PY_BIN" ] && [ -x "$PY_BIN" ] || die "Python interpreter not found or not executable: $PY_REQ"
else
  PY_BIN=$(command -v python3.12 2>/dev/null || command -v python3 2>/dev/null || true)
  [ -n "$PY_BIN" ] && [ -x "$PY_BIN" ] || die "python3.12 not found on PATH; pass --python <path-to-python3.12>"
fi
"$PY_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null \
  || die "Python >= 3.12 is required (got $("$PY_BIN" -V 2>&1 || echo unknown)); pass --python <python3.12>"

# ---- optional emulator / torch_harmonics build tools check
if [ "$WITH_EMU" = 1 ] || [ "$WITH_CAMULATOR" = 1 ] || [ "$WITH_SDYFF" = 1 ]; then
  if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
    NEED_CXX_FLAGS=""
    [ "$WITH_EMU" = 1 ] && NEED_CXX_FLAGS="$NEED_CXX_FLAGS --with-emu"
    [ "$WITH_CAMULATOR" = 1 ] && NEED_CXX_FLAGS="$NEED_CXX_FLAGS --with-camulator"
    [ "$WITH_SDYFF" = 1 ] && NEED_CXX_FLAGS="$NEED_CXX_FLAGS --with-sdyff"
    die "${NEED_CXX_FLAGS# } requires a C/C++ compiler (c++, g++, or clang++) on PATH (or rerun without${NEED_CXX_FLAGS})"
  fi
fi

TORCH_INDEX=${AIMIP_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
CAMULATOR_TORCH_INDEX=${AIMIP_CAMULATOR_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}
SDYFF_TORCH_INDEX=${AIMIP_SDYFF_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}
echo "install-native: version $VERSION, python $PY_BIN, site $SITE, with-emu $WITH_EMU, with-camulator $WITH_CAMULATOR, with-sdyff $WITH_SDYFF, prefix $PREFIX"

mkdir -p "$PREFIX"/{bin,data,build} || die "cannot create $PREFIX"

# ---- site profile (written and checked BEFORE environment creation so missing options fail immediately)
"$PY_BIN" - "$HERE/presets/$SITE.json" "$PREFIX/site.json" "$PART" "$ACC" "$GRES" "$RUNROOT" <<'PY' || die "site.json failed"
import json, sys
p = json.load(open(sys.argv[1])); part, acc, gres, rr = sys.argv[3:7]
if part: p["sbatch"]["partition"] = part
if acc == "none": p["sbatch"]["account"] = ""
elif acc: p["sbatch"]["account"] = acc
if gres == "none": p["sbatch"]["gres"] = ""
elif gres: p["sbatch"]["gres"] = gres
if rr: p["run_root"] = rr
p["runtime"] = "native"
bad = [k for k, v in p.get("sbatch", {}).items() if v == "CHANGE_ME"]
if bad: sys.exit(f"site preset needs --{' --'.join(bad)}")
json.dump(p, open(sys.argv[2], "w"), indent=2)
PY
SCHED=$("$PY_BIN" -c "import json;print(json.load(open('$PREFIX/site.json'))['scheduler'])") || die "cannot read scheduler from $PREFIX/site.json"
RROOT=$("$PY_BIN" -c "import json;print(json.load(open('$PREFIX/site.json'))['run_root'])") || die "cannot read run_root from $PREFIX/site.json"
EXTRA=$("$PY_BIN" -c "import json;print(' '.join(json.load(open('$PREFIX/site.json')).get('extra_ro_binds',[])))") || die "cannot read extra_ro_binds from $PREFIX/site.json"

ensure_world_readable() {
  find "$@" \
    \( -type d ! -perm -a+rx -exec chmod a+rx {} + \) -o \
    \( -type f -perm -u+x ! -perm -a+rx -exec chmod a+rx {} + \) -o \
    \( -type f ! -perm -a+r -exec chmod a+r {} + \) \
    || die "failed to set read/traverse permissions on $*"
}

write_launcher_and_version() {
  cp "$REPO/container/VERSION" "$PREFIX/VERSION" || die "cannot write $PREFIX/VERSION"
  cat > "$PREFIX/bin/aimip.new" <<'LAUNCHER' || die "cannot stage $PREFIX/bin/aimip.new"
#!/bin/bash
# AIMIP native launcher — generated by install/install-native.sh (do not edit by hand; re-run install-native.sh).
set -u
export LC_ALL=C
umask g-w,o-w
PREFIX=@PREFIX@
SCHEDULER=@SCHEDULER@
RUN_ROOT_TEMPLATE='@RUN_ROOT@'
EXTRA_RO="@EXTRA_RO@"
DATA=$PREFIX/data
U=$(id -un); U1=${U:0:1}
RUN_ROOT=${RUN_ROOT_TEMPLATE//'${USER}'/$U}; RUN_ROOT=${RUN_ROOT//'${U1}'/$U1}; RUN_ROOT=${RUN_ROOT//'${HOME}'/$HOME}

die() { echo "aimip: $*" >&2; exit 2; }
safe_path() {
  case "$1" in /*) ;; *) die "path must be absolute: $1";; esac
  case "$1" in *..*|*[!A-Za-z0-9._/+-]*) die "refusing path: $1";; esac
}

mkdir -p "$RUN_ROOT" 2>/dev/null || true
DATA_MODE=ro
case "${1:-}" in
  setup|fetch) mkdir -p "$DATA" 2>/dev/null; [ -w "$DATA" ] || die "$DATA is not writable by you: ask the install owner to run this"; DATA_MODE=rw;;
  run|ic|chat|mcp|ui|approve) mkdir -p "$DATA" 2>/dev/null || true; [ -w "$DATA" ] && DATA_MODE=rw;;
esac

if [ "${1:-}" = "chat" ] || [ "${1:-}" = "ui" ]; then
  if [ -n "${AIMIP_GEMINI_KEY_FILE:-}" ]; then safe_path "$AIMIP_GEMINI_KEY_FILE"
    [ -f "$AIMIP_GEMINI_KEY_FILE" ] && [ ! -L "$AIMIP_GEMINI_KEY_FILE" ] && [ -O "$AIMIP_GEMINI_KEY_FILE" ] \
      || die "AIMIP_GEMINI_KEY_FILE must be an existing regular file owned by you: $AIMIP_GEMINI_KEY_FILE"
  fi
  for f in AIMIP_GEMINI_VERTEX_TOKEN_FILE GOOGLE_APPLICATION_CREDENTIALS; do
    [ -n "${!f:-}" ] || continue; safe_path "${!f}"
    [ -f "${!f}" ] && [ ! -L "${!f}" ] && [ -O "${!f}" ] || die "$f must be an existing regular file owned by you: ${!f}"
  done
else
  unset GEMINI_API_KEY GOOGLE_API_KEY AIMIP_GEMINI_KEY_FILE AIMIP_GEMINI_VERTEX_TOKEN_FILE AIMIP_GEMINI_VERTEX_PROJECT AIMIP_GEMINI_VERTEX_LOCATION GOOGLE_APPLICATION_CREDENTIALS GOOGLE_CLOUD_PROJECT GCLOUD_PROJECT CLOUDSDK_AUTH_ACCESS_TOKEN OPENAI_API_KEY ANTHROPIC_API_KEY
fi

export AIMIP_HOME="$PREFIX"
export AIMIP_PLAYGROUND="$PREFIX"
export AIMIP_DATA="$DATA"
export AIMIP_SITE_CONFIG="$PREFIX/site.json"
export AIMIP_DEFAULT_OUT_ROOT="$RUN_ROOT"
export AIMIP_STATE="${AIMIP_STATE:-$RUN_ROOT/.aimip-state}"
export AIMIP_USER="$U"
export AIMIP_SCHEDULER="$SCHEDULER"
export AIMIP_EXTRA_RO_BINDS="$EXTRA_RO"
export AIMIP_DATA_MODE="$DATA_MODE"
export AIMIP_CHAT_LOG_DIR="${AIMIP_CHAT_LOG_DIR:-$RUN_ROOT/chat-logs}"
export AIMIP_IMAGE_SHA="native"
export AIMIP_FME_2026_10_OVERLAY="$PREFIX/overlay-fme-2026.10.0"
if [ -z "${AIMIP_EMU_PYTHON:-}" ] && [ -x "$PREFIX/venv-emu/bin/python" ]; then
  export AIMIP_EMU_PYTHON="$PREFIX/venv-emu/bin/python"
fi
if [ -z "${AIMIP_CAMULATOR_PYTHON:-}" ] && [ -x "$PREFIX/venv-camulator/bin/python" ]; then
  export AIMIP_CAMULATOR_PYTHON="$PREFIX/venv-camulator/bin/python"
fi
if [ -z "${AIMIP_SDYFF_PYTHON:-}" ] && [ -x "$PREFIX/venv-sdyff/bin/python" ]; then
  export AIMIP_SDYFF_PYTHON="$PREFIX/venv-sdyff/bin/python"
fi
export HF_HOME="${HF_HOME:-$DATA/.hf}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTHONNOUSERSITE=1
exec "$PREFIX/venv/bin/python" -I -B -m aimip "$@"   # -I: the current directory and PYTHONPATH cannot shadow the installed package
LAUNCHER
  sed -e "s#@PREFIX@#$PREFIX#" -e "s#@SCHEDULER@#$SCHED#" -e "s#@RUN_ROOT@#$RROOT#" -e "s#@EXTRA_RO@#$EXTRA#" \
      "$PREFIX/bin/aimip.new" > "$PREFIX/bin/aimip.tmp" \
    && rm -f "$PREFIX/bin/aimip.new" \
    && chmod 755 "$PREFIX/bin/aimip.tmp" \
    && mv "$PREFIX/bin/aimip.tmp" "$PREFIX/bin/aimip" \
    || die "cannot write $PREFIX/bin/aimip"
  ensure_world_readable "$PREFIX/bin" "$PREFIX/site.json" "$PREFIX/VERSION"
}

if [ "$DRY_RUN" = 1 ]; then
  write_launcher_and_version
  cat <<EOF
install-native (dry-run): wrote $PREFIX/site.json, $PREFIX/VERSION, and $PREFIX/bin/aimip
Commands that a full install runs:
  "$PY_BIN" -m venv "$PREFIX/venv"
  "$PREFIX/venv/bin/pip" install pip==25.2
  "$PREFIX/venv/bin/pip" install --index-url "$TORCH_INDEX" "torch==2.14.0"
  "$PREFIX/venv/bin/pip" install -c "$REPO/container/constraints.txt" "fme==2026.5.1" netCDF4 h5netcdf pyyaml pytest huggingface_hub scipy pandas
  "$PREFIX/venv/bin/pip" install -c "$REPO/container/constraints.txt" -r "$REPO/container/requirements-agent.txt"
  "$PREFIX/venv/bin/pip" install --no-deps "$REPO"
  "$PREFIX/venv/bin/pip" install --no-deps --target "$PREFIX/overlay-fme-2026.10.0" "fme==2026.10.0" "torch_harmonics==0.8.0" "zarrs==0.2.3"
EOF
  if [ "$WITH_EMU" = 1 ]; then
    cat <<EOF
  "$PY_BIN" -m venv "$PREFIX/venv-emu"
  "$PREFIX/venv-emu/bin/pip" install --index-url "$TORCH_INDEX" "torch==2.14.0" "torchvision==0.29.0"
  "$PREFIX/venv-emu/bin/pip" install -c "$REPO/container/constraints-emu.txt" earth2studio@8fa7acb nvidia-physicsnemo@1dacf8d ...
  "$PREFIX/venv-emu/bin/pip" install --no-build-isolation earth2grid@11dcf1b (with CUDA HEALPix padding kernel via nvcc 13.0)
  "$PREFIX/venv-emu/bin/pip" install --no-deps "$REPO"
  "$PREFIX/venv-emu/bin/python" -I "$REPO/src/aimip/geoarches_src.py" install "$PREFIX/geoarches"   (geoarches@4d0c77a1, ArchesWeather)
EOF
  fi
  if [ "$WITH_CAMULATOR" = 1 ]; then
    cat <<EOF
  "$PY_BIN" -m venv "$PREFIX/venv-camulator"
  "$PREFIX/venv-camulator/bin/pip" install -c "$REPO/container/constraints-camulator.txt" numpy==1.26.4 ...
  "$PREFIX/venv-camulator/bin/pip" install --index-url "$CAMULATOR_TORCH_INDEX" -c "$REPO/container/constraints-camulator.txt" "torch==2.4.1" "torchvision==0.19.1"
  "$PREFIX/venv-camulator/bin/pip" install -c "$REPO/container/constraints-camulator.txt" --no-build-isolation --no-deps "torch_harmonics==0.7.2"
  "$PREFIX/venv-camulator/bin/pip" install --no-build-isolation --no-deps "$REPO"
  "$PREFIX/venv-camulator/bin/python" -I -B -c 'import sys; from aimip.camulator import install_camulator_source_tree; install_camulator_source_tree(sys.argv[1])' "$PREFIX/venv-camulator"
EOF
  fi
  if [ "$WITH_SDYFF" = 1 ]; then
    cat <<EOF
  "$PY_BIN" -m venv "$PREFIX/venv-sdyff"
  "$PREFIX/venv-sdyff/bin/pip" install -c "$REPO/container/constraints-sdyff.txt" numpy==1.26.4 "pytorch-lightning==2.4.0" "hydra-core==1.3.2" "omegaconf==2.3.0" ...
  "$PREFIX/venv-sdyff/bin/pip" install --index-url "$SDYFF_TORCH_INDEX" -c "$REPO/container/constraints-sdyff.txt" "torch==2.4.1" "torchvision==0.19.1"
  "$PREFIX/venv-sdyff/bin/pip" install -c "$REPO/container/constraints-sdyff.txt" --no-build-isolation --no-deps "torch_harmonics==0.7.2"
  "$PREFIX/venv-sdyff/bin/pip" install --no-build-isolation --no-deps "nvidia-modulus @ https://github.com/ai2cm/modulus/archive/94f62e1ce2083640829ec12d80b00619c40a47f8.tar.gz#sha256=8785a70f2609746beaa5a3ff166b1a99393a9a969afef44c77fe1270ac5af246" "$REPO"
  "$PREFIX/venv-sdyff/bin/python" -I -B -c 'import sys; from aimip.sdyff import install_sdyff_apex_shim; install_sdyff_apex_shim(sys.argv[1])' "$PREFIX/venv-sdyff"
EOF
  fi
  exit 0
fi

# Remove any pre-existing launcher/VERSION so a failed install never leaves an active launcher behind.
rm -f "$PREFIX/bin/aimip" "$PREFIX/bin/aimip.new" "$PREFIX/bin/aimip.tmp" "$PREFIX/VERSION"

# ---- build main venv and fme 2026.10.0 overlay
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
LOG=$PREFIX/build/install-native-$VERSION.log
echo "install-native: creating $PREFIX/venv and $PREFIX/overlay-fme-2026.10.0 (log $LOG)"
{
  "$PY_BIN" -m venv "$PREFIX/venv" \
  && "$PREFIX/venv/bin/pip" install pip==25.2 \
  && "$PREFIX/venv/bin/pip" install --index-url "$TORCH_INDEX" "torch==2.14.0" \
  && "$PREFIX/venv/bin/pip" install -c "$REPO/container/constraints.txt" "fme==2026.5.1" netCDF4 h5netcdf pyyaml pytest huggingface_hub scipy pandas \
  && "$PREFIX/venv/bin/pip" install -c "$REPO/container/constraints.txt" -r "$REPO/container/requirements-agent.txt" \
  && "$PREFIX/venv/bin/pip" install --no-deps "$REPO" \
  && rm -rf "$PREFIX/overlay-fme-2026.10.0" \
  && "$PREFIX/venv/bin/pip" install --no-deps --target "$PREFIX/overlay-fme-2026.10.0" "fme==2026.10.0" "torch_harmonics==0.8.0" "zarrs==0.2.3" \
  && "$PREFIX/venv/bin/pip" freeze --all > "$PREFIX/installed-inventory.txt"
} > "$LOG" 2>&1 || { tail -30 "$LOG" >&2; die "venv build failed (see $LOG)"; }

# ---- optional emulator venv (DLESyM + ArchesClimate-SSP + ArchesWeather Python deps; same recipe as container/Dockerfile)
# geoarches (ArchesWeather) is unpacked into $PREFIX/geoarches from its pinned archive (src/aimip/geoarches_src.py).
if [ "$WITH_EMU" = 1 ]; then
  EMU_LOG=$PREFIX/build/install-native-emu-$VERSION.log
  echo "install-native: creating $PREFIX/venv-emu with CUDA earth2grid kernel (log $EMU_LOG)"
  {
    "$PY_BIN" -m venv "$PREFIX/venv-emu" \
    && "$PREFIX/venv-emu/bin/pip" install pip==25.2 \
    && "$PREFIX/venv-emu/bin/pip" install --index-url "$TORCH_INDEX" "torch==2.14.0" "torchvision==0.29.0" \
    && "$PREFIX/venv-emu/bin/pip" install -c "$REPO/container/constraints-emu.txt" \
      "earth2studio @ https://github.com/NVIDIA/earth2studio/archive/8fa7acb28b082e83a0718e8d2ed137fd375a11fe.tar.gz#sha256=db814396dff3fe021d9af70bf10a6f80577694416eeb8b70b8b504685df4ff20" \
      "nvidia-physicsnemo @ https://github.com/NVIDIA/physicsnemo/archive/1dacf8d691bb0ab987e8832debacab2dbe0043bb.tar.gz#sha256=ee6a06672258bbb7f20a4554725a5f22b9c504d67f57b4f49561d9cc577f09ae" \
      omegaconf hydra-core lightning tensordict timm diffusers lovely-tensors lovely-numpy matplotlib torchmetrics einops cftime netCDF4 pyyaml torch_harmonics \
      axial-attention fasteners \
    && rm -rf "$PREFIX/build/cuda-build" \
    && "$PREFIX/venv-emu/bin/pip" install --target "$PREFIX/build/cuda-build" \
      "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-crt==13.0.88" "nvidia-cuda-cccl==13.0.85" "nvidia-cuda-runtime==13.0.96" "nvidia-nvvm==13.0.88" \
    && ln -sfn lib "$PREFIX/build/cuda-build/nvidia/cu13/lib64" \
    && ln -sfn libcudart.so.13 "$PREFIX/build/cuda-build/nvidia/cu13/lib/libcudart.so" \
    && EMU_SITE=$("$PREFIX/venv-emu/bin/python" -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))') \
    && [ -n "$EMU_SITE" ] \
    && CUDA_HOME="$PREFIX/build/cuda-build/nvidia/cu13" \
      PATH="$PREFIX/build/cuda-build/nvidia/cu13/bin:$PATH" \
      CPATH="$EMU_SITE/nvidia/cu13/include" \
      TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX" \
      MAX_JOBS=4 \
      "$PREFIX/venv-emu/bin/pip" install --no-build-isolation \
      "earth2grid @ https://github.com/NVlabs/earth2grid/archive/11dcf1b0787a7eb6a8497a3a5a5e1fdcc31232d3.tar.gz#sha256=6ae3c5b020e1c9623a77debca1ef10f6a5127cffd8a084db8c965381f8928581" \
    && rm -rf "$PREFIX/build/cuda-build" \
    && "$PREFIX/venv-emu/bin/pip" install --no-deps \
      "cbottle @ https://github.com/NVlabs/cBottle/archive/f0f519987589cc0b368ab892a1adfb2fb30a5587.tar.gz#sha256=904e2e2e563b0c035f961e235d4081c4800fc4b1ab2767a507016ae2f2144fc1" \
      "$REPO" \
    && { "$PREFIX/venv-emu/bin/python" -I "$REPO/src/aimip/geoarches_src.py" verify "$PREFIX/geoarches" \
         || { rm -rf "$PREFIX/geoarches" \
              && "$PREFIX/venv-emu/bin/python" -I "$REPO/src/aimip/geoarches_src.py" install "$PREFIX/geoarches"; }; } \
    && "$PREFIX/venv-emu/bin/pip" freeze --all > "$PREFIX/installed-inventory-emu.txt"
  } > "$EMU_LOG" 2>&1 || { rm -rf "$PREFIX/build/cuda-build"; tail -30 "$EMU_LOG" >&2; die "venv-emu build failed (see $EMU_LOG); fix the error above or rerun without --with-emu to install the other models"; }
fi

ensure_world_readable "$PREFIX/venv" "$PREFIX/overlay-fme-2026.10.0"
[ "$WITH_EMU" = 1 ] && ensure_world_readable "$PREFIX/venv-emu" "$PREFIX/geoarches"

# ---- optional isolated CAMulator venv (torch 2.4.1 + torch_harmonics 0.7.2 + numpy 1.26.4; exact == version pins in
#      constraints-camulator.txt without --require-hashes, configurable AIMIP_CAMULATOR_TORCH_INDEX, matching venv/venv-emu)
if [ "$WITH_CAMULATOR" = 1 ]; then
  CAM_LOG=$PREFIX/build/install-native-camulator-$VERSION.log
  echo "install-native: creating $PREFIX/venv-camulator (torch 2.4.1, torch_harmonics 0.7.2, numpy 1.26.4; log $CAM_LOG)"
  {
    "$PY_BIN" -m venv "$PREFIX/venv-camulator" \
    && "$PREFIX/venv-camulator/bin/pip" install pip==25.2 \
    && "$PREFIX/venv-camulator/bin/pip" install -c "$REPO/container/constraints-camulator.txt" \
      setuptools setuptools-scm wheel ninja packaging "numpy==1.26.4" \
      fsspec filelock typing_extensions sympy networkx Jinja2 MarkupSafe mpmath pillow \
    && "$PREFIX/venv-camulator/bin/pip" install --index-url "$CAMULATOR_TORCH_INDEX" -c "$REPO/container/constraints-camulator.txt" \
      "torch==2.4.1" "torchvision==0.19.1" \
    && "$PREFIX/venv-camulator/bin/pip" install -c "$REPO/container/constraints-camulator.txt" \
      xarray netCDF4 h5netcdf h5py cftime scipy pandas pyyaml einops timm tqdm zarr numcodecs \
      scikit-learn numba bridgescaler dask distributed matplotlib gcsfs huggingface_hub safetensors \
      segmentation-models-pytorch torch-geometric torchmetrics \
    && PATH="$PREFIX/venv-camulator/bin:$PATH" "$PREFIX/venv-camulator/bin/pip" install \
      -c "$REPO/container/constraints-camulator.txt" --no-build-isolation --no-deps "torch_harmonics==0.7.2" \
    && "$PREFIX/venv-camulator/bin/pip" install --no-build-isolation --no-deps "$REPO" \
    && "$PREFIX/venv-camulator/bin/python" -I -B -c \
      'import sys; from aimip.camulator import install_camulator_source_tree; install_camulator_source_tree(sys.argv[1])' \
      "$PREFIX/venv-camulator" \
    && "$PREFIX/venv-camulator/bin/pip" freeze --all > "$PREFIX/installed-inventory-camulator.txt"
  } > "$CAM_LOG" 2>&1 || { tail -30 "$CAM_LOG" >&2; die "venv-camulator build failed (see $CAM_LOG); fix the error above or rerun without --with-camulator to install the other models"; }
  ensure_world_readable "$PREFIX/venv-camulator"
fi

# ---- optional isolated Spherical DYffusion venv (torch 2.4.1 + torch_harmonics 0.7.2 + ai2cm/modulus@94f62e1c#sha256=...;
#      exact == version pins in constraints-sdyff.txt without --require-hashes, configurable AIMIP_SDYFF_TORCH_INDEX)
if [ "$WITH_SDYFF" = 1 ]; then
  SDYFF_LOG=$PREFIX/build/install-native-sdyff-$VERSION.log
  echo "install-native: creating $PREFIX/venv-sdyff (torch 2.4.1, torch_harmonics 0.7.2, ai2cm/modulus@94f62e1c; log $SDYFF_LOG)"
  {
    "$PY_BIN" -m venv "$PREFIX/venv-sdyff" \
    && "$PREFIX/venv-sdyff/bin/pip" install pip==25.2 \
    && "$PREFIX/venv-sdyff/bin/pip" install -c "$REPO/container/constraints-sdyff.txt" \
      setuptools setuptools-scm wheel ninja packaging "numpy==1.26.4" \
      fsspec filelock typing_extensions sympy networkx Jinja2 MarkupSafe mpmath pillow \
    && "$PREFIX/venv-sdyff/bin/pip" install --index-url "$SDYFF_TORCH_INDEX" -c "$REPO/container/constraints-sdyff.txt" \
      "torch==2.4.1" "torchvision==0.19.1" \
    && "$PREFIX/venv-sdyff/bin/pip" install -c "$REPO/container/constraints-sdyff.txt" \
      "pytorch-lightning==2.4.0" "hydra-core==1.3.2" "omegaconf==2.3.0" \
      dacite tensordict tensorly tensorly-torch toolz torchmetrics einops \
      xarray netCDF4 h5netcdf h5py cftime scipy pandas pyyaml "ruamel.yaml" dask zarr numcodecs \
      tqdm matplotlib wandb huggingface_hub rich timm boto3 requests regex \
    && PATH="$PREFIX/venv-sdyff/bin:$PATH" "$PREFIX/venv-sdyff/bin/pip" install \
      -c "$REPO/container/constraints-sdyff.txt" --no-build-isolation --no-deps "torch_harmonics==0.7.2" \
    && "$PREFIX/venv-sdyff/bin/pip" install --no-build-isolation --no-deps \
      "nvidia-modulus @ https://github.com/ai2cm/modulus/archive/94f62e1ce2083640829ec12d80b00619c40a47f8.tar.gz#sha256=8785a70f2609746beaa5a3ff166b1a99393a9a969afef44c77fe1270ac5af246" \
      "$REPO" \
    && "$PREFIX/venv-sdyff/bin/python" -I -B -c \
      'import sys; from aimip.sdyff import install_sdyff_apex_shim; install_sdyff_apex_shim(sys.argv[1])' \
      "$PREFIX/venv-sdyff" \
    && "$PREFIX/venv-sdyff/bin/pip" freeze --all > "$PREFIX/installed-inventory-sdyff.txt"
  } > "$SDYFF_LOG" 2>&1 || { tail -30 "$SDYFF_LOG" >&2; die "venv-sdyff build failed (see $SDYFF_LOG); fix the error above or rerun without --with-sdyff to install the other models"; }
  ensure_world_readable "$PREFIX/venv-sdyff"
fi

# ---- version/import assertions and self-test in the new environment (launcher written ONLY after all pass)
echo "install-native: verifying installed versions and running test suite"
"$PREFIX/venv/bin/python" -I -c '
import os, sys, torch, fme, aimip
from fme.core import set_seed
import fme.ace.inference
want_v, venv_dir = sys.argv[1], os.path.realpath(sys.argv[2])
aimip_file = os.path.realpath(aimip.__file__)
assert aimip_file.startswith(venv_dir + os.sep), f"aimip imported from {aimip_file}, expected {venv_dir}"
assert aimip.__version__ == want_v, f"aimip {aimip.__version__} != {want_v}"
assert fme.__version__ == "2026.5.1", f"fme {fme.__version__} != 2026.5.1"
assert torch.__version__.split("+")[0] == "2.14.0", f"torch {torch.__version__} != 2.14.0"
print("AIMIP_NATIVE_OK", torch.__version__, fme.__version__, aimip.__version__)
' "$VERSION" "$PREFIX/venv" || die "import/version check failed"

"$PREFIX/venv/bin/python" -I -c '
import os, sys
ov = os.path.realpath(sys.argv[1])
sys.path.insert(0, ov)
import fme, fme.coupled.inference
assert os.path.realpath(fme.__file__).startswith(ov + os.sep), f"overlay fme imported from {fme.__file__}, expected {ov}"
assert fme.__version__ == "2026.10.0", f"overlay fme {fme.__version__} != 2026.10.0"
print("AIMIP_OVERLAY_OK", fme.__version__)
' "$PREFIX/overlay-fme-2026.10.0" || die "overlay check failed"

if [ "$WITH_EMU" = 1 ]; then
  "$PREFIX/venv-emu/bin/python" -I -c '
import os, sys, torch, earth2studio, physicsnemo, earth2grid, healpixpad_cuda, cbottle, aimip, aimip.emu_job
from earth2studio.models.px import DLESyMLatLon
from earth2studio.models.px.dlesym_v0_isccp_era5 import DLESyMv0_ISCCP_ERA5LatLon
from earth2studio.data import ARCO_ERA5, CBottle3D
want_v, venv_dir = sys.argv[1], os.path.realpath(sys.argv[2])
assert os.path.realpath(aimip.__file__).startswith(venv_dir + os.sep), f"venv-emu aimip imported from {aimip.__file__}"
assert aimip.__version__ == want_v, f"venv-emu aimip {aimip.__version__} != {want_v}"
assert torch.__version__.split("+")[0] == "2.14.0", f"venv-emu torch {torch.__version__} != 2.14.0"
print("AIMIP_EMU_OK", torch.__version__, earth2studio.__version__, physicsnemo.__version__)
import axial_attention, fasteners
from aimip.geoarches_src import code_dir, verify
verify(code_dir())
sys.path.insert(0, code_dir())
sys.dont_write_bytecode = True   # keep the geoarches tree identical to its pinned digest
import geoarches.lightning_modules.forecast, geoarches.lightning_modules.diffusion, geoarches.dataloaders.era5
verify(code_dir())
print("AIMIP_GEOARCHES_OK", code_dir())
' "$VERSION" "$PREFIX/venv-emu" || die "venv-emu import/version check failed (rerun without --with-emu to install the other models)"
fi

if [ "$WITH_CAMULATOR" = 1 ]; then
  "$PREFIX/venv-camulator/bin/python" -I -B -c '
import os, sys, numpy, torch, torch_harmonics, credit, aimip, aimip.emu_job
from credit.parser import credit_main_parser
from credit.models import load_model, load_model_name, model_types
from credit.models.crossformer import CrossFormer
from credit.models.camulator import Camulator
from credit.models.wxformer.crossformer import CrossFormer as WXFormer
from credit.models.checkpoint import load_model_state
from credit.postblock import (
    PostBlock,
    TracerFixer,
    GlobalMassFixer,
    GlobalWaterFixer,
    GlobalEnergyFixer,
    GlobalEnergyFixerUpDown,
)
from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
from credit.transforms.deprecated._transforms import ToTensor
from credit.preblock.scaler import Scaler
from credit.distributed import distributed_model_wrapper
from credit.output import load_metadata, make_xarray, save_netcdf_increment

want_v, venv_dir = sys.argv[1], os.path.realpath(sys.argv[2])
src_root = os.path.realpath(os.path.join(venv_dir, "src", "miles-credit"))
climate_dir = os.path.join(src_root, "climate")
assert os.path.realpath(aimip.__file__).startswith(venv_dir + os.sep), f"venv-camulator aimip imported from {aimip.__file__}"
assert os.path.realpath(credit.__file__).startswith(src_root + os.sep), f"venv-camulator credit imported from {credit.__file__}, expected {src_root}"
assert aimip.__version__ == want_v, f"venv-camulator aimip {aimip.__version__} != {want_v}"
assert model_types["camulator"][0] is Camulator, "model_types[\"camulator\"] did not resolve to Camulator"
assert os.path.isfile(os.path.join(climate_dir, "Quick_Climate.py")), f"missing {climate_dir}/Quick_Climate.py"
sys.path.insert(0, climate_dir)
from WindPP import post_process_wind_artifacts
import Model_State
from Model_State import initialize_camulator, CAMulatorStepper
assert Model_State.CREDIT_AVAILABLE and Model_State.POSTBLOCK_AVAILABLE and Model_State.WINDPP_AVAILABLE, (
    f"Model_State optional imports failed: CREDIT={Model_State.CREDIT_AVAILABLE}, "
    f"POSTBLOCK={Model_State.POSTBLOCK_AVAILABLE}, WINDPP={Model_State.WINDPP_AVAILABLE}"
)
assert int(numpy.__version__.split(".")[0]) < 2, f"venv-camulator numpy {numpy.__version__} must be < 2"
assert torch.__version__.split("+")[0] == "2.4.1", f"venv-camulator torch {torch.__version__} != 2.4.1"
assert torch_harmonics.__version__ == "0.7.2", f"venv-camulator torch_harmonics {torch_harmonics.__version__} != 0.7.2"
print("AIMIP_CAMULATOR_OK", torch.__version__, torch_harmonics.__version__, numpy.__version__, Camulator.__name__)
' "$VERSION" "$PREFIX/venv-camulator" || die "venv-camulator import/version check failed (rerun without --with-camulator to install the other models)"
fi

if [ "$WITH_SDYFF" = 1 ]; then
  "$PREFIX/venv-sdyff/bin/python" -I -B -c '
import os, sys, numpy, torch, torch_harmonics, pytorch_lightning, hydra, omegaconf, dacite, tensorly, tltorch, tensordict, toolz, boto3, apex, modulus, aimip, aimip.emu_job
from apex.normalization import FusedLayerNorm
from modulus.models.sfno.initialization import trunc_normal_
from modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet
want_v, venv_dir = sys.argv[1], os.path.realpath(sys.argv[2])
assert os.path.realpath(aimip.__file__).startswith(venv_dir + os.sep), f"venv-sdyff aimip imported from {aimip.__file__}"
assert os.path.realpath(apex.__file__).startswith(venv_dir + os.sep), f"venv-sdyff apex shim imported from {apex.__file__}"
assert FusedLayerNorm is torch.nn.LayerNorm, "venv-sdyff apex.normalization.FusedLayerNorm must alias torch.nn.LayerNorm"
assert aimip.__version__ == want_v, f"venv-sdyff aimip {aimip.__version__} != {want_v}"
assert int(numpy.__version__.split(".")[0]) < 2, f"venv-sdyff numpy {numpy.__version__} must be < 2"
assert torch.__version__.split("+")[0] == "2.4.1", f"venv-sdyff torch {torch.__version__} != 2.4.1"
assert torch_harmonics.__version__ == "0.7.2", f"venv-sdyff torch_harmonics {torch_harmonics.__version__} != 0.7.2"
print("AIMIP_SDYFF_OK", torch.__version__, torch_harmonics.__version__, pytorch_lightning.__version__, FusedLayerNorm.__name__)
' "$VERSION" "$PREFIX/venv-sdyff" || die "venv-sdyff import/version check failed (rerun without --with-sdyff to install the other models)"
fi

selftest_dir_ok() {  # selftest_dir_ok <existing dir>
  local d a m probe
  d=$(cd "$1" 2>/dev/null && pwd -P) || return 1
  case "$d" in *..*|*[!A-Za-z0-9._/+-]*) return 1;; esac
  a=$d
  while [ "$a" != / ]; do
    m=$(ls -ld "$a" | cut -c1-10)
    case "${m:5:1}${m:8:1}" in *w*) case "${m:9:1}" in t|T) ;; *) return 1;; esac;; esac
    a=$(dirname "$a")
  done
  probe=$(mktemp "$d/.aimip-exec-probe.XXXXXX") || return 1
  printf '#!/bin/sh\nexit 0\n' > "$probe"; chmod 700 "$probe"
  "$probe" 2>/dev/null; m=$?; rm -f "$probe"; return $m
}
expand_run_root() {  # expand_run_root <template>; same substitutions as container/aimip-wrapper.in, without eval
  local r=$1 u u1
  u=$(id -un); u1=${u:0:1}
  r=${r//'${USER}'/$u}; r=${r//'${U1}'/$u1}; r=${r//'${HOME}'/$HOME}
  printf '%s\n' "$r"
}
SELFTEST_RUN_ROOT=$(expand_run_root "$RROOT")
PTDIR=""
for c in "$PREFIX/build" "$SELFTEST_RUN_ROOT" "${TMPDIR:-}" /tmp; do
  [ -n "$c" ] && [ -d "$c" ] && selftest_dir_ok "$c" && { PTDIR=$(cd "$c" && pwd -P); break; }
done
[ -n "$PTDIR" ] || die "no safe directory for self-test pytest: $PREFIX/build, $SELFTEST_RUN_ROOT, \$TMPDIR and /tmp each have a group- or world-writable parent, characters the gate refuses, or forbid execution; rerun with TMPDIR=<a directory of your own>"
PYTEST_TMP=$(mktemp -d "$PTDIR/aimip-pytest.XXXXXX") || die "cannot create temporary pytest directory"
trap 'rm -rf "$PYTEST_TMP"' EXIT
( cd "$REPO" && umask 022 && TMPDIR="$PTDIR" HF_HUB_OFFLINE=1 "$PREFIX/venv/bin/python" -B -m pytest -q -p no:cacheprovider --basetemp="$PYTEST_TMP/run" tests ) || die "pytest failed"
rm -rf "$PYTEST_TMP"; trap - EXIT

write_launcher_and_version
"$PREFIX/bin/aimip" --version || die "launcher check failed"

cat <<EOF

Installed AIMIP (native mode) $VERSION in $PREFIX
Next:
  $PREFIX/bin/aimip setup --models all        # download checkpoints + data, verify aimip.lock (~215 GB)
                                               # or skip it: the first 'aimip run -m X' downloads only model X
  $PREFIX/bin/aimip models
  $PREFIX/bin/aimip run -m ace2-era5 --start 1979-01-01T00:00:00 --steps 1460 --submit   # then: aimip submit <dir>
  $PREFIX/bin/aimip chat                      # Gemini key or Vertex credentials: see docs/agent.md
EOF
