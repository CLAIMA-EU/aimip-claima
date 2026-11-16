#!/bin/bash
# AIMIP container installer. Fetches the published image (or builds it with --build), writes the site profile and the
# `aimip` command.
#   ./install/install-container.sh --prefix DIR [--site levante|generic-slurm|local]
#       [--runtime auto|apptainer|singularity|docker|podman] [--image REF | --build]
#       [--partition P] [--account A|none] [--gres G|none] [--run-root R]
# Afterwards:  DIR/bin/aimip setup --models all     (downloads models + data, verifies aimip.lock)
set -u
umask 022
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
VERSION=$(tr -d '[:space:]' < "$REPO/container/VERSION")
PREFIX=""; SITE=generic-slurm; RT=auto; PART=""; ACC=""; GRES=""; RUNROOT=""
IMAGE_REF="ghcr.io/clidyn/aimip-claima:$VERSION"; BUILD=0; USER_SET_IMAGE=0
die() { echo "install-container: $*" >&2; exit 2; }
need_val() { [ "$1" -ge 2 ] && [ -n "${2:-}" ] || die "option $3 requires a value"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) need_val $# "${2:-}" "$1"; PREFIX=$2; shift 2;;
    --site) need_val $# "${2:-}" "$1"; SITE=$2; shift 2;;
    --runtime) need_val $# "${2:-}" "$1"; RT=$2; shift 2;;
    --partition) need_val $# "${2:-}" "$1"; PART=$2; shift 2;;
    --account) need_val $# "${2:-}" "$1"; ACC=$2; shift 2;;
    --gres) need_val $# "${2:-}" "$1"; GRES=$2; shift 2;;
    --run-root) need_val $# "${2:-}" "$1"; RUNROOT=$2; shift 2;;
    --image) need_val $# "${2:-}" "$1"; IMAGE_REF=$2; USER_SET_IMAGE=1; shift 2;;
    --build) BUILD=1; shift;;
    -h|--help) sed -n 2,7p "$0"; exit 0;;
    *) die "unknown option $1";;
  esac
done
[ -n "$PREFIX" ] || die "--prefix is required"
case "$PREFIX" in /*) ;; *) PREFIX=$(pwd)/$PREFIX;; esac
case "$PREFIX" in *[!A-Za-z0-9._/+-]*|*..*) die "prefix must be a plain absolute path";; esac
case "$IMAGE_REF" in
  ""|-*|[!A-Za-z0-9]*|*[!A-Za-z0-9._/:@+-]*)
    die "--image must be a plain registry reference starting with an alphanumeric character (e.g. ghcr.io/org/name:tag)";;
esac
[ -f "$HERE/presets/$SITE.json" ] || die "no preset $SITE (have: $(ls "$HERE/presets" | sed 's/\.json$//' | tr '\n' ' '))"

image_registry_host() {
  local ref=$1 first
  case "$ref" in
    */*)
      first=${ref%%/*}
      case "$first" in *.*|*:*|localhost) printf '%s\n' "$first"; return;; esac
      ;;
  esac
  printf 'docker.io\n'
}
REG_HOST=$(image_registry_host "$IMAGE_REF")

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1"
  else
    shasum -a 256 "$1"
  fi
}

# ---- runtime
find_bin() { for c in "$@"; do command -v "$c" >/dev/null 2>&1 && { command -v "$c"; return 0; }; done; return 1; }
if [ "$RT" = auto ]; then
  for r in apptainer singularity podman docker; do B=$(find_bin "$r") && { RT=$r; break; }; done
  [ "$RT" = auto ] && [ -x /sw/spack-levante/apptainer-1.5.3-6acsty/bin/apptainer ] && RT=apptainer
fi
case "$RT" in
  apptainer) RTB=$(find_bin apptainer || echo /sw/spack-levante/apptainer-1.5.3-6acsty/bin/apptainer);;
  singularity|docker|podman) RTB=$(find_bin "$RT") || die "$RT not found";;
  *) die "no container runtime found (need apptainer, singularity, docker or podman)";;
esac
[ -x "$RTB" ] || die "runtime binary $RTB not executable"
echo "install-container: version $VERSION, runtime $RT ($RTB), site $SITE, prefix $PREFIX"
ARCH=$(uname -m); BASE_ARG=""
if [ "$ARCH" = aarch64 ] || [ "$ARCH" = arm64 ]; then
  BASE_ARG="python:3.12.11-bookworm"
  if [ "$BUILD" = 0 ] && [ "$USER_SET_IMAGE" = 0 ]; then
    BUILD=1
    echo "install-container: $ARCH host — the published image is x86_64 only; switching to --build automatically (base $BASE_ARG)"
  else
    echo "install-container: $ARCH host, base image by tag ($BASE_ARG; the pinned digest is amd64)"
  fi
fi

mkdir -p "$PREFIX"/{bin,images,data,build} || die "cannot create $PREFIX"
rm -f "$PREFIX/bin/aimip" "$PREFIX/bin/aimip.new" "$PREFIX/VERSION"
[ -e "$PREFIX/images/aimip-$VERSION.sif" ] && echo "install-container: an image for $VERSION already exists in $PREFIX — verifying and reusing it (delete $PREFIX/images to rebuild)"

# ---- site profile (written and checked BEFORE the long image build so missing options fail immediately)
python3 - "$HERE/presets/$SITE.json" "$PREFIX/site.json" "$PART" "$ACC" "$GRES" "$RUNROOT" "$RT" <<'PY' || die "site.json failed"
import json, sys
p = json.load(open(sys.argv[1])); part, acc, gres, rr, rt = sys.argv[3:8]
if part: p["sbatch"]["partition"] = part
if acc == "none": p["sbatch"]["account"] = ""
elif acc: p["sbatch"]["account"] = acc
if gres == "none": p["sbatch"]["gres"] = ""
elif gres: p["sbatch"]["gres"] = gres
if rr: p["run_root"] = rr
p["runtime"] = rt
bad = [k for k, v in p.get("sbatch", {}).items() if v == "CHANGE_ME"]
if bad: sys.exit(f"site preset needs --{' --'.join(bad)}")
json.dump(p, open(sys.argv[2], "w"), indent=2)
PY
SCHED=$(python3 -c "import json;print(json.load(open('$PREFIX/site.json'))['scheduler'])") || die "cannot read scheduler from $PREFIX/site.json"
RROOT=$(python3 -c "import json;print(json.load(open('$PREFIX/site.json'))['run_root'])") || die "cannot read run_root from $PREFIX/site.json"
EXTRA=$(python3 -c "import json;print(' '.join(json.load(open('$PREFIX/site.json')).get('extra_ro_binds',[])))") || die "cannot read extra_ro_binds from $PREFIX/site.json"

check_sif_version() {
  "$RTB" -s exec --containall --cleanenv --no-home "$1" python -I -B -c 'import aimip, sys; sys.exit(0 if aimip.__version__ == sys.argv[1] else 1)' "$VERSION"
}

# ---- build
cd "$REPO" || exit 2
LOG=$PREFIX/build/build-$VERSION.log
case "$RT" in
  apptainer|singularity)
    SIF=$PREFIX/images/aimip-$VERSION.sif
    if [ ! -e "$SIF" ] && [ "$BUILD" = 0 ]; then
      # The published image: converted to a SIF here, no root and no compiler needed.
      export APPTAINER_TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/aimip-pull.XXXXXX")
      export SINGULARITY_TMPDIR=$APPTAINER_TMPDIR; trap 'rm -rf "$APPTAINER_TMPDIR" "$SIF.incoming"' EXIT
      export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$PREFIX/build/cache} SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-$PREFIX/build/cache}
      echo "install-container: fetching docker://$IMAGE_REF into $SIF (log $LOG)"
      nice "$RTB" build --mksquashfs-args "-processors ${AIMIP_BUILD_THREADS:-4}" "$SIF.incoming" "docker://$IMAGE_REF" > "$LOG" 2>&1 \
        || { rm -f "$SIF.incoming"; tail -20 "$LOG" >&2; die "could not fetch $IMAGE_REF (a private image needs '$RTB registry login docker://$REG_HOST'; or build it here with --build)"; }
      check_sif_version "$SIF.incoming" \
        || { rm -f "$SIF.incoming"; die "the image $IMAGE_REF does not hold aimip $VERSION (check out the matching release of this repository)"; }
      mv "$SIF.incoming" "$SIF" || die "cannot move $SIF.incoming to $SIF"
      printf '%s\n' "docker://$IMAGE_REF" > "$PREFIX/image.ref"
    elif [ ! -e "$SIF" ]; then
      python3 "$REPO/container/gen_apptainer_def.py" "$REPO/container/Dockerfile" "$PREFIX/build/apptainer.def" "$BASE_ARG" || die "def generation failed"
      rm -rf "$PREFIX/build/ctx" && mkdir -p "$PREFIX/build/ctx" || die "cannot stage build context"
      cp -R "$REPO/pyproject.toml" "$REPO/LICENSE" "$REPO/THIRD_PARTY.md" "$REPO/THIRD_PARTY-PACKAGES.md" "$REPO/README.md" "$REPO/src" "$REPO/tests" "$REPO/container" "$REPO/install" "$PREFIX/build/ctx/" || die "cannot copy build context"
      find "$PREFIX/build/ctx" \( -name "__pycache__" -o -name "*.pyc" -o -name ".pytest_cache" \) -exec rm -rf {} + 2>/dev/null || true
      cd "$PREFIX/build/ctx" || exit 2
      export APPTAINER_TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/aimip-build.XXXXXX")
      export SINGULARITY_TMPDIR=$APPTAINER_TMPDIR; trap 'rm -rf "$APPTAINER_TMPDIR" "$PREFIX/build/ctx" "$SIF.incoming"' EXIT
      # Keep layer cache next to the build rather than under $HOME where HPC quotas are small.
      export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$PREFIX/build/cache} SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-$PREFIX/build/cache}
      echo "install-container: building $SIF (20-40 min; log $LOG)"
      # Bound mksquashfs worker threads so shared login nodes do not hit per-user thread limits.
      nice "$RTB" build --mksquashfs-args "-processors ${AIMIP_BUILD_THREADS:-4}" "$SIF.incoming" "$PREFIX/build/apptainer.def" > "$LOG" 2>&1 \
        || { rm -f "$SIF.incoming"; tail -20 "$LOG" >&2; die "image build failed"; }
      check_sif_version "$SIF.incoming" \
        || { rm -f "$SIF.incoming"; die "built image does not hold aimip $VERSION"; }
      mv "$SIF.incoming" "$SIF" || die "cannot move $SIF.incoming to $SIF"
      cd "$REPO" && rm -rf "$PREFIX/build/ctx"
      printf '%s\n' "local-build:$VERSION" > "$PREFIX/image.ref"
    else
      check_sif_version "$SIF" \
        || die "existing image $SIF does not hold aimip $VERSION (delete $SIF to rebuild or check out the matching release)"
      [ -f "$PREFIX/image.ref" ] || printf '%s\n' "existing-sif:$SIF" > "$PREFIX/image.ref"
    fi
    grep -E "AIMIP_IMAGE_OK|passed" "$LOG" 2>/dev/null | tail -2 || true
    ( cd "$PREFIX/images" && sha256_file "aimip-$VERSION.sif" | awk '{print $1}' ) > "$PREFIX/image.sha256" || die "cannot hash $SIF"
    rm -f "$PREFIX/image.stat"
    IMAGE=$SIF;;
  docker|podman)
    if [ "$BUILD" = 0 ]; then
      IMAGE=$IMAGE_REF
      echo "install-container: pulling $IMAGE (log $LOG)"
      "$RTB" pull "$IMAGE" > "$LOG" 2>&1 \
        || { tail -20 "$LOG" >&2; die "could not pull $IMAGE (a private image needs '$RTB login $REG_HOST'; or build it here with --build)"; }
      "$RTB" run --rm --entrypoint python "$IMAGE" -I -B -c 'import aimip, sys; sys.exit(0 if aimip.__version__ == sys.argv[1] else 1)' "$VERSION" \
        || die "the image $IMAGE does not hold aimip $VERSION (check out the matching release of this repository)"
      REPO_DIGEST=$("$RTB" image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null || true)
      if [ -n "$REPO_DIGEST" ] && [ "$REPO_DIGEST" != "$IMAGE_REF" ]; then
        printf '%s\n%s\n' "$IMAGE_REF" "$REPO_DIGEST" > "$PREFIX/image.ref"
      else
        printf '%s\n' "$IMAGE_REF" > "$PREFIX/image.ref"
      fi
    else
      IMAGE=aimip:$VERSION
      BA=(); [ -n "$BASE_ARG" ] && BA=(--build-arg "BASE_IMAGE=$BASE_ARG")
      echo "install-container: building $IMAGE (log $LOG)"
      "$RTB" build ${BA[@]+"${BA[@]}"} -f "$REPO/container/Dockerfile" -t "$IMAGE" "$REPO" > "$LOG" 2>&1 || { tail -20 "$LOG" >&2; die "image build failed"; }
      "$RTB" run --rm --entrypoint python "$IMAGE" -I -B -c 'import aimip, sys; sys.exit(0 if aimip.__version__ == sys.argv[1] else 1)' "$VERSION" \
        || die "built image $IMAGE does not hold aimip $VERSION"
      printf '%s\n' "local-build:$IMAGE" > "$PREFIX/image.ref"
    fi
    IMG_ID=$("$RTB" image inspect --format '{{.Id}}' "$IMAGE" | sed 's/^sha256://' | tr -d '[:space:]')
    [ -n "$IMG_ID" ] || die "cannot inspect $IMAGE"
    printf '%s\n' "$IMG_ID" > "$PREFIX/image.sha256" || die "cannot write $PREFIX/image.sha256"
    IMAGE="sha256:$IMG_ID";;
esac

# ---- the command
sed -e "s#@PREFIX@#$PREFIX#" -e "s#@RUNTIME@#$RT#" -e "s#@RUNTIME_BIN@#$RTB#" -e "s#@IMAGE@#$IMAGE#" \
    -e "s#@SCHEDULER@#$SCHED#" -e "s#@RUN_ROOT@#$RROOT#" -e "s#@EXTRA_RO@#$EXTRA#" "$REPO/container/aimip-wrapper.in" > "$PREFIX/bin/aimip.new" \
  && chmod 755 "$PREFIX/bin/aimip.new" \
  && mv "$PREFIX/bin/aimip.new" "$PREFIX/bin/aimip" \
  && cp "$REPO/container/VERSION" "$PREFIX/VERSION" \
  || die "cannot write $PREFIX/bin/aimip"
chmod -R a+rX "$PREFIX/bin" "$PREFIX/images" "$PREFIX/site.json" "$PREFIX/VERSION" "$PREFIX/image.sha256" "$PREFIX/image.ref" 2>/dev/null || true
echo "install-container: self-test"
# The self-test runs small fake executables and submits trial run directories through the wrapper. Its working directory
# must allow execution (TMPDIR may be mounted noexec) and must satisfy the wrapper's ancestry rule: no parent directory
# that is group- or world-writable without the sticky bit.
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
WTDIR=""
for c in "$PREFIX/build" "$SELFTEST_RUN_ROOT" "${TMPDIR:-}" /tmp; do
  [ -n "$c" ] && [ -d "$c" ] && selftest_dir_ok "$c" && { WTDIR=$(cd "$c" && pwd -P); break; }
done
[ -n "$WTDIR" ] || die "no directory for the wrapper self-test: $PREFIX/build, $SELFTEST_RUN_ROOT, \$TMPDIR and /tmp each have a group- or world-writable parent, characters the wrapper refuses, or forbid execution; rerun with TMPDIR=<a directory of your own>"
WT=$(TMPDIR=$WTDIR bash "$REPO/tests/wrapper/test_wrapper.sh" 2>&1) || { echo "$WT" | grep -E "FAIL|WRAPPER"; die "wrapper self-test failed (in $WTDIR)"; }
echo "$WT" | tail -1
"$PREFIX/bin/aimip" --version || die "self-test failed"
cat <<EOF

Installed AIMIP (container mode) $VERSION in $PREFIX
Next:
  $PREFIX/bin/aimip setup --models all        # download checkpoints + data, verify aimip.lock (~215 GB)
                                               # or skip it: the first 'aimip run -m X' downloads only model X
  $PREFIX/bin/aimip models
  $PREFIX/bin/aimip run -m ace2-era5 --start 1979-01-01T00:00:00 --steps 1460 --submit   # then: aimip submit <dir>
  $PREFIX/bin/aimip chat                      # Gemini key or Vertex credentials: see docs/agent.md
EOF
