#!/bin/bash
# Offline tests of container/aimip-wrapper.in with fake sbatch and container runtime.
# Runs under stock /bin/bash 3.2 as well as newer bash.
set -u
H=$(cd "$(dirname "$0")/../.." && pwd)
umask 022   # before any mkdir so trial run directories are never created group-writable
W=$(mktemp -d "${TMPDIR:-/tmp}/aimip-wtest.XXXXXX") || exit 2
case "$W" in */aimip-wtest.*) ;; *) echo "bad tmp dir $W"; exit 2;; esac
trap 'case "$W" in */aimip-wtest.*) rm -rf "$W";; esac' EXIT
P=$W/pre.v1; mkdir -p "$P/bin" "$W/fakebin" "$W/runs" "$W/home"
cat > "$W/fakebin/sbatch" <<'S'
#!/bin/bash
echo "$@" >> "$FAKE_LOG"; cat "${@: -1}" > "$FAKE_LOG.script"; echo 4242
S
cat > "$W/fakebin/fakert" <<'S'
#!/bin/bash
printf '%s\n' "$@" > "$FAKE_LOG.argv"; env > "$FAKE_LOG.env"; echo FAKERT_OK
S
chmod +x "$W/fakebin/sbatch" "$W/fakebin/fakert"
printf 'INITIAL_SIF_IMAGE' > "$P/img.sif"
if command -v sha256sum >/dev/null 2>&1; then sha256sum "$P/img.sif" | awk '{print $1}' > "$P/image.sha256"; else shasum -a 256 "$P/img.sif" | awk '{print $1}' > "$P/image.sha256"; fi
sed -e "s#@PREFIX@#$P#" -e "s#@RUNTIME@#apptainer#" -e "s#@RUNTIME_BIN@#$W/fakebin/fakert#" -e "s#@IMAGE@#$P/img.sif#" \
    -e "s#@SCHEDULER@#slurm#" -e "s#@RUN_ROOT@#$W/runs#" -e "s#@EXTRA_RO@##" "$H/container/aimip-wrapper.in" > "$P/bin/aimip"; chmod +x "$P/bin/aimip"
export PATH=$W/fakebin:$PATH FAKE_LOG=$W/log HOME=$W/home
pass=0; fail=0
ok()  { pass=$((pass+1)); echo "ok   $1"; }
bad() { fail=$((fail+1)); echo "FAIL $1"; }
mkjob() {  # mkjob <dir> <line1> <line2> <body...>
  local d=$1; shift; mkdir -p "$d"; rm -f "$d/slurm.jobid"; printf '%s\n' "$@" > "$d/job.sbatch"; }
ex() { echo "\"$P/bin/aimip\" exec-fme \"$W/runs/$1\""; }
umask 022
expect_submit() {  # expect_submit <name> <0|1> <dir>
  out=$("$P/bin/aimip" submit "$3" 2>&1); rc=$?
  if [ "$2" = 0 ] && [ $rc = 0 ]; then ok "$1"; elif [ "$2" = 1 ] && [ $rc != 0 ]; then ok "$1 (refused: ${out#aimip: })"; else bad "$1 rc=$rc out=$out"; fi; }
mkjob "$W/runs/r1" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --job-name=aimip-x' '#SBATCH --time=00:20:00' '#SBATCH --output='"$W/runs/r1"'/slurm-%j.out' '' 'set -euo pipefail' '' "$(ex r1)"
expect_submit "generated script is accepted" 0 "$W/runs/r1"
mkjob "$W/runs/r1h" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --cpus-per-task=32' '#SBATCH --hint=nomultithread' '#SBATCH --output='"$W/runs/r1h"'/slurm-%j.out' '' 'set -euo pipefail' '' "$(ex r1h)"
expect_submit "#SBATCH --hint=nomultithread accepted" 0 "$W/runs/r1h"
mkjob "$W/runs/r1m" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --hint=multithread' "$(ex r1m)"
expect_submit "any other #SBATCH --hint refused" 1 "$W/runs/r1m"
mkjob "$W/runs/r1eq" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --partition' "$(ex r1eq)"
expect_submit "#SBATCH key without = refused" 1 "$W/runs/r1eq"
mkjob "$W/runs/r2" '#!/usr/bin/env -S python3 -c print(314159)' '# aimip-generated v1' "$(ex r2)"
expect_submit "arbitrary shebang refused" 1 "$W/runs/r2"
mkjob "$W/runs/r3" '#!/bin/bash' '# aimip-generated v1' "\"$W/preXv1/bin/aimip\" exec-fme \"$W/runs/r3\""
expect_submit "regex-metachar prefix (pre.v1 vs preXv1) refused" 1 "$W/runs/r3"
mkjob "$W/runs/r4" '#!/bin/bash' '# aimip-generated v1' 'set -euo pipefail'
expect_submit "no exec line refused" 1 "$W/runs/r4"
mkjob "$W/runs/r5" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --wrap=id' "$(ex r5)"
expect_submit "#SBATCH --wrap refused" 1 "$W/runs/r5"
mkjob "$W/runs/r6" '#!/bin/bash' '# aimip-generated v1' "$(ex r6); id"
expect_submit "trailing command refused" 1 "$W/runs/r6"
mkjob "$W/runs/r7" '#!/bin/bash' '# aimip-generated v1' "\"$P/bin/aimip\" exec-fme \"$W/runs/../etc\""
expect_submit "dot-dot path refused" 1 "$W/runs/r7"
mkjob "$W/runs/r8" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --export=ALL' "$(ex r8)"
expect_submit "#SBATCH --export refused" 1 "$W/runs/r8"
# Run directory user check, permissions, and log/exec containment
mkjob "$W/runs/r10" '#!/bin/bash' '# aimip-generated v1' "$(ex r1)"
expect_submit "exec target outside the run dir refused" 1 "$W/runs/r10"
mkjob "$W/runs/r11" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --output=/tmp/x-%j.out' "$(ex r11)"
expect_submit "slurm log outside the run dir refused" 1 "$W/runs/r11"
mkdir -p "$W/outside_logs"
mkjob "$W/runs/r11a" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --output='"$W/runs/r11a"'/logs/slurm-%j.out' "$(ex r11a)"; ln -s "$W/outside_logs" "$W/runs/r11a/logs"
expect_submit "slurm log through escaping directory symlink refused" 1 "$W/runs/r11a"
mkjob "$W/runs/r11b" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --error='"$W/runs/r11b"'/slurm-%j.err' "$(ex r11b)"; ln -s "$W/outside_logs/victim.err" "$W/runs/r11b/slurm-%j.err"
expect_submit "symlinked slurm log file refused" 1 "$W/runs/r11b"
mkjob "$W/runs/r11c" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --output='"$W/runs/r11c"'/sub/slurm-%j.out' "$(ex r11c)"; mkdir -p "$W/runs/r11c/sub"
expect_submit "nested slurm log subdirectory refused" 1 "$W/runs/r11c"
mkjob "$W/runs/r11d" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --job-name=escape' '#SBATCH --output='"$W/runs/r11d"'/%x.out' "$(ex r11d)"; ln -s "$W/outside_logs/victim_x.out" "$W/runs/r11d/escape.out"
expect_submit "unsupported slurm %x log substitution refused" 1 "$W/runs/r11d"
mkjob "$W/runs/r11e" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --output='"$W/runs/r11e"'/slurm-%j.out' "$(ex r11e)"; ln -s "$W/outside_logs/victim_123.out" "$W/runs/r11e/slurm-123.out"
expect_submit "expanded slurm-%j.out symlink destination refused" 1 "$W/runs/r11e"
mkjob "$W/runs/r11f" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --output='"$W/runs/r11f"'/plain.out' "$(ex r11f)"
expect_submit "plain slurm log filename accepted" 0 "$W/runs/r11f"
mkjob "$W/runs/r12" '#!/bin/bash' '# aimip-generated v1' "$(ex r12)"; chmod g+w "$W/runs/r12"
expect_submit "group-writable run dir refused" 1 "$W/runs/r12"
mkjob "$W/runs/r13" '#!/bin/bash' '# aimip-generated v1' "$(ex r13)"; ln -s "$W/runs/r13" "$W/runs/r13link"
expect_submit "symlinked run dir refused" 1 "$W/runs/r13link"
mkjob "$W/runs/r14" '#!/bin/bash' '# aimip-generated v1' "$(ex r14)"; ln -s "$W/victim" "$W/runs/r14/slurm.jobid"
expect_submit "symlinked slurm.jobid refused (no write through it)" 1 "$W/runs/r14"
[ -e "$W/victim" ] && bad "wrote through the symlink" || ok "nothing written through the symlink"
mkjob "$W/runs/r15/spin" '#!/bin/bash' '# aimip-generated v1' "$(ex r15/spin/0)" "$(ex r15/spin/1)"
expect_submit "spin script with member exec lines accepted" 0 "$W/runs/r15/spin"
mkjob "$W/runs/r16" '#!/bin/bash' '# aimip-generated v1' '#SBATCH --output='"$W/runs/r16"'/../../x-%j.out' "$(ex r16)"
expect_submit "#SBATCH value with .. refused" 1 "$W/runs/r16"
mkdir -p "$W/runs/r17"; chmod g+w "$W/runs/r17"
"$P/bin/aimip" exec-fme "$W/runs/r17" >/dev/null 2>&1 && bad "exec-fme ran in a group-writable dir" || ok "exec-fme refuses a group-writable run dir at job time"
AIMIP_GEMINI_KEY_FILE=$W/nokey "$P/bin/aimip" chat >/dev/null 2>&1 && bad "missing key file accepted" || ok "missing key file refused"
AIMIP_GEMINI_VERTEX_TOKEN_FILE=$W/notoken "$P/bin/aimip" chat >/dev/null 2>&1 && bad "missing Vertex token file accepted" || ok "missing Vertex token file refused"
( umask 002; "$P/bin/aimip" --version >/dev/null 2>&1 )
grep -q "^umask g-w,o-w" "$P/bin/aimip" && ok "wrapper sets umask g-w,o-w" || bad "no umask in wrapper"
mkjob "$W/runs/r18" '#!/bin/bash' '# aimip-generated v1' "$(ex r18)"; echo "x: 1" > "$W/runs/r18/config.yaml"; chmod 0666 "$W/runs/r18/config.yaml"
expect_submit "0755 run dir with a 0666 config.yaml refused" 1 "$W/runs/r18"
mkdir -p "$W/open/sub"; chmod 0777 "$W/open"; mkjob "$W/open/sub/r19" '#!/bin/bash' '# aimip-generated v1' "\"$P/bin/aimip\" exec-fme \"$W/open/sub/r19\""
expect_submit "world-writable parent without sticky bit refused" 1 "$W/open/sub/r19"
chmod 1777 "$W/open"
expect_submit "world-writable parent WITH sticky bit accepted" 0 "$W/open/sub/r19"
mkdir -p "$W/shared/sub"; chmod 0777 "$W/shared"; ln -s "$W/shared/sub" "$W/runs/alias"
mkjob "$W/runs/alias/r20" '#!/bin/bash' '# aimip-generated v1' "\"$P/bin/aimip\" exec-fme \"$W/runs/alias/r20\""
expect_submit "caller-owned alias into a 0777 parent refused (physical ancestry)" 1 "$W/runs/alias/r20"
# sbatch receives the validated private copy rather than a mutable path inside the run directory
mkjob "$W/runs/r9" '#!/bin/bash' '# aimip-generated v1' "$(ex r9)"; : > "$FAKE_LOG"
"$P/bin/aimip" submit "$W/runs/r9" >/dev/null 2>&1
case "$(tail -1 "$FAKE_LOG")" in *"$W/runs/"*) bad "sbatch got a run-dir path";; *aimip-submit.*/job.sbatch*) ok "sbatch got the private copy";; *) bad "unexpected sbatch arg: $(tail -1 "$FAKE_LOG")";; esac
cmp -s "$FAKE_LOG.script" "$W/runs/r9/job.sbatch" && ok "copy content equals the checked script" || bad "copy differs"
# Credentials are never passed in argv, and only `chat` forwards them
GEMINI_API_KEY=SYNTHETIC_SECRET_123 "$P/bin/aimip" models >/dev/null 2>&1
grep -q SYNTHETIC_SECRET_123 "$FAKE_LOG.argv" && bad "key in argv (models)" || ok "key not in argv (models)"
grep -qE "^(APPTAINERENV|SINGULARITYENV)_GEMINI_API_KEY=" "$FAKE_LOG.env" && bad "key forwarded to a non-chat command" || ok "key not forwarded to non-chat command (only APPTAINERENV_ crosses --cleanenv)"
GEMINI_API_KEY=SYNTHETIC_SECRET_123 "$P/bin/aimip" chat >/dev/null 2>&1
grep -q SYNTHETIC_SECRET_123 "$FAKE_LOG.argv" && bad "key in argv (chat)" || ok "key not in argv (chat)"
grep -q "^APPTAINERENV_GEMINI_API_KEY=SYNTHETIC_SECRET_123" "$FAKE_LOG.env" && ok "chat gets the key via APPTAINERENV_" || bad "chat did not get the key"
# Empty arrays under set -u (bash 3.2)
"$P/bin/aimip" --version >/dev/null 2>&1 && ok "--version with no arrays filled" || bad "--version failed under $BASH_VERSION"
# --unsquash only when the installer wrote UNSQUASH=1 (its default on aarch64 hosts)
"$P/bin/aimip" models >/dev/null 2>&1
grep -qx -- "--unsquash" "$FAKE_LOG.argv" && bad "--unsquash passed without UNSQUASH=1" || ok "no --unsquash unless UNSQUASH=1"
sed -e "s#@PREFIX@#$P#" -e "s#@RUNTIME@#apptainer#" -e "s#@RUNTIME_BIN@#$W/fakebin/fakert#" -e "s#@IMAGE@#$P/img.sif#" \
    -e "s#@SCHEDULER@#slurm#" -e "s#@RUN_ROOT@#$W/runs#" -e "s#@EXTRA_RO@##" -e "s#@UNSQUASH@#1#" "$H/container/aimip-wrapper.in" > "$P/bin/aimip-unsq"
chmod +x "$P/bin/aimip-unsq"; rm -f "$FAKE_LOG.argv"
"$P/bin/aimip-unsq" models >/dev/null 2>&1
[ "$(sed -n 2p "$FAKE_LOG.argv" 2>/dev/null)" = exec ] && [ "$(sed -n 3p "$FAKE_LOG.argv" 2>/dev/null)" = --unsquash ] \
  && ok "UNSQUASH=1 runs the image with exec --unsquash" || bad "UNSQUASH=1 did not pass --unsquash after exec"
# Apptainer --env values must never contain commas (apptainer splits --env KEY=VAL on commas)
mkdir -p "$W/docbin"
for ucmd in id mkdir head tr cut dirname ls awk sed sha256sum shasum; do
  upath=$(command -v "$ucmd" 2>/dev/null || true)
  [ -n "$upath" ] && ln -s "$upath" "$W/docbin/$ucmd"
done
printf '#!/bin/bash\nexit 0\n' > "$W/docbin/sbatch"
printf '#!/bin/bash\necho "GPU 0: NVIDIA A100-SXM4-80GB, UUID: GPU-1234"\n' > "$W/docbin/nvidia-smi"
chmod +x "$W/docbin/sbatch" "$W/docbin/nvidia-smi"
PATH="$W/docbin" OMP_NUM_THREADS="8,16" "$P/bin/aimip" doctor >/dev/null 2>&1
grep -q "^AIMIP_HOST_SLURM_MISSING=squeue; sacct; scancel$" "$FAKE_LOG.argv" && ok "doctor missing slurm tools separated by '; '" || bad "unexpected AIMIP_HOST_SLURM_MISSING in argv"
awk 'prev=="--env" && index($0,",")>0 {found=1} {prev=$0} END {exit found}' "$FAKE_LOG.argv" && ok "no commas in any apptainer --env argument" || bad "comma leaked into apptainer --env argument"
# Installer: choice of the self-test directory (install-container.sh selftest_dir_ok / expand_run_root)
if [ -f "$H/install/install-container.sh" ]; then
  sed -n -e '/^selftest_dir_ok() {/,/^}/p' -e '/^expand_run_root() {/,/^}/p' "$H/install/install-container.sh" > "$W/selftest_fns.sh"
  . "$W/selftest_fns.sh"
  mkdir -p "$W/sel/plain" "$W/sel/with space" "$W/sel/gw/child"; chmod 777 "$W/sel/gw"
  selftest_dir_ok "$W/sel/plain" && ok "self-test dir: plain private dir accepted" || bad "self-test dir: plain dir refused"
  selftest_dir_ok "$W/sel/with space" && bad "self-test dir: path with a space accepted" || ok "self-test dir: path the wrapper refuses is skipped"
  selftest_dir_ok "$W/sel/gw/child" && bad "self-test dir: world-writable parent accepted" || ok "self-test dir: world-writable parent without sticky refused"
  chmod 755 "$W/sel/gw"
  u=$(id -un)
  [ "$(expand_run_root '/scratch/${U1}/${USER}/aimip-runs')" = "/scratch/${u:0:1}/$u/aimip-runs" ] && [ "$(expand_run_root '${HOME}/aimip-runs')" = "$HOME/aimip-runs" ] \
    && ok "self-test dir: run-root templates expanded like the wrapper" || bad "self-test dir: run-root template not expanded"
fi
# Wrapper `ui` and `ls` refresh `sacct.txt` in one batched `sacct` call for active runs and skip finished/terminal runs
mkdir -p "$W/uiruns/ui_1_to" "$W/uiruns/ui_2_can" "$W/uiruns/ui_3_done"
sed -e "s#@PREFIX@#$P#" -e "s#@RUNTIME@#apptainer#" -e "s#@RUNTIME_BIN@#$W/fakebin/fakert#" -e "s#@IMAGE@#$P/img.sif#" \
    -e "s#@SCHEDULER@#slurm#" -e "s#@RUN_ROOT@#$W/uiruns#" -e "s#@EXTRA_RO@##" "$H/container/aimip-wrapper.in" > "$P/bin/aimip-ui"
chmod +x "$P/bin/aimip-ui"
echo 9101 > "$W/uiruns/ui_1_to/slurm.jobid"; echo "9101|RUNNING|00:01:00|0:0" > "$W/uiruns/ui_1_to/sacct.txt"
echo 9102 > "$W/uiruns/ui_2_can/slurm.jobid"
echo 9103 > "$W/uiruns/ui_3_done/slurm.jobid"; printf '{"finished":"2026-11-16T00:00:00Z","rc":0}\n' > "$W/uiruns/ui_3_done/aimip-manifest.json"
cat > "$W/fakebin/sacct" <<'S'
#!/bin/bash
printf '%s\n' "$*" >> "$FAKE_LOG.sacct"
printf '9101|TIMEOUT|04:00:00|0:15\n9102|CANCELLED by 1000|00:02:10|0:15\n9103|COMPLETED|00:01:00|0:0\n'
S
chmod +x "$W/fakebin/sacct"
rm -f "$FAKE_LOG.sacct"
AIMIP_WRAPPER_SACCT_INTERVAL=0 "$P/bin/aimip-ui" ui >/dev/null 2>&1
[ -f "$FAKE_LOG.sacct" ] && [ "$(wc -l < "$FAKE_LOG.sacct" | tr -d ' ')" = "1" ] && ok "ui refreshes active runs in a single batched sacct call" || bad "ui sacct call count unexpected"
grep -q -- "-j 9101,9102 " "$FAKE_LOG.sacct" && ! grep -q "9103" "$FAKE_LOG.sacct" && ok "ui sacct batch queries active job IDs and skips finished manifest runs" || bad "unexpected job list in ui sacct batch"
grep -q "^9101|TIMEOUT|" "$W/uiruns/ui_1_to/sacct.txt" && ok "ui updates sacct.txt from RUNNING to TIMEOUT" || bad "ui did not update ui_1_to/sacct.txt"
grep -q "^9102|CANCELLED" "$W/uiruns/ui_2_can/sacct.txt" && ok "ui writes CANCELLED state to sacct.txt" || bad "ui did not write ui_2_can/sacct.txt"
"$P/bin/aimip-ui" ls >/dev/null 2>&1
[ "$(wc -l < "$FAKE_LOG.sacct" | tr -d ' ')" = "1" ] && ok "ls skips sacct once all runs are finished or terminal" || bad "ls queried sacct for terminal runs"
# Wrapper option-before-positional (`logs -f`), `approve` rw data mode, `--config` mount, and SIF/Docker image integrity
echo 9104 > "$W/uiruns/ui_4_follow/slurm.jobid" 2>/dev/null || { mkdir -p "$W/uiruns/ui_4_follow"; echo 9104 > "$W/uiruns/ui_4_follow/slurm.jobid"; }
cat > "$W/fakebin/sacct" <<'S'
#!/bin/bash
printf '9104|RUNNING|00:00:30|0:0\n'
S
chmod +x "$W/fakebin/sacct"
AIMIP_WRAPPER_SACCT_INTERVAL=0 "$P/bin/aimip-ui" logs -f ui_4_follow >/dev/null 2>&1
[ -f "$W/uiruns/ui_4_follow/sacct.txt" ] && ok "logs -f <name> refreshes sacct.txt when option precedes target" || bad "logs -f <name> missed sacct.txt"
mkdir -p "$P/data"
"$P/bin/aimip" approve deadbeef >/dev/null 2>&1
grep -q "^AIMIP_DATA_MODE=rw$" "$FAKE_LOG.argv" && ok "approve mounts writable data directory as rw" || bad "approve did not set AIMIP_DATA_MODE=rw"
mkdir -p "$W/cfgdir"; echo "model: ace2-era5" > "$W/cfgdir/exp.yaml"
( cd "$W/cfgdir" && "$P/bin/aimip" plan --config exp.yaml >/dev/null 2>&1 )
grep -q "^$W/cfgdir/exp.yaml:$W/cfgdir/exp.yaml:ro$" "$FAKE_LOG.argv" && ok "relative --config resolved and mounted :ro" || bad "relative --config not mounted :ro"
printf 'SIF_BYTES_ONE' > "$P/img.sif"
if command -v sha256sum >/dev/null 2>&1; then sha256sum "$P/img.sif" | awk '{print $1}' > "$P/image.sha256"; else shasum -a 256 "$P/img.sif" | awk '{print $1}' > "$P/image.sha256"; fi
ls -ln "$P/img.sif" 2>/dev/null | awk '{print $5"|"$6"|"$7"|"$8}' > "$P/image.stat"
cp -p "$P/img.sif" "$W/sif.ref"
mkdir -p "$W/runs/r_sif_ok"
"$P/bin/aimip" exec-fme "$W/runs/r_sif_ok" >/dev/null 2>&1 && ok "matching SIF sha256 accepted on exec-fme" || bad "matching SIF sha256 refused"
"$P/bin/aimip" run ace2-era5 --start 1979-01-01 --steps 1 --local >/dev/null 2>&1 && ok "matching SIF sha256 accepted on run --local" || bad "matching SIF sha256 refused on run --local"
printf 'SIF_BYTES_TWO' > "$P/img.sif"
touch -r "$W/sif.ref" "$P/img.sif" 2>/dev/null || true
rm -f "$FAKE_LOG.argv"
"$P/bin/aimip" exec-fme "$W/runs/r_sif_ok" >/dev/null 2>&1 && bad "replaced SIF accepted on exec-fme" || ok "replaced SIF refused on exec-fme"
"$P/bin/aimip" run ace2-era5 --start 1979-01-01 --steps 1 --local >/dev/null 2>&1 && bad "same-size/mtime replaced SIF accepted on run --local" || ok "same-size/mtime replaced SIF refused on run --local despite image.stat"
[ ! -e "$FAKE_LOG.argv" ] && ok "runtime never called on replaced SIF" || bad "runtime was called on replaced SIF"
printf 'SIF_BYTES_ONE' > "$P/img.sif"
rm -f "$P/image.sha256" "$FAKE_LOG.argv"
"$P/bin/aimip" run ace2-era5 --start 1979-01-01 --steps 1 --local >/dev/null 2>&1 && bad "missing image.sha256 accepted on run --local" || ok "missing image.sha256 refused on run --local"
[ ! -e "$FAKE_LOG.argv" ] && ok "runtime never called when image.sha256 is missing" || bad "runtime called when image.sha256 missing"
# Local detached submit fails closed when neither setsid nor python3 is available
if command -v sha256sum >/dev/null 2>&1; then sha256sum "$P/img.sif" | awk '{print $1}' > "$P/image.sha256"; else shasum -a 256 "$P/img.sif" | awk '{print $1}' > "$P/image.sha256"; fi
sed -e "s#@PREFIX@#$P#" -e "s#@RUNTIME@#apptainer#" -e "s#@RUNTIME_BIN@#$W/fakebin/fakert#" -e "s#@IMAGE@#$P/img.sif#" \
    -e "s#@SCHEDULER@#none#" -e "s#@RUN_ROOT@#$W/runs#" -e "s#@EXTRA_RO@##" "$H/container/aimip-wrapper.in" > "$P/bin/aimip-local"
chmod +x "$P/bin/aimip-local"
mkdir -p "$W/nosesbin"
for ucmd in bash id mkdir head tr cut dirname ls awk sed sha256sum shasum mktemp cat chmod mv rm rmdir basename sleep; do
  upath=$(command -v "$ucmd" 2>/dev/null || true)
  [ -n "$upath" ] && ln -s "$upath" "$W/nosesbin/$ucmd"
done
mkdir -p "$W/runs/r_noses"
printf '#!/bin/bash\n# aimip-generated v1\n\nset -euo pipefail\n"%s/bin/aimip" exec-fme "%s/runs/r_noses"\n' "$P" "$W" > "$W/runs/r_noses/job.sbatch"
noses_err=$(PATH="$W/nosesbin" "$P/bin/aimip-local" submit "$W/runs/r_noses" 2>&1); noses_rc=$?
if [ $noses_rc != 0 ] && case "$noses_err" in *"setsid or python3"*) true;; *) false;; esac; then
  ok "local submit fails closed without setsid/python3"
else
  bad "local submit without setsid/python3 rc=$noses_rc err=$noses_err"
fi
[ ! -e "$W/runs/r_noses/local.pid" ] && ok "no local.pid written when session launcher is missing" || bad "local.pid written despite missing session launcher"
echo "WRAPPER_TESTS pass=$pass fail=$fail bash=$BASH_VERSION"
case "$W" in */aimip-wtest.*) rm -rf "$W";; esac
[ $fail = 0 ]
