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
echo "WRAPPER_TESTS pass=$pass fail=$fail bash=$BASH_VERSION"
case "$W" in */aimip-wtest.*) rm -rf "$W";; esac
[ $fail = 0 ]
