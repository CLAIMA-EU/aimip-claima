#!/bin/bash
# End-to-end offline UI session test:
#   start `aimip ui` on ephemeral port -> curl models -> plan -> approve -> submit (fake sbatch) -> runs -> logs -> summary -> stop server
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd -P)
PY="${PYTHON_BIN:-python3}"
umask 022

RAW_W=$(mktemp -d "${TMPDIR:-/tmp}/aimip-ui-session.XXXXXX")
W=$(cd "$RAW_W" && pwd -P)
UI_PID=""
cleanup() {
  if [ -n "$UI_PID" ]; then
    kill "$UI_PID" 2>/dev/null || true
    wait "$UI_PID" 2>/dev/null || true
  fi
  rm -rf "$W"
}
trap cleanup EXIT

mkdir -p "$W/prefix/bin" "$W/data" "$W/runs" "$W/fakebin"

cat > "$W/prefix/bin/aimip" <<EOF
#!/bin/bash
exec "$PY" -B -m aimip "\$@"
EOF
chmod 755 "$W/prefix/bin/aimip"
printf '%s\n' "$("$PY" -c 'import sys; sys.path.insert(0, "'"$REPO"'/src"); import aimip; print(aimip.__version__)')" > "$W/prefix/VERSION"

cat > "$W/fakebin/nvidia-smi" <<'EOF'
#!/bin/bash
echo "GPU 0: Offline Test GPU"
EOF
cat > "$W/fakebin/sbatch" <<'EOF'
#!/bin/bash
echo 626200
EOF
cat > "$W/fakebin/squeue" <<'EOF'
#!/bin/bash
exit 0
EOF
cat > "$W/fakebin/sacct" <<'EOF'
#!/bin/bash
echo "626200|COMPLETED|00:00:45|0:0"
EOF
cat > "$W/fakebin/scancel" <<'EOF'
#!/bin/bash
exit 0
EOF
chmod 755 "$W/fakebin/"*

PYTHONPATH="$REPO/src" "$PY" - "$W" <<'PYSETUP'
import hashlib, json, pathlib, sys
import cftime, numpy as np, xarray as xr

w = pathlib.Path(sys.argv[1])
data = w / "data"
ckpt = data / "models" / "ace2-era5" / "ckpt.tar"
ckpt.parent.mkdir(parents=True)
ckpt.write_bytes(b"offline-ui-session-ckpt")
ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

ic_path = data / "models" / "ace2-era5" / "ic.nc"
dt = np.array([cftime.datetime(1979, 1, 1, 0, 0, 0, calendar="proleptic_gregorian")], dtype=object)
ds_ic = xr.Dataset(
    data_vars={
        "TMP2m": xr.DataArray(np.full((1, 4, 8), 285.0, dtype=np.float32), dims=("time", "latitude", "longitude")),
        "surface_temperature": xr.DataArray(np.full((1, 4, 8), 288.0, dtype=np.float32), dims=("time", "latitude", "longitude")),
        "time": xr.DataArray(dt, dims=("time",)),
    },
    coords={
        "latitude": xr.DataArray(np.array([-60.0, -20.0, 20.0, 60.0], dtype=np.float32), dims=("latitude",)),
        "longitude": xr.DataArray(np.linspace(0.0, 315.0, 8, dtype=np.float32), dims=("longitude",)),
    },
)
ds_ic.to_netcdf(str(ic_path), engine="netcdf4", encoding={"time": {"units": "hours since 1979-01-01 00:00:00", "calendar": "proleptic_gregorian"}})

fdir = data / "inputs" / "ace2-era5" / "amip-hist"
fdir.mkdir(parents=True)
(fdir / "forcing_1979.nc").write_bytes(b"forcing")

reg = {
    "schema_version": 3,
    "models": [
        {
            "name": "ace2-era5",
            "family": "fme",
            "fme_module": "fme.ace.inference",
            "checkpoint": str(ckpt),
            "checkpoint_sha256": ckpt_sha,
            "ic_files": [str(ic_path)],
            "forcing_sets": [{"name": "amip-hist", "dir": str(fdir), "window": ["1979-01-01T00:00:00", "2014-12-31T18:00:00"]}],
            "timestep_hours": 6,
            "stochastic": False,
            "default_n_ensemble_per_ic": 1,
            "notes": "UI session test model.",
        }
    ],
}
(data / "registry.json").write_text(json.dumps(reg, indent=2))
(w / "prefix" / "site.json").write_text(json.dumps({
    "name": "ci",
    "scheduler": "slurm",
    "runtime": "native",
    "sbatch": {"partition": "gpu", "account": "test_acct", "gres": "gpu:1", "cpus_per_task": 8, "mem": "32G", "max_time": "04:00:00"},
    "run_root": str(w / "runs"),
    "extra_ro_binds": [],
    "gpu_check": "nvidia-smi",
}, indent=2))
PYSETUP

export PATH="$W/fakebin:$PATH"
export PYTHONPATH="$REPO/src"
export AIMIP_HOME="$W/prefix"
export AIMIP_PLAYGROUND="$W/prefix"
export AIMIP_DATA="$W/data"
export AIMIP_REGISTRY="$W/data/registry.json"
export AIMIP_SITE_CONFIG="$W/prefix/site.json"
export AIMIP_DEFAULT_OUT_ROOT="$W/runs"
export AIMIP_RUN_ROOTS="$W/runs"
export AIMIP_STATE="$W/state"
export HF_HUB_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="127.0.0.1,localhost"

ucurl() { curl -q --noproxy '*' "$@"; }

echo "==> 1. start aimip ui on ephemeral port"
"$PY" -B -m aimip ui --host 127.0.0.1 --port 0 > "$W/ui.stdout" 2> "$W/ui.stderr" &
UI_PID=$!

BOOT_URL=""
for _ in $(seq 1 100); do
  if [ -s "$W/ui.stdout" ]; then
    BOOT_URL=$(sed -n 's/^AIMIP UI listening on //p' "$W/ui.stdout" | head -n 1)
    [ -n "$BOOT_URL" ] && break
  fi
  sleep 0.1
done
if [ -z "$BOOT_URL" ]; then
  echo "aimip ui failed to print startup URL" >&2
  cat "$W/ui.stderr" >&2 || true
  exit 1
fi

BASE_URL="${BOOT_URL%%\?*}"
BASE_URL="${BASE_URL%/}"
TOKEN="${BOOT_URL#*token=}"
TOKEN="${TOKEN%%&*}"
[ -n "$TOKEN" ]

echo "==> 2. verify CSP header, unauthenticated 401, and zero-cookie header token bootstrap"
ucurl -fsSI "$BASE_URL/" | grep -qi "Content-Security-Policy: default-src 'self'"
UNAUTH_CODE=$(ucurl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/v1/models")
[ "$UNAUTH_CODE" = "401" ]

COOKIE_CODE=$(ucurl -s -o /dev/null -w "%{http_code}" -H "Cookie: aimip_ui_token=$TOKEN" "$BASE_URL/api/v1/models")
[ "$COOKIE_CODE" = "401" ]

BOOT_HDRS="$W/boot.headers"
BOOT_CODE=$(ucurl -s -D "$BOOT_HDRS" -o /dev/null -w "%{http_code}" "$BOOT_URL")
[ "$BOOT_CODE" = "200" ]
! grep -qi "^Set-Cookie:" "$BOOT_HDRS"
grep -qi "^Cache-Control:.*no-store" "$BOOT_HDRS"

echo "==> 3. models -> plan -> approve -> submit"
ucurl -fsS -H "X-AIMIP-Token: $TOKEN" "$BASE_URL/api/v1/models?task=amip" | grep -q "ace2-era5"

PLAN_JSON=$(ucurl -fsS -H "X-AIMIP-Token: $TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"ace2-era5","start":"1979-01-01","length":"10d","name":"ui-demo"}' \
  "$BASE_URL/api/v1/plan")
PREFIX=$("$PY" -c 'import json,sys; d=json.loads(sys.argv[1]); assert d["ok"]; print(d["result"]["approval_prefix"])' "$PLAN_JSON")
FULL_DIGEST=$("$PY" -c 'import json,sys; d=json.loads(sys.argv[1]); assert d["ok"]; print(d["result"]["digest"])' "$PLAN_JSON")
[ ! -e "$W/runs/ui-demo" ]

APPROVE_JSON=$(ucurl -fsS -H "X-AIMIP-Token: $TOKEN" -H "Content-Type: application/json" \
  -d "{\"digest_prefix\":\"$PREFIX\",\"digest\":\"$FULL_DIGEST\",\"action\":\"submit\"}" \
  "$BASE_URL/api/v1/approve")
NONCE=$("$PY" -c 'import json,sys; d=json.loads(sys.argv[1]); assert d["ok"]; print(d["result"]["authorization_nonce"])' "$APPROVE_JSON")

SUBMIT_JSON=$(ucurl -fsS -H "X-AIMIP-Token: $TOKEN" -H "Content-Type: application/json" \
  -d "{\"nonce\":\"$NONCE\"}" \
  "$BASE_URL/api/v1/submit")
"$PY" -c 'import json,sys; d=json.loads(sys.argv[1]); assert d["ok"] and d["result"]["attempt_id"]=="626200"' "$SUBMIT_JSON"

printf '\033[32mINFO: ui step 40/40 complete\033[0m\n' > "$W/runs/ui-demo/slurm-626200.out"
PYTHONPATH="$REPO/src" "$PY" - "$W/runs/ui-demo" <<'PYOUT'
import pathlib, sys, cftime, numpy as np, xarray as xr
out = pathlib.Path(sys.argv[1])
vt = np.array([[cftime.datetime(1979, 1, 15, calendar="proleptic_gregorian")]], dtype=object)
ds = xr.Dataset(
    data_vars={
        "TMP2m": xr.DataArray(np.full((1, 1, 4, 8), 287.75, dtype=np.float32), dims=("sample", "time", "latitude", "longitude"), attrs={"units": "K"}),
        "counts": xr.DataArray(np.array([[40]], dtype=np.int32), dims=("sample", "time")),
    },
    coords={
        "time": xr.DataArray(np.array([0], dtype=np.int64), dims=("time",)),
        "latitude": xr.DataArray(np.array([-60.0, -20.0, 20.0, 60.0], dtype=np.float64), dims=("latitude",)),
        "longitude": xr.DataArray(np.linspace(0.0, 315.0, 8, dtype=np.float64), dims=("longitude",)),
        "valid_time": xr.DataArray(vt, dims=("sample", "time")),
    },
)
ds.to_netcdf(str(out / "monthly_mean_predictions.nc"), engine="netcdf4",
             encoding={"valid_time": {"units": "days since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian", "dtype": "int32"}})
PYOUT

echo "==> 4. runs -> logs -> summary (-> plot.png)"
ucurl -fsS -H "X-AIMIP-Token: $TOKEN" "$BASE_URL/api/v1/runs" | grep -q "ui-demo"
LOGS_JSON=$(ucurl -fsS -H "X-AIMIP-Token: $TOKEN" "$BASE_URL/api/v1/runs/ui-demo/logs")
"$PY" -c 'import json,sys; c=json.loads(sys.argv[1])["result"]["content"]; assert "INFO: ui step 40/40 complete" in c and "\x1b" not in c' "$LOGS_JSON"

SUMMARY_JSON=$(ucurl -fsS -H "X-AIMIP-Token: $TOKEN" "$BASE_URL/api/v1/runs/ui-demo/summary")
HAS_PLOT=$("$PY" -c 'import json,sys; d=json.loads(sys.argv[1])["result"]; assert abs(d["variables"]["TMP2m"]["series"][0]["ensemble_mean"] - 287.75) < 1e-4; print("1" if d.get("plot_available") else "0")' "$SUMMARY_JSON")
if [ "$HAS_PLOT" = "1" ]; then
  ucurl -fsS -H "X-AIMIP-Token: $TOKEN" "$BASE_URL/api/v1/runs/ui-demo/plot.png?var=TMP2m" > "$W/plot.png"
  "$PY" -c 'import pathlib,sys; b=pathlib.Path(sys.argv[1]).read_bytes(); assert b.startswith(b"\x89PNG\r\n\x1a\n")' "$W/plot.png"
fi

kill "$UI_PID" 2>/dev/null || true
wait "$UI_PID" 2>/dev/null || true
UI_PID=""

echo "UI_SESSION_OK"
