#!/bin/bash
# End-to-end offline console session test:
#   doctor -> models -> show -> cite -> plan -> run -> submit (fakes) -> watch -> logs -> ls -> summary -> completion
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="${PYTHON_BIN:-python3}"
umask 022

W=$(mktemp -d "${TMPDIR:-/tmp}/aimip-console-session.XXXXXX")
trap 'rm -rf "$W"' EXIT

mkdir -p "$W/prefix/bin" "$W/data" "$W/runs" "$W/fakebin"

cat > "$W/prefix/bin/aimip" <<EOF
#!/bin/bash
exec "$PY" -B -m aimip "\$@"
EOF
chmod 755 "$W/prefix/bin/aimip"
printf '%s\n' "$("$PY" -c 'import sys; sys.path.insert(0, "'"$REPO"'/src"); import aimip; print(aimip.__version__)')" > "$W/prefix/VERSION"

# Fake GPU & Slurm binaries for offline CI
cat > "$W/fakebin/nvidia-smi" <<'EOF'
#!/bin/bash
echo "GPU 0: Offline Test GPU"
EOF
cat > "$W/fakebin/sbatch" <<'EOF'
#!/bin/bash
echo 424200
EOF
cat > "$W/fakebin/squeue" <<'EOF'
#!/bin/bash
exit 0
EOF
cat > "$W/fakebin/sacct" <<'EOF'
#!/bin/bash
echo "424200|COMPLETED|00:01:12|0:0"
EOF
cat > "$W/fakebin/scancel" <<'EOF'
#!/bin/bash
exit 0
EOF
chmod 755 "$W/fakebin/"*

# Build a synthetic offline registry + IC + monthly dataset helper via Python
PYTHONPATH="$REPO/src" "$PY" - "$W" <<'PYSETUP'
import hashlib, json, pathlib, sys
import cftime, numpy as np, xarray as xr

w = pathlib.Path(sys.argv[1])
data = w / "data"
ckpt = data / "models" / "ace2-era5" / "ckpt.tar"
ckpt.parent.mkdir(parents=True)
ckpt.write_bytes(b"offline-session-ckpt")
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
            "notes": "Console session test model.",
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
export HF_HUB_OFFLINE=1

aimip() { "$PY" -B -m aimip "$@"; }

echo "==> 1. doctor"
aimip doctor | grep -q "Overall: ok"
aimip doctor --json | "$PY" -c 'import json,sys; d=json.load(sys.stdin); assert d["aimip"]=="1" and d["ok"] and d["result"]["ok"]'

echo "==> 2. models / show / cite"
aimip models --task amip --status certified | grep -q "ace2-era5"
aimip show ace2-era5 --examples | grep -q "ACE2-ERA5"
aimip cite ace2-era5 | grep -q "@"

echo "==> 3. plan (zero side effects)"
aimip plan ace2-era5 --start 1979-01-01 --length 10d --name demo | grep -q "40 steps"
[ ! -e "$W/runs/demo" ]

echo "==> 4. run --name demo"
aimip run ace2-era5 --start 1979-01-01 --length 10d --name demo --submit | grep -q "Next:     aimip submit"
[ -f "$W/runs/demo/plan.json" ]

echo "==> 5. submit demo"
aimip submit demo | grep -q "submitted job 424200"

# Populate synthetic job log + monthly output as if Slurm step finished
printf '\033[32mINFO: step 40/40 complete\033[0m\n' > "$W/runs/demo/slurm-424200.out"
PYTHONPATH="$REPO/src" "$PY" - "$W/runs/demo" <<'PYOUT'
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

echo "==> 6. watch / logs / ls / summary / completion"
aimip watch demo --interval 0.05 | grep -q "state=completed"
LOG_OUT=$(aimip logs demo)
printf '%s' "$LOG_OUT" | grep -q "INFO: step 40/40 complete"
case "$LOG_OUT" in *$'\033'*) echo "ANSI escape leaked in logs!" >&2; exit 1;; esac
aimip ls | grep -q "demo"
aimip summary demo --json | "$PY" -c 'import json,sys; d=json.load(sys.stdin); assert abs(d["result"]["variables"]["TMP2m"]["series"][0]["ensemble_mean"] - 287.75) < 1e-4'
aimip completion bash | grep -q "_aimip_completions"

echo "CONSOLE_SESSION_OK"
