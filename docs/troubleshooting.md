# Troubleshooting

| what you see | cause and remedy |
|---|---|
| `path must be absolute` / `refusing path` | a relative path or a character outside letters, digits and `. _ / + -` ([paths](slurm-and-paths.md#paths)) |
| `run/output paths must live under <run root>` | `--out` points outside your run root |
| `run directory is group- or world-writable` | `chmod go-w <dir>` |
| `run directory is not owned by you` | you tried to submit or read somebody else's run; copy what you need into your run root |
| `Timestamp ... was not found in shipped IC file(s)` | without `--ic`, `--start` must be a shipped date; `aimip ic list -m <model>` lists them |
| `Refusing run: end time ... exceeds forcing set ...` | the run would end after the forcing does; the message gives the largest number of steps that fits |
| `Model X is deterministic and shipped IC ... provides only 1 initial state` | `--members K` needs K different states: make them with `ic spin` and `collect`, or pass a file with K states |
| `... --no-fetch was specified` | the model data are missing; run `aimip fetch -m X` |
| `model X is not installed here` | the same, in an install you do not own: ask its owner to run `aimip fetch -m X` |
| `forcing sets 1xCO2, 3xCO2 all cover this window` | a continued `ace2-som` run whose scenario cannot be read from the previous manifest; pass `--forcing-set` |
| `not an aimip-generated script` / `unexpected line` | `job.sbatch` was edited; prepare the run again with the options of `run` |
| `Could not fetch the ERA5 initial state` | DLESyM reads its start state from Google Cloud when the run is prepared; run `aimip run` where there is internet |
| `environment is missing` (DLESyM, ArchesClimate-SSP, ArchesWeather) | container: reinstall with `install/install-container.sh`; native: install with `--with-emu` |
| `the geoarches source is not installed` (ArchesWeather) | the install predates ArchesWeather support: reinstall as above |
| `site preset needs --account` | give `--account <account>` or `--account none` to the installer |
| the job ends as `TIMEOUT` | rerun the segment with a larger `--time` or fewer steps |
| `State: failed` | read the end of `slurm-<jobid>.err` (or `local-run.log` with the `local` preset) and `inference_out.log` in the run directory |
| exit code 5, `unstable_from` in the manifest (DLESyM) | the rollout became unstable; the months before it are kept ([models](models.md)) |
| every container command takes minutes | Apptainer unpacks the image on a slow file system: `export APPTAINER_TMPDIR=/tmp` |
| `cuFFT error: CUFFT_INTERNAL_ERROR` at the first step (aarch64, GH200) | the image is read from a FUSE-mounted SIF: reinstall with this version (the `aimip` command then unpacks the image for each call), or `export APPTAINER_UNSQUASH=1` ([aarch64 and GH200](installation.md#aarch64-and-gh200)) |
| image build stops at `mksquashfs` | too many threads on a shared login node; `AIMIP_BUILD_THREADS=2` before the installer |
