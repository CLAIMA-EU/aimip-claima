# Troubleshooting

Troubleshooting guide for aimip-claima. The command is `aimip`. Start with these two commands; they find most problems:

```bash
aimip doctor             # checks the installation; every failed check says what to do
aimip logs <run name>    # the end of a run's log files
```

Every error message of aimip-claima says what was wrong, and most end with a `next:` line: the command to try instead.

## While installing

| what you see | what to do |
|---|---|
| `site preset needs --account` (or `--partition`) | give the option to the installer: `--account myproject`, or `--account none` if your cluster has no accounts |
| `no container runtime found` | install Apptainer, Singularity, Docker or Podman, or use [native mode](installation.md#5c-native-mode-no-container) |
| the installer cannot fetch the image | development versions have no published image: add `--build`. For a release, check internet access to `ghcr.io` (and `https_proxy`) |
| the image build stops at `mksquashfs` | too many threads on a shared login node: run the installer again with `AIMIP_BUILD_THREADS=2` in front of it |
| every `aimip` command takes minutes ("Converting SIF file to temporary sandbox") | Apptainer unpacks the image on a slow file system: `export APPTAINER_TMPDIR=/tmp` |
| `environment is missing` (DLESyM, ArchesClimate-SSP, ArchesWeather, ...) | container mode: install again with the current version; native mode: install with `--with-emu` (or `--with-camulator`, `--with-sdyff`) |
| `the geoarches source is not installed` (ArchesWeather) | the installation is older than ArchesWeather support: install again |

## While preparing a run (`aimip plan`, `aimip run`)

| what you see | what to do |
|---|---|
| `No shipped initial condition for '<model>' at <date>` | the model can start only on the dates the message lists (also in `aimip show <model>`); or pass your own state with `--ic`, or make one with `aimip ic spin` |
| `Refusing run: end time ... exceeds forcing set ...` | the run would end after the forcing data do; the message gives the largest `--steps` or `--end` that fits |
| `--output '...' needs a whole number of days` (or of other periods) | the run must be a whole number of output periods: use one of the `--steps` values the message gives |
| `--output '...': START and END must fall on the boundaries ...` | the period of an output must begin and end on its period boundaries, counted from the start of the run ([choosing what is written](user-guide.md#choosing-what-is-written)) |
| `--output is not available for <model> yet` | only the ACE models support `--output` so far; the other models write monthly means |
| `Model '<model>' is deterministic: --members K requires a multi-sample IC` | a deterministic model needs a different initial state per member: see [ensembles](user-guide.md#ensembles) |
| `forcing sets 1xCO2, 3xCO2 all cover this window` | the model has several forcing sets for these dates: choose one with `--forcing-set` |
| `model X is not installed here: ask the install owner to run 'aimip fetch -m X'` | someone else owns the installation; only the owner can download models |
| `... --no-fetch was specified` | the model is not downloaded yet: run `aimip fetch -m <model>` (or leave out `--no-fetch`) |
| `Could not fetch the ERA5 initial state` (DLESyM) | DLESyM downloads its start state when the run is prepared: run `aimip run` where there is internet |
| a download hangs or fails with a network error | the machine cannot reach the internet: download the model elsewhere and copy it in ([clusters without internet](installation.md#clusters-without-internet-for-example-marenostrum-5)), and use `--no-fetch` |
| `path must be absolute`, `refusing path` | use absolute paths with only letters, digits and `. _ / + -`; no spaces, no `~`, no `..` |
| `run/output paths must live under <run root>` | `--out` points outside your run root: use `--name` instead, or a path under the run root |
| `path will not be visible inside the container` | your `--ic` or `--lora` file is outside the run root and the installation: copy it into your run root |
| `Unknown model '<model>'` from `aimip ic list` | `ic list` reads the files of downloaded models: download the model first (`aimip fetch -m <model>`), or use `aimip show <model>` |

## While starting a run (`aimip submit`)

| what you see | what to do |
|---|---|
| `not an aimip-generated script`, `unexpected line` | `job.sbatch` was edited. aimip-claima submits only the scripts it wrote: prepare the run again with the options you need (`--time`, `--partition`, ...) |
| `run directory is group- or world-writable` | `chmod go-w <run directory>` |
| `run directory is not owned by you` | you tried to start somebody else's run: prepare your own |
| `plan changed, approve again` | something in the run (a file, a resource) changed after it was approved: approve the new plan ([console.md](console.md)) |

## While a run is running, or after it failed

| what you see | what to do |
|---|---|
| `State: failed` | `aimip logs <name>` shows the end of the logs; the full logs are `slurm-<jobid>.err`, `slurm-<jobid>.out` and `inference_out.log` (without Slurm: `local-run.log`) in the run directory |
| the job ends as `TIMEOUT` | the run was longer than the job time: prepare it again with a larger `--time`, or split it into [segments](user-guide.md#continuing-a-run) |
| `aimip warning: output file output_... without X` at the end of the job | the model has no variable `X` (a typing error, or another name): `ncdump -h monthly_mean_predictions.nc` lists the model's names |
| exit code 5, and `unstable_from` in `aimip-manifest.json` | the model became unstable (values out of range); aimip-claima stopped it and kept the months before. This is a property of the model ([models.md](models.md)), not an installation problem |
| `cuFFT error: CUFFT_INTERNAL_ERROR` at the first step (GH200, aarch64) | the image is read from a mounted file: install again with the current version, which unpacks it, or `export APPTAINER_UNSQUASH=1` ([aarch64](installation.md#special-situations)) |
| `CUDA out of memory` | too many members for one GPU: fewer `--members` per run, or a GPU with more memory |

## Still stuck

Open an issue at <https://github.com/CliDyn/aimip-claima/issues> with the command you ran, the output of
`aimip doctor` and the end of the log (`aimip logs <name>`). Remove account names and paths you do not want to
share.
