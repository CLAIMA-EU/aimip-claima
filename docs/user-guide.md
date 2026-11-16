# User guide

From the first eight-step test to ensembles, continued runs and LoRA checkpoints, and the files a run leaves behind.
The model facts are in [models.md](models.md); job scripts, wall time and path rules in
[slurm-and-paths.md](slurm-and-paths.md).

In the examples, `A` is the command and `R` your run root. `$A site show` prints the run-root template, for example
`/scratch/${U1}/${USER}/aimip-runs`, where `${U1}` is the first letter of your user name:

```bash
A=/path/to/prefix/bin/aimip
R=/path/to/your/run/root
```

## How it works

- **Preparing and submitting are two steps.** `aimip run … --submit` writes a run directory with `config.yaml`,
  `job.sbatch` and `aimip-manifest.json`. Nothing is queued until you run `aimip submit <dir>`; you can read the job
  script first. `--local` runs at once on the current machine instead; `--dry-run` (the default) only prepares and
  validates.
- **Every run is recorded.** The manifest holds the command line, the sha256 of the checkpoint and of the initial
  state, the forcing files, the seed, the software versions, the exit code and the wall time.
- **Models are downloaded on first use** by the owner of the install ([installation.md](installation.md)).

## First run

```bash
$A models                               # the fourteen models, their date windows and initial conditions
$A ic list -m ace2-era5                 # 24 shipped start dates: the 1st of each month of 1979 and of 2001
$A run -m ace2-era5 --start 1979-01-01T00:00:00 --steps 8 --out $R/first --submit
$A submit $R/first
$A status $R/first                      # State: completed, Exit code: 0
$A summary $R/first
```

```
Summary for <R>/first (1 member(s), 1 month(s), cos(lat) global mean):
MONTH    COUNTS  TMP2m (K)  surface_temperature (K)
-------  ------  ---------  -----------------------
1979-01  8       285.1578   285.8980
```

`COUNTS` is the number of model steps in that month's mean; 8 of 124 makes this a partial month. From-zero installs in
both modes reproduce these two numbers exactly.

## Choosing model, start and length

- **Model:** [models.md](models.md) lists for each model whether it is deterministic or stochastic, its forcing
  window and its shipped initial conditions.
- **Start:** `--start YYYY-MM-DDTHH:MM:SS`. Without `--ic` it must be a shipped date (`aimip ic list -m <model>`);
  to start on another date make an initial state first (below). DLESyM and ArchesClimate-SSP follow their own rules
  ([models.md](models.md)).
- **Length:** `--steps` counts model steps (6 hours for most models, 5 days for SamudrACE, 1 month for
  ArchesClimate-SSP). The run must end inside the forcing window; `aimip run` refuses a run that does not and gives
  the largest number of steps that fits.
- **Output:** monthly means by default; `--prediction` also writes every step (about 6 GB per member and year),
  `--no-monthly` switches the monthly means off (fme models only).

In the recipes below, every `run … --submit` only prepares the run; queue it with `$A submit <its --out directory>`
(shown where the order matters). With `--local` instead of `--submit` the run starts at once.

## Deterministic ensemble: different initial states at the same date

```bash
$A ic spin -m ace2-era5 --from 1979-01-01T00:00:00,1979-02-01T00:00:00,1979-03-01T00:00:00 \
           --to 1979-03-02T00:00:00 --out $R/spin --submit
$A submit $R/spin                                   # one job runs all three spin-ups
$A status $R/spin                                   # wait for "Spin members: 3/3 completed"
$A collect $R/spin --out $R/ic-19790302.nc          # three different states, all dated 1979-03-02
$A run -m ace2-era5 --start 1979-03-02T00:00:00 --steps 1460 --members 3 --ic $R/ic-19790302.nc --out $R/ens3 --submit
$A submit $R/ens3
```

Each `--from` date must be before `--to`. Here the members have 60, 29 and 1 days of spin-up; start dates further
apart give more different states (the usual AMIP way to sample internal variability).

## Stochastic ensemble

The stochastic models (`ace2s-shield-plus`, `hiro-ace`, `ace2s`, `hiro-acev2`, `samudrace-e3smv3`, `dlesym-v1.1`) make
members from model noise; one initial state is enough:

```bash
$A run -m ace2s --start 2014-01-01T00:00:00 --steps 1460 --members 4 --seed 1 --out $R/ace2s-m4 --submit
```

The same seed gives the same run again (default 0). The members are along `sample` in the output files.

## Continue a run

Every fme run writes `restart.nc`, the state after its last step. Start the next segment from it at the time the
previous one ended (`end_time` in the manifest):

```bash
$A run -m ace2-era5 --start 1979-01-03T00:00:00 --steps 8 --ic $R/first/restart.nc --out $R/first-cont --submit
```

- This is also how to cross from one forcing set to the next (for example `ace2-shield` from 2014 into 2015–2024).
- For a stochastic model give each segment a **different** `--seed`; the same seed replays the noise.
- For a coupled model pass the previous run directory: `--ic $R/cm4-y1`.
- Long runs are best split into segments: the job wall time is capped, and a failed segment costs one segment.

## Scenarios (`ace2-som`, `archesclimate-ssp`)

```bash
$A run -m ace2-som --start 2032-01-01T00:00:00 --steps 1460 --out $R/som-1x --submit                       # 1xCO2
$A run -m ace2-som --forcing-set 3xCO2 --start 2032-01-01T00:00:00 --steps 1460 --out $R/som-3x --submit   # 3xCO2
$A run -m archesclimate-ssp --start 2015-01-01T00:00:00 --steps 1029 --forcing-set ssp585 --out $R/arches --submit
```

## Coupled atmosphere–ocean runs

```bash
$A run -m samudrace-cm4 --start 0311-01-01T00:00:00 --steps 73 --out $R/cm4-y1 --submit               # one model year
$A run -m samudrace-cm4 --start 0312-01-01T00:00:00 --steps 73 --ic $R/cm4-y1 --out $R/cm4-y2 --submit
$A run -m dlesym-v1.1 --start 1995-01-01T00:00:00 --steps 29220 --members 3 --seed 1 --out $R/dlesym --submit   # 20 years
```

Outputs are in `atmosphere/` and `ocean/`; `summary` adds the ocean columns. DLESyM runs cannot be continued.

## LoRA checkpoints

```bash
$A lora info  $R/lora/my_ckpt.tar                                        # LoRA settings and tensor norms
$A lora scale $R/lora/my_ckpt.tar --alpha 0.5 --out $R/lora/my_ckpt_a05.tar   # a copy with another strength
$A run -m ace2-era5 --start 1979-01-01T00:00:00 --steps 1460 --lora $R/lora/my_ckpt_a05.tar --out $R/lora-a05 --submit
```

`--lora` replaces the model's checkpoint with yours (fme models only); `lora scale` changes only the strength
(`--alpha 0` switches the LoRA off) and never overwrites a file. fme checkpoints are PyTorch pickles: load only
checkpoints from a source you trust.

## What is in a run directory

| file | content |
|---|---|
| `aimip-manifest.json` | the record of the run (see above) |
| `config.yaml` | the fme inference configuration (for DLESyM and ArchesClimate-SSP a JSON job description) |
| `job.sbatch` | the job script |
| `monthly_mean_predictions.nc` | **the main result**: monthly means of every output variable |
| `autoregressive_predictions.nc` | every step; only with `--prediction` |
| `restart.nc` | the state after the last step (fme models) |
| `slurm-<jobid>.out/.err`, `inference_out.log` | logs; look here first when a run fails |
| `slurm.jobid`, `sacct.txt` | the job id and the last scheduler record, used by `status` |
| `local.pid`, `local-run.log` | with the `local` preset: the process id and the log, instead of the Slurm files |

Coupled models write `atmosphere/` and `ocean/` subdirectories with their own files.

### Reading the monthly means

`monthly_mean_predictions.nc` has the dimensions `(sample, time, lat, lon)`; `sample` is the ensemble member. In
fme files `time` is a month counter, the date of each month is in `valid_time(sample, time)` and `counts(sample,
time)` is the number of steps averaged:

```python
import xarray as xr

ds = xr.open_dataset("monthly_mean_predictions.nc")
t2m = ds["TMP2m"]                                   # (sample, time, lat, lon), K
t2m = t2m.where(ds["counts"] >= 112)                # 6-hourly fme models: drop partial months
ens_mean = t2m.mean("sample")
dates = ds["valid_time"].isel(sample=0)
```

The `counts >= 112` mask is for the 6-hourly fme atmosphere models. In DLESyM and ArchesClimate-SSP files `time` is
the date itself (the first of each month) and `counts` is the number of samples (6-hourly for the DLESyM atmosphere,
48-hourly for its ocean, 1 for ArchesClimate-SSP); SamudrACE steps 5 days. Variable names follow each model (fme:
`TMP2m`, `PRATEsfc`, …; E3SM-trained: `TREFHT`, `TS`; DLESyM: `t2m`, `sst`; ArchesClimate-SSP: `tas`, `pr`); 
`ncdump -h` shows the list. A monthly file of `ace2-era5` takes about 16 MB per member and month.
