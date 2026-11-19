# Using aimip-claima

This guide assumes aimip-claima is installed and on your `PATH` ([installation](installation.md)). The command is
`aimip`. It has four parts:

1. [The idea in one minute](#the-idea-in-one-minute): the few words this guide uses.
2. [Your first run, step by step](#your-first-run-step-by-step): about 20 minutes, most of it waiting for the
   download and the queue.
3. [Recipes](#recipes): longer runs, ensembles, AIMIP runs, scenarios, coupled models, runs without Slurm.
4. [The results](#the-results): the files a run writes and how to read them.

Every command has a full description in `aimip <command> --help`. What each model can do (its dates, its initial
states, its known limits) is in [models.md](models.md).

## The idea in one minute

- **A model** is one of the emulators aimip-claima knows, for example `ace2-era5`. `aimip models` lists them and
  `aimip show <model>` describes one.
- **A run** is one simulation with one model: a start date, a length, and optionally several ensemble members.
- **A run has a name** (`--name first`) and a directory, `<run root>/first`. Everything about the run is in that
  directory: its settings, the job script, the log files and the results. Every command takes the name.
- **A run goes through four steps:**

  | step | command | what happens |
  |---|---|---|
  | check | `aimip plan` | checks the request and prints what would happen. Changes nothing, downloads nothing |
  | prepare | `aimip run` | downloads the model the first time, then writes the run directory. Nothing runs yet |
  | start | `aimip submit` | puts the job in the Slurm queue (or, without Slurm, starts it on this machine) |
  | follow | `aimip watch`, `aimip status`, `aimip summary` | shows the state of the job, then the results |

  You can skip `plan`: `aimip run` makes the same checks. To prepare and run at once on the current machine, use
  `aimip run … --local`.

## Your first run, step by step

This run simulates two days with ACE2-ERA5, the Ai2 atmosphere model trained on ERA5. It is short, so you can check
the whole chain in a few minutes of GPU time.

### Step 1. Look at the model

```bash
aimip show ace2-era5
```

```
ace2-era5 — ACE2-ERA5 (Ai2)
  status:      certified, verified on 1x NVIDIA A100-SXM4-80GB (8 steps, container == native bit-identical, 1.3.0/1.3.1)   (task: amip, family: fme)
  step/cal:    6h step, calendar=proleptic_gregorian, stochastic=False
  grid:        180 x 360 Gaussian (1 deg)
  forcing:     amip-hist (1979-01-01..2014-12-31)
  ic:          Shipped ERA5 initial conditions: 12 monthly timestamps in 1979 (ic_1979.nc) and 12 monthly timestamps in 2001 (ic_2001.nc) from allenai/ACE2-ERA5.
  outputs:     TMP2m, surface_temperature, PRATEsfc, PRESsfc
  installed:   no (0/39 files, ~20.52 GiB)
  ...
```

This tells you what you need for a run:

- **step/cal:** the model advances in 6-hour steps;
- **forcing:** it is driven by observed sea surface temperatures, which are available from 1979 to 2014, so the run
  must lie inside that window;
- **ic:** it can start from the 1st of any month in 1979 or 2001, where its authors provide an initial state;
- **installed:** it is not downloaded yet; the first run downloads 20.5 GiB.

### Step 2. Check the plan

```bash
aimip plan ace2-era5 --start 1979-01-01 --steps 8 --name first
```

```
PLAN first  [digest d69e646baba1]
  model:       ace2-era5 (fme, status=certified, task=amip)
  period:      1979-01-01T00:00:00 -> 1979-01-03T00:00:00  (8 steps, dt=6h, calendar=proleptic_gregorian)
  ic:          /big/disk/aimip/data/inputs/ace2-era5/ic/ic_1979.nc @ 1979-01-01T00:00:00 (...)
  ensemble:    1 member(s), seed=0  (Single trajectory from shipped IC (pending fetch).)
  forcing:     amip-hist
  inputs:      0 cached, 39 to fetch (~20.52 GiB from Hugging Face / Zenodo)
  outputs:     monthly means (monthly_mean_predictions.nc)
  resources:   slurm partition=gpu account=myproject gres=gpu:1 wall=00:20:00 (...)
  output:      /big/disk/aimip-runs/first
  limits:      Authors' limits (allenai/ACE2-ERA5): "not expected to generalize accurately for large perturbations ..."
  digest:      d69e646baba1  (approval prefix: d69e646b)
```

Read it from top to bottom: 8 steps of 6 hours take the model from 1 January to 3 January 1979; it starts from the
shipped initial state; 20.5 GiB will be downloaded; the job asks Slurm for one GPU for 20 minutes; the results go to
`/big/disk/aimip-runs/first`. The `limits` line quotes what the model's authors say it should not be used for. The
digest identifies this exact plan; you need it only for [approvals](console.md).

If something is wrong, `plan` says what and how to fix it. For example, a start date without an initial state:

```
aimip error: No shipped initial condition for 'ace2-era5' at 1979-01-15T00:00:00 (shipped timestamps:
1979-01-01T00:00:00, 1979-02-01T00:00:00, ...; pass `--ic <restart.nc>` or spin up with `aimip ic spin`).
  next:    aimip show ace2-era5
```

### Step 3. Prepare the run

```bash
aimip run ace2-era5 --start 1979-01-01 --steps 8 --name first
```

The first time, this downloads the model (20.5 GiB) and checks every file, which takes from a few minutes to an
hour depending on your connection. Then it prints the plan card again and:

```
Prepared ace2-era5 run in /big/disk/aimip-runs/first (mode=dry-run, forcing_set=amip-hist, start=1979-01-01T00:00:00, end=1979-01-03T00:00:00, steps=8, members=1, seed=0).
Config:   /big/disk/aimip-runs/first/config.yaml
Manifest: /big/disk/aimip-runs/first/aimip-manifest.json
Plan:     /big/disk/aimip-runs/first/plan.json
Sbatch:   /big/disk/aimip-runs/first/job.sbatch
Next:     aimip submit /big/disk/aimip-runs/first   (or: aimip submit first)
```

Nothing is running yet. You can look at the job script (`job.sbatch`) before you start it. (`aimip run … --submit`
does the same as `aimip run`; it does not submit by itself.)

### Step 4. Start it

```bash
aimip submit first
```

```
submitted job 1234567   (then: aimip status /big/disk/aimip-runs/first  or: aimip watch first)
```

Without Slurm (`--site local`), `submit` starts the job in the background on this machine and prints its process id.

### Step 5. Follow it

```bash
aimip watch first        # prints a line whenever the state changes, and returns when the job has ended
aimip status first       # the state once
aimip logs first         # the end of the job's log files
aimip ls                 # all your runs
```

`aimip status` shows, among other things:

```
Model:         ace2-era5 (forcing_set=amip-hist)
State:         completed
SLURM sacct:   state=COMPLETED elapsed=00:01:36 exit=0:0
Exit code:     0 (...)
```

A run is `prepared` until you start it, then `pending` while it waits in the queue, `running`, and finally
`completed`, `failed` or `cancelled`. If a run fails,
`aimip logs first` shows the end of its log; see also [troubleshooting](troubleshooting.md). `aimip cancel first`
stops a queued or running job.

### Step 6. Look at the result

```bash
aimip summary first
```

```
Summary for /big/disk/aimip-runs/first (1 member(s), 1 month(s), cos(lat) global mean):
MONTH    COUNTS  TMP2m (K)  surface_temperature (K)  air_temperature_7 (K)
-------  ------  ---------  -----------------------  ---------------------
1979-01  8       285.1577   285.8979                 282.9760
```

`summary` prints the area-weighted global mean of each month. `COUNTS` is the number of model steps in the month: 8
of the 124 six-hour steps of January, so this is a partial month. The output above is from a GH200; on an A100,
`TMP2m` is `285.1578 K`. If you get these numbers, your installation gives the same answer as ours.

The results themselves are NetCDF files in the run directory, mainly `monthly_mean_predictions.nc`; see
[The results](#the-results).

## Recipes

In the recipes, `aimip run` prepares the run and `aimip submit <name>` starts it. Use `aimip run … --local` instead
to run at once on the current machine.

### Choosing the start and the length

- **Start:** `--start 1979-01-01` or `--start 1979-01-01T06:00:00`. The start needs an initial state: for most
  models a shipped one (`aimip show <model>` lists them; after the model is downloaded, `aimip ic list -m <model>`
  lists the exact times), or your own file with `--ic`. Some models have their own rules, for example DLESyM
  starts from ERA5 at any 6-hourly time, and ArchesWeather only on 1978-09-30 ([models.md](models.md)).
- **Length,** one of:
  - `--steps 1460`: a number of model steps (6 hours for most models; 1 day for ArchesWeather; 1 month for
    ArchesClimate-SSP and MD-1.5; 5 days for the SamudrACE models and Samudra-OM4);
  - `--length 10y`, `6m`, `30d` or `12h`: converted to steps with the model's calendar;
  - `--end 1989-01-01`: the end time.
- **The run must end inside the forcing window.** If it does not, aimip-claima says how far you can go:

  ```
  aimip error: Refusing run: end time 2019-01-01T00:00:00 (...) exceeds forcing set 'amip-hist' end
  2014-12-31T18:00:00 (maximum steps in 'amip-hist' from 1979-01-01T00:00:00 is 52595; use
  `--end 2014-12-31T18:00:00` or `--steps 52595`; ...).
  ```

### Choosing what is written

Every run writes the monthly means of all its variables (`monthly_mean_predictions.nc`). For the ACE, DLESyM and
ArchesWeather models you can add files with another frequency, only the variables you need, and only part of the run, with `--output`. Repeat it
for several files:

```bash
aimip run ace2-era5 --start 1979-01-01 --length 10y --name era5-10y \
    --output daily:TMP2m,PRATEsfc \
    --output step:TMP2m@1979-01-01..1979-02-01
```

The plan card lists every file with its size, so you can see what a run will write before it starts:

```
  outputs:     monthly means (monthly_mean_predictions.nc)
               + output_daily.nc: daily means of TMP2m, PRATEsfc, whole run (3653 per member, ~1.76 GiB)
               + output_6h.nc: states at every model step (6 h) of TMP2m, 1979-01-01T00:00:00 .. 1979-02-01T00:00:00 (124 per member, ~31 MiB)
```

Each `--output` has the form `FREQUENCY[:VARIABLES][@START..END]`:

| part | examples | meaning |
|---|---|---|
| FREQUENCY | `step` | the state after every model step (every 6 hours for the ACE models) |
| | `12h`, `daily`, `5d` | means over periods of that length; any multiple of the model step |
| | `monthly` | monthly means |
| VARIABLES | `TMP2m,PRATEsfc` | the variables to write, as the model names them; without it, all variables. `ncdump -h monthly_mean_predictions.nc` of an earlier run lists the names |
| START..END | `1979-01-01..1979-02-01` | only this part of the run; not with `monthly` |

How the times work:

- **Each value covers one period that ends at a model step.** With 6-hour steps, the daily mean of 1 January is the
  mean of the states at 06, 12, 18 and 24 h, and `step` output starts with the state at 06 h: the starting state
  is not repeated in these files (it is in `initial_condition.nc`). The time of each value, in `valid_time`, is the
  middle of its period, for example 15:00 for a daily mean.
- **START..END selects whole periods:** `step:TMP2m@1979-01-01..1979-01-02` gives the four states of 1 January (06,
  12, 18 and 24 h). START and END must fall on period boundaries, counted from the start of the run.
- **The run must be a whole number of periods:** a daily file needs a whole number of days. aimip says which
  `--steps` would fit.

The files are called `output_<frequency>.nc` (`output_daily.nc`, `output_6h.nc`; a second daily file is
`output_daily_2.nc`). A variable the model does not have is left out of the file without an error from the model;
aimip then warns at the end of the job (`output file output_daily.nc without X`) and records it in
`aimip-manifest.json` (`outputs_check`). In a settings file, list the outputs under `outputs:`:

```yaml
outputs:
  - daily:TMP2m,PRATEsfc
  - step:TMP2m@1979-01-01..1979-02-01
```

For DLESyM and ArchesWeather:

- **DLESyM** (`dlesym-*`): the atmosphere variables (`t2m`, `z500`, `tcwv`, … as in its monthly file), every 6 hours
  or as means, on the 0.25° output grid. These files are large: one variable at every step is about 6 GB per member
  and year before compression, its daily means about 1.5 GB (the plan card shows the size). The ocean and the precipitation stay
  monthly.
- **ArchesWeather and ArchesWeatherGen**: each step is the daily-mean state of one day, stamped with its date (00
  UTC). Periods start on the first output day, so `step:tas@1979-01-01..1979-02-01` gives the 31 days of January,
  and `5d` means cover 1–5 October, 6–10 October, … for a run from 1978-09-30. The pressure-level variables (`ta`,
  `zg`, `ua`, `va`, `hus`, `wap`) keep their 13 levels. aimip-claima checks the variable names when it plans the run.

The other models (the coupled SamudrACE models, Samudra-OM4, MD-1.5, CAMulator, cBottle-3d, Spherical DYffusion)
write monthly means only for now, and `--output` works on the command line and in settings files, not yet in the web
interface, MCP or the assistant.
`--prediction` still writes every step of all variables (about 6 GB per member and year for ACE), and
`--no-monthly` switches the monthly file off.

### How long it takes, and long runs

The `resources` line of the plan card gives the expected GPU time and the job time aimip-claima asks for (three times the
expected time, at least 20–45 minutes, at most the site limit, usually 12 hours). Some examples on one A100:

| model | one model year |
|---|---|
| ACE and SamudrACE models | about 4 minutes per member |
| DLESyM | about 40 seconds per member |
| ArchesWeather / ArchesWeatherGen | about 7 / 28 minutes per member |

A run that does not fit in one job must be split into segments, each continued from the previous one (next recipe).
aimip-claima warns when a run will not fit. Details: [slurm-and-paths.md](slurm-and-paths.md#wall-time).

### Continuing a run

Most models write `restart.nc`, the state after the last step. Start the next segment from it, at the time the
previous one ended:

```bash
aimip run ace2-era5 --start 1979-01-01 --length 10y --name era5-a
aimip submit era5-a
# after it has finished:
aimip run ace2-era5 --start 1989-01-01 --length 10y --ic /big/disk/aimip-runs/era5-a/restart.nc --name era5-b
aimip submit era5-b
```

- For a stochastic model, give each segment a **different** `--seed`; the same seed repeats the same noise.
- For the coupled SamudrACE models and for ArchesWeather/ArchesWeatherGen, pass the previous run directory:
  `--ic /big/disk/aimip-runs/era5-a`. ArchesWeather segments give the same result as one long run; leave out
  `--members` and `--seed`, they come from the restart.
- DLESyM runs cannot be continued yet: a DLESyM run must fit in one job.
- The same way, a run can cross from one forcing set to the next, for example `ace2-shield` from 2014 into 2015–2024.

### Ensembles

**Stochastic models** (the `TYPE` column of `aimip models` says `stochastic`: for example `ace2s`, `hiro-acev2`,
`dlesym-v1.1`, `archesweathergen`) make members from their own random noise. One initial state is enough:

```bash
aimip run ace2s --start 2014-01-01 --length 1y --members 4 --seed 1 --name ace2s-m4
```

The members are along the `sample` dimension of the output files. The same seed gives the same members again.

**Deterministic models** need a different initial state for each member. aimip-claima refuses otherwise:

```
aimip error: Model 'ace2-era5' is deterministic: --members 3 requires a multi-sample IC (prepare one with
`aimip ic spin` -> `aimip collect`, or use --members 1).
```

Make the states by spinning up from different dates to a common date (the usual AMIP way to sample internal
variability):

```bash
aimip ic spin -m ace2-era5 --from 1979-01-01,1979-02-01,1979-03-01 --to 1979-03-02 --out /big/disk/aimip-runs/spin --submit
aimip submit /big/disk/aimip-runs/spin      # one job runs the three spin-ups
aimip status /big/disk/aimip-runs/spin      # wait for "Spin members: 3/3 completed"
aimip collect /big/disk/aimip-runs/spin --out /big/disk/aimip-runs/ic-19790302.nc
aimip run ace2-era5 --start 1979-03-02 --length 1y --members 3 --ic /big/disk/aimip-runs/ic-19790302.nc --name era5-ens3
```

Here the members have 60, 29 and 1 days of spin-up; start dates further apart give more different states.

### AIMIP Phase 1 runs

These models are set up as their authors ran them for AIMIP Phase 1 (1978-10 to 2024-12, observed sea surface
temperature and sea ice):

| model | command | time on one GPU |
|---|---|---|
| ArchesWeather | `aimip run archesweather --start 1978-09-30 --end 2024-12-31 --name aw-aimip` | about 5 hours (A100) |
| ArchesWeatherGen, one member | `aimip run archesweathergen --start 1978-09-30 --end 1993-12-31 --seed 0 --name awg-s0-1`, then two more segments ([continuing](#continuing-a-run)) | about 21 hours (A100) |
| DLESyM, member r1 | `aimip run dlesym-aimip --start 1978-10-03 --steps 67552 --name dla-r1` | 36 minutes (GH200) |

The authors' other DLESyM members start on 1978-10-04 to 1978-10-07; aimip-claima tells you the largest `--steps` for each.
For more ArchesWeatherGen members, use `--seed 1`, `2`, … How close each model comes to the authors' submission is in
[models.md](models.md). The +2 K and +4 K experiments are not available yet.

### Scenarios

Some models have several forcing sets (the `FORCING_SETS` column of `aimip models`). Choose one with
`--forcing-set`:

```bash
aimip run ace2-som --start 2032-01-01 --length 1y --name som-1x                         # 1xCO2 (the default)
aimip run ace2-som --start 2032-01-01 --length 1y --forcing-set 3xCO2 --name som-3x
aimip run archesclimate-ssp --start 2015-01-01 --steps 1029 --forcing-set ssp585 --name arches-585
```

### Coupled atmosphere–ocean models

```bash
aimip run samudrace-cm4 --start 0311-01-01 --steps 73 --name cm4-y1                     # one model year
aimip run samudrace-cm4 --start 0312-01-01 --steps 73 --ic /big/disk/aimip-runs/cm4-y1 --name cm4-y2
aimip run dlesym-v1.1 --start 1995-01-01 --length 20y --members 3 --seed 1 --name dlesym-20y
```

Their results are in `atmosphere/` and `ocean/` subdirectories; `aimip summary` shows both.

### Running without Slurm

On a workstation, or inside an interactive allocation with a GPU:

```bash
aimip run ace2-era5 --start 1979-01-01 --steps 8 --name first-local --local
```

The model runs at once in your terminal; `aimip summary first-local` works as before. With the `local` site preset,
`aimip submit` starts a prepared run in the background instead (its log is `local-run.log` in the run directory).

### Running on a cluster

The job settings come from your site settings (`aimip site show`). Any of them can be changed for one run:

```bash
aimip run ace2-era5 --start 1979-01-01 --length 5y --name era5-5y \
    --partition gpu-long --account otherproject --time 06:00:00 --gres gpu:a100:1
```

Also `--cpus-per-task`, `--mem` and `--job-name`. aimip-claima writes the job script itself and `aimip submit` sends only
scripts that aimip-claima wrote: to change a setting, prepare the run again instead of editing `job.sbatch`. More on job
scripts, wall time and paths: [slurm-and-paths.md](slurm-and-paths.md).

### Writing the settings in a file

For runs you repeat, put the settings in a YAML file:

```yaml
# era5-decade.yaml
model: ace2-era5
start: 1979-01-01
length: 10y
name: era5-decade
time: "02:00:00"
```

```bash
aimip run --config era5-decade.yaml
aimip run --config era5-decade.yaml --name era5-decade-2 --start 2001-01-01    # flags override the file
```

The keys are the option names (`model`, `start`, `steps`, `length`, `end`, `members`, `seed`, `ic`, `forcing_set`,
`name`, `out`, `outputs`, `partition`, `account`, `gres`, `cpus_per_task`, `mem`, `time`, …). Every run directory contains an
`experiment.yaml` with the settings it was prepared with.

### Your own checkpoint (LoRA)

```bash
aimip lora info  /big/disk/aimip-runs/lora/my_ckpt.tar                         # its LoRA settings
aimip lora scale /big/disk/aimip-runs/lora/my_ckpt.tar --alpha 0.5 --out /big/disk/aimip-runs/lora/my_ckpt_a05.tar
aimip run ace2-era5 --start 1979-01-01 --length 1y --lora /big/disk/aimip-runs/lora/my_ckpt_a05.tar --name lora-a05
```

`--lora` replaces the model's checkpoint with yours (the ACE and SamudrACE models only); `lora scale` writes a copy
with another strength (`--alpha 0` switches the LoRA off). Checkpoints are Python pickles: load only checkpoints
from a source you trust. Your own files (`--ic`, `--lora`) must be inside your run root or the installation
([paths](slurm-and-paths.md#paths)).

### The assistant, the web interface and approvals

`aimip chat` plans runs in conversation, `aimip ui` opens a web page with the same forms, and `aimip mcp` serves
aimip-claima to other AI tools. None of them can start a run by itself: a person confirms the exact plan first. See
[console.md](console.md) and [agent.md](agent.md).

## The results

### Files in a run directory

| file | content |
|---|---|
| `monthly_mean_predictions.nc` | **the main result:** monthly means of every output variable |
| `monthly_levels.nc`, `monthly_precip.nc` | some models: variables on pressure levels, or precipitation, in separate files |
| `output_<frequency>.nc` | the files you asked for with [`--output`](#choosing-what-is-written), for example `output_daily.nc` |
| `autoregressive_predictions.nc` | every model step; only with `--prediction` (about 6 GB per member and year for ACE) |
| `restart.nc` | the state after the last step, to [continue the run](#continuing-a-run) |
| `aimip-manifest.json` | the record of the run: the command line, the checksums of the checkpoint, the initial state and the forcing, the seed, the software versions, the exit code and the run time |
| `experiment.yaml`, `plan.json` | the settings and the plan the run was prepared with |
| `config.yaml`, `job.sbatch` | the model configuration and the job script aimip-claima wrote |
| `slurm-<jobid>.out`, `slurm-<jobid>.err`, `inference_out.log` | the logs; look here first when a run fails (`aimip logs <name>` shows their end) |
| `local-run.log` | without Slurm: the log |

The coupled models write `atmosphere/` and `ocean/` subdirectories, each with its own files.

### Reading the monthly means

The files have the dimensions `(sample, time, lat, lon)`, where `sample` is the ensemble member:

```python
import xarray as xr

ds = xr.open_dataset("/big/disk/aimip-runs/first/monthly_mean_predictions.nc")
t2m = ds["TMP2m"]                           # (sample, time, lat, lon), in K
t2m = t2m.where(ds["counts"] >= 112)        # 6-hourly models: drop months with fewer than 28 days of steps
ens_mean = t2m.mean("sample")
dates = ds["valid_time"].isel(sample=0)     # the date of each month
```

`counts` is the number of model states averaged in each month. Two things differ between models:

- **The time axis.** In the files of the ACE and SamudrACE models, `time` is a month counter and the date is in
  `valid_time`. In the other models' files, `time` is the date itself.
- **The variable names** follow each model: `TMP2m`, `PRATEsfc` (ACE); `TREFHT`, `TS` (models trained on E3SM);
  `t2m`, `sst` (DLESyM); `tas`, `pr` (ArchesClimate-SSP, ArchesWeather); `TMP2m`, `t_850`, `q_850` (MD-1.5). `ncdump -h
  <file>` lists them.

A monthly file of `ace2-era5` takes about 16 MB per member and month.

## Command overview

| command | what it does |
|---|---|
| `aimip models`, `aimip show <model>`, `aimip cite <model>` | list the models, describe one, print its references |
| `aimip plan …` | check a run without changing anything |
| `aimip run …` | prepare a run (`--local`: and run it now) |
| `aimip submit <name>` | start a prepared run |
| `aimip watch <name>`, `aimip status <name>`, `aimip logs <name>`, `aimip ls` | follow runs |
| `aimip summary <name>` | global-mean monthly values of a run |
| `aimip cancel <name>` | stop a queued or running job |
| `aimip ic list -m <model>`, `aimip ic spin …`, `aimip collect …` | initial states: list them, spin new ones up, collect them into one file |
| `aimip fetch -m <model>`, `aimip setup --models all` | download models in advance |
| `aimip bundle -m <models> --out <dir>` | the image, the models' files and the source in one directory, for a cluster without internet ([installation](installation.md#clusters-without-internet-for-example-marenostrum-5)) |
| `aimip doctor`, `aimip site show` | check the installation, show the settings |
| `aimip lora info/scale …` | inspect or rescale a LoRA checkpoint |
| `aimip chat`, `aimip ui`, `aimip mcp`, `aimip approve` | the assistant, the web interface, the MCP server, approvals ([console.md](console.md)) |
| `aimip completion bash` | tab completion for your shell |

Every command accepts `--json` for scripts: the result is printed as one JSON object.
