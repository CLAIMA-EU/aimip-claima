# Models

aimip-claima registers the models below. The command is `aimip`. `aimip models` prints the same list from your install,
with the date windows and initial conditions.

| model | type | forcing / date window | shipped initial conditions | ensembles |
|---|---|---|---|---|
| `ace2-era5` | deterministic, trained on ERA5 | 1979–2014 | the 1st of each month in 1979 and in 2001 (ERA5 states) | different initial conditions (`ic spin`) |
| `ace2-shield` | deterministic, trained on SHiELD | 1979–2014; 2014–2024 with AIMIP SST; julian | one state, 1979-01-01 (from the ACE2S-SHiELD-plus package; ACE2-SHiELD ships none) | none from the shipped state |
| `ace2s-shield-plus` | stochastic | 1979–2014; 2014–2024; julian | one state, 1979-01-01 | `--members K` (model noise) |
| `ace2-eamv3` | deterministic, trained on E3SMv3 | 1994–2014; noleap calendar | one state stamped 1994-01-01 (**a 1971 model state**) | none from the shipped state |
| `hiro-ace` | stochastic; trained only on 9 years of X-SHiELD (2014–2022) | 2014–2024 (training period 2014–2022); 1994–2014 is our own extension outside the training period; julian | one state stamped 1994-01-01 (**a 2014 model state**) | `--members K` |
| `ace2-som` | deterministic, slab ocean | scenarios `1xCO2` (default) and `3xCO2`, nominal years 2032–2035 | one state per scenario, 2032-01-01 | pass several states with `--ic` |
| `ace2s` | stochastic (the `hiro-ace` checkpoint with the authors' forcing; trained on X-SHiELD 2014–2022) | 2014–2023 (the training period); julian calendar | `ic_2014.nc`, `ic_2023.nc`, 12 states each | `--members K` |
| `hiro-acev2` | stochastic, ACE2S-X-SHiELD global stage; `fme` 2026.10.0 | 2014–2023 (the training period); julian | `ic_2014.nc`, `ic_2023.nc`, 12 states each | `--members K` |
| `samudrace-cm4` | **coupled** atmosphere–ocean, deterministic, CM4 preindustrial control | `picontrol-0311`: labels 0311–0410, `picontrol-0151`: 0151-01-06 … 0250-12-27 (one forcing year repeated); noleap | five states 0311 … 0319, one state 0151-01-06 | different initial conditions (`ic spin`) |
| `samudrace-e3smv3` | **coupled**, stochastic; `fme` 2026.10.0 | one year, 0425-01-03 … 0425-12-29 (**at most 72 steps**); noleap | three states in one file | `--members K` |
| `dlesym-v1.0` | **coupled** atmosphere–ocean (HEALPix-64, output 0.25°), deterministic; Earth2Studio | not forced; any 6-hourly start 1940-01-04 … 2023-12-31 | ERA5 at `--start`, downloaded by `aimip run` | — |
| `dlesym-v1.1` | coupled, stochastic; adds 6-hour precipitation | as above | as above | `--members K` (noise seed + k) |
| `dlesym-v0-isccp` | coupled, deterministic | as above | as above | — |
| `dlesym-aimip` | AIMIP version: `dlesym-v0-isccp` with the SST prescribed from the AIMIP forcing, deterministic | AIMIP SST 1978-10 … 2024-12; `--start` 1978-10-03 … 2024-12-28 | ERA5 at `--start` (the authors' members: 1978-10-03 … 07, 00 UTC) | one run per start |
| `archesclimate-ssp` | monthly emulator of IPSL-CM6A-LR (144 × 144: 1.25° lat × 2.5° lon; atmosphere and upper-ocean temperature), deterministic | scenarios `ssp434` (default) and `ssp585`; outputs March 2015 … November 2100 | the scenario's 2015 state (`--start 2015-01-01T00:00:00`) | — |
| `archesweather` | ArchesWeather-Mx4, AIMIP version: four networks averaged at every step, deterministic, trained on ERA5 1979–2013 (validation 2014, test 2015–2024; 1°, 13 pressure levels, daily means) | monthly ERA5 SST and sea ice, 1978-10-01 … 2024-12-31 | the ERA5 daily mean of 1978-09-30 (`--start 1978-09-30T00:00:00`) | — |
| `archesweathergen` | ArchesWeatherGen, AIMIP version: stochastic (flow matching around ArchesWeather-Mx4; trained on ERA5 1979–2013, validation 2014, test 2015–2024) | as above | as above | `--members K` (noise seed + k) |
| `samudra-om4` | 1° global **ocean** emulator of GFDL OM4 (180 × 360, 19 depths 2.5–6000 m), deterministic | OM4 JRA55-do OMIP-2 surface fluxes 1975-01-08 … 2022-12-29 (julian; independent test period starts 2014-09-30T12:00:00) | two 5-day OM4 states ending at `--start`, staged from OSN by `aimip run` | — |
| `md-1.5` | 1.5° monthly latent-diffusion atmosphere emulator (121 × 240, 37 prognostic channels; `MD-1p5-latest.pth`), stochastic | AIMIP Phase 1 monthly SST/SIC (Zenodo 17065758, 556 month-start-centred points 1978-10-01 … 2025-01-01; training 1985–2014, test 2015–2022) | none published upstream — pass `--ic <era5_monthly.nc>` or stage `$AIMIP_DATA/inputs/md-1.5/ic/ic_<YYYY-MM-DD>.nc` | `--members K` (15 reverse DDPM steps/month) |
| `camulator` | 6-hourly 1° 32-level atmosphere emulator of CAM6 / ERA5-scaled climate target (MILES/CREDIT), deterministic | `camulator-cyclic-1yr` (1980–2100) and `camulator-branch-1980-2014` (1980–2014); noleap | 69 published noleap tensors (1980-01-01, Jan 1 & Jul 1 of 1981–2014) | — |
| `cbottle-3d` | conditional atmospheric **state generator** (not a time-stepping simulation; Earth2Studio `CBottle3D`, output 0.25°) | PCMDI-AMIP-1-1-9 monthly mid-point SST 1940-01-01 … 2022-12-16 (training range 1980–2017; no CO2 or sea-ice input) | none (independent diffusion draws given day-of-year, hour-of-day, and SST) | `--members K` (SHA-256 per member & batch) |
| `spherical-dyffusion` | 6-hourly 1° probabilistic atmosphere emulator of FV3GFS (180 × 360, 8 layers), stochastic | repeating 365-day annual SST cycle (labelled 2021 in reference files; nominal cycle months `0001-01…` in outputs); noleap | 1° FV3GFS reference simulation (Zenodo 10791087, 2021-01-01 … 2021-12-31) | `--members K` |

## Steps

| models | one step | one year |
|---|---|---|
| the eight `fme` atmosphere models, `dlesym-*`, `camulator`, `spherical-dyffusion` | 6 hours | 1460 steps (1464 in a leap year with a Gregorian or Julian calendar) |
| `samudrace-cm4`, `samudrace-e3smv3`, `samudra-om4` | 5 days | 73 steps (`samudra-om4` takes even `--steps >= 2`) |
| `archesweather`, `archesweathergen` | 1 day (daily-mean states) | 365 or 366 steps (at most 16894: 1978-10-01 … 2024-12-31) |
| `cbottle-3d` | 1 day (`daily_synoptic`: 00/06/12/18 UTC per day) | 365 or 366 steps |
| `archesclimate-ssp`, `md-1.5` | 1 month (`md-1.5` outputs one single monthly state per step, `counts = 1`) | 12 steps (`archesclimate-ssp` up to 1029; `md-1.5` up to 556) |

## Notes per model family

**Ai2 `fme` models.** Checkpoints, initial conditions and forcing come from Hugging Face (`allenai/*`, revisions
pinned in the lock file). `hiro-acev2` and `samudrace-e3smv3` need `fme` 2026.10.0; both versions are installed and
each run records the one it imported. For these two the seed is written into the configuration, so a run with the
same seed repeats exactly.

**Initial conditions of `ace2-eamv3` and `hiro-ace`.** The authors ship no 1994 state, so the 1994-01-01 states are
model states from another year (EAMv3: 1971-01-01; HiRO: the first state of 2014) with the date changed. The first
model year carries a spin-up from a mismatched state: leave out 1994 in an analysis.

**Deterministic members.** `ic spin` needs two or more shipped start dates, and two states of one deterministic run
carried to the same date are the same state. `ace2-shield`, `ace2-eamv3` and `ace2-som` therefore cannot make members
from their shipped states alone: pass a file with several different states (`--ic`), or use a stochastic model.

**`ace2-som`** runs two equilibrium climates; the years are labels. Choose the scenario with `--forcing-set 3xCO2`.
A continued run takes the scenario from the previous run's manifest, and refuses if it cannot tell.

**SamudrACE (coupled).** Outputs are in `atmosphere/` and `ocean/` inside the run directory; continue with
`--ic <previous run directory>`. `samudrace-cm4` is a preindustrial control, about 1 K colder than 1995–2014.
`samudrace-e3smv3` ships one year of forcing, so a run stops at 72 steps and cannot be continued.

**DLESyM** (Earth2Studio). Not forced, so any start with an ERA5 state works; `aimip run` downloads that state from
the public ARCO-ERA5 bucket (about one minute, 0.4 GB, into the run directory). A run cannot be continued: write the
whole period in one job (`--steps 29220` = 20 years, about 15 GPU-minutes per member). The authors built `dlesym-v1.0`
and `dlesym-v1.1` for forecasts of weeks to months; multi-year runs are outside what they validated, so check long runs for drift. `dlesym-v0-isccp` is
the checkpoint Earth2Studio documents for 100–1000-year climate runs. A run stops with exit code 5 at the first month
with non-finite values or a global-mean 2 m temperature outside 200–330 K, records `unstable_from` and keeps the
months before it. DLESyM's `sst` has values over land: mask land before use. `dlesym-v1.1` writes
`atmosphere/monthly_precip.nc`; its `tp06` (6-hour precipitation) is derived with ε = 1e-8, inferred from the
authors' pipeline, and its 3-hour OLR input is built as Earth2Studio does.

`dlesym-aimip` is the AIMIP Phase 1 setup of the same authors (code: `nathanielcresswellclay/DLESyM_aimip-2026`): the
`dlesym-v0-isccp` checkpoints, but the ocean model's SST is replaced at every step by the AIMIP forcing (Zenodo
17065758), interpolated linearly in time between its monthly stamps (the first of each month), as their "ground-truth"
ocean does. Land cells get the model's own SST interpolation and the field is mapped to HEALPix like the initial state.
The authors' five members start from ERA5 at 00 UTC on 1978-10-03, 04, 05, 06 and 07, one run each; `--steps 67552`
from 1978-10-03 is their run length (to 2024-12-28; at most 67568, to 2025-01-01). Checked against their submission:
2-m temperature within 0.1 K (RMSE) of their member r1 on days 2–3, diverging afterwards as weather does; over
1979–2024 the global-mean 2-m temperature is 287.32 K (their four members: 287.32–287.33 K), its trend 0.167 K per
decade (theirs: 0.165–0.172), and the mean seasonal cycle differs by at most 0.04 K. A full run takes about 36 minutes
on one GH200. The model was trained on 1983–2016, part of the AIMIP evaluation period. No +2 K / +4 K experiments yet.

**ArchesClimate-SSP.** `--steps` counts months. The run always starts from the scenario's 2015 state; the first
output month is March 2015. It emulates one climate model (IPSL-CM6A-LR) under one scenario forcing, not observed
weather. Outputs: `atmosphere/monthly_mean_predictions.nc` (`tas`, `psl`, `pr`, …), `atmosphere/monthly_levels.nc`
(17 pressure levels), `ocean/monthly_mean_predictions.nc` (`thetao_0`, the top level, °C) and
`ocean/monthly_thetao.nc` (`thetao` on 10 depths, °C). Pressure levels below the ground (mostly 1000–700 hPa
over land) and ocean values over land are NaN, with a mask that does not change in time.

**ArchesWeather and ArchesWeatherGen.** These are the versions the INRIA authors retrained for AIMIP (Singh et al.,
"Evaluating Skill and Stability of ArchesWeather and ArchesWeatherGen under Multi-Decadal Climate Simulations",
arXiv:2605.29976; trained on ERA5 1979–2013, validation 2014, test 2015–2024, so most of the AIMIP period is in-sample),
with checkpoints, initial state and forcing from Zenodo record 22795233. One step is one day of
daily-mean states on the 1° grid; the model sees the two previous days and the monthly SST and sea-ice cover of the
predicted day's month. `archesweather` averages the four ArchesWeather-M networks at every step, as the authors do in
their AIMIP runs. `archesweathergen` draws each day with 25 flow-matching steps around that average; members differ by
the noise seed (step i of member k uses seed + k + 1000 i, the authors' scheme — keep `--seed` values of separate runs
less than 1000 apart so they do not reuse daily noise). The authors' long-rollout script is
not published, so aimip-claima follows section 2.3 of the paper: predicted SST and sea ice over land (where the initial
state has no ocean value) are set to zero before they are fed back, and written as NaN. The only initial state is
the ERA5 daily mean of 1978-09-30 (the authors' member r2; their other members start on 1978-09-29 and
1978-10-01 … 03). Outputs: `atmosphere/monthly_mean_predictions.nc` (`tas`, `psl`, `uas`, `vas`, and the model's own
daily `sst` and `sic` averaged per month) and `atmosphere/monthly_levels.nc` (`zg`, `ua`, `va`, `ta`, `hus`, `wap`
on 13 pressure levels, `zg` = geopotential / 9.80665). On one A100 a whole run 1978-10 … 2024-12 takes about
5 hours for `archesweather` and about 21 hours per member for `archesweathergen` (about 28 minutes per member-year).
Every run writes `restart.nc` (the states of its last two days, and the running sums of a month it ended in), and
`--ic <run directory>` continues it with the same result as one longer run, so long `archesweathergen` runs go in
segments ([user-guide.md](user-guide.md#continuing-a-run)). The ArchesWeatherGen checkpoint also stores one
deterministic network of its own; like geoarches' loader with the published configuration, aimip-claima uses the four
ArchesWeather-M networks instead.

Compared with the authors' AIMIP submission (`allenai/aimip-phase1-submissions`, `ArchesWeather/ArchesWeather-V2`),
an `archesweather` run from 1978-09-30 gives a global-mean 2 m temperature of 286.95 K for October 1978 (authors'
member r2: 286.96 K) and 287.18 K for 1979 (authors' five members: 287.11–287.19 K). The runs are not bit-identical:
the authors ran on TPUs. Continued runs are bit-identical to one run on the same GPU type: `archesweather` in
segments of 40, 11 and 11 days (two of them ending inside a month) against 62 days, and `archesweathergen` with two
members in segments of 12 and 19 days against 31 days, on an A100.

**DLESyM, ArchesClimate-SSP and ArchesWeather software.** They run in a second Python environment (Earth2Studio 0.19
at commit 8fa7acb, PhysicsNeMo 1dacf8d, earth2grid 11dcf1b with its CUDA HEALPix padding kernel; every other package
pinned in `container/constraints-emu.txt`). The ArchesClimate-SSP source code (commit 06ce82dc) carries no licence,
so it is not part of the image or this repository: `aimip fetch -m archesclimate-ssp` (or the first run) downloads it
from its author's GitHub repository into `$AIMIP_DATA/code/ArchesClimate-SSP` and checks it against the tree digest
pinned in `src/aimip/setup_data.py`. The geoarches source (BSD-3-Clause, commit 4d0c77a1 of its `aimip` branch) is
unpacked at install time from its GitHub archive into `/opt/aimip/geoarches` (native: `<prefix>/geoarches`); the
archive's SHA-256 and the tree digest are pinned in `src/aimip/geoarches_src.py`, and every run checks the tree
again. The ArchesWeatherGen sampler is used with tensordict 0.15 (geoarches pins an older version); aimip-claima gives the
TensorDicts it passes in matching batch sizes so the arithmetic stays element by element. See
[THIRD_PARTY.md](../THIRD_PARTY.md) for the licences of everything the image installs.

**Samudra-OM4, MD-1.5, CAMulator, cBottle-3d and Spherical DYffusion.**
- `samudra-om4` runs in `venv-emu` (Apache-2.0 `M2LInES/Samudra@0c2a1bd8` UNet) and stages only the requested 5-day OM4 state pair + surface boundary fluxes from OSN (`s3://m2lines-pubs/Samudra/OM4`) into the run directory at prepare time. Dates before `2014-09-30T12:00:00` lie inside the 1975–2014 training/validation window. The OSN `OM4.zarr` store has 73 timestamps per year (`Jan 3 .. Dec 28`) with no time bounds or averaging attribute ("5-day mean" is the paper's statement); each leap year in `1976..2020` ends with a 6-day `Dec 28 -> Jan 3` step, so `Dec 31` of a leap year is covered by no 5-day state and gets weight `1/31` from the `Jan 3` state. Non-finite or out-of-band `thetao_0` (`[-5, 45]` °C) stops with exit code 5.
- `md-1.5` runs in `venv-emu` (`kjhall01/monthly-diffusion@b077d3bc`, MIT; `MD-1p5-latest.pth`, Hall & Molina 2026, arXiv:2604.13481). The first output month (`step 0`) is the deterministic encoder–decoder reconstruction of the initial state (`md/cvae.py:475-481`), not a forecast step; each subsequent step performs `T = 15` reverse DDPM steps (`md/diffusion.py:106` default; `config.pkl` omits `T`) batched across `--members` (with 5 static subgrid-orography and land-sea-mask channels `SDORSFC`, `ANORSFC`, `ISORSFC`, `SLORSFC`, `LSMSFC` embedded in `config.pkl`) and writes a single monthly state (`counts = 1`, not a sub-monthly average) across all 37 prognostic channels plus `restart.nc`. Driven by AIMIP Phase 1 monthly SST/SIC (Zenodo 17065758, 556 month-start-centred points `1978-10-01 .. 2025-01-01`); requires at least a 10-year spin-up (QBO bias in the first ~5 years). Upstream publishes no initial-state file: pass `--ic <path.nc>` (1.5° or 0.25° monthly-mean ERA5 NetCDF, or `<prev_run>/restart.nc`) or stage `$AIMIP_DATA/inputs/md-1.5/ic/ic_<YYYY-MM-DD>.nc`. Non-finite or out-of-band `TMP2m` (`[200, 330]` K) stops with exit code 5.
- `camulator` runs in its own isolated native environment `venv-camulator` (`install-native.sh --with-camulator`: PyTorch 2.4.1, `torch_harmonics` 0.7.2, `numpy<2`, and `WillyChap/miles-credit@d0eb9add` staged non-packaged under `venv-camulator/src/miles-credit` via `miles_credit_camulator.pth` because upstream `pyproject.toml` omits `credit.transforms.deprecated` and `credit.preblock`). Note that `checkpoint.pt00069.pt` uses 4 dynamic forcings (`SOLIN`, `SST`, `ICEFRAC`, `co2vmr_3d` over 1980–2014), differing from the 2-forcing 1979–2010 model in arXiv:2504.06007. Non-finite or out-of-band `TREFHT` (`[200, 330]` K) stops with exit code 5.
- `cbottle-3d` runs in `venv-emu` (`NVlabs/cBottle@f0f51998` via Earth2Studio `CBottle3D`). It is a conditional state generator without temporal memory: each day (`00/06/12/18` UTC, `batch_size=4`, chronological order) is drawn independently conditioned on `(day_of_year, hour_of_day)` and monthly mid-point SST (`PCMDI-AMIP-1-1-9` `tosbcs`), with no initial state, CO2, or sea-ice input; each 4-date batch derives a distinct seed via `SHA-256("aimip:cbottle-3d:v1:{seed}:{member_index}:{batch_index}") mod (2^31 - 1)` so diffusion draws are independent across batches and members. Non-finite or out-of-band `t2m` (`[200, 330]` K) stops with exit code 5.
- `spherical-dyffusion` runs in its own isolated native environment `venv-sdyff` (`install-native.sh --with-sdyff`: PyTorch 2.4.1, `torch_harmonics` 0.7.2, `ai2cm/modulus@94f62e1c` with `apex FusedLayerNorm` replaced by `torch.nn.LayerNorm` — numerically equivalent up to float rounding, while the published checkpoints configure `normalization_layer='instance_norm'` / `torch.nn.InstanceNorm2d` — and `Rose-STL-Lab/spherical-dyffusion@a10b3e7f` fetched into `$AIMIP_DATA/code/spherical-dyffusion`). It emulates a repeating 365-day annual SST cycle (labelled 2021 in Zenodo 10791087); output months use nominal cycle indices (`0001-01…`) over complete months (`coarsen_factor=4`), rename ocean-prescribed `surface_temperature` to `surface_temperature_prescribed_ocean`, and summarize predicted `air_temperature_7` (stopping with exit code 5 outside `[180, 340]` K).
- **Checkpoint verification gate.** Across all five models, SHA-256 verification on `O_NOFOLLOW` file descriptors (with trusted-owner and non-group/world-writable directory/file checks) is the primary integrity boundary before loading from that same handle; `weights_only=True` and closed pickle global allowlists serve as a second defense-in-depth layer (especially in `venv-camulator` and `venv-sdyff`, which pin `torch==2.4.1`).

## Forcing of the AMIP models

| model | 1979/1994–2014 | 2015–2024 |
|---|---|---|
| `ace2-era5` | Ai2's ERA5 forcing (`allenai/ACE2-ERA5`): SST and skin temperature, sea ice, land/ocean fractions, insolation, global-mean CO2 | — |
| `ace2-shield`, `ace2s-shield-plus` | Ai2's SHiELD+ AMIP forcing (`allenai/ACE2S-SHiELD-plus`) | AIMIP SST and sea ice (Zenodo 17065758, ERA5 0.25° monthly, regridded conservatively to the models' 1° grid, linear in time); insolation and statics from the model's reference year; CO2 from the ACE2-ERA5 series, extrapolated linearly after 2022 |
| `ace2-eamv3` | ERA5 SST and sea ice from the Ai2 ERA5 forcing; land fraction, orography, insolation cycle from the authors' 1971 file; noleap calendar (29 February dropped); no CO2 input | — |
| `hiro-ace` | SHiELD+ AMIP time-varying fields with HiRO's own statics (coastal cells take SST from the nearest SHiELD+ ocean cell) | as for SHiELD+ |
| `archesweather`, `archesweathergen` | the authors' AIMIP forcing (Zenodo 22795233): ERA5 monthly SST and sea-ice cover regridded conservatively to 1°, constant within each month, from October 1978 | the same file, through December 2024 |

The authors of HiRO-ACE state that ACE2S was trained only on 9 years of X-SHiELD (2014–2022) and is not expected to
generalize outside the forcing conditions used in training (model card `allenai/HiRO-ACE`; Perkins et al.,
"HiRO-ACE: Fast and skillful AI emulation and downscaling trained on a 3 km global storm-resolving model",
arXiv:2512.18224). The 1994–2014 window of `hiro-ace` is therefore outside its training period, with forcing we
build from another model's package and a 2014 state given a 1994 date: treat it as an experiment.

The other models use their authors' forcing packages unchanged (`allenai/ACE2-SOM`, `allenai/HiRO-ACE`,
`allenai/HiRO-ACEv2`, `allenai/SamudrACE-CM4-piControl`, `allenai/SamudrACE-E3SMv3`, `gclyne/ArchesClimate-data`,
Zenodo 22795233 for ArchesWeather).
Derived forcing files are rebuilt by `aimip setup`/`aimip fetch` on your machine and checked against the lock file.
