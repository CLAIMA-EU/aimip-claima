# Models

`aimip models` prints the same list from your install, with the date windows and initial conditions.

| model | type | forcing / date window | shipped initial conditions | ensembles |
|---|---|---|---|---|
| `ace2-era5` | deterministic, trained on ERA5 | 1979–2014 | the 1st of each month in 1979 and in 2001 (ERA5 states) | different initial conditions (`ic spin`) |
| `ace2-shield` | deterministic, trained on SHiELD | 1979–2014; 2014–2024 with AIMIP SST; julian | one state, 1979-01-01 (from the ACE2S-SHiELD-plus package; ACE2-SHiELD ships none) | none from the shipped state |
| `ace2s-shield-plus` | stochastic | 1979–2014; 2014–2024; julian | one state, 1979-01-01 | `--members K` (model noise) |
| `ace2-eamv3` | deterministic, trained on E3SMv3 | 1994–2014; noleap calendar | one state stamped 1994-01-01 (**a 1971 model state**) | none from the shipped state |
| `hiro-ace` | stochastic | 1994–2014; 2014–2024; julian | one state stamped 1994-01-01 (**a 2014 model state**) | `--members K` |
| `ace2-som` | deterministic, slab ocean | scenarios `1xCO2` (default) and `3xCO2`, nominal years 2032–2035 | one state per scenario, 2032-01-01 | pass several states with `--ic` |
| `ace2s` | stochastic (the `hiro-ace` checkpoint with the authors' forcing) | 2014–2023; julian calendar | `ic_2014.nc`, `ic_2023.nc`, 12 states each | `--members K` |
| `hiro-acev2` | stochastic, ACE2S-X-SHiELD global stage; `fme` 2026.10.0 | 2014–2023; julian | `ic_2014.nc`, `ic_2023.nc`, 12 states each | `--members K` |
| `samudrace-cm4` | **coupled** atmosphere–ocean, deterministic, CM4 preindustrial control | `picontrol-0311`: labels 0311–0410, `picontrol-0151`: 0151-01-06 … 0250-12-27 (one forcing year repeated); noleap | five states 0311 … 0319, one state 0151-01-06 | different initial conditions (`ic spin`) |
| `samudrace-e3smv3` | **coupled**, stochastic; `fme` 2026.10.0 | one year, 0425-01-03 … 0425-12-29 (**at most 72 steps**); noleap | three states in one file | `--members K` |
| `dlesym-v1.0` | **coupled** atmosphere–ocean (HEALPix-64, output 0.25°), deterministic; Earth2Studio | not forced; any 6-hourly start 1940-01-04 … 2023-12-31 | ERA5 at `--start`, downloaded by `aimip run` | — |
| `dlesym-v1.1` | coupled, stochastic; adds 6-hour precipitation | as above | as above | `--members K` (noise seed + k) |
| `dlesym-v0-isccp` | coupled, deterministic | as above | as above | — |
| `archesclimate-ssp` | monthly emulator of IPSL-CM6A-LR (144 × 144: 1.25° lat × 2.5° lon; atmosphere and upper-ocean temperature), deterministic | scenarios `ssp434` (default) and `ssp585`; outputs March 2015 … November 2100 | the scenario's 2015 state (`--start 2015-01-01T00:00:00`) | — |

## Steps

| models | one step | one year |
|---|---|---|
| the eight `fme` atmosphere models | 6 hours | 1460 steps (1464 in a leap year with a Gregorian or Julian calendar) |
| `samudrace-cm4`, `samudrace-e3smv3` | 5 days | 73 steps |
| `dlesym-*` | 6 hours | 1460 or 1464 steps |
| `archesclimate-ssp` | 1 month | 12 steps (at most 1029) |

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
the checkpoint Earth2Studio documents for 100–1000-year climate runs. A run stops with exit code 4 at the first month
with non-finite values or a global-mean 2 m temperature outside 200–330 K, records `unstable_from` and keeps the
months before it. DLESyM's `sst` has values over land: mask land before use. `dlesym-v1.1` writes
`atmosphere/monthly_precip.nc`; its `tp06` (6-hour precipitation) is derived with ε = 1e-8, inferred from the
authors' pipeline, and its 3-hour OLR input is built as Earth2Studio does.

**ArchesClimate-SSP.** `--steps` counts months. The run always starts from the scenario's 2015 state; the first
output month is March 2015. It emulates one climate model (IPSL-CM6A-LR) under one scenario forcing, not observed
weather. Outputs: `atmosphere/monthly_mean_predictions.nc` (`tas`, `psl`, `pr`, …), `atmosphere/monthly_levels.nc`
(17 pressure levels), `ocean/monthly_mean_predictions.nc` (`thetao_0`, the top level, °C) and
`ocean/monthly_thetao.nc` (`thetao` on 10 depths, °C). Pressure levels below the ground (mostly 1000–700 hPa
over land) and ocean values over land are NaN, with a mask that does not change in time.

**DLESyM and ArchesClimate-SSP software.** They run in a second Python environment (Earth2Studio 0.19 at commit
8fa7acb, PhysicsNeMo 1dacf8d, earth2grid 11dcf1b with its CUDA HEALPix padding kernel; every other package pinned in
`container/constraints-emu.txt`). The ArchesClimate-SSP source code (commit 06ce82dc) carries no licence, so it is
not part of the image or this repository: `aimip fetch -m archesclimate-ssp` (or the first run) downloads it from
its author's GitHub repository into `$AIMIP_DATA/code/ArchesClimate-SSP` and checks it against the tree digest pinned in
`src/aimip/setup_data.py`. See
[THIRD_PARTY.md](../THIRD_PARTY.md) for the licences of everything the image installs.

## Forcing of the AMIP models

| model | 1979/1994–2014 | 2015–2024 |
|---|---|---|
| `ace2-era5` | Ai2's ERA5 forcing (`allenai/ACE2-ERA5`): SST and skin temperature, sea ice, land/ocean fractions, insolation, global-mean CO2 | — |
| `ace2-shield`, `ace2s-shield-plus` | Ai2's SHiELD+ AMIP forcing (`allenai/ACE2S-SHiELD-plus`) | AIMIP SST and sea ice (Zenodo 17065758, ERA5 0.25° monthly, regridded conservatively to the models' 1° grid, linear in time); insolation and statics from the model's reference year; CO2 from the ACE2-ERA5 series, extrapolated linearly after 2022 |
| `ace2-eamv3` | ERA5 SST and sea ice from the Ai2 ERA5 forcing; land fraction, orography, insolation cycle from the authors' 1971 file; noleap calendar (29 February dropped); no CO2 input | — |
| `hiro-ace` | SHiELD+ AMIP time-varying fields with HiRO's own statics (coastal cells take SST from the nearest SHiELD+ ocean cell) | as for SHiELD+ |

The other models use their authors' forcing packages unchanged (`allenai/ACE2-SOM`, `allenai/HiRO-ACE`,
`allenai/HiRO-ACEv2`, `allenai/SamudrACE-CM4-piControl`, `allenai/SamudrACE-E3SMv3`, `gclyne/ArchesClimate-data`).
Derived forcing files are rebuilt by `aimip setup`/`aimip fetch` on your machine and checked against the lock file.
