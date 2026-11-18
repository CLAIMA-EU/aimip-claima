# aimip-claima

aimip-claima runs AI climate emulators as climate experiments, on a GPU workstation or an HPC cluster. The command is
`aimip`. You choose a model, a start date and a length; aimip-claima downloads the model from its authors, checks
every file, prepares the run, starts it on Slurm or on your machine, and keeps a record of exactly what was run.

```bash
aimip run ace2-era5 --start 1979-01-01 --length 1y --name my-first-year     # prepare one model year
aimip submit my-first-year                                                   # start it
aimip summary my-first-year                                                  # global-mean temperature per month
```

## Where to start

1. **[Install it](docs/installation.md):** about an hour, with a container (Apptainer, Singularity, Docker, Podman)
   or without one.
2. **[Make your first run](docs/user-guide.md#your-first-run-step-by-step):** a two-day test run, step by step,
   with the output you should see.
3. **[Recipes](docs/user-guide.md#recipes):** longer runs, ensembles, AIMIP runs, scenarios, coupled models.

## Models

`aimip models` lists them; `aimip show <model>` describes one. Details and known limits: [docs/models.md](docs/models.md).

| family | models | what they are |
|---|---|---|
| ACE (Ai2) | `ace2-era5`, `ace2-shield`, `ace2s-shield-plus`, `ace2-eamv3`, `hiro-ace`, `ace2-som`, `ace2s`, `hiro-acev2` | atmosphere emulators trained on ERA5 or on climate models, forced by sea surface temperature; `ace2-som` with a slab ocean |
| SamudrACE (Ai2) | `samudrace-cm4`, `samudrace-e3smv3` | coupled atmosphere–ocean emulators of GFDL CM4 and E3SMv3 |
| DLESyM (UW / NVIDIA) | `dlesym-v1.0`, `dlesym-v1.1`, `dlesym-v0-isccp`, `dlesym-aimip` | coupled atmosphere–ocean model on a HEALPix grid; `dlesym-aimip` with the AIMIP sea surface temperature |
| ArchesWeather (INRIA) | `archesweather`, `archesweathergen` | the AIMIP versions: daily atmosphere, deterministic and stochastic |
| ArchesClimate (INRIA / Mila) | `archesclimate-ssp` | monthly emulator of IPSL-CM6A-LR for SSP scenarios |
| others | `samudra-om4`, `md-1.5`, `camulator`, `cbottle-3d`, `spherical-dyffusion` | an ocean emulator of GFDL OM4, a monthly diffusion model, an emulator of CAM6, a generator of atmospheric states, an emulator of FV3GFS |

## What aimip-claima does for you

- **Downloads each model on first use** from where its authors publish it (Hugging Face, Zenodo, GitHub, ARCO-ERA5)
  and checks every file against a lock file of checksums, so everyone runs the same files.
- **Checks a run before it starts:** dates inside the forcing data, an initial state for the start date, enough job
  time. It also shows what the model's authors say the model should not be used for.
- **Prepares a run directory** with the model configuration, the job script and a record of the command, the
  checksums of the files used, the seed and the software versions.
- **Runs it** on Slurm or on the current machine, and summarises the results.
- Ensembles, continued runs, scenarios, LoRA checkpoints, an optional assistant (`aimip chat`) and a web interface
  (`aimip ui`).

## Versions

`main` is the development version (1.4.0.dev0), described by these pages. The latest release is
[v1.3.1](https://github.com/CliDyn/aimip-claima/tree/v1.3.1); its documentation is in that tag.

## Documentation

| page | |
|---|---|
| [installation.md](docs/installation.md) | installing, checking the installation, model data, special situations |
| [user-guide.md](docs/user-guide.md) | the first run step by step, recipes, reading the results |
| [models.md](docs/models.md) | every model: forcing, initial states, time steps, known limits, comparison with the authors' results |
| [troubleshooting.md](docs/troubleshooting.md) | error messages and what to do |
| [slurm-and-paths.md](docs/slurm-and-paths.md) | job scripts, wall time, run directories, path rules |
| [console.md](docs/console.md), [agent.md](docs/agent.md) | approvals, the assistant, the MCP server, the web interface |
| [testing.md](docs/testing.md) | how the installations were tested |
| [development.md](docs/development.md) | code layout, tests, the lock file, adding a model |

## Licence

The code is released under the MIT License ([LICENSE](LICENSE)). The models, weights and data are not part of this
repository; each belongs to its authors and is downloaded from them under their own licence. The licences of the
software in the image, and of the code downloaded on first use, are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
