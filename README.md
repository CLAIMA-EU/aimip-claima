# aimip-claima

aimip-claima runs AI climate emulators as climate experiments, on a GPU workstation or an HPC cluster. The command is
`aimip`. You choose a model, a start date and a length; aimip-claima downloads the model from its authors, checks
every file, prepares the run, starts it on Slurm or on your machine, and keeps a record of exactly what was run.

> **Requirements: Linux with an NVIDIA GPU.** aimip-claima needs a Linux system (x86-64 or ARM, for example an HPC
> cluster or a GPU workstation) with an NVIDIA GPU and a driver for CUDA 13 or newer. On a Windows PC with an NVIDIA GPU
> it runs inside WSL2 ([see below](#on-a-windows-pc-with-an-nvidia-gpu-wsl2)). **macOS is not supported**: many of the
> models need CUDA, which Macs do not have. From a Mac, install aimip-claima on a Linux GPU
> server and work there over SSH; the web console opens on your Mac through an SSH tunnel
> ([docs/console.md](docs/console.md)).

```bash
aimip run ace2-era5 --start 1979-01-01 --length 1y --name my-first-year     # prepare one model year
aimip submit my-first-year                                                   # start it
aimip summary my-first-year                                                  # global-mean temperature per month
aimip export-cf my-first-year                                                # CF copies for cdo, AQUA, xarray
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
- **Writes what you ask for:** monthly means by default; for the ACE, DLESyM and ArchesWeather models also daily or
  6-hourly files of chosen variables and periods (`--output`).
- **Exports standard copies:** `aimip export-cf` writes CF copies of the monthly means (time first, real dates, CMIP
  names where they match exactly) for `cdo`, AQUA and xarray; `aimip export-cmor` writes CMIP6-style `Amon` files. The
  run itself is never changed ([user guide](docs/user-guide.md#cf-copies-for-cdo-and-other-tools-aimip-export-cf)).
- Ensembles, continued runs, scenarios, LoRA checkpoints, an optional assistant (`aimip chat`) and a web interface
  (`aimip ui`).

## Versions

`main` is the development version (1.4.0.dev0), described by these pages. The latest release is
[v1.3.1](https://github.com/CliDyn/aimip-claima/tree/v1.3.1); its documentation is in that tag.

## Documentation

| page | |
|---|---|
| [installation.md](docs/installation.md) | installing, checking the installation, model data, special situations |
| [user-guide.md](docs/user-guide.md) | the first run step by step, recipes, reading the results, CF and CMIP6-style copies (`aimip export-cf`, `aimip export-cmor`) |
| [models.md](docs/models.md) | every model: forcing, initial states, time steps, known limits, comparison with the authors' results |
| [troubleshooting.md](docs/troubleshooting.md) | error messages and what to do |
| [slurm-and-paths.md](docs/slurm-and-paths.md) | job scripts, wall time, run directories, path rules |
| [console.md](docs/console.md), [agent.md](docs/agent.md) | approvals, the assistant, the MCP server, the web interface |
| [testing.md](docs/testing.md) | how the installations were tested |
| [development.md](docs/development.md) | code layout, tests, the lock file, adding a model |

## Known limits

- **GPUs tested:** NVIDIA A100 (x86-64) and GH200 (ARM). Other NVIDIA GPUs with a driver for CUDA 13 should work for
  most models but are not in the test record; start with the first test run of `ace2-era5` and compare its number with
  the one in the [user guide](docs/user-guide.md#your-first-run-step-by-step).
- **Newest GPUs (Blackwell: B100/B200, RTX 50 series) and ARM GPU nodes:** `camulator` and `spherical-dyffusion` run in
  their own environment pinned to PyTorch 2.4.1 for CUDA 12.1, which has no build for these GPUs or for ARM. Run these
  two models on an A100 or H100 node; all other models use the main environment (PyTorch for CUDA 13).
- **GPU memory:** the models were run on 80 GB GPUs. On cards with 16–24 GB the ACE and ArchesWeather models are
  expected to fit; the larger models (for example `cbottle-3d`, DLESyM at 0.25° output) may not.
- **No internet on the machine:** the DLESyM models and `samudra-om4` download starting data whenever a run is
  prepared, so they cannot run where no node reaches the internet (see `aimip bundle` in the
  [installation guide](docs/installation.md)).
- **macOS:** not supported (see Requirements above). **Windows:** through WSL2 only; tested with the first run of
  `ace2-era5` on a laptop RTX 4060 (8 GB), which gave the reference number.

<a id="on-a-windows-pc-with-an-nvidia-gpu-wsl2"></a>
<details>
<summary><b>Windows PC with an NVIDIA GPU (WSL2)</b></summary>

aimip-claima runs inside WSL2, the Linux that comes with Windows 10 and 11. A gaming or laptop GPU is enough for the
ACE models; start with the first test below.

1. Install the latest NVIDIA driver for Windows (the normal Game Ready or Studio driver). Do not install a driver
   inside Linux: WSL2 passes the GPU through.
2. In PowerShell: `wsl --install -d Ubuntu-24.04`, restart, open "Ubuntu" from the Start menu and create your Linux
   user.
3. In Ubuntu, check the GPU with `nvidia-smi` (it should show `CUDA Version: 13.0` or higher), then:

   ```bash
   sudo apt install -y git build-essential python3.12-venv
   git clone https://github.com/CliDyn/aimip-claima.git && cd aimip-claima
   install/install-native.sh --site local --prefix ~/aimip --run-root ~/aimip-runs
   ~/aimip/bin/aimip run ace2-era5 --start 1979-01-01 --steps 8 --name first
   ~/aimip/bin/aimip submit first
   ~/aimip/bin/aimip watch first       # wait until it says completed (about a minute)
   ~/aimip/bin/aimip summary first     # 1979-01 TMP2m: 285.1578 K
   ```

Keep everything in your Linux home (`~`), not under `/mnt/c`: the Windows disk is slow from Linux. WSL2 uses half of
the PC's memory by default; if a model runs out of memory, raise it in `%UserProfile%\.wslconfig` (`[wsl2]`,
`memory=16GB`) and run `wsl --shutdown`. On GPUs with 8–16 GB start with the ACE models; the larger models may not fit.

</details>

## Licence

The code is released under the MIT License ([LICENSE](LICENSE)). The models, weights and data are not part of this
repository; each belongs to its authors and is downloaded from them under their own licence. The licences of the
software in the image, and of the code downloaded on first use, are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
