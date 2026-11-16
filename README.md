# aimip-claima

Run AI climate emulators as climate experiments on a GPU machine or an HPC cluster, with one command: `aimip`.

The `main` branch is the development version (1.4.0.dev0: a fuller console, agent runs with human approval, an
MCP server and a web UI; see [docs/console.md](docs/console.md)). The latest release is
[v1.3.1](https://github.com/CliDyn/aimip-claima/tree/v1.3.1); install from that tag for the released version. No image is published for development versions, so on `main`
use `install/install-container.sh … --build` or the native installer.

```bash
aimip run -m ace2-era5 --start 1979-01-01T00:00:00 --steps 1460 --submit   # one model year
aimip submit <run-dir>
aimip summary <run-dir>                                                    # global-mean 2 m temperature per month
```

Fourteen models are supported: the Ai2 ACE family (`ace2-era5`, `ace2-shield`, `ace2s-shield-plus`, `ace2-eamv3`,
`hiro-ace`, `ace2-som`, `ace2s`, `hiro-acev2`), the coupled atmosphere–ocean SamudrACE models (`samudrace-cm4`,
`samudrace-e3smv3`), three DLESyM checkpoints (`dlesym-v1.0`, `dlesym-v1.1`, `dlesym-v0-isccp`) and
ArchesClimate-SSP (`archesclimate-ssp`). See [docs/models.md](docs/models.md).

What `aimip` does for you:

- **Downloads each model on first use** from where its authors publish it (Hugging Face, Zenodo, ARCO-ERA5, GitHub) and
  checks every file against a lock file (`src/aimip/aimip.lock`, sha256 and pinned revisions).
- **Prepares a run directory** with the model configuration, a job script and a manifest that records the command
  line, the checksums of the checkpoint and the initial state, the forcing files, the seed and the software versions.
- **Runs it** on Slurm (`aimip submit`) or directly on the current machine (`--local`).
- Ensembles (initial-condition spin-up or model noise), continued runs, scenario forcing sets, LoRA checkpoints, and
  an optional planning assistant (`aimip chat`).

## Two ways to install

| | container | native |
|---|---|---|
| what you need | Apptainer/Singularity (HPC) or Docker/Podman (+ NVIDIA Container Toolkit) | Python 3.12; for DLESyM/ArchesClimate also a C/C++ compiler |
| what you get | the published image with every library inside; on the host only a small wrapper | Python virtual environments in a directory of your choice |
| command | `install/install-container.sh --prefix DIR --site <preset>` | `install/install-native.sh --prefix DIR --site <preset> [--with-emu]` |
| best for | shared clusters, reproducibility, several users | a workstation or a cluster without containers |

Both give the same `DIR/bin/aimip` command and the same run directories. In our tests four of the five models
compared gave bit-identical results in both modes, and since 1.3.1 so does the coupled `samudrace-cm4`. See
[what was tested](docs/installation.md#what-was-tested).

### Container (Apptainer, Singularity, Docker or Podman)

```bash
git clone https://github.com/CliDyn/aimip-claima.git && cd aimip-claima
install/install-container.sh --prefix /big/disk/aimip --site generic-slurm --partition gpu --account myproject \
    --run-root /big/disk/aimip-runs --build
/big/disk/aimip/bin/aimip models
```

With `--build` the image is built on your machine from this checkout (30–40 minutes; Apptainer needs no root).
Without `--build` the installer fetches the ready image `ghcr.io/clidyn/aimip-claima:<version>` (x86_64, public, no
login needed). Only release versions have an image: check out a release tag (`git checkout v1.3.1`) to use it; on
`main` or if the image cannot be fetched, the installer stops and tells you to add `--build`.

### Native (no container)

```bash
git clone https://github.com/CliDyn/aimip-claima.git && cd aimip-claima
install/install-native.sh --prefix /big/disk/aimip --site local --run-root /big/disk/aimip-runs   # add --with-emu for DLESyM/ArchesClimate
/big/disk/aimip/bin/aimip models
```

Presets: `generic-slurm`, `local` (no scheduler) and `levante` (DKRZ). Details, disk space and checks:
[docs/installation.md](docs/installation.md).

## First run

```bash
A=/big/disk/aimip/bin/aimip
$A run -m ace2-era5 --start 1979-01-01T00:00:00 --steps 8 --submit   # two model days; downloads ace2-era5 (20.5 GiB) first
$A submit <run-dir printed above>
$A status <run-dir>
$A summary <run-dir>                                                   # January 1979: TMP2m 285.1578 K
```

Without Slurm, use `--local` instead of `--submit`.

## Documentation

| | |
|---|---|
| [docs/installation.md](docs/installation.md) | both install modes, presets, model data, checks |
| [docs/user-guide.md](docs/user-guide.md) | running experiments: first run, ensembles, continued runs, scenarios, coupled models, LoRA, outputs |
| [docs/models.md](docs/models.md) | the fourteen models: type, forcing, initial conditions, calendars, known limits |
| [docs/slurm-and-paths.md](docs/slurm-and-paths.md) | job scripts, wall time, run directories, path rules |
| [docs/agent.md](docs/agent.md) | the optional planning assistant `aimip chat` |
| [docs/console.md](docs/console.md) | development version: new commands, approval, agent runs, MCP server, web UI |
| [docs/troubleshooting.md](docs/troubleshooting.md) | error messages and what to do |
| [docs/development.md](docs/development.md) | layout, tests, the lock file, adding a model |

## Licence

The code is released under the MIT License ([LICENSE](LICENSE)). The models, weights and data are not part of this
repository; each belongs to its authors and is downloaded from them under their own licence.

Licences of the software installed by the image, and of the code downloaded on first use: [THIRD_PARTY.md](THIRD_PARTY.md).
