# Installation

`aimip` installs into one directory, the **prefix**. Both install modes put the command at `<prefix>/bin/aimip`,
the site settings at `<prefix>/site.json` and the model data at `<prefix>/data/`.

## What the machine needs

- Linux x86_64 with an NVIDIA GPU (A100, H100 or newer for every model; see "GPUs" below). On aarch64 (GH200) the
  container mode builds its own image; see [aarch64 and GH200](#aarch64-and-gh200).
- Internet access on the machine where you install and where you prepare runs (model downloads, the DLESyM initial
  state). Compute nodes need no internet.
- `bash`, `python3` (3.8+ for the installers themselves), `tar`, `sha256sum` or `shasum`.
- Disk: about 25 GB for the software (image and build cache, or the virtual environments), plus the model data:
  0.03–34 GB per model, about 258 GB for all twenty-one.
- Container mode: one of Apptainer, Singularity, Docker or Podman; Docker and Podman need the NVIDIA Container Toolkit
  for `--gpus all`. Native mode: Python 3.12 (the pins were made for 3.12; newer versions are untested), and for
  `--with-emu` also a C/C++ compiler.
- An NVIDIA driver that supports CUDA 13.0 (the PyTorch wheels are built for CUDA 13.0).

## Any cluster

| the cluster has | use |
|---|---|
| Apptainer or Singularity (most HPC systems) | container mode; the image is fetched, nothing is built |
| Docker or Podman (workstations, cloud) | container mode |
| no container runtime | native mode (Python 3.12 and a C/C++ compiler) |
| Slurm | presets `generic-slurm` or `levante`; `aimip submit` queues the job |
| another scheduler (PBS, LSF, …) or none | preset `local`; inside an interactive or batch allocation run `aimip run … --local`, or prepare the run on a node with internet (`aimip run … --dry-run`) and call its `DIR/bin/aimip exec-fme <run dir>` line from your own job script. Models must be fetched beforehand where there is internet (`aimip fetch -m X`); DLESyM downloads its start state when the run is prepared |

## Choose a site preset

| preset | scheduler | defaults |
|---|---|---|
| `generic-slurm` | Slurm | you give `--partition` and `--account` (or `none`); one GPU (`--gres gpu:1`) and 16 CPUs by default |
| `local` | none | runs on the current machine; `aimip submit` starts the job in the background |
| `levante` | Slurm (DKRZ Levante) | partition `gpu`, one A100-80GB, run root `/scratch/<first letter>/<user>/aimip-runs`; you give `--account` |

Every option can be changed later per run (`--partition`, `--account`, `--gres`, `--cpus-per-task`, `--mem`,
`--time`). The run root (where run directories go) is set with `--run-root`. With `generic-slurm` and `local` it
defaults to `$HOME/aimip-runs`; on a cluster your home directory usually has a small quota, so pass a directory on a
large disk (a ten-year run writes tens of gigabytes).

## Container mode

```bash
install/install-container.sh --prefix DIR --site <preset> [--account A|none] [--partition P] [--gres G|none] \
                             [--run-root R] [--runtime auto|apptainer|singularity|docker|podman] [--image REF | --build]
```

By default the installer **fetches the published image** `ghcr.io/clidyn/aimip-claima:<version>` (only release
versions have an image, and only if it is published and readable to you; otherwise use `--build`, which builds the
image from your checkout in 30–40 minutes) (the version of
your checkout; x86_64 — on aarch64 hosts it builds the image instead): Apptainer/Singularity convert it to a SIF file in `DIR/images/` (no root, no compiler),
Docker/Podman pull it. It then writes `site.json` and the wrapper `DIR/bin/aimip` and runs the wrapper's self-test;
it ends with `WRAPPER_TESTS pass=… fail=0` and `Installed …`. The log is `DIR/build/build-<version>.log`.

The image is the same everywhere: it was built once from `container/Dockerfile`, and the package tests ran inside it
while it was built. `--image REF` takes another tag or registry (for example a copy in your site's registry);
`--build` builds the image on this machine instead (20–40 minutes, needs internet; the build log then shows
`AIMIP_IMAGE_OK`, `AIMIP_OVERLAY_OK`, `AIMIP_EMU_OK` and a test line `... passed`). Run long installs in `tmux` or
with `nohup`.

Fetching was tested with Apptainer 1.5 (the installer uses `build --mksquashfs-args`); it needs about 25 GB of free space in `TMPDIR` for the
conversion, and HTTPS access to `ghcr.io` (set `https_proxy` if your site uses a proxy).

Without the image registry (no internet on the machine): fetch the image elsewhere
(`apptainer build aimip.sif docker://ghcr.io/clidyn/aimip-claima:<version>`), copy `aimip.sif` to
`DIR/images/aimip-<version>.sif`, then run the installer: it uses the file that is already there.

The container sees the install read-only and writes only to your run root and to the paths you pass with `--out`.

**Apptainer temporary directory.** On systems where Apptainer cannot mount the image directly it unpacks it for every
`aimip` call (it prints "Converting SIF file to temporary sandbox"). If `TMPDIR` points to a parallel file system this
takes minutes; point it at a local disk:

```bash
export APPTAINER_TMPDIR=/tmp
```

**Docker or Podman without Slurm.** Install with `--site local`; `aimip run … --local` then runs the model in a
container on this machine's GPU.

## Native mode

```bash
install/install-native.sh --prefix DIR --site <preset> [--account A|none] [--partition P] [--gres G|none] \
                          [--run-root R] [--with-emu] [--with-camulator] [--with-sdyff] [--python python3.12] [--dry-run]
```

`--dry-run` writes `site.json` and the launcher and checks the options without creating environments.

It creates, inside `DIR`:

| path | content |
|---|---|
| `venv/` | Python 3.12 with PyTorch 2.14 (CUDA 13.0 wheels), Ai2 `fme` 2026.5.1 and `aimip`, pinned like the image |
| `overlay-fme-2026.10.0/` | `fme` 2026.10.0, used only by `hiro-acev2` and `samudrace-e3smv3` |
| `venv-emu/` | with `--with-emu`: the environment for DLESyM, ArchesClimate-SSP, ArchesWeather, Samudra-OM4 and cBottle-3d (Earth2Studio, PhysicsNeMo, earth2grid with its CUDA padding kernel, `NVlabs/cBottle`, and ArchesClimate-SSP's and geoarches' Python dependencies), pinned like the image. The ArchesClimate-SSP source itself is downloaded on first use ([models](models.md)) |
| `geoarches/` | with `--with-emu`: the geoarches source for ArchesWeather (commit 4d0c77a1, checked against its pinned archive and tree digest) |
| `venv-camulator/` | with `--with-camulator`: isolated native environment for `camulator` (PyTorch 2.4.1, `torch_harmonics` 0.7.2, `numpy` 1.26.4, and `WillyChap/miles-credit@d0eb9add` staged non-packaged under `venv-camulator/src/miles-credit` via `miles_credit_camulator.pth`; pinned via `container/constraints-camulator.txt`) |
| `venv-sdyff/` | with `--with-sdyff`: isolated native environment for `spherical-dyffusion` (PyTorch 2.4.1, `torch_harmonics` 0.7.2, `ai2cm/modulus@94f62e1c` with a local `apex.normalization.FusedLayerNorm = torch.nn.LayerNorm` shim; pinned via `container/constraints-sdyff.txt`; `Rose-STL-Lab/spherical-dyffusion@a10b3e7f` is downloaded into `data/code/spherical-dyffusion` on first use) |
| `bin/aimip` | the launcher that sets the paths and starts `venv/bin/python -m aimip` |
| `site.json`, `data/` | site settings, model data |

The installer runs the package tests in the new environment and prints the next steps. Without `--with-emu` the
ten `fme` models work; DLESyM, ArchesClimate-SSP and ArchesWeather report that their environment is missing.

**Package and checkpoint pinning.** Like `venv/` and `venv-emu/`, `venv-camulator/` and `venv-sdyff/` pin exact `==`
package versions in `container/constraints-camulator.txt` and `container/constraints-sdyff.txt` (with `#sha256=` on
direct GitHub source archives such as `NVlabs/cBottle` and `ai2cm/modulus`, and `torch_harmonics==0.7.2` built with
`--no-deps`), without `--require-hashes` on wheel installs and with configurable PyTorch wheel indices
(`AIMIP_CAMULATOR_TORCH_INDEX` / `AIMIP_SDYFF_TORCH_INDEX`). Because those two environments pin `torch==2.4.1`
(predating PyTorch 2.6's hardened `WeightsUnpickler`), cryptographic SHA-256 verification on `O_NOFOLLOW` file
descriptors is the primary checkpoint gate, while `weights_only=True` and closed pickle allowlists act as a second
defense-in-depth layer.

## Model data

Two ways, in both modes:

- **On first use** (default): the first `aimip run -m X` downloads and verifies model X only, and prints what it
  downloads. `aimip fetch -m X` does the same without preparing a run; `--no-fetch` forbids downloads.
- **All at once:** `aimip setup --models all` (about 258 GB). On clusters that limit work on login nodes, run it in
  a CPU job. The downloads are about 163 GiB; the rest are forcing files built on your machine and the ArchesWeather
  checkpoints extracted from their zip.

| model | download |
|---|---|
| `ace2-era5` | 20.5 GiB |
| `ace2-shield`, `ace2s-shield-plus`, `hiro-ace`, `ace2-eamv3` | checkpoints 1.7–3.0 GiB each, plus the downloads their forcing is built from (SHiELD+ forcing 20.3 GiB, AIMIP SST 4.3 GiB, EAMv3 raw files 4.3 GiB) and the built forcing files |
| `ace2-som` 8.7 · `ace2s` 7.6 · `hiro-acev2` 8.9 · `samudrace-cm4` 3.1 · `samudrace-e3smv3` 2.5 GiB | |
| `dlesym-v1.0` 0.16 · `dlesym-v1.1` 0.04 · `dlesym-v0-isccp` 0.29 GiB | plus the ERA5 initial state per run (0.4 GB) |
| `archesclimate-ssp` | 33.5 GiB (both scenarios) |
| `archesweather`, `archesweathergen` | 2.0 GiB from Zenodo 22795233 (the checkpoint zip, the initial state, the forcing, statistics), plus 1.3 or 1.8 GiB of checkpoints extracted from the zip |
| `samudra-om4` 1.5 · `md-1.5` 0.03 (+ AIMIP SST above) · `camulator` 5.9 (+ 10.7 GiB on demand) · `cbottle-3d` 3.2 · `spherical-dyffusion` 25.3 GiB | `samudra-om4` also stages its OM4 window per run (~1.5 GB/yr) |

Every file is checked against `aimip.lock`: downloads by size and sha256, files built on your machine by sha256. Derived files (forcing built from the authors'
packages) are rebuilt on your machine and must match the lock bit for bit.

**Several users, one install.** Only the owner of the prefix downloads. Other users see the data read-only; a model
that is missing gives `model X is not installed here: ask the install owner to run 'aimip fetch -m X'`. Checkpoints,
initial-condition files and code directories must be owned by `root`, the user running `aimip`, or the owner of a
non-writable install/data prefix, and must not be writable by group or others (`chmod go-w`); files owned by another
colleague outside the prefix are refused.

## GPUs

All models run on one GPU. The DLESyM HEALPix padding kernel is compiled for compute capability 8.0 and 9.0 (A100,
H100, GH200). On older GPUs `dlesym-v1.1` needs the kernel built for them: add their compute capability to `TORCH_CUDA_ARCH_LIST`
in `container/Dockerfile` (then `--build`) or in `install/install-native.sh`.

## aarch64 and GH200

The published image is x86_64 only, so on aarch64 hosts the installer builds the image itself (it switches to
`--build`). The build needs internet access and a temporary directory on a local disk (`TMPDIR=/tmp`); on the GH200 node
we tested, with a 215 GB in-memory `/tmp`, it took 10 minutes. The image runs only on the architecture it was built for:
where the login nodes are x86_64 and the GPU nodes aarch64, install and run `aimip` on a GPU node (in
a batch job or an interactive allocation) and use `--local` instead of `--submit`.

With Apptainer or Singularity, the `aimip` command on aarch64 unpacks the image for each call
(`apptainer exec --unsquash`) into `APPTAINER_TMPDIR` (default `/tmp`; about 15 GB free space needed) and Apptainer
removes the copy when the call ends; on the GH200 nodes we tested this adds about 9 seconds per call. The reason:
on those nodes Apptainer runs without setuid and serves the image through a FUSE squashfs mount, and there every cuFFT
call failed with `cuFFT error: CUFFT_INTERNAL_ERROR`, so every `fme` model stopped at its first step. The same
`libcufft` file read from a normal file system worked. `install-container.sh --unsquash no` turns this off, for example
where Apptainer mounts images with the kernel's squashfs driver; `--unsquash yes` turns it on elsewhere. For an install
made before this option existed, `export APPTAINER_UNSQUASH=1` has the same effect.

## Check the install

```bash
A=DIR/bin/aimip
$A --version
$A models                       # twenty-one models
$A run -m ace2-era5 --start 1979-01-01T00:00:00 --steps 8 --out <run root>/check --submit   # prepares the run
$A submit <run root>/check                                                     # queues it (or use --local above instead)
```

When the job has finished, `$A summary <run-dir>` prints `TMP2m 285.1578 K` for January 1979. This number was
reproduced exactly by a from-zero install in each mode (see [what was tested](#what-was-tested)).

## What was tested

Release 1.3.0 was installed from an empty directory in both modes on one Slurm GPU cluster (Apptainer 1.5, NVIDIA
A100-80GB, Python 3.12 for native mode) with fresh
downloads, and five models were run in each mode (the `samudrace-cm4` runs with autotuning off were made later,
for 1.3.1):
`ace2-era5` (8 steps), `hiro-acev2` (2 members), `samudrace-cm4` (6 coupled steps), `dlesym-v1.1` (2 members, 62 days)
and `archesclimate-ssp` (24 months).

- `ace2-era5`, `hiro-acev2`, `dlesym-v1.1` and `archesclimate-ssp`: container and native outputs are bit-identical
  in all output variables. `ace2-era5` gives `TMP2m 285.1578 K` for January 1979 in both. For `dlesym-v1.1`, the
  2 m temperature of member 0 also equals, bit for bit, an earlier run made with a different installation.
- `samudrace-cm4`: `fme` switches on cuDNN autotuning (`torch.backends.cudnn.benchmark`) on GPUs, which picks
  convolution algorithms by timing them. With autotuning on, repeated runs of the same 6-step configuration gave
  one of two bit-different results: both appeared in five native runs, and four container runs all gave the same
  one of the two. The two results differ by order one in the winds after six coupled steps. `aimip` keeps
  autotuning off while `fme` runs a coupled model (`samudrace-cm4`, `samudrace-e3smv3`), and the run manifest
  records the setting (`cudnn_benchmark`). With it off, six native runs on two GPU nodes and two container runs
  were all bit-identical, so container and native agree for this model in this test. These results differ from
  both earlier results: a coupled run whose manifest has no `cudnn_benchmark` entry, or has `true`, cannot be
  reproduced bit for bit with the default setting. `AIMIP_CUDNN_BENCHMARK=1` restores `fme`'s behaviour in
  native mode; a container run does not receive this variable. `samudrace-e3smv3` uses the same coupled
  inference and was not tested.
- The eight uncoupled `fme` models keep `fme`'s default, autotuning on. With it, `ace2-era5` gave
  `TMP2m 285.1578 K` in every install we made, and container and native outputs were bit-identical for
  `ace2-era5` and `hiro-acev2`. This is what we observed, not a guarantee: the same setting made the coupled
  runs vary. With autotuning off, `ace2-era5` gave 285.1581 K.
- aarch64: the container mode was built and run on GH200 nodes of one Slurm cluster (Apptainer 1.5.3 without setuid, image built
  on the node). With this version `ace2-era5` (8 steps) gives `TMP2m 285.1577 K`; with 1.3.0, `ace2s`, `ace2-som` and
  `samudrace-cm4` also ran. Results on GH200 are close to the A100 results but not bit-identical (`ace2-era5`:
  285.1577 K against 285.1578 K with autotuning on, 285.1577 K against 285.1581 K with it off), so compare GH200 runs
  with GH200 runs. Not tested on aarch64: native mode, Docker/Podman, the other models.
- Not tested in 1.3.x: the other nine models were not compared between the two modes; Docker/Podman; local mode
  without a scheduler; schedulers other than Slurm.
- `archesweather` and `archesweathergen` (added after 1.3.1, container mode, Apptainer, one A100-80GB): an image
  built from this version, `aimip fetch` from Zenodo (2.0 GiB downloaded, 10 files extracted, all checked against the
  lock), then `aimip run --submit`: `archesweather` 62 days and `archesweathergen` 2 members × 31 days. Both were
  bit-identical to the same rollouts run directly in the emulator environment. A 457-day `archesweather` run agrees
  with the authors' AIMIP submission in global-mean 2 m temperature ([models](models.md)). Native mode and other
  GPUs were not tested for these two models.

## Uninstall

Delete the prefix and your run root; with Docker or Podman also remove the image (`docker rmi <image>`). Nothing else
is written, except `~/.aimip/` if you put agent credentials there.
