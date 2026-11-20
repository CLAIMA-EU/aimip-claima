# Installing aimip-claima

This page takes you from an empty directory to a working installation of aimip-claima and a first test run. The command
is `aimip`. Plan for about an hour: 10–45 minutes for the installer, then a 20 GB download and a few minutes on a GPU
for the test run.

Everything goes into one directory of your choice, the **prefix**:

```
/big/disk/aimip/            the prefix
├── bin/aimip               the command you will use
├── site.json               your cluster settings (partition, account, where runs go, ...)
├── images/                 container mode: the container image
├── venv/, venv-emu/, ...   native mode: the Python environments
└── data/                   model checkpoints and input data, downloaded when first needed
```

Your runs go to a second directory, the **run root**.

These pages describe the development version on `main`. For the latest release, check out its tag
(`git checkout v1.3.1`) and read the documentation in that checkout.

## 1. Check what you have

| you need | how to check |
|---|---|
| an NVIDIA GPU: A100, H100, GH200 or newer | `nvidia-smi` on a GPU node |
| an NVIDIA driver for CUDA 13.0 or newer | `nvidia-smi` shows `CUDA Version: 13.0` (or higher) in its first lines |
| a container runtime (Apptainer, Singularity, Docker or Podman), **or** Python 3.12 | `apptainer --version`, `singularity --version`, `docker --version`, `podman --version`, `python3.12 --version` |
| disk space: about 25 GB for the software, plus the models you use (20 GB for the first test model, about 260 GB for all of them) | `df -h /big/disk` |
| internet access where you install and where you prepare runs (none at all: see [clusters without internet](#clusters-without-internet-for-example-marenostrum-5)) | `python3 -c "import urllib.request; urllib.request.urlopen('https://huggingface.co')"` |
| `bash`, `git`, `python3` (3.8 or newer, for the installer itself), `tar`, `sha256sum` | |

Compute nodes do not need internet: models are downloaded when a run is prepared, normally on a login node.

On an older GPU (V100 and earlier) most models run, but `dlesym-v1.1` needs an extra build step; see
[Special situations](#special-situations).

## 2. Choose container or native mode

Answer one question: **does your machine have Apptainer, Singularity, Docker or Podman?**

- **Yes: use container mode.** Every library comes inside one image, so nothing on the machine can break it, and
  results are reproducible across machines. On HPC clusters this is usually Apptainer or Singularity.
- **No: use native mode.** The installer creates Python virtual environments. You need Python 3.12 and, for the
  DLESyM, ArchesClimate-SSP and ArchesWeather models, a C/C++ compiler.

Both modes give the same `aimip` command, the same run directories and, in our tests, the same numbers.

## 3. Choose two directories

- **The prefix** holds the software and the model data. Put it on a large disk, not in a home directory with a
  small quota: the models alone can take up to 260 GB.
- **The run root** is where your runs are written. A ten-year run writes tens of gigabytes. A scratch file system is
  fine, but copy results you want to keep if your site deletes old files there.

The examples below use `/big/disk/aimip` and `/big/disk/aimip-runs`. Replace them with your own paths. Paths may
contain only letters, digits and `. _ / + -` (no spaces).

## 4. Get the code

```bash
git clone https://github.com/CliDyn/aimip-claima.git
cd aimip-claima
```

## 5. Run the installer

Pick the case that matches your machine. Run the installer in `tmux` or `screen`, or with `nohup`: building the image
takes a while, and a dropped connection would stop it.

### 5a. Container mode on a Slurm cluster (Apptainer or Singularity)

```bash
export APPTAINER_TMPDIR=/tmp        # unpack images on a local disk (see "Apptainer is slow" below)

install/install-container.sh \
    --prefix /big/disk/aimip \
    --run-root /big/disk/aimip-runs \
    --site generic-slurm \
    --partition gpu \
    --account myproject \
    --build
```

| option | what to give |
|---|---|
| `--site generic-slurm` | the settings preset. Use `levante` on DKRZ Levante (then `--partition` is not needed), `local` for a machine without a scheduler |
| `--partition gpu` | the Slurm partition with GPUs on your cluster (`sinfo` lists them) |
| `--account myproject` | the account your GPU hours are charged to, or `none` if your cluster has no accounts |
| `--gres gpu:1` | optional: how to ask for one GPU, if your cluster needs another form (for example `gpu:a100:1`) |
| `--build` | builds the image on this machine (30–40 minutes, no root needed). Leave it out only for a release version: then the installer downloads the ready image `ghcr.io/clidyn/aimip-claima:<version>` instead |

The installer first writes the settings, so a missing option stops it within seconds. It then builds the image (log
in `/big/disk/aimip/build/build-<version>.log`), writes the command `/big/disk/aimip/bin/aimip` and tests it. It ends
like this:

```
WRAPPER_TESTS pass=... fail=0 bash=...
aimip-claima 1.4.0.dev0

Installed aimip-claima (container mode) 1.4.0.dev0 in /big/disk/aimip
Next:
  ...
```

The container sees your files read-only, except the run root and the data directory.

### 5b. Container mode on a workstation (Docker or Podman, no Slurm)

Docker and Podman need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
so that containers can use the GPU (`docker run --rm --gpus all ubuntu nvidia-smi` must work).

```bash
install/install-container.sh \
    --prefix /big/disk/aimip \
    --run-root /big/disk/aimip-runs \
    --site local \
    --runtime docker \
    --build
```

Use `--runtime podman` for Podman. With `--site local`, runs start on this machine instead of going to a queue.

### 5c. Native mode (no container)

```bash
install/install-native.sh \
    --prefix /big/disk/aimip \
    --run-root /big/disk/aimip-runs \
    --site generic-slurm \
    --partition gpu \
    --account myproject \
    --with-emu
```

The `--site`, `--partition` and `--account` options are the same as in 5a (`--site local` on a workstation). The
`--with-…` options choose which model environments to build:

| option | models | extra |
|---|---|---|
| none | the Ai2 models: the ACE family and SamudrACE | |
| `--with-emu` | also DLESyM, ArchesClimate-SSP, ArchesWeather and ArchesWeatherGen, Samudra-OM4, MD-1.5, cBottle-3d | needs a C/C++ compiler |
| `--with-camulator` | also CAMulator | a separate environment |
| `--with-sdyff` | also Spherical DYffusion | a separate environment |

`--python /path/to/python3.12` picks the interpreter if `python3.12` is not on your `PATH`. `--dry-run` checks the
options without installing anything. At the end the installer runs the test suite in the new environment and prints
the next steps.

## 6. Put `aimip` on your PATH

```bash
echo 'export PATH=/big/disk/aimip/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
aimip --version
```

Optional, tab completion of commands, model names and options:

```bash
echo 'source <(aimip completion bash)' >> ~/.bashrc     # zsh: aimip completion zsh
```

## 7. Check the installation

```bash
aimip doctor
```

```
PASS   site_config            site=generic-slurm scheduler=slurm runtime=apptainer partition=gpu account=myproject
PASS   runtime                container runtime=apptainer (...; image=/big/disk/aimip/images/aimip-1.4.0.dev0.sif)
INFO   gpu                    no GPU here — jobs run on GPU nodes via the scheduler (partition=gpu)
PASS   scheduler              Slurm tools found on host (sbatch, squeue, sacct, scancel)
PASS   data_root              /big/disk/aimip/data (... GiB free, 0 model(s) ready)
PASS   permissions            run_root=/big/disk/aimip-runs (owner-only writable, safe ancestry)
PASS   image_version          aimip-claima 1.4.0.dev0 matches /big/disk/aimip/VERSION
Overall: ok (7/7 passed)
```

On a login node without a GPU the `gpu` line is `INFO`, not a failure: the jobs run on the GPU nodes. Every failed
check prints what to do. `aimip site show` prints your settings, and `aimip models` lists the models.

## 8. Make a first test run

```bash
aimip run ace2-era5 --start 1979-01-01 --steps 8 --name first    # downloads ace2-era5 (20.5 GiB) the first time
aimip submit first
aimip watch first                                                 # waits until the job has finished
aimip summary first
```

The last command prints the global-mean 2 m temperature of January 1979:

```
MONTH    COUNTS  TMP2m (K)  surface_temperature (K)  air_temperature_7 (K)
-------  ------  ---------  -----------------------  ---------------------
1979-01  8       285.1577   285.8979                 282.9760
```

This is the output on a GH200. A `TMP2m` of `285.1578` on an A100 (`285.1577` on a GH200) means the installation
works. Without Slurm, run the first command with
`--local` added, and leave out `submit` and `watch`. The [user guide](user-guide.md) explains each step.

## Model data

Models are downloaded **when you first use them**: the first `aimip run` with a model downloads that model, checks
every file and prints what it downloads. To download in advance:

```bash
aimip fetch -m ace2-era5           # one model
aimip setup --models all           # all models: about 160 GiB of downloads, about 260 GB on disk
```

On clusters that limit work on login nodes, run `aimip setup` in a CPU job. `aimip show <model>` shows whether a
model is installed and how much it needs.

| model | download |
|---|---|
| `ace2-era5` | 20.5 GiB |
| `ace2-shield`, `ace2s-shield-plus`, `hiro-ace`, `ace2-eamv3` | checkpoints of 1.7–3.0 GiB each, plus the files their forcing is built from (SHiELD forcing 20.3 GiB, AIMIP SST 4.3 GiB, EAMv3 files 4.3 GiB) |
| `ace2-som` 8.7 · `ace2s` 7.6 · `hiro-acev2` 8.9 · `samudrace-cm4` 3.1 · `samudrace-e3smv3` 2.5 GiB | |
| `dlesym-v1.0` 0.16 · `dlesym-v1.1` 0.04 · `dlesym-v0-isccp` 0.29 GiB | plus 0.4 GB of ERA5 initial state for each run |
| `dlesym-aimip` | the `dlesym-v0-isccp` files plus the AIMIP SST forcing (4.3 GiB, shared with the ACE models that use it) |
| `archesclimate-ssp` | 33.5 GiB (both scenarios) |
| `archesweather`, `archesweathergen` | 2.0 GiB, plus 1.3 or 1.8 GiB of checkpoints unpacked from it |
| `samudra-om4` 1.5 · `md-1.5` 0.03 (plus the AIMIP SST) · `camulator` 5.9 · `cbottle-3d` 3.2 · `spherical-dyffusion` 25.3 GiB | `samudra-om4` also copies about 1.5 GB of ocean data per simulated year into each run |

Every file is checked against the lock file shipped with aimip-claima (`src/aimip/aimip.lock`): its size and its SHA-256
checksum must match, so you run exactly the files the model's authors published. Some forcing files are built on your
machine from the authors' data; they must match the lock bit for bit too.

### One installation for a whole group

Install once as one user (the owner); colleagues add `/big/disk/aimip/bin` to their `PATH` and use it with their
own run roots. Only the owner can download models: when someone else needs a model that is not there, aimip-claima says
`model X is not installed here: ask the install owner to run 'aimip fetch -m X'`.

aimip-claima refuses checkpoints and input files that someone else could have changed: they must belong to `root`, to you
or to the owner of the installation, and must not be writable by group or others (`chmod go-w <file>`).

## Special situations

**Using the container image directly.** In container mode `aimip` is a small host script that runs each command
inside the image (`<prefix>/images/aimip.sif`, or the Docker/Podman image) with the right mounts and environment, and
keeps runs inside your run root. To work with the image yourself:

```bash
aimip container-cmd run ace2-era5 --start 1979-01-01 --steps 8 --name first   # prints the apptainer/docker command
aimip container-shell                                                          # a shell inside the image
```

`container-cmd` prints the exact command the script would run for that `aimip` command (image, `-B` mounts, `--env`
settings, `--nv`) and runs nothing: copy it into your own job scripts or change it. `container-shell` opens a shell in
the image with the same mounts and settings; `aimip …` and `python -m aimip …` work inside. `submit`, `cancel`,
`watch`, `status`, `logs`, `chat` and `ui` are not available through `container-cmd`, since they act on the host (Slurm)
or pass credentials; run them with `aimip` as usual. When you change the printed command, the safety checks of the
script (paths inside your run root, no symlinks) are yours to keep.

**Compute nodes without internet.** Nothing to do: `aimip run` downloads what a run needs when it prepares the run,
on the machine where you type it. DLESyM (`dlesym-*`) also downloads its ERA5 start state then, and Samudra-OM4 its
ocean data.

**No Slurm, or another scheduler (PBS, LSF, ...).** Install with `--site local`. Inside an interactive or batch
allocation with a GPU, run `aimip run … --local`: the model runs at once. With your own job scripts: prepare the run
where there is internet (`aimip run …` without `--local`), and put the `aimip exec-fme <run dir>` line from its
`job.sbatch` into your job script.

**No internet anywhere on the cluster** (you can log in and copy files in, but nothing can be downloaded): see
[the next section](#clusters-without-internet-for-example-marenostrum-5).

**A web proxy.** Set `https_proxy` (and `http_proxy`) before installing and before `aimip run`.

**Apptainer is slow** (every `aimip` command takes minutes and prints "Converting SIF file to temporary sandbox").
Apptainer is unpacking the image on a slow shared file system. Point its temporary directory at a local disk, and add
the line to your `~/.bashrc`:

```bash
export APPTAINER_TMPDIR=/tmp
```

**The image build stops at `mksquashfs`** on a shared login node: too many threads. Run the installer again with
`AIMIP_BUILD_THREADS=2` in front of it.

**aarch64 machines (NVIDIA GH200).** There is no published aarch64 image, so the installer builds one (it switches
to `--build` by itself; about 10 minutes on a GH200 node with a fast local `/tmp`). The image runs only on the
architecture it was built for. Where the login nodes are x86_64 and the GPU nodes aarch64, install and use aimip-claima
on a GPU node, in a batch job or an interactive allocation, and run with `--local`.

On aarch64, the `aimip` command unpacks the image for every call (about 9 seconds and 15 GB in `APPTAINER_TMPDIR`). On
the GH200 nodes we tested, the fme models failed at their first step with `cuFFT error: CUFFT_INTERNAL_ERROR` when the
image was read from a mounted image file, and worked when it was unpacked. `--unsquash no` (or `yes`) on the installer
switches this off (or on, on other machines).

**GPUs older than A100.** The DLESyM HEALPix kernel is compiled for A100, H100 and GH200 (compute capability 8.0 and
9.0). For another GPU, add its compute capability to `TORCH_CUDA_ARCH_LIST` in `container/Dockerfile` (then install
with `--build`) or in `install/install-native.sh`.

## Clusters without internet (for example MareNostrum 5)

On some clusters no node can reach the internet. You can log in with SSH and copy files in with `scp` or `rsync`, but
the cluster cannot download anything: the installer cannot fetch or build the image, and aimip-claima cannot download
models. Prepare everything on a second machine that has internet, then copy it over.

The second machine needs Linux on the same processor type as the cluster (x86_64 for MareNostrum 5), Apptainer or
Singularity, and disk space for the models you want. It needs no GPU. A login node of another cluster or a Linux
workstation will do.

**On the machine with internet:**

1. Install aimip-claima in container mode as in [section 5](#5-run-the-installer), with `--site local` (these site
   settings are not copied; leave out `--build` for a release version):

   ```bash
   git clone https://github.com/CliDyn/aimip-claima.git
   cd aimip-claima
   install/install-container.sh --prefix /big/disk/aimip --run-root /big/disk/aimip-runs --site local --build
   ```

2. Download the models you will use:

   ```bash
   aimip fetch -m ace2-era5
   aimip fetch -m archesweather
   ```

3. Put everything the cluster needs for these models into one new directory, the bundle:

   ```bash
   aimip bundle -m ace2-era5,archesweather --out /big/disk/aimip-bundle
   ```

   It holds the container image, every file of the models (including the files their forcing is built from) and the
   aimip-claima source of the same version, with the steps for the cluster in `BUNDLE.md`. It is laid out like an
   installation directory. For ArchesWeather alone it is 7.2 GiB, 4.3 GiB of it the image.

4. Copy it to the cluster, to where the installation should live there:

   ```bash
   rsync -a /big/disk/aimip-bundle/ mycluster:/gpfs/projects/myproject/aimip/
   ```

**On the cluster:**

5. If the cluster provides Apptainer or Singularity as a module, load it (`module avail apptainer singularity`
   shows what there is). Then run the installer from the bundle, with the bundle as the prefix, the cluster's settings
   and **without** `--build`. It finds the image, checks that it holds the same version as the source, and downloads
   nothing:

   ```bash
   cd /gpfs/projects/myproject/aimip/aimip-claima
   install/install-container.sh --prefix /gpfs/projects/myproject/aimip \
       --run-root /gpfs/scratch/myproject/aimip-runs \
       --site generic-slurm --partition <partition> --account <project>
   ```

6. Check the copied files against the lock file (nothing is downloaded when they are all there) and the installation:

   ```bash
   export PATH=/gpfs/projects/myproject/aimip/bin:$PATH
   aimip fetch -m ace2-era5
   aimip fetch -m archesweather
   aimip doctor
   ```

   When you prepare runs, add `--no-fetch`: if a file is missing, the run then stops at once with a clear message
   instead of waiting for a network that is not there.

To add models later, fetch them on the machine with internet, make a bundle of the new models, copy its `data/`
directory into the installation on the cluster (`rsync -a <bundle>/data/ mycluster:/gpfs/projects/myproject/aimip/data/`)
and run `aimip fetch -m <model>` there. To update aimip-claima, make a new bundle with the new version and install
from it.

What does not work on such a cluster:

- **DLESyM** (`dlesym-*`) and **Samudra-OM4**: they download their starting data each time a run is prepared (ERA5
  from Google Cloud, ocean states from OSN); `aimip bundle` refuses them.
- **Native mode**: it installs Python packages from the internet.
- **CAMulator** from start dates other than 1980-01-01 and 1981-01-01: `aimip fetch` downloads only the files for
  these two. For another start, prepare one run with that start on the machine with internet (`aimip run camulator
  --start …` downloads the extra files) before you make the bundle; it includes the extra files it finds.
- **Docker or Podman** on the machine with internet: `aimip bundle` needs the Apptainer or Singularity image file.

**Job options that aimip-claima does not write,** such as the QoS that MareNostrum 5 uses to choose a queue: set them as
Slurm environment variables before `aimip submit`, for example `export SBATCH_QOS=<your QoS>`. `sbatch` reads
these variables, and they take precedence over the job script.

## Changing the settings later

`aimip site show` prints the current defaults. Every run can override them (`--partition`, `--account`, `--gres`,
`--cpus-per-task`, `--mem`, `--time`; see the [user guide](user-guide.md#running-on-a-cluster)). To change the
defaults, run the installer again with the same `--prefix` and the new options: it keeps the image and the data that
are already there.

## Updating

```bash
cd aimip-claima
git pull
install/install-container.sh --prefix /big/disk/aimip ... --build    # the same options as before
```

Run the installer again with the same options (native mode: `install/install-native.sh` as before). A new version
builds a new image; the model data are kept. Each run directory records the aimip-claima version it was prepared with.

## Uninstalling

Delete the prefix and your run root. With Docker or Podman also remove the image (`docker rmi <image>`). aimip-claima writes
nothing else, except `~/.aimip/` if you stored credentials for the assistant there.

## Installer options

| option | container | native | meaning |
|---|---|---|---|
| `--prefix DIR` | yes | yes | where to install (required) |
| `--site generic-slurm \| local \| levante` | yes | yes | the settings preset: any Slurm cluster, a machine without a scheduler, or DKRZ Levante |
| `--partition P`, `--account A\|none`, `--gres G\|none` | yes | yes | Slurm defaults for every run |
| `--run-root R` | yes | yes | where runs go (default: `$HOME/aimip-runs`; on Levante `/scratch/<letter>/<user>/aimip-runs`) |
| `--build` | yes | | build the image here instead of downloading it |
| `--image REF` | yes | | use another image, for example a copy in your site's registry |
| `--runtime auto\|apptainer\|singularity\|docker\|podman` | yes | | which container runtime to use (default: the first one found, in this order: Apptainer, Singularity, Podman, Docker) |
| `--unsquash auto\|yes\|no` | yes | | unpack the image for every call (default: on aarch64 only) |
| `--with-emu`, `--with-camulator`, `--with-sdyff` | | yes | which model environments to build (table in 5c) |
| `--python PATH` | | yes | the Python 3.12 interpreter to use |
| `--dry-run` | | yes | check the options and write the settings only |

The presets are in `install/presets/`. The defaults of `generic-slurm`: one GPU (`gpu:1`), 16 CPUs, 64 GB of memory,
at most 12 hours per job.

How the installations were tested, and which results agree bit for bit between the two modes, is in
[testing.md](testing.md).
