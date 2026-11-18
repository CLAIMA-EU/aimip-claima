# What was tested

How the installations of aimip-claima were checked, and which results agree bit for bit between container and native
mode. The command is `aimip`. The
expected value of the first test run (`TMP2m 285.1578 K` for January 1979 with `ace2-era5`, 8 steps, on an A100) comes
from these tests. What each model was compared against (for example the authors' AIMIP submissions) is in
[models.md](models.md).

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
  one of the two. The two results differ by order one in the winds after six coupled steps. aimip-claima keeps
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
