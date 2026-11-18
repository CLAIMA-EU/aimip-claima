# Job scripts, wall time and paths

This page describes the job scripts, wall-time rules and path checks of aimip-claima. The command is `aimip`.

## Job scripts

`aimip run … --submit` writes `job.sbatch` in the run directory:

Generated with the `generic-slurm` preset:

```bash
#!/bin/bash
# aimip-generated v1
#SBATCH --job-name=aimip-ace2-era5
#SBATCH --partition=gpu
#SBATCH --account=myproject
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=<run dir>/slurm-%j.out
#SBATCH --error=<run dir>/slurm-%j.err

set -euo pipefail
"<prefix>/bin/aimip" exec-fme "<run dir>"
```

`aimip submit <dir>` sends only scripts that aimip-claima generated: it checks every line (the marker, the allowed
`#SBATCH` options, `exec-fme` lines only for this install and this run directory; a spin job has one per member) and refuses anything else. To change
a setting, prepare the run again with the options below instead of editing the script.

Per-run options: `--partition`, `--account`, `--gres`, `--cpus-per-task`, `--mem`, `--time`, `--job-name`. Defaults
come from `<prefix>/site.json`; `aimip site show` prints them. `--hint=nomultithread` makes `--cpus-per-task` count
physical cores.

With the `local` preset there are no `#SBATCH` lines and `aimip submit` starts the script in the background on the
current machine.

## Wall time

aimip-claima sets `--time` itself: three times the expected run time with a minimum, capped at the site limit (12 hours
in the shipped presets).

| models | expected time on one A100 | minimum |
|---|---|---|
| fme models | 0.0644 GPU-hours (about 4 minutes) per model year and member | 20 minutes |
| DLESyM | about 40 s per model year and member | 45 minutes |
| ArchesClimate-SSP | about 2 s per month | 45 minutes |
| ArchesWeather | about 1.1 s per day (about 5 hours for 1978-10 … 2024-12) | 45 minutes |
| ArchesWeatherGen | about 4.6 s per day and member (about 21 hours per member for 1978-10 … 2024-12) | 45 minutes |

Short runs are dominated by start-up: an 8-step `ace2-era5` run finishes in under a minute of model time. When the
estimate exceeds the site limit, aimip-claima warns and uses the limit; for the fme models with a 12-hour limit that is
above about 60 member-years in one job. Split such runs into segments, or set `--time` yourself. DLESyM runs cannot
be continued yet: one run must fit in one job. ArchesWeather runs continue from `restart.nc`
([user-guide.md](user-guide.md#continuing-a-run)); a 12-hour job holds about 26 member-years of `archesweathergen`,
but the three-fold estimate reaches the 12-hour limit at about 8 member-years, so aimip-claima caps and warns above that.
Segments of up to about 20 member-years leave a margin.

Members of one run share one GPU and one job.

## Paths

- **Run root.** Run directories live under one directory, the run root (`aimip site show` prints its template;
  `${U1}` is the first letter of your user name). Without `--out`,
  `aimip run` names the directory `<run root>/<model>-<start>-<hash>`; with `--out` give an absolute path under the
  run root.
- **Allowed characters.** Paths must be absolute and may contain only letters, digits and `. _ / + -`; `..`, spaces
  and `~` are refused.
- **Ownership.** A run directory must be yours and must not be writable by your group or by others. aimip-claima creates
  it that way whatever your umask; a directory made writable by others is refused by `aimip submit`
  (`chmod go-w <dir>` fixes it).
- **Your own input files** (`--ic`, `--lora`) must be under the run root, the install, or a read-only root listed
  in `extra_ro_binds` of `site.json` (the Levante preset lists `/work`; to add one, edit `install/presets/<preset>.json` before installing — the
  installer writes `site.json` from it). In container mode the container sees nothing else; native mode applies the same rule.
- On clusters whose scratch file system is purged, copy results you want to keep elsewhere.
