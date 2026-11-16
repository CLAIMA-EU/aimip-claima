# Development

## Layout

| path | content |
|---|---|
| `src/aimip/` | the package: command line (`cli.py`), run preparation and execution (`runner.py`), model registry (`registry.py`, `registry.default.json`), downloads and the lock (`setup_data.py`, `aimip.lock`), derived forcing (`builders.py`), initial conditions and spin-up (`ic.py`), DLESyM/ArchesClimate support (`emu*.py`), LoRA tools (`lora.py`), status and summaries (`status.py`), the planning assistant (`agent/`) |
| `tests/` | the pytest suite (offline: synthetic data, fake runtimes) and `tests/wrapper/`, the shell tests of the host wrapper and the submit gate |
| `container/` | `Dockerfile` (the canonical image recipe), `gen_apptainer_def.py` (derives the Apptainer recipe from it), pinned requirements, the host wrapper `aimip-wrapper.in` |
| `install/` | `install-container.sh`, `install-native.sh`, site presets |
| `docs/` | this documentation |

## Tests

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0"
pip install -c container/constraints.txt numpy cftime netCDF4 h5netcdf xarray pyyaml pytest huggingface_hub \
    scipy pandas -r container/requirements-agent.txt
pip install --no-deps -e .
umask 022 && pytest -q
bash tests/wrapper/test_wrapper.sh
```

These are the steps CI runs.

The suite needs no GPU, no network and no model data. The image build and `install-native.sh` run it as their last
step, and CI (`.github/workflows/tests.yml`) on every push.

## The lock file

`src/aimip/aimip.lock` lists every downloaded or derived file with its sha256 (and size for downloads) and source (Hugging Face
repository and revision, Zenodo record, or the builder that derives it). `aimip setup` and `aimip fetch` refuse any
file that does not match. Derived files are built deterministically; changing a builder (including the attributes it
writes into a NetCDF file) changes their hashes, so regenerate the lock with `aimip setup --write-lock` from a
verified tree and review the diff.

## Adding a model

1. Pin its files: Hugging Face repository, revision and paths (see `emu_sources.json` for an example), sizes and
   sha256.
2. Add catalogue entries (`setup_data.build_artifact_catalog`) and lock entries.
3. Add the registry entry (`registry.build_default_registry_payload`) and regenerate `registry.default.json`.
4. If it does not run on `fme`, add a family runner (see `emu.py` / `emu_job.py`).
5. Add tests, and run the model once from a fresh install before release.

## Releases

The version is in `pyproject.toml`, `src/aimip/__init__.py` and `container/VERSION`. A new version makes the
container installer fetch (or, with `--build`, build) a new image instead of reusing an existing one. Whenever a pin
changes, regenerate `THIRD_PARTY-PACKAGES.md` from the built image with `container/list_licences.py`. Images are
published by `.github/workflows/image.yml` when a tag `v<version>` matching `container/VERSION` is pushed.
