"""The emulator job keeps multiprocessing's socket directory in the short /tmp even under a long inherited TMPDIR."""

from __future__ import annotations

import multiprocessing.util
import os
from pathlib import Path

import pytest

from aimip import emu_job


def test_setup_tmp_keeps_af_unix_dir_short_under_long_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    long_dir = tmp_path / ("very-long-run-directory-name-" * 4) / "tmp"
    long_dir.mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(long_dir))
    monkeypatch.setattr(emu_job.tempfile, "tempdir", None)
    monkeypatch.setenv("AIMIP_EMU_TMP", str(long_dir / "emu"))
    monkeypatch.setattr(multiprocessing.util.process.current_process(), "_config",
                        dict(multiprocessing.util.process.current_process()._config), raising=False)
    multiprocessing.util.process.current_process()._config.pop("tempdir", None)

    tmp = emu_job._setup_tmp()

    mp_dir = multiprocessing.util.get_temp_dir()
    assert not mp_dir.startswith(str(long_dir)), mp_dir
    assert len(os.path.join(mp_dir, "listener-xxxxxxxx")) < 107
    assert tmp == str(long_dir / "emu")                     # the job's own files still go to the run's temp dir
    assert os.environ["TMPDIR"] == tmp
