"""install-container.sh --build stages every path the Dockerfile copies (a sprint user's clean checkout failed:
`cp: cannot stat 'docs'` once the Dockerfile copied docs/ and the staging list did not)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_context_contains_every_dockerfile_copy_source() -> None:
    sources: set[str] = set()
    for line in (ROOT / "container" / "Dockerfile").read_text(encoding="utf-8").splitlines():
        if line.startswith("COPY ") and "--from" not in line:
            *srcs, _dest = line.split()[1:]
            sources.update(s.split("/")[0] for s in srcs)
    script = (ROOT / "install" / "install-container.sh").read_text(encoding="utf-8")
    staged_line = next(ln for ln in script.splitlines() if "cp -R" in ln and "build/ctx" in ln)
    staged = set(re.findall(r'"\$REPO/([^"/]+)"', staged_line))
    missing = sorted(sources - staged)
    assert not missing, f"install-container.sh does not stage {missing} for the Docker build context"
    for s in sources:
        assert (ROOT / s).exists(), s
