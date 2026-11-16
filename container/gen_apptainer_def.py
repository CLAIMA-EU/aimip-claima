"""Generate apptainer.def from container/Dockerfile so the two container recipes stay in sync."""
import os
import re
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    df = fh.read().replace("\\\n", " ")

base = re.search(r"ARG BASE_IMAGE=(\S+)", df).group(1)
if len(sys.argv) > 3 and sys.argv[3]:
    base = sys.argv[3]
torch_index = re.search(r"ARG TORCH_INDEX=(\S+)", df).group(1)
runs = [r.strip().replace("${TORCH_INDEX}", torch_index) for r in re.findall(r"^RUN (.+)$", df, re.M)]
copies = re.findall(r"^COPY (.+) (\S+)$", df, re.M)
files = []
for srcs, dst in copies:
    for s in srcs.split():
        if dst.endswith("/"):
            files.append(f"    {s} {dst.rstrip('/')}/{os.path.basename(s.rstrip('/'))}")
        else:
            files.append(f"    {s} {dst}")
post = "\n".join("    " + r.replace(" && ", " \\\n      && ") for r in runs)
out = f"""Bootstrap: docker
From: {base}
# GENERATED from container/Dockerfile by gen_apptainer_def.py — do not edit by hand.

%files
{chr(10).join(files)}

%post
    set -e
    umask 022
    export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 PATH=/opt/aimip/venv/bin:$PATH
{post}

%environment
    export PYTHONNOUSERSITE=1 PATH=/opt/aimip/venv/bin:$PATH

%runscript
    exec python -B -m aimip "$@"
"""
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    fh.write(out)
