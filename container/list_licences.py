"""Print name, version and declared licence of every installed distribution (tab separated).

Run it with each environment's Python inside the image to regenerate THIRD_PARTY-PACKAGES.md, e.g.
  apptainer exec -B container/list_licences.py:/x/list_licences.py:ro aimip.sif \
      /opt/aimip/venv/bin/python -I /x/list_licences.py
"""
import importlib.metadata as md, sys
rows = {}
for d in md.distributions():
    m = d.metadata
    name = m.get("Name") or "?"
    lic = m.get("License-Expression") or ""
    if not lic:
        cls = [c.split("::")[-1].strip() for c in (m.get_all("Classifier") or []) if c.startswith("License ::")]
        lic = "; ".join(cls)
    if not lic:
        l = (m.get("License") or "").strip().splitlines()
        lic = l[0][:60] if l else "UNKNOWN"
    rows[name.lower()] = (name, d.version, lic)
for k in sorted(rows):
    print("\t".join(rows[k]))
