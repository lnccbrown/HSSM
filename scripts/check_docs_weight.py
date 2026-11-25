#!/usr/bin/env python3
"""Guard the documentation tree against silent weight growth.

Committed notebook outputs are what make these docs usable without a GPU, but
they are also the dominant cost in this repository's history: notebooks are
~55% of all pack blobs, because every re-execution commits a fresh multi-MB
blob that stays forever. This check makes that cost visible at review time
rather than in a clone six months later.

It flags two things:
  * notebooks larger than --max-notebook-mb (default 4), and
  * non-notebook files under docs/ larger than --max-asset-mb (default 5),
    which are usually artifacts that belong in a release or on HuggingFace.

Existing offenders are grandfathered via ALLOWLIST so the check can be
switched on without a flag day; entries there are debt, not policy.
"""

import argparse
import json
import pathlib
import sys

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"

# Known-heavy paths that predate the check. Shrink and remove, do not extend.
# Sizes at the time of writing (2026-08): scientific_workflow 11.2 MB and
# plotting 5.4 MB (down from 10.9 before the retina fix).
ALLOWLIST = {
    "tutorials/scientific_workflow_hssm.ipynb",
    "tutorials/plotting.ipynb",
    "archive/hssm_tutorial_workshop_1.ipynb",
    "archive/hssm_tutorial_workshop_2.ipynb",
}
# Sampler-output caches the notebooks restore instead of re-sampling, and the
# model directory save_load_tutorial writes. Tracked deliberately, but listed
# file by file: a prefix rule would silently admit the next 200 MB artifact
# dropped into the same directory.
ALLOWLIST_ASSETS = {
    "tutorials/idata/basic_ddm/traces.nc",
    "tutorials/scientific_workflow_hssm/idata/angle_hier/traces.nc",
    "tutorials/scientific_workflow_hssm/idata/angle_hier_v2/traces.nc",
    "tutorials/scientific_workflow_hssm/idata/angle_hier_v3/traces.nc",
    "tutorials/scientific_workflow_hssm/idata/angle_hier_v4/traces.nc",
    "tutorials/scientific_workflow_hssm/idata/angle_v5/traces.nc",
    "tutorials/scientific_workflow_hssm/idata/basic_ddm/traces.nc",
    "tutorials/scientific_workflow_hssm/idata/ddm_hier/traces.nc",
    # written by save_load_tutorial when it runs; committed so the rendered
    # page shows a populated model directory (see #1186 for retiring these)
    "tutorials/hssm_models/test_model/traces.nc",
}


def _mb(p: pathlib.Path) -> float:
    return p.stat().st_size / 1_000_000


def main() -> int:
    """Report docs files over the size limits; return 1 if any."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-notebook-mb", type=float, default=4.0)
    ap.add_argument("--max-asset-mb", type=float, default=5.0)
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    args = ap.parse_args()

    offenders, report = [], []
    for p in sorted(DOCS.rglob("*")):
        if not p.is_file() or ".ipynb_checkpoints" in str(p):
            continue
        rel = str(p.relative_to(DOCS))
        if rel in ALLOWLIST or rel in ALLOWLIST_ASSETS:
            continue
        size = _mb(p)
        limit = args.max_notebook_mb if p.suffix == ".ipynb" else args.max_asset_mb
        if size > limit:
            offenders.append((rel, size, limit))
        report.append({"path": rel, "mb": round(size, 2)})

    if args.json:
        print(json.dumps({"offenders": offenders, "files": report}, indent=2))
    else:
        for rel, size, limit in offenders:
            print(f"TOO LARGE: {rel} — {size:.1f} MB (limit {limit:.0f} MB)")
        if offenders:
            print(
                "\nStrip stream outputs, drop `figure_format='retina'`, or move the "
                "artifact out of docs/ (release asset or HuggingFace)."
            )
        else:
            print(f"ok: no file over limit ({len(report)} checked)")
    return 1 if offenders else 0


if __name__ == "__main__":
    sys.exit(main())
