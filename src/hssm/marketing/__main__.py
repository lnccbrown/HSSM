"""Command-line entry point for the governed Tier-S experiment runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments import available_templates, run_experiment


def main() -> int:
    """Validate and execute one registered experiment request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?", type=Path, help="request JSON")
    parser.add_argument("output", nargs="?", type=Path, help="empty output directory")
    parser.add_argument(
        "--list-templates", action="store_true", help="print registered templates"
    )
    args = parser.parse_args()
    if args.list_templates:
        print(json.dumps(available_templates(), indent=2, sort_keys=True))
        return 0
    if args.request is None or args.output is None:
        parser.error("request and output are required unless --list-templates is used")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    manifest = run_experiment(request, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
