"""Manage notebook discovery, execution, aggregation, and Slurm submission."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import fnmatch
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import tomllib

DEFAULT_MANIFEST_PATH = Path("notebooks.toml")
DEFAULT_RUNS_DIR = Path(".cache/notebook-runs")


class ManifestValidationError(ValueError):
    """Raised when the notebook manifest and on-disk inventory disagree."""


class TeeWriter:
    """Write text to multiple file-like objects."""

    def __init__(self, *targets: Any):
        self.targets = targets

    def write(self, data: str) -> int:
        """Write a string to every target and flush it immediately."""
        for target in self.targets:
            target.write(data)
            target.flush()
        return len(data)

    def flush(self) -> None:
        """Flush every target."""
        for target in self.targets:
            target.flush()


@dataclasses.dataclass(frozen=True)
class NotebookSpec:
    """Notebook execution settings loaded from the manifest."""

    path: str
    enabled: bool
    timeout_seconds: int | None = None
    kernel_name: str | None = None
    notes: str | None = None
    slurm_cpus_per_task: int | None = None
    slurm_mem: str | None = None


@dataclasses.dataclass(frozen=True)
class NotebookManifest:
    """Top-level notebook manifest settings."""

    manifest_path: Path
    default_timeout_seconds: int
    default_kernel_name: str
    ignored_globs: tuple[str, ...]
    notebooks: tuple[NotebookSpec, ...]


def now_utc() -> str:
    """Return an ISO8601 UTC timestamp."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def make_run_dir(base_dir: Path | None) -> Path:
    """Create a timestamped run directory when one is not explicitly provided."""
    if base_dir is not None:
        return base_dir.resolve()

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (DEFAULT_RUNS_DIR / timestamp).resolve()


def load_manifest(manifest_path: Path) -> NotebookManifest:
    """Load and validate the notebook manifest."""
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    notebooks: list[NotebookSpec] = []
    seen_paths: set[str] = set()
    for raw_notebook in data.get("notebooks", []):
        path = raw_notebook["path"]
        if path in seen_paths:
            msg = f"Duplicate notebook entry in manifest: {path}"
            raise ManifestValidationError(msg)
        seen_paths.add(path)
        notebooks.append(
            NotebookSpec(
                path=path,
                enabled=bool(raw_notebook.get("enabled", True)),
                timeout_seconds=raw_notebook.get("timeout_seconds"),
                kernel_name=raw_notebook.get("kernel_name"),
                notes=raw_notebook.get("notes"),
                slurm_cpus_per_task=raw_notebook.get("slurm_cpus_per_task"),
                slurm_mem=raw_notebook.get("slurm_mem"),
            )
        )

    return NotebookManifest(
        manifest_path=manifest_path.resolve(),
        default_timeout_seconds=int(data.get("default_timeout_seconds", 10000)),
        default_kernel_name=str(data.get("default_kernel_name", "python3")),
        ignored_globs=tuple(data.get("ignored_globs", [])),
        notebooks=tuple(notebooks),
    )


def matches_any_glob(relative_path: str, patterns: tuple[str, ...]) -> bool:
    """Return whether a relative path matches any ignore glob."""
    return any(fnmatch.fnmatch(relative_path, pattern) for pattern in patterns)


def find_disk_notebooks(repo_root: Path, ignored_globs: tuple[str, ...]) -> list[str]:
    """List notebook paths on disk, excluding generated notebook artifacts."""
    notebook_paths: list[str] = []
    for notebook_path in sorted((repo_root / "docs").rglob("*.ipynb")):
        relative_path = notebook_path.relative_to(repo_root).as_posix()
        if matches_any_glob(relative_path, ignored_globs):
            continue
        notebook_paths.append(relative_path)
    return notebook_paths


def discover_inventory(
    manifest: NotebookManifest,
    include_disabled: bool,
    selected_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate the manifest against disk and return the ordered inventory."""
    repo_root = manifest.manifest_path.parent
    disk_paths = find_disk_notebooks(repo_root, manifest.ignored_globs)
    disk_path_set = set(disk_paths)
    manifest_paths = [spec.path for spec in manifest.notebooks]
    manifest_path_set = set(manifest_paths)

    missing_from_manifest = sorted(disk_path_set - manifest_path_set)
    missing_from_disk = sorted(manifest_path_set - disk_path_set)
    problems: list[str] = []
    if missing_from_manifest:
        problems.append(
            "Notebooks exist on disk but are missing from notebooks.toml: "
            + ", ".join(missing_from_manifest)
        )
    if missing_from_disk:
        problems.append(
            "Notebooks exist in notebooks.toml but not on disk: "
            + ", ".join(missing_from_disk)
        )
    if problems:
        raise ManifestValidationError("\n".join(problems))

    inventory: list[dict[str, Any]] = []
    for index, spec in enumerate(manifest.notebooks):
        if selected_paths is not None and spec.path not in selected_paths:
            continue
        if not include_disabled and not spec.enabled:
            continue
        inventory.append(
            {
                "index": index,
                "path": spec.path,
                "enabled": spec.enabled,
                "timeout_seconds": spec.timeout_seconds
                or manifest.default_timeout_seconds,
                "kernel_name": spec.kernel_name or manifest.default_kernel_name,
                "notes": spec.notes,
                "slurm_cpus_per_task": spec.slurm_cpus_per_task,
                "slurm_mem": spec.slurm_mem,
            }
        )

    return inventory


def ensure_run_layout(run_dir: Path) -> None:
    """Create the directory structure for a notebook run."""
    for directory in (
        run_dir,
        run_dir / "results",
        run_dir / "logs",
        run_dir / "executed",
        run_dir / "submitted",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def slugify_notebook_path(notebook_path: str) -> str:
    """Convert a notebook path into a filesystem-safe slug."""
    return notebook_path.replace("/", "__").replace(".ipynb", "")


def inventory_path(run_dir: Path) -> Path:
    """Return the path to the frozen inventory snapshot."""
    return run_dir / "inventory.json"


def result_path(run_dir: Path, index: int, notebook_path: str) -> Path:
    """Return the result file path for a notebook index."""
    return (
        run_dir / "results" / f"{index:03d}-{slugify_notebook_path(notebook_path)}.json"
    )


def write_json(path: Path, payload: Any) -> None:
    """Serialize JSON with a stable layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def initialize_run(
    manifest: NotebookManifest,
    run_dir: Path,
    selected_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Persist the inventory snapshot and pre-create skipped results."""
    ensure_run_layout(run_dir)
    inventory = discover_inventory(
        manifest,
        include_disabled=True,
        selected_paths=selected_paths,
    )
    snapshot = {
        "created_at": now_utc(),
        "manifest_path": manifest.manifest_path.as_posix(),
        "default_timeout_seconds": manifest.default_timeout_seconds,
        "default_kernel_name": manifest.default_kernel_name,
        "ignored_globs": list(manifest.ignored_globs),
        "inventory": inventory,
    }
    write_json(inventory_path(run_dir), snapshot)

    for entry in inventory:
        if entry["enabled"]:
            continue
        skipped_payload = {
            "index": entry["index"],
            "path": entry["path"],
            "status": "skipped",
            "needs_attention": False,
            "timeout_seconds": entry["timeout_seconds"],
            "kernel_name": entry["kernel_name"],
            "started_at": None,
            "finished_at": None,
            "duration_seconds": 0.0,
            "notes": entry.get("notes"),
            "error_summary": None,
            "log_path": None,
            "executed_notebook_path": None,
        }
        write_json(result_path(run_dir, entry["index"], entry["path"]), skipped_payload)

    return inventory


def load_inventory_snapshot(run_dir: Path) -> list[dict[str, Any]]:
    """Load the frozen inventory snapshot for a run."""
    snapshot = json.loads(inventory_path(run_dir).read_text(encoding="utf-8"))
    return list(snapshot["inventory"])


def clean_notebook_for_execution(source_path: Path, cleaned_path: Path) -> None:
    """Write a cleaned copy of a notebook without mutating the source file."""
    import nbformat

    notebook = nbformat.read(source_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell["execution_count"] = None
        cell["outputs"] = []
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, cleaned_path)


def execute_notebook(
    repo_root: Path,
    notebook_entry: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Execute one notebook and record its outcome."""
    from nbclient.exceptions import CellExecutionError, CellTimeoutError
    from nbconvert.preprocessors import ExecutePreprocessor

    notebook_path = repo_root / notebook_entry["path"]
    cleaned_input_path = run_dir / "executed" / "cleaned" / notebook_entry["path"]
    executed_output_path = run_dir / "executed" / notebook_entry["path"]
    log_slug = slugify_notebook_path(notebook_entry["path"])
    log_file = run_dir / "logs" / f"{notebook_entry['index']:03d}-{log_slug}.log"

    started_at = now_utc()
    started_monotonic = time.monotonic()
    status = "passed"
    error_summary: str | None = None

    clean_notebook_for_execution(notebook_path, cleaned_input_path)

    with log_file.open("w", encoding="utf-8") as log_handle:
        stdout_writer = TeeWriter(log_handle, sys.stdout)
        stderr_writer = TeeWriter(log_handle, sys.stderr)
        try:
            import nbformat

            notebook = nbformat.read(cleaned_input_path, as_version=4)
            preprocessor = ExecutePreprocessor(
                timeout=int(notebook_entry["timeout_seconds"]),
                kernel_name=str(notebook_entry["kernel_name"]),
                allow_errors=False,
            )
            with (
                contextlib.redirect_stdout(stdout_writer),
                contextlib.redirect_stderr(stderr_writer),
            ):
                preprocessor.preprocess(
                    notebook,
                    {"metadata": {"path": str(notebook_path.parent)}},
                )
            executed_output_path.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, executed_output_path)
        except CellTimeoutError as exc:
            status = "timed_out"
            error_summary = str(exc)
            stderr_writer.write("\n" + traceback.format_exc())
        except CellExecutionError as exc:
            status = "failed"
            error_summary = str(exc)
            stderr_writer.write("\n" + traceback.format_exc())
        except Exception as exc:  # pragma: no cover - defensive classification
            status = "failed"
            error_summary = str(exc)
            stderr_writer.write("\n" + traceback.format_exc())

    duration_seconds = round(time.monotonic() - started_monotonic, 3)
    needs_attention = status in {"failed", "timed_out"}
    return {
        "index": notebook_entry["index"],
        "path": notebook_entry["path"],
        "status": status,
        "needs_attention": needs_attention,
        "timeout_seconds": notebook_entry["timeout_seconds"],
        "kernel_name": notebook_entry["kernel_name"],
        "started_at": started_at,
        "finished_at": now_utc(),
        "duration_seconds": duration_seconds,
        "notes": notebook_entry.get("notes"),
        "error_summary": error_summary,
        "log_path": log_file.as_posix(),
        "executed_notebook_path": executed_output_path.as_posix()
        if status == "passed"
        else None,
    }


def summarize_results(inventory: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    """Aggregate per-notebook results into a summary report."""
    statuses = {
        "passed": 0,
        "failed": 0,
        "timed_out": 0,
        "skipped": 0,
        "missing_result": 0,
    }
    results: list[dict[str, Any]] = []

    for notebook_entry in inventory:
        result_file = result_path(
            run_dir, notebook_entry["index"], notebook_entry["path"]
        )
        if not result_file.exists():
            missing_result = {
                "index": notebook_entry["index"],
                "path": notebook_entry["path"],
                "status": "missing_result",
                "needs_attention": True,
                "timeout_seconds": notebook_entry["timeout_seconds"],
                "kernel_name": notebook_entry["kernel_name"],
                "started_at": None,
                "finished_at": None,
                "duration_seconds": None,
                "notes": notebook_entry.get("notes"),
                "error_summary": "No result artifact was produced for this notebook.",
                "log_path": None,
                "executed_notebook_path": None,
            }
            results.append(missing_result)
            statuses["missing_result"] += 1
            continue

        result_payload = json.loads(result_file.read_text(encoding="utf-8"))
        results.append(result_payload)
        statuses[result_payload["status"]] += 1

    attention = [result for result in results if result["needs_attention"]]
    skipped = [result for result in results if result["status"] == "skipped"]
    return {
        "generated_at": now_utc(),
        "run_dir": run_dir.as_posix(),
        "summary": {
            "total": len(inventory),
            **statuses,
            "needs_attention": len(attention),
        },
        "attention": attention,
        "skipped": skipped,
        "results": results,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown report."""
    summary = report["summary"]
    lines = [
        "# Notebook execution report",
        "",
        f"Generated at: {report['generated_at']}",
        f"Run directory: {report['run_dir']}",
        "",
        "## Summary",
        "",
        f"- Total notebooks: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Timed out: {summary['timed_out']}",
        f"- Skipped: {summary['skipped']}",
        f"- Missing result: {summary['missing_result']}",
        f"- Need attention: {summary['needs_attention']}",
        "",
        "## Needs attention",
        "",
    ]
    if report["attention"]:
        for item in report["attention"]:
            lines.append(
                f"- {item['path']} [{item['status']}]"
                + (f": {item['error_summary']}" if item.get("error_summary") else "")
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Skipped", ""])
    if report["skipped"]:
        for item in report["skipped"]:
            suffix = f": {item['notes']}" if item.get("notes") else ""
            lines.append(f"- {item['path']}{suffix}")
    else:
        lines.append("- None")

    lines.append("")
    return "\n".join(lines)


def aggregate_run(run_dir: Path) -> dict[str, Any]:
    """Aggregate a run and persist JSON plus Markdown reports."""
    inventory = load_inventory_snapshot(run_dir)
    report = summarize_results(inventory, run_dir)
    write_json(run_dir / "report.json", report)
    (run_dir / "report.md").write_text(render_markdown_report(report), encoding="utf-8")
    return report


def parse_key_value_pairs(items: list[str]) -> dict[str, str]:
    """Parse repeated KEY=VALUE CLI items."""
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            msg = f"Expected KEY=VALUE format, got: {item}"
            raise ValueError(msg)
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def enabled_indexes_from_inventory(inventory: list[dict[str, Any]]) -> list[str]:
    """Return the enabled inventory indexes as strings."""
    return [str(entry["index"]) for entry in inventory if entry["enabled"]]


def command_discover(args: argparse.Namespace) -> int:
    """Handle the discover subcommand."""
    manifest = load_manifest(args.manifest.resolve())
    inventory = discover_inventory(manifest, include_disabled=args.include_disabled)
    if args.format == "json":
        print(json.dumps(inventory, indent=2))
    else:
        for entry in inventory:
            status = "enabled" if entry["enabled"] else "disabled"
            print(f"{entry['index']:03d} {status} {entry['path']}")
    return 0


def command_prepare_run(args: argparse.Namespace) -> int:
    """Handle the prepare-run subcommand."""
    manifest = load_manifest(args.manifest.resolve())
    run_dir = make_run_dir(args.run_dir)
    inventory = initialize_run(manifest, run_dir)
    enabled_indexes = enabled_indexes_from_inventory(inventory)
    payload = {
        "run_dir": run_dir.as_posix(),
        "enabled_indexes": enabled_indexes,
        "array_spec": ",".join(enabled_indexes),
        "inventory_size": len(inventory),
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    elif args.format == "array":
        print(payload["array_spec"])
    else:
        print(f"run_dir={payload['run_dir']}")
        print(f"array_spec={payload['array_spec']}")
        print(f"inventory_size={payload['inventory_size']}")
    return 0


def command_run_all(args: argparse.Namespace) -> int:
    """Handle the local batch execution subcommand."""
    manifest = load_manifest(args.manifest.resolve())
    run_dir = make_run_dir(args.run_dir)
    selected_paths = {args.notebook} if args.notebook else None
    inventory = initialize_run(manifest, run_dir, selected_paths=selected_paths)
    repo_root = manifest.manifest_path.parent

    for notebook_entry in inventory:
        if not notebook_entry["enabled"]:
            continue
        if args.notebook and notebook_entry["path"] != args.notebook:
            continue
        result_payload = execute_notebook(repo_root, notebook_entry, run_dir)
        write_json(
            result_path(run_dir, notebook_entry["index"], notebook_entry["path"]),
            result_payload,
        )

    report = aggregate_run(run_dir)
    print(render_markdown_report(report))
    if args.fail_on_error and report["summary"]["needs_attention"] > 0:
        return 1
    return 0


def command_run_one(args: argparse.Namespace) -> int:
    """Handle the single notebook execution subcommand."""
    manifest = load_manifest(args.manifest.resolve())
    run_dir = args.run_dir.resolve()
    ensure_run_layout(run_dir)

    selected_paths = {args.notebook} if args.notebook else None

    if inventory_path(run_dir).exists():
        inventory = load_inventory_snapshot(run_dir)
    else:
        if args.index is not None:
            full_inventory = discover_inventory(manifest, include_disabled=True)
            matching_entry = next(
                (entry for entry in full_inventory if entry["index"] == args.index),
                None,
            )
            if matching_entry is None:
                msg = "Requested notebook was not found in the inventory."
                raise ValueError(msg)
            selected_paths = {matching_entry["path"]}
        inventory = initialize_run(manifest, run_dir, selected_paths=selected_paths)

    selected_entry: dict[str, Any] | None = None
    if args.index is not None:
        for entry in inventory:
            if entry["index"] == args.index:
                selected_entry = entry
                break
    elif args.notebook:
        for entry in inventory:
            if entry["path"] == args.notebook:
                selected_entry = entry
                break
    else:
        msg = "Either --index or --notebook is required."
        raise ValueError(msg)

    if selected_entry is None:
        msg = "Requested notebook was not found in the inventory."
        raise ValueError(msg)

    if not selected_entry["enabled"]:
        print(f"Skipping disabled notebook: {selected_entry['path']}")
        return 0

    repo_root = manifest.manifest_path.parent
    result_payload = execute_notebook(repo_root, selected_entry, run_dir)
    write_json(
        result_path(run_dir, selected_entry["index"], selected_entry["path"]),
        result_payload,
    )
    print(json.dumps(result_payload, indent=2))
    return 1 if args.fail_on_error and result_payload["needs_attention"] else 0


def command_aggregate(args: argparse.Namespace) -> int:
    """Handle the aggregate subcommand."""
    report = aggregate_run(args.run_dir.resolve())
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(render_markdown_report(report))
    return 0


def command_submit_slurm(args: argparse.Namespace) -> int:
    """Handle Slurm array submission."""
    manifest = load_manifest(args.manifest.resolve())
    run_dir = make_run_dir(args.run_dir)
    inventory = initialize_run(manifest, run_dir)
    enabled_indexes = enabled_indexes_from_inventory(inventory)
    if not enabled_indexes:
        print("No enabled notebooks to submit.")
        return 0

    export_vars = parse_key_value_pairs(args.export)
    export_vars.setdefault("PYTENSOR_FLAGS", os.environ.get("PYTENSOR_FLAGS", ""))
    export_items = [f"{key}={value}" for key, value in sorted(export_vars.items())]
    array_index_token = "__SLURM_ARRAY_TASK_ID__"
    command_parts = [
        "uv",
        "run",
        "python",
        "scripts/notebook_jobs.py",
        "run-one",
        "--manifest",
        str(manifest.manifest_path),
        "--run-dir",
        str(run_dir),
        "--index",
        array_index_token,
        "--fail-on-error",
    ]
    wrap_command = shlex.join(command_parts).replace(
        array_index_token,
        "${SLURM_ARRAY_TASK_ID}",
    )
    sbatch_command = [
        "sbatch",
        f"--array={','.join(enabled_indexes)}",
        f"--job-name={args.job_name}",
        f"--output={run_dir / 'submitted' / 'slurm-%A_%a.out'}",
        f"--error={run_dir / 'submitted' / 'slurm-%A_%a.err'}",
        f"--export={','.join(['ALL', *export_items])}",
    ]
    if args.partition:
        sbatch_command.append(f"--partition={args.partition}")
    if args.account:
        sbatch_command.append(f"--account={args.account}")
    if args.time_limit:
        sbatch_command.append(f"--time={args.time_limit}")
    if args.cpus_per_task:
        sbatch_command.append(f"--cpus-per-task={args.cpus_per_task}")
    if args.mem:
        sbatch_command.append(f"--mem={args.mem}")
    for sbatch_arg in args.sbatch_arg:
        sbatch_command.append(sbatch_arg)
    sbatch_command.extend(["--wrap", wrap_command])

    submission_preview = {
        "run_dir": run_dir.as_posix(),
        "inventory_size": len(inventory),
        "enabled_indexes": enabled_indexes,
        "sbatch_command": sbatch_command,
    }
    write_json(run_dir / "submitted" / "submission.json", submission_preview)

    if args.dry_run:
        print(shlex.join(sbatch_command))
        return 0

    completed = subprocess.run(
        sbatch_command, check=False, capture_output=True, text=True
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the notebook manifest TOML file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser(
        "discover", help="Validate and print the notebook inventory."
    )
    discover_parser.add_argument(
        "--include-disabled", action="store_true", help="Include disabled notebooks."
    )
    discover_parser.add_argument("--format", choices=("text", "json"), default="text")
    discover_parser.set_defaults(handler=command_discover)

    prepare_parser = subparsers.add_parser(
        "prepare-run",
        help="Create a run directory and print the enabled Slurm array indexes.",
    )
    prepare_parser.add_argument(
        "--run-dir", type=Path, help="Directory to store run artifacts."
    )
    prepare_parser.add_argument(
        "--format", choices=("text", "json", "array"), default="text"
    )
    prepare_parser.set_defaults(handler=command_prepare_run)

    run_all_parser = subparsers.add_parser(
        "run-all", help="Run all enabled notebooks locally."
    )
    run_all_parser.add_argument(
        "--run-dir", type=Path, help="Directory to store run artifacts."
    )
    run_all_parser.add_argument(
        "--notebook", help="Restrict local execution to one notebook path."
    )
    run_all_parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return a failing exit code when attention is needed.",
    )
    run_all_parser.set_defaults(handler=command_run_all)

    run_one_parser = subparsers.add_parser(
        "run-one", help="Run one notebook by index or path."
    )
    run_one_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory that stores run artifacts.",
    )
    run_one_parser.add_argument(
        "--index", type=int, help="Notebook index from the discovered inventory."
    )
    run_one_parser.add_argument("--notebook", help="Notebook path from the manifest.")
    run_one_parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return a failing exit code when the notebook needs attention.",
    )
    run_one_parser.set_defaults(handler=command_run_one)

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="Build a report from result artifacts."
    )
    aggregate_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory that stores run artifacts.",
    )
    aggregate_parser.add_argument(
        "--format", choices=("markdown", "json"), default="markdown"
    )
    aggregate_parser.set_defaults(handler=command_aggregate)

    submit_parser = subparsers.add_parser(
        "submit-slurm", help="Submit the enabled notebooks as a Slurm job array."
    )
    submit_parser.add_argument(
        "--run-dir", type=Path, help="Directory to store run artifacts."
    )
    submit_parser.add_argument("--job-name", default="hssm-notebooks")
    submit_parser.add_argument("--partition")
    submit_parser.add_argument("--account")
    submit_parser.add_argument(
        "--time-limit", help="Slurm time limit, for example 08:00:00."
    )
    submit_parser.add_argument("--cpus-per-task", type=int)
    submit_parser.add_argument("--mem", help="Slurm memory request, for example 16G.")
    submit_parser.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Additional raw sbatch arguments.",
    )
    submit_parser.add_argument(
        "--export",
        action="append",
        default=[],
        help="Extra KEY=VALUE environment exports.",
    )
    submit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch command without submitting it.",
    )
    submit_parser.set_defaults(handler=command_submit_slurm)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the notebook jobs CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ManifestValidationError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
