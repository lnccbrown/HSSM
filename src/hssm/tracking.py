"""Opt-in MLflow tracking for HSSM inference runs.

Wrap a fitting workflow in :func:`track` and HSSM records the run to MLflow
following the ecosystem schema (HSSMSpine ``_docs/mlflow-schema.md``,
``phase=infer``): which model and network were fitted, the sampler settings,
convergence metrics, and, optionally, the traces as an artifact. The same
``lineage_id`` that ssm-simulators stamps on training data and LANfactory
carries into the published network is read back from the HuggingFace
``manifest.json`` so one query shows data -> network -> fits.

Nothing here runs unless :func:`track` is active. MLflow is an optional
dependency (``pip install hssm[tracking]``). Every hook is best-effort: a
tracking failure is logged and never interrupts inference.

Example
-------
>>> import hssm
>>> with hssm.track(experiment="infer/ddm", run_name="pilot-01"):
...     model = hssm.HSSM(data, model="ddm")
...     model.sample(draws=1000, chains=4)
...     model.save_model()
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal

if TYPE_CHECKING:  # pragma: no cover
    from hssm.base import HSSMBase

_logger = logging.getLogger("hssm")

#: MLflow run-schema version this module emits (HSSMSpine ``_docs/mlflow-schema.md``).
MLFLOW_SCHEMA_VERSION = "2"

#: Filename of the network registry LANfactory's ``upload-hf`` maintains at the
#: root of the HuggingFace repository.
MANIFEST_FILENAME = "manifest.json"

# The last network fetched from HuggingFace, recorded by the ONNX loader so a
# tracker started later can still say which file this model runs on.
_LAST_NETWORK: dict[str, str] = {}

_ACTIVE: Tracker | None = None


# --------------------------------------------------------------------------- #
# Network provenance (called from distribution_utils.onnx_utils.model)
# --------------------------------------------------------------------------- #


def hf_revision_from_path(local_path: str | os.PathLike) -> str | None:
    """Commit sha of a file in the huggingface_hub cache, or None.

    ``hf_hub_download`` returns ``.../snapshots/<commit sha>/<file>``; reading
    the sha off the path costs no network round-trip.
    """
    path = Path(local_path)
    if path.parent.parent.name == "snapshots":
        return path.parent.name
    return None


def record_network(filename: str, local_path: str | os.PathLike) -> None:
    """Remember which network file was just fetched from HuggingFace."""
    _LAST_NETWORK.clear()
    _LAST_NETWORK["network_file"] = str(filename)
    if revision := hf_revision_from_path(local_path):
        _LAST_NETWORK["hf_revision"] = revision


def last_network() -> dict[str, str]:
    """Return the most recently fetched network (``network_file``, ``hf_revision``)."""
    return dict(_LAST_NETWORK)


def _manifest_entry_for(network_file: str) -> dict[str, Any] | None:
    """Return the ``manifest.json`` record whose root network is ``network_file``."""
    try:
        from huggingface_hub import hf_hub_download

        from hssm.distribution_utils.onnx_utils.model import REPO_ID

        with open(hf_hub_download(repo_id=REPO_ID, filename=MANIFEST_FILENAME)) as f:
            manifest = json.load(f)
    except Exception as e:  # noqa: BLE001 - provenance is best-effort
        _logger.debug("Could not read %s from HuggingFace: %s", MANIFEST_FILENAME, e)
        return None
    for entry in manifest.get("networks", []):
        if entry.get("onnx_root") == network_file or network_file in entry.get(
            "files", []
        ):
            return entry
    return None


# --------------------------------------------------------------------------- #
# Common tags (mirrors ssm-simulators / LANfactory)
# --------------------------------------------------------------------------- #


def _git_sha() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else None


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def common_run_tags(lineage_id: str) -> dict[str, str]:
    """Tags every phase of the schema carries (v2 "common" block)."""
    import getpass
    import socket

    tags = {
        "schema_version": MLFLOW_SCHEMA_VERSION,
        "lineage_id": lineage_id,
        "hostname": socket.gethostname(),
    }
    with contextlib.suppress(KeyError, OSError):  # no passwd entry in containers
        tags["user"] = getpass.getuser()
    if sha := _git_sha():
        tags["git_sha"] = sha
    if v := _package_version("hssm"):
        tags["hssm_version"] = v
    for env_key, tag in (
        ("SLURM_JOB_ID", "slurm_job_id"),
        ("SLURM_ARRAY_JOB_ID", "slurm_array_job_id"),
        ("SLURM_ARRAY_TASK_ID", "slurm_array_task_id"),
    ):
        if os.getenv(env_key):
            tags[tag] = os.environ[env_key]
    return tags


# --------------------------------------------------------------------------- #
# Model description
# --------------------------------------------------------------------------- #


def _jsonable(value: Any) -> Any:
    """Return a JSON-serialisable stand-in for arbitrary model-spec values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return repr(value)


def model_spec(model: HSSMBase) -> dict[str, Any]:
    """Return the user-facing specification of a model, as JSON-able data.

    Built from the constructor arguments HSSM already stores for
    ``save_model`` (``_init_args``), minus the data itself, so two fits differ
    in ``spec_sha256`` exactly when the modelling choices differ.
    """
    init_args = dict(getattr(model, "_init_args", {}) or {})
    init_args.pop("data", None)
    spec = _jsonable(init_args)
    spec["model"] = getattr(model, "model_name", None)
    spec["loglik_kind"] = getattr(model, "loglik_kind", None)
    spec["repr"] = repr(model) if hasattr(model, "__repr__") else None
    return spec


def spec_sha256(spec: dict[str, Any]) -> str:
    """Stable hash of :func:`model_spec` (``repr`` excluded)."""
    canonical = {k: v for k, v in spec.items() if k != "repr"}
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, default=repr).encode()
    ).hexdigest()


def data_sha256(data: Any) -> str | None:
    """Content hash of the observed DataFrame."""
    try:
        import pandas as pd

        return hashlib.sha256(
            pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()
        ).hexdigest()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# The tracker
# --------------------------------------------------------------------------- #


class Tracker:
    """State for one tracked inference run. Obtained from :func:`track`."""

    def __init__(
        self,
        mlflow: Any,
        run_id: str,
        *,
        log_artifacts: bool | Literal["all"],
        lineage_id: str | None,
    ) -> None:
        self._mlflow = mlflow
        self.run_id = run_id
        self.log_artifacts = log_artifacts
        self._explicit_lineage_id = lineage_id
        self.lineage_id: str | None = None
        self._model_logged = False
        self._sample_started: float | None = None

    # -- helpers -----------------------------------------------------------

    def _guard(self, what: str, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - tracking never kills inference
            _logger.warning("MLflow tracking: failed to %s: %s", what, e)

    def _resolve_lineage(self, network: dict[str, str]) -> dict[str, str]:
        """Lineage tags: from the manifest of the network in use, else minted."""
        tags: dict[str, str] = {}
        if self._explicit_lineage_id:
            tags["lineage_id"] = self._explicit_lineage_id
            tags["lineage_source"] = "user"
        elif network.get("network_file") and (
            entry := _manifest_entry_for(network["network_file"])
        ):
            if entry.get("lineage_id"):
                tags["lineage_id"] = entry["lineage_id"]
                tags["lineage_source"] = "hf_manifest"
            if entry.get("mlflow_run_id"):
                tags["mlflow_run_id_train"] = entry["mlflow_run_id"]
            if entry.get("run_uuid"):
                tags["run_uuid_train"] = entry["run_uuid"]
        if "lineage_id" not in tags:
            tags["lineage_id"] = uuid.uuid4().hex
            tags["lineage_source"] = "minted"
        return tags

    # -- public logging API -----------------------------------------------

    def log_model(self, model: HSSMBase) -> None:
        """Log what is being fitted: model, network, data shape, spec hash."""
        if self._model_logged:
            return
        self._model_logged = True
        self._guard("log model", self._log_model, model)

    def _log_model(self, model: HSSMBase) -> None:
        mlflow = self._mlflow
        network = last_network()
        loglik = getattr(getattr(model, "model_config", None), "loglik", None)
        if isinstance(loglik, str) and not network.get("network_file"):
            network["network_file"] = loglik

        params: dict[str, Any] = {
            "model": getattr(model, "model_name", None),
            "loglik_kind": getattr(model, "loglik_kind", None),
        }
        params.update(network)
        data = getattr(model, "data", None)
        if data is not None:
            params["n_trials"] = int(len(data))
            if "participant_id" in getattr(data, "columns", []):
                params["n_subjects"] = int(data["participant_id"].nunique())
        spec = model_spec(model)
        params["spec_sha256"] = spec_sha256(spec)
        for pkg in ("hssm", "pymc", "bambi"):
            if v := _package_version(pkg):
                params[f"{pkg}_version"] = v
        mlflow.log_params({k: v for k, v in params.items() if v is not None})

        tags = self._resolve_lineage(network)
        self.lineage_id = tags["lineage_id"]
        if data is not None and (h := data_sha256(data)):
            tags["data_sha256"] = h
        mlflow.set_tags(tags)
        if self.log_artifacts:
            mlflow.log_dict(spec, "model_spec.json")

    def sample_started(self) -> None:
        """Mark the start of ``sample()`` for the ``sampling_seconds`` metric."""
        self._sample_started = time.monotonic()

    def log_sample(self, model: HSSMBase, sampler: str, kwargs: dict) -> None:
        """Log sampler settings, convergence metrics and (optionally) traces."""
        self.log_model(model)
        self._guard("log sample", self._log_sample, model, sampler, kwargs)

    def _log_sample(self, model: HSSMBase, sampler: str, kwargs: dict) -> None:
        mlflow = self._mlflow
        params: dict[str, Any] = {"sampler": sampler}
        for key in ("draws", "tune", "chains", "target_accept", "cores"):
            if key in kwargs:
                params[key] = kwargs[key]
        traces = getattr(model, "traces", None)
        posterior = getattr(traces, "posterior", None)
        if posterior is not None:
            sizes = getattr(posterior, "sizes", {})
            params.setdefault("draws", int(sizes.get("draw", 0)))
            params.setdefault("chains", int(sizes.get("chain", 0)))
        mlflow.log_params(params)

        metrics: dict[str, float] = {}
        if self._sample_started is not None:
            metrics["sampling_seconds"] = time.monotonic() - self._sample_started
        stats = getattr(traces, "sample_stats", None)
        if stats is not None and "diverging" in stats:
            metrics["divergences"] = float(stats["diverging"].sum())

        summary = None
        try:
            import arviz as az

            summary = az.summary(traces)
        except Exception as e:  # noqa: BLE001
            _logger.debug("az.summary unavailable for tracking: %s", e)
        if summary is not None and len(summary):
            if "r_hat" in summary:
                metrics["r_hat_max"] = float(summary["r_hat"].max())
            if "ess_bulk" in summary:
                metrics["ess_bulk_min"] = float(summary["ess_bulk"].min())
            if "ess_tail" in summary:
                metrics["ess_tail_min"] = float(summary["ess_tail"].min())
        # Single-chain fits yield NaN r_hat/ESS; a NaN metric is noise in the UI.
        metrics = {k: v for k, v in metrics.items() if math.isfinite(v)}
        if metrics:
            mlflow.log_metrics(metrics)

        if self.log_artifacts:
            with tempfile.TemporaryDirectory() as tmp:
                if summary is not None:
                    path = Path(tmp) / "summary.csv"
                    summary.to_csv(path)
                    mlflow.log_artifact(str(path))
                if traces is not None and hasattr(traces, "to_netcdf"):
                    path = Path(tmp) / "traces.nc"
                    traces.to_netcdf(path)
                    mlflow.log_artifact(str(path))

    def log_saved_model(self, model_path: Path) -> None:
        """Attach ``model.pkl`` if ``log_artifacts="all"``; called by ``save_model``."""
        if self.log_artifacts != "all":
            return
        pkl = Path(model_path) / "model.pkl"
        if pkl.exists():
            self._guard("log model.pkl", self._mlflow.log_artifact, str(pkl))

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Attach a matplotlib figure, e.g. ``log_figure(fig, "plots/trace.png")``."""
        self._guard("log figure", self._mlflow.log_figure, figure, artifact_file)

    def log_metric(self, key: str, value: float) -> None:
        """Attach an extra metric (e.g. ``loo_elpd``) to the run."""
        self._guard(f"log metric {key}", self._mlflow.log_metric, key, value)


def active() -> Tracker | None:
    """Return the tracker of the enclosing :func:`track` block, or None."""
    return _ACTIVE


@contextlib.contextmanager
def track(
    experiment: str | None = None,
    run_name: str | None = None,
    *,
    tracking_uri: str | None = None,
    log_artifacts: bool | Literal["all"] = True,
    lineage_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Tracker]:
    """Record the enclosed HSSM workflow as one MLflow run (``phase=infer``).

    Parameters
    ----------
    experiment
        MLflow experiment name. Defaults to ``MLFLOW_EXPERIMENT_NAME``; the
        ecosystem convention is ``infer/<model>``.
    run_name
        Optional run name.
    tracking_uri
        Tracking server. Defaults to ``MLFLOW_TRACKING_URI`` (the lab's shared
        server), then MLflow's own default.
    log_artifacts
        ``True`` attaches ``traces.nc``, ``summary.csv`` and ``model_spec.json``;
        ``"all"`` additionally attaches ``model.pkl`` when ``save_model`` is
        called inside the block; ``False`` logs params/metrics only.
    lineage_id
        Override the lineage id. By default it is read from the HuggingFace
        ``manifest.json`` entry of the network in use, else minted.
    tags
        Extra tags. ``schema_version``, ``phase`` and ``lineage_id`` are
        reserved and rejected.

    Raises
    ------
    ImportError
        If MLflow is not installed (``pip install hssm[tracking]``).
    """
    global _ACTIVE  # noqa: PLW0603 - the whole point is process-wide state

    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow is required for hssm.track(). Install it with "
            "`pip install hssm[tracking]` (or `uv sync --extra tracking`)."
        ) from exc

    reserved = {"schema_version", "phase", "lineage_id"} & set(tags or {})
    if reserved:
        raise ValueError(f"tags may not override reserved schema tags: {reserved}")
    if _ACTIVE is not None:
        raise RuntimeError("hssm.track() blocks cannot be nested")

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    exp = experiment or os.getenv("MLFLOW_EXPERIMENT_NAME")
    if exp:
        mlflow.set_experiment(exp)

    run = mlflow.start_run(run_name=run_name)
    tracker = Tracker(
        mlflow, run.info.run_id, log_artifacts=log_artifacts, lineage_id=lineage_id
    )
    # The common block goes on immediately with a placeholder lineage; it is
    # overwritten with the resolved id once a model is logged.
    initial = common_run_tags(lineage_id or "pending")
    initial["phase"] = "infer"
    initial.update(tags or {})
    tracker._guard("set tags", mlflow.set_tags, initial)
    _logger.info("MLflow tracking: started run %s", run.info.run_id)

    _ACTIVE = tracker
    try:
        yield tracker
    except BaseException:
        _ACTIVE = None
        with contextlib.suppress(Exception):
            mlflow.end_run(status="FAILED")
        raise
    else:
        _ACTIVE = None
        with contextlib.suppress(Exception):
            if tracker.lineage_id is None:
                # No model was ever logged; give the run a real lineage id.
                mlflow.set_tag("lineage_id", lineage_id or uuid.uuid4().hex)
            mlflow.end_run()


__all__ = [
    "MLFLOW_SCHEMA_VERSION",
    "Tracker",
    "active",
    "common_run_tags",
    "data_sha256",
    "hf_revision_from_path",
    "last_network",
    "model_spec",
    "record_network",
    "spec_sha256",
    "track",
]
