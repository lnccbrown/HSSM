"""Opt-in MLflow tracking for HSSM inference runs.

Wrap a fitting workflow in :func:`track` and each fit is recorded as one MLflow
run: the model and its priors, the sampler settings, convergence diagnostics,
the traces, and anything you add yourself. Fits of the same data, or of the same
model specification, are tagged alike, so a study of several models stays
legible afterwards.

Nothing here runs unless :func:`track` is active, MLflow is an optional
dependency (``uv add "hssm[tracking]"``), and every hook is best-effort: a
tracking failure is logged and never interrupts inference.

Where the likelihood is a network published by the LAN pipeline, the run also
carries that network's provenance, which links a fit back to the data the
network was trained on. The schema shared across the ecosystem is documented in
HSSMSpine ``_docs/mlflow-schema.md``.

Example
-------
>>> import hssm
>>> data = hssm.load_data("cavanagh_theta")
>>> with hssm.track(experiment="my-study", run_name="ddm-basic"):
...     model = hssm.HSSM(data, model="ddm")
...     model.sample(draws=1000, chains=4)
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
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    from mlflow import MlflowClient
    from mlflow.entities import Run

    from hssm import HSSM
    from hssm.base import HSSMBase

_logger = logging.getLogger("hssm")


__all__ = [
    "Tracker",
    "active",
    "list_runs",
    "load_run",
    "track",
]

#: MLflow run-schema version this module emits (HSSMSpine ``_docs/mlflow-schema.md``).
MLFLOW_SCHEMA_VERSION = "2"

#: Filename of the network registry LANfactory's ``upload-hf`` maintains at the
#: root of the HuggingFace repository.
MANIFEST_FILENAME = "manifest.json"

#: Param keys this module logs itself. A caller-supplied param of the same name
#: would be a second value for an existing key, which MLflow rejects for the
#: *whole batch* — so they are refused up front, as reserved tags are.
RESERVED_PARAMS = frozenset(
    {
        "model",
        "loglik_kind",
        "network_file",
        "hf_revision",
        "n_trials",
        "n_subjects",
        "spec_sha256",
        "hssm_version",
        "pymc_version",
        "bambi_version",
        "sampler",
        "draws",
        "tune",
        "chains",
        "target_accept",
        "cores",
        "method",
        "niter",
        "backend",
    }
)

#: Artifacts :func:`load_run` rebuilds a model from; the traces are optional.
REBUILD_ARTIFACTS = frozenset({"data.parquet", "model_spec.json"})

#: How many runs :func:`list_runs` asks the server for per request.
RUNS_PAGE_SIZE = 1000

# The last network fetched from HuggingFace, recorded by the ONNX loader so the
# model being built can claim it. A ContextVar for the same reason as `_ACTIVE`
# below: two models constructed concurrently would otherwise interleave on one
# dict and could end up claiming each other's network. Always replaced
# wholesale, never mutated in place.
_LAST_NETWORK: ContextVar[dict[str, str] | None] = ContextVar(
    "hssm_last_network", default=None
)

# Which tracker the enclosing `track()` block installed. A ContextVar rather
# than a plain global: a new thread starts with a fresh context, so an untracked
# `sample()` running alongside a tracked one cannot see — and so cannot mutate —
# the tracked run's timing and model state.
_ACTIVE: ContextVar[Tracker | None] = ContextVar("hssm_active_tracker", default=None)


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
    record: dict[str, str] = {"network_file": str(filename)}
    if revision := hf_revision_from_path(local_path):
        record["hf_revision"] = revision
    _LAST_NETWORK.set(record)


def last_network() -> dict[str, str]:
    """Return the most recently fetched network (``network_file``, ``hf_revision``).

    Scoped to the execution context, so a download in another thread is not
    mistaken for this one's.
    """
    return dict(_LAST_NETWORK.get() or {})


def reset_network_record() -> None:
    """Forget the last recorded network.

    Called before a model builds its likelihood so that what is recorded
    afterwards belongs to *that* model. Without it a model which downloads
    nothing — an analytical likelihood, say — would inherit whichever network
    the previous model in the session happened to load.
    """
    _LAST_NETWORK.set(None)


def _manifest_entry_for(
    network_file: str, revision: str | None = None
) -> dict[str, Any] | None:
    """Return the ``manifest.json`` record whose root network is ``network_file``.

    ``revision`` is the repository commit the network itself came from. Reading
    the manifest at that same commit keeps the two consistent: the manifest on
    the default branch may have moved on since, and would then describe a
    different network under the same filename.
    """
    try:
        from huggingface_hub import hf_hub_download

        from hssm.distribution_utils.onnx_utils.model import REPO_ID

        # `revision=None` is hf_hub_download's own default (the default
        # branch), so the unpinned case needs no special handling.
        with open(
            hf_hub_download(
                repo_id=REPO_ID, filename=MANIFEST_FILENAME, revision=revision
            )
        ) as f:
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


def check_params(params: dict[str, Any] | None) -> None:
    """Raise if any key collides with one this module logs itself.

    MLflow refuses to record a second, different value for a param, and it
    rejects the entire batch when one key offends — so a silent collision would
    cost the run every other param too. Failing here says which key to rename.
    """
    clash = RESERVED_PARAMS & set(params or {})
    if clash:
        raise ValueError(
            f"params may not override what HSSM records itself: {sorted(clash)}"
        )


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


def _is_restorable(value: Any) -> bool:
    """Whether :func:`_jsonable` keeps ``value`` as data rather than ``repr`` text.

    A ``repr`` placeholder is an ordinary string in the JSON, so a spec that
    contains one looks valid but cannot be passed back to a constructor.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_restorable(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_is_restorable(v) for v in value)
    return False


def spec_is_restorable(model: HSSMBase) -> bool:
    """Whether :func:`model_spec` records ``model`` well enough to rebuild it."""
    init_args = dict(getattr(model, "_init_args", {}) or {})
    init_args.pop("data", None)
    return _is_restorable(init_args)


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
    """Content hash of the observed DataFrame, schema included.

    Values alone are not enough: renaming a column, or reading the same numbers
    back as a different dtype, gives a frame that means something else while
    hashing identically. Column labels and dtypes therefore go into the digest
    alongside the row and index values.
    """
    try:
        import pandas as pd

        digest = hashlib.sha256()
        digest.update(pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes())
        schema = [(str(c), str(dt)) for c, dt in zip(data.columns, data.dtypes)]
        digest.update(repr(schema).encode())
        return digest.hexdigest()
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
        dataset_name: str | None = None,
    ) -> None:
        self._mlflow = mlflow
        self.run_id = run_id
        self.log_artifacts = log_artifacts
        self._explicit_lineage_id = lineage_id
        self._dataset_name = dataset_name
        self._data_uri: str | None = None
        self.lineage_id: str | None = None
        self._model_logged = False
        self._sample_started: float | None = None
        self._vi_started: float | None = None

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
            entry := _manifest_entry_for(
                network["network_file"], network.get("hf_revision")
            )
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
        # The model carries the network it was built with, and an empty record
        # means it uses none. Falling back to `last_network()` here would undo
        # that: a model with no network of its own would pick up whichever one
        # was loaded most recently in the session.
        network = dict(getattr(model, "_tracking_network", None) or {})
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
        # What `load_run` checks before trying to rebuild the model.
        tags["model_class"] = type(model).__name__
        tags["spec_restorable"] = str(spec_is_restorable(model)).lower()
        mlflow.set_tags(tags)
        if self.log_artifacts:
            mlflow.log_dict(spec, "model_spec.json")
            if data is not None:
                self._guard("log data", self._log_data, data)
        if data is not None and self._dataset_name:
            self._guard("log dataset", self._log_dataset, data)

    def _log_data(self, data: Any) -> None:
        """Attach the data the model was fit to, as ``data.parquet``.

        The traces keep only the observed response, so without this a run
        cannot say which covariates a regression used. Parquet keeps the
        dtypes that a CSV round trip would lose. Logged before the dataset is
        registered, so the Datasets panel can point at this copy.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.parquet"
            data.to_parquet(path)
            self._mlflow.log_artifact(str(path))
        self._data_uri = self._mlflow.get_artifact_uri("data.parquet")

    def _log_dataset(self, data: Any) -> None:
        """Register the observed data as an MLflow dataset.

        This fills the run's "Datasets" panel with the column schema and row
        count, and makes the UI's dataset filter work. When ``data.parquet``
        was logged, the entry's source points at it. It needs a name, which a
        DataFrame does not carry, so it happens only when the caller supplies
        one — an unnamed dataset would show up identically for every study.
        """
        import mlflow.data

        # `from_pandas` exists at runtime in mlflow>=3.14 but neither checker
        # can see it, so both are told to stand down here.
        # pyrefly: ignore[missing-attribute]
        from_pandas = mlflow.data.from_pandas  # type: ignore[attr-defined]
        try:
            dataset = from_pandas(data, name=self._dataset_name, source=self._data_uri)
        except Exception:  # noqa: BLE001 - a missing link must not cost the entry
            if self._data_uri is None:
                raise
            # MLflow resolves a dataset source by URI scheme, and some stores
            # have no resolver — notably a server run with --serve-artifacts,
            # whose artifact URIs are `mlflow-artifacts:/...`. Register the
            # dataset without the link to the stored copy rather than lose it.
            dataset = from_pandas(data, name=self._dataset_name)
        self._mlflow.log_input(dataset)

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

        self._log_idata(traces, metrics, artifact_name="traces.nc")

    def _log_idata(
        self,
        idata: Any,
        metrics: dict[str, float],
        *,
        artifact_name: str,
    ) -> None:
        """Summarise an inference result: posterior metrics, summary, the idata.

        Shared by the MCMC and variational paths, which differ in what they call
        the result and in which metrics they bring with them.
        """
        mlflow = self._mlflow
        summary = None
        try:
            import arviz as az

            summary = az.summary(idata)
        except Exception as e:  # noqa: BLE001
            _logger.debug("az.summary unavailable for tracking: %s", e)
        if summary is not None and len(summary):
            if "r_hat" in summary:
                metrics["r_hat_max"] = float(summary["r_hat"].max())
            if "ess_bulk" in summary:
                metrics["ess_bulk_min"] = float(summary["ess_bulk"].min())
            if "ess_tail" in summary:
                metrics["ess_tail_min"] = float(summary["ess_tail"].min())
        # Single-chain fits (and every VI fit) yield NaN r_hat/ESS; a NaN metric
        # is noise in the UI.
        metrics = {k: v for k, v in metrics.items() if math.isfinite(v)}
        if metrics:
            mlflow.log_metrics(metrics)

        if self.log_artifacts:
            with tempfile.TemporaryDirectory() as tmp:
                if summary is not None:
                    path = Path(tmp) / "summary.csv"
                    summary.to_csv(path)
                    mlflow.log_artifact(str(path))
                if idata is not None and hasattr(idata, "to_netcdf"):
                    path = Path(tmp) / artifact_name
                    idata.to_netcdf(path)
                    mlflow.log_artifact(str(path))

    def vi_started(self) -> None:
        """Mark the start of ``vi()`` for the ``vi_seconds`` metric."""
        self._vi_started = time.monotonic()

    def log_vi(self, model: HSSMBase, kwargs: dict) -> None:
        """Log variational settings, the approximation's loss, and the idata."""
        self.log_model(model)
        self._guard("log vi", self._log_vi, model, kwargs)

    def _log_vi(self, model: HSSMBase, kwargs: dict) -> None:
        mlflow = self._mlflow
        params: dict[str, Any] = {}
        for key in ("method", "niter", "draws", "backend"):
            if kwargs.get(key) is not None:
                params[key] = kwargs[key]
        mlflow.log_params(params)

        metrics: dict[str, float] = {}
        if self._vi_started is not None:
            metrics["vi_seconds"] = time.monotonic() - self._vi_started
        # `hist` is the ELBO trace of the fitted approximation: its last entry
        # is where optimisation ended up, its minimum the best it reached.
        hist = getattr(getattr(model, "vi_approx", None), "hist", None)
        if hist is not None and len(hist):
            metrics["elbo_final"] = float(hist[-1])
            metrics["elbo_min"] = float(min(hist))

        idata = getattr(model, "_inference_obj_vi", None)
        self._log_idata(idata, metrics, artifact_name="vi_traces.nc")

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

    def log_param(self, key: str, value: Any) -> None:
        """Attach one of your own parameters to the run.

        Use this for the dimensions *you* are varying — a dataset version, a
        prior set, which subjects were included — so runs can be compared on
        them in the MLflow UI. Keys this module logs itself are refused; see
        :data:`RESERVED_PARAMS`.
        """
        self.log_params({key: value})

    def log_params(self, params: dict[str, Any]) -> None:
        """Attach several of your own parameters at once. See :meth:`log_param`."""
        check_params(params)
        self._guard("log params", self._mlflow.log_params, params)


def active() -> Tracker | None:
    """Return the tracker of the enclosing :func:`track` block, or None.

    Scoped to the execution context: work started in another thread sees None
    unless it opened a block of its own.
    """
    return _ACTIVE.get()


@contextlib.contextmanager
def track(
    experiment: str | None = None,
    run_name: str | None = None,
    *,
    tracking_uri: str | None = None,
    log_artifacts: bool | Literal["all"] = True,
    lineage_id: str | None = None,
    tags: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    dataset_name: str | None = None,
) -> Iterator[Tracker]:
    """Record the enclosed HSSM workflow as one MLflow run (``phase=infer``).

    One block records one fit. Fitting twice inside a single block is **not
    supported**: the second fit's settings collide with the first's, MLflow
    refuses to change a recorded parameter, and the second fit's settings,
    metrics and artifacts are dropped — leaving a run that silently describes
    only the first. Open one block per fit.

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
        ``True`` attaches ``traces.nc``, ``summary.csv``, ``model_spec.json``
        and ``data.parquet`` — the data the model was fit to, so note that it
        is copied into every run, including runs sent to a shared server;
        ``"all"`` additionally attaches ``model.pkl`` when ``save_model`` is
        called inside the block; ``False`` logs params/metrics only.
    lineage_id
        Override the lineage id. By default it is read from the HuggingFace
        ``manifest.json`` entry of the network in use, else minted.
    tags
        Extra tags. ``schema_version``, ``phase`` and ``lineage_id`` are
        reserved and rejected.
    params
        Your own parameters for this run — the dimensions you are varying, such
        as a dataset version or a prior set — so runs can be compared on them in
        the MLflow UI. Keys HSSM records itself are rejected; see
        :data:`RESERVED_PARAMS`. More can be added later with
        :meth:`Tracker.log_param`.
    dataset_name
        Name to register the observed data under, which fills the run's
        "Datasets" panel with the column schema and row count and lets the UI
        filter by dataset. Omitted, no dataset is recorded: a DataFrame carries
        no name of its own, and an invented one would look the same for every
        study. The data is identified either way by the ``data_sha256`` tag.

    Raises
    ------
    ImportError
        If MLflow is not installed (``uv add "hssm[tracking]"``).
    RuntimeError
        If the block is nested inside another.
    ValueError
        If ``tags`` or ``params`` collide with what HSSM records itself.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow is required for hssm.track(). Install it with "
            '`uv add "hssm[tracking]"` (or `pip install "hssm[tracking]"`).'
        ) from exc

    reserved = {"schema_version", "phase", "lineage_id"} & set(tags or {})
    if reserved:
        raise ValueError(f"tags may not override reserved schema tags: {reserved}")
    check_params(params)
    if _ACTIVE.get() is not None:
        raise RuntimeError("hssm.track() blocks cannot be nested")

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    exp = experiment or os.getenv("MLFLOW_EXPERIMENT_NAME")
    if exp:
        mlflow.set_experiment(exp)

    run = mlflow.start_run(run_name=run_name)
    tracker = Tracker(
        mlflow,
        run.info.run_id,
        log_artifacts=log_artifacts,
        lineage_id=lineage_id,
        dataset_name=dataset_name,
    )
    # The common block goes on immediately with a placeholder lineage; it is
    # overwritten with the resolved id once a model is logged.
    initial = common_run_tags(lineage_id or "pending")
    initial["phase"] = "infer"
    initial.update(tags or {})
    tracker._guard("set tags", mlflow.set_tags, initial)
    if params:
        tracker._guard("log params", mlflow.log_params, params)
    _logger.info("MLflow tracking: started run %s", run.info.run_id)

    token = _ACTIVE.set(tracker)
    try:
        yield tracker
    except BaseException:
        _ACTIVE.reset(token)
        with contextlib.suppress(Exception):
            mlflow.end_run(status="FAILED")
        raise
    else:
        _ACTIVE.reset(token)
        with contextlib.suppress(Exception):
            if tracker.lineage_id is None:
                # No model was ever logged; give the run a real lineage id.
                mlflow.set_tag("lineage_id", lineage_id or uuid.uuid4().hex)
            mlflow.end_run()


def load_run(run_id: str, tracking_uri: str | None = None) -> HSSM:
    """Rebuild a fitted model from a run recorded by :func:`track`.

    The model is constructed again from the run's ``model_spec.json`` and
    ``data.parquet``, and its ``traces.nc`` and ``vi_traces.nc`` are attached
    when the run has them.

    Parameters
    ----------
    run_id
        The MLflow run to rebuild.
    tracking_uri
        Tracking server to read from. Defaults to the current MLflow tracking
        URI, which is left unchanged either way.

    Returns
    -------
    HSSM
        The rebuilt model, with its posterior restored.

    Raises
    ------
    ImportError
        If MLflow is not installed (``uv add "hssm[tracking]"``).
    NotImplementedError
        If the run fitted a model class other than ``HSSM``.
    ValueError
        If the run kept no data or spec (it was logged with
        ``log_artifacts=False``), or if its spec holds arguments that were
        recorded as text, such as ``bmb.Prior`` objects or custom functions.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow is required for hssm.load_run(). Install it with "
            '`uv add "hssm[tracking]"` (or `pip install "hssm[tracking]"`).'
        ) from exc
    import arviz as az
    import pandas as pd

    from hssm import HSSM

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    tags = client.get_run(run_id).data.tags
    model_class = tags.get("model_class", "HSSM")
    if model_class != "HSSM":
        raise NotImplementedError(
            f"Run {run_id} fitted a {model_class}; load_run rebuilds HSSM models only."
        )
    not_restorable = ValueError(
        f"Run {run_id} cannot be rebuilt from its spec: some constructor arguments "
        "(e.g. bmb.Prior objects or custom functions) were recorded as text."
    )
    if tags.get("spec_restorable") == "false":
        raise not_restorable

    names = {a.path for a in client.list_artifacts(run_id)}
    missing = sorted(REBUILD_ARTIFACTS - names)
    if missing:
        raise ValueError(
            f"Run {run_id} has no {', '.join(missing)} to rebuild from; it was "
            "probably logged with log_artifacts=False."
        )

    with tempfile.TemporaryDirectory() as tmp:

        def fetch(name: str) -> str:
            return mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=name,
                dst_path=tmp,
                tracking_uri=tracking_uri,
            )

        data = pd.read_parquet(fetch("data.parquet"))
        spec = json.loads(Path(fetch("model_spec.json")).read_text())
        spec.pop("repr", None)
        try:
            model = HSSM(data, **spec)
        except Exception as exc:
            # Runs logged before `spec_restorable` existed carry no verdict;
            # a failed rebuild is the first sign their spec held text.
            if "spec_restorable" in tags:
                raise
            raise not_restorable from exc
        # Loaded into memory: the files go when the temporary directory does.
        if "traces.nc" in names:
            model.restore_traces(az.from_netcdf(fetch("traces.nc")).load())
        if "vi_traces.nc" in names:
            model.restore_vi_traces(az.from_netcdf(fetch("vi_traces.nc")).load())
    return model


@dataclass(frozen=True)
class RunRow:
    """One run as a row of the `list_runs` table; the fields are its columns."""

    run_id: str
    run_name: str | None
    experiment: str | None
    start_time: pd.Timestamp
    status: str
    user: str | None
    model: str | None
    loglik_kind: str | None
    dataset_name: str | None
    n_trials: int | None
    restorable: bool

    @classmethod
    def from_run(cls, run: Run, *, experiment: str | None, restorable: bool) -> RunRow:
        """Read ``run`` into a row.

        ``experiment`` and ``restorable`` are not held by the run itself: the
        first needs the server's experiment names, the second the run's
        artifact listing.
        """
        params = run.data.params
        return cls(
            run_id=run.info.run_id,
            run_name=run.info.run_name,
            experiment=experiment,
            start_time=_start_time(run),
            status=run.info.status,
            user=run.data.tags.get("user"),
            model=params.get("model"),
            loglik_kind=params.get("loglik_kind"),
            dataset_name=_dataset_name(run),
            n_trials=_n_trials(run),
            restorable=restorable,
        )


def list_runs(tracking_uri: str | None = None) -> pd.DataFrame:
    """List the runs recorded by :func:`track`, newest first.

    Runs logged by other tools on the same server are left out.

    Parameters
    ----------
    tracking_uri
        Tracking server to read from. Defaults to the current MLflow tracking
        URI, which is left unchanged either way.

    Returns
    -------
    pd.DataFrame
        One row per run, with columns ``run_id``, ``run_name``, ``experiment``,
        ``start_time``, ``status``, ``user``, ``model``, ``loglik_kind``,
        ``dataset_name``, ``n_trials`` and ``restorable``, which says whether
        :func:`load_run` can rebuild the run.

    Raises
    ------
    ImportError
        If MLflow is not installed (``uv add "hssm[tracking]"``).
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow is required for hssm.list_runs(). Install it with "
            '`uv add "hssm[tracking]"` (or `pip install "hssm[tracking]"`).'
        ) from exc
    import pandas as pd

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiments = {e.experiment_id: e.name for e in client.search_experiments()}
    rows = [
        RunRow.from_run(
            run,
            experiment=experiments.get(run.info.experiment_id),
            restorable=_is_rebuildable(client, run),
        )
        for run in _tracked_runs(client, list(experiments))
    ]
    return pd.DataFrame(
        [asdict(row) for row in rows], columns=[f.name for f in fields(RunRow)]
    )


def _tracked_runs(
    client: MlflowClient, experiment_ids: list[str], page_size: int = RUNS_PAGE_SIZE
) -> Iterator[Run]:
    """Yield every run :func:`track` recorded in ``experiment_ids``, newest first.

    MLflow returns results a page at a time; this follows the page tokens so
    a server with more runs than one page still lists them all.
    """
    if not experiment_ids:  # MLflow would otherwise reject the empty search
        return
    page_token = None
    while True:
        page = client.search_runs(
            experiment_ids,
            filter_string="tags.phase = 'infer'",
            order_by=["attributes.start_time DESC"],
            max_results=page_size,
            page_token=page_token,
        )
        yield from page
        page_token = page.token
        if not page_token:
            return


def _start_time(run: Run) -> pd.Timestamp:
    """Return when the run started, as a UTC timestamp (MLflow stores epoch ms)."""
    import pandas as pd

    return pd.to_datetime(run.info.start_time, unit="ms", utc=True)


def _n_trials(run: Run) -> int | None:
    """Return the number of trials fitted, or None if no model was logged."""
    n_trials = run.data.params.get("n_trials")
    return int(n_trials) if n_trials is not None else None


def _dataset_name(run: Run) -> str | None:
    """Return the name given to ``track(dataset_name=...)``, or None."""
    datasets = run.inputs.dataset_inputs if run.inputs else []
    return datasets[0].dataset.name if datasets else None


def _is_rebuildable(client: MlflowClient, run: Run) -> bool:
    """Whether :func:`load_run` would rebuild ``run``, without rebuilding it."""
    tags = run.data.tags
    if tags.get("model_class") != "HSSM" or tags.get("spec_restorable") != "true":
        return False
    names = {a.path for a in client.list_artifacts(run.info.run_id)}
    return REBUILD_ARTIFACTS <= names
