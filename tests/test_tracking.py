"""Tests for opt-in MLflow tracking of inference runs (hssm.track)."""

import contextlib
import json
import math
import threading
import time

import pandas as pd
import pytest

import hssm
from hssm import tracking

mlflow = pytest.importorskip("mlflow")


@pytest.fixture(autouse=True)
def _isolated_mlflow(tmp_path, monkeypatch):
    """Sqlite tracking store per test; no env leakage; clean network record."""
    original_uri = mlflow.get_tracking_uri()
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    # Even with a sqlite backend, MLflow puts *artifacts* under a relative
    # ./mlruns. Run from tmp_path so that lands with the rest of the test's
    # scratch instead of in the developer's working directory.
    monkeypatch.chdir(tmp_path)
    tracking.reset_network_record()
    uri = f"sqlite:///{(tmp_path / 'tracking.db').absolute()}"
    mlflow.set_tracking_uri(uri)
    # mlflow remembers the active experiment *id* process-wide; a previous
    # test's id does not exist in this fresh store, so point at Default.
    mlflow.set_experiment("Default")
    yield uri
    if mlflow.active_run() is not None:
        mlflow.end_run()
    tracking.reset_network_record()
    with contextlib.suppress(Exception):
        mlflow.set_tracking_uri(original_uri)


def _run(run_id):
    return mlflow.tracking.MlflowClient().get_run(run_id)


class TestNetworkProvenance:
    """Which ONNX network a fit used, and where it came from."""

    def test_hf_revision_read_from_cache_path_without_a_network_call(self, tmp_path):
        """The commit sha is the directory name under `snapshots/`.

        Reading it off the path keeps provenance free: no request to the Hub.
        A path outside the cache layout has no revision to report.
        """
        p = tmp_path / "models--franklab--HSSM" / "snapshots" / "abc123" / "ddm.onnx"
        assert tracking.hf_revision_from_path(p) == "abc123"
        assert tracking.hf_revision_from_path(tmp_path / "ddm.onnx") is None

    def test_recorded_network_is_readable_with_its_revision(self, tmp_path):
        """What the loader records is what a tracker started later reads back."""
        p = tmp_path / "snapshots" / "deadbeef" / "angle.onnx"
        tracking.record_network("angle.onnx", p)
        assert tracking.last_network() == {
            "network_file": "angle.onnx",
            "hf_revision": "deadbeef",
        }

    def test_downloading_a_network_leaves_a_record_for_the_tracker(
        self, tmp_path, monkeypatch
    ):
        """The loader runs during `HSSM(...)`, before any tracker may exist."""
        from hssm.distribution_utils.onnx_utils import model as onnx_model

        local = tmp_path / "snapshots" / "c0ffee" / "ddm.onnx"
        local.parent.mkdir(parents=True)
        local.write_bytes(b"")
        monkeypatch.setattr(
            onnx_model, "hf_hub_download", lambda repo_id, filename: str(local)
        )
        assert onnx_model.download_hf("ddm.onnx") == str(local)
        assert tracking.last_network() == {
            "network_file": "ddm.onnx",
            "hf_revision": "c0ffee",
        }

    def test_manifest_entry_found_by_either_root_or_member_filename(
        self, tmp_path, monkeypatch
    ):
        """A network is identified by its published root name or any of its files.

        `upload-hf` publishes one canonical root network plus the full artifact
        set, so a fit may name either; an unpublished network matches nothing.
        """
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "networks": [
                        {
                            "model": "ddm",
                            "network_type": "lan",
                            "onnx_root": "ddm.onnx",
                            "files": ["x_lan_ddm__model.onnx"],
                            "lineage_id": "lin-ddm",
                            "mlflow_run_id": "train-run",
                            "run_uuid": "x",
                        }
                    ],
                }
            )
        )
        import huggingface_hub

        monkeypatch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda repo_id, filename, revision=None: str(manifest),
        )
        assert tracking._manifest_entry_for("ddm.onnx")["lineage_id"] == "lin-ddm"
        assert tracking._manifest_entry_for("x_lan_ddm__model.onnx")["run_uuid"] == "x"
        assert tracking._manifest_entry_for("angle.onnx") is None

    def test_manifest_is_read_at_the_network_s_own_revision(
        self, tmp_path, monkeypatch
    ):
        """The manifest must be fetched at the commit the network came from.

        The manifest on the default branch moves on as networks are published,
        so reading it there would describe whatever is current rather than what
        this fit actually ran.
        """
        import huggingface_hub

        def manifest_at(rev, lineage):
            path = tmp_path / f"manifest-{rev}.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "networks": [{"onnx_root": "ddm.onnx", "lineage_id": lineage}],
                    }
                )
            )
            return path

        old, new = manifest_at("old", "lin-old"), manifest_at("new", "lin-new")
        monkeypatch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda repo_id, filename, revision=None: str(
                old if revision == "old" else new
            ),
        )

        at_old = tracking._manifest_entry_for("ddm.onnx", "old")
        assert at_old["lineage_id"] == "lin-old"
        # No revision recorded: fall back to the default branch, as before.
        assert tracking._manifest_entry_for("ddm.onnx")["lineage_id"] == "lin-new"


class TestHelpers:
    """Run identity, the reserved-key guards, and block lifecycle."""

    def test_every_run_carries_schema_version_lineage_and_origin(self):
        """The tags shared with ssm-simulators and LANfactory are always set."""
        tags = tracking.common_run_tags("lin")
        assert tags["schema_version"] == "2"
        assert tags["lineage_id"] == "lin"
        assert tags["hostname"] and tags["hssm_version"]

    def test_spec_hash_ignores_key_order_and_repr_but_not_the_model(self):
        """Two fits share `spec_sha256` exactly when the specification matches.

        Key order and the human-readable `repr` are presentation, not
        specification, so they must not change the digest; the model must.
        """
        a = {"model": "ddm", "include": [{"name": "v"}], "repr": "A"}
        b = {"include": [{"name": "v"}], "model": "ddm", "repr": "B"}
        assert tracking.spec_sha256(a) == tracking.spec_sha256(b)
        assert tracking.spec_sha256({"model": "angle"}) != tracking.spec_sha256(a)

    def test_tags_cannot_overwrite_the_schema_tags(self):
        """Overwriting `phase` would make the run unfindable by its own schema."""
        with pytest.raises(ValueError, match="reserved"):
            with hssm.track(tags={"phase": "x"}):
                pass

    def test_nested_track_blocks_are_refused(self):
        """One active run per process: a nested block would silently split a fit."""
        with hssm.track():
            with pytest.raises(RuntimeError, match="nested"):
                with hssm.track():
                    pass

    def test_block_that_fits_nothing_still_closes_a_complete_run(self):
        """A run with no model must still finish with a real lineage id.

        The id is normally resolved when a model is logged; without one it is
        minted at exit rather than left as the internal placeholder.
        """
        with hssm.track(experiment="infer/empty", run_name="noop") as t:
            run_id = t.run_id
        run = _run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags["phase"] == "infer"
        assert len(run.data.tags["lineage_id"]) == 32

    def test_failure_inside_the_block_is_recorded_and_does_not_leak(self):
        """A raised error marks the run FAILED and leaves no tracker active.

        A leaked tracker would attach the next fit in the process to a dead run.
        """
        with pytest.raises(RuntimeError):
            with hssm.track() as t:
                run_id = t.run_id
                raise RuntimeError("boom")
        assert _run(run_id).info.status == "FAILED"
        assert tracking.active() is None


class TestTrackedFit:
    """A real (tiny) fit inside hssm.track: analytical ddm, 1 chain."""

    @pytest.fixture
    def fitted(self, data_ddm):
        """Run a tiny analytical ddm fit inside a track block."""
        with hssm.track(
            experiment="infer/ddm", run_name="t", lineage_id="lin-explicit"
        ) as t:
            model = hssm.HSSM(data_ddm, model="ddm")
            model.sample(draws=10, chains=1, tune=10, progressbar=False)
            run_id = t.run_id
        return _run(run_id), model

    def test_fit_records_model_sampler_and_convergence(self, fitted):
        """One fit produces the model, sampler and diagnostic record in full."""
        run, _ = fitted
        p, t, m = run.data.params, run.data.tags, run.data.metrics
        assert p["model"] == "ddm"
        assert p["loglik_kind"] == "analytical"
        assert p["sampler"] == "pymc"
        assert p["draws"] == "10" and p["chains"] == "1" and p["tune"] == "10"
        assert p["n_trials"] == "100"
        assert len(p["spec_sha256"]) == 64
        assert "hssm_version" in p and "pymc_version" in p
        assert t["schema_version"] == "2" and t["phase"] == "infer"
        assert t["lineage_id"] == "lin-explicit" and t["lineage_source"] == "user"
        assert len(t["data_sha256"]) == 64
        assert m["sampling_seconds"] > 0
        assert "divergences" in m
        # one chain -> r_hat is NaN and must be dropped rather than logged as NaN
        assert "r_hat_max" not in m
        assert all(math.isfinite(v) for v in m.values())  # no NaNs at all
        assert run.info.status == "FINISHED"

    def test_fit_attaches_spec_summary_and_traces(self, fitted):
        """The three artifacts a fit is expected to leave behind."""
        run, _ = fitted
        names = {
            a.path
            for a in mlflow.tracking.MlflowClient().list_artifacts(run.info.run_id)
        }
        assert {"model_spec.json", "summary.csv", "traces.nc"} <= names

    def test_changing_the_specification_changes_the_spec_hash(self, data_ddm):
        """Adding `p_outlier` is a different model, so a different digest."""
        with hssm.track() as t1:
            hssm.HSSM(data_ddm, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
        with hssm.track() as t2:
            hssm.HSSM(data_ddm, model="ddm", p_outlier=0.1).sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
        assert (
            _run(t1.run_id).data.params["spec_sha256"]
            != _run(t2.run_id).data.params["spec_sha256"]
        )


class TestSaveModelHook:
    """What `log_artifacts` includes, and what it deliberately leaves out."""

    def test_pickled_model_attached_only_when_log_artifacts_is_all(self, data_ddm):
        """`log_artifacts="all"` adds `model.pkl`; plain `True` does not.

        The pickle is the largest artifact and the one most often unwanted, so
        it sits behind the widest setting rather than the default.
        """
        # `save_model` writes relative to the cwd, which `_isolated_mlflow`
        # already points at this test's tmp_path.
        for mode, expect in (("all", True), (True, False)):
            with hssm.track(log_artifacts=mode) as t:
                model = hssm.HSSM(data_ddm, model="ddm")
                model.sample(draws=5, chains=1, tune=5, progressbar=False)
                model.save_model(model_name=f"m-{mode}", save_traces_only=False)
            names = {
                a.path for a in mlflow.tracking.MlflowClient().list_artifacts(t.run_id)
            }
            assert ("model.pkl" in names) is expect, mode

    def test_log_artifacts_false_keeps_metrics_but_writes_no_files(self, data_ddm):
        """Turning artifacts off must not cost the run its metrics too."""
        with hssm.track(log_artifacts=False) as t:
            hssm.HSSM(data_ddm, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
        client = mlflow.tracking.MlflowClient()
        assert client.list_artifacts(t.run_id) == []
        # one chain -> r_hat is NaN and dropped; ESS is finite and must be present
        assert "ess_bulk_min" in client.get_run(t.run_id).data.metrics


def test_untracked_sample_is_unaffected(data_ddm):
    """No track() block: sample() must not touch MLflow at all."""
    assert tracking.active() is None
    hssm.HSSM(data_ddm, model="ddm").sample(
        draws=5, chains=1, tune=5, progressbar=False
    )
    assert mlflow.active_run() is None


def test_untracked_save_model_is_unaffected(data_ddm, tmp_path, monkeypatch):
    """No track() block: save_model() must not touch MLflow either.

    `save_model` carries the third `tracking.active()` hook, alongside the ones
    in `sample()` and `vi()`.
    """
    monkeypatch.chdir(tmp_path)
    assert tracking.active() is None
    model = hssm.HSSM(data_ddm, model="ddm")
    model.sample(draws=5, chains=1, tune=5, progressbar=False)
    model.save_model(model_name="untracked", save_traces_only=False)
    assert mlflow.active_run() is None


class TestUserParams:
    """`params=` and `log_param`: the dimensions a user varies themselves."""

    def test_params_argument_reaches_the_run(self):
        """Params passed to track() land on the run."""
        with hssm.track(experiment="study", params={"prior_v_sd": 2.0}) as t:
            run_id = t.run_id
        assert _run(run_id).data.params["prior_v_sd"] == "2.0"

    def test_params_can_be_added_after_the_block_opens(self):
        """Not everything worth recording is known when the run starts."""
        with hssm.track(experiment="study") as t:
            t.log_param("note", "pilot only")
            t.log_params({"cohort": "A", "excluded": "3"})
            run_id = t.run_id
        p = _run(run_id).data.params
        assert p["note"] == "pilot only"
        assert p["cohort"] == "A" and p["excluded"] == "3"

    def test_reserved_params_rejected_by_track(self):
        """A key HSSM logs itself is refused up front, not silently dropped."""
        with pytest.raises(ValueError, match="model"):
            with hssm.track(params={"model": "mine"}):
                pass

    def test_reserved_params_rejected_by_log_param(self):
        """The guard applies to the handle too, not just to `track()`."""
        with hssm.track() as t:
            with pytest.raises(ValueError, match="draws"):
                t.log_param("draws", 5)

    def test_user_params_survive_a_fit(self, data_ddm):
        """User params coexist with the ones the fit logs."""
        with hssm.track(experiment="study", params={"dataset": "v2"}) as t:
            hssm.HSSM(data_ddm, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
            run_id = t.run_id
        p = _run(run_id).data.params
        assert p["dataset"] == "v2"  # ours
        assert p["model"] == "ddm" and p["sampler"] == "pymc"  # HSSM's


class TestDataHash:
    """`data_sha256` identifies the data, schema included."""

    def test_same_frame_hashes_the_same_every_time(self, data_ddm):
        """The digest groups a study's fits, so it must not drift between calls."""
        assert tracking.data_sha256(data_ddm) == tracking.data_sha256(data_ddm)

    def test_renamed_column_changes_digest(self, data_ddm):
        """Same numbers under a different name are not the same dataset."""
        renamed = data_ddm.rename(columns={data_ddm.columns[0]: "renamed"})
        assert tracking.data_sha256(renamed) != tracking.data_sha256(data_ddm)

    def test_changed_dtype_changes_digest(self, data_ddm):
        """Reading the same values back as another dtype is a different frame."""
        col = data_ddm.columns[0]
        retyped = data_ddm.astype({col: "float32"})
        assert tracking.data_sha256(retyped) != tracking.data_sha256(data_ddm)


class TestNetworkAttribution:
    """Network provenance belongs to the model, not to the process."""

    def test_model_reports_the_network_recorded_while_it_was_built(self, data_ddm):
        """A run logs the model's own snapshot, not the session's latest network.

        The network record is a moving target, so reading it at log time would make a
        model report whichever network was loaded most recently — the wrong one
        in a notebook that builds several models.
        """
        model = hssm.HSSM(data_ddm, model="ddm")
        model._tracking_network = {"network_file": "net_a.onnx", "hf_revision": "aaa"}
        tracking.record_network("net_b.onnx", "/tmp/snapshots/bbb/net_b.onnx")

        # An explicit lineage id keeps the manifest lookup (a network call) out
        # of the test; it is not what is under test here.
        with hssm.track(experiment="study", lineage_id="lin") as t:
            t.log_model(model)
            run_id = t.run_id
        p = _run(run_id).data.params
        assert p["network_file"] == "net_a.onnx"
        assert p["hf_revision"] == "aaa"

    def test_model_that_loads_no_network_inherits_none(self, data_ddm):
        """An analytical model must not pick up an earlier model's network.

        Construction clears the record before building the likelihood, so a
        model that downloads nothing ends up with nothing rather than with
        whatever the previous model in the session loaded.
        """
        tracking.record_network("net_a.onnx", "/tmp/snapshots/aaa/net_a.onnx")
        model = hssm.HSSM(data_ddm, model="ddm", loglik_kind="analytical")
        assert model._tracking_network == {}

        with hssm.track(experiment="study", lineage_id="lin") as t:
            t.log_model(model)
            run_id = t.run_id
        assert "net_a.onnx" not in _run(run_id).data.params.get("network_file", "")


class TestTrackedVI:
    """`vi()` records the variational settings, its loss, and the idata."""

    @pytest.fixture
    def fitted_vi(self, data_ddm):
        """Run a tiny ADVI fit inside a track block."""
        with hssm.track(experiment="study", run_name="advi", lineage_id="lin") as t:
            model = hssm.HSSM(data_ddm, model="ddm")
            model.vi(method="advi", niter=100, draws=10, progressbar=False)
            run_id = t.run_id
        return _run(run_id), model

    def test_vi_records_its_settings_and_where_the_elbo_landed(self, fitted_vi):
        """Variational settings, duration and ELBO reach the run.

        r_hat is excluded deliberately: VI produces a single draw set, so the
        statistic is NaN and would be noise in the UI rather than information.
        """
        run, _ = fitted_vi
        p, m = run.data.params, run.data.metrics
        assert p["method"] == "advi"
        assert p["niter"] == "100" and p["draws"] == "10"
        assert p["model"] == "ddm"  # the model block still runs
        assert m["vi_seconds"] > 0
        assert "elbo_final" in m and "elbo_min" in m
        # VI yields a single draw set: r_hat is NaN and must not be logged.
        assert "r_hat_max" not in m
        assert all(math.isfinite(v) for v in m.values())
        assert run.info.status == "FINISHED"

    def test_artifacts_include_the_vi_idata(self, fitted_vi):
        """The approximate posterior is attached under its own name."""
        run, _ = fitted_vi
        names = {
            a.path
            for a in mlflow.tracking.MlflowClient().list_artifacts(run.info.run_id)
        }
        assert {"model_spec.json", "summary.csv", "vi_traces.nc"} <= names

    def test_untracked_vi_is_unaffected(self, data_ddm):
        """No track() block: vi() must not touch MLflow."""
        assert tracking.active() is None
        hssm.HSSM(data_ddm, model="ddm").vi(
            method="advi", niter=50, draws=10, progressbar=False
        )
        assert mlflow.active_run() is None


class TestDatasetRegistration:
    """`dataset_name` fills MLflow's own Datasets panel."""

    def test_named_dataset_is_registered_with_schema_and_row_count(self, data_ddm):
        """A name turns the observed data into an MLflow dataset input.

        MLflow's Datasets panel is fed by `log_input`, not by params or tags,
        so without this the panel reads "None" however much else is recorded.
        """
        with hssm.track(experiment="study", lineage_id="lin", dataset_name="ddm") as t:
            hssm.HSSM(data_ddm, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
            run_id = t.run_id
        inputs = mlflow.tracking.MlflowClient().get_run(run_id).inputs.dataset_inputs
        assert len(inputs) == 1
        dataset = inputs[0].dataset
        assert dataset.name == "ddm"
        assert '"num_rows": 100' in dataset.profile
        assert "rt" in dataset.schema

    def test_no_dataset_recorded_without_a_name(self, data_ddm):
        """Unnamed data stays out of the panel rather than inventing a label.

        Every study would otherwise show the same generic dataset name, which
        is worse than an empty panel. `data_sha256` still identifies the data.
        """
        with hssm.track(experiment="study", lineage_id="lin") as t:
            hssm.HSSM(data_ddm, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
            run_id = t.run_id
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        assert run.inputs.dataset_inputs == []
        assert len(run.data.tags["data_sha256"]) == 64


class TestContextIsolation:
    """The active tracker is scoped to its execution context, not the process."""

    def test_other_thread_sees_no_tracker(self):
        """A thread that did not open a block must not find one."""
        seen = {}

        def worker():
            seen["active"] = tracking.active()

        with hssm.track(experiment="study", lineage_id="lin"):
            assert tracking.active() is not None
            t = threading.Thread(target=worker)
            t.start()
            t.join()
        assert seen["active"] is None

    def test_untracked_fit_in_another_thread_does_not_spoil_the_tracked_run(
        self, data_ddm
    ):
        """A concurrent untracked fit must not consume the tracked run's state.

        With a process-wide tracker the other thread's `sample()` finds it, logs
        *its* model and marks the tracker as done. The tracked fit that follows
        is then skipped as "already logged", so the run describes the wrong fit.
        The two fits here use different numbers of trials, which is what makes
        the mix-up visible.
        """
        other_data = data_ddm.head(40)

        def worker():
            hssm.HSSM(other_data, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )

        with hssm.track(experiment="study", lineage_id="lin") as t:
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            # The tracked fit happens *after* the other thread has finished.
            hssm.HSSM(data_ddm, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
            run_id = t.run_id

        run = _run(run_id)
        # 100 trials is this block's own fit; 40 would be the other thread's.
        assert run.data.params["n_trials"] == "100"
        assert run.data.metrics["sampling_seconds"] > 0

    def test_network_record_is_per_context(self):
        """A download in another thread must not become this one's network.

        Construction clears the record and reads it back a moment later; if that
        record were shared across threads, a model built concurrently could slot
        its own download into the gap and be claimed by the wrong model.
        """
        seen = {}

        def worker():
            tracking.record_network("other.onnx", "/tmp/snapshots/bbb/other.onnx")
            seen["in_thread"] = tracking.last_network().get("network_file")

        tracking.record_network("mine.onnx", "/tmp/snapshots/aaa/mine.onnx")
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert seen["in_thread"] == "other.onnx"  # the thread saw its own
        assert tracking.last_network()["network_file"] == "mine.onnx"  # ours intact


def _artifacts(run_id):
    return {a.path for a in mlflow.tracking.MlflowClient().list_artifacts(run_id)}


def _download_data(run_id):
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="data.parquet"
    )
    return pd.read_parquet(path)


class TestDataArtifact:
    """The data a model was fit to is kept with its run, as `data.parquet`."""

    def test_fitted_data_round_trips(self, data_ddm):
        """The stored copy is the frame the posterior was fit to, exactly."""
        with hssm.track(experiment="study", lineage_id="lin") as t:
            model = hssm.HSSM(data_ddm, model="ddm")
            model.sample(draws=5, chains=1, tune=5, progressbar=False)
            run_id = t.run_id
        pd.testing.assert_frame_equal(_download_data(run_id), model.data)

    def test_covariates_survive(self, data_ddm_reg):
        """What the traces lose — a regression's covariates — the artifact keeps.

        `traces.nc` holds only the observed response, so without this a run
        could not say which predictors a regression used.
        """
        with hssm.track(experiment="study", lineage_id="lin") as t:
            hssm.HSSM(
                data_ddm_reg,
                model="ddm",
                include=[{"name": "v", "formula": "v ~ x + y"}],
            ).sample(draws=5, chains=1, tune=5, progressbar=False)
            run_id = t.run_id
        assert {"x", "y"} <= set(_download_data(run_id).columns)

    def test_not_written_when_artifacts_are_off(self, data_ddm):
        """`log_artifacts=False` keeps the data out of the run."""
        with hssm.track(log_artifacts=False, lineage_id="lin") as t:
            hssm.HSSM(data_ddm, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
            run_id = t.run_id
        assert "data.parquet" not in _artifacts(run_id)

    def test_named_dataset_points_at_the_stored_copy(self, data_ddm):
        """The Datasets panel entry links to the artifact, not just a schema."""
        with hssm.track(experiment="study", lineage_id="lin", dataset_name="ddm") as t:
            hssm.HSSM(data_ddm, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
            run_id = t.run_id
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        source = json.loads(run.inputs.dataset_inputs[0].dataset.source)
        assert source["uri"].endswith("data.parquet")

    def test_a_failed_write_costs_only_the_data(self, data_ddm, monkeypatch):
        """If the parquet cannot be written, the fit and the rest of the run stand.

        `to_parquet` can fail on unusual frames, or when pyarrow is missing
        because MLflow was installed without the `tracking` extra.
        """

        def refuse(*args, **kwargs):
            raise ImportError("pyarrow is not installed")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", refuse)
        with hssm.track(experiment="study", lineage_id="lin", dataset_name="ddm") as t:
            hssm.HSSM(data_ddm, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
            run_id = t.run_id
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        names = _artifacts(run_id)
        assert run.info.status == "FINISHED"
        assert "data.parquet" not in names
        assert {"model_spec.json", "summary.csv", "traces.nc"} <= names
        # The dataset is still registered, just without a stored copy to link.
        assert run.inputs.dataset_inputs[0].dataset.name == "ddm"

    def test_dataset_survives_a_proxied_artifact_store(self, data_ddm, monkeypatch):
        """A server run with --serve-artifacts must still get its Datasets entry.

        Such a server hands out `mlflow-artifacts:/...` URIs, which MLflow has no
        dataset-source resolver for. Linking the entry to the stored copy then
        fails, and the entry must be registered without the link rather than
        dropped. Faking the URI reproduces it: resolution is by scheme alone.
        """
        monkeypatch.setattr(
            mlflow,
            "get_artifact_uri",
            lambda path=None: f"mlflow-artifacts:/1/abc/artifacts/{path}",
        )
        with hssm.track(experiment="study", lineage_id="lin", dataset_name="ddm") as t:
            hssm.HSSM(data_ddm, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
            run_id = t.run_id
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        assert [d.dataset.name for d in run.inputs.dataset_inputs] == ["ddm"]
        assert "data.parquet" in _artifacts(run_id)


def _posterior(idata):
    return idata["posterior"].to_dataset()


class TestLoadRun:
    """`hssm.load_run` rebuilds a fitted model from what its run recorded."""

    @pytest.fixture
    def fitted_reg(self, data_ddm_reg):
        """A regression with a dict prior: everything the spec can store as is."""
        with hssm.track(experiment="study", lineage_id="lin") as t:
            model = hssm.HSSM(
                data_ddm_reg,
                model="ddm",
                include=[
                    {
                        "name": "v",
                        "formula": "v ~ 1 + x",
                        "prior": {"x": {"name": "Normal", "mu": 0.0, "sigma": 0.5}},
                    }
                ],
            )
            model.sample(draws=5, chains=1, tune=5, progressbar=False)
        return t.run_id, model

    def test_rebuilt_model_has_the_same_specification(self, fitted_reg):
        """Same modelling choices, same data: the spec hashes must agree."""
        run_id, model = fitted_reg
        loaded = hssm.load_run(run_id)
        assert isinstance(loaded, hssm.HSSM)
        assert tracking.spec_sha256(tracking.model_spec(loaded)) == (
            tracking.spec_sha256(tracking.model_spec(model))
        )
        pd.testing.assert_frame_equal(loaded.data, model.data)

    def test_rebuilt_model_carries_the_fitted_posterior(self, fitted_reg):
        """The draws come back attached, not just the model that made them."""
        run_id, model = fitted_reg
        loaded = hssm.load_run(run_id)
        assert _posterior(loaded.traces).equals(_posterior(model.traces))

    def test_vi_run_restores_the_vi_posterior(self, data_ddm):
        """A VI fit stores `vi_traces.nc`, which must land back on the model."""
        with hssm.track(lineage_id="lin") as t:
            model = hssm.HSSM(data_ddm, model="ddm")
            model.vi(method="advi", niter=100, draws=10, progressbar=False)
        loaded = hssm.load_run(t.run_id)
        assert _posterior(loaded._inference_obj_vi).equals(
            _posterior(model._inference_obj_vi)
        )

    def test_run_records_class_and_that_its_spec_is_restorable(self, fitted_reg):
        """The tags `load_run` checks before trying to rebuild anything."""
        run_id, _ = fitted_reg
        tags = _run(run_id).data.tags
        assert tags["model_class"] == "HSSM"
        assert tags["spec_restorable"] == "true"

    def test_object_prior_marks_the_spec_not_restorable(self, data_ddm):
        """A `bmb.Prior` is stored as text, which no constructor accepts.

        The JSON alone looks valid, so the run must say so at logging time.
        """
        import bambi as bmb

        with hssm.track(lineage_id="lin") as t:
            hssm.HSSM(
                data_ddm,
                model="ddm",
                include=[{"name": "v", "prior": bmb.Prior("Normal", mu=0, sigma=1)}],
            ).sample(draws=5, chains=1, tune=5, progressbar=False)
        assert _run(t.run_id).data.tags["spec_restorable"] == "false"
        with pytest.raises(ValueError, match="cannot be rebuilt from its spec"):
            hssm.load_run(t.run_id)

    def test_run_without_artifacts_cannot_be_rebuilt(self, data_ddm):
        """`log_artifacts=False` keeps no data or spec, so nothing to rebuild from."""
        with hssm.track(log_artifacts=False, lineage_id="lin") as t:
            hssm.HSSM(data_ddm, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
        with pytest.raises(ValueError, match="log_artifacts"):
            hssm.load_run(t.run_id)

    def test_other_model_classes_are_refused(self, fitted_reg):
        """Subclasses record different constructor arguments than `HSSM`."""
        run_id, _ = fitted_reg
        mlflow.tracking.MlflowClient().set_tag(run_id, "model_class", "RLSSM")
        with pytest.raises(NotImplementedError, match="RLSSM"):
            hssm.load_run(run_id)

    def test_explicit_tracking_uri_leaves_the_global_one_alone(
        self, fitted_reg, _isolated_mlflow
    ):
        """Loading from a named server must not redirect later tracked runs."""
        run_id, _ = fitted_reg
        mlflow.set_tracking_uri("sqlite:///elsewhere.db")
        loaded = hssm.load_run(run_id, tracking_uri=_isolated_mlflow)
        assert isinstance(loaded, hssm.HSSM)
        assert mlflow.get_tracking_uri() == "sqlite:///elsewhere.db"


LIST_RUNS_COLUMNS = [
    "run_id",
    "run_name",
    "experiment",
    "start_time",
    "status",
    "user",
    "model",
    "loglik_kind",
    "dataset_name",
    "n_trials",
    "restorable",
]


def _empty_hssm_run(**kwargs):
    """A run recorded by hssm.track with nothing fitted inside it."""
    with hssm.track(**kwargs) as t:
        pass
    return t.run_id


def _other_tools_run():
    """A plain MLflow run, as ssm-simulators or LANfactory would log it."""
    with mlflow.start_run() as run:
        pass
    return run.info.run_id


def _only_row(runs, run_id):
    rows = runs[runs["run_id"] == run_id]
    assert len(rows) == 1
    return rows.iloc[0]


class TestListRuns:
    """`hssm.list_runs` shows the runs `hssm.track` recorded, one row each."""

    def test_runs_listed_newest_first_with_their_experiment(self):
        """The table a user reads a `run_id` off: curated columns, newest on top."""
        _empty_hssm_run(experiment="study-a", run_name="first")
        time.sleep(0.01)  # distinct start times, so the order is defined
        _empty_hssm_run(experiment="study-b", run_name="second")
        runs = hssm.list_runs()
        assert list(runs.columns) == LIST_RUNS_COLUMNS
        assert list(runs["run_name"]) == ["second", "first"]
        assert list(runs["experiment"]) == ["study-b", "study-a"]
        assert (runs["status"] == "FINISHED").all()

    def test_fitted_run_reports_model_data_and_that_it_can_be_rebuilt(self, data_ddm):
        """A fit fills in the model columns, and `load_run` can rebuild it."""
        with hssm.track(lineage_id="lin", dataset_name="ddm-sim") as t:
            hssm.HSSM(data_ddm, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
        row = _only_row(hssm.list_runs(), t.run_id)
        assert row["model"] == "ddm"
        assert row["loglik_kind"] == "analytical"
        assert row["dataset_name"] == "ddm-sim"
        assert row["n_trials"] == 100
        assert row["restorable"]

    def test_runs_not_recorded_by_hssm_are_left_out(self):
        """Other tools on a shared server log runs too; they are not HSSM fits."""
        other = _other_tools_run()
        hssm_run = _empty_hssm_run()
        run_ids = set(hssm.list_runs()["run_id"])
        assert hssm_run in run_ids
        assert other not in run_ids

    def test_object_prior_run_is_not_restorable(self, data_ddm):
        """The column agrees with what `load_run` would refuse."""
        import bambi as bmb

        with hssm.track(lineage_id="lin") as t:
            hssm.HSSM(
                data_ddm,
                model="ddm",
                include=[{"name": "v", "prior": bmb.Prior("Normal", mu=0, sigma=1)}],
            ).sample(draws=5, chains=1, tune=5, progressbar=False)
        assert not _only_row(hssm.list_runs(), t.run_id)["restorable"]

    def test_run_without_artifacts_is_not_restorable(self, data_ddm):
        """No stored data or spec means nothing for `load_run` to rebuild from."""
        with hssm.track(log_artifacts=False, lineage_id="lin") as t:
            hssm.HSSM(data_ddm, model="ddm").sample(
                draws=5, chains=1, tune=5, progressbar=False
            )
        assert not _only_row(hssm.list_runs(), t.run_id)["restorable"]

    def test_empty_store_gives_an_empty_table_with_the_columns(self):
        """Code that reads the columns must not break before the first run."""
        runs = hssm.list_runs()
        assert runs.empty
        assert list(runs.columns) == LIST_RUNS_COLUMNS

    def test_explicit_tracking_uri_leaves_the_global_one_alone(self, _isolated_mlflow):
        """Listing a named server must not redirect later tracked runs."""
        hssm_run = _empty_hssm_run()
        mlflow.set_tracking_uri("sqlite:///elsewhere.db")
        runs = hssm.list_runs(tracking_uri=_isolated_mlflow)
        assert list(runs["run_id"]) == [hssm_run]
        assert mlflow.get_tracking_uri() == "sqlite:///elsewhere.db"


class TestRestorableValues:
    """`_is_restorable`: which constructor arguments survive the JSON spec."""

    @pytest.mark.parametrize(
        "value",
        [None, True, 1, 1.5, "ddm", [1, "a"], (0.0, 1.0), {"v": {"prior": [1]}}],
    )
    def test_plain_values_are_kept_as_data(self, value):
        """What `_jsonable` writes unchanged can be read back unchanged."""
        assert tracking._is_restorable(value)

    @pytest.mark.parametrize(
        "value",
        [object(), {1: "a"}, {"v": object()}, [object()], {1, 2}, print],
    )
    def test_anything_else_is_written_as_text(self, value):
        """Objects, non-string keys, sets and functions become `repr` text."""
        assert not tracking._is_restorable(value)

    def test_model_with_dict_priors_is_restorable(self, data_ddm):
        """The spec of a model built from plain values needs nothing else."""
        model = hssm.HSSM(
            data_ddm,
            model="ddm",
            include=[{"name": "v", "prior": {"name": "Normal", "mu": 0, "sigma": 1}}],
        )
        assert tracking.spec_is_restorable(model)

    def test_model_with_an_object_prior_is_not_restorable(self, data_ddm):
        """A `bmb.Prior` in the arguments is enough to lose the spec."""
        import bambi as bmb

        model = hssm.HSSM(
            data_ddm,
            model="ddm",
            include=[{"name": "v", "prior": bmb.Prior("Normal", mu=0, sigma=1)}],
        )
        assert not tracking.spec_is_restorable(model)


class TestTrackedRunsSearch:
    """`_tracked_runs`: every HSSM run on the server, across result pages."""

    def test_follows_every_page_newest_first(self):
        """With one run per page, all runs still come back, in order."""
        names = ["first", "second", "third"]
        for name in names:
            _empty_hssm_run(run_name=name)
            time.sleep(0.01)  # distinct start times, so the order is defined
        _other_tools_run()
        client = mlflow.tracking.MlflowClient()
        experiment_ids = [e.experiment_id for e in client.search_experiments()]
        runs = tracking._tracked_runs(client, experiment_ids, page_size=1)
        assert [r.info.run_name for r in runs] == names[::-1]

    def test_no_experiments_means_no_runs(self):
        """An empty list of experiments must not search everything instead."""
        client = mlflow.tracking.MlflowClient()
        assert list(tracking._tracked_runs(client, [])) == []


class TestRunRow:
    """`_run_row`: one run as a row of the `list_runs` table."""

    def test_row_has_every_column_and_reads_the_run(self):
        """Names, times and recorded params end up in their columns."""
        run_id = _empty_hssm_run(run_name="r")
        client = mlflow.tracking.MlflowClient()
        client.log_param(run_id, "model", "ddm")
        client.log_param(run_id, "n_trials", "42")
        row = tracking._run_row(client.get_run(run_id), "study", restorable=True)
        assert list(row) == LIST_RUNS_COLUMNS
        assert row["run_id"] == run_id and row["run_name"] == "r"
        assert row["experiment"] == "study"
        assert row["status"] == "FINISHED"
        assert row["model"] == "ddm"
        assert row["n_trials"] == 42  # a number, not the param's string
        assert row["start_time"].tzinfo is not None
        assert row["restorable"] is True

    def test_unfitted_run_leaves_model_columns_empty(self):
        """A block that fitted nothing has no model, data or trial count."""
        run_id = _empty_hssm_run()
        run = mlflow.tracking.MlflowClient().get_run(run_id)
        row = tracking._run_row(run, "Default", restorable=False)
        assert row["model"] is None and row["loglik_kind"] is None
        assert row["dataset_name"] is None and row["n_trials"] is None


class TestIsRebuildable:
    """`_is_rebuildable`: what `load_run` needs, checked without rebuilding."""

    def _run(self, *, model_class="HSSM", spec_restorable="true", artifacts=True):
        client = mlflow.tracking.MlflowClient()
        run_id = _empty_hssm_run()
        client.set_tag(run_id, "model_class", model_class)
        client.set_tag(run_id, "spec_restorable", spec_restorable)
        if artifacts:
            for name in ("data.parquet", "model_spec.json"):
                client.log_text(run_id, "", name)
        return client, client.get_run(run_id)

    def test_hssm_run_with_restorable_spec_and_artifacts(self):
        """All three conditions met."""
        assert tracking._is_rebuildable(*self._run())

    def test_other_model_class(self):
        """`load_run` rebuilds `HSSM` only."""
        assert not tracking._is_rebuildable(*self._run(model_class="RLSSM"))

    def test_spec_recorded_as_text(self):
        """A spec holding `repr` text cannot be passed to a constructor."""
        assert not tracking._is_rebuildable(*self._run(spec_restorable="false"))

    def test_missing_artifacts(self):
        """No stored data or spec, nothing to rebuild from."""
        assert not tracking._is_rebuildable(*self._run(artifacts=False))

    def test_run_from_before_the_tags(self):
        """Without the tags there is no verdict, so it is not offered."""
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(_empty_hssm_run())
        assert not tracking._is_rebuildable(client, run)
