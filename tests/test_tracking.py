"""Tests for opt-in MLflow tracking of inference runs (hssm.track)."""

import contextlib
import json
import math
import shutil
from pathlib import Path

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
    tracking._LAST_NETWORK.clear()
    uri = f"sqlite:///{(tmp_path / 'tracking.db').absolute()}"
    mlflow.set_tracking_uri(uri)
    # mlflow remembers the active experiment *id* process-wide; a previous
    # test's id does not exist in this fresh store, so point at Default.
    mlflow.set_experiment("Default")
    yield uri
    if mlflow.active_run() is not None:
        mlflow.end_run()
    tracking._LAST_NETWORK.clear()
    if (Path.cwd() / "mlruns").exists():
        shutil.rmtree(Path.cwd() / "mlruns")
    with contextlib.suppress(Exception):
        mlflow.set_tracking_uri(original_uri)


def _run(run_id):
    return mlflow.tracking.MlflowClient().get_run(run_id)


class TestNetworkProvenance:
    """Tests for network provenance."""

    def test_hf_revision_parsed_from_cache_path(self, tmp_path):
        """Hf revision parsed from cache path."""
        p = tmp_path / "models--franklab--HSSM" / "snapshots" / "abc123" / "ddm.onnx"
        assert tracking.hf_revision_from_path(p) == "abc123"
        assert tracking.hf_revision_from_path(tmp_path / "ddm.onnx") is None

    def test_record_and_last_network(self, tmp_path):
        """Record and last network."""
        p = tmp_path / "snapshots" / "deadbeef" / "angle.onnx"
        tracking.record_network("angle.onnx", p)
        assert tracking.last_network() == {
            "network_file": "angle.onnx",
            "hf_revision": "deadbeef",
        }

    def test_loader_records_hf_downloads(self, tmp_path, monkeypatch):
        """download_hf / load_onnx_model must leave a record for the tracker."""
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

    def test_manifest_lookup_by_root_or_folder_file(self, tmp_path, monkeypatch):
        """Manifest lookup by root or folder file."""
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
            huggingface_hub, "hf_hub_download", lambda repo_id, filename: str(manifest)
        )
        assert tracking._manifest_entry_for("ddm.onnx")["lineage_id"] == "lin-ddm"
        assert tracking._manifest_entry_for("x_lan_ddm__model.onnx")["run_uuid"] == "x"
        assert tracking._manifest_entry_for("angle.onnx") is None


class TestHelpers:
    """Tests for helpers."""

    def test_common_tags(self):
        """Common tags."""
        tags = tracking.common_run_tags("lin")
        assert tags["schema_version"] == "2"
        assert tags["lineage_id"] == "lin"
        assert tags["hostname"] and tags["hssm_version"]

    def test_spec_hash_stable_and_repr_excluded(self):
        """Spec hash stable and repr excluded."""
        a = {"model": "ddm", "include": [{"name": "v"}], "repr": "A"}
        b = {"include": [{"name": "v"}], "model": "ddm", "repr": "B"}
        assert tracking.spec_sha256(a) == tracking.spec_sha256(b)
        assert tracking.spec_sha256({"model": "angle"}) != tracking.spec_sha256(a)

    def test_reserved_tags_rejected(self):
        """Reserved tags rejected."""
        with pytest.raises(ValueError, match="reserved"):
            with hssm.track(tags={"phase": "x"}):
                pass

    def test_nested_track_rejected(self):
        """Nested track rejected."""
        with hssm.track():
            with pytest.raises(RuntimeError, match="nested"):
                with hssm.track():
                    pass

    def test_empty_block_still_gets_lineage_and_finishes(self):
        """Empty block still gets lineage and finishes."""
        with hssm.track(experiment="infer/empty", run_name="noop") as t:
            run_id = t.run_id
        run = _run(run_id)
        assert run.info.status == "FINISHED"
        assert run.data.tags["phase"] == "infer"
        assert len(run.data.tags["lineage_id"]) == 32

    def test_exception_marks_run_failed_and_clears_active(self):
        """Exception marks run failed and clears active."""
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
        """Fitted."""
        with hssm.track(
            experiment="infer/ddm", run_name="t", lineage_id="lin-explicit"
        ) as t:
            model = hssm.HSSM(data_ddm, model="ddm")
            model.sample(draws=10, chains=1, tune=10, progressbar=False)
            run_id = t.run_id
        return _run(run_id), model

    def test_params_tags_metrics(self, fitted):
        """Params tags metrics."""
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

    def test_artifacts(self, fitted):
        """Artifacts."""
        run, _ = fitted
        names = {
            a.path
            for a in mlflow.tracking.MlflowClient().list_artifacts(run.info.run_id)
        }
        assert {"model_spec.json", "summary.csv", "traces.nc"} <= names

    def test_spec_hash_differs_by_model_choices(self, data_ddm):
        """Spec hash differs by model choices."""
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
    """Tests for save model hook."""

    def test_model_pkl_attached_only_with_all(self, data_ddm, tmp_path, monkeypatch):
        """Model pkl attached only with all."""
        monkeypatch.chdir(tmp_path)
        for mode, expect in (("all", True), (True, False)):
            with hssm.track(log_artifacts=mode) as t:
                model = hssm.HSSM(data_ddm, model="ddm")
                model.sample(draws=5, chains=1, tune=5, progressbar=False)
                model.save_model(model_name=f"m-{mode}", save_traces_only=False)
            names = {
                a.path for a in mlflow.tracking.MlflowClient().list_artifacts(t.run_id)
            }
            assert ("model.pkl" in names) is expect, mode

    def test_metrics_only_when_artifacts_off(self, data_ddm):
        """Metrics only when artifacts off."""
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
