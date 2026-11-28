"""HSSM must import and run normally when MLflow is not installed.

`tests/test_tracking.py` skips itself without MLflow, so nothing there would
notice a top-level `import mlflow` creeping into `hssm.tracking` — and that
would break every user who has not asked for the `tracking` extra. These tests
run in a subprocess with MLflow made unimportable, so they are meaningful
whether or not the extra is present in the test environment.
"""

import subprocess
import sys
import textwrap

# Refuses any import of mlflow, then exercises the paths a user without the
# extra still goes through.
_BLOCK_MLFLOW = """
import sys


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError(f"No module named {name!r}")
        return None


sys.meta_path.insert(0, _Blocker())
"""


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_MLFLOW + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # the assertions below report the failure with its stderr
    )


def test_hssm_imports_without_mlflow():
    """Importing HSSM, and the tracking module itself, must not need MLflow."""
    result = _run(
        """
        import hssm
        from hssm import tracking

        assert tracking.active() is None
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout


def test_network_provenance_works_without_mlflow():
    """The ONNX loader records provenance on every model, tracked or not.

    `record_network` is called from `distribution_utils.onnx_utils.model`, which
    runs for every user of an approximate-differentiable likelihood.
    """
    result = _run(
        """
        from hssm import tracking

        tracking.record_network("ddm.onnx", "/tmp/snapshots/abc123/ddm.onnx")
        assert tracking.last_network() == {
            "network_file": "ddm.onnx",
            "hf_revision": "abc123",
        }
        tracking.reset_network_record()
        assert tracking.last_network() == {}
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout


def test_track_raises_a_useful_error_without_mlflow():
    """Asking for tracking without the extra must say how to install it."""
    result = _run(
        """
        import hssm

        try:
            with hssm.track(experiment="x"):
                pass
        except ImportError as e:
            assert "hssm[tracking]" in str(e), str(e)
            print("ok")
        else:
            raise AssertionError("track() did not raise without mlflow")
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout
