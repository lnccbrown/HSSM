"""HSSM must import and run normally when LANfactory is not installed.

LANfactory, which brings torch, is only in the ``inspect`` extra, so nothing in
``hssm`` or ``hssm.plotting`` may import it at module level. These tests run in
a subprocess with LANfactory made unimportable, so they are meaningful whether
or not it is present in the test environment.
"""

import subprocess
import sys
import textwrap

# Refuses any import of lanfactory, then exercises the paths a user without the
# extra still goes through.
_BLOCK_LANFACTORY = """
import sys


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "lanfactory" or name.startswith("lanfactory."):
            raise ImportError(f"No module named {name!r}")
        return None


sys.meta_path.insert(0, _Blocker())
"""


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_LANFACTORY + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # the assertions below report the failure with its stderr
    )


def test_hssm_imports_without_lanfactory():
    """Importing HSSM and its plotting module must not need LANfactory."""
    result = _run(
        """
        import hssm
        import hssm.plotting
        from hssm.plotting import plot_kde_vs_lan, plot_lan_manifold

        assert "lanfactory" not in sys.modules
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout


def test_inspection_plots_raise_a_useful_error_without_lanfactory():
    """Asking for an inspection plot without the extra must say how to install it.

    The import is checked before the model is looked at, so any object will do.
    """
    result = _run(
        """
        from hssm.plotting import plot_kde_vs_lan, plot_lan_manifold

        calls = [
            lambda: plot_kde_vs_lan(object(), {"v": 0.5}),
            lambda: plot_lan_manifold(object()),
        ]
        for call in calls:
            try:
                call()
            except ImportError as e:
                assert "hssm[inspect]" in str(e), str(e)
            else:
                raise AssertionError("the plot did not raise without lanfactory")
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout
