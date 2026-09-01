"""Operational heartbeat tests for the causal-study cell subprocess."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import scripts.truncated_hierarchy_causal_runner as runner
from scripts.truncated_hierarchy_causal_contract import build_plan, load_manifest


@pytest.fixture(scope="module")
def unit():
    """Return one real planned unit for operational identity fields."""
    return build_plan(load_manifest(), "smoke")[0]


def _observations(stderr: str) -> list[dict[str, object]]:
    return [
        json.loads(line.removeprefix(runner.CELL_OBSERVATION_PREFIX))
        for line in stderr.splitlines()
        if line.startswith(runner.CELL_OBSERVATION_PREFIX)
    ]


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_pid_exit(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.01)
    return not _pid_exists(pid)


def _pid_file_ready(path: Path) -> bool:
    return _read_pid_file(path) is not None


def _read_pid_file(path: Path) -> int | None:
    try:
        pid = int(path.read_text())
    except (OSError, ValueError):
        return None
    return pid if pid > 0 else None


def _cleanup_recorded_processes(
    child_pid_path: Path, grandchild_pid_path: Path
) -> None:
    """Best-effort cleanup even when readiness failed after one PID appeared."""
    child_pid = _read_pid_file(child_pid_path)
    grandchild_pid = _read_pid_file(grandchild_pid_path)
    recorded_pids = [pid for pid in (child_pid, grandchild_pid) if pid is not None]
    if child_pid is not None and any(_pid_exists(pid) for pid in recorded_pids):
        try:
            os.killpg(child_pid, signal.SIGKILL)
        except OSError:
            pass
    for pid in recorded_pids:
        if _pid_exists(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def _pipe_holding_tree_program(child_pid_path: Path, grandchild_pid_path: Path) -> str:
    grandchild_program = (
        "import os,time; "
        "from pathlib import Path; "
        f"Path({str(grandchild_pid_path)!r}).write_text(str(os.getpid())); "
        "print('grandchild-ready', flush=True); "
        "time.sleep(60)"
    )
    return (
        "import os,subprocess,sys,time; "
        "from pathlib import Path; "
        f"Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
        f"subprocess.Popen([sys.executable, '-c', {grandchild_program!r}]); "
        "print('child-ready', flush=True); "
        "time.sleep(60)"
    )


def test_observed_process_emits_live_heartbeat_and_preserves_capture(
    unit, tmp_path: Path, capsys
) -> None:
    """Heartbeat logging never enters either captured child stream."""
    program = (
        "import sys,time; "
        "print('stdout-before', flush=True); "
        "print('stderr-before', file=sys.stderr, flush=True); "
        "time.sleep(0.12); "
        "print('stdout-after'); "
        "print('stderr-after', file=sys.stderr)"
    )

    def prior_sigterm_handler(_signum: int, _frame: object) -> None:
        return

    original_sigterm_handler = signal.signal(signal.SIGTERM, prior_sigterm_handler)
    try:
        completed = runner._run_observed_cell_process(
            unit,
            [sys.executable, "-c", program],
            cwd=tmp_path,
            environment=os.environ.copy(),
            heartbeat_seconds=0.02,
        )
        assert signal.getsignal(signal.SIGTERM) is prior_sigterm_handler
    finally:
        signal.signal(signal.SIGTERM, original_sigterm_handler)

    captured = capsys.readouterr()
    observations = _observations(captured.err)
    assert completed.returncode == 0
    assert completed.stdout == "stdout-before\nstdout-after\n"
    assert completed.stderr == "stderr-before\nstderr-after\n"
    assert captured.out == ""
    assert [record["event"] for record in observations][0] == "cell-start"
    assert [record["event"] for record in observations][-1] == "cell-end"
    heartbeats = [
        record for record in observations if record["event"] == "cell-heartbeat"
    ]
    assert heartbeats
    assert all(record["cell_id"] == unit.cell_id for record in observations)
    assert all(
        record["child_pid"] == observations[0]["child_pid"] for record in observations
    )
    assert all(
        record["resource_status"] in {"available", "partial", "unavailable"}
        for record in heartbeats
    )
    assert observations[-1]["returncode"] == 0
    assert observations[-1]["termination"] == "child-exited"


def test_observed_process_preserves_scientific_failure_returncode_and_streams(
    unit, tmp_path: Path, capsys
) -> None:
    """The runner still receives return code 10 and exact child diagnostics."""
    program = (
        "import sys; "
        "print('failed-out', end=''); "
        "print('failed-err', end='', file=sys.stderr); "
        "raise SystemExit(10)"
    )

    completed = runner._run_observed_cell_process(
        unit,
        [sys.executable, "-c", program],
        cwd=tmp_path,
        environment=os.environ.copy(),
        heartbeat_seconds=1.0,
    )

    observations = _observations(capsys.readouterr().err)
    assert completed.returncode == 10
    assert completed.stdout == "failed-out"
    assert completed.stderr == "failed-err"
    assert [record["event"] for record in observations] == [
        "cell-start",
        "cell-end",
    ]
    assert observations[-1]["returncode"] == 10


def test_resource_telemetry_failure_cannot_change_child_result(
    unit, tmp_path: Path, capsys, monkeypatch
) -> None:
    """A failed optional resource probe degrades to heartbeat-only logging."""

    def fail_resource_probe(_child_pid: int) -> dict[str, object]:
        raise PermissionError("resource access denied")

    monkeypatch.setattr(runner, "_process_resource_snapshot", fail_resource_probe)
    completed = runner._run_observed_cell_process(
        unit,
        [sys.executable, "-c", "import time; print('ok'); time.sleep(0.08)"],
        cwd=tmp_path,
        environment=os.environ.copy(),
        heartbeat_seconds=0.01,
    )

    observations = _observations(capsys.readouterr().err)
    heartbeats = [
        record for record in observations if record["event"] == "cell-heartbeat"
    ]
    assert completed.returncode == 0
    assert completed.stdout == "ok\n"
    assert completed.stderr == ""
    assert heartbeats
    assert all(record["resource_status"] == "unavailable" for record in heartbeats)
    assert all(
        record["resource_error_types"] == ["PermissionError"] for record in heartbeats
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-session regression")
def test_interrupt_kills_child_and_pipe_holding_grandchild_before_propagating(
    unit, tmp_path: Path, capsys, monkeypatch
) -> None:
    """An inherited pipe cannot strand cleanup or leave either process alive."""
    child_pid_path = tmp_path / "child.pid"
    grandchild_pid_path = tmp_path / "grandchild.pid"
    child_program = _pipe_holding_tree_program(child_pid_path, grandchild_pid_path)

    def interrupt_after_tree_starts(_child_pid: int) -> dict[str, object]:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if _pid_file_ready(child_pid_path) and _pid_file_ready(grandchild_pid_path):
                break
            time.sleep(0.01)
        raise KeyboardInterrupt

    monkeypatch.setattr(
        runner, "_process_resource_snapshot", interrupt_after_tree_starts
    )
    started = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            runner._run_observed_cell_process(
                unit,
                [sys.executable, "-c", child_program],
                cwd=tmp_path,
                environment=os.environ.copy(),
                heartbeat_seconds=0.02,
            )
        elapsed = time.monotonic() - started

        child_pid = _read_pid_file(child_pid_path)
        grandchild_pid = _read_pid_file(grandchild_pid_path)
        assert child_pid is not None
        assert grandchild_pid is not None
        assert elapsed < runner.CELL_CLEANUP_SECONDS + 2.0
        assert _wait_for_pid_exit(child_pid)
        assert _wait_for_pid_exit(grandchild_pid)
    finally:
        _cleanup_recorded_processes(child_pid_path, grandchild_pid_path)

    observations = _observations(capsys.readouterr().err)
    assert [record["event"] for record in observations] == [
        "cell-start",
        "cell-end",
    ]
    assert observations[-1]["returncode"] == -signal.SIGKILL
    assert observations[-1]["termination"] == "observer-interrupted"


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal-session regression")
def test_sigterm_observer_exits_143_and_reaps_pipe_holding_process_tree(
    tmp_path: Path,
) -> None:
    """External SIGTERM reaches bounded group cleanup before observer exit."""
    child_pid_path = tmp_path / "child.pid"
    grandchild_pid_path = tmp_path / "grandchild.pid"
    child_program = _pipe_holding_tree_program(child_pid_path, grandchild_pid_path)
    repository = Path(runner.__file__).resolve().parents[1]
    observer_program = (
        "import os,sys; "
        "from pathlib import Path; "
        "from scripts.truncated_hierarchy_causal_contract import "
        "build_plan,load_manifest; "
        "from scripts.truncated_hierarchy_causal_runner import "
        "_run_observed_cell_process; "
        "unit=build_plan(load_manifest(), 'smoke')[0]; "
        f"_run_observed_cell_process(unit, [sys.executable, '-c', {child_program!r}], "
        f"cwd=Path({str(tmp_path)!r}), environment=os.environ.copy(), "
        "heartbeat_seconds=30.0)"
    )
    observer = subprocess.Popen(
        [sys.executable, "-c", observer_program],
        cwd=repository,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if _pid_file_ready(child_pid_path) and _pid_file_ready(grandchild_pid_path):
                break
            if observer.poll() is not None:
                break
            time.sleep(0.01)
        child_pid = _read_pid_file(child_pid_path)
        grandchild_pid = _read_pid_file(grandchild_pid_path)
        assert child_pid is not None
        assert grandchild_pid is not None
        assert observer.poll() is None

        started = time.monotonic()
        os.kill(observer.pid, signal.SIGTERM)
        _stdout, stderr = observer.communicate(
            timeout=runner.CELL_CLEANUP_SECONDS + 5.0
        )
        elapsed = time.monotonic() - started

        assert observer.returncode == 128 + signal.SIGTERM
        assert elapsed < runner.CELL_CLEANUP_SECONDS + 4.0
        assert _wait_for_pid_exit(child_pid)
        assert _wait_for_pid_exit(grandchild_pid)
        observations = _observations(stderr)
        assert [record["event"] for record in observations] == [
            "cell-start",
            "cell-end",
        ]
        assert observations[-1]["returncode"] == -signal.SIGKILL
        assert observations[-1]["termination"] == "observer-interrupted"
    finally:
        if observer.poll() is None:
            observer.terminate()
            try:
                observer.communicate(timeout=runner.CELL_CLEANUP_SECONDS + 1.0)
            except subprocess.TimeoutExpired:
                observer.kill()
                observer.communicate(timeout=runner.CELL_CLEANUP_SECONDS + 1.0)
        _cleanup_recorded_processes(child_pid_path, grandchild_pid_path)
