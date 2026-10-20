"""Tests for hssm.rl.utils — validate_balanced_panel."""

import re

import numpy as np
import pandas as pd
import pytest

from hssm.rl.utils import validate_balanced_panel


def _make_panel(
    n_participants: int, n_trials: int, participant_col: str = "participant_id"
) -> pd.DataFrame:
    """Return a perfectly balanced, contiguous panel DataFrame."""
    ids = np.repeat(range(n_participants), n_trials)
    return pd.DataFrame(
        {participant_col: ids, "rt": np.random.rand(n_participants * n_trials)}
    )


class TestValidateBalancedPanelHappyPath:
    def test_returns_correct_shape(self) -> None:
        """Returns (n_participants, n_trials) for a balanced panel."""
        df = _make_panel(5, 20)
        n_p, n_t = validate_balanced_panel(df)
        assert n_p == 5
        assert n_t == 20

    def test_single_participant(self) -> None:
        """Single-participant panel is trivially balanced."""
        df = _make_panel(1, 10)
        n_p, n_t = validate_balanced_panel(df)
        assert n_p == 1
        assert n_t == 10

    def test_custom_participant_col(self) -> None:
        """Works when participant column has a non-default name."""
        df = _make_panel(3, 8, participant_col="subj_id")
        n_p, n_t = validate_balanced_panel(df, participant_col="subj_id")
        assert n_p == 3
        assert n_t == 8


class TestValidateBalancedPanelMissingColumn:
    def test_missing_participant_col_raises(self) -> None:
        """Raises ValueError when participant_col is absent from the DataFrame."""
        df = pd.DataFrame({"rt": [0.5, 0.6]})
        with pytest.raises(ValueError, match="not found in data"):
            validate_balanced_panel(df)

    def test_wrong_participant_col_name_raises(self) -> None:
        """Raises ValueError when a wrong column name is supplied."""
        df = _make_panel(2, 5)
        with pytest.raises(ValueError, match="not found in data"):
            validate_balanced_panel(df, participant_col="subject")


class TestValidateBalancedPanelNaN:
    def test_nan_participant_id_raises(self) -> None:
        """Raises ValueError when participant_col contains NaN."""
        df = _make_panel(3, 4)
        df.loc[df.index[0], "participant_id"] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            validate_balanced_panel(df)


class TestValidateBalancedPanelUnbalanced:
    def test_unbalanced_panel_raises(self) -> None:
        """Raises ValueError when participants have different trial counts."""
        df = _make_panel(3, 5)
        unbalanced = df.iloc[:-1].copy()  # drop one row → participant 2 has 4 trials
        with pytest.raises(ValueError, match="balanced panels"):
            validate_balanced_panel(unbalanced)

    def test_one_participant_fewer_trials_raises(self) -> None:
        """Raises ValueError when one participant has fewer trials than others."""
        df = _make_panel(3, 5)
        # Drop last 2 rows of participant 2 so counts differ.
        mask = ~(
            (df["participant_id"] == 2)
            & (df.index >= df[df["participant_id"] == 2].index[-2])
        )
        unbalanced = df[mask].copy()
        with pytest.raises(ValueError, match="balanced panels"):
            validate_balanced_panel(unbalanced)


class TestValidateBalancedPanelContiguity:
    def test_interleaved_participants_raises(self) -> None:
        """Raises ValueError when participants' rows are interleaved (not contiguous).

        The RL likelihood reshapes data as (n_participants, n_trials, ...) by row
        position, so interleaved rows would silently corrupt trial sequences.
        """
        # Build interleaved data: [0, 1, 2, 0, 1, 2, ...]  (3 participants × 4 trials)
        ids = np.tile([0, 1, 2], 4)
        df = pd.DataFrame({"participant_id": ids, "rt": np.random.rand(12)})
        with pytest.raises(ValueError, match="contiguous"):
            validate_balanced_panel(df)

    def test_sorted_data_passes(self) -> None:
        """Sorting an interleaved panel by participant_id makes it valid."""
        ids = np.tile([0, 1, 2], 4)
        df = pd.DataFrame({"participant_id": ids, "rt": np.random.rand(12)})
        df_sorted = df.sort_values("participant_id").reset_index(drop=True)
        n_p, n_t = validate_balanced_panel(df_sorted)
        assert n_p == 3
        assert n_t == 4


class TestValidateBalancedPanelSuggestedSort:
    """The sort suggested by the contiguity error must keep trial order.

    The RL likelihood takes each participant's trial order from row order, so a
    sort that makes rows contiguous but reorders trials within a participant
    passes validation while silently corrupting the learning sequence. pandas'
    default sort (``kind="quicksort"``) is not stable.
    """

    @staticmethod
    def _interleaved_panel(
        n_participants: int, n_trials: int, participant_col: str
    ) -> pd.DataFrame:
        """Return a panel exported trial by trial: interleaved, in trial order."""
        return pd.DataFrame(
            {
                participant_col: np.tile(np.arange(n_participants), n_trials),
                "trial_id": np.repeat(np.arange(n_trials), n_participants),
                "rt": np.random.default_rng(0).random(n_participants * n_trials),
            }
        )

    @pytest.mark.parametrize("participant_col", ["participant_id", "subj_id"])
    def test_error_message_suggests_stable_sort(self, participant_col: str) -> None:
        """The error message suggests a stable sort on the given column."""
        df = self._interleaved_panel(3, 4, participant_col)
        with pytest.raises(ValueError, match="contiguous") as exc_info:
            validate_balanced_panel(df, participant_col=participant_col)
        message = str(exc_info.value)
        assert f"data.sort_values('{participant_col}', kind='stable')" in message
        assert (
            f"data.sort_values(['{participant_col}', 'trial_id'], kind='stable')"
            in message
        )

    @pytest.mark.parametrize("participant_col", ["participant_id", "subj_id"])
    def test_suggested_sorts_preserve_trial_order(self, participant_col: str) -> None:
        """Every sort quoted in the message keeps each participant's trial order.

        Size of the FULL_RUN setting of docs/tutorials/rlssm_basic.ipynb
        (15 participants x 150 trials).
        """
        n_participants, n_trials = 15, 150
        df = self._interleaved_panel(n_participants, n_trials, participant_col)
        with pytest.raises(ValueError, match="contiguous") as exc_info:
            validate_balanced_panel(df, participant_col=participant_col)
        snippets = re.findall(r"data\.sort_values\([^)]*\)", str(exc_info.value))
        assert snippets
        for snippet in snippets:
            # Run the quoted code as a user would, on the interleaved panel.
            remedied = eval(snippet, {"__builtins__": {}}, {"data": df})
            assert validate_balanced_panel(
                remedied, participant_col=participant_col
            ) == (n_participants, n_trials)
            for _, trials in remedied.groupby(participant_col)["trial_id"]:
                assert trials.is_monotonic_increasing, snippet
