"""Utility functions for reinforcement learning + SSM models."""

import pandas as pd


def validate_balanced_panel(
    data: pd.DataFrame,
    participant_col: str = "participant_id",
) -> tuple[int, int]:
    """Validate that data forms a balanced panel and return its shape.

    A balanced panel requires every participant to have exactly the same number
    of trials (rows in *data*).

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to validate.
    participant_col : str, optional
        Name of the column that identifies participants.
        Defaults to ``"participant_id"``.

    Returns
    -------
    tuple[int, int]
        ``(n_participants, n_trials)`` where *n_trials* is the number of rows
        per participant.

    Raises
    ------
    ValueError
        If *participant_col* is not present in *data*, or if the panel is
        unbalanced (participants have different trial counts).
    """
    if participant_col not in data.columns:
        raise ValueError(
            f"Column '{participant_col}' not found in data. "
            "Please provide the correct participant column name via "
            "`participant_col`."
        )

    counts = data.groupby(participant_col).size()
    if counts.nunique() != 1:
        raise ValueError(
            "Data must form balanced panels: all participants must have the "
            f"same number of trials. Observed trial counts: {dict(counts)}"
        )

    return int(len(counts)), int(counts.iloc[0])
