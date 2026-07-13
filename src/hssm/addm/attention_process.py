"""Pluggable attention (drift) processes for the aDDM.

A name->callable registry, the aDDM analog of ``RLSSMConfig.learning_process``.
An attention process maps the per-trial covariates to the padded per-stage drift
array ``mu`` that the likelihood kernel consumes.

The default :func:`standard_alternating` *is* the vendored default drift logic
(it delegates to ``_build_addm_mu_array_data``), so the default attention process
and the default kernel agree by construction. Future variants (bias, drift
offsets, non-alternating gaze) register by name in :data:`ATTENTION_PROCESSES`.
"""

from collections.abc import Callable


def standard_alternating(eta, kappa, r1, r2, flag, d, max_d):
    """Default aDDM drift array.

    Delegates to the vendored builder so the default attention process and the
    default kernel (which builds the same array internally) agree exactly.

    Parameters
    ----------
    eta, kappa
        Attentional discount and drift scaling.
    r1, r2, flag, d
        Per-trial covariates (item ratings, first-item indicator, stage count).
    max_d
        Padded stage dimension (``sacc_array.shape[1]``).

    Returns
    -------
    jax.Array, shape (n_trials, max_d)
        Padded per-trial, per-stage drift array.
    """
    # Imported lazily to avoid an import cycle: importing ``likelihoods.jax``
    # runs ``likelihoods/__init__`` which imports ``builder``, which imports this
    # module. The kernel is only needed when drift is actually computed.
    from .likelihoods.jax import _build_addm_mu_array_data

    return _build_addm_mu_array_data(eta, kappa, r1, r2, flag, d, max_d)


ATTENTION_PROCESSES: dict[str, Callable] = {
    "standard_alternating": standard_alternating,
}


def resolve_attention_process(attention_process: str | Callable) -> Callable:
    """Resolve a registry name or a callable to a concrete attention process.

    Parameters
    ----------
    attention_process
        Either a key of :data:`ATTENTION_PROCESSES` or a callable with the
        :func:`standard_alternating` signature.

    Returns
    -------
    Callable
        The resolved attention-process callable.
    """
    if callable(attention_process):
        return attention_process
    if isinstance(attention_process, str):
        try:
            return ATTENTION_PROCESSES[attention_process]
        except KeyError:
            raise ValueError(
                f"Unknown attention_process {attention_process!r}; "
                f"available: {sorted(ATTENTION_PROCESSES)}."
            ) from None
    raise TypeError(
        "attention_process must be a registry name (str) or a callable, "
        f"got {type(attention_process)!r}."
    )
