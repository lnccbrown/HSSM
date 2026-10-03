from .._types import DefaultConfig  # noqa: D100


def get_ddm_uniform_st_config() -> DefaultConfig:
    """
    Get the default configuration for the DDM with uniform ndt variability.

    Trial non-decision time is ``Uniform(t - st, t + st)``, so ``st`` is the
    kernel half-width and the kernel SD is ``st / sqrt(3)``.

    ``ddm_normal_st`` declares the same ``st`` bounds, but there ``st`` is the
    standard deviation of an unbounded ``Normal(t, st)`` kernel. The numbers
    match; the dispersions do not. ``st = 0.25`` is an SD of ``0.144`` here
    and of ``0.25`` there, and this kernel has a hard support edge at
    ``t - st`` where that one has none. Do not read a shared ``st`` value as
    a shared amount of non-decision-time variability.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the model,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["v", "a", "z", "t", "st"],
        "choices": [-1, 1],
        "description": "The DDM with uniform variability in non-decision time",
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "ddm_uniform_st.onnx",
                "backend": "jax",
                # ssm-simulators ships this simulator as ``ddm_st`` (same
                # half-width ``st``); ``ddm_uniform_st`` becomes an alias for
                # it upstream in a separate ssm-simulators PR. Without this,
                # posterior/prior predictive sampling raises
                # ``Unknown model 'ddm_uniform_st'``.
                "rv": "ddm_st",
                "default_priors": {},
                # These are the LAN's training box, not modelling choices: the
                # network is only defined on the region it was trained on, and
                # HSSM uses these bounds to keep sampling inside it. They are
                # narrower than the family norm for that reason alone - plain
                # ``ddm`` allows ``t`` from 0.0 and ``z`` in (0.1, 0.9), while
                # this network saw ``t >= 0.25`` and ``z`` in (0.3, 0.7).
                "bounds": {
                    "v": (-3.0, 3.0),
                    "a": (0.3, 2.5),
                    "z": (0.3, 0.7),
                    "t": (0.25, 2.25),
                    "st": (1e-3, 0.25),
                },
                "extra_fields": None,
            },
        },
    }
