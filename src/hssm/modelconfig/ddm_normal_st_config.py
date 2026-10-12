from .._types import DefaultConfig  # noqa: D100


def get_ddm_normal_st_config() -> DefaultConfig:
    """
    Get the default configuration for the DDM with Normal ndt variability.

    Trial non-decision time is ``Normal(t, st)``, so ``st`` is the kernel SD
    and the kernel support is unbounded. HSSM floors the likelihood at the
    practical edge ``t - 3 * st`` (``ndt_edge_shift``).

    ``ddm_st`` also has an ``st`` parameter, but there ``st`` is the
    half-width of a ``Uniform(t - st, t + st)`` kernel whose SD is
    ``st / sqrt(3)``. The numbers match; the dispersions do not. ``st = 0.25``
    is an SD of ``0.25`` here and of ``0.144`` there, and that kernel has a
    hard support edge at ``t - st`` where this one has none. Do not read a
    shared ``st`` value as a shared amount of non-decision-time variability.

    The parameter bounds are the network's training box.

    Because the kernel is unbounded, simulated RTs can be non-positive near
    the lower ``t`` bound. The ssm-simulators config measures 8.9% at
    ``t = st = 0.25, a = 0.3`` and 0.013% on the 3-sigma edge
    (``3 * st = t``), so filter non-positive RTs out of posterior-predictive
    samples before comparing them with data.

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
        "description": "The DDM with Normal variability in non-decision time",
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "ddm_normal_st.onnx",
                "backend": "jax",
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
                # The Normal(t, st) kernel has no hard edge, so this is a
                # practical one: the support is taken to start at t - 3 * st,
                # about three kernel SDs below t. (ssms declares no edge for
                # ddm_normal_st.)
                "ndt_edge_shift": {"param": "st", "scale": 3.0},
            },
        },
    }
