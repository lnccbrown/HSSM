from .._types import DefaultConfig  # noqa: D100


def get_ddm_normal_st_config() -> DefaultConfig:
    """
    Get the default configuration for the DDM with Normal ndt variability.

    Trial non-decision time is ``Normal(t, st)``, so ``st`` is the kernel SD
    and the kernel support is unbounded.

    ``ddm_uniform_st`` declares the same ``st`` bounds, but there ``st`` is the
    half-width of a ``Uniform(t - st, t + st)`` kernel whose SD is
    ``st / sqrt(3)``. The numbers match; the dispersions do not. ``st = 0.25``
    is an SD of ``0.25`` here and of ``0.144`` there, and this kernel has no
    support edge at all where that one stops at ``t - st``. Do not read a
    shared ``st`` value as a shared amount of non-decision-time variability.

    The ``st`` lower bound of ``1e-3`` is the network's true training-box
    edge, and it is also a defect corner of this particular network: below
    ``st = 0.01`` the network puts its maximum on the bound where the exact
    likelihood has an interior maximum, and the gradient there points into
    the bound. Chains trap, ``st`` collapses to about ``0.002``, and
    per-subject recovery of ``v`` and ``a`` degrades while rhat and ESS stay
    clean, so nothing warns. Constrain ``st`` to ``(0.01, 0.25)`` for
    inference - e.g. ``include=[{"name": "st", "bounds": (0.01, 0.25)}]`` -
    which restores recovery (measured at ``st = 0.05``: correlation with
    truth for ``v`` +0.24 -> +0.97, coverage 6/12 -> 11/12).

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
                # The Normal ndt kernel is unbounded (real density extends below
                # t - st), so the admissibility floor sits at the kernel's
                # practical 3-sigma edge, t - 3 * st.
                "ndt_edge_width": 3.0,
                "default_priors": {},
                # These are the LAN's training box, not modelling choices: the
                # network is only defined on the region it was trained on, and
                # HSSM uses these bounds to keep sampling inside it. They are
                # narrower than the family norm for that reason alone - plain
                # ``ddm`` allows ``t`` from 0.0 and ``z`` in (0.1, 0.9), while
                # this network saw ``t >= 0.25`` and ``z`` in (0.3, 0.7). The
                # ``st`` floor of 1e-3 is the box edge but not a safe operating
                # point; see this function's docstring.
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
