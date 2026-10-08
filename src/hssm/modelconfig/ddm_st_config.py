from .._types import DefaultConfig  # noqa: D100


def get_ddm_st_config() -> DefaultConfig:
    """
    Get the default configuration for the DDM with uniform ndt variability.

    Trial non-decision time is ``Uniform(t - st, t + st)``, so ``st`` is the
    kernel half-width and the kernel SD is ``st / sqrt(3)``.

    Here ``st`` is the HALF-width of the uniform non-decision-time kernel
    (``t ± st``), unlike ``full_ddm`` and HDDM, where ``st`` is the full width
    (``t ± st/2``).

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
                "loglik": "ddm_st.onnx",
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
                    "st": (1e-3, 0.35),
                },
                "extra_fields": None,
                # ssms draws the non-decision time as t + U(-st, st): st is the
                # half-width, so the support starts at t - st. (hddm_wfpt's full_ddm
                # reads st as the full width and declares 0.5.)
                "ndt_edge_shift": {"param": "st", "scale": 1.0},
            },
        },
    }
