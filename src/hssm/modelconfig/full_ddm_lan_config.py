from .._types import DefaultConfig  # noqa: D100


def get_full_ddm_lan_config() -> DefaultConfig:
    """
    Get the default configuration for the full DDM served by a LAN.

    The seven-parameter DDM with trial-to-trial variability in drift (``sv``),
    starting point (``sz``) and non-decision time (``st``), with a likelihood
    approximation network (``full_ddm_lan.onnx``) in place of ``full_ddm``'s
    blackbox integrals, so the model samples with NUTS. The parameter bounds
    are the network's training box, which is the ``full_ddm_lan`` config in
    ``ssm-simulators``.

    The variability parameters follow the simulator's conventions, which are
    not ``full_ddm``'s (HDDM's):

    - ``st`` is the HALF-width of the uniform non-decision-time kernel
      (``t ± st``), so the support starts at ``t - st``. ``full_ddm`` reads
      ``st`` as the full width (``t ± st/2``).
    - ``sz`` is the HALF-width of the uniform starting-point kernel as a
      fraction of ``a`` (``z ± sz``). ``full_ddm`` and HDDM read ``sz`` as the
      full width (``z ± sz/2``), so ``sz_hddm = 2 * sz``.
    - ``sv`` is the standard deviation of the Normal drift kernel, as in
      ``full_ddm`` and ``ddm_sdv``.

    ``sz`` is not recoverable on its own at any design the network was
    validated on (12 subjects by 150 to 1,800 trials): its posterior is the
    prior. It is free here so that the other six parameters are integrated
    over it rather than fitted at a pinned value; treat it as a nuisance
    parameter and leave its posterior out of inferences. ``sv`` is bounded
    below at 0.3, the floor of the training box: below it ``sv`` is not
    identifiable at ordinary trial counts.

    Returns
    -------
    DefaultConfig
        A dictionary containing the default configuration settings for the model,
        including response variables, model parameters, choices, description,
        and likelihood specifications.
    """
    return {
        "response": ["rt", "response"],
        "list_params": ["v", "a", "z", "t", "sz", "sv", "st"],
        "choices": [-1, 1],
        "description": "The full DDM with a likelihood approximation network",
        "likelihoods": {
            "approx_differentiable": {
                "loglik": "full_ddm_lan.onnx",
                "backend": "jax",
                "default_priors": {},
                # These are the LAN's training box, not modelling choices: the
                # network is only defined on the region it was trained on, and
                # HSSM uses these bounds to keep sampling inside it. ``sv``
                # starts at 0.3 and ``st`` reaches 0.35 for that reason alone;
                # ``full_ddm``'s blackbox likelihood is unbounded there.
                "bounds": {
                    "v": (-3.0, 3.0),
                    "a": (0.3, 2.5),
                    "z": (0.3, 0.7),
                    "t": (0.25, 2.25),
                    "sz": (1e-3, 0.2),
                    "sv": (0.3, 2.5),
                    "st": (1e-3, 0.35),
                },
                "extra_fields": None,
                # ssms draws the non-decision time as t + U(-st, st): st is the
                # half-width, so the support starts at t - st. (full_ddm's
                # blackbox reads st as the full width and declares 0.5.)
                "ndt_edge_shift": {"param": "st", "scale": 1.0},
            },
        },
    }
