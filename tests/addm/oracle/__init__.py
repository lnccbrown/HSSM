# efpt NumPy reference backend -- vendored TEST ORACLE (not shipped in the package).
# Source: efficient-fpt @ d97a451479141acef845195610f0f9d85824844e
#         MIT License, Copyright (c) 2025 Sicheng Liu.
# Vendored verbatim except import retargeting to flatten the backend into this
# self-contained  package. Kept at efpt's ORIGINAL DEFAULT_TRUNC_NUM=100
# so it is an INDEPENDENT numerical reference for the HSSM-vendored jax kernel
# (do not sync to that kernel's local modifications).

from .single_stage import (
    fptd_basic, q_basic, fptd_single, q_single,
    log_fptd_basic, log_q_basic, log_fptd_single, log_q_single,
)
from .multi_stage import (
    compute_homog_multistage_logfptds_and_lognpd,
    filter_and_group,
)

__all__ = [
    "fptd_basic", "q_basic", "fptd_single", "q_single",
    "log_fptd_basic", "log_q_basic", "log_fptd_single", "log_q_single",
    "compute_homog_multistage_logfptds_and_lognpd", "filter_and_group",
]
