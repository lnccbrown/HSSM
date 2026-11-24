# efpt NumPy reference backend -- vendored TEST ORACLE (not shipped in the package).
# Source: efficient-fpt @ d97a451479141acef845195610f0f9d85824844e
#         MIT License, Copyright (c) 2025 Sicheng Liu.
# Vendored verbatim except import retargeting to flatten the backend into this
# self-contained  package. Kept at efpt's ORIGINAL DEFAULT_TRUNC_NUM=100
# so it is an INDEPENDENT numerical reference for the HSSM-vendored jax kernel
# (do not sync to that kernel's local modifications).

"""Package-wide default values for computational parameters."""

DEFAULT_MID_QUAD_ORDER = 20
DEFAULT_LAST_QUAD_ORDER = 30
# Legacy compatibility alias for callers that still pass a single ``order``.
DEFAULT_QUADRATURE_ORDER = DEFAULT_LAST_QUAD_ORDER
DEFAULT_TRUNC_NUM = 100
DEFAULT_THRESHOLD = 1e-20
DEFAULT_CHUNK_SIZE = 200
