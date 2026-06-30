# Vendored from efficient-fpt @ d97a451; do not edit in place — re-vendor instead.
"""Package-wide default values for computational parameters."""

DEFAULT_MID_QUAD_ORDER = 20
DEFAULT_LAST_QUAD_ORDER = 30
# Legacy compatibility alias for callers that still pass a single ``order``.
DEFAULT_QUADRATURE_ORDER = DEFAULT_LAST_QUAD_ORDER
DEFAULT_TRUNC_NUM = 6
DEFAULT_THRESHOLD = 1e-20
DEFAULT_CHUNK_SIZE = 200
