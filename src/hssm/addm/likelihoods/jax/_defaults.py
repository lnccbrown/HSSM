# Vendored from efficient-fpt @ d97a451; do not edit in place — re-vendor instead.
"""Package-wide default values for computational parameters."""

DEFAULT_MID_QUAD_ORDER = 20
DEFAULT_LAST_QUAD_ORDER = 30
# Legacy compatibility alias for callers that still pass a single ``order``.
DEFAULT_QUADRATURE_ORDER = DEFAULT_LAST_QUAD_ORDER
# HSSM-authored change from efpt upstream (100). The JAX kernel unrolls this FPT
# truncation series into the compiled graph, and its reverse-mode gradient (VJP)
# materializes per-term intermediates — trunc=100 can exhaust RAM at compile/grad
# time. trunc=6 keeps the graph tractable; the cost is ~6e-4 relative error in the
# FPT density vs the efpt NumPy oracle (tests/addm/oracle), acceptable for
# inference. Keep this <= 6 on the JAX path. See likelihoods/jax/NOTICE.md.
DEFAULT_TRUNC_NUM = 6
DEFAULT_THRESHOLD = 1e-20
DEFAULT_CHUNK_SIZE = 200
