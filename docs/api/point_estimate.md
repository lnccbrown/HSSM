`hssm.PointEstimate` is what [`HSSM.find_MAP`](hssm.md) and
[`HSSM.find_MLE`](hssm.md) return, and what the `model.map` and `model.mle`
properties hold. It is a `dict` subclass, so it can be passed anywhere a plain
point dictionary was accepted before — notably `model.sample(initvals=...)` —
while also carrying the optimizer metadata and the ArviZ-friendly exporters
documented below. See the
[point estimation tutorial](../tutorials/map_mle.ipynb) for worked examples.

::: hssm.PointEstimate
    options:
      # The project default suppresses attribute docs, but for this class the
      # attributes *are* the API — everything the optimizer reports back lives
      # there rather than on a method.
      show_docstring_attributes: true
