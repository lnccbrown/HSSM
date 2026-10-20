# Releases and migration

Use the [changelog](../changelog.md) as the versioned source of truth. A release
story is never inferred from a roadmap item: **planned**, **in shadow**,
**released**, and **deprecated** are different states.

## HSSM 0.5.0

Version 0.5.0 includes model-cartoon uncertainty displays, new gamma-drift model
families, safer prior behavior, link/API consistency work, and compatibility
updates. It also carries breaking changes. Read the exact
[0.5.0 changelog and migration notes](../changelog.md#050) before upgrading.

Useful routes:

- [Plot posterior and predictive results](../tutorials/plotting.ipynb)
- [Model cartoon gallery](../tutorials/cartoon_gallery.ipynb)
- [Link functions and safe priors](../tutorials/link_functions.ipynb)
- [Random-slope prior diagnostics](../tutorials/random_slope_safe_priors.ipynb)
- [Installation and compatible Python versions](../getting_started/installation.md)

Every future release announcement should link a released tag or changelog entry,
name who is affected, put migration before celebration, and retain a correction
route through the issue tracker.
