<img src="docs/images/mainlogo.png" alt="HSSM logo" width="175">

# HSSM — Hierarchical Sequential Sampling Modeling

[![Paper DOI](https://img.shields.io/badge/paper-10.64898%2F2026.06.05.730398-blue)](https://doi.org/10.64898/2026.06.05.730398)
[![PyPI](https://img.shields.io/pypi/v/hssm)](https://pypi.org/project/hssm/)
[![Run tests](https://github.com/lnccbrown/HSSM/actions/workflows/run_tests.yml/badge.svg)](https://github.com/lnccbrown/HSSM/actions/workflows/run_tests.yml)
[![codecov](https://codecov.io/gh/lnccbrown/HSSM/branch/main/graph/badge.svg)](https://codecov.io/gh/lnccbrown/HSSM)

HSSM is a Python toolbox for hierarchical Bayesian modeling of choice and
response-time data with sequential sampling models. It supports trial-wise and
hierarchical regression, reinforcement-learning models, posterior diagnostics,
model comparison, and custom likelihoods through a high-level PyMC and Bambi
interface. HSSM is a
[BRAINSTORM](https://ccbs.carney.brown.edu/brainstorm) project at Brown
University.

## Install

Use Python 3.12, 3.13, or 3.14 in a fresh environment:

```bash
pip install hssm
```

The [installation guide](https://lnccbrown.github.io/HSSM/getting_started/installation/)
covers uv, CUDA extras, Colab, development installs, and troubleshooting.

## Start with the documentation

The [HSSM documentation](https://lnccbrown.github.io/HSSM/) is the canonical
source for durable guidance. Begin with the
[quickstart](https://lnccbrown.github.io/HSSM/getting_started/getting_started/),
then follow the
[main tutorial](https://lnccbrown.github.io/HSSM/tutorials/main_tutorial/).
The [ecosystem map](https://lnccbrown.github.io/HSSM/ecosystem/) explains when
work belongs in HSSM or one of its sibling projects.

## Contributing and support

- Read the [contribution guide](docs/CONTRIBUTING.md) and
  [local development setup](docs/local_development.md).
- Ask modeling questions in
  [GitHub Discussions](https://github.com/lnccbrown/HSSM/discussions).
- Report bugs and request features through
  [GitHub Issues](https://github.com/lnccbrown/HSSM/issues).

## Citation

Please cite Fengler et al., *HSSM: A Widely Applicable Toolbox for Hierarchical
Bayesian Neurocognitive Modeling* ([paper DOI](https://doi.org/10.64898/2026.06.05.730398)).
For version-specific software citation, use the
[Zenodo archive](https://doi.org/10.5281/zenodo.17247695).

## License

HSSM carries the Brown University license in [LICENSE](LICENSE). Copyright
2023 Brown University. All Rights Reserved.
