# HSSM open issues — inventory

Snapshot of every **open** issue in [`lnccbrown/hssm`](https://github.com/lnccbrown/hssm/issues) as of **2026-08-16**. Pull requests are excluded.

**Total open issues: 108**

Machine-readable companions: [`open_issues.csv`](open_issues.csv), [`open_issues.json`](open_issues.json). Regenerate all three with `uv run python issue_inventory/generate_inventory.py`.

## By category

| Category | Count | Share |
| --- | ---: | ---: |
| Tests / Test suite | 14 | 13% |
| CI / Release | 8 | 7% |
| RLSSM | 14 | 13% |
| Architecture / API | 19 | 18% |
| Models / Likelihoods | 10 | 9% |
| Sampling / Inference | 12 | 11% |
| aDDM | 4 | 4% |
| Documentation | 6 | 6% |
| p_outlier | 5 | 5% |
| Priors | 5 | 5% |
| Plotting | 3 | 3% |
| BayesFlow / SBI | 3 | 3% |
| Other / Bugs | 5 | 5% |

## By year opened

| Year | Count |
| --- | ---: |
| 2023 | 2 |
| 2024 | 14 |
| 2025 | 38 |
| 2026 | 54 |

## Most frequent reporters

| Author | Count |
| --- | ---: |
| AlexanderFengler | 44 |
| cpaniaguam | 20 |
| digicosmos86 | 13 |
| krishnbera | 10 |
| frankmj | 3 |
| AndrewZhang599 | 2 |
| morgbead | 2 |
| YuanboBQ | 2 |
| igrahek | 2 |
| jainraj | 2 |

## Labels in use

| Label | Count |
| --- | ---: |
| `enhancement` | 14 |
| `bug` | 8 |
| `chore` | 7 |
| `github_actions` | 5 |
| `good first issue` | 5 |
| `refactor` | 3 |
| `documentation` | 2 |
| `release` | 2 |
| `upstream` | 2 |
| `dependencies` | 1 |
| `drift` | 1 |
| `likeliihood` | 1 |
| `model` | 1 |
| `nice to have` | 1 |
| `pipeline` | 1 |

71 of 108 open issues (66%) carry no label.

## Oldest still-open issues

| # | Title | Opened | Age (days) |
| --- | --- | --- | ---: |
| [#285](https://github.com/lnccbrown/hssm/issues/285) | Add note on `a` parameter meaning to docs | 2023-09-22 | 1059 |
| [#312](https://github.com/lnccbrown/hssm/issues/312) | Add some example datasets to package for testing robustness of priors | 2023-11-01 | 1019 |
| [#347](https://github.com/lnccbrown/hssm/issues/347) | Add DIC function as util method | 2024-01-29 | 930 |
| [#353](https://github.com/lnccbrown/hssm/issues/353) | LANs for full DDM | 2024-02-15 | 913 |
| [#387](https://github.com/lnccbrown/hssm/issues/387) | Specifying priors for categorical variables in regression does not work | 2024-04-05 | 863 |
| [#412](https://github.com/lnccbrown/hssm/issues/412) | Add `ddm_sdv` onnx model to HF | 2024-05-06 | 832 |
| [#449](https://github.com/lnccbrown/hssm/issues/449) | Prior bounds should override explicit bounds | 2024-05-25 | 813 |
| [#456](https://github.com/lnccbrown/hssm/issues/456) | `nan` grads when running `find_MAP()` on `analytic`, `ddm` | 2024-06-11 | 796 |
| [#458](https://github.com/lnccbrown/hssm/issues/458) | Add test configuration for `test_hssm.py` that includes categorial covariates | 2024-06-13 | 794 |
| [#462](https://github.com/lnccbrown/hssm/issues/462) | Convergence issues when running model with categorical covariates in hierarchy | 2024-06-14 | 793 |

## Most discussed issues

| # | Title | Comments |
| --- | --- | ---: |
| [#387](https://github.com/lnccbrown/hssm/issues/387) | Specifying priors for categorical variables in regression does not work | 11 |
| [#494](https://github.com/lnccbrown/hssm/issues/494) | using hierarchical models with random slope resutling in sampling problem | 8 |
| [#1085](https://github.com/lnccbrown/hssm/issues/1085) | New aDDM estimation function | 5 |
| [#844](https://github.com/lnccbrown/hssm/issues/844) | Broadcasting error when using default intercept (1) on DDM model | 4 |
| [#721](https://github.com/lnccbrown/hssm/issues/721) | Posterior predictive checks within HSSM | 4 |
| [#1052](https://github.com/lnccbrown/hssm/issues/1052) | Choice-only RLSSM: ssms.rl presets (ssm-simulators >= 0.13) not buildable through the RV path - smoke tests skipped | 3 |
| [#881](https://github.com/lnccbrown/hssm/issues/881) | Fully support choice only models | 3 |
| [#754](https://github.com/lnccbrown/hssm/issues/754) | `p_outlier` has strictly equality above 0? | 3 |
| [#512](https://github.com/lnccbrown/hssm/issues/512) | Cannot use specific parameter title for custom model config | 3 |
| [#462](https://github.com/lnccbrown/hssm/issues/462) | Convergence issues when running model with categorical covariates in hierarchy | 3 |

## Full listing by category

### Tests / Test suite (14)

| # | Title | Author | Labels | Opened | Comments |
| --- | --- | --- | --- | --- | ---: |
| [#1139](https://github.com/lnccbrown/hssm/issues/1139) | Reduce the number of parameters in integration tests | digicosmos86 | — | 2026-08-05 | 1 |
| [#1138](https://github.com/lnccbrown/hssm/issues/1138) | Move integration tests to tests/integration folder | digicosmos86 | — | 2026-08-05 | 1 |
| [#1120](https://github.com/lnccbrown/hssm/issues/1120) | Misc unit tests moves | cpaniaguam | — | 2026-07-28 | 1 |
| [#1103](https://github.com/lnccbrown/hssm/issues/1103) | Organize plotting tests | cpaniaguam | — | 2026-07-27 | 1 |
| [#1089](https://github.com/lnccbrown/hssm/issues/1089) | Group unit tests | cpaniaguam | — | 2026-07-24 | 1 |
| [#1088](https://github.com/lnccbrown/hssm/issues/1088) | Restructure tests | digicosmos86 | — | 2026-07-24 | 1 |
| [#1081](https://github.com/lnccbrown/hssm/issues/1081) | tests: batch of small measured trims (~5-8 min slow suite, ~1-2 min fast suite) | AlexanderFengler | `chore` | 2026-07-15 | 1 |
| [#1080](https://github.com/lnccbrown/hssm/issues/1080) | tests: test_plotting_cartoon re-samples predictives in every parametrization - ship predictive groups in the cartoon fixtures | AlexanderFengler | `refactor` | 2026-07-15 | 1 |
| [#1079](https://github.com/lnccbrown/hssm/issues/1079) | tests: test_sample_posterior_predictive - 4 cases x ~128 s predict over all 500 posterior draws; use a thinned fixture variant | AlexanderFengler | `refactor` | 2026-07-15 | 1 |
| [#1078](https://github.com/lnccbrown/hssm/issues/1078) | tests: reduce the cloned 20-row sampler-dispatch grids to a coverage-preserving 9-row design | AlexanderFengler | `refactor`, `chore` | 2026-07-15 | 1 |
| [#1075](https://github.com/lnccbrown/hssm/issues/1075) | tests: pytest addopts cleanup - drop global --cov/--exitfirst/--capture=no, add --durations, raise --timeout | AlexanderFengler | `chore` | 2026-07-15 | 1 |
| [#1074](https://github.com/lnccbrown/hssm/issues/1074) | tests: test_addm_waic_loo costs 16.6 min per CI run - deterministic timeout + untimed rerun; replace MCMC with a synthetic posterior | AlexanderFengler | `bug`, `chore` | 2026-07-15 | 1 |
| [#894](https://github.com/lnccbrown/hssm/issues/894) | MCMC sampling hangs in tests/test_save_load.py::test_save_load_vi_mcmc possibly after external dependency updates | cpaniaguam | — | 2026-02-11 | 1 |
| [#458](https://github.com/lnccbrown/hssm/issues/458) | Add test configuration for `test_hssm.py` that includes categorial covariates | AlexanderFengler | — | 2024-06-13 | 0 |

### CI / Release (8)

| # | Title | Author | Labels | Opened | Comments |
| --- | --- | --- | --- | --- | ---: |
| [#1185](https://github.com/lnccbrown/hssm/issues/1185) | drift: HSSM scheduled checks failing | github-actions | `drift` | 2026-08-13 | 1 |
| [#1083](https://github.com/lnccbrown/hssm/issues/1083) | Tracking: eliminate the release-CI timeout and speed up the test suite | AlexanderFengler | `github_actions`, `release` | 2026-07-15 | 1 |
| [#1082](https://github.com/lnccbrown/hssm/issues/1082) | CI: pilot pytest-xdist on one slow batch | AlexanderFengler | `enhancement`, `github_actions` | 2026-07-15 | 1 |
| [#1077](https://github.com/lnccbrown/hssm/issues/1077) | CI: rebalance the slow-test batches (19.8 / 22.9 / 53.3 min -> ~29 / 29 / 29) | AlexanderFengler | `github_actions`, `chore` | 2026-07-15 | 1 |
| [#1076](https://github.com/lnccbrown/hssm/issues/1076) | CI: coverage.yml re-runs the full suite serially (~2 h) on every main push - upload per-batch coverage instead | AlexanderFengler | `github_actions`, `chore` | 2026-07-15 | 1 |
| [#1073](https://github.com/lnccbrown/hssm/issues/1073) | CI: release publish gate runs the full test suite serially in one 90-min job - reuse the 3-batch slow split | AlexanderFengler | `bug`, `github_actions`, `release` | 2026-07-15 | 1 |
| [#1072](https://github.com/lnccbrown/hssm/issues/1072) | Publish workflow: workflow_dispatch builds main instead of the release tag; no version-vs-tag guard; main version regressed to 0.4.0 | AlexanderFengler | — | 2026-07-15 | 1 |
| [#1061](https://github.com/lnccbrown/hssm/issues/1061) | CI: test job can hang 90 min when HuggingFace download stalls during pytest collection | AlexanderFengler | — | 2026-07-14 | 1 |

### RLSSM (14)

| # | Title | Author | Labels | Opened | Comments |
| --- | --- | --- | --- | --- | ---: |
| [#1052](https://github.com/lnccbrown/hssm/issues/1052) | Choice-only RLSSM: ssms.rl presets (ssm-simulators >= 0.13) not buildable through the RV path - smoke tests skipped | AlexanderFengler | — | 2026-07-13 | 3 |
| [#939](https://github.com/lnccbrown/hssm/issues/939) | Add save_model / load_model to RLSSM | cpaniaguam | — | 2026-03-27 | 0 |
| [#872](https://github.com/lnccbrown/hssm/issues/872) | Now that we want to be type-safe, we should make protocols for learning process and decision process functions with different metatdata requirements | cpaniaguam | — | 2026-01-09 | 0 |
| [#871](https://github.com/lnccbrown/hssm/issues/871) | Generalize target param computation framework | cpaniaguam | — | 2026-01-09 | 0 |
| [#870](https://github.com/lnccbrown/hssm/issues/870) | Support padding in RLSSM | cpaniaguam | — | 2026-01-09 | 0 |
| [#813](https://github.com/lnccbrown/hssm/issues/813) | Plotting utilities for RLSSM models | krishnbera | — | 2025-09-21 | 0 |
| [#812](https://github.com/lnccbrown/hssm/issues/812) | RLSSM simulators | krishnbera | — | 2025-09-21 | 0 |
| [#811](https://github.com/lnccbrown/hssm/issues/811) | RLSSM posterior predictive checks | krishnbera | — | 2025-09-21 | 0 |
| [#810](https://github.com/lnccbrown/hssm/issues/810) | Incorporate RLSSM likelihoods | krishnbera | — | 2025-09-21 | 0 |
| [#809](https://github.com/lnccbrown/hssm/issues/809) | Creating pymc Distribution objects | krishnbera | — | 2025-09-21 | 0 |
| [#808](https://github.com/lnccbrown/hssm/issues/808) | Parameter processing for RLSSM models | krishnbera | — | 2025-09-21 | 0 |
| [#807](https://github.com/lnccbrown/hssm/issues/807) | Adjust logic in the HSSM class for missing data and deadlines | krishnbera | — | 2025-09-21 | 0 |
| [#806](https://github.com/lnccbrown/hssm/issues/806) | Add data validators for RLSSM models | krishnbera | — | 2025-09-21 | 0 |
| [#804](https://github.com/lnccbrown/hssm/issues/804) | Full integration of RLSSM | krishnbera | — | 2025-09-21 | 0 |

### Architecture / API (19)

| # | Title | Author | Labels | Opened | Comments |
| --- | --- | --- | --- | --- | ---: |
| [#1146](https://github.com/lnccbrown/hssm/issues/1146) | hssm.load_data() typing | digicosmos86 | — | 2026-08-06 | 1 |
| [#1046](https://github.com/lnccbrown/hssm/issues/1046) | Add outputs to recommend GPU usage | frankmj | — | 2026-07-09 | 1 |
| [#1031](https://github.com/lnccbrown/hssm/issues/1031) | Generalize the model-config registry to support subclass models (aDDM, RLSSM) | AlexanderFengler | — | 2026-07-05 | 1 |
| [#972](https://github.com/lnccbrown/hssm/issues/972) | Populate supported models from registry | cpaniaguam | — | 2026-06-01 | 0 |
| [#952](https://github.com/lnccbrown/hssm/issues/952) | Deprecate legacy `include` in model classes | cpaniaguam | — | 2026-04-28 | 0 |
| [#942](https://github.com/lnccbrown/hssm/issues/942) | Create base.py anew to facilitate merge conflict resolution | cpaniaguam | — | 2026-04-02 | 0 |
| [#934](https://github.com/lnccbrown/hssm/issues/934) | Access configs from config objects | cpaniaguam | — | 2026-03-13 | 0 |
| [#932](https://github.com/lnccbrown/hssm/issues/932) | Clarify support for `choices` as int in HSSM class | cpaniaguam | — | 2026-03-11 | 1 |
| [#930](https://github.com/lnccbrown/hssm/issues/930) | Pass configs via dependency injection into model classes (BaseModelConfig-only; dict supported) | cpaniaguam | `dependencies` | 2026-03-11 | 0 |
| [#895](https://github.com/lnccbrown/hssm/issues/895) | Make _make_model_distribution an abstract method in HSSMBase | cpaniaguam | — | 2026-02-11 | 0 |
| [#873](https://github.com/lnccbrown/hssm/issues/873) | Create the `HSSMBase` class that `HSSM` and `RLSSM` classes both inheirt from | digicosmos86 | — | 2026-01-13 | 0 |
| [#855](https://github.com/lnccbrown/hssm/issues/855) | `params_is_reg` argument should accept a dictionary. | AlexanderFengler | — | 2025-12-02 | 0 |
| [#842](https://github.com/lnccbrown/hssm/issues/842) | fixed vector parameters should be possible | AlexanderFengler | `enhancement` | 2025-11-02 | 0 |
| [#841](https://github.com/lnccbrown/hssm/issues/841) | Print out should include whether model uses centered or non-centered parameterization | AlexanderFengler | `enhancement`, `good first issue` | 2025-10-29 | 0 |
| [#739](https://github.com/lnccbrown/hssm/issues/739) | Refactor: A more general JAX to Pytensor Op wrapper that handles both differentiable and non-differentiable cases | digicosmos86 | — | 2025-06-25 | 2 |
| [#702](https://github.com/lnccbrown/hssm/issues/702) | Consider allowing passing a pymc distribution directly when applying lapse distribution | cpaniaguam | — | 2025-04-05 | 0 |
| [#697](https://github.com/lnccbrown/hssm/issues/697) | consistent treatment of `dt` parameter | AlexanderFengler | `enhancement`, `chore` | 2025-03-30 | 0 |
| [#659](https://github.com/lnccbrown/hssm/issues/659) | Import config functions on demand | cpaniaguam | — | 2025-02-14 | 2 |
| [#512](https://github.com/lnccbrown/hssm/issues/512) | Cannot use specific parameter title for custom model config | AndrewZhang599 | — | 2024-07-20 | 3 |

### Models / Likelihoods (10)

| # | Title | Author | Labels | Opened | Comments |
| --- | --- | --- | --- | --- | ---: |
| [#925](https://github.com/lnccbrown/hssm/issues/925) | Incorporate choice-only simulator from `ssm-simulators` once implemented | digicosmos86 | — | 2026-03-10 | 0 |
| [#881](https://github.com/lnccbrown/hssm/issues/881) | Fully support choice only models | AlexanderFengler | — | 2026-01-22 | 3 |
| [#865](https://github.com/lnccbrown/hssm/issues/865) | Ensure ssm_logp_func are compatible with blackbox/analytical likelihoods | cpaniaguam | — | 2025-12-16 | 0 |
| [#801](https://github.com/lnccbrown/hssm/issues/801) | New Analytical Models: Poisson Race Model | AlexanderFengler | `enhancement`, `good first issue` | 2025-09-12 | 1 |
| [#800](https://github.com/lnccbrown/hssm/issues/800) | New Analytical Models: Extrema Detection | AlexanderFengler | `enhancement`, `good first issue` | 2025-09-12 | 0 |
| [#799](https://github.com/lnccbrown/hssm/issues/799) | New Analytical Models: Racing Diffusion | AlexanderFengler | `enhancement`, `good first issue` | 2025-09-12 | 0 |
| [#798](https://github.com/lnccbrown/hssm/issues/798) | New models: ExGaussian and Shifted Wald | AlexanderFengler | `enhancement`, `good first issue` | 2025-09-12 | 1 |
| [#645](https://github.com/lnccbrown/hssm/issues/645) | Add go / no-go capability and opn / cpn networks for: 'ddm', 'angle', 'weibull' | AlexanderFengler | `pipeline` | 2025-01-31 | 1 |
| [#412](https://github.com/lnccbrown/hssm/issues/412) | Add `ddm_sdv` onnx model to HF | jainraj | `bug` | 2024-05-06 | 3 |
| [#353](https://github.com/lnccbrown/hssm/issues/353) | LANs for full DDM | igrahek | `enhancement` | 2024-02-15 | 1 |

### Sampling / Inference (12)

| # | Title | Author | Labels | Opened | Comments |
| --- | --- | --- | --- | --- | ---: |
| [#1102](https://github.com/lnccbrown/hssm/issues/1102) | add MAP /MLE functions | frankmj | — | 2026-07-27 | 1 |
| [#1045](https://github.com/lnccbrown/hssm/issues/1045) | add function for Savage Dickey Ratio test | frankmj | — | 2026-07-09 | 1 |
| [#1038](https://github.com/lnccbrown/hssm/issues/1038) | Report metrics on inference speed changes before and after the PyMC6 migration | digicosmos86 | — | 2026-07-07 | 1 |
| [#869](https://github.com/lnccbrown/hssm/issues/869) | High number of divergences in DDM sampling | morgbead | — | 2026-01-07 | 0 |
| [#816](https://github.com/lnccbrown/hssm/issues/816) | Failing parameter recovery on truncated data | ddgpalmer | — | 2025-10-01 | 1 |
| [#721](https://github.com/lnccbrown/hssm/issues/721) | Posterior predictive checks within HSSM | AbsDey | — | 2025-05-03 | 4 |
| [#720](https://github.com/lnccbrown/hssm/issues/720) | Boundary parameter a model instability without intercept leads to NaN r_hat | YuanboBQ | — | 2025-04-24 | 1 |
| [#587](https://github.com/lnccbrown/hssm/issues/587) | Reconciling the issues with under-the-hood truncation in HSSM | digicosmos86 | — | 2024-09-20 | 0 |
| [#494](https://github.com/lnccbrown/hssm/issues/494) | using hierarchical models with random slope resutling in sampling problem | YuanboBQ | — | 2024-07-17 | 8 |
| [#462](https://github.com/lnccbrown/hssm/issues/462) | Convergence issues when running model with categorical covariates in hierarchy | AlexanderFengler | — | 2024-06-14 | 3 |
| [#456](https://github.com/lnccbrown/hssm/issues/456) | `nan` grads when running `find_MAP()` on `analytic`, `ddm` | AlexanderFengler | `bug` | 2024-06-11 | 0 |
| [#347](https://github.com/lnccbrown/hssm/issues/347) | Add DIC function as util method | AlexanderFengler | `enhancement` | 2024-01-29 | 0 |

### aDDM (4)

| # | Title | Author | Labels | Opened | Comments |
| --- | --- | --- | --- | --- | ---: |
| [#1085](https://github.com/lnccbrown/hssm/issues/1085) | New aDDM estimation function | JamesWeiChen | — | 2026-07-22 | 5 |
| [#1039](https://github.com/lnccbrown/hssm/issues/1039) | aDDM model cartoon: condition on observed fixations + honor continuation policy (currently generic-SSM re-simulation) | AlexanderFengler | — | 2026-07-08 | 1 |
| [#1012](https://github.com/lnccbrown/hssm/issues/1012) | [ADDM] Review of v0 of ADDM subclass integration | AndrewZhang599 | — | 2026-06-26 | 1 |
| [#958](https://github.com/lnccbrown/hssm/issues/958) | Isolate single stage aDDM from `efficient-fpt` repo and make it an analytical `angle` likelihood | AlexanderFengler | `model`, `likeliihood` | 2026-05-10 | 0 |

### Documentation (6)

| # | Title | Author | Labels | Opened | Comments |
| --- | --- | --- | --- | --- | ---: |
| [#916](https://github.com/lnccbrown/hssm/issues/916) | Add poisson race tutorial to docs properly | AlexanderFengler | — | 2026-03-03 | 0 |
| [#777](https://github.com/lnccbrown/hssm/issues/777) | Fix header and title for HSSM MathPsych tutorial 2025 | AlexanderFengler | — | 2025-08-03 | 0 |
| [#661](https://github.com/lnccbrown/hssm/issues/661) | Update docs/tutorials/main_tutorial.ipynb | cpaniaguam | `documentation` | 2025-02-14 | 1 |
| [#602](https://github.com/lnccbrown/hssm/issues/602) | A tutorial for fitting the DDM to conflic task | suwangcn | — | 2024-11-16 | 1 |
| [#472](https://github.com/lnccbrown/hssm/issues/472) | (Documentation) Add details in huggingface repo for further development of LANs | jainraj | — | 2024-06-24 | 1 |
| [#285](https://github.com/lnccbrown/hssm/issues/285) | Add note on `a` parameter meaning to docs | AlexanderFengler | `documentation` | 2023-09-22 | 1 |

### p_outlier (5)

| # | Title | Author | Labels | Opened | Comments |
| --- | --- | --- | --- | --- | ---: |
| [#794](https://github.com/lnccbrown/hssm/issues/794) | Always include `p_outlier` in `make_distribution` | digicosmos86 | — | 2025-09-10 | 0 |
| [#793](https://github.com/lnccbrown/hssm/issues/793) | Always include `p_outlier` in `make_hssm_rv` | digicosmos86 | — | 2025-09-10 | 0 |
| [#792](https://github.com/lnccbrown/hssm/issues/792) | Always add `p_outlier` to `list_params` in HSSM | digicosmos86 | — | 2025-09-10 | 0 |
| [#791](https://github.com/lnccbrown/hssm/issues/791) | fix confusion around `p_outlier` | digicosmos86 | — | 2025-09-10 | 0 |
| [#754](https://github.com/lnccbrown/hssm/issues/754) | `p_outlier` has strictly equality above 0? | AlexanderFengler | `enhancement` | 2025-07-03 | 3 |

### Priors (5)

| # | Title | Author | Labels | Opened | Comments |
| --- | --- | --- | --- | --- | ---: |
| [#685](https://github.com/lnccbrown/hssm/issues/685) | Default priors are the same for the centered and non-centered parametrization | igrahek | — | 2025-03-12 | 1 |
| [#642](https://github.com/lnccbrown/hssm/issues/642) | Unify prior definitions within code base | cpaniaguam | `nice to have` | 2025-01-28 | 0 |
| [#449](https://github.com/lnccbrown/hssm/issues/449) | Prior bounds should override explicit bounds | AlexanderFengler | `enhancement` | 2024-05-25 | 0 |
| [#387](https://github.com/lnccbrown/hssm/issues/387) | Specifying priors for categorical variables in regression does not work | eort | `bug`, `upstream` | 2024-04-05 | 11 |
| [#312](https://github.com/lnccbrown/hssm/issues/312) | Add some example datasets to package for testing robustness of priors | AlexanderFengler | `enhancement` | 2023-11-01 | 0 |

### Plotting (3)

| # | Title | Author | Labels | Opened | Comments |
| --- | --- | --- | --- | --- | ---: |
| [#1128](https://github.com/lnccbrown/hssm/issues/1128) | plot_model_cartoon: remove the deprecation shims (two minor releases after #1124) | AlexanderFengler | — | 2026-08-02 | 1 |
| [#1126](https://github.com/lnccbrown/hssm/issues/1126) | plot_model_cartoon: kind="kde" with NDT-boundary-corrected kernels | AlexanderFengler | — | 2026-08-02 | 2 |
| [#856](https://github.com/lnccbrown/hssm/issues/856) | hssm.graph() does not render in marimo notebook | Jovan-Kemp | — | 2025-12-06 | 1 |

### BayesFlow / SBI (3)

| # | Title | Author | Labels | Opened | Comments |
| --- | --- | --- | --- | --- | ---: |
| [#954](https://github.com/lnccbrown/hssm/issues/954) | Proper Bayesfow integration | AlexanderFengler | `enhancement` | 2026-05-06 | 2 |
| [#776](https://github.com/lnccbrown/hssm/issues/776) | BayesFlow native contribution pipeline | AlexanderFengler | — | 2025-08-02 | 0 |
| [#775](https://github.com/lnccbrown/hssm/issues/775) | Example: BayesFlow Likelihood as serialized JAX function | AlexanderFengler | — | 2025-08-02 | 0 |

### Other / Bugs (5)

| # | Title | Author | Labels | Opened | Comments |
| --- | --- | --- | --- | --- | ---: |
| [#1189](https://github.com/lnccbrown/hssm/issues/1189) | az.summary HTML repr ignores pandas display.max_rows (arviz-stats SummaryDataFrame) | AlexanderFengler | — | 2026-08-13 | 1 |
| [#1092](https://github.com/lnccbrown/hssm/issues/1092) | Fixing a parameter with backend='jax' fails with a bare AssertionError in specifyshape | AlexanderFengler | — | 2026-07-26 | 1 |
| [#959](https://github.com/lnccbrown/hssm/issues/959) | Misleading initial values output after HSSM clears PyMC initial-value registry | krishnbera | `bug` | 2026-05-11 | 0 |
| [#844](https://github.com/lnccbrown/hssm/issues/844) | Broadcasting error when using default intercept (1) on DDM model | morgbead | `bug` | 2025-11-04 | 4 |
| [#498](https://github.com/lnccbrown/hssm/issues/498) | Different coordinates for some parameters in the inference data | hyang336 | `bug`, `upstream` | 2024-07-18 | 2 |

---

Generated from the GitHub issues API by `issue_inventory/generate_inventory.py`. Issue bodies and comment threads are not duplicated here — follow the links for the full discussion. Categories are assigned in the generator's `RAW` table and are an editorial grouping, not GitHub labels.
