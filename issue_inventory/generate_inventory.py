"""Generate the HSSM open-issue inventory (JSON + CSV + Markdown).

Data source: GitHub REST/GraphQL listing of open issues in lnccbrown/hssm,
snapshotted on SNAPSHOT_DATE. Pull requests are excluded (GitHub's issue
listing for this repo returned issues only).
"""

import csv
import html
import json
from collections import Counter, OrderedDict
from datetime import date, datetime
from pathlib import Path

SNAPSHOT_DATE = date(2026, 8, 16)
REPO = "lnccbrown/hssm"
OUT_DIR = Path(__file__).resolve().parent

# The snapshot itself: one row per open issue, kept as a dense table so the diff of a
# future re-snapshot stays readable. Long lines and hand alignment are intentional.
# fmt: off
# ruff: noqa: E501
# number, title, author, labels, comments, created_at, updated_at, category
RAW = [
    (1189, "az.summary HTML repr ignores pandas display.max_rows (arviz-stats SummaryDataFrame)", "AlexanderFengler", [], 1, "2026-08-13T22:31:15Z", "2026-08-13T22:31:19Z", "Other / Bugs"),
    (1185, "drift: HSSM scheduled checks failing", "github-actions", ["drift"], 1, "2026-08-13T20:36:44Z", "2026-08-13T20:36:48Z", "CI / Release"),
    (1146, "hssm.load_data() typing", "digicosmos86", [], 1, "2026-08-06T13:32:08Z", "2026-08-06T13:32:13Z", "Architecture / API"),
    (1139, "Reduce the number of parameters in integration tests", "digicosmos86", [], 1, "2026-08-05T15:30:57Z", "2026-08-05T15:31:01Z", "Tests / Test suite"),
    (1138, "Move integration tests to tests/integration folder", "digicosmos86", [], 1, "2026-08-05T15:30:08Z", "2026-08-05T15:30:12Z", "Tests / Test suite"),
    (1128, "plot_model_cartoon: remove the deprecation shims (two minor releases after #1124)", "AlexanderFengler", [], 1, "2026-08-02T15:59:40Z", "2026-08-02T15:59:43Z", "Plotting"),
    (1126, "plot_model_cartoon: kind=&#34;kde&#34; with NDT-boundary-corrected kernels", "AlexanderFengler", [], 2, "2026-08-02T15:59:37Z", "2026-08-02T20:13:47Z", "Plotting"),
    (1120, "Misc unit tests moves", "cpaniaguam", [], 1, "2026-07-28T19:52:46Z", "2026-07-28T20:42:27Z", "Tests / Test suite"),
    (1103, "Organize plotting tests", "cpaniaguam", [], 1, "2026-07-27T17:34:57Z", "2026-07-27T17:35:02Z", "Tests / Test suite"),
    (1102, "add MAP /MLE functions", "frankmj", [], 1, "2026-07-27T15:57:29Z", "2026-07-27T15:57:33Z", "Sampling / Inference"),
    (1092, "Fixing a parameter with backend=&#39;jax&#39; fails with a bare AssertionError in specifyshape", "AlexanderFengler", [], 1, "2026-07-26T04:09:36Z", "2026-07-26T04:09:40Z", "Other / Bugs"),
    (1089, "Group unit tests", "cpaniaguam", [], 1, "2026-07-24T15:54:00Z", "2026-07-24T15:54:16Z", "Tests / Test suite"),
    (1088, "Restructure tests", "digicosmos86", [], 1, "2026-07-24T13:48:08Z", "2026-07-24T13:48:12Z", "Tests / Test suite"),
    (1085, "New aDDM estimation function", "JamesWeiChen", [], 5, "2026-07-22T12:24:02Z", "2026-07-29T20:44:08Z", "aDDM"),
    (1083, "Tracking: eliminate the release-CI timeout and speed up the test suite", "AlexanderFengler", ["github_actions", "release"], 1, "2026-07-15T18:34:57Z", "2026-07-15T18:35:02Z", "CI / Release"),
    (1082, "CI: pilot pytest-xdist on one slow batch", "AlexanderFengler", ["enhancement", "github_actions"], 1, "2026-07-15T18:34:55Z", "2026-07-15T18:34:59Z", "CI / Release"),
    (1081, "tests: batch of small measured trims (~5-8 min slow suite, ~1-2 min fast suite)", "AlexanderFengler", ["chore"], 1, "2026-07-15T18:34:54Z", "2026-07-15T18:35:11Z", "Tests / Test suite"),
    (1080, "tests: test_plotting_cartoon re-samples predictives in every parametrization - ship predictive groups in the cartoon fixtures", "AlexanderFengler", ["refactor"], 1, "2026-07-15T18:34:52Z", "2026-07-15T18:34:57Z", "Tests / Test suite"),
    (1079, "tests: test_sample_posterior_predictive - 4 cases x ~128 s predict over all 500 posterior draws; use a thinned fixture variant", "AlexanderFengler", ["refactor"], 1, "2026-07-15T18:34:50Z", "2026-07-15T18:34:54Z", "Tests / Test suite"),
    (1078, "tests: reduce the cloned 20-row sampler-dispatch grids to a coverage-preserving 9-row design", "AlexanderFengler", ["refactor", "chore"], 1, "2026-07-15T18:34:48Z", "2026-07-15T18:34:53Z", "Tests / Test suite"),
    (1077, "CI: rebalance the slow-test batches (19.8 / 22.9 / 53.3 min -> ~29 / 29 / 29)", "AlexanderFengler", ["github_actions", "chore"], 1, "2026-07-15T18:34:47Z", "2026-07-15T18:34:51Z", "CI / Release"),
    (1076, "CI: coverage.yml re-runs the full suite serially (~2 h) on every main push - upload per-batch coverage instead", "AlexanderFengler", ["github_actions", "chore"], 1, "2026-07-15T18:34:45Z", "2026-07-15T18:34:49Z", "CI / Release"),
    (1075, "tests: pytest addopts cleanup - drop global --cov/--exitfirst/--capture=no, add --durations, raise --timeout", "AlexanderFengler", ["chore"], 1, "2026-07-15T18:34:43Z", "2026-07-15T18:34:47Z", "Tests / Test suite"),
    (1074, "tests: test_addm_waic_loo costs 16.6 min per CI run - deterministic timeout + untimed rerun; replace MCMC with a synthetic posterior", "AlexanderFengler", ["bug", "chore"], 1, "2026-07-15T18:34:41Z", "2026-07-15T18:34:45Z", "Tests / Test suite"),
    (1073, "CI: release publish gate runs the full test suite serially in one 90-min job - reuse the 3-batch slow split", "AlexanderFengler", ["bug", "github_actions", "release"], 1, "2026-07-15T18:34:39Z", "2026-07-15T18:34:43Z", "CI / Release"),
    (1072, "Publish workflow: workflow_dispatch builds main instead of the release tag; no version-vs-tag guard; main version regressed to 0.4.0", "AlexanderFengler", [], 1, "2026-07-15T18:29:25Z", "2026-07-15T18:29:30Z", "CI / Release"),
    (1061, "CI: test job can hang 90 min when HuggingFace download stalls during pytest collection", "AlexanderFengler", [], 1, "2026-07-14T01:02:09Z", "2026-07-14T01:02:13Z", "CI / Release"),
    (1052, "Choice-only RLSSM: ssms.rl presets (ssm-simulators &gt;= 0.13) not buildable through the RV path - smoke tests skipped", "AlexanderFengler", [], 3, "2026-07-13T04:23:24Z", "2026-07-13T21:00:45Z", "RLSSM"),
    (1046, "Add outputs to recommend GPU usage", "frankmj", [], 1, "2026-07-09T15:55:48Z", "2026-07-09T15:55:51Z", "Architecture / API"),
    (1045, "add function for Savage Dickey Ratio test", "frankmj", [], 1, "2026-07-09T15:53:54Z", "2026-07-09T15:53:58Z", "Sampling / Inference"),
    (1039, "aDDM model cartoon: condition on observed fixations + honor continuation policy (currently generic-SSM re-simulation)", "AlexanderFengler", [], 1, "2026-07-08T12:24:09Z", "2026-07-08T12:24:12Z", "aDDM"),
    (1038, "Report metrics on inference speed changes before and after the PyMC6 migration", "digicosmos86", [], 1, "2026-07-07T18:14:52Z", "2026-07-07T18:14:55Z", "Sampling / Inference"),
    (1031, "Generalize the model-config registry to support subclass models (aDDM, RLSSM)", "AlexanderFengler", [], 1, "2026-07-05T19:16:51Z", "2026-07-05T19:16:54Z", "Architecture / API"),
    (1012, "[ADDM] Review of v0 of ADDM subclass integration", "AndrewZhang599", [], 1, "2026-06-26T06:06:36Z", "2026-06-26T06:06:40Z", "aDDM"),
    (972, "Populate supported models from registry", "cpaniaguam", [], 0, "2026-06-01T20:27:51Z", "2026-06-01T20:28:45Z", "Architecture / API"),
    (959, "Misleading initial values output after HSSM clears PyMC initial-value registry", "krishnbera", ["bug"], 0, "2026-05-11T17:52:52Z", "2026-05-11T17:52:52Z", "Other / Bugs"),
    (958, "Isolate single stage aDDM from `efficient-fpt` repo and make it an analytical `angle` likelihood", "AlexanderFengler", ["model", "likeliihood"], 0, "2026-05-10T04:42:27Z", "2026-05-10T04:42:27Z", "aDDM"),
    (954, "Proper Bayesfow integration", "AlexanderFengler", ["enhancement"], 2, "2026-05-06T00:11:04Z", "2026-05-20T17:59:24Z", "BayesFlow / SBI"),
    (952, "Deprecate legacy `include` in model classes", "cpaniaguam", [], 0, "2026-04-28T19:02:36Z", "2026-04-28T19:02:36Z", "Architecture / API"),
    (942, "Create base.py anew to facilitate merge conflict resolution", "cpaniaguam", [], 0, "2026-04-02T17:13:17Z", "2026-04-02T17:13:34Z", "Architecture / API"),
    (939, "Add save_model / load_model to RLSSM", "cpaniaguam", [], 0, "2026-03-27T14:34:38Z", "2026-03-27T14:34:38Z", "RLSSM"),
    (934, "Access configs from config objects", "cpaniaguam", [], 0, "2026-03-13T13:40:00Z", "2026-03-13T13:40:00Z", "Architecture / API"),
    (932, "Clarify support for `choices` as int in HSSM class", "cpaniaguam", [], 1, "2026-03-11T18:58:16Z", "2026-03-11T19:01:11Z", "Architecture / API"),
    (930, "Pass configs via dependency injection into model classes (BaseModelConfig-only; dict supported)", "cpaniaguam", ["dependencies"], 0, "2026-03-11T14:26:51Z", "2026-03-11T14:26:51Z", "Architecture / API"),
    (925, "Incorporate choice-only simulator from `ssm-simulators` once implemented", "digicosmos86", [], 0, "2026-03-10T13:33:43Z", "2026-03-10T13:34:14Z", "Models / Likelihoods"),
    (916, "Add poisson race tutorial to docs properly", "AlexanderFengler", [], 0, "2026-03-03T16:54:31Z", "2026-03-03T16:54:31Z", "Documentation"),
    (895, "Make _make_model_distribution an abstract method in HSSMBase", "cpaniaguam", [], 0, "2026-02-11T18:38:26Z", "2026-02-11T18:38:26Z", "Architecture / API"),
    (894, "MCMC sampling hangs in tests/test_save_load.py::test_save_load_vi_mcmc possibly after external dependency updates", "cpaniaguam", [], 1, "2026-02-11T17:53:16Z", "2026-02-11T17:53:46Z", "Tests / Test suite"),
    (881, "Fully support choice only models", "AlexanderFengler", [], 3, "2026-01-22T06:17:19Z", "2026-01-25T15:40:22Z", "Models / Likelihoods"),
    (873, "Create the `HSSMBase` class that `HSSM` and `RLSSM` classes both inheirt from", "digicosmos86", [], 0, "2026-01-13T15:47:54Z", "2026-01-13T15:48:02Z", "Architecture / API"),
    (872, "Now that we want to be type-safe, we should make protocols for learning process and decision process functions with different metatdata requirements", "cpaniaguam", [], 0, "2026-01-09T20:13:15Z", "2026-01-09T20:13:15Z", "RLSSM"),
    (871, "Generalize target param computation framework", "cpaniaguam", [], 0, "2026-01-09T20:11:08Z", "2026-01-09T20:11:08Z", "RLSSM"),
    (870, "Support padding in RLSSM", "cpaniaguam", [], 0, "2026-01-09T16:06:19Z", "2026-01-09T16:06:39Z", "RLSSM"),
    (869, "High number of divergences in DDM sampling", "morgbead", [], 0, "2026-01-07T15:11:21Z", "2026-01-07T15:45:03Z", "Sampling / Inference"),
    (865, "Ensure ssm_logp_func are compatible with blackbox/analytical likelihoods", "cpaniaguam", [], 0, "2025-12-16T15:08:00Z", "2025-12-16T15:08:00Z", "Models / Likelihoods"),
    (856, "hssm.graph() does not render in marimo notebook", "Jovan-Kemp", [], 1, "2025-12-06T21:18:15Z", "2025-12-11T19:07:33Z", "Plotting"),
    (855, "`params_is_reg` argument should accept a dictionary.", "AlexanderFengler", [], 0, "2025-12-02T18:05:45Z", "2025-12-02T18:05:45Z", "Architecture / API"),
    (844, "Broadcasting error when using default intercept (1) on DDM model", "morgbead", ["bug"], 4, "2025-11-04T22:18:31Z", "2025-12-12T19:28:19Z", "Other / Bugs"),
    (842, "fixed vector parameters should be possible", "AlexanderFengler", ["enhancement"], 0, "2025-11-02T03:57:11Z", "2025-11-02T19:15:50Z", "Architecture / API"),
    (841, "Print out should include whether model uses centered or non-centered parameterization", "AlexanderFengler", ["enhancement", "good first issue"], 0, "2025-10-29T15:28:57Z", "2025-10-29T15:28:57Z", "Architecture / API"),
    (816, "Failing parameter recovery on truncated data", "ddgpalmer", [], 1, "2025-10-01T22:33:32Z", "2025-10-15T18:27:28Z", "Sampling / Inference"),
    (813, "Plotting utilities for RLSSM models", "krishnbera", [], 0, "2025-09-21T18:48:06Z", "2025-09-21T18:53:39Z", "RLSSM"),
    (812, "RLSSM simulators", "krishnbera", [], 0, "2025-09-21T18:47:46Z", "2026-05-20T00:03:14Z", "RLSSM"),
    (811, "RLSSM posterior predictive checks", "krishnbera", [], 0, "2025-09-21T18:47:19Z", "2025-09-21T18:53:26Z", "RLSSM"),
    (810, "Incorporate RLSSM likelihoods", "krishnbera", [], 0, "2025-09-21T18:46:51Z", "2025-09-21T20:11:53Z", "RLSSM"),
    (809, "Creating pymc Distribution objects", "krishnbera", [], 0, "2025-09-21T18:42:37Z", "2025-09-21T18:42:37Z", "RLSSM"),
    (808, "Parameter processing for RLSSM models", "krishnbera", [], 0, "2025-09-21T18:39:01Z", "2025-09-21T19:39:24Z", "RLSSM"),
    (807, "Adjust logic in the HSSM class for missing data and deadlines", "krishnbera", [], 0, "2025-09-21T18:37:34Z", "2025-09-21T19:30:53Z", "RLSSM"),
    (806, "Add data validators for RLSSM models", "krishnbera", [], 0, "2025-09-21T18:36:34Z", "2025-09-21T19:38:54Z", "RLSSM"),
    (804, "Full integration of RLSSM", "krishnbera", [], 0, "2025-09-21T18:15:14Z", "2025-09-21T18:15:14Z", "RLSSM"),
    (801, "New Analytical Models: Poisson Race Model", "AlexanderFengler", ["enhancement", "good first issue"], 1, "2025-09-12T14:50:16Z", "2025-09-18T23:07:23Z", "Models / Likelihoods"),
    (800, "New Analytical Models: Extrema Detection", "AlexanderFengler", ["enhancement", "good first issue"], 0, "2025-09-12T14:43:51Z", "2025-09-12T14:43:51Z", "Models / Likelihoods"),
    (799, "New Analytical Models: Racing Diffusion", "AlexanderFengler", ["enhancement", "good first issue"], 0, "2025-09-12T14:42:08Z", "2025-09-12T14:42:08Z", "Models / Likelihoods"),
    (798, "New models: ExGaussian and Shifted Wald", "AlexanderFengler", ["enhancement", "good first issue"], 1, "2025-09-12T14:40:40Z", "2025-10-03T15:45:33Z", "Models / Likelihoods"),
    (794, "Always include `p_outlier` in `make_distribution`", "digicosmos86", [], 0, "2025-09-10T14:13:27Z", "2025-09-10T14:13:27Z", "p_outlier"),
    (793, "Always include `p_outlier` in `make_hssm_rv`", "digicosmos86", [], 0, "2025-09-10T14:12:30Z", "2025-09-10T14:12:30Z", "p_outlier"),
    (792, "Always add `p_outlier` to `list_params` in HSSM", "digicosmos86", [], 0, "2025-09-10T14:10:12Z", "2025-09-10T14:10:12Z", "p_outlier"),
    (791, "fix confusion around `p_outlier`", "digicosmos86", [], 0, "2025-09-10T14:09:38Z", "2025-09-10T14:09:38Z", "p_outlier"),
    (777, "Fix header and title for HSSM MathPsych tutorial 2025", "AlexanderFengler", [], 0, "2025-08-03T20:24:46Z", "2025-08-03T20:24:46Z", "Documentation"),
    (776, "BayesFlow native contribution pipeline", "AlexanderFengler", [], 0, "2025-08-02T20:13:47Z", "2025-08-02T20:13:47Z", "BayesFlow / SBI"),
    (775, "Example: BayesFlow Likelihood as serialized JAX function", "AlexanderFengler", [], 0, "2025-08-02T20:10:52Z", "2025-08-02T20:11:08Z", "BayesFlow / SBI"),
    (754, "`p_outlier` has strictly equality above 0?", "AlexanderFengler", ["enhancement"], 3, "2025-07-03T08:59:48Z", "2025-08-05T13:51:54Z", "p_outlier"),
    (739, "Refactor: A more general JAX to Pytensor Op wrapper that handles both differentiable and non-differentiable cases", "digicosmos86", [], 2, "2025-06-25T12:56:41Z", "2025-08-05T14:05:45Z", "Architecture / API"),
    (721, "Posterior predictive checks within HSSM", "AbsDey", [], 4, "2025-05-03T18:05:17Z", "2025-05-06T17:55:34Z", "Sampling / Inference"),
    (720, "Boundary parameter a model instability without intercept leads to NaN r_hat", "YuanboBQ", [], 1, "2025-04-24T13:51:04Z", "2025-05-31T03:36:30Z", "Sampling / Inference"),
    (702, "Consider allowing passing a pymc distribution directly when applying lapse distribution", "cpaniaguam", [], 0, "2025-04-05T23:31:24Z", "2025-04-05T23:31:24Z", "Architecture / API"),
    (697, "consistent treatment of `dt` parameter", "AlexanderFengler", ["enhancement", "chore"], 0, "2025-03-30T19:18:41Z", "2025-08-02T19:40:33Z", "Architecture / API"),
    (685, "Default priors are the same for the centered and non-centered parametrization", "igrahek", [], 1, "2025-03-12T00:46:55Z", "2025-03-19T02:37:42Z", "Priors"),
    (661, "Update docs/tutorials/main_tutorial.ipynb", "cpaniaguam", ["documentation"], 1, "2025-02-14T19:57:02Z", "2025-03-13T01:32:25Z", "Documentation"),
    (659, "Import config functions on demand", "cpaniaguam", [], 2, "2025-02-14T17:21:19Z", "2025-08-02T19:43:54Z", "Architecture / API"),
    (645, "Add go / no-go capability and opn / cpn networks for: &#39;ddm&#39;, &#39;angle&#39;, &#39;weibull&#39;", "AlexanderFengler", ["pipeline"], 1, "2025-01-31T13:31:55Z", "2025-04-09T15:50:35Z", "Models / Likelihoods"),
    (642, "Unify prior definitions within code base", "cpaniaguam", ["nice to have"], 0, "2025-01-28T18:41:05Z", "2025-01-28T18:42:13Z", "Priors"),
    (602, "A tutorial for fitting the DDM to conflic task", "suwangcn", [], 1, "2024-11-16T00:47:28Z", "2024-11-28T03:44:30Z", "Documentation"),
    (587, "Reconciling the issues with under-the-hood truncation in HSSM", "digicosmos86", [], 0, "2024-09-20T13:34:56Z", "2024-09-20T13:36:39Z", "Sampling / Inference"),
    (512, "Cannot use specific parameter title for custom model config", "AndrewZhang599", [], 3, "2024-07-20T15:47:12Z", "2024-08-22T15:25:46Z", "Architecture / API"),
    (498, "Different coordinates for some parameters in the inference data", "hyang336", ["bug", "upstream"], 2, "2024-07-18T22:58:51Z", "2025-01-21T13:32:59Z", "Other / Bugs"),
    (494, "using hierarchical models with random slope resutling in sampling problem", "YuanboBQ", [], 8, "2024-07-17T04:07:06Z", "2024-07-17T16:28:36Z", "Sampling / Inference"),
    (472, "(Documentation) Add details in huggingface repo for further development of LANs", "jainraj", [], 1, "2024-06-24T05:50:26Z", "2024-06-25T02:00:22Z", "Documentation"),
    (462, "Convergence issues when running model with categorical covariates in hierarchy", "AlexanderFengler", [], 3, "2024-06-14T16:51:58Z", "2025-01-21T13:41:42Z", "Sampling / Inference"),
    (458, "Add test configuration for `test_hssm.py` that includes categorial covariates", "AlexanderFengler", [], 0, "2024-06-13T15:42:29Z", "2024-06-13T15:42:29Z", "Tests / Test suite"),
    (456, "`nan` grads when running `find_MAP()` on `analytic`, `ddm`", "AlexanderFengler", ["bug"], 0, "2024-06-11T02:29:21Z", "2024-06-13T15:31:23Z", "Sampling / Inference"),
    (449, "Prior bounds should override explicit bounds", "AlexanderFengler", ["enhancement"], 0, "2024-05-25T23:47:23Z", "2024-05-25T23:47:24Z", "Priors"),
    (412, "Add `ddm_sdv` onnx model to HF", "jainraj", ["bug"], 3, "2024-05-06T13:47:47Z", "2024-06-21T05:50:32Z", "Models / Likelihoods"),
    (387, "Specifying priors for categorical variables in regression does not work", "eort", ["bug", "upstream"], 11, "2024-04-05T13:18:49Z", "2026-07-30T16:34:48Z", "Priors"),
    (353, "LANs for full DDM", "igrahek", ["enhancement"], 1, "2024-02-15T22:57:31Z", "2024-05-23T14:56:48Z", "Models / Likelihoods"),
    (347, "Add DIC function as util method", "AlexanderFengler", ["enhancement"], 0, "2024-01-29T15:18:42Z", "2024-01-29T15:18:43Z", "Sampling / Inference"),
    (312, "Add some example datasets to package for testing robustness of priors", "AlexanderFengler", ["enhancement"], 0, "2023-11-01T18:45:56Z", "2023-11-01T18:45:56Z", "Priors"),
    (285, "Add note on `a` parameter meaning to docs", "AlexanderFengler", ["documentation"], 1, "2023-09-22T20:15:40Z", "2023-09-27T17:37:43Z", "Documentation"),
]
# fmt: on

CATEGORY_ORDER = [
    "Tests / Test suite",
    "CI / Release",
    "RLSSM",
    "Architecture / API",
    "Models / Likelihoods",
    "Sampling / Inference",
    "aDDM",
    "Documentation",
    "p_outlier",
    "Priors",
    "Plotting",
    "BayesFlow / SBI",
    "Other / Bugs",
]


def parse_ts(value: str) -> datetime:
    """Parse a GitHub ISO-8601 UTC timestamp."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def build_records():
    """Normalize the RAW table into sorted, enriched issue records."""
    records = []
    for number, title, author, labels, comments, created, updated, category in RAW:
        created_dt = parse_ts(created)
        updated_dt = parse_ts(updated)
        records.append(
            OrderedDict(
                number=number,
                title=html.unescape(title),
                url=f"https://github.com/{REPO}/issues/{number}",
                state="open",
                author=author,
                labels=labels,
                comments=comments,
                created_at=created,
                updated_at=updated,
                age_days=(SNAPSHOT_DATE - created_dt.date()).days,
                days_since_update=(SNAPSHOT_DATE - updated_dt.date()).days,
                category=category,
            )
        )
    records.sort(key=lambda r: r["number"], reverse=True)
    return records


def write_json(records):
    """Write the full snapshot as JSON and return the path."""
    payload = {
        "repository": REPO,
        "snapshot_date": SNAPSHOT_DATE.isoformat(),
        "state_filter": "open",
        "total": len(records),
        "issues": records,
    }
    path = OUT_DIR / "open_issues.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def write_csv(records):
    """Write the flat one-row-per-issue CSV and return the path."""
    path = OUT_DIR / "open_issues.csv"
    fields = [
        "number",
        "title",
        "url",
        "state",
        "author",
        "labels",
        "comments",
        "created_at",
        "updated_at",
        "age_days",
        "days_since_update",
        "category",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["labels"] = ";".join(record["labels"])
            writer.writerow(row)
    return path


def write_markdown(records):
    """Write the human-facing categorized Markdown report and return the path."""
    by_category = {c: [] for c in CATEGORY_ORDER}
    for record in records:
        by_category[record["category"]].append(record)

    label_counts = Counter(label for r in records for label in r["labels"])
    author_counts = Counter(r["author"] for r in records)
    year_counts = Counter(r["created_at"][:4] for r in records)

    lines = [
        "# HSSM open issues — inventory",
        "",
        f"Snapshot of every **open** issue in [`{REPO}`](https://github.com/{REPO}/issues) "
        f"as of **{SNAPSHOT_DATE.isoformat()}**. Pull requests are excluded.",
        "",
        f"**Total open issues: {len(records)}**",
        "",
        "Machine-readable companions: [`open_issues.csv`](open_issues.csv), "
        "[`open_issues.json`](open_issues.json). Regenerate all three with "
        "`uv run python issue_inventory/generate_inventory.py`.",
        "",
        "## By category",
        "",
        "| Category | Count | Share |",
        "| --- | ---: | ---: |",
    ]
    for category in CATEGORY_ORDER:
        count = len(by_category[category])
        lines.append(f"| {category} | {count} | {count / len(records):.0%} |")
    lines += ["", "## By year opened", "", "| Year | Count |", "| --- | ---: |"]
    for year in sorted(year_counts):
        lines.append(f"| {year} | {year_counts[year]} |")

    lines += [
        "",
        "## Most frequent reporters",
        "",
        "| Author | Count |",
        "| --- | ---: |",
    ]
    for author, count in author_counts.most_common(10):
        lines.append(f"| {author} | {count} |")

    lines += ["", "## Labels in use", "", "| Label | Count |", "| --- | ---: |"]
    for label, count in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{label}` | {count} |")

    unlabeled = sum(1 for r in records if not r["labels"])
    lines += [
        "",
        f"{unlabeled} of {len(records)} open issues ({unlabeled / len(records):.0%}) carry no label.",
        "",
        "## Oldest still-open issues",
        "",
        "| # | Title | Opened | Age (days) |",
        "| --- | --- | --- | ---: |",
    ]
    for record in sorted(records, key=lambda r: r["created_at"])[:10]:
        lines.append(
            f"| [#{record['number']}]({record['url']}) | {record['title']} "
            f"| {record['created_at'][:10]} | {record['age_days']} |"
        )

    lines += [
        "",
        "## Most discussed issues",
        "",
        "| # | Title | Comments |",
        "| --- | --- | ---: |",
    ]
    for record in sorted(records, key=lambda r: -r["comments"])[:10]:
        lines.append(
            f"| [#{record['number']}]({record['url']}) | {record['title']} | {record['comments']} |"
        )

    lines += ["", "## Full listing by category", ""]
    for category in CATEGORY_ORDER:
        bucket = by_category[category]
        lines += [
            f"### {category} ({len(bucket)})",
            "",
            "| # | Title | Author | Labels | Opened | Comments |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
        for record in bucket:
            labels = ", ".join(f"`{lab}`" for lab in record["labels"]) or "—"
            lines.append(
                f"| [#{record['number']}]({record['url']}) | {record['title']} "
                f"| {record['author']} | {labels} | {record['created_at'][:10]} "
                f"| {record['comments']} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "Generated from the GitHub issues API by `issue_inventory/generate_inventory.py`. "
        "Issue bodies and comment threads are not duplicated here — follow the links for "
        "the full discussion. Categories are assigned in the generator's `RAW` table and "
        "are an editorial grouping, not GitHub labels.",
    ]

    path = OUT_DIR / "OPEN_ISSUES.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    """Regenerate all three inventory artifacts."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()
    assert len(records) == 108, len(records)
    assert len({r["number"] for r in records}) == len(records), (
        "duplicate issue numbers"
    )
    unknown = {r["category"] for r in records} - set(CATEGORY_ORDER)
    assert not unknown, unknown
    for path in (write_json(records), write_csv(records), write_markdown(records)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
