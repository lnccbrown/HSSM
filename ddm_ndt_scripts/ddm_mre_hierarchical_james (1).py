"""
Minimal reproducible example: hierarchical DDM, `analytical` vs
`approx_differentiable`, with link functions as a second factor
================================================================
Companion to lnccbrown/HSSM#1085.

Simulates a hierarchical DDM dataset from KNOWN group-level parameters, then
fits the SAME model twice, changing only `loglik_kind`. Everything else --
data, formulas, random-effect structure, priors, seed, sampler settings -- is
held fixed, so any difference in sampling behaviour is attributable to the
likelihood implementation alone.

The generating parameters are the posterior means of a hierarchical
`analytical` fit to our real 44-subject dataset (8,427 trials), so the
simulated data is faithful to the case where we originally saw the problem.

    # on the real data (needs ddm_ladder_real_data.csv in the working directory)
    python ddm_mre_hierarchical_james.py --data real --lik analytical
    python ddm_mre_hierarchical_james.py --data real --lik approx_differentiable
    python ddm_mre_hierarchical_james.py --data real --lik approx_differentiable --links log_logit

    # same three on simulated data
    python ddm_mre_hierarchical_james.py --lik analytical

`--links log_logit` tests @frankmj's suggestion that link functions on bounded
parameters aid convergence. On our *real* data with the hierarchical aDDM it
raised the step size by ~1000x but did not stop the tree depth from saturating.

Observed on our real data (44 subjects, 8,427 trials, identical rows across all
three runs, 4 chains, 1000 tune / 1000 draws, max_tree_depth=10):

                          analytical   approx_diff      approx_diff
                                         (identity)    (log_logit)
    max r_hat                  1.010         3.432          2.302
    min ess_bulk                 345             4              5
    tree depth saturation    0% (med 5)      100%           100%
    final step size            0.108      3.63e-04       5.79e-04
    divergences                    0             0              0
    wall clock               7.8 min     348.9 min      347.7 min

`link_settings="log_logit"` (which puts gen_logit on every bounded parameter --
v -3-3, a 0.3-2.5, z 0-1, t 0-2 under approx_differentiable) moves things in the
right direction but does not fix it: the tree depth still saturates on 100% of
iterations and ESS stays in single digits. Zero divergences throughout.

An earlier version of this file argued that bounds could not be involved because
0.000% of posterior *draws* fell outside them. That reasoning is invalid -- draws
leaving the support are rejected, so accepted draws are in-bounds by
construction. The log_logit arm above is the proper test.

The same real dataset used for these three runs is included as
`ddm_ladder_real_data.csv` (rt, response, vdiff, participant_id).

Three corrections since the table above was produced -- rerun before quoting it:

  1. Priors are now specified explicitly instead of taking HSSM's defaults.
     The defaults are Normal(0, 0.25) on the LINK scale, i.e. centred on the
     midpoint of each parameter's bounds; under approx_differentiable that is a
     prior centred at t = 1.0 s and a = 1.4, which puts the values this dataset
     supports 2-3 prior SDs into the tail. Priors here are stated in parameter
     units and mapped onto whichever link each arm actually uses, so both arms
     encode the same belief (see the PRIORS block).
  2. Group offsets have `mu` pinned at 0. With a common intercept in the
     formula, a second estimated mean is redundant with it. HSSM already does
     this for `1|participant_id`, but for a random *slope* it emits
     Normal(mu=Normal(0, 0.25), ...) -- and since the offsets are non-centred,
     that mu never reaches the likelihood. PyMC warns about it directly:
     "free random variables that do not influence the likelihood:
     'v_vdiff|participant_id_mu'".
  3. v now carries a random slope as well as a random intercept:
     `v ~ 1 + vdiff + (1 + vdiff|participant_id)`. The simulator draws a
     matching per-subject slope so the generative process is not a special case
     of the fitted model with the slope variance pinned at zero.

`--no-ndt-guard` -- the non-decision-time guard as a factor
-----------------------------------------------------------
HSSM forbids `t` from exceeding a trial's RT with `ensure_positive_ndt`
(`distribution_utils/dist.py`), which overwrites the log-likelihood with the
floor `LOGP_LB = -66.1` wherever `rt - t <= 1e-15`. Both arms get the identical
guard, but it does very different things to them:

  analytical   the Navarro-Fuss density has ALREADY collapsed (logp ~ -67, and
               below the p_outlier lapse floor of log(0.05/20) = -5.99) by the
               time t reaches rt, so the guard overwrites a value that is
               already there. Measured jump through the mixture: +0.0000.
  LAN          the network cannot represent a collapse to zero and saturates at
               logp ~ -8 to -9, which is the same order as the lapse floor. The
               guard therefore drops a live value onto the floor, producing a
               genuine STEP DISCONTINUITY. Measured jump: -0.04 to -0.12.

That step is negligible against a log-posterior of ~13,000 but is a third to a
half of the dH budget NUTS' step-size adaptation targets (target_accept 0.8
=> dH ~ 0.22), and unlike ordinary integration error it does not shrink with the
step size -- only the crossing RATE does. Measured at the posterior mean of the
real-data runs, switching the guard off moves |dH| at eps=0.03 from 0.853 to
0.0069 for the LAN and leaves the analytical at 0.0065 either way (see section
11 of ddm_data_comparison.ipynb and ndt_guard_dh_precompute.py).

This flag turns the guard off for a real fit, which is the end-to-end test that
static measurement cannot give: does r_hat/ESS actually recover?

    python ddm_mre_hierarchical_james.py --data real --lik approx_differentiable \
        --links log_logit --no-ndt-guard

READ THE RESULT WITH CARE. This is a DIAGNOSTIC, not a fix. The analytical
likelihood floors itself internally (`logp_ddm` in likelihoods/analytical.py has
its own `pt.where(rt <= epsilon, LOGP_LB, ...)`), so the guard is decorative
there; the LAN has no such protection and the guard is load-bearing. With it
off, the penalty for `t` overshooting a trial's RT falls from infinite to ~0.1
log units, so `t` is only weakly identified from above and is expected to drift
UPWARD. Clean r_hat here would confirm the mechanism while producing a biased
`t` -- which is why the real fix is a CONTINUOUS guard (e.g. multiplying the
density by a smooth window, `logp + log_sigmoid((rt - t) / tau)`), not a missing
one.
"""

import argparse
import os

# JAX preallocates 75% of VRAM by default; the hierarchical gradient for 44
# subjects peaks above that on a 24 GB card. Must be set before importing jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import numpy as np
import pandas as pd

# Posterior means from a hierarchical `analytical` fit to the real dataset.
TRUE = dict(
    v_intercept=0.0104, v_vdiff=0.2541,   # drift: intercept + slope on value diff
    a=1.1063, z=0.5068, t=0.7139,          # group-level means
    sd_v=0.0655, sd_a=0.1902, sd_z=0.0287, sd_t=0.1274,   # between-subject SDs
    # ASSUMED: the reference fit had no random slope on vdiff, so there is no
    # posterior SD to copy. Set to ~20% of the slope itself.
    sd_v_vdiff=0.05,
)
SEED = 99

# ---------------------------------------------------------------------------
# Priors, stated once in PARAMETER units and mapped onto whichever link is live.
#
# This indirection is necessary because `link_settings="log_logit"` picks a link
# from each parameter's bounds, and the bounds differ by `loglik_kind`:
#
#     analytical             v (-inf, inf)   a (0, inf)     z (0, 1)   t (0, inf)
#     approx_differentiable  v (-3, 3)       a (0.3, 2.5)   z (0, 1)   t (0, 2)
#
# so the same setting yields a mix of identity/log/gen_logit for `analytical`
# but gen_logit on all four for `approx_differentiable`. Handing both arms the
# same link-scale numbers would encode *different* beliefs about the parameters
# and make the comparison meaningless -- which is the whole point of the MRE.
# Stating the prior in parameter units and transferring it per-arm keeps the
# encoded belief identical no matter which link is in play.
#
# For reference, HSSM's own default is Normal(0, 0.25) on the link scale, i.e.
# centred on the midpoint of the bounds. Under approx_differentiable that means
# a prior centred at t = 1.0 s and a = 1.4, which puts the values this dataset
# actually supports 2-3 prior SDs into the tail.
#
# Centres are generic and physically sensible rather than copied from TRUE, so
# the simulated-data recovery check stays honest and the real-data arm does not
# reuse a posterior from the same rows.
# ---------------------------------------------------------------------------
PRIORS = {
    # param: centre, prior SD, and the HalfNormal scale for the group SD,
    #        all in parameter units
    "v": dict(loc=0.0, sd=1.0,  grp=0.50),
    "a": dict(loc=1.2, sd=0.5,  grp=0.25),
    "z": dict(loc=0.5, sd=0.15, grp=0.075),
    "t": dict(loc=0.5, sd=0.3,  grp=0.15),
}
# The vdiff slope is a rate of change, not a location, so it crosses the link by
# the delta-method rule (divide by the slope of the inverse link), never by the
# logit. Same rule the between-subject SDs use.
V_SLOPE = dict(sd=0.5, grp=0.25)

# The approx_differentiable bounds are the LAN's TRAINING BOX, not the DDM's
# mathematical support. `--bounds lan` hands them to the analytical likelihood
# too, which is the only way to compare the two likelihoods with the induced
# parameterisation held fixed: bounds pick the link (log for (0, inf),
# gen_logit for a finite interval), so the two arms otherwise never share a
# geometry. The reverse cell -- the LAN under analytical's (0, inf) bounds --
# is not runnable, because the network has no valid output outside its box.
LAN_BOUNDS = {"v": (-3.0, 3.0), "a": (0.3, 2.5), "z": (0.0, 1.0), "t": (0.0, 2.0)}


def link_of(bounds, links):
    """Mirror of HSSM's `RegressionParam.set_loglogit_link` bounds dispatch."""
    lo, hi = bounds
    if links != "log_logit":
        return "identity"
    if np.isneginf(lo) and np.isposinf(hi):
        return "identity"
    if lo == 0.0 and np.isposinf(hi):
        return "log"
    if np.isfinite(lo) and np.isfinite(hi):
        return "gen_logit"
    return "identity"


def loc_to_link(x, bounds, kind):
    """A location in parameter units -> the linear-predictor scale."""
    lo, hi = bounds
    if kind == "identity":
        return float(x)
    if kind == "log":
        return float(np.log(x))
    u = (x - lo) / (hi - lo)
    return float(np.log(u / (1.0 - u)))


def link_to_loc(x, bounds, kind):
    """Inverse of `loc_to_link`: linear-predictor scale -> parameter units."""
    lo, hi = bounds
    if kind == "identity":
        return float(x)
    if kind == "log":
        return float(np.exp(x))
    return float(lo + (hi - lo) / (1.0 + np.exp(-x)))


def dparam_dlink(x, bounds, kind):
    """d(parameter)/d(linear predictor), evaluated at x."""
    lo, hi = bounds
    if kind == "identity":
        return 1.0
    if kind == "log":
        return float(x)
    u = (x - lo) / (hi - lo)
    return float((hi - lo) * u * (1.0 - u))


def scale_to_link(s, x, bounds, kind):
    """A slope or SD in parameter units -> the linear-predictor scale."""
    return float(s / dparam_dlink(x, bounds, kind))


def build_include(lik, links, which_bounds="native", t_mode="hierarchical",
                  centered=False, slope_scale=1.0):
    """Formulas + explicit priors for both arms. Returns (include, bounds).

    `which_bounds="lan"` substitutes the LAN training box for whatever the
    likelihood declares. The priors follow automatically: they are specified in
    parameter units, so swapping bounds re-expresses the same belief on the new
    link rather than changing it.

    `centered=True` switches the random effects to the centred parameterisation.
    That is not just a flag on the prior -- it changes what the group term MEANS,
    so the formula has to change with it:

        non-centred   p ~ 1 + (1|g)   p_j = Intercept + z_j * sigma,  z_j ~ N(0,1)
        centred       p ~ 0 + (1|g)   p_j ~ Normal(mu, sigma)

    In the non-centred form the group term holds *offsets* around a common
    intercept, so it needs `1 +` to supply that intercept and its own `mu` is
    pinned at 0 (bambi's non-centred path discards `mu` outright -- see
    `backend/terms.py`, which builds `offset * sigma` and never reads it).
    In the centred form the group term holds the per-subject values themselves
    and carries the group mean in its own `mu` hyperprior, so keeping `1 +`
    would put two group means in the model, additively unidentified with each
    other. Hence `0 +`, and the Intercept prior moves onto the group `mu`.
    Same for the random slope on vdiff: `0 + (1 + vdiff|g)` drops the common
    slope, and V_SLOPE's prior becomes the slope's group-mean hyperprior.
    """
    from hssm.modelconfig.ddm_config import get_ddm_config

    bounds = (dict(LAN_BOUNDS) if which_bounds == "lan"
              else get_ddm_config()["likelihoods"][lik]["bounds"])
    include = []
    regressed = ("v", "a", "z") if t_mode == "fixed" else ("v", "a", "z", "t")
    for p in regressed:
        b, spec = bounds[p], PRIORS[p]
        kind = link_of(b, links)
        # Every number below is the same belief in parameter units, mapped onto
        # whichever link this arm uses; only where it is ATTACHED changes.
        mu_link = loc_to_link(spec["loc"], b, kind)
        sd_link = scale_to_link(spec["sd"], spec["loc"], b, kind)
        grp_link = scale_to_link(spec["grp"], spec["loc"], b, kind)
        # `slope_scale` carries the vdiff rescaling into the slope prior. If the
        # regressor is divided by sigma, the slope it multiplies is multiplied by
        # sigma for the SAME drift range, so the prior has to follow -- otherwise
        # z-scoring silently tightens the prior on v by a factor of sigma and the
        # two arms no longer encode the same belief.
        vsd_link = scale_to_link(V_SLOPE["sd"] * slope_scale, spec["loc"], b, kind)
        vgrp_link = scale_to_link(V_SLOPE["grp"] * slope_scale, spec["loc"], b, kind)

        if centered:
            prior = {
                # The group mean lives here now, not in a common intercept, so
                # it takes the Intercept prior verbatim. `noncentered: False` is
                # what makes bambi honour `mu` at all.
                "1|participant_id": {
                    "name": "Normal",
                    "mu": {"name": "Normal", "mu": mu_link, "sigma": sd_link},
                    "sigma": {"name": "HalfNormal", "sigma": grp_link},
                    "noncentered": False,
                },
            }
            if p == "v":
                prior["vdiff|participant_id"] = {
                    "name": "Normal",
                    "mu": {"name": "Normal", "mu": 0.0, "sigma": vsd_link},
                    "sigma": {"name": "HalfNormal", "sigma": vgrp_link},
                    "noncentered": False,
                }
                formula = "v ~ 0 + (1 + vdiff|participant_id)"
            else:
                formula = f"{p} ~ 0 + (1|participant_id)"
        else:
            prior = {
                "Intercept": {"name": "Normal", "mu": mu_link, "sigma": sd_link},
                # mu pinned at 0. The common intercept already carries the group
                # mean, so a second estimated mean is redundant with it. HSSM
                # does this for `1|...` on its own, but NOT for a random slope:
                # that path gets Normal(mu=Normal(0, 0.25), ...), and because the
                # offsets are non-centred the extra mu never reaches the
                # likelihood at all -- PyMC warns it is a free variable that does
                # not influence it.
                "1|participant_id": {
                    "name": "Normal", "mu": 0.0,
                    "sigma": {"name": "HalfNormal", "sigma": grp_link},
                },
            }
            if p == "v":
                prior["vdiff"] = {"name": "Normal", "mu": 0.0, "sigma": vsd_link}
                prior["vdiff|participant_id"] = {
                    "name": "Normal", "mu": 0.0,
                    "sigma": {"name": "HalfNormal", "sigma": vgrp_link},
                }
                formula = "v ~ 1 + vdiff + (1 + vdiff|participant_id)"
            else:
                formula = f"{p} ~ 1 + (1|participant_id)"
        include.append({"name": p, "formula": formula, "prior": prior})
    if t_mode == "fixed":
        # Same mechanism the aDDM script uses for b: a scalar prior pins the
        # parameter, so PyMC builds no value variable for it at all.
        include.append({"name": "t", "prior": float(TRUE["t"])})
    return include, bounds


def vi_start_point(model, initvals):
    """Map our initvals onto the transformed space that `pm.fit(start=...)` wants.

    Ported from addm_hierarchical_james_bug.py, with one addition the aDDM script
    did not need: this model is run CENTRED as well as non-centred, and those
    have different per-subject latent variables.

        non-centred   {p}_1|participant_id_offset   N(0,1) offsets  -> start at 0
        centred       {p}_1|participant_id          the subject values
                                                    themselves -> start at the
                                                    group mean, since that is
                                                    what `mu` says they are

    Scalars keep the aDDM rules: `{p}_Intercept` / `{p}_1|participant_id_mu` are
    unbounded under log_logit links so the key already matches, while
    `{p}_1|participant_id_sigma` is HalfNormal -> log-transformed, so both the
    key (`_log__`) and the value (logged) have to change. Every key is probed
    against the real `initial_point()` rather than assumed.
    """
    start = model.pymc_model.initial_point()
    mapped, skipped = [], []
    for k, v in initvals.items():
        if k in start:
            start[k] = np.asarray(v, dtype=np.asarray(start[k]).dtype)
            mapped.append(k)
        elif f"{k}_log__" in start:
            kt = f"{k}_log__"
            start[kt] = np.asarray(np.log(max(float(v), 1e-8)),
                                   dtype=np.asarray(start[kt]).dtype)
            mapped.append(kt)
        else:
            skipped.append(k)

    # Per-subject layer. Non-centred offsets are N(0,1) so 0 is the mode;
    # centred subject values are drawn around `mu`, so `mu` is the mode.
    for k in list(start):
        if k.endswith("_offset"):
            start[k] = np.zeros_like(np.asarray(start[k]))
        elif "|participant_id" in k and not k.endswith(("_mu", "_sigma", "_log__")):
            stem = k.rsplit("|", 1)[0]                    # e.g. "v_1" / "v_vdiff"
            mu_key = f"{k}_mu"
            if mu_key in initvals:
                start[k] = np.full_like(np.asarray(start[k]),
                                        float(initvals[mu_key]))
                mapped.append(f"{k} (<- {mu_key})")
    return start, mapped, skipped


def vi_start_sigma(start, sigma):
    """Initial ADVI std-dev around each start value (small = trust the start)."""
    return {k: np.full_like(np.asarray(v), sigma, dtype="float32")
            for k, v in start.items()}


class VITrace:
    """Record the variational mean (and marginal SD) during optimisation.

    Ported verbatim from addm_hierarchical_james_bug.py. The ELBO alone says
    whether the objective settled, not whether any individual parameter did --
    a flat ELBO is consistent with two parameters trading off against each other
    forever. This records the approximation itself so each parameter's
    trajectory can be inspected after the fact.

    PyMC calls callbacks as ``(approx, loss, i)``. Parameters are read with
    ``get_value()`` rather than ``approx.mean.eval()``: under ``backend="jax"``
    the fit writes ``jax.Array`` into the shared storage, and any later
    default-backend compiled function breaks on it (the exact bug
    ``hssm._vi_compat.coerce_approx_params_to_numpy`` exists to undo). Reading
    storage compiles nothing, and ``np.asarray`` converts a jax.Array fine.

    ``approx.ordering`` maps each variable name to its slice of the flat mean
    vector, which is what makes the saved trace sliceable by parameter name.

    fullrank_advi stores ``[L_tril, mu]`` where Sigma = L @ L.T, so the marginal
    SD of latent i is the norm of row i of L. meanfield stores ``[rho, mu]``
    with SD = softplus(rho).
    """

    def __init__(self, every=10):
        self.every = max(int(every), 1)
        self.iters, self.means, self.sds = [], [], []
        self.names = None

    def __call__(self, approx, loss, i):
        if i % self.every:
            return
        if self.names is None and getattr(approx, "ordering", None):
            self.names = [(nm, int(sl.start), int(sl.stop))
                          for nm, (_, sl, *_rest) in approx.ordering.items()]
        params = {getattr(pp, "name", ""): pp for pp in approx.params}
        if "mu" not in params:
            return
        mu = np.asarray(params["mu"].get_value()).copy()
        self.iters.append(int(i))
        self.means.append(mu)

        n = mu.size
        if "L_tril" in params:                      # fullrank: Sigma = L L^T
            tril = np.asarray(params["L_tril"].get_value())
            L = np.zeros((n, n), dtype=float)
            L[np.tril_indices(n)] = tril
            self.sds.append(np.sqrt((L ** 2).sum(axis=1)))
        elif "rho" in params:                       # meanfield: sd = softplus
            rho = np.asarray(params["rho"].get_value())
            self.sds.append(np.log1p(np.exp(-np.abs(rho))) + np.maximum(rho, 0))

    def save(self, path):
        """Write the trace, with the name->slice map needed to read it back."""
        if not self.means:
            return None
        names = self.names or []
        np.savez_compressed(
            path,
            iters=np.asarray(self.iters, dtype=int),
            means=np.stack(self.means),
            sds=np.stack(self.sds) if self.sds else np.zeros((0, 0)),
            var_names=np.array([n for n, _, _ in names], dtype=object),
            var_start=np.array([s for _, s, _ in names], dtype=int),
            var_stop=np.array([e for _, _, e in names], dtype=int),
            allow_pickle=True,
        )
        return path


def build_initvals(include, regressed, centered):
    """Starting values for VI, read straight off the priors we just built.

    The PRIORS block is already stated as "where we believe this parameter is",
    so the prior centre is the natural start point -- no separate INIT_GUESS
    table like the aDDM script needs, because there the initvals deliberately
    avoid the generative truth and here the priors already do (their centres are
    generic, not copied from TRUE).
    """
    def arg(prior, name):
        """Read one argument off a prior spec.

        Has to accept both shapes: `build_include` writes plain dicts, but by
        the time the model is constructed HSSM has converted every entry to a
        `bmb.Prior`, whose arguments live in `.args` rather than by subscript.
        Nested hyperpriors get the same treatment recursively.
        """
        if isinstance(prior, dict):
            return prior[name]
        return getattr(prior, "args", {})[name]

    initvals = {}
    for spec in include:
        p = spec["name"]
        if p not in regressed or "prior" not in spec:
            continue
        pr = spec["prior"]
        if not isinstance(pr, dict):      # scalar-pinned parameter, nothing to set
            continue
        for term, prior in pr.items():
            if "|" in term:                          # group term
                if centered:                         # mu hyperprior carries it
                    initvals[f"{p}_{term}_mu"] = arg(arg(prior, "mu"), "mu")
                initvals[f"{p}_{term}_sigma"] = arg(arg(prior, "sigma"), "sigma")
            elif term == "Intercept":
                initvals[f"{p}_Intercept"] = arg(prior, "mu")
            else:                                    # common slope, e.g. vdiff
                initvals[f"{p}_{term}"] = arg(prior, "mu")
    return {k: float(v) for k, v in initvals.items()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lik", default="analytical",
                   choices=["analytical", "approx_differentiable"])
    p.add_argument("--t-mode", default="hierarchical",
                   choices=["hierarchical", "fixed"],
                   help="'fixed' pins t at its generative value for every "
                        "subject and does not sample it, isolating whether t is "
                        "what makes the LAN arm hard to sample")
    p.add_argument("--centered", action="store_true",
                   help="centred random effects: p_j ~ Normal(mu, sigma) sampled "
                        "directly, with the group mean carried by the random "
                        "term's own mu. Switches the formulas from `1 + (1|g)` to "
                        "`0 + (1|g)` (and v to `0 + (1 + vdiff|g)`), since a "
                        "common intercept alongside an estimated group mean would "
                        "be additively unidentified")
    p.add_argument("--bounds", default="native", choices=["native", "lan"],
                   help="'native' uses each likelihood's own bounds; 'lan' forces "
                        "the LAN training box on both, so the parameterisation "
                        "(which bounds determine) is held fixed across arms")
    p.add_argument("--links", default="default", choices=["default", "log_logit"],
                   help="'default' leaves HSSM's identity link on every parameter; "
                        "'log_logit' asks HSSM to pick a link from each parameter's "
                        "bounds (identity if both infinite, log for (0,inf), "
                        "gen_logit when both are finite)")
    p.add_argument("--data", default="simulated", choices=["simulated", "real"],
                   help="'real' reads ddm_ladder_real_data.csv (the exact 8,427 "
                        "trials behind the table above) from the working directory")
    p.add_argument("--subjects", type=int, default=44)
    p.add_argument("--trials", type=int, default=192,
                   help="trials per subject (median of the real dataset)")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--max-tree-depth", type=int, default=10)
    p.add_argument("--target-accept", type=float, default=0.8)
    p.add_argument("--inference", default="nuts",
                   choices=["nuts", "advi", "fullrank_advi"],
                   help="'nuts' (default) runs the numpyro NUTS sampler. "
                        "'advi'/'fullrank_advi' run PyMC variational inference "
                        "via HSSM's model.vi(). fullrank_advi models the full "
                        "posterior covariance rather than assuming independence, "
                        "which is the variant that stands a chance on a "
                        "hierarchical model where the group SD and the "
                        "per-subject values are strongly correlated.")
    p.add_argument("--vi-niter", type=int, default=50000,
                   help="VI optimisation steps (default 50000, matching HSSM's "
                        "own fullrank_advi tutorial)")
    p.add_argument("--vi-draws", type=int, default=1000,
                   help="draws taken from the fitted approximation (default 1000)")
    p.add_argument("--vi-lr", type=float, default=0.005,
                   help="adagrad_window learning rate (default 0.005). ADVI keeps "
                        "the LAST iterate, not the best, so a rate that lets the "
                        "optimiser step back out of its minimum puts that drift "
                        "straight into the posterior.")
    p.add_argument("--vi-start-sigma", type=float, default=0.01,
                   help="initial ADVI std-dev around each start value "
                        "(default 0.01; small = trust the start point)")
    p.add_argument("--vi-track", action="store_true",
                   help="record the variational mean/SD during optimisation and "
                        "save it to <tag>_vitrack.npz, so per-parameter "
                        "convergence can be plotted instead of just the ELBO")
    p.add_argument("--vi-track-every", type=int, default=10,
                   help="record every Nth iteration (default 10). 44 subjects is "
                        "~230 latent dims, so every-10 over 50k iterations is "
                        "~9 MB rather than ~92 MB")
    p.add_argument("--zscore-vdiff", action="store_true",
                   help="z-score the vdiff regressor over the pooled data before "
                        "fitting. vdiff is integers in [-3, 3]; dividing by its SD "
                        "puts the regressor on the same scale as the intercept, "
                        "which is the standard conditioning fix for a correlated "
                        "intercept/slope pair in a hierarchical regression.")
    p.add_argument("--zscore-prior", default="rescale",
                   choices=["rescale", "keep"],
                   help="what to do with the slope prior when --zscore-vdiff is "
                        "on. 'rescale' (default) multiplies V_SLOPE by the SD of "
                        "vdiff, so the prior encodes the SAME belief about the "
                        "drift range as the un-z-scored run. 'keep' leaves the "
                        "prior at its raw-units value, which -- since sigma > 1 -- "
                        "makes it TIGHTER in drift terms. Use 'keep' only to test "
                        "the prior's effect deliberately.")
    p.add_argument("--no-ndt-guard", action="store_true",
                   help="DIAGNOSTIC: disable HSSM's ensure_positive_ndt, which "
                        "overwrites the log-likelihood with LOGP_LB=-66.1 wherever "
                        "rt - t <= 1e-15. Replaces it with the identity BEFORE the "
                        "model is built, so the discontinuity never enters the "
                        "pytensor graph; nothing else changes. The analytical "
                        "likelihood floors itself internally and is unaffected, so "
                        "this is only meaningful on the LAN arm. Expect t to drift "
                        "upward -- see the module docstring.")
    p.add_argument("--idata-subdir", default="",
                   help="subfolder of --idata-dir to write the netcdf into "
                        "(created if absent). Defaults to 'noguard' when "
                        "--no-ndt-guard is set, so guarded and unguarded runs "
                        "cannot overwrite each other.")
    p.add_argument("--save-idata", dest="save_idata", action="store_true",
                   default=True,
                   help="write the full InferenceData to --idata-dir (default on). "
                        "Needed for anything that works with posterior draws "
                        "rather than summaries -- e.g. evaluating a second "
                        "likelihood's gradient at this posterior's draws.")
    p.add_argument("--no-save-idata", dest="save_idata", action="store_false")
    p.add_argument("--idata-dir",
                   default="/users/azhan378/scratch/ddm_hier_idata",
                   help="where to write the netcdf (created if absent). Kept off "
                        "the repo because hierarchical idata runs to hundreds of MB.")
    return p.parse_args()


def simulate(n_subj, n_trials, rng):
    """Hierarchical DDM: per-subject params from the group distribution, then
    (rt, response) from the ssms `ddm` simulator."""
    from ssms.basic_simulators.simulator import simulator

    # Per-subject deviations, non-centred exactly as bambi/HSSM parameterise them.
    v_j = TRUE["v_intercept"] + rng.normal(0, TRUE["sd_v"], n_subj)
    a_j = TRUE["a"] + rng.normal(0, TRUE["sd_a"], n_subj)
    z_j = TRUE["z"] + rng.normal(0, TRUE["sd_z"], n_subj)
    t_j = TRUE["t"] + rng.normal(0, TRUE["sd_t"], n_subj)
    # Random slope on vdiff, so the generative process matches the fitted
    # `(1 + vdiff|participant_id)` structure rather than being a special case
    # of it with the slope variance pinned at zero.
    vs_j = TRUE["v_vdiff"] + rng.normal(0, TRUE["sd_v_vdiff"], n_subj)

    rows = []
    for j in range(n_subj):
        # Value difference on the real experiment's scale: integers in [-3, 3].
        vdiff = rng.integers(-3, 4, n_trials).astype(float)
        v = v_j[j] + vs_j[j] * vdiff
        theta = dict(v=v, a=np.full(n_trials, a_j[j]),
                     z=np.full(n_trials, z_j[j]), t=np.full(n_trials, t_j[j]))
        out = simulator(theta, model="ddm", n_samples=1, random_state=SEED + j)
        rows.append(pd.DataFrame(dict(
            rt=np.squeeze(out["rts"]).astype(float),
            response=np.squeeze(out["choices"]).astype(int),
            vdiff=vdiff,
            participant_id=str(j + 1),
        )))
    df = pd.concat(rows, ignore_index=True)
    # Same RT filter as the real analysis (~mean+3SD), so the simulated and real
    # datasets have comparable support. Without it the simulated tail runs to
    # ~11 s where the real data is capped at 6 s.
    df = df[(df.rt > 0.2) & (df.rt < 6.0)].reset_index(drop=True)
    return df, dict(v=v_j, v_vdiff=vs_j, a=a_j, z=z_j, t=t_j)


def main():
    args = parse_args()

    import jax
    print("JAX devices:", jax.devices(), "| backend:", jax.default_backend())

    import hssm
    print("HSSM", hssm.__version__)

    # region ===== optional: disable the non-decision-time guard =====
    # `ensure_positive_ndt` is defined in distribution_utils/dist.py and called
    # from `make_distribution` in that SAME module, as a bare name. Python
    # resolves a bare global at call time against the module's namespace, so
    # rebinding the attribute here -- before hssm.HSSM() builds the graph below
    # -- is what the logp construction will pick up. Nothing else is touched:
    # the likelihood, the priors, the p_outlier mixture that runs immediately
    # after the guard, all stay exactly as they were.
    if args.no_ndt_guard:
        import hssm.distribution_utils.dist as _dist
        _original_guard = _dist.ensure_positive_ndt
        _dist.ensure_positive_ndt = (
            lambda data, logp, list_params, dist_params: logp
        )
        # Confirm the swap is live in the namespace dist.py actually reads, and
        # that the replacement really is a pass-through, rather than trusting
        # that the assignment above did what it looks like it did.
        assert _dist.ensure_positive_ndt is not _original_guard
        _probe = np.array([1.0, 2.0, 3.0])
        assert _dist.ensure_positive_ndt(None, _probe, [], []) is _probe
        print("\n  [--no-ndt-guard] ensure_positive_ndt -> identity "
              "(patched before model construction)")
        print("  DIAGNOSTIC ONLY: t is no longer forbidden from exceeding a "
              "trial's RT and is expected to drift upward.")
    # endregion

    # "constant for every subject" has to hold in the GENERATIVE process too,
    # not just in the fitted model. Without this the simulator still draws
    # t_j ~ N(0.7139, 0.1274) per subject while the model pins t at 0.7139 for
    # everyone, so `--t-mode fixed` would confound "t's hierarchy removed" with
    # "model misspecified", and the unmodelled spread would leak into a and z.
    # Applied before simulate() reads it. (No effect under --data real, where
    # the generative t is whatever the subjects actually had.)
    if args.t_mode == "fixed":
        TRUE["sd_t"] = 0.0

    if args.data == "real":
        df = pd.read_csv("ddm_ladder_real_data.csv")
        df["participant_id"] = df.participant_id.astype(int).astype(str)
        print(f"\nreal data: {len(df)} trials, {df.participant_id.nunique()} subjects "
              f"| rt median {df.rt.median():.2f}s, P(resp=1) {(df.response==1).mean():.3f}")
    else:
        rng = np.random.default_rng(SEED)
        df, _ = simulate(args.subjects, args.trials, rng)
        print(f"\nsimulated {len(df)} trials, {args.subjects} subjects "
              f"| rt median {df.rt.median():.2f}s, P(resp=1) {(df.response==1).mean():.3f}")
        # Real data for reference: 8,427 trials, rt median 1.567, P(resp=1) 0.512.

    # region ===== optional z-scoring of the vdiff regressor =====
    # v = v0 + v1 * vdiff.  Substituting vdiff = sigma * z + mu gives
    #     v = (v0 + v1 * mu)  +  (v1 * sigma) * z,
    # so the slope in z-units is sigma times the slope in raw units, and the
    # intercept picks up v1 * mu. `slope_scale` carries the first of those into
    # the prior; the second is reported so a non-zero mean cannot pass silently.
    slope_scale = 1.0
    vdiff_mu = vdiff_sd = None
    if args.zscore_vdiff:
        vdiff_mu = float(df.vdiff.mean())
        vdiff_sd = float(df.vdiff.std(ddof=0))
        if not np.isfinite(vdiff_sd) or vdiff_sd <= 0:
            raise SystemExit("vdiff has zero variance; cannot z-score it.")
        df = df.copy()
        df["vdiff"] = (df.vdiff - vdiff_mu) / vdiff_sd
        if args.zscore_prior == "rescale":
            slope_scale = vdiff_sd
        print(f"\n  vdiff z-scored: mu {vdiff_mu:+.4f}, sd {vdiff_sd:.4f} "
              f"-> range [{df.vdiff.min():+.3f}, {df.vdiff.max():+.3f}]")
        print(f"    slope prior: {args.zscore_prior}"
              + (f" (V_SLOPE x {slope_scale:.4f}, so the encoded belief about "
                 f"the drift range is unchanged)" if args.zscore_prior == "rescale"
                 else " (left at raw-units value -- TIGHTER in drift terms)"))
        if abs(vdiff_mu) > 1e-6:
            print(f"    [note] mean is {vdiff_mu:+.4f}, not 0, so the intercept "
                  f"absorbs v1 * mu = slope x {vdiff_mu:+.4f}; the intercept prior "
                  f"is NOT adjusted for this.")
    # endregion

    # Identical model both ways; only `loglik_kind` differs between runs. The
    # priors are stated in parameter units and transferred onto each arm's link,
    # so "identical" holds in the sense that matters -- the same belief about v,
    # a, z and t -- rather than merely the same numbers on two different scales.
    include, bounds = build_include(args.lik, args.links, args.bounds, args.t_mode,
                                    args.centered, slope_scale=slope_scale)
    link_kw = {"link_settings": "log_logit"} if args.links == "log_logit" else {}

    print(f"\nbuilding ddm with loglik_kind={args.lik!r}, links={args.links!r}, "
          f"bounds={args.bounds!r}, t={args.t_mode!r}, "
          f"parameterisation={'centred' if args.centered else 'non-centred'!r}")
    if args.t_mode == "fixed":
        print(f"  t = {TRUE['t']:g} (constant for every subject, not sampled)")
    for spec in include:
        # the fixed-t entry carries a scalar prior and no formula
        print("  " + (spec["formula"] if "formula" in spec
                      else f"{spec['name']} = {spec['prior']} (fixed)"))
    def group_mean_prior(pr, term):
        """(mu, sd) of the group mean, wherever this parameterisation put it.

        Non-centred: on the common term ('Intercept' / 'vdiff').
        Centred: on the random term's own `mu` hyperprior.
        """
        if args.centered:
            m = pr[term]["mu"]
            return m["mu"], m["sigma"]
        return pr[term]["mu"], pr[term]["sigma"]

    print("\n  priors (parameter units -> link scale):")
    print(f"    {'param':6}{'bounds':>18}{'link':>11}{'centre':>10}{'sd':>9}"
          f"{'group sd':>10}")
    regressed = ("v", "a", "z") if args.t_mode == "fixed" else ("v", "a", "z", "t")
    for p in regressed:
        b, spec = bounds[p], PRIORS[p]
        kind = link_of(b, args.links)
        pr = include[list(regressed).index(p)]["prior"]
        mu, sd = group_mean_prior(pr, "1|participant_id" if args.centered
                                  else "Intercept")
        print(f"    {p:6}{str(tuple(b)):>18}{kind:>11}{mu:>10.3f}{sd:>9.3f}"
              f"{pr['1|participant_id']['sigma']['sigma']:>10.3f}"
              f"   (= {spec['loc']:g} +/- {spec['sd']:g} in parameter units)")
    vpr = include[0]["prior"]
    vmu, vsd = group_mean_prior(vpr, "vdiff|participant_id" if args.centered
                                else "vdiff")
    print(f"    {'v:vdiff':6}{'':>18}{'':>11}{vmu:>10.3f}{vsd:>9.3f}"
          f"{vpr['vdiff|participant_id']['sigma']['sigma']:>10.3f}"
          f"   (slope; delta-method transfer)")

    if args.bounds == "lan":
        # Same box the LAN declares, so `set_loglogit_link` picks gen_logit on
        # every parameter for this arm as well.
        link_kw["model_config"] = {"bounds": dict(LAN_BOUNDS)}
    model = hssm.HSSM(data=df[["rt", "response", "vdiff", "participant_id"]],
                      model="ddm", loglik_kind=args.lik, include=include, **link_kw)

    import time
    t0 = time.time()
    tracer = None
    if args.inference == "nuts":
        # NOTE: `chain_method` is NOT passed. pymc 6.2's `_sample_external_nuts`
        # calls `sample_jax_nuts` with a fixed argument list that omits it, so any
        # value given here is silently discarded and numpyro always gets its own
        # default of "parallel" -- one chain per visible device via pmap. Request
        # as many GPUs as chains (see run_ddm_mre_hierarchical_james.sh); with
        # fewer, numpyro warns and falls back to running them sequentially.
        idata = model.sample(sampler="numpyro", draws=args.draws, tune=args.tune,
                             chains=args.chains, cores=1, random_seed=SEED,
                             target_accept=args.target_accept,
                             nuts={"max_tree_depth": args.max_tree_depth})
    else:
        import pymc as pm
        # VI's `start` lives in PyMC's unconstrained space, so the initvals have
        # to be remapped (group SDs get logged); see vi_start_point.
        raw_initvals = build_initvals(include, regressed, args.centered)
        named = set(model.pymc_model.named_vars)
        dropped = sorted(k for k in raw_initvals
                         if k not in named and f"{k}_log__" not in named)
        raw_initvals = {k: v for k, v in raw_initvals.items() if k not in dropped}
        if dropped:
            print(f"\n  [warn] initvals not in the model, ignored: {dropped}")
        start, mapped, skipped = vi_start_point(model, raw_initvals)
        # PyMC accepts `start_sigma` only for meanfield advi; fullrank_advi
        # raises NotImplementedError, so it is passed conditionally.
        vi_kw = {}
        if args.inference == "advi":
            vi_kw["start_sigma"] = vi_start_sigma(start, args.vi_start_sigma)
        print(f"\n  VI start: {len(mapped)} variables set from initvals"
              + (f", {len(skipped)} unmapped: {skipped}" if skipped else "")
              + (f" | start_sigma {args.vi_start_sigma:g}" if vi_kw
                 else " | start_sigma n/a for fullrank"))
        print(f"  {args.inference} | niter {args.vi_niter} | draws "
              f"{args.vi_draws} | adagrad_window(lr={args.vi_lr:g})")
        if args.vi_track:
            # pm.callbacks is not exposed at the PyMC root in this version.
            from pymc.variational.callbacks import Tracker
            tracer = VITrace(every=args.vi_track_every)
            # Tracker tries fn() first and falls back to fn(approx, hist, i);
            # VITrace takes the 3-arg form, so it lands on the fallback.
            vi_kw["callbacks"] = [Tracker(trace=tracer)]
            print(f"  tracking variational mean/SD every "
                  f"{args.vi_track_every} iterations")
        # ignore_mcmc_start_point_defaults: otherwise HSSM overwrites `start`
        # with its own MCMC initval defaults and our start point is discarded.
        idata = model.vi(method=args.inference, niter=args.vi_niter,
                         draws=args.vi_draws, random_seed=SEED,
                         start=start,
                         ignore_mcmc_start_point_defaults=True,
                         obj_optimizer=pm.adagrad_window(learning_rate=args.vi_lr),
                         **vi_kw)
    dt = time.time() - t0

    import arviz as az
    # arviz rounds summary values to STRINGS by default; round_to="none" keeps
    # them numeric so the comparisons below are real comparisons.
    summ = az.summary(idata, round_to="none")
    # arviz 1.x names the interval columns after the (ci_kind, ci_prob) pair, so
    # the default 89% ETI and a 94% HDI land in disjoint columns and can sit in
    # the same table. Keeping both means the CSVs written before this change
    # stay directly comparable on `eti89_*`.
    hdi = az.summary(idata, round_to="none", ci_kind="hdi", ci_prob=0.94)
    for i, col in enumerate(("hdi94_lb", "hdi94_ub")):
        summ.insert(summ.columns.get_loc("eti89_ub") + 1 + i, col, hdi[col])

    tag = (f"{args.data}_{args.lik}_{args.links}_{args.bounds}bounds"
           f"_t{args.t_mode}_{'centred' if args.centered else 'noncentred'}"
           # z-scoring changes what the slope MEANS, so it must not share a
           # filename with a raw-units run.
           + (f"_vdiffz{'' if args.zscore_prior == 'rescale' else 'keepprior'}"
              if args.zscore_vdiff else "")
           # the guard changes the target distribution, so it must not share a
           # filename with a guarded run
           + ("_noguard" if args.no_ndt_guard else "")
           # VI results depend strongly on the learning rate, so it belongs in
           # the filename -- otherwise an lr sweep silently overwrites itself.
           + ("" if args.inference == "nuts"
              else f"_{args.inference}_lr{args.vi_lr:g}"))

    print("\n" + "=" * 70)
    print(f"{args.inference.upper()} DIAGNOSTICS  --  loglik_kind={args.lik}, "
          f"links={args.links}")
    print("=" * 70)
    if args.inference == "nuts":
        print(f"  wall clock          {dt/60:.1f} min ({dt/(args.draws+args.tune):.3f} s/iteration)")
        print(f"  max r_hat           {summ.r_hat.max():.3f}")
        print(f"  min ess_bulk        {summ.ess_bulk.min():.0f}")
        print(f"  min ess_tail        {summ.ess_tail.min():.0f}")
        ss = idata.sample_stats
        if "diverging" in ss:
            print(f"  divergences         {int(ss.diverging.values.sum())} / {ss.diverging.size}")
        if "tree_depth" in ss:
            td = ss.tree_depth.values
            print(f"  tree depth          median {np.median(td):.0f} / cap {args.max_tree_depth}, "
                  f"saturated {float((td >= args.max_tree_depth).mean()):.1%} of iterations")
        if "step_size" in ss:
            print(f"  final step size     {float(ss.step_size.values[:, -1].mean()):.3e}")
        if "acceptance_rate" in ss:
            print(f"  acceptance          {float(ss.acceptance_rate.values.mean()):.3f}")
    else:
        print(f"  wall clock          {dt/60:.1f} min "
              f"({dt/max(args.vi_niter, 1)*1000:.2f} ms/iteration)")
        # r_hat/ESS are meaningless here: VI draws are iid from the fitted
        # approximation, so they measure the sampler that isn't running. The
        # ELBO trace is the thing that says whether the fit converged.
        loss = np.asarray(getattr(model.vi_approx, "hist", []), dtype=float)
        if loss.size:
            nn = max(len(loss) // 100, 50)
            tail, mid = loss[-nn:], loss[len(loss)//2: len(loss)//2 + nn]
            drift = float(np.median(mid) - np.median(tail))
            print(f"  ELBO loss           final {loss[-1]:.1f} | last-{nn} median "
                  f"{np.median(tail):.1f} | last-{nn} sd {np.std(tail):.1f}")
            print(f"  still improving     midpoint median {np.median(mid):.1f} -> "
                  f"tail {np.median(tail):.1f}  (drop {drift:.1f}; "
                  f"{'PLATEAUED' if abs(drift) < np.std(tail) else 'STILL MOVING'})")
            if not np.isfinite(loss).all():
                print(f"  [warn] {int((~np.isfinite(loss)).sum())} non-finite "
                      f"ELBO values -- the optimiser left the network's valid region")
            np.save(f"mre_summary_{tag}_elbo.npy", loss)
            print(f"  ELBO trace saved    mre_summary_{tag}_elbo.npy")
        else:
            print("  [warn] no ELBO history on vi_approx")
        if tracer is not None:
            pth = tracer.save(f"mre_summary_{tag}_vitrack.npz")
            if pth:
                print(f"  param trace         {len(tracer.means)} snapshots x "
                      f"{tracer.means[0].size} latent dims -> {pth}")
            else:
                print("  [warn] tracking was on but nothing was recorded")

    # Where the group mean ended up depends on the parameterisation, and the
    # posterior for it is on the LINK scale while TRUE is in parameter units --
    # so back-transform before comparing, or `a = 1.11` reads as `+0.08`.
    def gm_name(p, term):
        if args.centered:
            return f"{p}_{term}|participant_id_mu"
        return f"{p}_Intercept" if term == "1" else f"{p}_{term}"

    rows = [(gm_name("v", "1"), "v", "v_intercept", "loc"),
            (gm_name("v", "vdiff"), "v", "v_vdiff", "slope"),
            (gm_name("a", "1"), "a", "a", "loc"),
            (gm_name("z", "1"), "z", "z", "loc"),
            (gm_name("t", "1"), "t", "t", "loc")]

    header = ("recovery (link -> natural vs truth)" if args.data == "simulated"
              else "group-level posterior means (link -> natural)")
    if args.inference != "nuts":
        header += "  [VI approximation, not MCMC]"
    print(f"\n  {header}:")
    for name, p, key, how in rows:
        if name not in summ.index:
            continue
        b, kind = bounds[p], link_of(bounds[p], args.links)
        link_val = float(summ.loc[name, "mean"])
        if how == "loc":
            nat = link_to_loc(link_val, b, kind)
        else:
            # a slope crosses back by the delta method, evaluated at the
            # posterior mean of this parameter's own location
            at = link_to_loc(float(summ.loc[gm_name(p, "1"), "mean"]), b, kind)
            nat = link_val * dparam_dlink(at, b, kind)
        # On z-scored vdiff the recovered slope is in z-units, so the
        # generative value has to be pushed onto the same scale before it can
        # be compared: v1_z = v1_raw * sigma.
        truth_val = TRUE[key] * (vdiff_sd if (how == "slope" and args.zscore_vdiff
                                              and vdiff_sd) else 1.0)
        truth = f"{truth_val:+.4f}" if args.data == "simulated" else ""
        print(f"    {name:<26}{link_val:>10.4f}{nat:>12.4f}   {truth}")

    summ.to_csv(f"mre_summary_{tag}.csv")
    print(f"\nwrote mre_summary_{tag}.csv")

    if args.save_idata:
        subdir = args.idata_subdir or ("noguard" if args.no_ndt_guard else "")
        out_dir = os.path.join(args.idata_dir, subdir) if subdir else args.idata_dir
        os.makedirs(out_dir, exist_ok=True)
        out_nc = os.path.join(out_dir, f"idata_{tag}.nc")
        idata.to_netcdf(out_nc)
        print(f"wrote {out_nc} "
              f"({os.path.getsize(out_nc) / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()