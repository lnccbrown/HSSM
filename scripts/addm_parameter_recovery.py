"""Off-CI parameter-recovery check for the attentional DDM (aDDM).

Ties the whole stack together: simulate a dataset with the ported ssm-simulators
aDDM engine (CE-1), fit it with ``hssm.aDDM`` (the vendored JAX likelihood), and
verify the posterior recovers the ground-truth parameters.

Run (needs the aDDM ssm-simulators build with ``cssm.addm`` installed)::

    uv run --no-sync python scripts/addm_parameter_recovery.py
    uv run --no-sync python scripts/addm_parameter_recovery.py --n-trials 2000

Not part of the CI suite (it samples). The fast, no-sampling sim<->likelihood
cross-check lives in ``tests/addm/test_addm_ppc.py::test_addm_sim_likelihood_recovery``.

Note on fixations: a simulated decision can land before a pre-generated saccade
would occur, so the *observed* fixations are truncated at the response time (real
aDDM data is naturally truncated this way; the likelihood requires ``rt >=
sacc[d-1]``). The full "keep saccading past the last observed fixation" refinement
is a follow-up (Andrew's C6).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# Ground-truth parameters (t = 0 so rt is pure decision time; sigma fixed at 1).
TRUTH = dict(eta=0.4, kappa=1.0, a=1.5, b=0.2, x0=0.0, t=0.0)
SIGMA = 1.0
MAX_D = 8


def simulate_addm_dataset(n_trials: int, seed: int = 0) -> pd.DataFrame:
    """Simulate an aDDM dataset with observed fixations truncated at the response."""
    import cssm

    rng = np.random.default_rng(seed)
    r1 = rng.integers(1, 6, n_trials).astype(np.float64)
    r2 = rng.integers(1, 6, n_trials).astype(np.float64)
    flag = rng.integers(0, 2, n_trials).astype(np.int64)
    d = rng.integers(2, MAX_D + 1, n_trials).astype(np.int32)
    sacc = np.zeros((n_trials, MAX_D))
    for i in range(n_trials):
        sacc[i, 1 : d[i]] = np.sort(rng.uniform(0.1, 1.2, d[i] - 1))

    col = lambda v: np.full(n_trials, v, dtype=np.float64)  # noqa: E731
    out = cssm.addm(
        col(TRUTH["eta"]),
        col(TRUTH["kappa"]),
        col(TRUTH["a"]),
        col(TRUTH["b"]),
        col(TRUTH["x0"]),
        col(TRUTH["t"]),
        col(999.0),
        col(SIGMA),
        sigma=col(SIGMA),
        r1=r1,
        r2=r2,
        flag=flag,
        sacc_array=sacc,
        d=d,
        n_samples=1,
        n_trials=n_trials,
        random_state=seed + 1,
        delta_t=0.001,
        max_t=10.0,
    )
    rt = np.asarray(out["rts"]).reshape(-1).astype(np.float64)
    ch = np.asarray(out["choices"]).reshape(-1).astype(np.float64)
    keep = rt != -999.0

    # Truncate observed fixations at rt (only fixations that started before it).
    rows = np.flatnonzero(keep)
    d_obs = np.array(
        [max(int((sacc[i, : d[i]] < rt[i]).sum()), 1) for i in rows], dtype=int
    )
    df = pd.DataFrame(
        {
            "rt": rt[keep],
            "response": ch[keep],
            "r1": r1[keep],
            "r2": r2[keep],
            "flag": flag[keep].astype(int),
            "d": d_obs,
            "sigma": np.full(rows.size, SIGMA),
        }
    )
    df["sacc_array"] = pd.Series([sacc[i].copy() for i in rows], index=df.index)
    return df


def run_recovery(n_trials: int, draws: int, tune: int, seed: int) -> bool:
    """Simulate, fit, and report whether each parameter is recovered within 2 sd."""
    import arviz as az

    import hssm

    hssm.set_floatX("float64")
    df = simulate_addm_dataset(n_trials, seed=seed)
    p_up = (df["response"] == 1).mean()
    print(
        f"Simulated {len(df)} trials (of {n_trials}); "
        f"mean rt {df['rt'].mean():.3f}, P(choice=+1) {p_up:.3f}"
    )

    model = hssm.aDDM(data=df)
    idata = model.sample(
        draws=draws,
        tune=tune,
        chains=2,
        cores=1,
        idata_kwargs={"log_likelihood": False},
    )
    summary = az.summary(idata, var_names=list(TRUTH), kind="stats")

    print("\nParameter recovery (posterior vs truth):")
    print(f"{'param':>6}  {'truth':>7}  {'mean':>8}  {'sd':>7}  {'z':>7}  within2sd")
    ok = True
    for p in TRUTH:
        if p not in summary.index:
            continue
        mean, sd = summary.loc[p, "mean"], summary.loc[p, "sd"]
        z = (mean - TRUTH[p]) / sd if sd > 0 else np.inf
        within = abs(z) <= 2.0
        ok &= within
        print(
            f"{p:>6}  {TRUTH[p]:>7.3f}  {mean:>8.3f}  {sd:>7.3f}  {z:>7.2f}  {within}"
        )
    print("\nRESULT:", "RECOVERED (all within 2 sd)" if ok else "OUT OF TOLERANCE")
    return ok


def main() -> None:
    """CLI entry point: run the recovery check and exit non-zero if out of tolerance."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-trials", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--tune", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ok = run_recovery(args.n_trials, args.draws, args.tune, args.seed)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
