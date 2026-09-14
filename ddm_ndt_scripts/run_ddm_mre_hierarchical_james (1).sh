#!/bin/bash
#SBATCH --job-name=ddm_mre
#SBATCH --account=carney-frankmj-condo2
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ddm_mre_%j.out
#SBATCH --error=logs/ddm_mre_%j.err
#
# Hierarchical DDM MRE: analytical vs approx_differentiable (lnccbrown/HSSM#1085).
#
# 2 GPUs, 2 chains, one chain per GPU running genuinely in parallel, 6 h cap.
#
# WHY 2 CHAINS: numpyro's chain_method defaults to "parallel", which pmaps one
# chain onto each visible device. pymc 6.2 drops any chain_method passed through
# HSSM, so this default is the only behaviour available. With more chains than
# GPUs numpyro warns and runs them SEQUENTIALLY, which would put the ~349 min
# approx_differentiable arm well past the 6 h wall. Two chains still give r_hat
# and ESS, just with less power than four -- raise both together if you want 4:
#   sbatch --gres=gpu:4 --export=ALL,CHAINS=4 run_ddm_mre_hierarchical_james.sh
#
# Submit the three arms of the comparison:
#   sbatch --export=ALL,LIK=analytical,LINKS=default            run_...james.sh
#   sbatch --export=ALL,LIK=approx_differentiable,LINKS=default run_...james.sh
#   sbatch --export=ALL,LIK=approx_differentiable,LINKS=log_logit run_...james.sh
#
# The non-decision-time guard as a factor (see ddm_data_comparison.ipynb S11):
#   sbatch --gres=gpu:4 --export=ALL,LIK=approx_differentiable,LINKS=log_logit,\
#          CHAINS=4,DATA=real,NDT_GUARD=off run_ddm_mre_hierarchical_james.sh
#
# Other overrides: DATA (real|simulated), CENTERED (0|1), DRAWS, TUNE,
# MAX_TREE_DEPTH, TARGET_ACCEPT, SUBJECTS, TRIALS, NDT_GUARD (on|off),
# IDATA_SUBDIR.

set -euo pipefail

REPO_DIR="/users/azhan378/data/azhang/HSSMSpine/repos/HSSM"
cd "$REPO_DIR"          # the script reads ddm_ladder_real_data.csv from the cwd
mkdir -p logs

# --- Tunables ----------------------------------------------------------------
LIK="${LIK:-approx_differentiable}"
LINKS="${LINKS:-log_logit}"
BOUNDS="${BOUNDS:-native}"   # native | lan (lan = force the LAN box on both arms)
T_MODE="${T_MODE:-hierarchical}"  # hierarchical | fixed (fixed = t not sampled)
# 1 = centred random effects. Also switches the formulas to `0 + (1|g)`, since
# the centred group term carries its own mean and a common intercept alongside
# it would be additively unidentified.
CENTERED="${CENTERED:-0}"
DATA="${DATA:-real}"
CHAINS="${CHAINS:-2}"           # keep equal to the GPU count
DRAWS="${DRAWS:-1000}"
TUNE="${TUNE:-1000}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-10}"
TARGET_ACCEPT="${TARGET_ACCEPT:-0.8}"
SUBJECTS="${SUBJECTS:-44}"
TRIALS="${TRIALS:-192}"
# nuts | advi | fullrank_advi. VI ignores CHAINS entirely (it is an optimiser,
# not a sampler), so a VI job only needs one GPU no matter what CHAINS says.
INFERENCE="${INFERENCE:-nuts}"
ZSCORE="${ZSCORE:-0}"                 # 1 = z-score the vdiff regressor
ZSCORE_PRIOR="${ZSCORE_PRIOR:-rescale}"   # rescale | keep
IDATA_DIR="${IDATA_DIR:-/users/azhan378/scratch/ddm_hier_idata}"
VI_NITER="${VI_NITER:-50000}"
VI_DRAWS="${VI_DRAWS:-1000}"
VI_LR="${VI_LR:-0.005}"
VI_TRACK="${VI_TRACK:-1}"
VI_TRACK_EVERY="${VI_TRACK_EVERY:-10}"
# on | off. "off" disables HSSM's ensure_positive_ndt, which overwrites the
# log-likelihood with LOGP_LB=-66.1 wherever rt - t <= 1e-15. That guard is a
# no-op for the analytical likelihood (which has already collapsed below the
# p_outlier lapse floor by then) but a step discontinuity for the LAN, so this
# is the end-to-end test of whether that step is what collapses the step size.
# DIAGNOSTIC ONLY -- with it off, t is expected to drift upward. See the module
# docstring in ddm_mre_hierarchical_james.py.
NDT_GUARD="${NDT_GUARD:-on}"
# subfolder of IDATA_DIR for the netcdf; the script defaults it to "noguard"
# when the guard is off, so guarded and unguarded runs cannot collide.
IDATA_SUBDIR="${IDATA_SUBDIR:-}"

# --- Environment -------------------------------------------------------------
# The repo's own venv: hssm installed editable from this checkout, so the fit
# exercises whatever branch is currently checked out.
source "$REPO_DIR/.venv/bin/activate"

# Multi-GPU numpyro on Oscar deadlocks in an NCCL peer-to-peer collective at the
# end of sampling; disabling P2P is the known mitigation and costs nothing here.
export NCCL_P2P_DISABLE=1
# CUDA stays the default backend, but a CPU device must remain available:
# numpyro's progress-bar io_callback places its inputs there and dies without it.
export JAX_PLATFORMS=cuda,cpu

N_GPU=$(echo "${CUDA_VISIBLE_DEVICES:-}" | awk -F, '{print NF}')
echo "================================================================"
echo "job          ${SLURM_JOB_ID:-<none>} on $(hostname)"
echo "branch       $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO_DIR" rev-parse --short HEAD)"
echo "python       $(which python)"
echo "GPUs         ${CUDA_VISIBLE_DEVICES:-<unset>}  (${N_GPU} visible)"
echo "chains       $CHAINS   <- must equal the GPU count for true parallelism"
echo "arm          lik=$LIK  links=$LINKS  bounds=$BOUNDS  t=$T_MODE  data=$DATA"
echo "ndt guard    $NDT_GUARD$([ "$NDT_GUARD" = "off" ] && echo "   <- DIAGNOSTIC: t may drift upward")"
echo "inference    $INFERENCE$([ "$INFERENCE" = "nuts" ] || echo "  niter=$VI_NITER lr=$VI_LR track=$VI_TRACK")"
echo "vdiff        $([ "$ZSCORE" = "1" ] && echo "z-scored (prior: $ZSCORE_PRIOR)" || echo "raw integers -3..3")"
echo "idata ->     $IDATA_DIR"
echo "param.       $([ "$CENTERED" = "1" ] && echo centred || echo non-centred)"
echo "start        $(date)"
echo "================================================================"
if [ "$INFERENCE" = "nuts" ] && [ "${N_GPU}" -ne "${CHAINS}" ]; then
  echo "[warn] ${CHAINS} chains but ${N_GPU} GPU(s): numpyro will fall back to" \
       "running chains SEQUENTIALLY, which may exceed the 6 h wall." >&2
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
echo

# Absolute interpreter path: srun does not carry the activated venv PATH.
"$REPO_DIR/.venv/bin/python" -u ddm_mre_hierarchical_james.py \
    --lik "$LIK" \
    --links "$LINKS" \
    --bounds "$BOUNDS" \
    --t-mode "$T_MODE" \
    $([ "$CENTERED" = "1" ] && echo --centered) \
    --data "$DATA" \
    --chains "$CHAINS" \
    --draws "$DRAWS" \
    --tune "$TUNE" \
    --max-tree-depth "$MAX_TREE_DEPTH" \
    --target-accept "$TARGET_ACCEPT" \
    --subjects "$SUBJECTS" \
    --trials "$TRIALS" \
    --inference "$INFERENCE" \
    --vi-niter "$VI_NITER" \
    --vi-draws "$VI_DRAWS" \
    --vi-lr "$VI_LR" \
    --vi-track-every "$VI_TRACK_EVERY" \
    $([ "$VI_TRACK" = "1" ] && echo --vi-track) \
    --idata-dir "$IDATA_DIR" \
    --zscore-prior "$ZSCORE_PRIOR" \
    $([ "$NDT_GUARD" = "off" ] && echo --no-ndt-guard) \
    $([ -n "$IDATA_SUBDIR" ] && echo --idata-subdir "$IDATA_SUBDIR") \
    $([ "$ZSCORE" = "1" ] && echo --zscore-vdiff)

echo
echo "finished $(date)"
ls -la "$REPO_DIR"/mre_summary_*.csv 2>/dev/null || echo "  (no summary CSV written)"
