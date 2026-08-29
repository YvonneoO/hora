#!/usr/bin/env bash
#SBATCH --job-name=hora_ckpt_sweep
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/hora_ckpt_sweep_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/hora_ckpt_sweep_%j.err
#
# Reconstruct a success-rate-vs-training-step curve by re-evaluating every
# periodic checkpoint from a completed run, using the fixed/correctly-masked
# success computation -- the live training-time TB 'success' curve is
# unusable (reset_buf indexing bug), but the checkpoints and training itself
# were never affected by it, so this recovers the curve without retraining.
#
# Required (--export=ALL,...): CKPT_DIR (e.g. .../v1_ponly_seed0/stage1_nn),
#   OUT_CSV, TAG (label for the output filename)
# Optional: PRIV_INFO (default False), COARSE_TACTILE (default False),
#   TARGET_EPISODES (default 200)

set -euo pipefail

if [[ -z "${CKPT_DIR:-}" || -z "${OUT_CSV:-}" ]]; then
  echo "CKPT_DIR and OUT_CSV must both be set" >&2
  exit 2
fi

YQQ=/scratch/project/prj-02-phai-lab/yqq
cd "${YQQ}"
source "${YQQ}/env.sh"
mkdir -p "${YQQ}/logs" "${YQQ}/tmp" "$(dirname "${OUT_CSV}")"

export TMPDIR="${YQQ}/tmp"
export HF_HOME="${YQQ}/.cache/huggingface"
export TORCH_HOME="${YQQ}/.cache/torch"
export TORCH_EXTENSIONS_DIR="${YQQ}/.cache/torch_extensions"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1

export ISAACGYM_ENV="${YQQ}/envs/bidexhands_isaacgym_py38"
export ISAACGYM_ROOT="${YQQ}/isaacgym"
export HORA_ROOT="${YQQ}/hora"

export PATH="${ISAACGYM_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${ISAACGYM_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ISAACGYM_ROOT}/python:${PYTHONPATH:-}"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "OUT_CSV=${OUT_CSV}"

cd "${HORA_ROOT}"
"${ISAACGYM_ENV}/bin/python" sweep_checkpoint_eval.py "${CKPT_DIR}" "${OUT_CSV}" \
  --priv_info "${PRIV_INFO:-False}" \
  --coarse_tactile "${COARSE_TACTILE:-False}" \
  --target_episodes "${TARGET_EPISODES:-200}"

echo "---SWEEP_DONE---"
