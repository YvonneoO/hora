#!/usr/bin/env bash
#SBATCH --job-name=hora_v4_inspect
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/hora_v4_inspect_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/hora_v4_inspect_%j.err
#
# Qualitative rollout of the v4_spread privileged teacher (priv_info=True) --
# a few episodes WITH the QA side-by-side video enabled, to visually inspect
# what the actual rotation behavior looks like. This is the reference the
# P-only ablation arm is being compared against (P-only converges to "hold,
# don't drop" with 0% rotation-threshold success -- this checks whether
# v4_spread genuinely rotates, or also just holds).

set -euo pipefail

YQQ=/scratch/project/prj-02-phai-lab/yqq
cd "${YQQ}"
source "${YQQ}/env.sh"
module load GCCcore/13.3.0 FFmpeg/7.0.2
mkdir -p "${YQQ}/logs" "${YQQ}/tmp"

export TMPDIR="${YQQ}/tmp"
export HF_HOME="${YQQ}/.cache/huggingface"
export TORCH_HOME="${YQQ}/.cache/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export HORA_ROOT="${YQQ}/hora"
export DEXTEROUSHANDS_ROOT="${YQQ}/DexterousHands"
export PYTHON_BIN="${YQQ}/envs/bidexhands_isaacgym_py38/bin/python"
export PYTHON_ENV_ROOT="${YQQ}/envs/bidexhands_isaacgym_py38"
export GPU_ID=0
export GRAPHICS_DEVICE_ID=0

export CHECKPOINT="${HORA_ROOT}/outputs/ShadowHandHora/v4_spread/stage1_nn/best.pth"
export OUTPUT_DIR="${OUTPUT_DIR:-${HORA_ROOT}/outputs/ShadowHandHora/v4_spread_inspect_${SLURM_JOB_ID}}"
export TARGET_EPISODES="${TARGET_EPISODES:-3}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-11}"
export WRITE_QA_VIDEO=1
export QA_VIDEO_EVERY=1
export QA_VIDEO_LIMIT=0
export KEEP_ONLY_SUCCESS=0
export KEEP_RGB_FRAMES=1
export ALLOW_EXISTING_OUTPUT=1

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TARGET_EPISODES=${TARGET_EPISODES}"

bash "${HORA_ROOT}/collect_hora_shadow_v4_tennis_wilor_raw_rigid.sh"

echo "---INSPECT_DONE---"
find "${OUTPUT_DIR}" -iname "*.mp4"
