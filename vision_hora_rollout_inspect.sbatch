#!/usr/bin/env bash
#SBATCH --job-name=hora_inspect
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/hora_inspect_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/hora_inspect_%j.err
#
# General qualitative rollout inspector for any ShadowHandHora tactile-SR
# ablation checkpoint (generalizes vision_v4_spread_rollout_inspect.sbatch,
# which hardcoded priv_info=True for the teacher only). Calls the tennis QA
# collector directly (bypassing collect_hora_shadow_v4_tennis_wilor_raw_rigid.sh's
# hardcoded priv_info=True) with the checkpoint's own actual training obs
# config, so the loaded network architecture matches the checkpoint.
#
# Required (--export=ALL,...): CHECKPOINT (abs path to a .pth), TAG (short
#   label used in the output dir name, e.g. ponly_seed0)
# Optional: PRIV_INFO (default False), COARSE_TACTILE (default False),
#   TARGET_EPISODES (default 3), ROLLOUT_SEED (default 11), MAX_STEPS
#   (default 900)

set -euo pipefail

if [[ -z "${CHECKPOINT:-}" || -z "${TAG:-}" ]]; then
  echo "CHECKPOINT and TAG must both be set" >&2
  exit 2
fi

YQQ=/scratch/project/prj-02-phai-lab/yqq
cd "${YQQ}"
source "${YQQ}/env.sh"
module load GCCcore/13.3.0 FFmpeg/7.0.2
mkdir -p "${YQQ}/logs" "${YQQ}/tmp"

export TMPDIR="${YQQ}/tmp"
export HF_HOME="${YQQ}/.cache/huggingface"
export TORCH_HOME="${YQQ}/.cache/torch"
export TORCH_EXTENSIONS_DIR="${YQQ}/.cache/torch_extensions"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

HORA_ROOT="${YQQ}/hora"
DEXTEROUSHANDS_ROOT="${YQQ}/DexterousHands"
ISAACGYM_ROOT="${YQQ}/isaacgym"
ISAACGYM_ENV="${YQQ}/envs/bidexhands_isaacgym_py38"
PYTHON_BIN="${ISAACGYM_ENV}/bin/python"

export DEXTEROUSHANDS_ROOT
export PATH="${ISAACGYM_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${ISAACGYM_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${HORA_ROOT}:${DEXTEROUSHANDS_ROOT}/bidexhands:${ISAACGYM_ROOT}/python:${DEXTEROUSHANDS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PRIV_INFO="${PRIV_INFO:-False}"
COARSE_TACTILE="${COARSE_TACTILE:-False}"
TARGET_EPISODES="${TARGET_EPISODES:-3}"
ROLLOUT_SEED="${ROLLOUT_SEED:-11}"
MAX_STEPS="${MAX_STEPS:-900}"
OUTPUT_DIR="${OUTPUT_DIR:-${HORA_ROOT}/outputs/ShadowHandHora/inspect_${TAG}_${SLURM_JOB_ID}}"

export HORA_TACTILE_DIR="${OUTPUT_DIR}"
export HORA_TARGET_EPISODES="${TARGET_EPISODES}"
export HORA_MAX_STEPS="${MAX_STEPS}"
export HORA_FRAME_STRIDE=2
export HORA_VIDEO_FPS=30
export HORA_VIDEO_WIDTH=960
export HORA_VIDEO_HEIGHT=720
export HORA_CHEST_EYE_OFFSET="0.32,0.0,0.80"
export HORA_CHEST_TARGET_OFFSET="0.0,0.0,0.08"
export HORA_HAND_COLOR_RGB="0.42,0.52,0.56"
export HORA_OBJECT_COLOR_RGB="0.40,0.58,0.28"
export HORA_KEEP_COMPONENT_VIDEOS=0
export HORA_KEEP_RGB_FRAMES=1
export HORA_WRITE_QA_VIDEO=1
export HORA_QA_VIDEO_EVERY=1
export HORA_QA_VIDEO_LIMIT=0
export HORA_KEEP_ONLY_SUCCESS=0

mkdir -p "${OUTPUT_DIR}"
cd "${HORA_ROOT}"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "TAG=${TAG}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "PRIV_INFO=${PRIV_INFO}"
echo "COARSE_TACTILE=${COARSE_TACTILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

"${PYTHON_BIN}" "${HORA_ROOT}/collect_hora_shadow_v4_wilor_raw_rigid.py" \
  task=ShadowHandHora \
  headless=True \
  pipeline=cpu \
  task.env.numEnvs=1 \
  test=True \
  task.env.object.type=simple_tennis_ball \
  task.env.grasp_cache_name=v4_spread \
  task.env.coarseTactile="${COARSE_TACTILE}" \
  task.env.randomization.randomizeMass=False \
  task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False \
  task.env.randomization.randomizePDGains=False \
  task.env.randomization.randomizeScale=True \
  task.env.randomization.randomizeScaleList="[0.7]" \
  task.env.forceScale=0 \
  task.env.randomForceProbScalar=0 \
  train.algo=PPO \
  train.ppo.priv_info="${PRIV_INFO}" \
  train.ppo.output_name="inspect_${TAG}" \
  checkpoint="${CHECKPOINT}" \
  seed="${ROLLOUT_SEED}" \
  sim_device=cuda:0 \
  rl_device=cuda:0 \
  graphics_device_id=0

echo "---INSPECT_DONE: ${TAG}---"
find "${OUTPUT_DIR}" -iname "*.mp4"
