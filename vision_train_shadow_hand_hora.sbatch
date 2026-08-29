#!/usr/bin/env bash
#SBATCH --job-name=hora_train
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/hora_train_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/hora_train_%j.err
#
# Fresh-from-scratch ShadowHandHora PPO training on VISION H200, same task
# setup (simple_tennis_ball, v4_spread grasp cache) as the existing teacher
# checkpoint, but with the observation space controlled by PRIV_INFO. This
# is the "does tactile help SR" ablation's per-arm training launcher
# (VTDexManip-style: each modality variant is trained as its own PPO policy,
# no distillation) -- PRIV_INFO=False gives the genuinely blind proprio+
# history-only "P" baseline, PRIV_INFO=True reproduces the original
# privileged teacher for reference. COARSE_TACTILE=True additionally appends
# a 217-taxel/hand coarse pressure channel (see allegro_hand_hora.py's
# _setup_coarse_tactile_tables) -- requires DEXTEROUSHANDS_ROOT, exported
# below, since it reuses bidexhands' egotouch_taxels.py geometry.
#
# Required (--export=ALL,...): OUTPUT_NAME (e.g. ShadowHandHora/v1_ponly_seed0)
# Optional: PRIV_INFO (default False), COARSE_TACTILE (default False), SEED
#           (default 0), NUM_ENVS (default task yaml's own default, 16384),
#           MAX_AGENT_STEPS (default the task yaml's own default, 1.5e9 --
#           rely on --time to bound a first-pass run instead of guessing a
#           cutoff)

set -euo pipefail

if [[ -z "${OUTPUT_NAME:-}" ]]; then
  echo "OUTPUT_NAME must be set (--export=ALL,OUTPUT_NAME=ShadowHandHora/<name>)" >&2
  exit 2
fi

YQQ=/scratch/project/prj-02-phai-lab/yqq
cd "${YQQ}"
source "${YQQ}/env.sh"
mkdir -p "${YQQ}/logs" "${YQQ}/tmp"

export TMPDIR="${YQQ}/tmp"
export HF_HOME="${YQQ}/.cache/huggingface"
export PIP_CACHE_DIR="${YQQ}/.cache/pip"
export TORCH_HOME="${YQQ}/.cache/torch"
export TORCH_EXTENSIONS_DIR="${YQQ}/.cache/torch_extensions"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1

export ISAACGYM_ENV="${YQQ}/envs/bidexhands_isaacgym_py38"
export ISAACGYM_ROOT="${YQQ}/isaacgym"
export HORA_ROOT="${YQQ}/hora"
export DEXTEROUSHANDS_ROOT="${YQQ}/DexterousHands"

export PATH="${ISAACGYM_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${ISAACGYM_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${ISAACGYM_ROOT}/python:${PYTHONPATH:-}"

PRIV_INFO="${PRIV_INFO:-False}"
COARSE_TACTILE="${COARSE_TACTILE:-False}"
SEED="${SEED:-0}"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "OUTPUT_NAME=${OUTPUT_NAME}"
echo "PRIV_INFO=${PRIV_INFO}"
echo "COARSE_TACTILE=${COARSE_TACTILE}"
echo "SEED=${SEED}"
nvidia-smi || true

cd "${HORA_ROOT}"

EXTRA_ARGS=()
[[ -n "${NUM_ENVS:-}" ]] && EXTRA_ARGS+=("task.env.numEnvs=${NUM_ENVS}")
[[ -n "${MAX_AGENT_STEPS:-}" ]] && EXTRA_ARGS+=("train.ppo.max_agent_steps=${MAX_AGENT_STEPS}")
EXTRA_ARGS+=("task.env.coarseTactile=${COARSE_TACTILE}")

"${ISAACGYM_ENV}/bin/python" train.py \
  task=ShadowHandHora \
  headless=True \
  pipeline=gpu \
  test=False \
  task.env.object.type=simple_tennis_ball \
  task.env.grasp_cache_name=v4_spread \
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
  train.ppo.output_name="${OUTPUT_NAME}" \
  seed="${SEED}" \
  sim_device=cuda:0 \
  rl_device=cuda:0 \
  graphics_device_id=0 \
  "${EXTRA_ARGS[@]}"

echo "---TRAINING DONE: ${OUTPUT_NAME}---"
echo "Checkpoints under: ${HORA_ROOT}/outputs/${OUTPUT_NAME}/stage1_nn/"
