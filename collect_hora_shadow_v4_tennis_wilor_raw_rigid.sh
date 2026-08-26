#!/usr/bin/env bash
# Reproduce the HORA ShadowHand v4 tennis-ball rollout using the same WiLoR
# chest view and raw rigid-contact EgoTouch projection as the DexterousHands
# multi_success_wilor_view_raw_rigid collector.
set -euo pipefail

if [[ -n "${HORA_ROOT:-}" ]]; then
  hora_root="${HORA_ROOT}"
elif [[ -d /lp-dev/qianqian/hora ]]; then
  hora_root=/lp-dev/qianqian/hora
elif [[ -d /workspace/hora ]]; then
  hora_root=/workspace/hora
else
  echo "Cannot find HORA root. Set HORA_ROOT." >&2
  exit 2
fi

if [[ -n "${DEXTEROUSHANDS_ROOT:-}" ]]; then
  dex_root="${DEXTEROUSHANDS_ROOT}"
elif [[ -d /lp-dev/qianqian/DexterousHands/bidexhands ]]; then
  dex_root=/lp-dev/qianqian/DexterousHands
elif [[ -d /workspace/DexterousHands/bidexhands ]]; then
  dex_root=/workspace/DexterousHands
elif [[ -d /workspace/bidexhands ]]; then
  dex_root=/workspace
else
  echo "Cannot find DexterousHands root for the EgoTouch taxel mapper." >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-/lp-dev/qianqian/envs/rlgpu/bin/python}"
python_env_root="${PYTHON_ENV_ROOT:-$(cd "$(dirname "${python_bin}")/.." && pwd)}"
gpu_id="${GPU_ID:-4}"
seed="${ROLLOUT_SEED:-0}"
checkpoint="${CHECKPOINT:-${hora_root}/outputs/ShadowHandHora/v4_spread/stage1_nn/best.pth}"
output_dir="${OUTPUT_DIR:-${hora_root}/outputs/ShadowHandHora/v4_spread_wilor_view_raw_rigid_recheck}"
collector="${COLLECTOR:-${hora_root}/collect_hora_shadow_v4_wilor_raw_rigid.py}"

for required in "${python_bin}" "${checkpoint}" "${collector}" "${dex_root}/bidexhands/tactile_collection/egotouch_taxels.py"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file is missing: ${required}" >&2
    exit 3
  fi
done

if [[ -e "${output_dir}/summary.json" && "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing run: ${output_dir}" >&2
  echo "Set OUTPUT_DIR to a new directory, or ALLOW_EXISTING_OUTPUT=1 explicitly." >&2
  exit 4
fi

cd "${hora_root}"
mkdir -p "${output_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHON="${python_bin}"
export PYTHONUNBUFFERED=1
export DEXTEROUSHANDS_ROOT="${dex_root}"
export PYTHONPATH="${hora_root}:${dex_root}/bidexhands:${dex_root}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${python_env_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export HORA_TACTILE_DIR="${output_dir}"
export HORA_TARGET_EPISODES="${TARGET_EPISODES:-2}"
export HORA_MAX_STEPS="${MAX_STEPS:-900}"
export HORA_FRAME_STRIDE=2
export HORA_VIDEO_FPS=30
export HORA_VIDEO_WIDTH=960
export HORA_VIDEO_HEIGHT=720
export HORA_CHEST_EYE_OFFSET="0.32,0.0,0.80"
export HORA_CHEST_TARGET_OFFSET="0.0,0.0,0.08"
export HORA_HAND_COLOR_RGB="0.42,0.52,0.56"
export HORA_OBJECT_COLOR_RGB="0.40,0.58,0.28"
export HORA_KEEP_COMPONENT_VIDEOS=0
export HORA_KEEP_RGB_FRAMES="${KEEP_RGB_FRAMES:-1}"
export HORA_WRITE_QA_VIDEO="${WRITE_QA_VIDEO:-0}"
export HORA_QA_VIDEO_EVERY="${QA_VIDEO_EVERY:-1}"
export HORA_QA_VIDEO_LIMIT="${QA_VIDEO_LIMIT:-0}"
export HORA_KEEP_ONLY_SUCCESS="${KEEP_ONLY_SUCCESS:-0}"

exec "${python_bin}" "${collector}" \
  task=ShadowHandHora \
  headless=True \
  pipeline=cpu \
  task.env.numEnvs=1 \
  test=True \
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
  train.ppo.priv_info=True \
  train.ppo.output_name=ShadowHandHora/v4_spread \
  checkpoint="${checkpoint}" \
  seed="${seed}" \
  sim_device=cuda:0 \
  rl_device=cuda:0 \
  graphics_device_id=0
