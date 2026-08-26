#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:?usage: run_cube_stage1.sh GPU_ID [smoke|full]}"
mode="${2:-full}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${HORA_PYTHON:-/lp-dev/qianqian/envs/rlgpu/bin/python}"
test -x "${python_bin}"

test -f cache/shadow_cube_grasp_50k_s07.npy

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/lp-dev/qianqian/envs/rlgpu/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ "${mode}" == "smoke" ]]; then
  warm_checkpoint=outputs/ShadowHandRWS/rws_v1/stage1_nn/best.pth
  num_envs=1024
  minibatch_size=16384
  max_steps=2097152
  output_name=ShadowHandCubeRotation/cube_z_smoke
elif [[ "${mode}" == "full" ]]; then
  warm_checkpoint=outputs/ShadowHandCubeRotation/cube_z_smoke/stage1_nn/best.pth
  num_envs=2048
  minibatch_size=32768
  max_steps=300000000
  output_name=ShadowHandCubeRotation/cube_z_stage1
else
  echo "mode must be smoke or full" >&2
  exit 2
fi

test -f "${warm_checkpoint}"

"${python_bin}" train.py \
  task=ShadowHandCubeRotation \
  headless=True \
  pipeline=gpu \
  num_envs="${num_envs}" \
  seed=0 \
  checkpoint="${warm_checkpoint}" \
  train.ppo.minibatch_size="${minibatch_size}" \
  train.ppo.output_name="${output_name}" \
  train.ppo.max_agent_steps="${max_steps}"
