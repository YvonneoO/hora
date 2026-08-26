#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:?usage: eval_cube_stage1.sh GPU_ID [CHECKPOINT]}"
checkpoint="${2:-outputs/ShadowHandCubeRotation/cube_z_stage1/stage1_nn/best.pth}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${HORA_PYTHON:-/lp-dev/qianqian/envs/rlgpu/bin/python}"
test -x "${python_bin}"

test -f "${checkpoint}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/lp-dev/qianqian/envs/rlgpu/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${python_bin}" train.py \
  task=ShadowHandCubeRotation \
  headless=True \
  pipeline=gpu \
  num_envs=256 \
  test=True \
  task.on_evaluation=True \
  train.ppo.output_name=ShadowHandCubeRotation/cube_z_stage1_eval \
  checkpoint="${checkpoint}"
