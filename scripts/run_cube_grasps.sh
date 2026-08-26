#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:?usage: run_cube_grasps.sh GPU_ID}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${HORA_PYTHON:-/lp-dev/qianqian/envs/rlgpu/bin/python}"
test -x "${python_bin}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/lp-dev/qianqian/envs/rlgpu/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${python_bin}" gen_grasp.py \
  task=ShadowHandCubeGrasp \
  headless=True \
  pipeline=gpu \
  task.env.numEnvs=8192 \
  test=True \
  task.env.controller.controlFrequencyInv=8 \
  task.env.episodeLength=80 \
  task.env.controller.torque_control=False \
  task.env.genGrasps=True \
  task.env.graspCacheTarget=50000 \
  task.env.baseObjScale=0.7 \
  task.env.grasp_cache_name=shadow_cube \
  task.env.object.type=cube \
  task.env.randomization.randomizeMass=True \
  task.env.randomization.randomizeMassLower=0.05 \
  task.env.randomization.randomizeMassUpper=0.051 \
  task.env.randomization.randomizeCOM=False \
  task.env.randomization.randomizeFriction=False \
  task.env.randomization.randomizePDGains=False \
  task.env.randomization.randomizeScale=False \
  train.ppo.priv_info=True

"${python_bin}" - <<'PY'
import numpy as np
path = "cache/shadow_cube_grasp_50k_s07.npy"
grasps = np.load(path)
assert grasps.ndim == 2 and grasps.shape[1] == 31, grasps.shape
assert grasps.shape[0] >= 50000, grasps.shape
assert np.isfinite(grasps).all()
print(f"validated {path}: shape={grasps.shape}", flush=True)
PY
