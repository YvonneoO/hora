#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
mkdir -p outputs/ShadowHandCubeRotation/pipeline
status_file=outputs/ShadowHandCubeRotation/pipeline/status.txt

echo "queued $(date -u +%FT%TZ): waiting for the first free DexterousHands GPU in 0-3" | tee "${status_file}"
selected_gpu=""
while [[ -z "${selected_gpu}" ]]; do
  for gpu in 0 1 2 3; do
    if ! pgrep -af "pipeline_worker.py --gpu ${gpu}( |$)" >/dev/null; then
      selected_gpu="${gpu}"
      break
    fi
  done
  if [[ -z "${selected_gpu}" ]]; then
    sleep 30
  fi
done

echo "selected_gpu=${selected_gpu} $(date -u +%FT%TZ)" | tee "${status_file}"
if [[ -f cache/shadow_cube_grasp_50k_s07.npy ]]; then
  /lp-dev/qianqian/envs/rlgpu/bin/python - <<'PY'
import numpy as np
path = "cache/shadow_cube_grasp_50k_s07.npy"
grasps = np.load(path)
assert grasps.shape == (50000, 31), grasps.shape
assert np.isfinite(grasps).all()
print(f"validated existing {path}: shape={grasps.shape}", flush=True)
PY
  echo "grasp_cache_ready gpu=${selected_gpu} $(date -u +%FT%TZ)" | tee "${status_file}"
else
  echo "grasp_generation gpu=${selected_gpu} $(date -u +%FT%TZ)" | tee "${status_file}"
  scripts/run_cube_grasps.sh "${selected_gpu}"
fi
echo "smoke_training gpu=${selected_gpu} $(date -u +%FT%TZ)" | tee "${status_file}"
scripts/run_cube_stage1.sh "${selected_gpu}" smoke
test -f outputs/ShadowHandCubeRotation/cube_z_smoke/stage1_nn/best.pth

echo "full_training gpu=${selected_gpu} $(date -u +%FT%TZ)" | tee "${status_file}"
scripts/run_cube_stage1.sh "${selected_gpu}" full
echo "evaluation gpu=${selected_gpu} $(date -u +%FT%TZ)" | tee "${status_file}"
scripts/eval_cube_stage1.sh "${selected_gpu}"
echo "complete gpu=${selected_gpu} $(date -u +%FT%TZ)" | tee "${status_file}"
