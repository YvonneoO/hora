#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
mkdir -p outputs/ShadowHandCubeRotation/pipeline
status_file=outputs/ShadowHandCubeRotation/pipeline/status.txt
log_file=outputs/ShadowHandCubeRotation/pipeline/pipeline.log

echo "queued $(date -u +%FT%TZ): waiting for DexterousHands GPU-0 worker" | tee "${status_file}"
while pgrep -af 'pipeline_worker.py --gpu 0|CUDA_VISIBLE_DEVICES=0.*train.py|shadow_hand_catch_over2_underarm' >/dev/null; do
  sleep 30
done

echo "grasp_generation $(date -u +%FT%TZ)" | tee "${status_file}"
scripts/run_cube_grasps.sh 0
echo "smoke_training $(date -u +%FT%TZ)" | tee "${status_file}"
scripts/run_cube_stage1.sh 0 smoke
test -f outputs/ShadowHandCubeRotation/cube_z_smoke/stage1_nn/best.pth

echo "full_training $(date -u +%FT%TZ)" | tee "${status_file}"
scripts/run_cube_stage1.sh 0 full
echo "evaluation $(date -u +%FT%TZ)" | tee "${status_file}"
scripts/eval_cube_stage1.sh 0
echo "complete $(date -u +%FT%TZ)" | tee "${status_file}"
