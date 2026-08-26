#!/usr/bin/env bash
# Formal HORA v4 tennis-ball collection launcher.
#
# Default policy:
# - collect 3 shards x 1000 kept successful trajectories on one GPU;
# - keep RGB frames + trajectory_env0.npz + pressure_grids.npz per trajectory;
# - skip side-by-side RGB+tactile videos by default; render QA videos offline if needed;
# - require full 2pi rotation, no object drop, and nonzero rigid contacts.
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

runner="${RUNNER:-${hora_root}/collect_hora_shadow_v4_tennis_wilor_raw_rigid.sh}"
if [[ ! -f "${runner}" ]]; then
  echo "Missing runner: ${runner}" >&2
  exit 3
fi

gpu_id="${GPU_ID:-4}"
base_seed="${BASE_SEED:-6100}"
shards="${NUM_SHARDS:-3}"
episodes_per_shard="${EPISODES_PER_SHARD:-1000}"
write_qa_video="${WRITE_QA_VIDEO:-0}"
qa_every="${QA_VIDEO_EVERY:-100}"
qa_limit="${QA_VIDEO_LIMIT:-10}"
stamp="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
base_output="${BASE_OUTPUT_DIR:-${hora_root}/outputs/ShadowHandHora/v4_spread_wilor_raw_rigid_bulk_3k_${stamp}}"

mkdir -p "${base_output}"
echo "HORA bulk output: ${base_output}"
echo "GPU=${gpu_id} shards=${shards} episodes_per_shard=${episodes_per_shard} base_seed=${base_seed}"

for shard in $(seq 0 "$((shards - 1))"); do
  shard_seed=$((base_seed + shard))
  shard_output="${base_output}/shard_$(printf "%02d" "${shard}")_seed_${shard_seed}"
  echo "START shard=${shard} seed=${shard_seed} output=${shard_output}"
  GPU_ID="${gpu_id}" \
  ROLLOUT_SEED="${shard_seed}" \
  TARGET_EPISODES="${episodes_per_shard}" \
  KEEP_ONLY_SUCCESS=1 \
  WRITE_QA_VIDEO="${write_qa_video}" \
  QA_VIDEO_EVERY="${qa_every}" \
  QA_VIDEO_LIMIT="${qa_limit}" \
  KEEP_RGB_FRAMES=1 \
  OUTPUT_DIR="${shard_output}" \
  "${runner}" 2>&1 | tee "${shard_output}.log"
done

echo "DONE HORA bulk output: ${base_output}"
