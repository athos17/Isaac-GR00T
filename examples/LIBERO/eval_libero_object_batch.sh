#!/usr/bin/env bash
set -euo pipefail

# Batch client-side LIBERO Object evaluation.
# Assumes the GR00T policy server is already running, for example:
#   uv run python gr00t/eval/run_gr00t_server.py \
#     --model-path /local/yangshuo/lyh/Isaac-GR00T/outputs/libero_object \
#     --embodiment-tag LIBERO_PANDA \
#     --use-sim-policy-wrapper

PROJECT_ROOT="${PROJECT_ROOT:-/local/yangshuo/lyh/Isaac-GR00T}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/outputs/libero_object}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${PROJECT_ROOT}/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python}"

POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5555}"
N_EPISODES="${N_EPISODES:-20}"
N_ENVS="${N_ENVS:-5}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
SEED="${SEED:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_PATH}/eval_libero_object/${TIMESTAMP}}"
LOG_DIR="${OUTPUT_DIR}/logs"
VIDEO_ROOT="${VIDEO_ROOT:-${OUTPUT_DIR}/videos}"
SUMMARY_FILE="${OUTPUT_DIR}/summary.tsv"

# If you installed EGL without sudo under /local/yangshuo/lyh/local/egl, this keeps
# MuJoCo/PyOpenGL able to find libEGL.so. It is harmless when the directory is absent.
LOCAL_EGL_LIB="/local/yangshuo/lyh/local/egl/usr/lib/x86_64-linux-gnu"
if [[ -d "${LOCAL_EGL_LIB}" ]]; then
  export LD_LIBRARY_PATH="${LOCAL_EGL_LIB}:${LD_LIBRARY_PATH:-}"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

TASKS=(
  "libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_butter_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_milk_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket"
)

mkdir -p "${LOG_DIR}" "${VIDEO_ROOT}"
printf "task\tsuccess_rate\tlog\tvideo_dir\n" > "${SUMMARY_FILE}"

echo "Project root: ${PROJECT_ROOT}"
echo "Checkpoint expected on server: ${MODEL_PATH}"
echo "Policy server: ${POLICY_HOST}:${POLICY_PORT}"
echo "Episodes per task: ${N_EPISODES}, envs: ${N_ENVS}, max steps: ${MAX_EPISODE_STEPS}, action steps: ${N_ACTION_STEPS}"
echo "Output dir: ${OUTPUT_DIR}"

for task in "${TASKS[@]}"; do
  task_slug="${task#libero_sim/}"
  log_file="${LOG_DIR}/${task_slug}.log"
  video_dir="${VIDEO_ROOT}/${task_slug}"

  echo
  echo "===== Evaluating ${task} ====="

  cmd=(
    "${LIBERO_PYTHON}" "${PROJECT_ROOT}/gr00t/eval/rollout_policy.py"
    --n-episodes "${N_EPISODES}"
    --policy-client-host "${POLICY_HOST}"
    --policy-client-port "${POLICY_PORT}"
    --max-episode-steps "${MAX_EPISODE_STEPS}"
    --env-name "${task}"
    --n-action-steps "${N_ACTION_STEPS}"
    --n-envs "${N_ENVS}"
    --video-dir "${video_dir}"
  )
  if [[ -n "${SEED}" ]]; then
    cmd+=(--seed "${SEED}")
  fi

  "${cmd[@]}" 2>&1 | tee "${log_file}"

  success_rate="$(awk '/success rate:/ {rate=$NF} END {print rate}' "${log_file}")"
  success_rate="${success_rate:-NA}"
  printf "%s\t%s\t%s\t%s\n" "${task}" "${success_rate}" "${log_file}" "${video_dir}" >> "${SUMMARY_FILE}"
  echo "Task success rate: ${success_rate}"
done

echo
echo "===== LIBERO Object summary ====="
column -t -s $'\t' "${SUMMARY_FILE}" || cat "${SUMMARY_FILE}"
echo "Summary saved to: ${SUMMARY_FILE}"
