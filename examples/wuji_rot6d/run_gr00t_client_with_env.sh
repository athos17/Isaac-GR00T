#!/usr/bin/env bash
set -o pipefail

REPO_ROOT="${GR00T_REPO_ROOT:-/home/zxc/Desktop/GR00T/Isaac-GR00T}"
SDK="${ASTRIBOT_SDK_ROOT:-/home/zxc/cenyj/astribot_sdk/astribot_sdk_ros2-master}"
SHIM="${ASTRIBOT_PYTHON_SHIMS:-/home/zxc/cenyj/astribot_sdk/python_shims}"
WUJI_SETUP="${WUJI_HAND_SETUP:-/home/zxc/Desktop/wuji/wuji-teleop/wujihandros2/install/setup.bash}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
CLIENT="${REPO_ROOT}/examples/wuji_rot6d/run_gr00t_client.py"

source_if_present() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        # shellcheck disable=SC1090
        source "${path}"
    else
        echo "[run_gr00t_client_with_env.sh][WARN] setup file not found: ${path}" >&2
    fi
}

source_if_present "${ROS_SETUP}"
source_if_present "${SDK}/env.sh"
source_if_present "${SDK}/install/setup.sh"
source_if_present "${WUJI_SETUP}"
source_if_present "${REPO_ROOT}/.venv/bin/activate"

export PYTHONPATH="${SHIM}:${SDK}/third_party/software/astribot_ros_middleware/lib/python3.10/site-packages:${SDK}/astribot_msgs/local/lib/python3.10/dist-packages:${SDK}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${SDK}/astribot_msgs/lib:${SDK}/astribot_msgs/local/lib:${SDK}/astribot_sdk/core/common/robotics_library_py:${SDK}/astribot_sdk/core/common/whole_body_control/third_party:${SDK}/third_party/third_pkg/pinocchio/lib:${SDK}/third_party/drake/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"
set -e
python - "${CLIENT}" "$@" <<'PY'
import importlib
import runpy
import sys

client_path = sys.argv[1]
client_args = sys.argv[2:]

pkg = importlib.import_module("robotics_library_py")
print("[bootstrap] robotics_library_py =", getattr(pkg, "__file__", None))
print("[bootstrap] robotics_library_py.__path__ =", list(getattr(pkg, "__path__", [])))

if not hasattr(pkg, "__path__"):
    raise RuntimeError("robotics_library_py was loaded as a module, not a package")

sys.argv = [client_path, *client_args]
runpy.run_path(client_path, run_name="__main__")
PY
