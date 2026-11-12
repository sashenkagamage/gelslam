# Source this file to activate the GelSLAM environment in the current shell:
#   source ~/gelslam/env.sh

export MAMBA_EXE="${MAMBA_EXE:-$HOME/micromamba/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

if ! command -v micromamba >/dev/null 2>&1; then
    eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
fi

micromamba activate gelslam
# shellcheck disable=SC1091
source "$HOME/gelslam/install/setup.bash"

# ROS nodes otherwise load Ubuntu's libstdc++, which is too old for conda Qt/OpenCV.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
