#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${PYTHON:-python3}"
task_venv="${VENV_DIR:-$task_root/.venv}"
task_skip_system=0
for task_arg in "$@"; do
    case "$task_arg" in
        --skip-system) task_skip_system=1 ;;
        -h|--help)
            echo 'Usage: scripts/setup_linux.sh [--skip-system]'
            echo 'Installs Ubuntu/Debian packages, Python dependencies, pinned sources, and builds py_x7.'
            echo 'Environment: PYTHON (Python 3.12/3.13), VENV_DIR, BUILD_DIR, BUILD_JOBS.'
            exit 0 ;;
        *) echo "Unknown option: $task_arg" >&2; exit 2 ;;
    esac
done
[[ "$(uname -s)" == Linux ]] || { echo 'Linux is required.' >&2; exit 1; }
if (( ! task_skip_system )); then
    command -v apt-get >/dev/null || { echo 'Install the documented system packages, then use --skip-system.' >&2; exit 1; }
    task_sudo=()
    if (( EUID != 0 )); then task_sudo=(sudo); fi
    "${task_sudo[@]}" apt-get update
    "${task_sudo[@]}" apt-get install -y build-essential cmake ninja-build libeigen3-dev libyaml-cpp-dev \
        python3 python3-dev python3-venv python3-pip git patch ca-certificates \
        libgomp1 libglib2.0-0 adb v4l-utils
fi
if ! "$task_python" -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info < (3, 14) else 1)'; then
    if [[ -n "${PYTHON:-}" ]]; then
        echo 'PYTHON must select Python 3.12 or 3.13 for the pinned dependencies.' >&2
        exit 1
    fi
    task_bootstrap="$task_root/.bootstrap"
    if [[ ! -x "$task_bootstrap/bin/python" ]]; then
        python3 -m venv "$task_bootstrap"
    fi
    "$task_bootstrap/bin/python" -m pip install 'uv==0.13.8'
    export UV_PYTHON_INSTALL_DIR="$task_root/.python"
    export UV_CACHE_DIR="$task_root/.cache/uv"
    "$task_bootstrap/bin/uv" python install 3.13.14
    task_python="$("$task_bootstrap/bin/uv" python find --managed-python 3.13.14)"
fi
if [[ ! -e "$task_venv" ]]; then
    "$task_python" -m venv "$task_venv"
fi
[[ -x "$task_venv/bin/python" ]] || { echo "Invalid Python environment: $task_venv" >&2; exit 1; }
"$task_venv/bin/python" -c 'import sys; sys.exit("The existing environment must use Python 3.12 or 3.13; select a new VENV_DIR.") if not (3, 12) <= sys.version_info < (3, 14) else None'
"$task_venv/bin/python" -m pip install --upgrade pip
"$task_venv/bin/python" -m pip install -r "$task_root/requirements.txt"
"$task_venv/bin/python" -m pip check
"$task_venv/bin/python" "$task_root/scripts/fetch_dependencies.py"
VENV_DIR="$task_venv" "$task_root/scripts/build.sh"
echo 'Setup complete. Start video-only mode with scripts/run.sh teleop.'
