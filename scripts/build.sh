#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_venv="${VENV_DIR:-$task_root/.venv}"
task_python="$task_venv/bin/python"
task_build="${BUILD_DIR:-$task_root/cpp/build}"
[[ -x "$task_python" ]] || { echo 'Run scripts/setup_linux.sh first.' >&2; exit 1; }
task_pybind="$("$task_python" -m pybind11 --cmakedir)"
task_prefix="$("$task_python" -c 'import sys; print(sys.prefix)')"
cmake -S "$task_root/cpp" -B "$task_build" -G Ninja \
    -DPython3_EXECUTABLE="$task_python" \
    -Dpybind11_DIR="$task_pybind" \
    -DCMAKE_PREFIX_PATH="$task_prefix${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$task_build" --parallel "${BUILD_JOBS:-2}"
PYTHONPATH="$task_root/python${PYTHONPATH:+:$PYTHONPATH}" "$task_python" -c \
    'import py_x7; print("Built extension:", py_x7.__file__)'
