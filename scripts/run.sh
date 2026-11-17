#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_venv="${VENV_DIR:-$task_root/.venv}"
task_command="${1:-teleop}"
if (( $# )); then shift; fi
case "$task_command" in
    teleop) task_entry=run_teleop.py ;;
    replay) task_entry=run_replay.py ;;
    inspect) task_entry=episode_viewer.py ;;
    -h|--help)
        echo 'Usage: scripts/run.sh {teleop|replay|inspect} [arguments...]'
        echo 'Default: teleop in video-only mode. Use teleop --with_arms to enable robot control.'
        exit 0 ;;
    *) echo "Unknown command: $task_command" >&2; exit 2 ;;
esac
[[ -x "$task_venv/bin/python" ]] || { echo 'Run scripts/setup_linux.sh first.' >&2; exit 1; }
exec "$task_venv/bin/python" "$task_root/python/$task_entry" "$@"
