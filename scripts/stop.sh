#!/bin/bash
set -u

# stop_sam3d.sh
# Stop and clean sam3d processor container, then check GPU memory/process leftovers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep consistent with start_sam3d.sh
CONTAINER_NAME="sim-proc-sam3d"
GPU_ID="0"

# Graceful stop timeout in seconds.
STOP_TIMEOUT=30

# Whether to kill GPU leftover processes on GPU_ID after container removal.
# Default: false, because blindly killing GPU processes may affect other services.
KILL_GPU_LEFTOVERS=false

usage() {
    cat <<EOF
Usage:
  ./stop_sam3d.sh [options]

Options:
  --timeout SECONDS          Graceful stop timeout before force removal. Default: ${STOP_TIMEOUT}
  --gpu GPU_ID               GPU index to check. Default: ${GPU_ID}
  --kill-gpu-leftovers       Kill leftover compute processes still using the target GPU.
  -h, --help                 Show this help.

Examples:
  ./stop_sam3d.sh
  ./stop_sam3d.sh --timeout 60
  ./stop_sam3d.sh --gpu 0 --kill-gpu-leftovers
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --timeout requires a value."
                exit 2
            fi
            STOP_TIMEOUT="$2"
            shift 2
            ;;
        --gpu)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --gpu requires a value."
                exit 2
            fi
            GPU_ID="$2"
            shift 2
            ;;
        --kill-gpu-leftovers)
            KILL_GPU_LEFTOVERS=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            usage
            exit 2
            ;;
    esac
done

echo "[INFO] Stopping sam3d service..."
echo "[INFO] container=${CONTAINER_NAME}, gpu=${GPU_ID}, timeout=${STOP_TIMEOUT}s"

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] docker command not found."
    exit 1
fi

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

container_running() {
    docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | grep -Fxq "true"
}

print_container_state() {
    if container_exists; then
        STATUS=$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "unknown")
        RUNNING=$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || echo "unknown")
        EXIT_CODE=$(docker inspect -f '{{.State.ExitCode}}' "${CONTAINER_NAME}" 2>/dev/null || echo "unknown")
        echo "[INFO] container state: status=${STATUS}, running=${RUNNING}, exit_code=${EXIT_CODE}"
    else
        echo "[INFO] container state: not found"
    fi
}

echo "[INFO] Current container state:"
print_container_state

# 1) Gracefully stop the container if it is running.
if container_exists; then
    if container_running; then
        echo "[INFO] Sending SIGTERM via docker stop, timeout=${STOP_TIMEOUT}s..."
        if docker stop -t "${STOP_TIMEOUT}" "${CONTAINER_NAME}" >/dev/null 2>&1; then
            echo "[INFO] Container stopped gracefully."
        else
            echo "[WARN] docker stop failed or timed out."
        fi
    else
        echo "[INFO] Container exists but is not running."
    fi

    # 2) Remove the container after stop.
    # Use rm -f as a safety net. If the container is still alive, this forces SIGKILL.
    if container_exists; then
        echo "[INFO] Removing container: ${CONTAINER_NAME}"
        if docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1; then
            echo "[INFO] Container removed."
        else
            echo "[WARN] Failed to remove container. Current state:"
            print_container_state
        fi
    fi
else
    echo "[INFO] Container not found, nothing to stop."
fi

# 3) Give Docker/NVIDIA runtime a moment to release GPU contexts.
sleep 2

echo "[INFO] Checking GPU status..."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi command not found. Cannot verify GPU memory release."
    echo "[INFO] Done."
    exit 0
fi

echo "========== nvidia-smi =========="
nvidia-smi || true
echo "================================"

# 4) Query compute apps on the target GPU.
# This returns PIDs of compute processes using the GPU.
GPU_PIDS=$(
    nvidia-smi \
        --id="${GPU_ID}" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF {print $1}' \
    | sort -u
)

if [[ -z "${GPU_PIDS}" ]]; then
    echo "[INFO] No compute processes found on GPU ${GPU_ID}."
    echo "[INFO] GPU memory should be released."
    echo "[INFO] Done."
    exit 0
fi

echo "[WARN] Compute processes still found on GPU ${GPU_ID}:"
for pid in ${GPU_PIDS}; do
    if ps -p "${pid}" >/dev/null 2>&1; then
        echo "----- PID ${pid} -----"
        ps -o pid,ppid,user,stat,etime,cmd -p "${pid}" || true
    else
        echo "PID ${pid} no longer exists."
    fi
done

# 5) Optional: kill leftover GPU processes.
if [[ "${KILL_GPU_LEFTOVERS}" == "true" ]]; then
    echo "[WARN] --kill-gpu-leftovers enabled."
    echo "[WARN] Sending SIGTERM to leftover GPU processes on GPU ${GPU_ID}..."

    for pid in ${GPU_PIDS}; do
        if ps -p "${pid}" >/dev/null 2>&1; then
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done

    sleep 5

    STILL_ALIVE=""
    for pid in ${GPU_PIDS}; do
        if ps -p "${pid}" >/dev/null 2>&1; then
            STILL_ALIVE="${STILL_ALIVE} ${pid}"
        fi
    done

    if [[ -n "${STILL_ALIVE}" ]]; then
        echo "[WARN] Some GPU processes are still alive after SIGTERM:${STILL_ALIVE}"
        echo "[WARN] Sending SIGKILL..."
        for pid in ${STILL_ALIVE}; do
            kill -KILL "${pid}" 2>/dev/null || true
        done
        sleep 2
    fi

    echo "[INFO] GPU status after killing leftovers:"
    echo "========== nvidia-smi =========="
    nvidia-smi || true
    echo "================================"

    REMAINING=$(
        nvidia-smi \
            --id="${GPU_ID}" \
            --query-compute-apps=pid \
            --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF {print $1}' \
        | sort -u
    )

    if [[ -z "${REMAINING}" ]]; then
        echo "[INFO] No remaining compute processes on GPU ${GPU_ID}."
    else
        echo "[ERROR] GPU ${GPU_ID} still has remaining compute processes:"
        echo "${REMAINING}"
        echo "[ERROR] You may need to inspect them manually:"
        echo "ps -fp ${REMAINING}"
        exit 1
    fi
else
    echo "[WARN] GPU ${GPU_ID} still has processes."
    echo "[WARN] I did not kill them automatically to avoid killing unrelated services."
    echo "[INFO] To force-kill GPU leftovers, run:"
    echo "./stop_sam3d.sh --gpu ${GPU_ID} --kill-gpu-leftovers"
fi

echo "[INFO] Done."