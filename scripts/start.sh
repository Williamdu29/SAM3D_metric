#!/bin/bash
set -u

source ../bin/env.sh

CONTAINER_NAME="sim-proc-sam3d"
IMAGE_NAME="sam3d-runtime:0.1"

# 清理旧容器，避免同名冲突
if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    echo "[INFO] Removing old container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true # 先强制删除旧容器，忽略错误
fi

# 统一收集启动参数
EXEC_PARAM="$*"

# 容器内启动脚本：
# 1) 优先自动寻找 conda.sh 并激活 segman
# 2) 若找不到 conda，则手工兜底设置 CONDA_PREFIX / CUDA_HOME
# 3) 最后执行 /app/sam3d/run.sh
INNER_BOOTSTRAP=$(cat <<'EOF'
set -e

echo "[BOOT] pwd=$(pwd)"
echo "[BOOT] trying to prepare conda env..."

CONDA_SH=""

for p in \
    /opt/conda/etc/profile.d/conda.sh \
    /root/miniconda3/etc/profile.d/conda.sh \
    /root/anaconda3/etc/profile.d/conda.sh \
    /usr/local/miniconda3/etc/profile.d/conda.sh \
    /usr/local/anaconda3/etc/profile.d/conda.sh \
    /home/ubuntu/miniconda3/etc/profile.d/conda.sh \
    /home/ubuntu/anaconda3/etc/profile.d/conda.sh
do
    if [ -f "$p" ]; then
        CONDA_SH="$p"
        break
    fi
done

if [ -n "$CONDA_SH" ]; then
    echo "[BOOT] found conda.sh: $CONDA_SH"
    # shellcheck disable=SC1090
    source "$CONDA_SH"

    if conda env list 2>/dev/null | awk '{print $1}' | grep -Fxq "segman"; then
        echo "[BOOT] activating conda env: segman"
        conda activate segman
        echo "[BOOT] CONDA_PREFIX=${CONDA_PREFIX:-}"
    else
        echo "[WARN] conda found, but env 'segman' not found"
        export CONDA_PREFIX="${CONDA_PREFIX:-/opt/conda}"
        export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
        echo "[BOOT] fallback CONDA_PREFIX=$CONDA_PREFIX"
        echo "[BOOT] fallback CUDA_HOME=$CUDA_HOME"
    fi
else
    echo "[WARN] conda.sh not found, using fallback env vars"
    if [ -d /opt/conda/envs/segman ]; then
        export CONDA_PREFIX=/opt/conda/envs/segman
    elif [ -d /opt/conda ]; then
        export CONDA_PREFIX=/opt/conda
    elif [ -d /usr/local/cuda ]; then
        export CONDA_PREFIX=/usr/local/cuda
    else
        export CONDA_PREFIX=/opt/conda
    fi

    export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"

    echo "[BOOT] fallback CONDA_PREFIX=$CONDA_PREFIX"
    echo "[BOOT] fallback CUDA_HOME=$CUDA_HOME"
fi

echo "[BOOT] final CONDA_PREFIX=${CONDA_PREFIX:-}"
echo "[BOOT] final CUDA_HOME=${CUDA_HOME:-}"
echo "[BOOT] launching /app/sam3d/run.sh"

exec /app/sam3d/run.sh
EOF
)

if [[ "${1:-}" == "debug" ]]; then
    echo "[INFO] Starting sam3d engine in interactive debug mode."

    docker run -it --rm \
        --gpus "device=0" \
        --volume="${PROC_HOME}/sam3d:/app/sam3d" \
        --volume="${DATA_HOME}/minio_data/4dsimulator:/data" \
        --volume="${LIB_HOME}/sam-3d-objects:/app/lib/sam-3d-objects" \
        --volume="${PROC_HOME}/framework:/app/framework" \
        --volume="${PROC_HOME}/conf:/app/conf" \
        -e PYTHONPATH="/app:/app/lib/sam-3d-objects" \
        -e PROC_CONF="${EXEC_PARAM}" \
        --net=host \
        --name "${CONTAINER_NAME}" \
        "${IMAGE_NAME}" \
        /bin/bash -lc "${INNER_BOOTSTRAP}"
else
    echo "[INFO] Starting sam3d engine in daemon mode."

    CID=$(docker run -d \
        --gpus "device=0" \
        --volume="${PROC_HOME}/sam3d:/app/sam3d" \
        --volume="${DATA_HOME}/minio_data/4dsimulator:/data" \
        --volume="${LIB_HOME}/sam-3d-objects:/app/lib/sam-3d-objects" \
        --volume="${PROC_HOME}/framework:/app/framework" \
        --volume="${PROC_HOME}/conf:/app/conf" \
        -e PYTHONPATH="/app:/app/lib/sam-3d-objects" \
        -e PROC_CONF="${EXEC_PARAM}" \
        --net=host \
        --name "${CONTAINER_NAME}" \
        "${IMAGE_NAME}" \
        /bin/bash -lc "${INNER_BOOTSTRAP}" 2>&1)

    RC=$?
    if [[ ${RC} -ne 0 ]]; then
        echo "[ERROR] docker run failed."
        echo "${CID}"
        exit ${RC}
    fi

    echo "[INFO] Container started, id=${CID}"
    sleep 5

    STATUS=$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "unknown")
    RUNNING=$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || echo "false")
    EXIT_CODE=$(docker inspect -f '{{.State.ExitCode}}' "${CONTAINER_NAME}" 2>/dev/null || echo "unknown")
    ERROR_MSG=$(docker inspect -f '{{.State.Error}}' "${CONTAINER_NAME}" 2>/dev/null || true)

    echo "[INFO] status=${STATUS}, running=${RUNNING}, exit_code=${EXIT_CODE}"

    if [[ "${RUNNING}" != "true" ]]; then
        echo "[ERROR] Container exited shortly after start."
        if [[ -n "${ERROR_MSG}" ]]; then
            echo "[ERROR] docker state error: ${ERROR_MSG}"
        fi

        echo "========== container logs =========="
        docker logs "${CONTAINER_NAME}" 2>&1 || true
        echo "===================================="

        exit 1
    fi

    echo "[INFO] Container is running."
    echo "[INFO] Follow logs with:"
    echo "docker logs -f ${CONTAINER_NAME}"
fi
