# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ARG MAX_JOBS=4

ENV TZ=Etc/UTC \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:/opt/conda/envs/sam3d-objects/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121" \
    PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html" \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    pkg-config \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libegl1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global http.version HTTP/1.1 \
    && git config --global http.lowSpeedLimit 1000 \
    && git config --global http.lowSpeedTime 300 \
    && git config --global core.compression 0

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba

WORKDIR /opt/SAM3D

COPY libs/SAM3D/ /opt/SAM3D/

RUN micromamba env create -y -f /opt/SAM3D/environments/default.yml \
    && micromamba clean -a -y

SHELL ["micromamba", "run", "-n", "sam3d-objects", "/bin/bash", "-lc"]

# 1) 固定 pip，避免 nvidia-pyindex 老包和新版 pip 冲突。
# 2) 在任何 pip install -e 之前，先 patch requirements，避免 editable metadata 缓存旧依赖。
# 3) MoGe 改成本地路径，避免 pip 在安装依赖时自己 git clone。
# 4) flash_attn 从 p3d 依赖中移除，后面手动用 --no-build-isolation 安装。
RUN set -eux; \
    python -m pip install --no-cache-dir "pip==23.0.1" "setuptools<70" "wheel<0.45" packaging ninja psutil wrapt; \
    python -m pip --version; \
    echo "[SAM3D Docker] clone MoGe manually before dependency resolution"; \
    mkdir -p /opt/third_party; \
    for i in 1 2 3 4 5; do \
        git clone --filter=blob:none https://github.com/microsoft/MoGe.git /opt/third_party/MoGe && break; \
        echo "[SAM3D Docker] MoGe clone failed, retry ${i}/5"; \
        rm -rf /opt/third_party/MoGe; \
        sleep 10; \
    done; \
    cd /opt/third_party/MoGe; \
    git checkout a8c37341bc0325ca99b9d57981cc3bb2bd3e255b; \
    cd /opt/SAM3D; \
    python - <<'PY'
from pathlib import Path

def rewrite_lines(path: str, transform):
    p = Path(path)
    if not p.exists():
        print(f"[SAM3D Docker] {path} not found, skip")
        return
    lines = p.read_text().splitlines()
    new_lines = transform(lines)
    p.write_text("\n".join(new_lines) + "\n")
    print(f"[SAM3D Docker] patched {path}:")
    for line in new_lines:
        if any(key.lower() in line.lower() for key in ["moge", "nvidia-pyindex", "flash_attn", "flash-attn"]):
            print("  ", line)

def patch_requirements(lines):
    out = []
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # nvidia-pyindex 只是添加 NVIDIA pip index；Dockerfile 已经通过 PIP_EXTRA_INDEX_URL 设置了。
        if lower == "nvidia-pyindex==1.0.9" or lower.startswith("nvidia-pyindex"):
            continue

        # MoGe 改成本地 file URL，避免 pip install 时实时 git clone。
        if "moge @ git+https://github.com/microsoft/moge.git" in lower:
            out.append("MoGe @ file:///opt/third_party/MoGe")
            continue

        out.append(line)
    return out

def patch_p3d(lines):
    out = []
    for line in lines:
        lower = line.strip().lower().replace("-", "_")
        # flash_attn 手动提前安装，不让 pip 在 .[p3d] 里走隔离构建。
        if lower.startswith("flash_attn"):
            continue
        out.append(line)
    return out

rewrite_lines("requirements.txt", patch_requirements)
rewrite_lines("requirements.p3d.txt", patch_p3d)

# 删除可能由历史构建残留的 metadata，避免 p3d 继续读取旧依赖。
for pattern in ["*.egg-info", "build", "dist"]:
    for p in Path(".").glob(pattern):
        if p.is_dir():
            import shutil
            shutil.rmtree(p)
        else:
            p.unlink()
PY

# 先装 torch 栈。注意：必须早于 flash-attn 和 p3d。
RUN set -eux; \
    echo "[SAM3D Docker] install torch stack explicitly"; \
    python -m pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121; \
    python - <<'PY'
import torch
print("[SAM3D Docker] torch:", torch.__version__)
print("[SAM3D Docker] torch cuda version:", torch.version.cuda)
print("[SAM3D Docker] cuda available during build:", torch.cuda.is_available())
PY

# 单独安装 flash-attn，并禁用 build isolation。
# 如果机器内存不够，把 compose 里的 MAX_JOBS 改成 2。
RUN set -eux; \
    echo "[SAM3D Docker] install flash-attn without build isolation"; \
    python -m pip install --no-cache-dir packaging ninja psutil wrapt; \
    MAX_JOBS="${MAX_JOBS}" python -m pip install --no-cache-dir flash-attn==2.8.3 --no-build-isolation; \
    python - <<'PY'
import flash_attn
print("[SAM3D Docker] flash_attn import: ok")
PY

# 安装 SAM3D core + dev。
RUN set -eux; \
    echo "[SAM3D Docker] install sam3d_objects [dev]"; \
    rm -rf /opt/SAM3D/*.egg-info /opt/SAM3D/build /opt/SAM3D/dist; \
    python -m pip install --no-cache-dir -e '.[dev]'

# 安装 p3d。此时 requirements.p3d.txt 里应该只剩 pytorch3d。
RUN set -eux; \
    echo "[SAM3D Docker] verify requirements.p3d.txt before p3d install"; \
    cat requirements.p3d.txt; \
    if grep -Ei "flash[_-]attn" requirements.p3d.txt; then \
        echo "[SAM3D Docker] ERROR: flash_attn still exists in requirements.p3d.txt"; \
        exit 1; \
    fi; \
    rm -rf /opt/SAM3D/*.egg-info /opt/SAM3D/build /opt/SAM3D/dist; \
    python -m pip install --no-cache-dir -e '.[p3d]'; \
    python - <<'PY'
import torch
import pytorch3d
import flash_attn
print("[SAM3D Docker] p3d stage ok")
print("[SAM3D Docker] torch:", torch.__version__)
print("[SAM3D Docker] pytorch3d import: ok")
print("[SAM3D Docker] flash_attn import: ok")
PY

# 安装 inference。Kaolin 通过 PIP_FIND_LINKS 指向 torch-2.5.1_cu121 wheel。
RUN set -eux; \
    echo "[SAM3D Docker] install sam3d_objects [inference]"; \
    rm -rf /opt/SAM3D/*.egg-info /opt/SAM3D/build /opt/SAM3D/dist; \
    python -m pip install --no-cache-dir -e '.[inference]'

# 官方 hydra patch + 最终导入验证。
RUN set -eux; \
    chmod +x ./patching/hydra; \
    ./patching/hydra; \
    python - <<'PY'
import torch
print("[SAM3D Docker] torch:", torch.__version__)
print("[SAM3D Docker] torch cuda version:", torch.version.cuda)
print("[SAM3D Docker] cuda available at build:", torch.cuda.is_available())

import pytorch3d
print("[SAM3D Docker] pytorch3d import: ok")

import flash_attn
print("[SAM3D Docker] flash_attn import: ok")

import kaolin
print("[SAM3D Docker] kaolin:", getattr(kaolin, "__version__", "unknown"))

import moge
print("[SAM3D Docker] moge import: ok")
PY

SHELL ["/bin/bash", "-lc"]

COPY docker/entrypoint.sh /opt/entrypoint.sh

RUN chmod +x /opt/entrypoint.sh \
    && groupadd -g ${GROUP_ID} appuser \
    && useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash appuser \
    && mkdir -p /workspace /cache/huggingface /cache/torch \
    && chown -R appuser:appuser /workspace /cache /opt/entrypoint.sh

USER appuser
WORKDIR /workspace

ENV HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    PYTHONPATH=/workspace:/workspace/src:/workspace/libs/SAM3D:/opt/SAM3D

EXPOSE 5000

ENTRYPOINT ["/opt/entrypoint.sh"]

CMD ["python", "-u", "src/service/run_sam3d_service.py"]