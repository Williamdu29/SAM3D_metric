# SAM3D_metric docker配置

镜像里只固化“运行环境”，代码仓库和 checkpoints 不打进镜像，而是在运行时挂载进去

```
docker/Dockerfile      # 定义 SAM3D + CUDA + Python 环境
docker/compose.yaml    # 定义如何挂载你的本地代码和 checkpoints
docker/entrypoint.sh   # 容器启动前做环境变量和检查
docker/.env            # 存放 UID/GID、CUDA 架构等本机配置
.dockerignore          # 控制构建镜像时不要把 checkpoints 等大文件打进去
```


## 先检查服务器 GPU Docker 环境

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

验证 Docker 能看到 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## 进入擦仓库根目录

先确认官方 SAM3D 源码目录存在：

```bash
ls libs/SAM3D
ls libs/SAM3D/environments/default.yml
```

SAM3D 官方安装流程就是基于 environments/default.yml 创建环境，然后执行 pip install -e '.[dev]'、pip install -e '.[p3d]'、pip install -e '.[inference]'，并设置 PyTorch CUDA 12.1 和 Kaolin 相关安装源



## 新建 docker 目录

执行

```bash
nano docker/Dockerfile
```

写入

```dockerfile

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

ENV TZ=Etc/UTC \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:/opt/conda/envs/sam3d-objects/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
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

RUN set -eux; \
    python -m pip install --no-cache-dir "pip==23.0.1" "setuptools<70" "wheel<0.45"; \
    python -m pip --version; \
    echo "[SAM3D Docker] remove fragile nvidia-pyindex if present"; \
    grep -R "nvidia-pyindex" -n pyproject.toml requirements*.txt || true; \
    sed -i '/^[[:space:]]*nvidia-pyindex[[:space:]=<>!~].*/d;/^[[:space:]]*nvidia-pyindex[[:space:]]*$/d' requirements*.txt || true; \
    echo "[SAM3D Docker] clone MoGe manually before pip dependency resolution"; \
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
    echo "[SAM3D Docker] patch SAM3D requirements.txt: MoGe git URL -> local file path"; \
    python - <<'PY'
from pathlib import Path

req = Path("requirements.txt")
text = req.read_text()

old_lines = [
    "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b",
    "MoGe @ git+https://github.com/microsoft/moge.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b",
    "moge @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b",
    "moge @ git+https://github.com/microsoft/moge.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b",
]

for old in old_lines:
    text = text.replace(old, "MoGe @ file:///opt/third_party/MoGe")

req.write_text(text)
print("[SAM3D Docker] patched requirements.txt")
print("[SAM3D Docker] MoGe lines after patch:")
for line in req.read_text().splitlines():
    if "MoGe" in line or "moge" in line:
        print(line)
PY

RUN set -eux; \
    python -m pip install --no-cache-dir -e '.[dev]'

RUN set -eux; \
    python -m pip install --no-cache-dir -e '.[p3d]'

RUN set -eux; \
    git ls-remote https://github.com/microsoft/MoGe.git >/tmp/moge-lsremote.txt; \
    head -n 5 /tmp/moge-lsremote.txt; \
    python -m pip install --no-cache-dir -e '.[inference]'

RUN set -eux; \
    chmod +x ./patching/hydra; \
    ./patching/hydra; \
    python - <<'PY'
import torch
print("[SAM3D Docker] torch:", torch.__version__)
print("[SAM3D Docker] cuda available at build:", torch.cuda.is_available())

try:
    import pytorch3d
    print("[SAM3D Docker] pytorch3d import: ok")
except Exception as e:
    print("[SAM3D Docker] pytorch3d import failed:", repr(e))
    raise

try:
    import kaolin
    print("[SAM3D Docker] kaolin:", getattr(kaolin, "__version__", "unknown"))
except Exception as e:
    print("[SAM3D Docker] kaolin import failed:", repr(e))
    raise

try:
    import moge
    print("[SAM3D Docker] moge import: ok")
except Exception as e:
    print("[SAM3D Docker] moge import failed:", repr(e))
    raise
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
```


这里有一个关键设计：构建镜像时，Docker 会把 libs/SAM3D/ 复制到 /opt/SAM3D 并安装官方依赖；运行容器时，我们会再把你本地的 libs/SAM3D 挂载到 /opt/SAM3D，这样容器运行时看到的是你本地当前版本的 SAM3D 源码。


## 新建 `docker/entrypoint.sh`

执行：

```bash
nano docker/entrypoint.sh
```

写入

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export PYTHONPATH="/workspace:/workspace/src:/workspace/libs/SAM3D:/opt/SAM3D:${PYTHONPATH:-}"
export SAM3D_CHECKPOINT_DIR="${SAM3D_CHECKPOINT_DIR:-/workspace/checkpoints}"

echo "[SAM3D Docker] workspace: $(pwd)"
echo "[SAM3D Docker] python: $(which python)"
echo "[SAM3D Docker] checkpoint dir: ${SAM3D_CHECKPOINT_DIR}"

python - <<'PY'
import torch
print("[SAM3D Docker] torch:", torch.__version__)
print("[SAM3D Docker] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[SAM3D Docker] gpu:", torch.cuda.get_device_name(0))
PY

exec "$@"
```


赋予执行权限：

```bash
chmod +x docker/entrypoint.sh
```

## 新建 `docker/compose.yaml`

执行

```bash
nano docker/compose.yaml
```

写入：

```yaml
services:
  sam3d-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
      network: host
      args:
        USER_ID: ${HOST_UID:-1000}
        GROUP_ID: ${HOST_GID:-1000}
        TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}

    image: sam3d-api:cu121
    container_name: sam3d-api

    working_dir: /workspace

    volumes:
      # 挂载整个项目仓库：
      # 宿主机 ~/work/sam-3d-objects -> 容器 /workspace
      - type: bind
        source: ..
        target: /workspace

      # 让 /opt/SAM3D 指向你当前仓库里的 SAM3D 官方源码
      - type: bind
        source: ../libs/SAM3D
        target: /opt/SAM3D

      # 缓存 HuggingFace / Torch，避免重复下载
      - sam3d_hf_cache:/cache/huggingface
      - sam3d_torch_cache:/cache/torch

    ports:
      # 宿主机 5000 -> 容器 5000
      # 之后你用 curl http://localhost:5000/health 测试
      - "5000:5000"

    environment:
      # 项目路径
      SAM3D_PROJECT_ROOT: /workspace

      # SAM3D 官方源码路径
      SAM3D_ROOT: /workspace/libs/SAM3D

      # Flask 监听配置
      SAM3D_SERVICE_HOST: 0.0.0.0
      SAM3D_SERVICE_PORT: "5000"

      # 你的 pipeline.yaml 在宿主机 checkpoints/pipeline.yaml
      # 容器内对应 /workspace/checkpoints/pipeline.yaml
      SAM3D_CONFIG: /workspace/checkpoints/pipeline.yaml
      SAM3D_CHECKPOINT_DIR: /workspace/checkpoints

      # 输出目录
      SAM3D_OUTPUT_ROOT: /workspace/Outputs/service_outputs

      # Python import 路径
      PYTHONPATH: /workspace:/workspace/src:/workspace/libs/SAM3D:/opt/SAM3D

      # 缓存目录
      HF_HOME: /cache/huggingface
      TORCH_HOME: /cache/torch

      # 默认不在启动时预加载模型，先让服务快速起来，再手动调用 /load_model
      SAM3D_PRELOAD_MODEL: "0"

      # 根据你代码里的环境变量接口设置
      SAM3D_RELEASE_AFTER_RECON: "0"
      SAM3D_COMPILE_MODEL: "0"
      SAM3D_VERBOSE: "1"
      SAM3D_VOXEL_SIZE: "0.003"
      SAM3D_ICP_MAX_ITER: "40"
      SAM3D_MESH_POISSON_DEPTH: "8"
      SAM3D_MESH_DENSITY_Q: "0.02"

      # NVIDIA runtime 环境
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility

    shm_size: "16gb"
    ipc: host

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    command: ["python", "-u", "src/service/run_sam3d_service.py"]

volumes:
  sam3d_hf_cache:
  sam3d_torch_cache:
```

这里 context: .. 的意思是：虽然 Compose 文件在 docker/compose.yaml，但 Docker 构建上下文使用上一层，也就是你的项目根目录。Compose 支持用 -f 指定配置文件，且路径会按第一个 Compose 文件的位置作为基础路径处理；Docker Compose GPU 配置里必须设置 capabilities: [gpu]，否则部署 GPU service 会报错

## 新建 `docker/.env`

执行

```bash
cat > docker/.env <<EOF
HOST_UID=$(id -u)
HOST_GID=$(id -g)
TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0
EOF
```

如果你知道服务器显卡型号，可以把 TORCH_CUDA_ARCH_LIST 改小一点，构建会更快：

```
T4       -> 7.5
A100     -> 8.0
RTX 3090 -> 8.6
RTX 4090 -> 8.9
L4       -> 8.9
H100     -> 9.0
```

## 新建根目录 `.dockerignore`

执行：

```bash
nano .dockerignore
```

写入：

```dockerignore
.git
.gitignore

__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.mypy_cache/
.ruff_cache/

.venv/
venv/
env/
conda/
*.egg-info/

.DS_Store
.idea/
.vscode/

logs/
tmp/
temp/
outputs/
output/
runs/

# 权重和 checkpoint 不打进镜像，运行时通过 volume 挂载
checkpoints/

# 常见大文件输出
*.ply
*.obj
*.glb
*.gltf
*.npz
*.npy
*.pth
*.pt
*.ckpt

# 保留 Docker 部署文件
!docker/
!docker/**

# 保留官方 SAM3D 源码，构建环境时需要
!libs/
!libs/SAM3D/
!libs/SAM3D/**
```

## 检查 Flask 服务监听地址

你的 `src/service/run_sam3d_service.py` 必须让 Flask 监听 `0.0.0.0`，不能只监听 `127.0.0.1`。

如果你现在是：

```python
app.run(host="127.0.0.1", port=5000)
```

请改成：

```py
app.run(host="0.0.0.0", port=5000)
```

否则服务在容器里启动了，宿主机也可能访问不到。

## 确认 checkpoints 里到底哪个是 pipeline.yaml

在宿主机项目根目录执行：

```bash
find checkpoints -maxdepth 3 -name "pipeline.yaml" -print
```

如果输出是：

```
checkpoints/pipeline.yaml
```

那么 Docker 里对应路径是：

```
/workspace/checkpoints/pipeline.yaml
```

先记住这个路径，下一步要写进 `docker/compose.yaml`


## 修改 `docker/compose.yaml`


打开：

```bash
nano docker/compose.yaml
```

把 ports 和 environment 改成下面这种。这里我统一使用 5000 作为容器内部端口和宿主机访问端口

```yaml
services:
  sam3d-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
      args:
        USER_ID: ${HOST_UID:-1000}
        GROUP_ID: ${HOST_GID:-1000}
        TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}

    image: sam3d-api:cu121
    container_name: sam3d-api

    working_dir: /workspace

    volumes:
      # 挂载整个项目仓库：
      # 宿主机 ~/work/sam-3d-objects -> 容器 /workspace
      - type: bind
        source: ..
        target: /workspace

      # 让 /opt/SAM3D 指向你当前仓库里的 SAM3D 官方源码
      - type: bind
        source: ../libs/SAM3D
        target: /opt/SAM3D

      # 缓存 HuggingFace / Torch，避免重复下载
      - sam3d_hf_cache:/cache/huggingface
      - sam3d_torch_cache:/cache/torch

    ports:
      # 宿主机 5000 -> 容器 5000
      # 之后你用 curl http://localhost:5000/health 测试
      - "5000:5000"

    environment:
      # 项目路径
      SAM3D_PROJECT_ROOT: /workspace

      # SAM3D 官方源码路径
      SAM3D_ROOT: /workspace/libs/SAM3D

      # Flask 监听配置
      SAM3D_SERVICE_HOST: 0.0.0.0
      SAM3D_SERVICE_PORT: "5000"

      # 你的 pipeline.yaml 在宿主机 checkpoints/pipeline.yaml
      # 容器内对应 /workspace/checkpoints/pipeline.yaml
      SAM3D_CONFIG: /workspace/checkpoints/pipeline.yaml
      SAM3D_CHECKPOINT_DIR: /workspace/checkpoints

      # 输出目录
      SAM3D_OUTPUT_ROOT: /workspace/Outputs/service_outputs

      # Python import 路径
      PYTHONPATH: /workspace:/workspace/src:/workspace/libs/SAM3D:/opt/SAM3D

      # 缓存目录
      HF_HOME: /cache/huggingface
      TORCH_HOME: /cache/torch

      # 默认不在启动时预加载模型，先让服务快速起来，再手动调用 /load_model
      SAM3D_PRELOAD_MODEL: "0"

      # 根据你代码里的环境变量接口设置
      SAM3D_RELEASE_AFTER_RECON: "0"
      SAM3D_COMPILE_MODEL: "0"
      SAM3D_VERBOSE: "1"
      SAM3D_VOXEL_SIZE: "0.003"
      SAM3D_ICP_MAX_ITER: "40"
      SAM3D_MESH_POISSON_DEPTH: "8"
      SAM3D_MESH_DENSITY_Q: "0.02"

      # NVIDIA runtime 环境
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility

    shm_size: "16gb"
    ipc: host

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    command: ["python", "-u", "src/service/run_sam3d_service.py"]

volumes:
  sam3d_hf_cache:
  sam3d_torch_cache:
```

第一，Docker Compose 可以用 environment 给容器注入环境变量，所以我们不需要把所有路径硬编码到 Python 文件里。

第二，GPU 配置里 capabilities: [gpu] 是必须的，Docker 官方文档明确说不设置这个字段会导致部署报错。

第三，因为你的 Compose 文件放在 docker/compose.yaml，后面运行命令会用 -f docker/compose.yaml 指定配置文件。Docker Compose 官方支持用 -f 指定不在当前目录的 Compose 文件。


## 构建镜像

在项目根目录：

```bash
docker compose --env-file docker/.env -f docker/compose.yaml build
```

如果你想看完整构建日志：

```bash
docker compose --env-file docker/.env -f docker/compose.yaml build --progress=plain
```

## 启动服务

```bash
docker compose --env-file docker/.env -f docker/compose.yaml up
```

正常你应该看到类似：

```
[SAM3D Docker] workspace: /workspace
[SAM3D Docker] checkpoint dir: /workspace/checkpoints
[SAM3D Docker] cuda available: True
Starting SAM3D reconstruction service ... host=0.0.0.0, port=5000
```

如果你想后台启动：

```bash
docker compose --env-file docker/.env -f docker/compose.yaml up -d
```

查看日志：

```bash
sudo docker compose --progress plain --env-file docker/.env -f docker/compose.yaml build
```