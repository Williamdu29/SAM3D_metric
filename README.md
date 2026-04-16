# SAM3DReconstructor

`SAM3DReconstructor` 是一个用于 **单物体 3D 重建** 的封装类。



1. **实例化类**：完成配置初始化、路径准备、SAM3D 模型加载
2. **调用 `recon()`**：输入 RGB / Depth / Mask / K，自动完成重建并输出结果文件

下游需要准备好输入数据，并按要求传参即可。

---

# 1. 类的功能

这个类会自动完成以下事情：

- 从 RGB / Depth / Mask / K 构建 anchor 点云
- 运行或加载 SAM3D 原始点云
- 做尺度对齐和刚体配准
- 输出对齐后的 metric point cloud
- 基于点云重建 mesh
- 保存指标文件
- 保存每一步耗时

---

# 2. 下游需要做两步

##  第一步：实例化类

也就是创建一个 `SAM3DReconstructor` 对象。

## 第二步：调用 `recon(...)`

把这一条样本对应的：

- RGB 图
- Depth 图
- Mask 图
- K.txt

### `SAM3DReconstructor()` 参数说明

- `sam3d_root`: SAM3D 仓库根目录，通常是项目目录，例如 `/home/dct/work/sam-3d-objects`。
- `sam3d_config`: SAM3D checkpoint pipeline 配置文件路径，例 `checkpoints/pipeline.yaml`。
- `out_dir`: 输出目录，结果点云、mesh、对齐指标和 timings 将写入该目录。
- `voxel_size`: 点云下采样体素大小，默认 `0.0025`，影响 anchor 构建与对齐精度。
- `icp_max_iter`: ICP 刚体配准最大迭代次数，默认 `40`。
- `mesh_poisson_depth`: Poisson 重建深度，默认 `8`。
- `mesh_density_q`: Poisson mesh density 截断分位数，默认 `0.02`。
- `sam_compile`: 是否编译 SAM 模型，默认 `False`，关闭可以加快初始化。
- `verbose`: 是否打印详细进度和诊断信息。

### `recon(...)` 参数说明

- `rgb_path`: RGB 图像路径。
- `depth_path`: 深度图路径。
- `mask_path`: 二值 mask 图路径，SAM3D 将仅处理该区域。
- `k_path`: 相机内参矩阵 `K.txt` 路径，加载为 `3x3` 矩阵。
- `run_sam3d`: `True` 时运行 SAM3D 推理生成原始点云；如果已有 raw SAM PLY，可设为 `False`。
- `output_prefix`: 输出文件名前缀，例如 `sam3d_metric_general`。
- `raw_sam_name`: 原始 SAM 点云 PLY 文件名，默认 `sam3d_raw_v8.ply`。
- `anchor_name`: anchor 点云文件名，默认 `anchor_metric_visible_surface_v8.ply`。
- `save_metrics_npy`: 是否保存指标 `.npy` 文件。
- `save_metrics_json`: 是否保存指标 `.json` 文件。
- `seed`: 随机种子，保证 SAM3D 推理与对齐流程可重复。

### `recon()` 返回字段说明

返回值是一个字典，常见字段包括：

- `anchor_pcd_path`: anchor 点云路径。
- `raw_sam_pcd_path`: 原始 SAM 点云路径。
- `metric_full_pcd_path`: 对齐后的 full metric 点云路径。
- `metric_visible_pcd_path`: 对齐后的 visible metric 点云路径。
- `metric_mesh_ply_path`: 生成 mesh PLY 路径。
- `metric_mesh_obj_path`: 生成 mesh OBJ 路径。
- `metrics_npy_path`: 保存的指标 `.npy` 文件路径。
- `metrics_json_path`: 保存的指标 `.json` 文件路径。
- `timings_json_path`: 每一步耗时结果路径。

---


# 3. 推荐的最小使用方式

```python
from sam3d_reconstructor import SAM3DReconstructor

reconstructor = SAM3DReconstructor(
    sam3d_root="/home/dct/work/sam-3d-objects",
    sam3d_config="/home/user/datas/hc/data/ckpts/sam3d-obj/models/checkpoints/pipeline.yaml",
    out_dir="/home/dct/work/sam-3d-objects/cookie_output",
)

result = reconstructor.recon(
    rgb_path="/mnt/ws_shard/dct/SKU/cookie/images/1775628522763.jpg",
    depth_path="/mnt/ws_shard/dct/SKU/cookie/depth/1775628522763.png",
    mask_path="/mnt/ws_shard/dct/SKU/cookie/masks/maskdata/1775628522763.png",
    k_path="/mnt/ws_shard/dct/SKU/cookie/K.txt",
)




```



# 4. Docker 环境打包

本项目提供“开箱即用”的交付方式：  
将 **SAM3D 项目代码** 和 **已经验证可运行的 Conda 环境** 一起打包进 Docker 镜像中。  
使用者无需重新创建 Python / CUDA / 依赖环境，只需要导入镜像并启动容器，即可直接运行项目。

## 1. 项目说明



当前可运行环境信息如下：

\- Conda 环境名：`sam3d-obj`
\- 项目代码目录：`/home/dct/work/sam-3d-objects/`

本仓库的交付目标是：

1. 在开发机器上，将 `sam3d-obj` 环境完整打包；
2. 将项目代码与打包后的环境一起构建到 Docker 镜像中；
3. 将 Docker 镜像导出成压缩包；
4. 其他用户拿到压缩包后，只需要解压、导入 Docker 镜像并启动容器，即可直接运行项目，无需重新配置环境。


\---

## 2. 整体流程概览

完整流程：

1. **确认本地 Conda 环境可正常运行**
2. **使用 conda-pack 打包 Conda 环境**
3. **准备 Dockerfile，把代码和环境一起放进镜像**
4. **构建 Docker 镜像并导出为交付压缩包**
5. **用户端导入镜像并运行项目**


\---

## 3. 确认本地环境可运行

### 3.1 激活环境

```bash
conda activate sam3d-obj
```

### 3.2 进入项目目录

```bash
cd /home/dct/work/sam-3d-objects/
```

### 3.3 运行前向推理

```bash
python3 /home/dct/work/sam-3d-objects/metric_V1/run_V1.5.py
```

## 4. 使用 conda-pack 打包现有 Conda 环境

```bash
conda install -c conda-forge conda-pack
```

### 4.2 打包环境

建议先切换到一个临时的输出目录，例如：

```bash
mkdir -p /home/dct/work/sam3d-docker-build/
cd /home/dct/work/sam3d-docker-build/
```

然后再执行打包

```bash
conda pack -n sam3d-obj -o sam3d-obj.tar.gz
```

执行完打包后，会生成文件 `sam3d-obj.tar.gz`，这个就是打包后的conda环境归档。



### 4.3 打包完环境后建议检查

检查文件是否存在

```bash
ls -lh sam3d-obj.tar.gz
```



## 5. 准备Docker构建目录

建议新建一个专门用于制作镜像的目录，例如

```bash
mkdir -p /home/dct/work/sam3d-docker-build/docker_package/
cd /home/dct/work/sam3d-docker-build/docker_package/
```

在目录中准备以下文件

```
docker_package/
├── Dockerfile
├── sam3d-obj.tar.gz
└── sam-3d-objects/
```

#### 5.1 复制项目源代码

```bash
cp -r /home/dct/work/sam-3d-objects ./sam-3d-objects
```

#### 5.2 复制conda打包文件

```bash
cp /home/dct/work/sam3d-docker-build/sam3d-obj.tar.gz ./
```



## 6. 编写dockerfile

```dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bash \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 复制 Conda 环境压缩包
COPY sam3d-obj.tar.gz /opt/

# 解压环境到固定目录
RUN mkdir -p /opt/conda_env && \
    tar -xzf /opt/sam3d-obj.tar.gz -C /opt/conda_env && \
    rm /opt/sam3d-obj.tar.gz

# 运行 conda-unpack 修复路径
RUN /opt/conda_env/bin/conda-unpack

# 复制项目代码
COPY sam-3d-objects /workspace/sam-3d-objects

WORKDIR /workspace/sam-3d-objects

# 设置环境变量，使容器默认使用打包好的 Python 环境
ENV PATH=/opt/conda_env/bin:$PATH

# 默认进入 bash，也可以改成你的启动命令
ENTRYPOINT ["/bin/bash"]
```

##  7. 构建Docker镜像

在```docker_package/```目录下执行：

```bash
docker build -t sam3d:latest .
```

构建完成后可以查看镜像：

```bash
docker images | grep sam3d
```

如果显示：```sam3d:latest```，则说明镜像构建成功

## 8. 本地测试镜像是否可运行

交付之前必须先测试自己的镜像

### 8.1 启动容器

```bash
docker run --rm -it sam3d:latest
```

进入容器后，执行：

```bash
cd /workspace/sam-3d-objects
python --version
which python
```

预期看到：

python指向 ```/opt/conda_env/bin/python```

Python 版本与原环境一致

### 8.2 测试项目启动

在容器中执行项目启动命令

```bash
python3 /home/dct/work/sam-3d-objects/metric_V1/run_V1.5.py
```



## 9. 导出docker镜像给其他用户

为了方便别人直接使用，可以把镜像导出成 tar 文件，再压缩发送。

### 9.1 导出镜像

```bash
docker save -o sam3d_docker_image.tar sam3d:latest
```



### 9.2 压缩并交付

```bash
gzip sam3d_docker_image.tar	
```

压缩后得到

```sam3d_docker_image.tar.gz```这个文件就是最终可以发给其他用户的交付包。

## 10. 用户使用方法

### 10.1 解压镜像包

```bash
gunzip sam3d_docker_image.tar.gz
```

得到：```sam3d_docker_image.tar```

### 10.2 导入docker镜像

```bash
docker load -i sam3d_docker_image.tar
```

### 10.3 启动容器

```bash
docker run --rm -it sam3d:latest
```

进入容器后：

```bash
cd /workspace/sam-3d-objects
python3 metric_V1/run_V1.5.py
```







