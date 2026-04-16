# AIGC 3D Reconstruction Model

## V1

## Overall Idea of the V1 Strategy

This project implements an **anchor-driven metric-scale recovery pipeline** inspired by **Any6D**.

All the updated and useful code is in **metric_V1** foldder. Check `README_V1.md` for help.

The core idea is:
1. Recover a **visible point cloud with real physical size** from a single **RGB-D anchor image**.
2. Use **SAM3D** to generate a **normalized full-object point cloud**.
3. Gradually align the SAM3D result to real-world scale through:
   - rotation candidate search,
   - isotropic scaling,
   - translation initialization,
   - rigid ICP,
   - projection constraints, and
   - PCA-based principal-axis dimension correction.

In short, the pipeline uses:

- **Anchor input** to provide **metric scale**
- **SAM3D output** to provide **complete geometry**
- **Metric alignment** to combine the two

This is conceptually close to Any6D: starting from a single RGB-D anchor, the method aligns an image-to-3D normalized shape into a **metric-scale object shape** for downstream **pose and size estimation**.

---

## Pipeline Overview

The pipeline has two branches:

### 1. Anchor branch
**Input:** `RGB + Depth + Mask + K`

This branch does **not** aim to reconstruct the full object.  
Instead, it extracts a **visible surface point cloud in the real-world coordinate system** and estimates scale-related anchors such as:

- object width in the image,
- object height in the image,
- depth to the object,
- approximate real-world size.

### 2. SAM3D branch
**Input:** `run_sam3d(rgb, mask)`

This branch generates a **3D point cloud from a single RGB image and mask**.  
However, the result has the following properties:

- the overall shape is relatively complete,
- the coordinate frame is not strictly aligned with the camera frame,
- the scale is **normalized**, not metric.

This is consistent with the typical behavior described for single-view image-to-3D models in Any6D: they can recover a full object shape, but the output is usually in a normalized space and cannot be directly used for metric pose or size estimation.

---

## Pipeline Details

## 1. Building the Metric-Scale Anchor

### 1.1 Mask preprocessing
The input mask is processed with **morphological opening and closing**, and only the **largest connected component** is kept.

Purpose:
- stabilize the foreground region,
- suppress depth holes,
- reduce edge noise,
- remove small fragments that may disturb later size estimation.

### 1.2 Depth unit check
The pipeline first determines the unit of the depth map.

### 1.3 Estimating anchor scale targets
The mask is **adaptively eroded** to avoid unstable boundaries and reduce background leakage.

Within the eroded mask, the pipeline computes the **median depth**:

```text
z_med
```

This represents the approximate distance from the object to the camera.

Then the **pinhole camera model** is used to convert pixel width and height into approximate metric size:

```text
metric_w = px_w * z_med / fx
metric_h = px_h * z_med / fy
```

This is close in spirit to the public description of Any6D, which first obtains a coarse object size estimate and then jointly refines pose and size, although the implementation here is more engineering-oriented and does not reproduce the full render-and-compare optimizer from the paper.

### 1.4 Generating a metric point cloud from the RGB-D anchor
The RGB-D anchor is converted into a **visible point cloud with real physical scale**.

---

## 2. SAM3D Output: Complete Shape Without True Scale

`run_sam3d()` feeds the RGB image and mask into the model and finally saves the result using `save_ply()`.

By default, the generated point cloud has three properties:

- the object shape is relatively complete,
- the coordinate system is not strictly aligned with the camera system,
- the scale is normalized instead of real.

So the SAM3D output is treated as a **normalized object shape**, not a metric one.

---

## 3. Searching for the Best Rotation + Scale + Translation Initialization

### 3.1 Downsampling and outlier removal
Both the anchor point cloud and the SAM3D point cloud undergo:

- voxel downsampling,
- outlier removal.

This improves the stability of later steps such as:

- PCA,
- ICP,
- Chamfer-distance evaluation.

### 3.2 PCA frame construction
`compute_pca_frame()` is applied to both the anchor point cloud and the SAM3D point cloud.

Then the initial rotation is constructed as:

```text
base_R = anchor_axes @ sam_axes.T
```

This rotates the principal axes of the SAM3D shape toward the anchor principal axes.

Because PCA has **sign ambiguity**, the pipeline uses `generate_sign_flip_rotations()` to enumerate **four sign-flip variants**.

This means:
- first perform a coarse pose alignment using geometric principal axes,
- then enumerate several symmetric candidates,
- so that the direction signs of the principal axes are not accidentally flipped.

### 3.3 Initial scale from anchor width and height
For each rotation candidate, the pipeline computes the robust XY extent of the rotated SAM3D point cloud in camera coordinates:

```text
full_xy_raw
```

Then it solves for the initial scale using:

```text
scale0 = solve_scale_xy(target_xy, full_xy_raw)
```

This scale is intended to make the XY dimensions of SAM3D match the real metric width and height estimated from the anchor.

After that, the pipeline further multiplies the scale by several perturbation factors:

```text
[0.94, 0.98, 1.00, 1.02, 1.06]
```

The purpose is to avoid trusting a single closed-form scale solution and instead perform a small local search around it.

---

## 4. Rigid Alignment Using Only the Visible Surface

### 4.1 Extracting the camera-visible surface
The full SAM3D point cloud is converted into a **camera-visible surface** using:

```text
extract_visible_surface_from_camera()
```

The underlying implementation is based on Open3D's `hidden_point_removal`.

The camera is assumed to be located at the origin, and both the anchor point cloud and later projections are expressed in the camera coordinate system.

Why do this?

- The anchor point cloud comes from the depth map, so it only contains the **visible side** of the object.
- The SAM3D output is a **full object**.

If the full SAM3D point cloud is directly aligned to the anchor with ICP, the hidden back side can interfere with registration.

Therefore, the pipeline first extracts the visible surface from SAM3D and then aligns:

- **anchor:** visible surface with metric scale
- **SAM3D visible:** visible surface with normalized scale

This becomes a **visible-to-visible registration** problem.

### 4.2 Translation initialization from visible centers
`translation_from_visible_center()` is used for translation initialization:

- **XY:** align the center of the visible SAM surface to the center of the anchor
- **Z:** align the median depth of the visible SAM surface to:

```text
target_z = z_med
```

This roughly places the scaled SAM3D visible surface at the real position of the anchor before ICP refinement.

### 4.3 Rigid ICP refinement
`icp_refine_rigid()` performs **rigid ICP only**.

It does **not** use:
- non-rigid deformation,
- anisotropic scaling.

Two explicit design principles are enforced:

```text
never use anisotropic scaling
Thickness/Z is never used for scale
```

The reason is clear:
- the object's learned shape proportions should be preserved,
- only global **isotropic scaling** is allowed.

If anisotropic scaling were allowed, the visible surface might fit the anchor better, but the full shape learned by SAM3D would become distorted. In that case, the mesh dimensions might look correct while the object geometry becomes unrealistic.

---

## 5. Three-Stage Scale Correction

### 5.1 Full-point-cloud XY correction: `s_corr_xy`
After rigid ICP, the refined full point cloud `sam_refined_full` is evaluated again.

Its robust width and height in the camera XY plane are measured:

```text
full_xy_after_icp
```

Then `solve_scale_xy()` is applied again to obtain a correction factor.

This checks whether the full object still matches the anchor-derived metric width and height after initial scaling and rigid ICP.

The correction is clipped to:

```text
[0.94, 1.06]
```

So this is only a **small correction**, not a complete re-scaling.

### 5.2 Conservative projection-bbox correction: `s_corr_proj`
The corrected visible point cloud `sam_corr_vis` is projected back into the image to obtain the predicted mask bounding box size:

```text
proj_px_wh
```

This is then compared to the original anchor mask bounding box:

```text
target_px_wh
```

The resulting correction factor `s_corr_proj` is heavily constrained:

```text
[0.97, 1.00]
```

This means:

- it can only shrink,
- it can never enlarge.

In other words, the projection correction is **one-sided and conservative**.

Why so conservative?

Because the projected bounding box is very sensitive to:
- occlusion,
- mask boundaries,
- dilation kernels.

If projection were allowed to dominate the scale, the object could easily become larger and larger during correction. Therefore, projection is only used as a conservative regularizer to prevent contour overflow, not as the main source of scale.

### 5.3 Object-frame correction using principal-axis dimensions: `s_corr_obj`
Finally, `robust_pca_extents()` is applied to `sam_mid_full` to measure the object size along the first three principal directions in the object frame.

Then `solve_scale_object_dims()` is used to match these dimensions to `target_xy`.

Why do this?

Earlier correction `s_corr_xy` measures width and height in the **camera frame**, but final object dimensions are often measured in tools such as MeshLab along the **object's own principal directions**.

So it is more reasonable to perform a final adjustment in the **object frame**.

The height-related constraint is given a larger weight:

```text
ww = 0.18
wh = 0.82
```

This reflects the empirical observation that, for many everyday objects in single-view settings, the height-like major direction is more stable than the width.

---

## 6. Multi-Metric Scoring and Best-Candidate Selection

Each candidate is scored using `score_alignment()`.

The final score is **not** based only on ICP. It combines multiple metrics:

- full-point-cloud width/height error (`size_score`)
- symmetric Chamfer distance
- mask IoU
- depth MAE
- coverage
- ICP RMSE / fitness

So the pipeline is not searching for the candidate with the smallest nearest-point registration error only.  
Instead, it seeks the best overall solution that is jointly consistent in:

- geometric dimensions,
- 2D projection,
- depth agreement,
- visible-surface overlap.

This is also aligned with the spirit of Any6D, where object alignment is not judged only in 3D but also by 2D/3D consistency.

---

## 7. Output: Full / Visible Point Clouds and Mesh Reconstruction

The pipeline saves:

- `full_final`: full point cloud at metric scale
- `visible_final`: visible-surface point cloud at metric scale

Then `reconstruct_mesh_from_pcd()` is used to perform **Poisson reconstruction** from the metric point cloud.

### Important note
The mesh reconstruction stage itself does **not** introduce any extra scale-alignment module.

It only reconstructs a surface from an already metric-aligned point cloud.

Although `calibrate_mesh_to_pointcloud()` exists in the codebase, it is not actually called inside `main()`. Instead:

```text
mesh_iso_corr = 1.0
```

is fixed directly.

So, in the current version, **metric-scale recovery happens mainly at the point-cloud stage, not at the mesh stage**.

---

## V1 Results

| Object | Ground Truth Size | Reconstruction Time |
|---|---|---:|
| Pretz | 16.0 × 8.4 × 2.5 | - |
| Sea Salt Soda Crackers | 18.8 × 9.0 × 7.1 | - |
| Vitasoy | 10.5 × 6.3 × 4.0 | - |
| Oreo | 21.0 × 5.5 × 4.8 | — |
| Tissue Pack | 8.2 × 6.5 | — |
| Mobile Phone | 16.3 × 8.0 | — |
| Jasmine Honey Tea | 21 × 6 | — |

---

## Limitations of V1

Both SAM3D and Any6D are **monocular scale-estimation models**, so frame selection is very important.

The current principles for selecting the reconstruction frame are:

- keep the object as close to the image center as possible,
- make the grasped edge as close as possible to `cos = 0` relative to the camera,
- under the above constraint, try to expose two visible faces,
- minimize distortion along the height direction,
- ensure the object does not go outside the image,
- avoid reflective objects.

### 1. Approximate anchor size estimation
`metric_w` and `metric_h` are still fundamentally approximated from:
- the 2D bounding box,
- a single representative depth value.

This works relatively well for:
- near-frontal views,
- objects with limited thickness.

But it can introduce systematic error for:
- oblique views,
- slender structures,
- objects with strong perspective foreshortening.

### 2. Thickness is intentionally excluded from scale estimation
The real metric scale is mainly constrained by:

- XY size,
- bounding box,
- the first two principal-axis dimensions.

Thickness **Z** is explicitly not used.

This avoids contamination from thickness noise, but for objects where thickness is the key dimension, the scale recovery can become weaker.

### 3. Engineering-oriented refinement rather than full Any6D reproduction
This method can be viewed as:

> **Any6D-style coarse metric alignment + engineering-oriented refinement**

It is **not** a full reproduction of Any6D's joint pose-size render-and-compare optimization.

This trade-off was made mainly for implementation efficiency and time cost.





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





