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

## 第一步：实例化类

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