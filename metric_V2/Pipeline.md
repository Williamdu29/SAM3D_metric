这段代码不是单纯“跑 SAM3D 然后导出模型”，而是一个把 SAM3D 单图重建结果拉回真实相机米制坐标系的工程化 pipeline。核心思想是：

SAM3D 负责从 RGB + mask 生成完整 3D 物体形状；RGB-D 深度图负责提供真实尺度、真实距离和相机坐标约束；Open3D 点云配准负责把 SAM3D 的无尺度/弱尺度重建结果对齐到真实米制空间；最后把同一套变换应用到 SAM3D 原生 mesh 上并导出。

SAM 3D Objects 本身的定位是从单张图像和目标 mask 生成物体的 3D 模型，包括形状、纹理、姿态与布局；官方用法中，推理接口会接收 image/mask，并输出可导出的 3D 表示，例如 Gaussian splat 点云和 mesh/glb 等对象。本代码项目正是在这个基础上额外引入 RGB-D 深度和相机内参，把结果转换成真实尺度。

## 1. 整体流程总览

主函数 main() 把整个项目分成 5 步：

```python
[1/5] Build anchor data from RGB-D
[2/5] Run SAM3D
[3/5] Solve metric alignment
[4/5] Apply transform and export mesh
[5/5] Done
```

每一步的职责如下：


| 阶段 | 作用 | 输入 | 输出 |
| :--- | :--- | :--- | :--- |
| 1. RGB-D anchor 构建 | 从深度图 + mask + K 内参生成真实尺度点云 | RGB、depth、mask、K | anchor_data |
| 2. SAM3D 推理 | 从 RGB + mask 生成 SAM3D 原生 3D 结果 | RGB、mask、pipeline.yaml | raw 点云 .ply + raw mesh |
| 3. 米制对齐搜索 | 把 SAM3D 点云缩放、旋转、平移、ICP 到 RGB-D anchor 点云 | anchor 点云 + SAM3D 点云 | 最优对齐参数 best |
| 4. mesh 变换导出 | 把求出的点云对齐变换应用到 SAM3D 原生 mesh | raw mesh + best | 米制 mesh .ply |
| 5. 统计输出 | 打印时间、内存、诊断信息 | timing | 日志 |


这个设计很重要：点云只用于求变换，最终导出的 mesh 仍然来自 SAM3D 原生 mesh，而不是从点云重新重建 mesh。
这意味着保留了 SAM3D 的完整形状和纹理/mesh 拓扑，同时用 RGB-D 数据修正了真实尺度和位姿。


## 2. 输入配置部分

代码开头定义了几个路径：

```python
RGB_PATH = "/mnt/dct/data/replay/mouse/images/62.png"
DEPTH_PATH = "/mnt/dct/data/replay/mouse/depths/62.png"
MASK_PATH = "/mnt/dct/data/replay/mouse/masks/maskdata/62.png"
K_PATH = "/mnt/dct/SKU/cookie/K.txt"
```

```python
SAM3D_ROOT = "/home/dct/work/sam-3d-objects"
SAM3D_CONFIG = "/home/user/datas/hc/data/ckpts/sam3d-obj/models/checkpoints/pipeline.yaml"
```

直接把 SAM3D repo 的 notebook 目录加入了 Python 路径：

```python
sys.path.append(os.path.join(SAM3D_ROOT, "notebook"))
from inference import Inference, load_image
```

SAM3D 官方推理流程也是以 Inference 类作为主要入口，读取 pipeline.yaml 和 checkpoints，然后通过 Inference.__call__() 处理 image/mask，生成输出字典。本项目代码和这个设计是一致的。




## 3. 第一阶段：构建 RGB-D anchor 点云

```python
anchor_data = build_anchor_data(RGB_PATH, DEPTH_PATH, MASK_PATH, K_PATH)
```

这一步是整个代码最关键的“真实尺度来源”。SAM3D 从单张 RGB 图像推 3D，本身很难保证绝对物理尺寸。这里用深度图和相机内参生成一个真实米制点云，作为后续对齐目标。

```python
mask_bin = morph_clean((mask > 0).astype(np.uint8), ksize=5)
```

1. 开运算：去除小噪声；
2. 闭运算：填补小孔洞；
3. 保留最大连通域：只保留最大的目标区域。


### 核心函数：

```python
scale_targets = estimate_anchor_scale_targets(mask_bin, depth_m, K)
```

核心函数输出：

```python
{
    "pixel_w": px_w,
    "pixel_h": px_h,
    "z_med": z_med,
    "z_q10": z_q10,
    "z_q90": z_q90,
    "metric_w": metric_w,
    "metric_h": metric_h,
    ...
}
```

### 自适应腐蚀 mask

```py
inner_mask, erode_k, used_erode = erode_mask_adaptive(mask, ratio=0.12)
```

mask 边界附近常常有：

深度空洞；
背景混入；
物体边缘飞点；
深度相机边界噪声。

所以它先把 mask 往里收一点，只用更可靠的中心区域估计深度。

### 用内部 mask 估计深度中位数

```python
z_med = float(np.median(valid_depth_inner))
z_q10 = float(np.percentile(valid_depth_inner, 10))
z_q90 = float(np.percentile(valid_depth_inner, 90))
```

这里不用平均值，而是用中位数，是为了抗噪声。

深度中位数 z_med 后面有三个用途：

估算目标真实宽高；
作为目标物体的中心深度；
在深度图中过滤离群点。

### 用 bbox 像素尺寸 + 深度 + 内参估算真实宽高

```python
metric_w = px_w * z_med / fx
metric_h = px_h * z_med / fy
```

> 这个公式来自针孔相机模型

所以一个物体在图像中的像素宽度 px_w，在深度 Z 处对应的真实宽度约为：

```python
真实宽度 = 像素宽度 × 深度 / fx
```


这一步得到的 metric_w、metric_h 并不一定非常精确，因为它假设目标整体处在近似同一深度平面上。但对于后续做尺度初始化非常有用。


### 用深度带过滤目标点

```python
z_med = scale_targets["z_med"]
band = max(0.035 * z_med, 0.012)
valid_depth_mask = np.isfinite(depth_m) & (depth_m > 1e-6) & (np.abs(depth_m - z_med) <= band)
final_mask = morph_clean(mask_bin & valid_depth_mask.astype(np.uint8), ksize=3)
```

这一步非常关键。

它不是直接用原始 mask 里的所有深度点，而是只保留接近 z_med 的点：

也就是深度容忍带：

至少 1.2 cm；
或者目标距离的 3.5%。

这样做的目的：

1. 去掉 mask 内背景点；
2. 去掉错误深度点；
3. 去掉明显不属于目标前表面的点；
4. 构建一个更干净的 anchor 点云。

但这里也有一个隐含风险：
如果目标本身有明显厚度，或者沿深度方向变化很大，这个过滤会把真实结构的一部分删掉。

代码更偏向“用可见正面表面做配准”，而不是完整利用全部深度形状。

### 深度图反投影为 3D 点云

```python
pts, _, _ = depth_to_points(depth_m, final_mask, K)
```

核心公式

```py
X = (xs - cx) * z / fx # 描述空间中物体相对于相机的位置
Y = (ys - cy) * z / fy
Z = z
```

也就是说，最终得到的是相机坐标系下的点：

```python
x: 相机右方向或图像横向方向
y: 相机下方向或图像纵向方向
z: 相机前方深度方向
```

### anchor_data 的最终内容

```python
return {
    "rgb": rgb,
    "H": H,
    "W": W,
    "depth_m": depth_m,
    "mask_bin": mask_bin,
    "final_mask": final_mask,
    "K": K,
    "pcd": pcd,
    "pts": pts,
    "depth_mode": depth_mode,
    "scale_targets": scale_targets,
    "anchor_center": anchor_center,
    "anchor_xy_extent": anchor_xy_extent,
}
```


## 4. 第二阶段：运行 SAM3D


```python
raw_sam_path, raw_mesh, output = run_sam3d(
    RGB_PATH,
    MASK_PATH,
    SAM3D_CONFIG,
    save_ply_path=tmp_raw_ply,
)
```

### 初始化 SAM3D 推理器

```py
inference = Inference(config_path, compile=False)
```


这里加载 pipeline.yaml。根据 SAM3D 的公开说明，pipeline.yaml 是推理配置入口，Inference 会加载配置和权重，然后通过调用接口处理 image/mask。

compile=False 的含义是禁用编译优化。优点是：

启动更稳定；
debug 更方便；
兼容性更好。

缺点是可能速度慢一些。

SAM3D 输出字典里，代码强制要求两个 key：

```py
if "glb" not in output:
    raise KeyError(...)
if "gs" not in output:
    raise KeyError(...)
```

其中

```py
raw_mesh = output["glb"]
output["gs"].save_ply(save_ply_path)
```


output["glb"]：SAM3D 原生 mesh；
output["gs"]：Gaussian splat 或点云式表示，可以 save_ply()。

SAM3D 官方/文档里也提到输出字典会映射 "gs" 到 GaussianSplat 对象，并提供 save_ply() 这类导出方法。


### 为什么要保存 output["gs"] 为临时 PLY

因为后面的配准过程需要 Open3D 点云

```py
sam_pcd = load_pcd(raw_sam_path)
```

而 Open3D 可以方便地读取 .ply 点云。
所以代码是：

SAM3D 输出 gs；
gs 保存成 .ply；
Open3D 读取 .ply；
用这个点云做尺度和位姿对齐；
对齐参数再应用回 output["glb"] mesh。

这是一个非常合理的桥接方式。



## 5. 第三阶段：米制尺度和姿态对齐

```py
best = search_metric_alignment(anchor_data, sam_pcd)
```

它的目标是求一套变换，把 SAM3D 点云从“模型自身坐标系”转换到“真实相机米制坐标系”。

完整变换大致包括：

```
SAM3D 原始点云
→ PCA 初始旋转
→ 初始尺度缩放
→ 初始平移到 anchor 附近
→ ICP 刚性精配准
→ 多种尺度修正
→ 最终平移校正
→ 得到 full_final / visible_final
```

### anchor 点云和 SAM 点云降采样

```py
VOXEL_SIZE = 0.0025
```

也就是 2.5 mm。

体素降采样的目的：

1. 降低点数；
2. 加快 ICP；
3. 降低噪声；
4. 让点云密度更均匀。

然后又分别去除离群点：

```py
anchor_pts = remove_outliers_np(anchor_pts, nb_neighbors=20, std_ratio=1.2)
sam_pts = remove_outliers_np(sam_pts, nb_neighbors=20, std_ratio=1.5)
```


### 确定目标宽高和目标深度


```py
target_xy = np.array([
    anchor_data["scale_targets"]["metric_w"],
    anchor_data["scale_targets"]["metric_h"],
], dtype=np.float64)

target_z = float(anchor_data["scale_targets"]["z_med"])
```

这里的 target_xy 是从 RGB-D 估出来的真实宽高。
target_z 是目标中心深度。

后续所有缩放和平移都围绕这几个量展开。


### 用 PCA 求初始姿态

```py
anchor_pca_center, anchor_axes, _ = compute_pca_frame(anchor_pts)
sam_center, sam_axes, _ = compute_pca_frame(sam_pts)
base_R = anchor_axes @ sam_axes.T
```

compute_pca_frame() 做的是：

1. 取点云 robust center；
2. 计算协方差矩阵；
3. 求特征值和特征向量；
4. 按特征值从大到小排序；
5. 得到主轴方向。

代码：

```py
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(evals)[::-1]
evecs = evecs[:, order]
```


PCA 的含义：

第一主轴：点云最长变化方向；
第二主轴：次长方向；
第三主轴：最短方向。



### 生成4个符号翻转旋转

```py
candidate_Rs = generate_sign_flip_rotations(base_R)
```

PCA 有一个问题：
特征向量的方向是不唯一的。

也就是说，某个主轴可以是 +x，也可以是 -x，数学上都成立。
所以需要尝试不同符号翻转


### 初始尺度估计 `solve_scale_xy`



对于每个候选旋转:

```py
sam_rot_full = centered_similarity(sam_pts, scale=1.0, R=R, center=sam_center)
full_xy_raw, _, _ = robust_extent_xy(sam_rot_full, low=2.0, high=98.0)
scale0 = solve_scale_xy(target_xy, full_xy_raw)
```

这里先把 SAM 点云旋转到 anchor PCA 坐标方向，然后计算它在 XY 平面上的 robust 宽高。


solve_scale_xy() 求一个统一尺度 s，让 SAM 的 XY 宽高尽量接近 RGB-D 估计的目标宽高

```py
def solve_scale_xy(target_xy, src_xy, ww=0.32, wh=0.68):
    target_w, target_h = float(target_xy[0]), float(target_xy[1])
    src_w, src_h = float(src_xy[0]), float(src_xy[1])
    num = ww * src_w * target_w + wh * src_h * target_h
    den = ww * src_w * src_w + wh * src_h * src_h + 1e-12
    s = float(num / den)
    return max(s, 1e-8)
```

这是一个加权最小二乘形式，目标是最小化：

`ww * (s * src_w - target_w)^2 + wh * (s * src_h - target_h)^2`

默认：`ww=0.32, wh=0.68`

说明更相信高度方向，或者说希望高度匹配权重更大。


### 多尺度扰动搜索

```py
for mult in [0.94, 0.98, 1.00, 1.02, 1.06]:
    s = scale0 * mult
```

没有只用一个尺度，而是在初始尺度附近搜索 5 个候选：

0.94
0.98
1.00
1.02
1.06

这是为了避免初始 bbox 尺度估计不准。

这个设计很工程化：
PCA + bbox 得到的尺度只是粗估，后面通过搜索和评分选最优。


### 提取从相机可见的 SAM 表面

```py
sam_scaled_vis = extract_visible_surface_from_camera(sam_scaled_full)
```

RGB-D anchor 点云只来自相机可见表面，而 SAM3D 的 full 点云/mesh 可能包含完整物体，包括背面和不可见部分。
如果直接拿完整 SAM 点云和 RGB-D 可见点云配准，会产生偏差。

### 初始平移：让可见表面中心对齐 anchor

```py
vis_center = robust_center(visible_pts)
return np.array([
    anchor_center[0] - vis_center[0],
    anchor_center[1] - vis_center[1],
    target_z - np.median(visible_pts[:, 2]),
])
```

1. x 方向对齐 anchor center；
2. y 方向对齐 anchor center；
3. z 方向让 SAM 可见表面的中位深度对齐目标深度 target_z。


```py
sam_init_full = sam_scaled_full + t0
sam_init_vis = sam_scaled_vis + t0
```

这样得到一个初始落在相机米制空间附近的 SAM 点云。


### ICP 精配准

```py
threshold = max(0.008, 4.0 * VOXEL_SIZE)
sam_refined_vis, T_icp, fitness, rmse = icp_refine_rigid(
    sam_init_vis,
    anchor_pts,
    threshold
)
```

ICP 阈值使用 1 cm

点到点 ICP：`TransformationEstimationPointToPoint()`

输出：

T_icp：4×4 刚性变换矩阵；
fitness：匹配比例；
rmse：内点均方根误差。

然后将同一个 ICP 变换应用到 full 点云：

```py
sam_refined_full = apply_4x4_to_points(sam_init_full, T_icp)
```

注意：
ICP 是在 visible 点云上求的，但变换必须同步作用到 full 点云，这样完整 mesh 才能一起被正确移动。

### 第一次尺度修正：XY extent correction

ICP 是刚性配准，不会改变尺度。
所以 ICP 后再检查 full 点云的 XY 宽高

```py
full_xy_after_icp, _, _ = robust_extent_xy(sam_refined_full, low=2.0, high=98.0)
s_corr_xy = solve_scale_xy(target_xy, full_xy_after_icp)
s_corr_xy = np.clip(s_corr_xy, 0.94, 1.06)
```

这个修正限制在 [0.94, 1.06]，只允许小幅度尺度补偿，避免过拟合。

然后以 anchor center 为中心缩放：

```py
sam_corr_full = centered_similarity(sam_refined_full, scale=s_corr_xy, center=anchor_center)
sam_corr_vis = centered_similarity(sam_refined_vis, scale=s_corr_xy, center=anchor_center)
```

再重新平移：

```py
t1 = translation_from_visible_center(...)
```

### 第二次尺度修正：投影 bbox correction

这一步是把 3D 点重新投影到图像上，看它的 2D bbox 是否和原始 mask bbox 一致

```py
proj_px_wh = projected_mask_bbox_wh(sam_corr_vis, anchor_data, dilate_ksize=5)
s_corr_proj = solve_projected_bbox_scale(target_px_wh, proj_px_wh, ww=0.18, wh=0.82)
s_corr_proj = np.clip(s_corr_proj, 0.97, 1.00)
```

前面 solve_scale_xy() 是在 3D 米制 XY 平面比较宽高。
这里则是：

1. 把 SAM 可见点云投影回图像；
2. 得到预测 mask 的 bbox；
3. 和真实 mask bbox 比较；
4. 根据 2D 投影大小再修正尺度。

project_points() 使用相机投影公式：

```py
u = fx * x / z + cx
v = fy * y / z + cy
```

render_depth_and_mask() 生成一个稀疏深度图和 mask：

```py
depth_img[vi, ui] = np.minimum(depth_img[vi, ui], zi)
mask_img[np.isfinite(depth_img)] = 1
```

也就是做一个简单的 z-buffer。

这里 s_corr_proj 被限制在：

```py
0.97 到 1.00
```

这一步只允许缩小，不允许放大。
SAM3D 结果容易投影偏大，所以用投影 bbox 做保守收缩。

### 第三次尺度修正：object-frame correction

前面 XY 和投影 bbox 都是基于相机平面/图像平面。
这一步基于物体自身 PCA 主轴尺寸：

```py
obj_ext_mid, _, _, _, _ = robust_pca_extents(sam_mid_full, low=2.0, high=98.0)
s_corr_obj = solve_scale_object_dims(target_xy, obj_ext_mid, ww=0.18, wh=0.82)
s_corr_obj = np.clip(s_corr_obj, 0.95, 1.01)
```

robust_pca_extents() 先对 full 点云做 PCA，然后在主轴坐标系下计算 robust extents。

他看的是：`物体自身最长轴、次长轴、最短轴上的尺寸`

而不是相机 XY 平面的尺寸。


`solve_scale_object_dims()` 里:

```py
src_h = src_ext_sorted[0]
src_w = src_ext_sorted[1]
tgt_h = target_hw[1]
tgt_w = target_hw[0]
```

这里把最长轴当作高度方向，第二长轴当作宽度方向。
然后同样用加权最小二乘求尺度。

这一步的潜在问题是：
如果物体的最长轴并不对应图像高度，或者鼠标这类物体的“长度”大于“高度”，那么变量名 src_h 可能语义不准确。

但从代码运行角度，它只是用最长轴和次长轴拟合目标 bbox 的两个尺寸。


### 最终尺度融合

代码先做一个预融合：

```py
s_corr_pre_obj = sqrt(s_corr_xy * s_corr_proj)
```

然后：

```py
s_corr = s_corr_pre_obj * s_corr_obj
s_corr = np.clip(s_corr, 0.94, 1.02)
```

最后：

```py
sam_final_full = centered_similarity(sam_refined_full, scale=s_corr, center=anchor_center)
sam_final_vis = centered_similarity(sam_refined_vis, scale=s_corr, center=anchor_center)
t2 = translation_from_visible_center(...)
sam_final_full += t2
sam_final_vis += t2
```


最终得到：

sam_final_full：完整 SAM 点云对齐到米制相机坐标；
sam_final_vis：可见 SAM 表面对齐到米制相机坐标。


## 6. 对齐质量评分

```py
metrics = score_alignment(...)
```

综合了多个指标

```py
score = (
    5.0 * size_score +
    1.0 * chamfer_n +
    0.9 * (1.0 - proj["iou"]) +
    0.4 * (1.0 - proj["coverage"]) +
    0.35 * depth_n +
    0.2 * rmse / diag -
    0.15 * fitness
)
```

> 不是单看 ICP，也不是单看 bbox，而是综合评估。


### 尺寸误差

```py
rel_err_full = abs(full_final_xy - target_xy) / target_xy
size_score = 0.32 * rel_err_w + 0.68 * rel_err_h
```

尺寸误差的权重最大，说明这个 pipeline 最关心最终模型的真实物理尺寸是否合理。

### Chamfer distance

```py
mean_cd, med_cd = compute_symmetric_chamfer(anchor_pts, sam_vis_refined)
```

Chamfer distance 是双向最近邻距离：

```py
anchor → SAM visible
SAM visible → anchor
```

代码计算：

```py
d1.mean() + d2.mean()
d1.median() + d2.median()
```

并用 `diag` 做归一化：

```py
chamfer_n = mean_cd / diag
```

这用于衡量 3D 几何贴合程度。


### 投影 IOU


```py
proj = projection_metrics(sam_vis_refined, anchor_data)
```

里面会把 SAM 可见点云投影成 mask，然后和真实 mask 算 IoU：

```py
iou = binary_iou(mask_gt, mask_pred)
```

### 深度 MAE

```py
abs_depth = abs(depth_pred[overlap] - depth_gt[overlap])
depth_mae = mean(abs_depth)
```

### coverage

```py
coverage = overlap.sum() / mask_gt.sum()
```

这衡量 SAM 投影点覆盖了多少真实 mask 区域。


### ICP fitness 和 rmse

```py
- 0.15 * fitness
+ 0.2 * rmse / diag
```

fitness 越高越好，所以是负号；
rmse 越大越差，所以是正号。

不过这里 ICP 指标权重并不算特别大，说明作者知道 ICP 只是局部几何贴合，不能完全代表最终模型质量。


## 7. 把变换应用到 SAM3D 原生 mesh

前面所有对齐搜索都是在点云上做的。
但最终要导出的是 mesh：

```py
metric_mesh = apply_alignment_to_trimesh(raw_mesh, best)
```

核心函数：

```py
def apply_alignment_to_vertices(vertices, align):
    V = centered_similarity(
        V,
        scale=align["scale_pre"],
        R=align["R"],
        center=align["sam_center"],
    )
    V = V + align["t0"]
    V = apply_4x4_to_points(V, align["T_icp"])
    V = centered_similarity(
        V,
        scale=align["scale_correction"],
        center=align["anchor_center"],
    )
    V = V + align["t2"]
    return V
```

这一步非常重要：
它必须严格复现前面对点云做过的变换顺序。

也就是：

```
1. 以 SAM center 为中心做初始 scale_pre + R
2. 加 t0
3. 应用 ICP 4×4 变换
4. 以 anchor center 为中心做最终 scale correction
5. 加 t2
```

如果这个顺序错了，mesh 和对齐点云就不一致。

### 支持 Trimesh 和 Scene

```py
if isinstance(mesh, trimesh.Trimesh):
    mesh.vertices = apply_alignment_to_vertices(mesh.vertices, align)

if isinstance(mesh, trimesh.Scene):
    for name, geom in mesh.geometry.items():
        ...
```

这里兼容两种 SAM3D 输出：

单个 trimesh.Trimesh；
包含多个 geometry 的 trimesh.Scene。

如果是 Scene，就遍历每个 geometry，并分别变换顶点。

### 导出米制 mesh

```py
export_mesh_as_ply(metric_mesh, EXPORT_METRIC_MESH_PATH)
```

export_mesh_as_ply() 会：

1. 检查后缀必须是 .ply；
2. 自动创建父目录；
3. 如果是 Trimesh，直接导出；
4. 如果是 Scene，先合并成一个 mesh 再导出。

## 8. 额外导出：PCA 主轴对齐 mesh

```py
axis_mesh, mesh_align_axes, mesh_align_bbox_center = pca_align_trimesh_to_origin(metric_mesh)
export_mesh_as_ply(axis_mesh, EXPORT_METRIC_MESH_AXIS_ALIGNED_PATH)
```

这一步不是相机坐标下的真实位姿导出，而是为了得到一个规范化朝向的物体模型。

### PCA 对齐逻辑

```py
center, axes, _ = compute_pca_frame(V)
```

PCA 得到三个主轴

```py
axes[:, 0] = 最长轴
axes[:, 1] = 第二长轴
axes[:, 2] = 第三轴
```


然后进行强制重排

```py
axes_reordered = np.column_stack([
    axes[:, 1],   # x
    axes[:, 0],   # y
    axes[:, 2],   # z
])
```

```
第二长轴 → x
最长轴   → y
第三轴   → z
```


这很适合某些物体规范化，例如把鼠标的长轴放到 y 方向。



最终得到一个：

主轴对齐；
中心在原点；
尺度仍然是米；
但不再保留相机位姿的 mesh。
