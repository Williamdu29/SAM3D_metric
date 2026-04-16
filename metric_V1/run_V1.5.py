
import os
import sys
import copy
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import time
import torch
import psutil

# ============================================================
# Config
# ============================================================
RGB_PATH = "/mnt/ws_shard/dct/SKU/soda_cookie/rgb/1775641324013.jpg"
DEPTH_PATH = "/mnt/ws_shard/dct/SKU/soda_cookie/depth/1775641324013.png"
MASK_PATH = "/mnt/ws_shard/dct/SKU/soda_cookie/masks/maskdata/1775641324013.png"
K_PATH = "/mnt/ws_shard/dct/SKU/cookie/K.txt"

SAM3D_ROOT = "/home/dct/work/sam-3d-objects"
SAM3D_CONFIG = "/home/user/datas/hc/data/ckpts/sam3d-obj/models/checkpoints/pipeline.yaml"
OUT_DIR = os.path.join(SAM3D_ROOT, "soda_cookie_output")
os.makedirs(OUT_DIR, exist_ok=True)

RUN_SAM3D = True
VOXEL_SIZE = 0.0025
ICP_MAX_ITER = 40
MESH_POISSON_DEPTH = 8
MESH_DENSITY_Q = 0.02

# v8 design notes:
# 1) Preserve V5/V7's isotropic-only geometry preservation: never use anisotropic scaling.
# 2) Keep the main metric alignment on FULL cloud camera-frame XY extents and rigid ICP.
# 3) Thickness/Z is never used for scale.
# 4) Make the projection-bbox correction one-sided and conservative: it may shrink slightly, but never enlarge.
# 5) Add a final object-frame isotropic calibration on the FULL cloud using PCA extents, because the user measures
#    object size in MeshLab along the object's principal dimensions, not camera-frame XY.
# 6) Final Poisson mesh is isotropically calibrated back to the corrected point cloud extents so mesh inflation does not
#    change the final measured scale.

sys.path.append(os.path.join(SAM3D_ROOT, "notebook"))
from inference import Inference, load_image  # noqa: E402


def load_mask(mask_path): # 加载掩码图像并将其转换为二值掩码
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    return (mask > 0).astype(np.uint8) 


def run_sam3d(rgb_path, mask_path, config_path, save_path): # 运行SAM3D模型进行推理，并将结果保存为PLY文件
    inference = Inference(config_path, compile=False)
    image = load_image(rgb_path)
    mask = load_mask(mask_path)
    output = inference(image, mask, seed=42)
    output["gs"].save_ply(save_path)
    return save_path


def load_pcd(path): # 加载点云数据并检查其有效性
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pcd


def save_pcd(path, pcd):
    ok = o3d.io.write_point_cloud(path, pcd)
    if not ok:
        raise RuntimeError(f"Failed to save point cloud: {path}")


def save_mesh(path, mesh):
    ok = o3d.io.write_triangle_mesh(path, mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to save mesh: {path}")


def np_to_pcd(points): # 将NumPy数组转换为Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def pcd_to_np(pcd):
    return np.asarray(pcd.points).astype(np.float64)


def voxel_downsample_pcd(pcd, voxel_size=VOXEL_SIZE): # 对点云进行体素下采样，以减少点云的密度并加快后续处理速度
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def robust_center(points): # 返回点云的鲁棒中心，即每个维度上的中位数，以减少异常值的影响
    return np.median(points, axis=0)


def robust_extent(points, low=2.0, high=98.0): # 返回点云的鲁棒范围，即每个维度上指定百分位数之间的距离，以减少异常值的影响
    lo = np.percentile(points, low, axis=0)
    hi = np.percentile(points, high, axis=0)
    return hi - lo, lo, hi


def robust_extent_xy(points, low=2.0, high=98.0): # 返回点云在XY平面上的鲁棒范围
    lo = np.percentile(points[:, :2], low, axis=0) # points[:, :2]表示只考虑点云的前两列，即X和Y坐标
    hi = np.percentile(points[:, :2], high, axis=0)
    return hi - lo, lo, hi


def compute_pca_frame(points): # 计算点云的PCA坐标系，包括中心、主轴和特征值，以便进行后续的对齐和尺度调整
    center = robust_center(points) 
    X = points - center[None, :] # X 是点云相对于中心的坐标矩阵，shape为(N, 3)，其中N是点云中的点数
    cov = X.T @ X / max(len(X) - 1, 1) # cov 是点云的协方差矩阵，shape为(3, 3)，表示点云在三个维度上的分布情况
    evals, evecs = np.linalg.eigh(cov) # evals 是协方差矩阵的特征值，evecs 是对应的特征向量，shape为(3,)，(3, 3)，分别表示点云在主轴方向上的方差和主轴的方向
    order = np.argsort(evals)[::-1] # order 是特征值从大到小的索引，用于将主轴按照方差大小排序
    evals = evals[order] # 将特征值按照order排序，使得evals[0]是最大的特征值，evals[1]是第二大的特征值，evals[2]是最小的特征值
    evecs = evecs[:, order] # 
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1.0
    return center, evecs, evals


def generate_sign_flip_rotations(base_R): # 生成基于base_R的8个旋转矩阵，通过对base_R的列进行符号翻转来实现，以考虑可能的轴反转情况
    flips = [
        np.diag([1, 1, 1]),
        np.diag([1, -1, -1]),
        np.diag([-1, 1, -1]),
        np.diag([-1, -1, 1]),
    ]
    rots = []
    for F in flips:
        R = base_R @ F
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1.0
        rots.append(R)
    return rots # 返回一个包含8个旋转矩阵的列表，每个矩阵都是base_R经过不同列符号翻转后的结果，且保证每个矩阵的行列式为正，以保持右手坐标系的性质


def centered_similarity(points, scale=1.0, R=None, t=None, center=None): # 对点云进行中心化的相似变换，包括缩放、旋转和平移，以便进行对齐和尺度调整
    X = points.copy()
    if center is not None:
        X = X - center[None, :]
    if R is not None:
        X = X @ R.T # 如果提供了旋转矩阵R，则将点云X乘以R的转置来实现旋转变换
    X = X * float(scale)
    if center is not None:
        X = X + center[None, :]
    if t is not None:
        X = X + t[None, :]
    return X


def apply_4x4_to_points(points, T): # 将一个4x4的变换矩阵T应用到点云上，以实现点云的刚性变换，包括旋转和平移
    X = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    Y = (T @ X.T).T
    return Y[:, :3]


def largest_component(mask): # 从二值掩码中提取最大的连通组件，以去除小的噪声区域，返回一个新的二值掩码，其中只有最大的连通组件被保留
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return (labels == keep).astype(np.uint8)


def morph_clean(mask, ksize=5): # 对二值掩码进行形态学清理操作，包括开运算和闭运算，以去除小的噪声区域和填补小的空洞，然后提取最大的连通组件，返回一个新的二值掩码，其中只有主要的连通组件被保留
    kernel = np.ones((ksize, ksize), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = largest_component(mask)
    return mask.astype(np.uint8)


def infer_depth_meters(depth): # 推断深度图像的单位是否为毫米，并将其转换为米，如果已经是米则直接返回，同时返回一个字符串表示推断的结果
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        raise ValueError("Depth image has no valid values.")
    vmax = float(np.percentile(valid, 99.5))
    if depth.dtype == np.uint16 or vmax > 100.0:
        return depth.astype(np.float32) / 1000.0, "mm_to_m"
    if vmax > 20.0:
        return depth.astype(np.float32) / 1000.0, "heuristic_mm_to_m"
    return depth.astype(np.float32), "already_m"


def depth_to_points(depth_m, mask, K): # 将深度图像转换为点云坐标，首先根据相机内参K计算每个像素对应的3D坐标，然后根据掩码mask筛选出有效的点云坐标，返回一个包含有效点云坐标的NumPy数组，以及对应的像素坐标和深度值
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ys, xs = np.where(mask > 0)
    z = depth_m[ys, xs]
    valid = np.isfinite(z) & (z > 1e-6) # 筛选出深度值有效的像素坐标，要求深度值必须是有限的且大于一个小的阈值，以去除无效的深度值
    xs, ys, z = xs[valid], ys[valid], z[valid]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    Z = z
    pts = np.stack([X, Y, Z], axis=1).astype(np.float32)
    return pts, xs, ys


def remove_outliers_np(points, nb_neighbors=20, std_ratio=1.5): # 使用Open3D的统计离群点移除方法来过滤点云中的离群点，首先将NumPy数组转换为Open3D点云对象，然后调用remove_statistical_outlier方法来移除离群点，最后将过滤后的点云对象转换回NumPy数组并返回，如果输入的点云数量不足以进行离群点移除，则直接返回原始点云
    if len(points) < max(nb_neighbors + 5, 20):
        return points
    pcd = np_to_pcd(points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_to_np(pcd)


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask.")
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return x0, y0, x1, y1


def erode_mask_adaptive(mask, ratio=0.12, min_k=3, max_k=31): # 根据掩码的大小自适应地选择腐蚀核的大小，以去除掩码边缘可能存在的噪声，同时保留足够的有效区域，首先计算掩码的边界框，然后根据边界框的宽高和给定的比例来计算腐蚀核的大小，并将其限制在最小值和最大值之间，最后使用OpenCV的erode函数对掩码进行腐蚀操作，并根据腐蚀后的掩码面积与原始掩码面积的比例来判断是否使用腐蚀后的掩码，返回腐蚀后的掩码、使用的核大小以及一个布尔值表示是否使用了腐蚀后的掩码
    x0, y0, x1, y1 = mask_bbox(mask)
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    k = int(round(min(bw, bh) * ratio))
    k = max(min_k, min(k, max_k))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    if eroded.sum() < max(64, 0.1 * mask.sum()):
        return mask.astype(np.uint8), k, False
    return eroded.astype(np.uint8), k, True


def estimate_anchor_scale_targets(mask, depth_m, K): # 根据掩码和深度图像来估计锚点的尺度目标，包括像素宽高、深度中位数、深度10%分位数、深度90%分位数、实际宽高等，首先对掩码进行腐蚀操作以去除边缘噪声，然后从腐蚀后的掩码中提取有效的深度值，并计算深度的统计量，接着根据掩码的边界框和相机内参K来计算像素宽高和实际宽高，最后将这些尺度目标以字典的形式返回
    mask = mask.astype(np.uint8)
    inner_mask, erode_k, used_erode = erode_mask_adaptive(mask, ratio=0.12)

    valid_depth_inner = depth_m[inner_mask > 0]
    valid_depth_inner = valid_depth_inner[np.isfinite(valid_depth_inner) & (valid_depth_inner > 1e-6)]
    if valid_depth_inner.size < 20:
        valid_depth_inner = depth_m[mask > 0]
        valid_depth_inner = valid_depth_inner[np.isfinite(valid_depth_inner) & (valid_depth_inner > 1e-6)]
        if valid_depth_inner.size == 0:
            raise ValueError("No valid depth inside mask.")

    z_med = float(np.median(valid_depth_inner))
    z_q10 = float(np.percentile(valid_depth_inner, 10))
    z_q90 = float(np.percentile(valid_depth_inner, 90))

    x0, y0, x1, y1 = mask_bbox(mask)
    px_w = float(x1 - x0 + 1)
    px_h = float(y1 - y0 + 1)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    metric_w = px_w * z_med / fx
    metric_h = px_h * z_med / fy

    return {
        "pixel_w": px_w,
        "pixel_h": px_h,
        "z_med": z_med,
        "z_q10": z_q10,
        "z_q90": z_q90,
        "metric_w": float(metric_w),
        "metric_h": float(metric_h),
        "inner_mask": inner_mask,
        "erode_k": int(erode_k),
        "used_erode": bool(used_erode),
    }




def robust_pca_extents(points, low=2.0, high=98.0): # 计算点云的PCA坐标系和在该坐标系下的鲁棒范围，首先计算点云的中心和主轴，然后将点云投影到主轴上，并计算投影后的每个维度上的鲁棒范围，最后返回按照范围大小排序的范围、中心、主轴以及每个维度上的低分位数和高分位数
    center, axes, _ = compute_pca_frame(points)
    proj = (points - center[None, :]) @ axes
    ext, lo, hi = robust_extent(proj, low=low, high=high)
    order = np.argsort(ext)[::-1]
    return ext[order], center, axes, lo[order], hi[order]


def solve_scale_object_dims(target_hw, src_ext_sorted, ww=0.20, wh=0.80): # 解决对象尺度问题，根据目标尺度和源尺度来计算缩放因子，保持各向同性缩放
    # target_hw: [width, height] in meters. Object-frame primary extent should match height,
    # second extent should roughly match width. Keep isotropic scaling only.
    src_h = float(src_ext_sorted[0])
    src_w = float(src_ext_sorted[1])
    tgt_h = float(target_hw[1])
    tgt_w = float(target_hw[0])
    num = wh * src_h * tgt_h + ww * src_w * tgt_w
    den = wh * src_h * src_h + ww * src_w * src_w + 1e-12
    return max(float(num / den), 1e-8)


def isotropically_scale_mesh_about_center(mesh, scale, center): # 对网格进行中心化的各向同性缩放，首先将网格的顶点坐标转换为NumPy数组，然后根据给定的中心和缩放因子来调整顶点坐标，最后将调整后的顶点坐标重新赋值给网格，并计算顶点法线以更新网格的几何属性，返回调整后的网格对象
    mesh = copy.deepcopy(mesh)
    V = np.asarray(mesh.vertices).astype(np.float64)
    V = (V - center[None, :]) * float(scale) + center[None, :]
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.compute_vertex_normals()
    return mesh


def calibrate_mesh_to_pointcloud(mesh, ref_points):
    mesh_pts = np.asarray(mesh.vertices).astype(np.float64)
    if len(mesh_pts) == 0 or len(ref_points) == 0:
        return mesh, 1.0, np.array([-1.0, -1.0], dtype=np.float64)
    mesh_ext, mesh_center, _, _, _ = robust_pca_extents(mesh_pts, low=2.0, high=98.0)
    ref_ext, ref_center, _, _, _ = robust_pca_extents(ref_points, low=2.0, high=98.0)
    s_mesh = solve_scale_object_dims(np.array([ref_ext[1], ref_ext[0]], dtype=np.float64), mesh_ext, ww=0.25, wh=0.75)
    s_mesh = float(np.clip(s_mesh, 0.94, 1.02))
    center = 0.5 * (mesh_center + ref_center)
    mesh_out = isotropically_scale_mesh_about_center(mesh, s_mesh, center)
    mesh_ext_after, _, _, _, _ = robust_pca_extents(np.asarray(mesh_out.vertices).astype(np.float64), low=2.0, high=98.0)
    return mesh_out, s_mesh, mesh_ext_after[:2].copy()

def build_anchor_data(rgb_path, depth_path, mask_path, K_path): # 从RGB-D数据构建锚点数据，包括读取RGB图像、深度图像和掩码图像，推断深度单位，计算点云坐标，去除离群点，并计算锚点的中心和范围等信息，返回一个包含这些信息的字典，以便后续的对齐和评估步骤使用
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(depth_path)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)

    K = np.loadtxt(K_path).reshape(3, 3)
    depth_m, depth_mode = infer_depth_meters(depth)

    mask_bin = morph_clean((mask > 0).astype(np.uint8), ksize=5)
    scale_targets = estimate_anchor_scale_targets(mask_bin, depth_m, K) # 根据掩码和深度图像来估计锚点的尺度目标，包括像素宽高、深度中位数、深度10%分位数、深度90%分位数、实际宽高等，首先对掩码进行腐蚀操作以去除边缘噪声，然后从腐蚀后的掩码中提取有效的深度值，并计算深度的统计量，接着根据掩码的边界框和相机内参K来计算像素宽高和实际宽高，最后将这些尺度目标以字典的形式返回

    z_med = scale_targets["z_med"] # 深度的中位数，表示锚点所在位置的典型深度值，通常用于后续的尺度调整和对齐步骤
    band = max(0.035 * z_med, 0.012) # 根据深度的中位数来计算一个深度范围的带宽，用于筛选出与锚点位置相近的有效深度值，以去除远离锚点位置的噪声点
    valid_depth_mask = np.isfinite(depth_m) & (depth_m > 1e-6) & (np.abs(depth_m - z_med) <= band)
    final_mask = morph_clean(mask_bin & valid_depth_mask.astype(np.uint8), ksize=3)

    pts, _, _ = depth_to_points(depth_m, final_mask, K) # 这里返回的点云是根据深度图像和掩码计算得到的点云坐标，首先根据相机内参K计算每个像素对应的3D坐标，然后根据掩码筛选出有效的点云坐标，返回一个包含有效点云坐标的NumPy数组，以及对应的像素坐标和深度值
    pts = remove_outliers_np(pts, nb_neighbors=20, std_ratio=1.2)
    if len(pts) == 0:
        raise ValueError("Anchor point cloud is empty after filtering.")

    pcd = np_to_pcd(pts) # 将点云坐标转换为Open3D点云对象，以便进行后续的处理和保存
    anchor_center = robust_center(pts)
    anchor_xy_extent, _, _ = robust_extent_xy(pts, low=2.0, high=98.0)

    return {
        "rgb": rgb,
        "H": H,
        "W": W,
        "depth_m": depth_m,
        "mask_bin": mask_bin.astype(np.uint8),
        "final_mask": final_mask.astype(np.uint8),
        "K": K,
        "pcd": pcd,
        "pts": pts,
        "depth_mode": depth_mode,
        "scale_targets": scale_targets,
        "anchor_center": anchor_center,
        "anchor_xy_extent": anchor_xy_extent,
    }


def estimate_hidden_point_radius(points):
    center = robust_center(points)
    d = np.linalg.norm(points - center[None, :], axis=1)
    return float(max(np.percentile(d, 95) * 6.0, 1e-3))


def extract_visible_surface_from_camera(points, camera_location=None, radius=None):
    if camera_location is None:
        camera_location = np.zeros(3, dtype=np.float64)
    pcd = np_to_pcd(points)
    if len(pcd.points) == 0:
        return points.copy()
    if radius is None:
        radius = estimate_hidden_point_radius(points)
    _, visible_idx = pcd.hidden_point_removal(camera_location, radius)
    vis = pcd.select_by_index(visible_idx)
    return pcd_to_np(vis)


def compute_symmetric_chamfer(src_pts, tgt_pts):
    src = np_to_pcd(src_pts)
    tgt = np_to_pcd(tgt_pts)
    d1 = np.asarray(src.compute_point_cloud_distance(tgt), dtype=np.float64)
    d2 = np.asarray(tgt.compute_point_cloud_distance(src), dtype=np.float64)
    if len(d1) == 0 or len(d2) == 0:
        return np.inf, np.inf
    return float(d1.mean() + d2.mean()), float(np.median(d1) + np.median(d2))


def project_points(points, K, H, W):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    if valid.sum() == 0:
        return np.empty((0,), np.int32), np.empty((0,), np.int32), np.empty((0,), np.float32)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    return ui[inside], vi[inside], z[inside].astype(np.float32)


def render_depth_and_mask(points, K, H, W):
    depth_img = np.full((H, W), np.inf, dtype=np.float32)
    mask_img = np.zeros((H, W), dtype=np.uint8)
    ui, vi, zi = project_points(points, K, H, W)
    if len(ui) == 0:
        return depth_img, mask_img
    order = np.argsort(zi)
    ui = ui[order]
    vi = vi[order]
    zi = zi[order]
    depth_img[vi, ui] = np.minimum(depth_img[vi, ui], zi)
    mask_img[np.isfinite(depth_img)] = 1
    return depth_img, mask_img


def binary_iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def projection_metrics(points, anchor_data):
    H = anchor_data["H"]
    W = anchor_data["W"]
    K = anchor_data["K"]
    depth_gt = anchor_data["depth_m"]
    mask_gt = anchor_data["mask_bin"] > 0
    depth_pred, mask_pred = render_depth_and_mask(points, K, H, W)
    iou = binary_iou(mask_gt, mask_pred > 0)
    overlap = mask_gt & np.isfinite(depth_pred) & np.isfinite(depth_gt) & (depth_gt > 1e-6)
    if overlap.sum() == 0:
        return {"iou": iou, "depth_mae": np.inf, "coverage": 0.0}
    abs_depth = np.abs(depth_pred[overlap] - depth_gt[overlap])
    return {
        "iou": iou,
        "depth_mae": float(np.mean(abs_depth)),
        "coverage": float(overlap.sum() / max(mask_gt.sum(), 1)),
    }

def projected_mask_bbox_wh(points, anchor_data, dilate_ksize=5):
    H = anchor_data["H"]
    W = anchor_data["W"]
    K = anchor_data["K"]
    _, mask_pred = render_depth_and_mask(points, K, H, W)
    if mask_pred.sum() == 0:
        return None
    if dilate_ksize > 1:
        kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        mask_pred = cv2.dilate(mask_pred.astype(np.uint8), kernel, iterations=1)
    ys, xs = np.where(mask_pred > 0)
    if len(xs) == 0:
        return None
    px_w = float(xs.max() - xs.min() + 1)
    px_h = float(ys.max() - ys.min() + 1)
    return np.array([px_w, px_h], dtype=np.float64)


def solve_projected_bbox_scale(target_px_wh, src_px_wh, ww=0.25, wh=0.75):
    if src_px_wh is None:
        return 1.0
    sw, sh = float(src_px_wh[0]), float(src_px_wh[1])
    tw, th = float(target_px_wh[0]), float(target_px_wh[1])
    if sw <= 1e-8 or sh <= 1e-8:
        return 1.0
    num = ww * sw * tw + wh * sh * th
    den = ww * sw * sw + wh * sh * sh + 1e-12
    return max(float(num / den), 1e-8)


def icp_refine_rigid(source_pts, target_pts, threshold):
    src = np_to_pcd(source_pts)
    tgt = np_to_pcd(target_pts)
    reg = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITER),
    )
    refined = apply_4x4_to_points(source_pts, reg.transformation)
    return refined, reg.transformation, float(reg.fitness), float(reg.inlier_rmse)


def solve_scale_xy(target_xy, src_xy, ww=0.32, wh=0.68):
    target_w, target_h = float(target_xy[0]), float(target_xy[1])
    src_w, src_h = float(src_xy[0]), float(src_xy[1])
    num = ww * src_w * target_w + wh * src_h * target_h
    den = ww * src_w * src_w + wh * src_h * src_h + 1e-12
    s = float(num / den)
    return max(s, 1e-8)


def translation_from_visible_center(visible_pts, anchor_center, target_z):
    vis_center = robust_center(visible_pts)
    return np.array([
        anchor_center[0] - vis_center[0],
        anchor_center[1] - vis_center[1],
        target_z - np.median(visible_pts[:, 2]),
    ], dtype=np.float64)


def score_alignment(anchor_data, anchor_pts, sam_vis_refined, full_final_xy, target_xy, fitness, rmse):
    rel_err_full = np.abs(full_final_xy - target_xy) / np.maximum(target_xy, 1e-8)
    mean_cd, med_cd = compute_symmetric_chamfer(anchor_pts, sam_vis_refined)
    proj = projection_metrics(sam_vis_refined, anchor_data)
    diag = float(np.linalg.norm([target_xy[0], target_xy[1], anchor_data["scale_targets"]["z_med"]]) + 1e-8)
    size_score = float(0.32 * rel_err_full[0] + 0.68 * rel_err_full[1])
    chamfer_n = mean_cd / diag if np.isfinite(mean_cd) else 1e6
    depth_n = proj["depth_mae"] / max(anchor_data["scale_targets"]["z_med"], 1e-6) if np.isfinite(proj["depth_mae"]) else 1e6
    score = (
        5.0 * size_score +
        1.0 * chamfer_n +
        0.9 * (1.0 - proj["iou"]) +
        0.4 * (1.0 - proj["coverage"]) +
        0.35 * depth_n +
        0.2 * rmse / max(diag, 1e-8) -
        0.15 * fitness
    )
    return {
        "score": float(score),
        "mean_cd": float(mean_cd),
        "med_cd": float(med_cd),
        "iou": float(proj["iou"]),
        "depth_mae": float(proj["depth_mae"]),
        "coverage": float(proj["coverage"]),
        "size_rel_err_full": rel_err_full,
    }


def search_metric_alignment(anchor_data, sam_pcd): # 传入锚点数据和SAM3D点云对象，首先对两者进行体素下采样和离群点移除，然后计算它们的PCA坐标系，并生成基于PCA对齐的候选旋转矩阵，接着对于每个候选旋转矩阵，进行一系列的尺度调整、可见表面提取、ICP精细化等步骤，最后根据对齐结果计算各种评估指标，并返回一个包含这些指标的字典，以便后续的选择和分析
    anchor_pts = pcd_to_np(voxel_downsample_pcd(anchor_data["pcd"], VOXEL_SIZE))
    anchor_pts = remove_outliers_np(anchor_pts, nb_neighbors=20, std_ratio=1.2)
    if len(anchor_pts) == 0:
        raise RuntimeError("Anchor point cloud is empty after downsampling.")

    sam_pts = pcd_to_np(voxel_downsample_pcd(sam_pcd, VOXEL_SIZE))
    sam_pts = remove_outliers_np(sam_pts, nb_neighbors=20, std_ratio=1.5)
    if len(sam_pts) == 0:
        raise RuntimeError("SAM3D point cloud is empty after downsampling.")

    anchor_center = anchor_data["anchor_center"] # 返回锚点的中心坐标，通常是根据锚点点云计算得到的一个代表性的中心位置
    target_xy = np.array([
        anchor_data["scale_targets"]["metric_w"],
        anchor_data["scale_targets"]["metric_h"],
    ], dtype=np.float64) # 返回锚点的尺度目标，包括实际宽度和高度
    target_z = float(anchor_data["scale_targets"]["z_med"]) # 返回锚点的深度中位数，表示锚点所在位置的典型深度值

    anchor_pca_center, anchor_axes, _ = compute_pca_frame(anchor_pts) # 计算锚点点云的PCA坐标系，包括中心和主轴，以便后续的对齐和旋转调整
    sam_center, sam_axes, _ = compute_pca_frame(sam_pts)
    _ = anchor_pca_center # 这里anchor_pca_center没有被直接使用，但它是计算base_R的基础，确保了旋转矩阵是基于锚点的PCA坐标系来生成的

    base_R = anchor_axes @ sam_axes.T # 旋转矩阵
    candidate_Rs = generate_sign_flip_rotations(base_R) # 生成基于base_R的候选旋转矩阵，通过对base_R的列进行不同的符号翻转来产生多个旋转矩阵，以增加对齐的鲁棒性和覆盖更多的可能的旋转情况

    best = None
    for ridx, R in enumerate(candidate_Rs):
        sam_rot_full = centered_similarity(sam_pts, scale=1.0, R=R, center=sam_center) # 计算对齐旋转后的SAM3D点云坐标
        full_xy_raw, _, _ = robust_extent_xy(sam_rot_full, low=2.0, high=98.0)
        scale0 = solve_scale_xy(target_xy, full_xy_raw) # 返回一个初始的缩放因子，是根据目标尺度和旋转对齐后的SAM3D点云的尺度来计算的

        for mult in [0.94, 0.98, 1.00, 1.02, 1.06]:
            s = scale0 * mult
            sam_scaled_full = centered_similarity(sam_pts, scale=s, R=R, center=sam_center) # 计算旋转对齐后再进行缩放调整后的SAM3D点云坐标
            sam_scaled_vis = extract_visible_surface_from_camera(sam_scaled_full)
            if len(sam_scaled_vis) < 100:
                continue # 如果提取的可见表面点云数量不足100个，则跳过当前的缩放因子，继续尝试下一个缩放因子

            t0 = translation_from_visible_center(sam_scaled_vis, anchor_center, target_z) # 平移向量
            sam_init_full = sam_scaled_full + t0[None, :] # 对齐的SAM3D点云坐标经过初始的平移调整后的坐标
            sam_init_vis = sam_scaled_vis + t0[None, :]

            threshold = max(0.008, 4.0 * VOXEL_SIZE)
            sam_refined_vis, T_icp, fitness, rmse = icp_refine_rigid(sam_init_vis, anchor_pts, threshold)
            sam_refined_full = apply_4x4_to_points(sam_init_full, T_icp)

            # V7: keep isotropic-only shape preservation. First use FULL cloud XY correction,
            # then add a very small projection-bbox isotropic correction in image space.
            full_xy_after_icp, _, _ = robust_extent_xy(sam_refined_full, low=2.0, high=98.0) # 返回经过ICP精细化后的SAM3D点云的尺度信息
            s_corr_xy = solve_scale_xy(target_xy, full_xy_after_icp)
            s_corr_xy = float(np.clip(s_corr_xy, 0.94, 1.06)) # 计算尺度校正因子，并将其限制在0.94到1.06之间，以避免过度的缩放调整

            sam_corr_full = centered_similarity(sam_refined_full, scale=s_corr_xy, center=anchor_center) # 计算经过尺度校正后的SAM3D点云坐标
            sam_corr_vis = centered_similarity(sam_refined_vis, scale=s_corr_xy, center=anchor_center)
            t1 = translation_from_visible_center(sam_corr_vis, anchor_center, target_z)
            sam_corr_full = sam_corr_full + t1[None, :] 
            sam_corr_vis = sam_corr_vis + t1[None, :]

            target_px_wh = np.array([
                anchor_data["scale_targets"]["pixel_w"],
                anchor_data["scale_targets"]["pixel_h"],
            ], dtype=np.float64) # 返回锚点的像素宽高尺度目标，是根据掩码的边界框和相机内参K计算得到的，表示在图像平面上锚点区域的宽度和高度，以像素为单位
            proj_px_wh = projected_mask_bbox_wh(sam_corr_vis, anchor_data, dilate_ksize=5) # 计算经过尺度校正后的SAM3D点云在图像平面上的投影边界框的像素宽高
            s_corr_proj = solve_projected_bbox_scale(target_px_wh, proj_px_wh, ww=0.18, wh=0.82) # 计算投影边界框的尺度校正因子，是根据目标像素宽高和投影像素宽高来计算的
            # V8: projection correction is one-sided and conservative. It may shrink a little, but it never enlarges.
            s_corr_proj = float(np.clip(s_corr_proj, 0.97, 1.00))

            # First isotropic correction remains conservative.
            s_corr_pre_obj = float(np.sqrt(s_corr_xy * s_corr_proj)) # 计算一个预先的尺度校正因子，是基于之前的尺度校正因子（s_corr_xy和s_corr_proj）的几何平均来计算的，以保持各向同性的缩放调整
            s_corr_pre_obj = float(np.clip(s_corr_pre_obj, 0.97, 1.02))

            sam_mid_full = centered_similarity(sam_refined_full, scale=s_corr_pre_obj, center=anchor_center) 
            sam_mid_vis = centered_similarity(sam_refined_vis, scale=s_corr_pre_obj, center=anchor_center)
            t_mid = translation_from_visible_center(sam_mid_vis, anchor_center, target_z)
            sam_mid_full = sam_mid_full + t_mid[None, :]
            sam_mid_vis = sam_mid_vis + t_mid[None, :]

            # V8: final isotropic object-frame calibration. This preserves shape ratio and targets what MeshLab measures.
            obj_ext_mid, _, _, _, _ = robust_pca_extents(sam_mid_full, low=2.0, high=98.0)
            s_corr_obj = solve_scale_object_dims(target_xy, obj_ext_mid, ww=0.18, wh=0.82) # 根据目标尺度和经过预先尺度校正后的SAM3D点云的尺度来计算一个最终的尺度校正因子
            s_corr_obj = float(np.clip(s_corr_obj, 0.95, 1.01))

            s_corr = float(s_corr_pre_obj * s_corr_obj)
            s_corr = float(np.clip(s_corr, 0.94, 1.02))

            sam_final_full = centered_similarity(sam_refined_full, scale=s_corr, center=anchor_center)
            sam_final_vis = centered_similarity(sam_refined_vis, scale=s_corr, center=anchor_center)
            t2 = translation_from_visible_center(sam_final_vis, anchor_center, target_z)
            sam_final_full = sam_final_full + t2[None, :]
            sam_final_vis = sam_final_vis + t2[None, :]

            full_xy_final, _, _ = robust_extent_xy(sam_final_full, low=2.0, high=98.0)
            vis_xy_final, _, _ = robust_extent_xy(sam_final_vis, low=2.0, high=98.0)
            proj_px_wh_final = projected_mask_bbox_wh(sam_final_vis, anchor_data, dilate_ksize=5)
            metrics = score_alignment(anchor_data, anchor_pts, sam_final_vis, full_xy_final, target_xy, fitness, rmse)


            result = {
                "score": metrics["score"],
                "R_idx": ridx,
                "R": R,
                "scale_pre": float(s),
                "scale_correction_xy": float(s_corr_xy),
                "scale_correction_proj": float(s_corr_proj),
                "scale_correction_obj": float(s_corr_obj),
                "scale_correction_pre_obj": float(s_corr_pre_obj),
                "scale_correction": float(s_corr),
                "scale": float(s * s_corr),
                "T_icp": T_icp,
                "fitness": float(fitness),
                "rmse": float(rmse),
                "mean_cd": metrics["mean_cd"],
                "med_cd": metrics["med_cd"],
                "iou": metrics["iou"],
                "depth_mae": metrics["depth_mae"],
                "coverage": metrics["coverage"],
                "target_wh": target_xy.copy(),
                "target_px_wh": target_px_wh.copy(),
                "projected_px_wh": np.array(proj_px_wh_final if proj_px_wh_final is not None else [-1.0, -1.0], dtype=np.float64),
                "aligned_full_wh": full_xy_final,
                "aligned_vis_wh": vis_xy_final,
                "size_rel_err_full_wh": metrics["size_rel_err_full"],
                "raw_full_wh_before_scale": np.array(full_xy_raw, dtype=np.float64),
                "raw_full_wh_after_icp_pre_corr": np.array(full_xy_after_icp, dtype=np.float64),
                "full_final": sam_final_full,
                "visible_final": sam_final_vis,
            }
            if best is None or result["score"] < best["score"]:
                best = result

    if best is None:
        raise RuntimeError("Failed to find a valid alignment.")
    return best


def estimate_normals_inplace(pcd, radius=None):
    pts = pcd_to_np(pcd)
    if len(pts) == 0:
        return pcd
    if radius is None:
        ext, _, _ = robust_extent(pts)
        radius = max(float(np.linalg.norm(ext) / 60.0), 0.003)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40))
    try:
        pcd.orient_normals_consistent_tangent_plane(20)
    except Exception:
        pass
    return pcd


def reconstruct_mesh_from_pcd(pcd):
    pcd = copy.deepcopy(pcd)
    pcd = voxel_downsample_pcd(pcd, VOXEL_SIZE)
    pcd = estimate_normals_inplace(pcd)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=MESH_POISSON_DEPTH)
    densities = np.asarray(densities)
    keep = densities > np.quantile(densities, MESH_DENSITY_Q)
    mesh.remove_vertices_by_mask(~keep)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.03, bbox.get_center())
    mesh = mesh.crop(bbox)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        pts = pcd_to_np(pcd)
        ext, _, _ = robust_extent(pts)
        alpha = max(float(np.linalg.norm(ext) / 40.0), 0.002)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
    return mesh


def align_mesh_to_origin_and_axes(mesh):
    """
    将 mesh 做 PCA 主轴对齐：
    1) 让 mesh 的三个主方向与 xyz 轴平行
    2) 再把旋转后的包围盒中心平移到原点
    """
    mesh = copy.deepcopy(mesh)
    V = np.asarray(mesh.vertices).astype(np.float64)
    if len(V) == 0:
        return mesh, np.eye(3), np.zeros(3, dtype=np.float64)

    # 1) 用已有的 PCA 函数求主轴
    center, axes, _ = compute_pca_frame(V)

    # 2) 处理符号不确定性，尽量让主轴方向和世界坐标轴方向一致
    #    这里只是为了结果更稳定，不影响“平行于xyz轴”这个目标
    if axes[0, 0] < 0:
        axes[:, 0] *= -1.0
    if axes[1, 1] < 0:
        axes[:, 1] *= -1.0
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1.0

    # 3) 旋转到世界坐标轴
    # 原始点: (V - center)
    # PCA坐标: (V - center) @ axes
    # 所以新的顶点直接写成 PCA 坐标即可
    V_aligned = (V - center[None, :]) @ axes

    mesh.vertices = o3d.utility.Vector3dVector(V_aligned)

    # 4) 再把旋转后的包围盒中心平移到原点
    # 不建议直接用 get_center()，因为那是顶点均值中心
    aabb = mesh.get_axis_aligned_bounding_box()
    bbox_center = aabb.get_center()

    V_aligned = np.asarray(mesh.vertices).astype(np.float64)
    V_aligned = V_aligned - bbox_center[None, :]
    mesh.vertices = o3d.utility.Vector3dVector(V_aligned)

    # 5) 更新法线
    mesh.compute_vertex_normals()

    return mesh, axes, bbox_center


def print_gpu_mem(tag=""):
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available")
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory.max_memory_allocated() / 1024**3
    reserv = torch.cuda.memory.max_memory_reserved() / 1024**3
    now_alloc = torch.cuda.memory.memory_allocated() / 1024**3
    now_reserv = torch.cuda.memory.memory_reserved() / 1024**3
    print(f"[{tag}] GPU now allocated   = {now_alloc:.2f} GB")
    print(f"[{tag}] GPU now reserved    = {now_reserv:.2f} GB")
    print(f"[{tag}] GPU peak allocated  = {alloc:.2f} GB")
    print(f"[{tag}] GPU peak reserved   = {reserv:.2f} GB")

def print_ram_mem(tag=""):
    vm = psutil.virtual_memory()
    used = (vm.total - vm.available) / 1024**3
    total = vm.total / 1024**3
    print(f"[{tag}] RAM used            = {used:.2f} / {total:.2f} GB")



def main():
    anchor_path = os.path.join(OUT_DIR, "anchor_metric_visible_surface_v8.ply") # 锚点的PLY文件路径，包含从RGB-D数据构建的锚点点云，通常用于后续的对齐和评估步骤

    raw_sam_path = os.path.join(OUT_DIR, "sam3d_raw_v8.ply") # SAM3D原始点云的PLY文件路径，包含从SAM3D模型生成的未经处理的点云数据，通常用于后续的对齐和评估步骤

    out_metric_pcd = os.path.join(OUT_DIR, "sam3d_metric_general_full.ply") # SAM3D度量对齐后的完整点云的PLY文件路径，包含经过尺度和位姿调整后的SAM3D点云数据，通常用于后续的评估和可视化步骤

    out_metric_vis = os.path.join(OUT_DIR, "sam3d_metric_general_visible.ply") # SAM3D度量对齐后的可见点云的PLY文件路径，包含经过尺度和位姿调整后的SAM3D点云数据，通常用于后续的评估和可视化步骤

    out_metric_mesh_ply = os.path.join(OUT_DIR, "sam3d_metric_general_mesh.ply") # SAM3D度量对齐后的网格的PLY文件路径，包含经过尺度和位姿调整后的SAM3D网格数据，通常用于后续的评估和可视化步骤

    out_metric_mesh_obj = os.path.join(OUT_DIR, "sam3d_metric_general_mesh.obj") # SAM3D度量对齐后的网格的OBJ文件路径，包含经过尺度和位姿调整后的SAM3D网格数据，通常用于后续的评估和可视化步骤


    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print_gpu_mem("start")
    print_ram_mem("start")

    print("[1/6] Build anchor data from RGB-D...")
    anchor_data = build_anchor_data(RGB_PATH, DEPTH_PATH, MASK_PATH, K_PATH) # 返回一个包含从RGB-D数据构建的锚点数据的字典，包括RGB图像、深度图像、掩码图像、相机内参、点云坐标、尺度目标等信息，这些信息用于后续的对齐和评估步骤
    save_pcd(anchor_path, anchor_data["pcd"])

    st = anchor_data["scale_targets"] 
    print("Anchor diagnostics:")
    print("  depth mode                         =", anchor_data["depth_mode"]) # 深度图的单位
    print("  erode kernel                       =", st["erode_k"]) # 用于腐蚀掩码的内核大小
    print("  used eroded inner mask             =", st["used_erode"]) # 是否使用了腐蚀后的掩码
    print("  bbox pixel width                   =", st["pixel_w"]) # 掩码边界框的像素宽度
    print("  bbox pixel height                  =", st["pixel_h"]) # 掩码边界框的像素高度
    print("  robust median depth [m]            =", st["z_med"]) # 深度的中位数，表示锚点所在位置的典型深度值，通常用于后续的尺度调整和对齐步骤
    print("  robust depth q10/q90 [m]           =", st["z_q10"], st["z_q90"]) # 深度的10%分位数和90%分位数，表示锚点所在位置的深度范围，通常用于后续的尺度调整和对齐步骤
    print("  target width [m]                   =", st["metric_w"]) # 根据掩码的边界框和相机内参K计算得到的目标宽度，表示锚点所在对象的实际宽度，通常用于后续的尺度调整和对齐步骤
    print("  target height [m]                  =", st["metric_h"]) # 锚点所在对象的实际高度，通常用于后续的尺度调整和对齐步骤
    print("  anchor visible XY extent [m]       =", anchor_data["anchor_xy_extent"]) # 锚点的可见XY范围，表示锚点所在对象在XY平面上的范围，通常用于后续的尺度调整和对齐步骤

    print_gpu_mem("after_anchor")
    print_ram_mem("after_anchor")


    print("\n[2/6] Run or load SAM3D...")
    if RUN_SAM3D or (not os.path.exists(raw_sam_path)):
        run_sam3d(RGB_PATH, MASK_PATH, SAM3D_CONFIG, raw_sam_path) # 先运行SAM3D模型来生成原始点云数据
    print("SAM3D raw point cloud:", raw_sam_path)

    print_gpu_mem("after_sam3d")
    print_ram_mem("after_sam3d")


    sam_pcd = load_pcd(raw_sam_path) # 加载SAM3D原始点云数据，返回一个Open3D点云对象

    print("\n[3/6] Solve metric scale from FULL cloud width/height, then refine rigid pose...")
    best = search_metric_alignment(anchor_data, sam_pcd)
    print("Best result:")
    print("  rotation candidate                 =", best["R_idx"])
    print("  pre-correction scale               =", best["scale_pre"])
    print("  final scale correction (isotropic) =", best["scale_correction"])
    print("  XY extent correction               =", best["scale_correction_xy"])
    print("  projected bbox correction          =", best["scale_correction_proj"])
    print("  object-frame correction            =", best["scale_correction_obj"])
    print("  pre-object blend correction        =", best["scale_correction_pre_obj"])
    print("  FIXED isotropic scale              =", best["scale"])
    print("  target width/height [m]            =", best["target_wh"])
    print("  aligned FULL width/height [m]      =", best["aligned_full_wh"])
    print("  aligned visible width/height [m]   =", best["aligned_vis_wh"])
    print("  target projected bbox [px]         =", best["target_px_wh"])
    print("  aligned projected bbox [px]        =", best["projected_px_wh"])
    print("  size relative error FULL w/h       =", best["size_rel_err_full_wh"])
    print("  raw FULL width/height pre-scale    =", best["raw_full_wh_before_scale"])
    print("  FULL width/height pre-corr postICP =", best["raw_full_wh_after_icp_pre_corr"])
    print("  chamfer(mean) [m]                  =", best["mean_cd"])
    print("  chamfer(median) [m]                =", best["med_cd"])
    print("  mask IoU                           =", best["iou"])
    print("  depth MAE [m]                      =", best["depth_mae"])
    print("  coverage                           =", best["coverage"])
    print("  ICP fitness                        =", best["fitness"])
    print("  ICP rmse                           =", best["rmse"])
    print("  final score                        =", best["score"])

    print_gpu_mem("after_alignment")
    print_ram_mem("after_alignment")

    print("\n[4/6] Save metric point clouds...")
    metric_pcd = np_to_pcd(best["full_final"])
    metric_vis_pcd = np_to_pcd(best["visible_final"])
    save_pcd(out_metric_pcd, metric_pcd)
    save_pcd(out_metric_vis, metric_vis_pcd)
    print("Saved full metric point cloud   :", out_metric_pcd)
    print("Saved visible metric point cloud:", out_metric_vis)

    print("\n[5/6] Reconstruct metric mesh...")
    mesh_iso_corr = 1.0
    mesh = reconstruct_mesh_from_pcd(metric_pcd)

    #  传进去mesh 输出中心的mesh
    # 新增：mesh主轴对齐 + 平移到原点
    mesh, mesh_align_axes, mesh_align_bbox_center = align_mesh_to_origin_and_axes(mesh)

    save_mesh(out_metric_mesh_ply, mesh)
    save_mesh(out_metric_mesh_obj, mesh)
    print("Saved metric mesh (PLY):", out_metric_mesh_ply)
    print("Saved metric mesh (OBJ):", out_metric_mesh_obj)

    print_gpu_mem("after_mesh")
    print_ram_mem("after_mesh")

    print("\n[6/6] Save metrics...")
    np.save(
        os.path.join(OUT_DIR, "best_alignment_metrics.npy"),
        {
            "scale": best["scale"],
            "mesh_iso_correction": float(mesh_iso_corr),
            "mesh_align_axes": mesh_align_axes,
            "mesh_align_bbox_center": mesh_align_bbox_center,
            "scale_pre": best["scale_pre"],
            "scale_correction": best["scale_correction"],
            "scale_correction_xy": best["scale_correction_xy"],
            "scale_correction_proj": best["scale_correction_proj"],
            "scale_correction_obj": best["scale_correction_obj"],
            "scale_correction_pre_obj": best["scale_correction_pre_obj"],
            "score": best["score"],
            "iou": best["iou"],
            "depth_mae": best["depth_mae"],
            "coverage": best["coverage"],
            "mean_cd": best["mean_cd"],
            "med_cd": best["med_cd"],
            "target_wh": best["target_wh"],
            "aligned_full_wh": best["aligned_full_wh"],
            "aligned_vis_wh": best["aligned_vis_wh"],
            "size_rel_err_full_wh": best["size_rel_err_full_wh"],
            "anchor_scale_targets": anchor_data["scale_targets"],
            "anchor_xy_extent": anchor_data["anchor_xy_extent"],
        },
        allow_pickle=True,
    )
    print("Saved debug files to:", OUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
