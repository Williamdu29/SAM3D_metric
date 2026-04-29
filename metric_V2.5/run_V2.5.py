import os
import sys
import copy
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import torch
import psutil
import trimesh
import tempfile
import time

# ============================================================
# User Config (V2)
# ============================================================
RGB_PATH = "/mnt/dct/data/replay/shampoo/images/77.png"
DEPTH_PATH = "/mnt/dct/data/replay/shampoo/depths/77.png"
MASK_PATH = "/mnt/dct/data/replay/shampoo/masks/maskdata/77.png"
K_PATH = "/mnt/dct/SKU/cookie/K.txt"

SAM3D_ROOT = "/home/dct/work/sam-3d-objects"
SAM3D_CONFIG = "/home/user/datas/hc/data/ckpts/sam3d-obj/models/checkpoints/pipeline.yaml"

RUN_SAM3D = True

# 可选 debug 输出；不给就不存
DEBUG_METRIC_PCD_PATH = None
DEBUG_METRIC_VIS_PCD_PATH = None

# 最终导出
EXPORT_METRIC_MESH_PATH = "/home/dct/work/sam-3d-objects/Outputs_V2.5/shampoo/sam3d_metric_mesh.ply"
EXPORT_METRIC_MESH_AXIS_ALIGNED_PATH = "/home/dct/work/sam-3d-objects/Outputs_V2.5/shampoo/sam3d_metric_mesh_axis_aligned.ply"

VOXEL_SIZE = 0.0025
ICP_MAX_ITER = 40

# ============================================================
# Mesh simplification config
# ============================================================
SIMPLIFY_MESH = True

# 目标面片数：60万 -> 5万，通常已经明显变小
TARGET_FACE_COUNT = 50000

# trimesh 简化强度：0 慢但质量好，10 快但质量差；建议 3~5
SIMPLIFY_AGGRESSION = 4

# 是否在简化后做基础清理
CLEAN_SIMPLIFIED_MESH = True

sys.path.append(os.path.join(SAM3D_ROOT, "notebook"))
from inference import Inference, load_image  # noqa: E402


# ============================================================
# Basic IO
# ============================================================
def load_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    return (mask > 0).astype(np.uint8)


def save_pcd(path, pcd):
    ok = o3d.io.write_point_cloud(path, pcd)
    if not ok:
        raise RuntimeError(f"Failed to save point cloud: {path}")


def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {path}")
    return pcd


def np_to_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd


def pcd_to_np(pcd):
    return np.asarray(pcd.points).astype(np.float64)


def make_temp_ply_path(prefix="sam3d_raw_", suffix=".ply"):
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return path


# ============================================================
# SAM3D inference
# ============================================================
def run_sam3d(rgb_path, mask_path, config_path, save_ply_path=None):
    """
    返回:
        raw_pcd_path: 原始点云缓存路径
        raw_mesh:     SAM3D 原生 mesh，优先取 output["glb"]
        output:       原始输出字典
    """
    inference = Inference(config_path, compile=False)
    image = load_image(rgb_path)
    mask = load_mask(mask_path)
    output = inference(image, mask, seed=42)

    print("check the output of SAM3D inference:")
    print(type(output))
    print(output.keys() if hasattr(output, "keys") else output)
    print("type(output['glb']) =", type(output["glb"]))
    print("type(output['gs'])  =", type(output["gs"]))

    if "glb" not in output:
        raise KeyError('SAM3D output does not contain "glb".')

    if "gs" not in output:
        raise KeyError('SAM3D output does not contain "gs".')

    raw_mesh = output["glb"]

    if save_ply_path is None:
        save_ply_path = make_temp_ply_path()

    output["gs"].save_ply(save_ply_path)

    if not os.path.exists(save_ply_path):
        raise RuntimeError(f"SAM3D gs.save_ply did not create file: {save_ply_path}")

    return save_ply_path, raw_mesh, output


# ============================================================
# Geometry helpers
# ============================================================
def voxel_downsample_pcd(pcd, voxel_size=VOXEL_SIZE):
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def robust_center(points):
    return np.median(points, axis=0)


def robust_extent(points, low=2.0, high=98.0):
    lo = np.percentile(points, low, axis=0)
    hi = np.percentile(points, high, axis=0)
    return hi - lo, lo, hi


def robust_extent_xy(points, low=2.0, high=98.0):
    lo = np.percentile(points[:, :2], low, axis=0)
    hi = np.percentile(points[:, :2], high, axis=0)
    return hi - lo, lo, hi


def compute_pca_frame(points):
    center = robust_center(points)
    X = points - center[None, :]
    cov = X.T @ X / max(len(X) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1.0
    return center, evecs, evals


def generate_sign_flip_rotations(base_R):
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
    return rots


def centered_similarity(points, scale=1.0, R=None, t=None, center=None):
    X = points.copy()
    if center is not None:
        X = X - center[None, :]
    if R is not None:
        X = X @ R.T
    X = X * float(scale)
    if center is not None:
        X = X + center[None, :]
    if t is not None:
        X = X + t[None, :]
    return X


def apply_4x4_to_points(points, T):
    X = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    Y = (T @ X.T).T
    return Y[:, :3]


def remove_outliers_np(points, nb_neighbors=20, std_ratio=1.5):
    if len(points) < max(nb_neighbors + 5, 20):
        return points
    pcd = np_to_pcd(points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd_to_np(pcd)


# ============================================================
# Mask / depth / anchor
# ============================================================
def largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return (labels == keep).astype(np.uint8)


def morph_clean(mask, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = largest_component(mask)
    return mask.astype(np.uint8)


def infer_depth_meters(depth):
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        raise ValueError("Depth image has no valid values.")
    vmax = float(np.percentile(valid, 99.5))
    if depth.dtype == np.uint16 or vmax > 100.0:
        return depth.astype(np.float32) / 1000.0, "mm_to_m"
    if vmax > 20.0:
        return depth.astype(np.float32) / 1000.0, "heuristic_mm_to_m"
    return depth.astype(np.float32), "already_m"


def depth_to_points(depth_m, mask, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ys, xs = np.where(mask > 0)
    z = depth_m[ys, xs]
    valid = np.isfinite(z) & (z > 1e-6)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    Z = z
    pts = np.stack([X, Y, Z], axis=1).astype(np.float32)
    return pts, xs, ys


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty mask.")
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return x0, y0, x1, y1


def erode_mask_adaptive(mask, ratio=0.12, min_k=3, max_k=31):
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


def estimate_anchor_scale_targets(mask, depth_m, K):
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


def build_anchor_data(rgb_path, depth_path, mask_path, K_path):
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
    scale_targets = estimate_anchor_scale_targets(mask_bin, depth_m, K)

    z_med = scale_targets["z_med"]
    band = max(0.035 * z_med, 0.012)
    valid_depth_mask = np.isfinite(depth_m) & (depth_m > 1e-6) & (np.abs(depth_m - z_med) <= band)
    final_mask = morph_clean(mask_bin & valid_depth_mask.astype(np.uint8), ksize=3)

    pts, _, _ = depth_to_points(depth_m, final_mask, K)
    pts = remove_outliers_np(pts, nb_neighbors=20, std_ratio=1.2)
    if len(pts) == 0:
        raise ValueError("Anchor point cloud is empty after filtering.")

    pcd = np_to_pcd(pts)
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


# ============================================================
# Alignment evaluation
# ============================================================
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


def robust_pca_extents(points, low=2.0, high=98.0):
    center, axes, _ = compute_pca_frame(points)
    proj = (points - center[None, :]) @ axes
    ext, lo, hi = robust_extent(proj, low=low, high=high)
    order = np.argsort(ext)[::-1]
    return ext[order], center, axes, lo[order], hi[order]


def solve_scale_object_dims(target_hw, src_ext_sorted, ww=0.20, wh=0.80):
    src_h = float(src_ext_sorted[0])
    src_w = float(src_ext_sorted[1])
    tgt_h = float(target_hw[1])
    tgt_w = float(target_hw[0])
    num = wh * src_h * tgt_h + ww * src_w * tgt_w
    den = wh * src_h * src_h + ww * src_w * src_w + 1e-12
    return max(float(num / den), 1e-8)


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


# ============================================================
# Main alignment search (point-cloud domain only)
# ============================================================
def search_metric_alignment(anchor_data, sam_pcd):
    anchor_pts = pcd_to_np(voxel_downsample_pcd(anchor_data["pcd"], VOXEL_SIZE))
    anchor_pts = remove_outliers_np(anchor_pts, nb_neighbors=20, std_ratio=1.2)
    if len(anchor_pts) == 0:
        raise RuntimeError("Anchor point cloud is empty after downsampling.")

    sam_pts = pcd_to_np(voxel_downsample_pcd(sam_pcd, VOXEL_SIZE))
    sam_pts = remove_outliers_np(sam_pts, nb_neighbors=20, std_ratio=1.5)
    if len(sam_pts) == 0:
        raise RuntimeError("SAM3D point cloud is empty after downsampling.")

    anchor_center = anchor_data["anchor_center"]
    target_xy = np.array([
        anchor_data["scale_targets"]["metric_w"],
        anchor_data["scale_targets"]["metric_h"],
    ], dtype=np.float64)
    target_z = float(anchor_data["scale_targets"]["z_med"])

    anchor_pca_center, anchor_axes, _ = compute_pca_frame(anchor_pts)
    sam_center, sam_axes, _ = compute_pca_frame(sam_pts)
    _ = anchor_pca_center

    base_R = anchor_axes @ sam_axes.T
    candidate_Rs = generate_sign_flip_rotations(base_R)

    best = None
    for ridx, R in enumerate(candidate_Rs):
        sam_rot_full = centered_similarity(sam_pts, scale=1.0, R=R, center=sam_center)
        full_xy_raw, _, _ = robust_extent_xy(sam_rot_full, low=2.0, high=98.0)
        scale0 = solve_scale_xy(target_xy, full_xy_raw)

        for mult in [0.94, 0.98, 1.00, 1.02, 1.06]:
            s = scale0 * mult
            sam_scaled_full = centered_similarity(sam_pts, scale=s, R=R, center=sam_center)
            sam_scaled_vis = extract_visible_surface_from_camera(sam_scaled_full)
            if len(sam_scaled_vis) < 100:
                continue

            t0 = translation_from_visible_center(sam_scaled_vis, anchor_center, target_z)
            sam_init_full = sam_scaled_full + t0[None, :]
            sam_init_vis = sam_scaled_vis + t0[None, :]

            threshold = max(0.008, 4.0 * VOXEL_SIZE)
            sam_refined_vis, T_icp, fitness, rmse = icp_refine_rigid(sam_init_vis, anchor_pts, threshold)
            sam_refined_full = apply_4x4_to_points(sam_init_full, T_icp)

            full_xy_after_icp, _, _ = robust_extent_xy(sam_refined_full, low=2.0, high=98.0)
            s_corr_xy = solve_scale_xy(target_xy, full_xy_after_icp)
            s_corr_xy = float(np.clip(s_corr_xy, 0.94, 1.06))

            sam_corr_full = centered_similarity(sam_refined_full, scale=s_corr_xy, center=anchor_center)
            sam_corr_vis = centered_similarity(sam_refined_vis, scale=s_corr_xy, center=anchor_center)
            t1 = translation_from_visible_center(sam_corr_vis, anchor_center, target_z)
            sam_corr_full = sam_corr_full + t1[None, :]
            sam_corr_vis = sam_corr_vis + t1[None, :]

            target_px_wh = np.array([
                anchor_data["scale_targets"]["pixel_w"],
                anchor_data["scale_targets"]["pixel_h"],
            ], dtype=np.float64)
            proj_px_wh = projected_mask_bbox_wh(sam_corr_vis, anchor_data, dilate_ksize=5)
            s_corr_proj = solve_projected_bbox_scale(target_px_wh, proj_px_wh, ww=0.18, wh=0.82)
            s_corr_proj = float(np.clip(s_corr_proj, 0.97, 1.00))

            s_corr_pre_obj = float(np.sqrt(s_corr_xy * s_corr_proj))
            s_corr_pre_obj = float(np.clip(s_corr_pre_obj, 0.97, 1.02))

            sam_mid_full = centered_similarity(sam_refined_full, scale=s_corr_pre_obj, center=anchor_center)
            sam_mid_vis = centered_similarity(sam_refined_vis, scale=s_corr_pre_obj, center=anchor_center)
            t_mid = translation_from_visible_center(sam_mid_vis, anchor_center, target_z)
            sam_mid_full = sam_mid_full + t_mid[None, :]
            sam_mid_vis = sam_mid_vis + t_mid[None, :]

            obj_ext_mid, _, _, _, _ = robust_pca_extents(sam_mid_full, low=2.0, high=98.0)
            s_corr_obj = solve_scale_object_dims(target_xy, obj_ext_mid, ww=0.18, wh=0.82)
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
                "R": R.copy(),
                "sam_center": sam_center.copy(),
                "anchor_center": anchor_center.copy(),

                "scale_pre": float(s),
                "t0": t0.copy(),
                "T_icp": T_icp.copy(),

                "scale_correction_xy": float(s_corr_xy),
                "scale_correction_proj": float(s_corr_proj),
                "scale_correction_obj": float(s_corr_obj),
                "scale_correction_pre_obj": float(s_corr_pre_obj),
                "scale_correction": float(s_corr),
                "t2": t2.copy(),

                "scale": float(s * s_corr),

                "fitness": float(fitness),
                "rmse": float(rmse),
                "mean_cd": metrics["mean_cd"],
                "med_cd": metrics["med_cd"],
                "iou": metrics["iou"],
                "depth_mae": metrics["depth_mae"],
                "coverage": metrics["coverage"],

                "target_wh": target_xy.copy(),
                "target_px_wh": target_px_wh.copy(),
                "projected_px_wh": np.array(
                    proj_px_wh_final if proj_px_wh_final is not None else [-1.0, -1.0],
                    dtype=np.float64
                ),
                "aligned_full_wh": full_xy_final,
                "aligned_vis_wh": vis_xy_final,
                "size_rel_err_full_wh": metrics["size_rel_err_full"],

                "full_final": sam_final_full,
                "visible_final": sam_final_vis,
            }
            if best is None or result["score"] < best["score"]:
                best = result

    if best is None:
        raise RuntimeError("Failed to find a valid alignment.")
    return best


# ============================================================
# Trimesh helpers
# ============================================================
def apply_alignment_to_vertices(vertices, align):
    V = np.asarray(vertices, dtype=np.float64).copy()

    V = centered_similarity(
        V,
        scale=align["scale_pre"],
        R=align["R"],
        center=align["sam_center"],
    )
    V = V + align["t0"][None, :]
    V = apply_4x4_to_points(V, align["T_icp"])
    V = centered_similarity(
        V,
        scale=align["scale_correction"],
        center=align["anchor_center"],
    )
    V = V + align["t2"][None, :]

    return V


def apply_alignment_to_trimesh(raw_mesh, align):
    """
    将点云域求出来的 metric alignment 应用到 SAM3D native mesh。

    这里统一返回 trimesh.Trimesh：
      - raw_mesh 是 Trimesh：直接变换 vertices
      - raw_mesh 是 Scene：先合并为单个 Trimesh，再变换 vertices

    这样后续 simplify / export / axis-align 都只处理单 mesh，最稳。
    """
    if isinstance(raw_mesh, trimesh.Scene):
        mesh = merge_scene_to_single_trimesh(raw_mesh)
    elif isinstance(raw_mesh, trimesh.Trimesh):
        mesh = raw_mesh.copy()
    else:
        raise TypeError(f"Unsupported mesh type: {type(raw_mesh)}")

    if len(mesh.vertices) == 0:
        raise ValueError("SAM3D native mesh has no vertices.")

    mesh.vertices = apply_alignment_to_vertices(mesh.vertices, align)

    return mesh


def ensure_parent_dir(file_path):
    """
    确保输出文件的父目录存在。
    例如:
        /a/b/c/out.ply
    会自动创建:
        /a/b/c
    """
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    if parent_dir != "":
        os.makedirs(parent_dir, exist_ok=True)


def export_mesh_as_ply(mesh, out_path):
    """
    将 metric mesh 导出为 .ply
    - 如果是 Trimesh：直接导出
    - 如果是 Scene：先合并成单个 Trimesh 再导出
    - 自动创建输出目录
    - 导出后检查文件是否真实存在
    """
    out_path = os.path.abspath(out_path)

    if not out_path.lower().endswith(".ply"):
        raise ValueError(f"PLY export path must end with .ply, got: {out_path}")

    parent_dir = os.path.dirname(out_path)
    os.makedirs(parent_dir, exist_ok=True)

    print("[EXPORT] parent dir        =", parent_dir)
    print("[EXPORT] parent dir exists =", os.path.exists(parent_dir))
    print("[EXPORT] target file       =", out_path)

    if isinstance(mesh, trimesh.Trimesh):
        if len(mesh.vertices) == 0:
            raise ValueError("Cannot export empty Trimesh: mesh has no vertices.")
        if len(mesh.faces) == 0:
            raise ValueError("Cannot export empty Trimesh: mesh has no faces.")

        print("[EXPORT] mesh vertices     =", f"{len(mesh.vertices):,}")
        print("[EXPORT] mesh faces        =", f"{len(mesh.faces):,}")

        mesh.export(out_path, file_type="ply")

    elif isinstance(mesh, trimesh.Scene):
        merged = merge_scene_to_single_trimesh(mesh)
        if len(merged.vertices) == 0:
            raise ValueError("Cannot export empty merged mesh: mesh has no vertices.")
        if len(merged.faces) == 0:
            raise ValueError("Cannot export empty merged mesh: mesh has no faces.")

        print("[EXPORT] merged vertices   =", f"{len(merged.vertices):,}")
        print("[EXPORT] merged faces      =", f"{len(merged.faces):,}")

        merged.export(out_path, file_type="ply")

    else:
        raise TypeError(f"Unsupported mesh type for PLY export: {type(mesh)}")

    if not os.path.exists(out_path):
        raise RuntimeError(f"Export finished but file does not exist: {out_path}")

    file_size_mb = os.path.getsize(out_path) / 1024**2
    print("[EXPORT] saved successfully =", out_path)
    print("[EXPORT] file size MB       =", f"{file_size_mb:.2f}")


def merge_scene_to_single_trimesh(scene):
    """
    将 trimesh.Scene 合并成单个 trimesh.Trimesh。

    优先使用 scene.dump(concatenate=True)，这样可以尽量保留 Scene graph
    中每个 geometry node 的 transform。
    """
    if isinstance(scene, trimesh.Trimesh):
        return scene.copy()

    if not isinstance(scene, trimesh.Scene):
        raise TypeError(f"Unsupported type: {type(scene)}")

    try:
        merged = scene.dump(concatenate=True)
        if isinstance(merged, trimesh.Trimesh) and len(merged.vertices) > 0:
            return merged
    except Exception as e:
        print("[MESH] scene.dump(concatenate=True) failed, fallback to geometry concat.")
        print("       reason:", repr(e))

    meshes = []
    for geom in scene.geometry.values():
        if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
            meshes.append(geom.copy())

    if len(meshes) == 0:
        raise ValueError("No valid mesh in trimesh.Scene")

    return trimesh.util.concatenate(meshes)


def pca_align_trimesh_to_origin(raw_mesh):
    """
    把 mesh 做 PCA 主轴对齐，并强制：
      - 最长轴 -> y
      - 第二长轴 -> x
      - 第三轴 -> z
    然后再把 AABB 中心平移到原点
    """
    if isinstance(raw_mesh, trimesh.Scene):
        mesh = merge_scene_to_single_trimesh(raw_mesh)
    else:
        mesh = raw_mesh.copy()

    V = np.asarray(mesh.vertices, dtype=np.float64)
    if len(V) == 0:
        return mesh, np.eye(3), np.zeros(3, dtype=np.float64)

    center, axes, _ = compute_pca_frame(V)

    # axes 的列目前是：
    # axes[:, 0] = 最长轴
    # axes[:, 1] = 第二长轴
    # axes[:, 2] = 第三轴
    #
    # 我们想要：
    # x <- 第二长轴
    # y <- 最长轴
    # z <- 第三轴
    axes_reordered = np.column_stack([
        axes[:, 1],   # x
        axes[:, 0],   # y
        axes[:, 2],   # z
    ])

    # 先尽量稳定方向
    if axes_reordered[0, 0] < 0:
        axes_reordered[:, 0] *= -1.0
    if axes_reordered[1, 1] < 0:
        axes_reordered[:, 1] *= -1.0

    # 保证右手系
    if np.linalg.det(axes_reordered) < 0:
        axes_reordered[:, 2] *= -1.0

    # 投影到新的坐标轴顺序
    V_aligned = (V - center[None, :]) @ axes_reordered

    # AABB 中心移到原点
    bbox_min = V_aligned.min(axis=0)
    bbox_max = V_aligned.max(axis=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    V_aligned = V_aligned - bbox_center[None, :]

    mesh.vertices = V_aligned
    return mesh, axes_reordered, bbox_center


def get_mesh_stats(mesh):
    """
    返回 mesh 的顶点数和面片数。
    支持 trimesh.Trimesh 和 trimesh.Scene。
    """
    if isinstance(mesh, trimesh.Trimesh):
        return int(len(mesh.vertices)), int(len(mesh.faces))

    if isinstance(mesh, trimesh.Scene):
        total_vertices = 0
        total_faces = 0
        for geom in mesh.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                total_vertices += int(len(geom.vertices))
                total_faces += int(len(geom.faces))
        return total_vertices, total_faces

    raise TypeError(f"Unsupported mesh type for stats: {type(mesh)}")


def print_mesh_stats(tag, mesh):
    v, f = get_mesh_stats(mesh)
    print(f"[MESH] {tag:<35} vertices = {v:,}, faces = {f:,}")


def trimesh_to_open3d_mesh(mesh):
    """
    trimesh.Trimesh -> open3d.geometry.TriangleMesh
    只保留 vertices / faces / vertex colors。
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got: {type(mesh)}")

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))

    # 尽量保留 vertex color
    try:
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            vc = np.asarray(mesh.visual.vertex_colors)
            if len(vc) == len(mesh.vertices):
                if vc.shape[1] >= 3:
                    colors = vc[:, :3].astype(np.float64)
                    if colors.max() > 1.0:
                        colors = colors / 255.0
                    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    except Exception:
        pass

    return o3d_mesh


def open3d_to_trimesh_mesh(o3d_mesh, reference_mesh=None):
    """
    open3d.geometry.TriangleMesh -> trimesh.Trimesh
    尽量保留 Open3D 里的 vertex color。
    """
    vertices = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    faces = np.asarray(o3d_mesh.triangles, dtype=np.int64)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    try:
        colors = np.asarray(o3d_mesh.vertex_colors)
        if len(colors) == len(vertices):
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
            mesh.visual.vertex_colors = np.concatenate([colors, alpha], axis=1)
    except Exception:
        pass

    return mesh


def clean_trimesh_basic(mesh):
    """
    基础 mesh 清理。
    注意：不同 trimesh 版本的 API 可能略有差异，所以这里都用 try 包住。
    """
    mesh = mesh.copy()

    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass

    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.merge_vertices()
    except Exception:
        pass

    try:
        mesh.fix_normals()
    except Exception:
        pass

    return mesh


def simplify_trimesh_with_trimesh(mesh, target_face_count, aggression=4):
    """
    优先使用 trimesh 自带的 quadric decimation。
    """
    if len(mesh.faces) <= target_face_count:
        return mesh.copy()

    try:
        simplified = mesh.simplify_quadric_decimation(
            face_count=int(target_face_count),
            aggression=int(aggression),
        )
    except TypeError:
        # 兼容某些旧版本参数名不完整的情况
        simplified = mesh.simplify_quadric_decimation(
            face_count=int(target_face_count),
        )

    if simplified is None:
        raise RuntimeError("trimesh simplify_quadric_decimation returned None.")

    if not isinstance(simplified, trimesh.Trimesh):
        raise TypeError(f"trimesh simplification returned unsupported type: {type(simplified)}")

    return simplified


def simplify_trimesh_with_open3d(mesh, target_face_count):
    """
    使用 Open3D 的 quadric decimation 作为 fallback。
    """
    if len(mesh.faces) <= target_face_count:
        return mesh.copy()

    o3d_mesh = trimesh_to_open3d_mesh(mesh)

    # Open3D 简化
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(target_face_count)
    )

    # 简化后清理
    try:
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_non_manifold_edges()
        o3d_mesh.remove_unreferenced_vertices()
        o3d_mesh.compute_vertex_normals()
    except Exception:
        pass

    return open3d_to_trimesh_mesh(o3d_mesh, reference_mesh=mesh)


def simplify_single_trimesh(mesh, target_face_count=50000, aggression=4, clean=True):
    """
    对单个 trimesh.Trimesh 做简化。
    逻辑：
      1. 如果面片数已经小于目标，不处理
      2. 优先用 trimesh 简化
      3. 如果 trimesh 简化失败，用 Open3D fallback
      4. 最后做基础清理
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got: {type(mesh)}")

    original_faces = int(len(mesh.faces))
    original_vertices = int(len(mesh.vertices))

    if original_faces == 0 or original_vertices == 0:
        raise ValueError("Cannot simplify empty mesh.")

    if original_faces <= target_face_count:
        print(
            f"[MESH] skip simplification: faces = {original_faces:,}, "
            f"target = {target_face_count:,}"
        )
        return clean_trimesh_basic(mesh) if clean else mesh.copy()

    print(
        f"[MESH] simplifying mesh: vertices = {original_vertices:,}, "
        f"faces = {original_faces:,}, target faces = {target_face_count:,}"
    )

    try:
        simplified = simplify_trimesh_with_trimesh(
            mesh,
            target_face_count=target_face_count,
            aggression=aggression,
        )
        method = "trimesh"
    except Exception as e:
        print("[MESH] trimesh simplification failed, fallback to Open3D.")
        print("       reason:", repr(e))
        simplified = simplify_trimesh_with_open3d(
            mesh,
            target_face_count=target_face_count,
        )
        method = "open3d"

    if clean:
        simplified = clean_trimesh_basic(simplified)

    new_vertices = int(len(simplified.vertices))
    new_faces = int(len(simplified.faces))

    print(
        f"[MESH] simplification done by {method}: "
        f"vertices {original_vertices:,} -> {new_vertices:,}, "
        f"faces {original_faces:,} -> {new_faces:,}"
    )

    return simplified

def simplify_mesh(mesh, target_face_count=50000, aggression=4, clean=True):
    """
    简化 trimesh.Trimesh 或 trimesh.Scene。

    推荐策略：
      - Trimesh：直接简化到 target_face_count
      - Scene：先合并为单个 Trimesh，再整体简化

    这里选择“先合并再简化”，因为你最终本来就是导出 PLY；
    PLY 更适合单 mesh，下游处理也更方便。
    """
    if isinstance(mesh, trimesh.Trimesh):
        return simplify_single_trimesh(
            mesh,
            target_face_count=target_face_count,
            aggression=aggression,
            clean=clean,
        )

    if isinstance(mesh, trimesh.Scene):
        print("[MESH] input is trimesh.Scene, merge to single Trimesh before simplification.")
        merged = merge_scene_to_single_trimesh(mesh)
        return simplify_single_trimesh(
            merged,
            target_face_count=target_face_count,
            aggression=aggression,
            clean=clean,
        )

    raise TypeError(f"Unsupported mesh type for simplification: {type(mesh)}")


# ============================================================
# Timing helpers
# ============================================================
def format_seconds(seconds):
    """
    将秒数格式化为更容易读的形式。
    例如:
        65.23 -> 1m 5.23s
        3725.8 -> 1h 2m 5.80s
    """
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.2f}s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.2f}s"


def print_step_time(step_name, elapsed):
    print(f"[TIMER] {step_name:<45} {format_seconds(elapsed)}")



# ============================================================
# Diagnostics
# ============================================================
def print_gpu_mem(tag=""):
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available")
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.max_memory_allocated() / 1024**3
    reserv = torch.cuda.max_memory_reserved() / 1024**3
    now_alloc = torch.cuda.memory_allocated() / 1024**3
    now_reserv = torch.cuda.memory_reserved() / 1024**3
    print(f"[{tag}] GPU now allocated   = {now_alloc:.2f} GB")
    print(f"[{tag}] GPU now reserved    = {now_reserv:.2f} GB")
    print(f"[{tag}] GPU peak allocated  = {alloc:.2f} GB")
    print(f"[{tag}] GPU peak reserved   = {reserv:.2f} GB")


def print_ram_mem(tag=""):
    vm = psutil.virtual_memory()
    used = (vm.total - vm.available) / 1024**3
    total = vm.total / 1024**3
    print(f"[{tag}] RAM used            = {used:.2f} / {total:.2f} GB")


# ============================================================
# Main
# ============================================================
def main():
    tmp_raw_ply = None
    timing = {}

    print("[DEBUG] running script              =", os.path.abspath(__file__))
    print("[DEBUG] current working directory   =", os.getcwd())
    print("[DEBUG] EXPORT_METRIC_MESH_PATH     =", EXPORT_METRIC_MESH_PATH)
    print("[DEBUG] EXPORT_AXIS_ALIGNED_PATH    =", EXPORT_METRIC_MESH_AXIS_ALIGNED_PATH)

    total_start = time.perf_counter()

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print_gpu_mem("start")
        print_ram_mem("start")

        # ------------------------------------------------------------
        # [1/5] Build anchor data
        # ------------------------------------------------------------
        step_start = time.perf_counter()

        print("[1/5] Build anchor data from RGB-D...")
        anchor_data = build_anchor_data(RGB_PATH, DEPTH_PATH, MASK_PATH, K_PATH)

        st = anchor_data["scale_targets"]
        print("Anchor diagnostics:")
        print("  depth mode                         =", anchor_data["depth_mode"])
        print("  erode kernel                       =", st["erode_k"])
        print("  used eroded inner mask             =", st["used_erode"])
        print("  bbox pixel width                   =", st["pixel_w"])
        print("  bbox pixel height                  =", st["pixel_h"])
        print("  robust median depth [m]            =", st["z_med"])
        print("  robust depth q10/q90 [m]           =", st["z_q10"], st["z_q90"])
        print("  target width [m]                   =", st["metric_w"])
        print("  target height [m]                  =", st["metric_h"])
        print("  anchor visible XY extent [m]       =", anchor_data["anchor_xy_extent"])

        print_gpu_mem("after_anchor")
        print_ram_mem("after_anchor")

        timing["1_build_anchor"] = time.perf_counter() - step_start
        print_step_time("[1/5] Build anchor data", timing["1_build_anchor"])

        # ------------------------------------------------------------
        # [2/5] Run SAM3D
        # ------------------------------------------------------------
        step_start = time.perf_counter()

        print("\n[2/5] Run SAM3D and get raw point cloud + raw mesh...")
        if not RUN_SAM3D:
            raise RuntimeError("This V2 code expects RUN_SAM3D=True.")

        tmp_raw_ply = make_temp_ply_path()
        raw_sam_path, raw_mesh, output = run_sam3d(
            RGB_PATH,
            MASK_PATH,
            SAM3D_CONFIG,
            save_ply_path=tmp_raw_ply,
        )

        print("SAM3D raw point cloud:", raw_sam_path)
        print("SAM3D raw mesh type   :", type(raw_mesh))

        print_gpu_mem("after_sam3d")
        print_ram_mem("after_sam3d")

        sam_pcd = load_pcd(raw_sam_path)

        timing["2_run_sam3d"] = time.perf_counter() - step_start
        print_step_time("[2/5] Run SAM3D", timing["2_run_sam3d"])

        # ------------------------------------------------------------
        # [3/5] Solve metric alignment
        # ------------------------------------------------------------
        step_start = time.perf_counter()

        print("\n[3/5] Solve metric scale/pose on point cloud...")
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
        print("  chamfer(mean) [m]                  =", best["mean_cd"])
        print("  chamfer(median) [m]                =", best["med_cd"])
        print("  mask IoU                           =", best["iou"])
        print("  depth MAE [m]                      =", best["depth_mae"])
        print("  coverage                           =", best["coverage"])
        print("  ICP fitness                        =", best["fitness"])
        print("  ICP rmse                           =", best["rmse"])
        print("  final score                        =", best["score"])

        if DEBUG_METRIC_PCD_PATH is not None:
            save_pcd(DEBUG_METRIC_PCD_PATH, np_to_pcd(best["full_final"]))
            print("Saved debug metric full pcd:", DEBUG_METRIC_PCD_PATH)

        if DEBUG_METRIC_VIS_PCD_PATH is not None:
            save_pcd(DEBUG_METRIC_VIS_PCD_PATH, np_to_pcd(best["visible_final"]))
            print("Saved debug metric visible pcd:", DEBUG_METRIC_VIS_PCD_PATH)

        timing["3_solve_alignment"] = time.perf_counter() - step_start
        print_step_time("[3/5] Solve metric alignment", timing["3_solve_alignment"])

        # ------------------------------------------------------------
        # [4/5] Apply transform and export mesh
        # ------------------------------------------------------------
        step_start = time.perf_counter()

        print("\n[4/5] Apply solved transform to SAM3D native mesh...")
        metric_mesh = apply_alignment_to_trimesh(raw_mesh, best)

        print_mesh_stats("metric mesh before simplification", metric_mesh)

        if SIMPLIFY_MESH:
            metric_mesh = simplify_mesh(
                metric_mesh,
                target_face_count=TARGET_FACE_COUNT,
                aggression=SIMPLIFY_AGGRESSION,
                clean=CLEAN_SIMPLIFIED_MESH,
            )
            print_mesh_stats("metric mesh after simplification", metric_mesh)
        else:
            print("[MESH] simplification disabled.")

        print("[DEBUG] final metric mesh export abs path =", os.path.abspath(EXPORT_METRIC_MESH_PATH))
        export_mesh_as_ply(metric_mesh, EXPORT_METRIC_MESH_PATH)
        print("Saved metric mesh (.ply):", os.path.abspath(EXPORT_METRIC_MESH_PATH))

        if EXPORT_METRIC_MESH_AXIS_ALIGNED_PATH is not None:
            axis_mesh, mesh_align_axes, mesh_align_bbox_center = pca_align_trimesh_to_origin(metric_mesh)

            print_mesh_stats("axis-aligned mesh before export", axis_mesh)

            print("[DEBUG] final axis-aligned mesh export abs path =", os.path.abspath(EXPORT_METRIC_MESH_AXIS_ALIGNED_PATH))
            export_mesh_as_ply(axis_mesh, EXPORT_METRIC_MESH_AXIS_ALIGNED_PATH)
            print("Saved axis-aligned metric mesh (.ply):", os.path.abspath(EXPORT_METRIC_MESH_AXIS_ALIGNED_PATH))
            print("  mesh_align_axes shape =", mesh_align_axes.shape)
            print("  mesh_align_bbox_center =", mesh_align_bbox_center)

        print_gpu_mem("after_mesh")
        print_ram_mem("after_mesh")

        timing["4_apply_export_mesh"] = time.perf_counter() - step_start
        print_step_time("[4/5] Apply transform and export mesh", timing["4_apply_export_mesh"])

        # ------------------------------------------------------------
        # [5/5] Final summary
        # ------------------------------------------------------------
        step_start = time.perf_counter()

        print("\n[5/5] Done.")
        print("V2 pipeline summary:")
        print("  - point cloud is used only for metric alignment")
        print("  - final mesh comes from SAM3D native mesh (output['glb'])")
        print("  - no mesh reconstruction from aligned point cloud")

        timing["5_summary"] = time.perf_counter() - step_start
        print_step_time("[5/5] Final summary", timing["5_summary"])

    finally:
        cleanup_start = time.perf_counter()

        if tmp_raw_ply is not None and os.path.exists(tmp_raw_ply):
            try:
                os.remove(tmp_raw_ply)
                print("Removed temp raw point cloud:", tmp_raw_ply)
            except Exception as e:
                print("Warning: failed to remove temp raw point cloud:", e)

        timing["cleanup"] = time.perf_counter() - cleanup_start
        total_elapsed = time.perf_counter() - total_start

        print("\n" + "=" * 72)
        print("Timing summary")
        print("=" * 72)

        ordered_keys = [
            "1_build_anchor",
            "2_run_sam3d",
            "3_solve_alignment",
            "4_apply_export_mesh",
            "5_summary",
            "cleanup",
        ]

        for key in ordered_keys:
            if key in timing:
                print(f"{key:<28} {format_seconds(timing[key])}")

        print("-" * 72)
        print(f"{'total':<28} {format_seconds(total_elapsed)}")
        print("=" * 72)


if __name__ == "__main__":
    main()