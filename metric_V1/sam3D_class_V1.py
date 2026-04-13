import os
import sys
import time
import json
import copy
from contextlib import contextmanager

import cv2
import numpy as np
import open3d as o3d
from PIL import Image


class SAM3DReconstructor:
    """
    用法：
        reconstructor = SAM3DReconstructor(
            sam3d_root="/home/dct/work/sam-3d-objects",
            sam3d_config="/home/user/datas/hc/data/ckpts/sam3d-obj/models/checkpoints/pipeline.yaml",
            out_dir="/home/dct/work/sam-3d-objects/cookie_output",
        )

        result = reconstructor.recon(
            rgb_path="xxx.jpg",
            depth_path="xxx.png",
            mask_path="xxx.png",
            k_path="K.txt",
            run_sam3d=True,
        )
    """

    def __init__(
        self,
        sam3d_root,
        sam3d_config,
        out_dir=None,
        voxel_size=0.0025,
        icp_max_iter=40,
        mesh_poisson_depth=8,
        mesh_density_q=0.02,
        sam_compile=False,
        verbose=True,
        ):
        self.sam3d_root = sam3d_root
        self.sam3d_config = sam3d_config
        self.default_out_dir = out_dir

        self.voxel_size = voxel_size
        self.icp_max_iter = icp_max_iter
        self.mesh_poisson_depth = mesh_poisson_depth
        self.mesh_density_q = mesh_density_q
        self.sam_compile = sam_compile
        self.verbose = verbose

        # 只有提供了默认输出目录时才创建
        if self.default_out_dir is not None:
            os.makedirs(self.default_out_dir, exist_ok=True)

        # 关键：必须先初始化，再用 _timer
        self.init_timings = {}
        self.total_timings = {}

        notebook_dir = os.path.join(self.sam3d_root, "notebook")
        if notebook_dir not in sys.path:
            sys.path.append(notebook_dir)

        with self._timer("init/load_sam3d_model"):
            from inference import Inference, load_image  # noqa: E402

            self.Inference = Inference
            self.load_image = load_image
            self.inference = self.Inference(self.sam3d_config, compile=self.sam_compile)

    # ============================================================
    # Timer helpers
    # ============================================================
    @contextmanager
    def _timer(self, name):
        start = time.perf_counter()
        if self.verbose:
            print(f"[Timer Start] {name}")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if not hasattr(self, "total_timings") or self.total_timings is None:
                self.total_timings = {}
            self.total_timings[name] = elapsed
            if self.verbose:
                print(f"[Timer End]   {name}: {elapsed:.4f}s")

    def _reset_timings(self):
        self.total_timings = {}

    def _save_timings(self, save_path):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.total_timings, f, indent=2, ensure_ascii=False)

    def _print_timing_summary(self):
        print("\n================ Timing Summary ================")
        total = 0.0
        for k, v in self.total_timings.items():
            print(f"{k:40s}: {v:.4f}s")
            total += v
        print("------------------------------------------------")
        print(f"{'TOTAL':40s}: {total:.4f}s")
        print("================================================\n")

    # ============================================================
    # Basic I/O
    # ============================================================
    @staticmethod
    def load_mask(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        return (mask > 0).astype(np.uint8)

    def run_sam3d(self, rgb_path, mask_path, save_path, seed=42):
        image = self.load_image(rgb_path)
        mask = self.load_mask(mask_path)
        output = self.inference(image, mask, seed=seed)
        output["gs"].save_ply(save_path)
        return save_path

    @staticmethod
    def load_pcd(path):
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            raise ValueError(f"Empty point cloud: {path}")
        return pcd

    @staticmethod
    def save_pcd(path, pcd):
        ok = o3d.io.write_point_cloud(path, pcd)
        if not ok:
            raise RuntimeError(f"Failed to save point cloud: {path}")

    @staticmethod
    def save_mesh(path, mesh):
        ok = o3d.io.write_triangle_mesh(path, mesh, write_ascii=False)
        if not ok:
            raise RuntimeError(f"Failed to save mesh: {path}")

    @staticmethod
    def np_to_pcd(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        return pcd

    @staticmethod
    def pcd_to_np(pcd):
        return np.asarray(pcd.points).astype(np.float64)

    def voxel_downsample_pcd(self, pcd, voxel_size=None):
        if voxel_size is None:
            voxel_size = self.voxel_size
        return pcd.voxel_down_sample(voxel_size=voxel_size)

    # ============================================================
    # Geometry utils
    # ============================================================
    @staticmethod
    def robust_center(points):
        return np.median(points, axis=0)

    @staticmethod
    def robust_extent(points, low=2.0, high=98.0):
        lo = np.percentile(points, low, axis=0)
        hi = np.percentile(points, high, axis=0)
        return hi - lo, lo, hi

    @staticmethod
    def robust_extent_xy(points, low=2.0, high=98.0):
        lo = np.percentile(points[:, :2], low, axis=0)
        hi = np.percentile(points[:, :2], high, axis=0)
        return hi - lo, lo, hi

    def compute_pca_frame(self, points):
        center = self.robust_center(points)
        X = points - center[None, :]
        cov = X.T @ X / max(len(X) - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        if np.linalg.det(evecs) < 0:
            evecs[:, 2] *= -1.0
        return center, evecs, evals

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def apply_4x4_to_points(points, T):
        X = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
        Y = (T @ X.T).T
        return Y[:, :3]

    # ============================================================
    # Mask / depth helpers
    # ============================================================
    @staticmethod
    def largest_component(mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if num_labels <= 1:
            return mask.astype(np.uint8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        return (labels == keep).astype(np.uint8)

    def morph_clean(self, mask, ksize=5):
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = self.largest_component(mask)
        return mask.astype(np.uint8)

    @staticmethod
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

    @staticmethod
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

    def remove_outliers_np(self, points, nb_neighbors=20, std_ratio=1.5):
        if len(points) < max(nb_neighbors + 5, 20):
            return points
        pcd = self.np_to_pcd(points)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return self.pcd_to_np(pcd)

    @staticmethod
    def mask_bbox(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            raise ValueError("Empty mask.")
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        return x0, y0, x1, y1

    def erode_mask_adaptive(self, mask, ratio=0.12, min_k=3, max_k=31):
        x0, y0, x1, y1 = self.mask_bbox(mask)
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

    def estimate_anchor_scale_targets(self, mask, depth_m, K):
        mask = mask.astype(np.uint8)
        inner_mask, erode_k, used_erode = self.erode_mask_adaptive(mask, ratio=0.12)

        valid_depth_inner = depth_m[inner_mask > 0]
        valid_depth_inner = valid_depth_inner[
            np.isfinite(valid_depth_inner) & (valid_depth_inner > 1e-6)
        ]
        if valid_depth_inner.size < 20:
            valid_depth_inner = depth_m[mask > 0]
            valid_depth_inner = valid_depth_inner[
                np.isfinite(valid_depth_inner) & (valid_depth_inner > 1e-6)
            ]
            if valid_depth_inner.size == 0:
                raise ValueError("No valid depth inside mask.")

        z_med = float(np.median(valid_depth_inner))
        z_q10 = float(np.percentile(valid_depth_inner, 10))
        z_q90 = float(np.percentile(valid_depth_inner, 90))

        x0, y0, x1, y1 = self.mask_bbox(mask)
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

    def robust_pca_extents(self, points, low=2.0, high=98.0):
        center, axes, _ = self.compute_pca_frame(points)
        proj = (points - center[None, :]) @ axes
        ext, lo, hi = self.robust_extent(proj, low=low, high=high)
        order = np.argsort(ext)[::-1]
        return ext[order], center, axes, lo[order], hi[order]

    @staticmethod
    def solve_scale_object_dims(target_hw, src_ext_sorted, ww=0.20, wh=0.80):
        src_h = float(src_ext_sorted[0])
        src_w = float(src_ext_sorted[1])
        tgt_h = float(target_hw[1])
        tgt_w = float(target_hw[0])
        num = wh * src_h * tgt_h + ww * src_w * tgt_w
        den = wh * src_h * src_h + ww * src_w * src_w + 1e-12
        return max(float(num / den), 1e-8)

    @staticmethod
    def isotropically_scale_mesh_about_center(mesh, scale, center):
        mesh = copy.deepcopy(mesh)
        V = np.asarray(mesh.vertices).astype(np.float64)
        V = (V - center[None, :]) * float(scale) + center[None, :]
        mesh.vertices = o3d.utility.Vector3dVector(V)
        mesh.compute_vertex_normals()
        return mesh

    def calibrate_mesh_to_pointcloud(self, mesh, ref_points):
        mesh_pts = np.asarray(mesh.vertices).astype(np.float64)
        if len(mesh_pts) == 0 or len(ref_points) == 0:
            return mesh, 1.0, np.array([-1.0, -1.0], dtype=np.float64)
        mesh_ext, mesh_center, _, _, _ = self.robust_pca_extents(mesh_pts, low=2.0, high=98.0)
        ref_ext, ref_center, _, _, _ = self.robust_pca_extents(ref_points, low=2.0, high=98.0)
        s_mesh = self.solve_scale_object_dims(
            np.array([ref_ext[1], ref_ext[0]], dtype=np.float64),
            mesh_ext,
            ww=0.25,
            wh=0.75,
        )
        s_mesh = float(np.clip(s_mesh, 0.94, 1.02))
        center = 0.5 * (mesh_center + ref_center)
        mesh_out = self.isotropically_scale_mesh_about_center(mesh, s_mesh, center)
        mesh_ext_after, _, _, _, _ = self.robust_pca_extents(
            np.asarray(mesh_out.vertices).astype(np.float64), low=2.0, high=98.0
        )
        return mesh_out, s_mesh, mesh_ext_after[:2].copy()

    # ============================================================
    # Anchor build
    # ============================================================
    def build_anchor_data(self, rgb_path, depth_path, mask_path, K):
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

        
        K = np.asarray(K, dtype=float)

        if K.size != 9:
            raise ValueError(
                f"K must contain exactly 9 elements, but got size={K.size}, shape={K.shape}"
            )

        K = K.reshape(3, 3)

        if not np.isfinite(K).all():
            raise ValueError("K contains NaN or Inf.")


        depth_m, depth_mode = self.infer_depth_meters(depth)

        mask_bin = self.morph_clean((mask > 0).astype(np.uint8), ksize=5)
        scale_targets = self.estimate_anchor_scale_targets(mask_bin, depth_m, K)

        z_med = scale_targets["z_med"]
        band = max(0.035 * z_med, 0.012)
        valid_depth_mask = np.isfinite(depth_m) & (depth_m > 1e-6) & (np.abs(depth_m - z_med) <= band)
        final_mask = self.morph_clean(mask_bin & valid_depth_mask.astype(np.uint8), ksize=3)

        pts, _, _ = self.depth_to_points(depth_m, final_mask, K)
        pts = self.remove_outliers_np(pts, nb_neighbors=20, std_ratio=1.2)
        if len(pts) == 0:
            raise ValueError("Anchor point cloud is empty after filtering.")

        pcd = self.np_to_pcd(pts)
        anchor_center = self.robust_center(pts)
        anchor_xy_extent, _, _ = self.robust_extent_xy(pts, low=2.0, high=98.0)

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
    # Alignment helpers
    # ============================================================
    def estimate_hidden_point_radius(self, points):
        center = self.robust_center(points)
        d = np.linalg.norm(points - center[None, :], axis=1)
        return float(max(np.percentile(d, 95) * 6.0, 1e-3))

    def extract_visible_surface_from_camera(self, points, camera_location=None, radius=None):
        if camera_location is None:
            camera_location = np.zeros(3, dtype=np.float64)
        pcd = self.np_to_pcd(points)
        if len(pcd.points) == 0:
            return points.copy()
        if radius is None:
            radius = self.estimate_hidden_point_radius(points)
        _, visible_idx = pcd.hidden_point_removal(camera_location, radius)
        vis = pcd.select_by_index(visible_idx)
        return self.pcd_to_np(vis)

    def compute_symmetric_chamfer(self, src_pts, tgt_pts):
        src = self.np_to_pcd(src_pts)
        tgt = self.np_to_pcd(tgt_pts)
        d1 = np.asarray(src.compute_point_cloud_distance(tgt), dtype=np.float64)
        d2 = np.asarray(tgt.compute_point_cloud_distance(src), dtype=np.float64)
        if len(d1) == 0 or len(d2) == 0:
            return np.inf, np.inf
        return float(d1.mean() + d2.mean()), float(np.median(d1) + np.median(d2))

    @staticmethod
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

    def render_depth_and_mask(self, points, K, H, W):
        depth_img = np.full((H, W), np.inf, dtype=np.float32)
        mask_img = np.zeros((H, W), dtype=np.uint8)
        ui, vi, zi = self.project_points(points, K, H, W)
        if len(ui) == 0:
            return depth_img, mask_img
        order = np.argsort(zi)
        ui = ui[order]
        vi = vi[order]
        zi = zi[order]
        depth_img[vi, ui] = np.minimum(depth_img[vi, ui], zi)
        mask_img[np.isfinite(depth_img)] = 1
        return depth_img, mask_img

    @staticmethod
    def binary_iou(a, b):
        a = a.astype(bool)
        b = b.astype(bool)
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        if union == 0:
            return 0.0
        return float(inter / union)

    def projection_metrics(self, points, anchor_data):
        H = anchor_data["H"]
        W = anchor_data["W"]
        K = anchor_data["K"]
        depth_gt = anchor_data["depth_m"]
        mask_gt = anchor_data["mask_bin"] > 0
        depth_pred, mask_pred = self.render_depth_and_mask(points, K, H, W)
        iou = self.binary_iou(mask_gt, mask_pred > 0)
        overlap = mask_gt & np.isfinite(depth_pred) & np.isfinite(depth_gt) & (depth_gt > 1e-6)
        if overlap.sum() == 0:
            return {"iou": iou, "depth_mae": np.inf, "coverage": 0.0}
        abs_depth = np.abs(depth_pred[overlap] - depth_gt[overlap])
        return {
            "iou": iou,
            "depth_mae": float(np.mean(abs_depth)),
            "coverage": float(overlap.sum() / max(mask_gt.sum(), 1)),
        }

    def projected_mask_bbox_wh(self, points, anchor_data, dilate_ksize=5):
        H = anchor_data["H"]
        W = anchor_data["W"]
        K = anchor_data["K"]
        _, mask_pred = self.render_depth_and_mask(points, K, H, W)
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

    @staticmethod
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

    def icp_refine_rigid(self, source_pts, target_pts, threshold):
        src = self.np_to_pcd(source_pts)
        tgt = self.np_to_pcd(target_pts)
        reg = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.icp_max_iter),
        )
        refined = self.apply_4x4_to_points(source_pts, reg.transformation)
        return refined, reg.transformation, float(reg.fitness), float(reg.inlier_rmse)

    @staticmethod
    def solve_scale_xy(target_xy, src_xy, ww=0.32, wh=0.68):
        target_w, target_h = float(target_xy[0]), float(target_xy[1])
        src_w, src_h = float(src_xy[0]), float(src_xy[1])
        num = ww * src_w * target_w + wh * src_h * target_h
        den = ww * src_w * src_w + wh * src_h * src_h + 1e-12
        s = float(num / den)
        return max(s, 1e-8)

    def translation_from_visible_center(self, visible_pts, anchor_center, target_z):
        vis_center = self.robust_center(visible_pts)
        return np.array(
            [
                anchor_center[0] - vis_center[0],
                anchor_center[1] - vis_center[1],
                target_z - np.median(visible_pts[:, 2]),
            ],
            dtype=np.float64,
        )

    def score_alignment(self, anchor_data, anchor_pts, sam_vis_refined, full_final_xy, target_xy, fitness, rmse):
        rel_err_full = np.abs(full_final_xy - target_xy) / np.maximum(target_xy, 1e-8)
        mean_cd, med_cd = self.compute_symmetric_chamfer(anchor_pts, sam_vis_refined)
        proj = self.projection_metrics(sam_vis_refined, anchor_data)
        diag = float(np.linalg.norm([target_xy[0], target_xy[1], anchor_data["scale_targets"]["z_med"]]) + 1e-8)
        size_score = float(0.32 * rel_err_full[0] + 0.68 * rel_err_full[1])
        chamfer_n = mean_cd / diag if np.isfinite(mean_cd) else 1e6
        depth_n = proj["depth_mae"] / max(anchor_data["scale_targets"]["z_med"], 1e-6) if np.isfinite(proj["depth_mae"]) else 1e6
        score = (
            5.0 * size_score
            + 1.0 * chamfer_n
            + 0.9 * (1.0 - proj["iou"])
            + 0.4 * (1.0 - proj["coverage"])
            + 0.35 * depth_n
            + 0.2 * rmse / max(diag, 1e-8)
            - 0.15 * fitness
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

    def search_metric_alignment(self, anchor_data, sam_pcd):
        anchor_pts = self.pcd_to_np(self.voxel_downsample_pcd(anchor_data["pcd"], self.voxel_size))
        anchor_pts = self.remove_outliers_np(anchor_pts, nb_neighbors=20, std_ratio=1.2)
        if len(anchor_pts) == 0:
            raise RuntimeError("Anchor point cloud is empty after downsampling.")

        sam_pts = self.pcd_to_np(self.voxel_downsample_pcd(sam_pcd, self.voxel_size))
        sam_pts = self.remove_outliers_np(sam_pts, nb_neighbors=20, std_ratio=1.5)
        if len(sam_pts) == 0:
            raise RuntimeError("SAM3D point cloud is empty after downsampling.")

        anchor_center = anchor_data["anchor_center"]
        target_xy = np.array(
            [
                anchor_data["scale_targets"]["metric_w"],
                anchor_data["scale_targets"]["metric_h"],
            ],
            dtype=np.float64,
        )
        target_z = float(anchor_data["scale_targets"]["z_med"])

        anchor_pca_center, anchor_axes, _ = self.compute_pca_frame(anchor_pts)
        sam_center, sam_axes, _ = self.compute_pca_frame(sam_pts)
        _ = anchor_pca_center

        base_R = anchor_axes @ sam_axes.T
        candidate_Rs = self.generate_sign_flip_rotations(base_R)

        best = None
        for ridx, R in enumerate(candidate_Rs):
            sam_rot_full = self.centered_similarity(sam_pts, scale=1.0, R=R, center=sam_center)
            full_xy_raw, _, _ = self.robust_extent_xy(sam_rot_full, low=2.0, high=98.0)
            scale0 = self.solve_scale_xy(target_xy, full_xy_raw)

            for mult in [0.94, 0.98, 1.00, 1.02, 1.06]:
                s = scale0 * mult
                sam_scaled_full = self.centered_similarity(sam_pts, scale=s, R=R, center=sam_center)
                sam_scaled_vis = self.extract_visible_surface_from_camera(sam_scaled_full)
                if len(sam_scaled_vis) < 100:
                    continue

                t0 = self.translation_from_visible_center(sam_scaled_vis, anchor_center, target_z)
                sam_init_full = sam_scaled_full + t0[None, :]
                sam_init_vis = sam_scaled_vis + t0[None, :]

                threshold = max(0.008, 4.0 * self.voxel_size)
                sam_refined_vis, T_icp, fitness, rmse = self.icp_refine_rigid(
                    sam_init_vis, anchor_pts, threshold
                )
                sam_refined_full = self.apply_4x4_to_points(sam_init_full, T_icp)

                full_xy_after_icp, _, _ = self.robust_extent_xy(sam_refined_full, low=2.0, high=98.0)
                s_corr_xy = self.solve_scale_xy(target_xy, full_xy_after_icp)
                s_corr_xy = float(np.clip(s_corr_xy, 0.94, 1.06))

                sam_corr_full = self.centered_similarity(sam_refined_full, scale=s_corr_xy, center=anchor_center)
                sam_corr_vis = self.centered_similarity(sam_refined_vis, scale=s_corr_xy, center=anchor_center)
                t1 = self.translation_from_visible_center(sam_corr_vis, anchor_center, target_z)
                sam_corr_full = sam_corr_full + t1[None, :]
                sam_corr_vis = sam_corr_vis + t1[None, :]

                target_px_wh = np.array(
                    [
                        anchor_data["scale_targets"]["pixel_w"],
                        anchor_data["scale_targets"]["pixel_h"],
                    ],
                    dtype=np.float64,
                )
                proj_px_wh = self.projected_mask_bbox_wh(sam_corr_vis, anchor_data, dilate_ksize=5)
                s_corr_proj = self.solve_projected_bbox_scale(target_px_wh, proj_px_wh, ww=0.18, wh=0.82)
                s_corr_proj = float(np.clip(s_corr_proj, 0.97, 1.00))

                s_corr_pre_obj = float(np.sqrt(s_corr_xy * s_corr_proj))
                s_corr_pre_obj = float(np.clip(s_corr_pre_obj, 0.97, 1.02))

                sam_mid_full = self.centered_similarity(sam_refined_full, scale=s_corr_pre_obj, center=anchor_center)
                sam_mid_vis = self.centered_similarity(sam_refined_vis, scale=s_corr_pre_obj, center=anchor_center)
                t_mid = self.translation_from_visible_center(sam_mid_vis, anchor_center, target_z)
                sam_mid_full = sam_mid_full + t_mid[None, :]
                sam_mid_vis = sam_mid_vis + t_mid[None, :]

                obj_ext_mid, _, _, _, _ = self.robust_pca_extents(sam_mid_full, low=2.0, high=98.0)
                s_corr_obj = self.solve_scale_object_dims(target_xy, obj_ext_mid, ww=0.18, wh=0.82)
                s_corr_obj = float(np.clip(s_corr_obj, 0.95, 1.01))

                s_corr = float(s_corr_pre_obj * s_corr_obj)
                s_corr = float(np.clip(s_corr, 0.94, 1.02))

                sam_final_full = self.centered_similarity(sam_refined_full, scale=s_corr, center=anchor_center)
                sam_final_vis = self.centered_similarity(sam_refined_vis, scale=s_corr, center=anchor_center)
                t2 = self.translation_from_visible_center(sam_final_vis, anchor_center, target_z)
                sam_final_full = sam_final_full + t2[None, :]
                sam_final_vis = sam_final_vis + t2[None, :]

                full_xy_final, _, _ = self.robust_extent_xy(sam_final_full, low=2.0, high=98.0)
                vis_xy_final, _, _ = self.robust_extent_xy(sam_final_vis, low=2.0, high=98.0)
                proj_px_wh_final = self.projected_mask_bbox_wh(sam_final_vis, anchor_data, dilate_ksize=5)
                metrics = self.score_alignment(
                    anchor_data, anchor_pts, sam_final_vis, full_xy_final, target_xy, fitness, rmse
                )

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
                    "projected_px_wh": np.array(
                        proj_px_wh_final if proj_px_wh_final is not None else [-1.0, -1.0],
                        dtype=np.float64,
                    ),
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

    # ============================================================
    # Mesh reconstruction
    # ============================================================
    def estimate_normals_inplace(self, pcd, radius=None):
        pts = self.pcd_to_np(pcd)
        if len(pts) == 0:
            return pcd
        if radius is None:
            ext, _, _ = self.robust_extent(pts)
            radius = max(float(np.linalg.norm(ext) / 60.0), 0.003)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
        )
        try:
            pcd.orient_normals_consistent_tangent_plane(20)
        except Exception:
            pass
        return pcd

    def reconstruct_mesh_from_pcd(self, pcd):
        pcd = copy.deepcopy(pcd)
        pcd = self.voxel_downsample_pcd(pcd, self.voxel_size)
        pcd = self.estimate_normals_inplace(pcd)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.mesh_poisson_depth
        )
        densities = np.asarray(densities)
        keep = densities > np.quantile(densities, self.mesh_density_q)
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
            pts = self.pcd_to_np(pcd)
            ext, _, _ = self.robust_extent(pts)
            alpha = max(float(np.linalg.norm(ext) / 40.0), 0.002)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            mesh.compute_vertex_normals()

        return mesh



    def align_mesh_to_origin_and_axes(self, mesh):
        """
        将 mesh 做 PCA 主轴对齐：
        1) 让 mesh 的三个主方向与 xyz 轴平行
        2) 再把旋转后的包围盒中心平移到原点

        返回:
            mesh_aligned: 对齐后的 mesh
            axes: PCA 主轴（列向量）
            bbox_center: 对齐后用于平移归零的包围盒中心
        """
        mesh = copy.deepcopy(mesh)
        V = np.asarray(mesh.vertices).astype(np.float64)

        if len(V) == 0:
            return mesh, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

        # 1) PCA 主轴
        center, axes, _ = self.compute_pca_frame(V) # 主轴是物体最长-次长-最短的方向，且是右手坐标系

        # 2) 让结果尽量稳定：减少主轴正负号随机翻转
        if axes[0, 0] < 0:
            axes[:, 0] *= -1.0
        if axes[1, 1] < 0:
            axes[:, 1] *= -1.0
        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1.0

        # 3) 旋转到 PCA 坐标系（即让主轴对齐到 xyz）
        V_aligned = (V - center[None, :]) @ axes
        mesh.vertices = o3d.utility.Vector3dVector(V_aligned) # 物体的最长方向对齐到 x 轴，次长对齐到 y 轴，最短对齐到 z 轴

        # 4) 使用旋转后的 AABB 中心做平移，让 mesh 落到原点
        aabb = mesh.get_axis_aligned_bounding_box()
        bbox_center = aabb.get_center()

        V_aligned = np.asarray(mesh.vertices).astype(np.float64)
        V_aligned = V_aligned - bbox_center[None, :]
        mesh.vertices = o3d.utility.Vector3dVector(V_aligned)

        mesh.compute_vertex_normals()
        return mesh, axes, bbox_center

    




    # ============================================================
    # Logging helpers
    # ============================================================
    def print_anchor_diagnostics(self, anchor_data):
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

    @staticmethod
    def print_best_result(best):
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

    
    def _parse_k_input(self, k_input):
        """
        将上游传入的 1x9 / 9 元素相机内参，解析为 3x3 numpy 数组。
        支持 list / tuple / numpy.ndarray。
        """
        k_array = np.asarray(k_input, dtype=float)

        if k_array.size != 9:
            raise ValueError(
                f"k_input must contain exactly 9 elements, but got {k_array.size}. "
                f"Received shape: {k_array.shape}"
            )

        k_matrix = k_array.reshape(3, 3)
        return k_matrix

    # ============================================================
    # Main API
    # ============================================================
    def recon(
        self,
        rgb_path,
        depth_path,
        mask_path,
        k_input,
        out_dir = None,
        run_sam3d=True,
        output_prefix="sam3d_metric_general",
        raw_sam_name="sam3d_raw_v8.ply",
        anchor_name="anchor_metric_visible_surface_v8.ply",
        save_metrics_npy=True,
        save_metrics_json=True,
        seed=42,
    ):
        self._reset_timings()

        # 1. 先决定本次调用实际使用的输出目录
        current_out_dir = out_dir if out_dir is not None else self.default_out_dir

        if current_out_dir is None:
            raise ValueError(
                "out_dir is None. Please provide out_dir in recon() "
                "or set a default out_dir in __init__()."
            )

        # 2. 确保本次输出目录存在
        os.makedirs(current_out_dir, exist_ok=True)

        # 3. 后续所有输出路径都基于 current_out_dir
        anchor_path = os.path.join(current_out_dir, anchor_name)
        raw_sam_path = os.path.join(current_out_dir, raw_sam_name)

        out_metric_pcd = os.path.join(current_out_dir, f"{output_prefix}_full.ply")
        out_metric_vis = os.path.join(current_out_dir, f"{output_prefix}_visible.ply")
        out_metric_mesh_ply = os.path.join(current_out_dir, f"{output_prefix}_mesh.ply")
        out_metric_mesh_obj = os.path.join(current_out_dir, f"{output_prefix}_mesh.obj")
        out_metrics_npy = os.path.join(current_out_dir, "best_alignment_metrics.npy")
        out_metrics_json = os.path.join(current_out_dir, "best_alignment_metrics.json")
        out_timings_json = os.path.join(current_out_dir, "timings.json")

        with self._timer("step1/build_anchor_data"):
            print("[1/6] Build anchor data from RGB-D...")

            K = self._parse_k_input(k_input)
            anchor_data = self.build_anchor_data(rgb_path, depth_path, mask_path, K)

            self.save_pcd(anchor_path, anchor_data["pcd"])
            self.print_anchor_diagnostics(anchor_data)

        with self._timer("step2/run_or_load_sam3d"):
            print("\n[2/6] Run or load SAM3D...")
            if run_sam3d or (not os.path.exists(raw_sam_path)):
                self.run_sam3d(rgb_path, mask_path, raw_sam_path, seed=seed)
            print("SAM3D raw point cloud:", raw_sam_path)
            sam_pcd = self.load_pcd(raw_sam_path)

        with self._timer("step3/search_metric_alignment"):
            print("\n[3/6] Solve metric scale from FULL cloud width/height, then refine rigid pose...")
            best = self.search_metric_alignment(anchor_data, sam_pcd)
            self.print_best_result(best)

        with self._timer("step4/save_metric_pointclouds"):
            print("\n[4/6] Save metric point clouds...")
            metric_pcd = self.np_to_pcd(best["full_final"])
            metric_vis_pcd = self.np_to_pcd(best["visible_final"])
            self.save_pcd(out_metric_pcd, metric_pcd)
            self.save_pcd(out_metric_vis, metric_vis_pcd)
            print("Saved full metric point cloud   :", out_metric_pcd)
            print("Saved visible metric point cloud:", out_metric_vis)

        with self._timer("step5/reconstruct_metric_mesh"):
            print("\n[5/6] Reconstruct metric mesh...")
            mesh = self.reconstruct_mesh_from_pcd(metric_pcd)

            # V1.5版本新增功能，对应run_V1.5.py中新增的对齐后mesh输出。这个对齐是基于PCA主轴的，目的是让输出mesh在主轴方向上与xyz轴对齐，并且把包围盒中心平移到原点，方便后续使用。
            mesh, mesh_align_axes, mesh_align_bbox_center = self.align_mesh_to_origin_and_axes(mesh)

            mesh_iso_corr = 1.0
            self.save_mesh(out_metric_mesh_ply, mesh)
            self.save_mesh(out_metric_mesh_obj, mesh)
            print("Saved metric mesh (PLY):", out_metric_mesh_ply)
            print("Saved metric mesh (OBJ):", out_metric_mesh_obj)
            print("Mesh alignment axes:\n", mesh_align_axes)
            print("Mesh bbox center moved to origin from:", mesh_align_bbox_center)


        with self._timer("step6/save_metrics"):
            print("\n[6/6] Save metrics...")
            metrics_dict = {
                "scale": float(best["scale"]),
                "mesh_iso_correction": float(mesh_iso_corr),
                "mesh_align_axes": np.asarray(mesh_align_axes).tolist(),
                "mesh_align_bbox_center": np.asarray(mesh_align_bbox_center).tolist(),
                "scale_pre": float(best["scale_pre"]),
                "scale_correction": float(best["scale_correction"]),
                "scale_correction_xy": float(best["scale_correction_xy"]),
                "scale_correction_proj": float(best["scale_correction_proj"]),
                "scale_correction_obj": float(best["scale_correction_obj"]),
                "scale_correction_pre_obj": float(best["scale_correction_pre_obj"]),
                "score": float(best["score"]),
                "iou": float(best["iou"]),
                "depth_mae": float(best["depth_mae"]),
                "coverage": float(best["coverage"]),
                "mean_cd": float(best["mean_cd"]),
                "med_cd": float(best["med_cd"]),
                "target_wh": np.asarray(best["target_wh"]).tolist(),
                "aligned_full_wh": np.asarray(best["aligned_full_wh"]).tolist(),
                "aligned_vis_wh": np.asarray(best["aligned_vis_wh"]).tolist(),
                "size_rel_err_full_wh": np.asarray(best["size_rel_err_full_wh"]).tolist(),
                "anchor_scale_targets": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in anchor_data["scale_targets"].items()
                },
                "anchor_xy_extent": np.asarray(anchor_data["anchor_xy_extent"]).tolist(),
            }

            if save_metrics_npy:
                np.save(out_metrics_npy, metrics_dict, allow_pickle=True)

            if save_metrics_json:
                with open(out_metrics_json, "w", encoding="utf-8") as f:
                    json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

            print("Saved debug files to:", current_out_dir)

        with self._timer("step7/save_timings"):
            self._save_timings(out_timings_json)

        self._print_timing_summary()
        print("Done.")

        return {
            "out_dir": current_out_dir,
            "anchor_pcd_path": anchor_path,
            "raw_sam_pcd_path": raw_sam_path,
            "metric_full_pcd_path": out_metric_pcd,
            "metric_visible_pcd_path": out_metric_vis,
            "metric_mesh_ply_path": out_metric_mesh_ply,
            "metric_mesh_obj_path": out_metric_mesh_obj,
            "metrics_npy_path": out_metrics_npy if save_metrics_npy else None,
            "metrics_json_path": out_metrics_json if save_metrics_json else None,
            "timings_json_path": out_timings_json,
            "metrics": metrics_dict,
            "timings": self.total_timings.copy(),
        }