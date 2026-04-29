import os
import sys
import time
import json
import copy
from contextlib import contextmanager, nullcontext

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import trimesh


class SAM3DReconstructor:
    """
    SAM3D native mesh metric reconstruction wrapper.

    Main idea:
        1. Build metric RGB-D anchor point cloud from depth + mask + camera intrinsics.
        2. Run SAM3D once to get:
              - output['gs']  -> raw SAM3D point cloud, used only for alignment.
              - output['glb'] -> native SAM3D mesh, used as final mesh source.
        3. Solve metric similarity / rigid alignment in point-cloud domain.
        4. Apply the solved transform to SAM3D native mesh vertices.
        5. Export metric mesh and optional PCA axis-aligned mesh.

    Expected usage:
        reconstructor = SAM3DReconstructor(
            sam3d_root="/home/dct/work/sam-3d-objects",
            sam3d_config="/home/user/datas/hc/data/ckpts/sam3d-obj/models/checkpoints/pipeline.yaml",
            out_dir=None,
            voxel_size=0.003,
            icp_max_iter=40,
            sam_compile=False,
            verbose=True,
        )

        result = reconstructor.recon(
            rgb_path="xxx.jpg",
            depth_path="xxx_depth.png",
            mask_path="xxx_mask.png",
            k_input=[fx, 0, cx, 0, fy, cy, 0, 0, 1],
            out_dir="/path/to/output",
            run_sam3d=True,
        )
    """

    def __init__(
        self,
        sam3d_root,
        sam3d_config,
        out_dir=None,
        voxel_size=0.003,
        icp_max_iter=40,
        mesh_poisson_depth=8,
        mesh_density_q=0.02,
        sam_compile=False,
        verbose=True,
    ):
        self.sam3d_root = os.path.abspath(sam3d_root)
        self.sam3d_config = sam3d_config
        self.default_out_dir = out_dir

        self.voxel_size = float(voxel_size)
        self.icp_max_iter = int(icp_max_iter)
        self.mesh_poisson_depth = int(mesh_poisson_depth)
        self.mesh_density_q = float(mesh_density_q)
        self.sam_compile = bool(sam_compile)
        self.verbose = bool(verbose)

        if self.default_out_dir is not None:
            os.makedirs(self.default_out_dir, exist_ok=True)

        self.init_timings = {}
        self.total_timings = {}

        self._prepare_sam3d_import_path()

        with self._timer("init/load_sam3d_model", target="init"):
            from inference import Inference, load_image  # noqa: E402

            self.Inference = Inference
            self.load_image = load_image
            self.inference = self.Inference(self.sam3d_config, compile=self.sam_compile)

    # ============================================================
    # Init / timer helpers
    # ============================================================
    def _prepare_sam3d_import_path(self):
        repo_root = os.path.abspath(self.sam3d_root)
        notebook_dir = os.path.join(repo_root, "notebook")

        for p in (repo_root, notebook_dir):
            try:
                sys.path.remove(p)
            except ValueError:
                pass

        sys.path.insert(0, repo_root)
        sys.path.insert(0, notebook_dir)

    @contextmanager
    def _timer(self, name, target="total"):
        start = time.perf_counter()
        if self.verbose:
            print(f"[Timer Start] {name}")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if target == "init":
                self.init_timings[name] = elapsed
            else:
                self.total_timings[name] = elapsed
            if self.verbose:
                print(f"[Timer End]   {name}: {elapsed:.4f}s")

    def _reset_timings(self):
        self.total_timings = {}

    def _save_timings(self, save_path):
        self.ensure_parent_dir(save_path)
        payload = {
            "init_timings": self.init_timings,
            "run_timings": self.total_timings,
            "run_total_excluding_init": float(sum(self.total_timings.values())),
            "total_including_init_loaded_once": float(
                sum(self.init_timings.values()) + sum(self.total_timings.values())
            ),
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _print_timing_summary(self):
        print("\n================ Timing Summary ================")
        if len(self.init_timings) > 0:
            print("[Init timings: model is loaded once in __init__]")
            for k, v in self.init_timings.items():
                print(f"{k:40s}: {v:.4f}s")
            print("------------------------------------------------")
        total = 0.0
        print("[Current recon timings]")
        for k, v in self.total_timings.items():
            print(f"{k:40s}: {v:.4f}s")
            total += v
        print("------------------------------------------------")
        print(f"{'RUN TOTAL':40s}: {total:.4f}s")
        print("================================================\n")

    # ============================================================
    # Basic I/O
    # ============================================================
    @staticmethod
    def ensure_parent_dir(file_path):
        parent_dir = os.path.dirname(os.path.abspath(file_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    @staticmethod
    def load_mask(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask = np.asarray(mask)
        return (mask > 0).astype(np.uint8)

    def _sam3d_cache_context(self):
        try:
            from sam3d_objects.model.backbone.tdfy_dit.modules.attention.ca_cache import (  # noqa: E501
                cross_attn_kv_cache,
            )

            pipeline = getattr(self.inference, "_pipeline", None)
            if pipeline is None:
                return nullcontext()
            return cross_attn_kv_cache(pipeline, verbose=self.verbose)
        except Exception:
            return nullcontext()

    def run_sam3d(self, rgb_path, mask_path, save_path, seed=42, return_mesh=False):
        """
        Run SAM3D and save raw GS point cloud as PLY.

        Args:
            rgb_path: RGB image path.
            mask_path: binary/object mask path.
            save_path: raw SAM3D point cloud PLY output path.
            seed: SAM3D inference seed.
            return_mesh: when True, returns native mesh from output['glb'].

        Returns:
            save_path, or (save_path, raw_mesh, output) when return_mesh=True.
        """
        self.ensure_parent_dir(save_path)

        image = self.load_image(rgb_path)
        mask = self.load_mask(mask_path)

        with self._sam3d_cache_context():
            output = self.inference(image, mask, seed=seed)

        if not hasattr(output, "__contains__"):
            raise TypeError(f"Unexpected SAM3D output type: {type(output)}")

        if "gs" not in output:
            raise KeyError("SAM3D inference did not return 'gs' in output.")

        output["gs"].save_ply(save_path)

        if not os.path.exists(save_path):
            raise RuntimeError(f"SAM3D gs.save_ply did not create file: {save_path}")

        if return_mesh:
            if "glb" not in output:
                raise KeyError("SAM3D inference did not return 'glb' in output, cannot return mesh.")
            return save_path, output["glb"], output

        return save_path

    @staticmethod
    def load_pcd(path):
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            raise ValueError(f"Empty point cloud: {path}")
        return pcd

    @staticmethod
    def save_pcd(path, pcd):
        SAM3DReconstructor.ensure_parent_dir(path)
        ok = o3d.io.write_point_cloud(path, pcd)
        if not ok:
            raise RuntimeError(f"Failed to save point cloud: {path}")

    @staticmethod
    def save_mesh(path, mesh):
        SAM3DReconstructor.ensure_parent_dir(path)
        ok = o3d.io.write_triangle_mesh(path, mesh, write_ascii=False)
        if not ok:
            raise RuntimeError(f"Failed to save mesh: {path}")

    @staticmethod
    def np_to_pcd(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        return pcd

    @staticmethod
    def pcd_to_np(pcd):
        return np.asarray(pcd.points).astype(np.float64)

    def voxel_downsample_pcd(self, pcd, voxel_size=None):
        if voxel_size is None:
            voxel_size = self.voxel_size
        return pcd.voxel_down_sample(voxel_size=float(voxel_size))

    # ============================================================
    # Geometry utils
    # ============================================================
    @staticmethod
    def robust_center(points):
        points = np.asarray(points, dtype=np.float64)
        if len(points) == 0:
            raise ValueError("Cannot compute center of empty points.")
        return np.median(points, axis=0)

    @staticmethod
    def robust_extent(points, low=2.0, high=98.0):
        points = np.asarray(points, dtype=np.float64)
        if len(points) == 0:
            raise ValueError("Cannot compute extent of empty points.")
        lo = np.percentile(points, low, axis=0)
        hi = np.percentile(points, high, axis=0)
        return hi - lo, lo, hi

    @staticmethod
    def robust_extent_xy(points, low=2.0, high=98.0):
        points = np.asarray(points, dtype=np.float64)
        if len(points) == 0:
            raise ValueError("Cannot compute XY extent of empty points.")
        lo = np.percentile(points[:, :2], low, axis=0)
        hi = np.percentile(points[:, :2], high, axis=0)
        return hi - lo, lo, hi

    def compute_pca_frame(self, points):
        points = np.asarray(points, dtype=np.float64)
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
        X = np.asarray(points, dtype=np.float64).copy()
        if center is not None:
            center = np.asarray(center, dtype=np.float64)
            X = X - center[None, :]
        if R is not None:
            X = X @ np.asarray(R, dtype=np.float64).T
        X = X * float(scale)
        if center is not None:
            X = X + center[None, :]
        if t is not None:
            t = np.asarray(t, dtype=np.float64)
            X = X + t[None, :]
        return X

    @staticmethod
    def apply_4x4_to_points(points, T):
        points = np.asarray(points, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64).reshape(4, 4)
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
        kernel = np.ones((int(ksize), int(ksize)), np.uint8)
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
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
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
        points = np.asarray(points)
        if len(points) < max(nb_neighbors + 5, 20):
            return points
        pcd = self.np_to_pcd(points)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio)
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
        k = int(round(min(bw, bh) * float(ratio)))
        k = max(int(min_k), min(k, int(max_k)))
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
        points = np.asarray(points, dtype=np.float64)
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
        num = float(wh) * src_h * tgt_h + float(ww) * src_w * tgt_w
        den = float(wh) * src_h * src_h + float(ww) * src_w * src_w + 1e-12
        return max(float(num / den), 1e-8)

    # ============================================================
    # Anchor build
    # ============================================================
    def _parse_k_input(self, k_input):
        k_array = np.asarray(k_input, dtype=float)
        if k_array.size != 9:
            raise ValueError(
                f"k_input must contain exactly 9 elements, but got {k_array.size}. "
                f"Received shape: {k_array.shape}"
            )
        K = k_array.reshape(3, 3)
        if not np.isfinite(K).all():
            raise ValueError("K contains NaN or Inf.")
        if abs(K[2, 2]) < 1e-12:
            raise ValueError("K[2, 2] is zero; invalid camera intrinsic matrix.")
        return K

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

        K = self._parse_k_input(K)
        depth_m, depth_mode = self.infer_depth_meters(depth)

        mask_bin = self.morph_clean((mask > 0).astype(np.uint8), ksize=5)
        scale_targets = self.estimate_anchor_scale_targets(mask_bin, depth_m, K)

        z_med = scale_targets["z_med"]
        band = max(0.035 * z_med, 0.012)
        valid_depth_mask = (
            np.isfinite(depth_m)
            & (depth_m > 1e-6)
            & (np.abs(depth_m - z_med) <= band)
        )
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
        points = np.asarray(points, dtype=np.float64)
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

    def compute_symmetric_chamfer(self, src_pts, tgt_pts, src_pcd=None, tgt_pcd=None):
        if src_pcd is None:
            src_pcd = self.np_to_pcd(src_pts)
        if tgt_pcd is None:
            tgt_pcd = self.np_to_pcd(tgt_pts)
        d1 = np.asarray(src_pcd.compute_point_cloud_distance(tgt_pcd), dtype=np.float64)
        d2 = np.asarray(tgt_pcd.compute_point_cloud_distance(src_pcd), dtype=np.float64)
        if len(d1) == 0 or len(d2) == 0:
            return np.inf, np.inf
        return float(d1.mean() + d2.mean()), float(np.median(d1) + np.median(d2))

    @staticmethod
    def project_points(points, K, H, W):
        points = np.asarray(points, dtype=np.float64)
        if len(points) == 0:
            return np.empty((0,), np.int32), np.empty((0,), np.int32), np.empty((0,), np.float32)

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

    def projection_metrics(self, points, anchor_data, depth_pred=None, mask_pred=None):
        H = anchor_data["H"]
        W = anchor_data["W"]
        K = anchor_data["K"]
        depth_gt = anchor_data["depth_m"]
        mask_gt = anchor_data["mask_bin"] > 0

        if depth_pred is None or mask_pred is None:
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

    @staticmethod
    def _bbox_wh_from_mask(mask_pred, dilate_ksize=5):
        if mask_pred is None or mask_pred.sum() == 0:
            return None
        if dilate_ksize > 1:
            kernel = np.ones((int(dilate_ksize), int(dilate_ksize)), np.uint8)
            mask_pred = cv2.dilate(mask_pred.astype(np.uint8), kernel, iterations=1)
        ys, xs = np.where(mask_pred > 0)
        if len(xs) == 0:
            return None
        px_w = float(xs.max() - xs.min() + 1)
        px_h = float(ys.max() - ys.min() + 1)
        return np.array([px_w, px_h], dtype=np.float64)

    def projected_mask_bbox_wh(self, points, anchor_data, dilate_ksize=5):
        H = anchor_data["H"]
        W = anchor_data["W"]
        K = anchor_data["K"]
        _, mask_pred = self.render_depth_and_mask(points, K, H, W)
        return self._bbox_wh_from_mask(mask_pred, dilate_ksize=dilate_ksize)

    def render_and_bbox(self, points, anchor_data, dilate_ksize=5):
        K = anchor_data["K"]
        H = anchor_data["H"]
        W = anchor_data["W"]
        depth_pred, mask_pred = self.render_depth_and_mask(points, K, H, W)
        bbox_wh = self._bbox_wh_from_mask(mask_pred, dilate_ksize=dilate_ksize)
        return depth_pred, mask_pred, bbox_wh

    @staticmethod
    def solve_projected_bbox_scale(target_px_wh, src_px_wh, ww=0.25, wh=0.75):
        if src_px_wh is None:
            return 1.0
        sw, sh = float(src_px_wh[0]), float(src_px_wh[1])
        tw, th = float(target_px_wh[0]), float(target_px_wh[1])
        if sw <= 1e-8 or sh <= 1e-8:
            return 1.0
        num = float(ww) * sw * tw + float(wh) * sh * th
        den = float(ww) * sw * sw + float(wh) * sh * sh + 1e-12
        return max(float(num / den), 1e-8)

    def icp_refine_rigid(self, source_pts, target_pts, threshold, target_pcd=None, max_iter=None):
        src = self.np_to_pcd(source_pts)
        tgt = target_pcd if target_pcd is not None else self.np_to_pcd(target_pts)
        if max_iter is None:
            max_iter = self.icp_max_iter
        reg = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            float(threshold),
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter)),
        )
        refined = self.apply_4x4_to_points(source_pts, reg.transformation)
        return refined, reg.transformation, float(reg.fitness), float(reg.inlier_rmse)

    @staticmethod
    def solve_scale_xy(target_xy, src_xy, ww=0.32, wh=0.68):
        target_w, target_h = float(target_xy[0]), float(target_xy[1])
        src_w, src_h = float(src_xy[0]), float(src_xy[1])
        num = float(ww) * src_w * target_w + float(wh) * src_h * target_h
        den = float(ww) * src_w * src_w + float(wh) * src_h * src_h + 1e-12
        s = float(num / den)
        return max(s, 1e-8)

    def translation_from_visible_center(self, visible_pts, anchor_center, target_z):
        vis_center = self.robust_center(visible_pts)
        return np.array(
            [
                anchor_center[0] - vis_center[0],
                anchor_center[1] - vis_center[1],
                float(target_z) - np.median(visible_pts[:, 2]),
            ],
            dtype=np.float64,
        )

    def score_alignment(
        self,
        anchor_data,
        anchor_pts,
        sam_vis_refined,
        full_final_xy,
        target_xy,
        fitness,
        rmse,
        anchor_pcd=None,
        depth_pred=None,
        mask_pred=None,
    ):
        rel_err_full = np.abs(full_final_xy - target_xy) / np.maximum(target_xy, 1e-8)
        mean_cd, med_cd = self.compute_symmetric_chamfer(
            anchor_pts,
            sam_vis_refined,
            src_pcd=anchor_pcd,
        )
        proj = self.projection_metrics(
            sam_vis_refined,
            anchor_data,
            depth_pred=depth_pred,
            mask_pred=mask_pred,
        )
        diag = float(
            np.linalg.norm([target_xy[0], target_xy[1], anchor_data["scale_targets"]["z_med"]])
            + 1e-8
        )
        size_score = float(0.32 * rel_err_full[0] + 0.68 * rel_err_full[1])
        chamfer_n = mean_cd / diag if np.isfinite(mean_cd) else 1e6
        depth_n = (
            proj["depth_mae"] / max(anchor_data["scale_targets"]["z_med"], 1e-6)
            if np.isfinite(proj["depth_mae"])
            else 1e6
        )
        score = (
            5.0 * size_score
            + 1.0 * chamfer_n
            + 0.9 * (1.0 - proj["iou"])
            + 0.4 * (1.0 - proj["coverage"])
            + 0.35 * depth_n
            + 0.2 * float(rmse) / max(diag, 1e-8)
            - 0.15 * float(fitness)
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

    def _finalize_alignment_candidate(
        self,
        anchor_data,
        anchor_pts,
        anchor_pcd,
        sam_refined_full,
        sam_refined_vis,
        target_xy,
        target_z,
        anchor_center,
        target_px_wh,
        fitness,
        rmse,
    ):
        full_xy_after_icp, _, _ = self.robust_extent_xy(sam_refined_full, low=2.0, high=98.0)
        s_corr_xy = self.solve_scale_xy(target_xy, full_xy_after_icp)
        s_corr_xy = float(np.clip(s_corr_xy, 0.94, 1.06))

        sam_corr_full = self.centered_similarity(sam_refined_full, scale=s_corr_xy, center=anchor_center)
        sam_corr_vis = self.centered_similarity(sam_refined_vis, scale=s_corr_xy, center=anchor_center)
        t1 = self.translation_from_visible_center(sam_corr_vis, anchor_center, target_z)
        sam_corr_full = sam_corr_full + t1[None, :]
        sam_corr_vis = sam_corr_vis + t1[None, :]

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
        depth_final, mask_final, proj_px_wh_final = self.render_and_bbox(
            sam_final_vis,
            anchor_data,
            dilate_ksize=5,
        )
        metrics = self.score_alignment(
            anchor_data,
            anchor_pts,
            sam_final_vis,
            full_xy_final,
            target_xy,
            fitness,
            rmse,
            anchor_pcd=anchor_pcd,
            depth_pred=depth_final,
            mask_pred=mask_final,
        )

        return {
            "score": metrics["score"],
            "scale_correction_xy": float(s_corr_xy),
            "scale_correction_proj": float(s_corr_proj),
            "scale_correction_obj": float(s_corr_obj),
            "scale_correction_pre_obj": float(s_corr_pre_obj),
            "scale_correction": float(s_corr),
            "t2": t2.copy(),
            "fitness": float(fitness),
            "rmse": float(rmse),
            "mean_cd": metrics["mean_cd"],
            "med_cd": metrics["med_cd"],
            "iou": metrics["iou"],
            "depth_mae": metrics["depth_mae"],
            "coverage": metrics["coverage"],
            "projected_px_wh": np.array(
                proj_px_wh_final if proj_px_wh_final is not None else [-1.0, -1.0],
                dtype=np.float64,
            ),
            "aligned_full_wh": full_xy_final,
            "aligned_vis_wh": vis_xy_final,
            "size_rel_err_full_wh": metrics["size_rel_err_full"],
            "raw_full_wh_after_icp_pre_corr": np.array(full_xy_after_icp, dtype=np.float64),
            "full_final": sam_final_full,
            "visible_final": sam_final_vis,
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

        anchor_pcd = self.np_to_pcd(anchor_pts)
        icp_search_iter = min(20, self.icp_max_iter)

        anchor_center = anchor_data["anchor_center"]
        target_xy = np.array(
            [
                anchor_data["scale_targets"]["metric_w"],
                anchor_data["scale_targets"]["metric_h"],
            ],
            dtype=np.float64,
        )
        target_z = float(anchor_data["scale_targets"]["z_med"])
        target_px_wh = np.array(
            [
                anchor_data["scale_targets"]["pixel_w"],
                anchor_data["scale_targets"]["pixel_h"],
            ],
            dtype=np.float64,
        )

        _, anchor_axes, _ = self.compute_pca_frame(anchor_pts)
        sam_center, sam_axes, _ = self.compute_pca_frame(sam_pts)

        base_R = anchor_axes @ sam_axes.T
        candidate_Rs = self.generate_sign_flip_rotations(base_R)

        best = None
        for ridx, R in enumerate(candidate_Rs):
            sam_rot_full = self.centered_similarity(sam_pts, scale=1.0, R=R, center=sam_center)
            full_xy_raw, _, _ = self.robust_extent_xy(sam_rot_full, low=2.0, high=98.0)
            scale0 = self.solve_scale_xy(target_xy, full_xy_raw)

            for mult in [0.94, 0.98, 1.00, 1.02, 1.06]:
                s = float(scale0 * mult)
                sam_scaled_full = self.centered_similarity(sam_pts, scale=s, R=R, center=sam_center)
                sam_scaled_vis = self.extract_visible_surface_from_camera(sam_scaled_full)
                if len(sam_scaled_vis) < 100:
                    continue

                t0 = self.translation_from_visible_center(sam_scaled_vis, anchor_center, target_z)
                sam_init_full = sam_scaled_full + t0[None, :]
                sam_init_vis = sam_scaled_vis + t0[None, :]

                threshold = max(0.008, 4.0 * self.voxel_size)
                sam_refined_vis, T_icp, fitness, rmse = self.icp_refine_rigid(
                    sam_init_vis,
                    anchor_pts,
                    threshold,
                    target_pcd=anchor_pcd,
                    max_iter=icp_search_iter,
                )
                sam_refined_full = self.apply_4x4_to_points(sam_init_full, T_icp)

                finalized = self._finalize_alignment_candidate(
                    anchor_data=anchor_data,
                    anchor_pts=anchor_pts,
                    anchor_pcd=anchor_pcd,
                    sam_refined_full=sam_refined_full,
                    sam_refined_vis=sam_refined_vis,
                    target_xy=target_xy,
                    target_z=target_z,
                    anchor_center=anchor_center,
                    target_px_wh=target_px_wh,
                    fitness=fitness,
                    rmse=rmse,
                )

                result = {
                    **finalized,
                    "R_idx": int(ridx),
                    "R": R.copy(),
                    "sam_center": sam_center.copy(),
                    "anchor_center": anchor_center.copy(),
                    "scale_pre": float(s),
                    "t0": t0.copy(),
                    "T_icp": T_icp.copy(),
                    "scale": float(s * finalized["scale_correction"]),
                    "target_wh": target_xy.copy(),
                    "target_px_wh": target_px_wh.copy(),
                    "raw_full_wh_before_scale": np.array(full_xy_raw, dtype=np.float64),
                    "_sam_init_vis": sam_init_vis,
                    "_sam_init_full": sam_init_full,
                    "_threshold": threshold,
                    "_scale_pre": float(s),
                }

                if best is None or result["score"] < best["score"]:
                    best = result

        if best is None:
            raise RuntimeError("Failed to find a valid alignment.")

        if self.icp_max_iter > icp_search_iter:
            t0_best = best["t0"].copy()
            sam_refined_vis_full, T_icp_full, fitness_full, rmse_full = self.icp_refine_rigid(
                best["_sam_init_vis"],
                anchor_pts,
                best["_threshold"],
                target_pcd=anchor_pcd,
                max_iter=self.icp_max_iter,
            )
            sam_refined_full_full = self.apply_4x4_to_points(best["_sam_init_full"], T_icp_full)

            finalized = self._finalize_alignment_candidate(
                anchor_data=anchor_data,
                anchor_pts=anchor_pts,
                anchor_pcd=anchor_pcd,
                sam_refined_full=sam_refined_full_full,
                sam_refined_vis=sam_refined_vis_full,
                target_xy=target_xy,
                target_z=target_z,
                anchor_center=anchor_center,
                target_px_wh=target_px_wh,
                fitness=fitness_full,
                rmse=rmse_full,
            )

            best = {
                **finalized,
                "R_idx": int(best["R_idx"]),
                "R": best["R"].copy(),
                "sam_center": sam_center.copy(),
                "anchor_center": anchor_center.copy(),
                "scale_pre": float(best["_scale_pre"]),
                "t0": t0_best,
                "T_icp": T_icp_full.copy(),
                "scale": float(best["_scale_pre"] * finalized["scale_correction"]),
                "target_wh": target_xy.copy(),
                "target_px_wh": target_px_wh.copy(),
                "raw_full_wh_before_scale": best["raw_full_wh_before_scale"],
            }

        for k in ("_sam_init_vis", "_sam_init_full", "_threshold", "_scale_pre"):
            best.pop(k, None)
        return best

    # ============================================================
    # Mesh helpers
    # ============================================================
    def apply_alignment_to_vertices(self, vertices, align):
        V = np.asarray(vertices, dtype=np.float64).copy()

        V = self.centered_similarity(
            V,
            scale=align["scale_pre"],
            R=align["R"],
            center=align["sam_center"],
        )
        V = V + align["t0"][None, :]
        V = self.apply_4x4_to_points(V, align["T_icp"])
        V = self.centered_similarity(
            V,
            scale=align["scale_correction"],
            center=align["anchor_center"],
        )
        V = V + align["t2"][None, :]
        return V

    def _as_single_trimesh(self, mesh_or_scene):
        """
        Convert trimesh.Trimesh or trimesh.Scene into one Trimesh.

        For Scene, to_mesh() is preferred because it bakes scene graph transforms.
        """
        if isinstance(mesh_or_scene, trimesh.Trimesh):
            mesh = mesh_or_scene.copy()
            if len(mesh.vertices) == 0:
                raise ValueError("Input Trimesh has no vertices.")
            return mesh

        if isinstance(mesh_or_scene, trimesh.Scene):
            if hasattr(mesh_or_scene, "to_mesh"):
                merged = mesh_or_scene.to_mesh()
                if isinstance(merged, trimesh.Trimesh) and len(merged.vertices) > 0:
                    return merged

            dumped = mesh_or_scene.dump(concatenate=True)
            if isinstance(dumped, trimesh.Trimesh) and len(dumped.vertices) > 0:
                return dumped

            dumped = mesh_or_scene.dump()
            meshes = [
                g for g in dumped
                if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0
            ]
            if len(meshes) == 0:
                raise ValueError("No valid Trimesh geometry found in trimesh.Scene.")
            return trimesh.util.concatenate(meshes)

        raise TypeError(f"Unsupported mesh type: {type(mesh_or_scene)}")

    def apply_alignment_to_trimesh(self, raw_mesh, align):
        mesh = self._as_single_trimesh(raw_mesh)
        mesh.vertices = self.apply_alignment_to_vertices(mesh.vertices, align)
        try:
            mesh._cache.clear()
        except Exception:
            pass
        return mesh

    def export_trimesh(self, mesh, out_path):
        self.ensure_parent_dir(out_path)
        ext = os.path.splitext(out_path)[1].lower().lstrip(".")
        if ext == "":
            raise ValueError(f"Output path has no extension: {out_path}")

        mesh_to_export = self._as_single_trimesh(mesh)
        if len(mesh_to_export.vertices) == 0:
            raise ValueError(f"Cannot export empty mesh to {out_path}")

        mesh_to_export.export(out_path, file_type=ext)
        if not os.path.exists(out_path):
            raise RuntimeError(f"Failed to export mesh: {out_path}")

    def pca_align_trimesh_to_origin(self, raw_mesh, longest_axis="y"):
        mesh = self._as_single_trimesh(raw_mesh)
        V = np.asarray(mesh.vertices, dtype=np.float64)
        if len(V) == 0:
            return mesh, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

        center, axes, _ = self.compute_pca_frame(V)

        if longest_axis == "y":
            axes_reordered = np.column_stack([
                axes[:, 1],
                axes[:, 0],
                axes[:, 2],
            ])
        elif longest_axis == "x":
            axes_reordered = np.column_stack([
                axes[:, 0],
                axes[:, 1],
                axes[:, 2],
            ])
        else:
            raise ValueError(f'Unsupported longest_axis="{longest_axis}", expected "x" or "y".')

        if axes_reordered[0, 0] < 0:
            axes_reordered[:, 0] *= -1.0
        if axes_reordered[1, 1] < 0:
            axes_reordered[:, 1] *= -1.0
        if np.linalg.det(axes_reordered) < 0:
            axes_reordered[:, 2] *= -1.0

        V_aligned = (V - center[None, :]) @ axes_reordered
        bbox_min = V_aligned.min(axis=0)
        bbox_max = V_aligned.max(axis=0)
        bbox_center = 0.5 * (bbox_min + bbox_max)
        V_aligned = V_aligned - bbox_center[None, :]

        mesh.vertices = V_aligned
        try:
            mesh._cache.clear()
        except Exception:
            pass
        return mesh, axes_reordered, bbox_center

    # Legacy fallback mesh reconstruction, kept for compatibility only.
    def estimate_normals_inplace(self, pcd, radius=None):
        pts = self.pcd_to_np(pcd)
        if len(pts) == 0:
            return pcd
        if radius is None:
            ext, _, _ = self.robust_extent(pts)
            radius = max(float(np.linalg.norm(ext) / 60.0), 0.003)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40))
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
            pcd,
            depth=self.mesh_poisson_depth,
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

    # ============================================================
    # Main API
    # ============================================================
    def recon(
        self,
        rgb_path,
        depth_path,
        mask_path,
        k_input,
        out_dir=None,
        run_sam3d=True,
        output_prefix="sam3d_metric_general",
        raw_sam_name="sam3d_raw_v8.ply",
        anchor_name="anchor_metric_visible_surface_v8.ply",
        save_metrics_npy=True,
        save_metrics_json=True,
        seed=42,
    ):
        self._reset_timings()

        current_out_dir = out_dir if out_dir is not None else self.default_out_dir
        if current_out_dir is None:
            raise ValueError(
                "out_dir is None. Please provide out_dir in recon() "
                "or set a default out_dir in __init__()."
            )
        os.makedirs(current_out_dir, exist_ok=True)

        anchor_path = os.path.join(current_out_dir, anchor_name)
        raw_sam_path = os.path.join(current_out_dir, raw_sam_name)

        out_metric_pcd = os.path.join(current_out_dir, f"{output_prefix}_full.ply")
        out_metric_vis = os.path.join(current_out_dir, f"{output_prefix}_visible.ply")
        out_metric_mesh_ply = os.path.join(current_out_dir, f"{output_prefix}_mesh.ply")
        out_metric_mesh_obj = os.path.join(current_out_dir, f"{output_prefix}_mesh.obj")
        out_metric_mesh_axis_ply = os.path.join(current_out_dir, f"{output_prefix}_mesh_axis_aligned.ply")
        out_metric_mesh_axis_obj = os.path.join(current_out_dir, f"{output_prefix}_mesh_axis_aligned.obj")
        out_metrics_npy = os.path.join(current_out_dir, "best_alignment_metrics.npy")
        out_metrics_json = os.path.join(current_out_dir, "best_alignment_metrics.json")
        out_timings_json = os.path.join(current_out_dir, "timings.json")

        with self._timer("step1/build_anchor_data"):
            print("[1/7] Build anchor data from RGB-D...")
            K = self._parse_k_input(k_input)
            anchor_data = self.build_anchor_data(rgb_path, depth_path, mask_path, K)
            self.save_pcd(anchor_path, anchor_data["pcd"])
            self.print_anchor_diagnostics(anchor_data)

        with self._timer("step2/run_sam3d"):
            print("\n[2/7] Run SAM3D...")
            if not run_sam3d:
                raise RuntimeError(
                    "This native-mesh pipeline requires run_sam3d=True, because raw_mesh comes "
                    "from output['glb']. To support run_sam3d=False, add a cached native mesh path."
                )

            raw_sam_path, raw_mesh, sam3d_output = self.run_sam3d(
                rgb_path,
                mask_path,
                raw_sam_path,
                seed=seed,
                return_mesh=True,
            )
            _ = sam3d_output
            print("SAM3D raw point cloud:", raw_sam_path)
            print("SAM3D raw mesh type   :", type(raw_mesh))
            sam_pcd = self.load_pcd(raw_sam_path)

        with self._timer("step3/search_metric_alignment"):
            print("\n[3/7] Solve metric scale and pose on point cloud...")
            best = self.search_metric_alignment(anchor_data, sam_pcd)
            self.print_best_result(best)

        with self._timer("step4/save_metric_pointclouds"):
            print("\n[4/7] Save metric point clouds...")
            metric_pcd = self.np_to_pcd(best["full_final"])
            metric_vis_pcd = self.np_to_pcd(best["visible_final"])
            self.save_pcd(out_metric_pcd, metric_pcd)
            self.save_pcd(out_metric_vis, metric_vis_pcd)
            print("Saved full metric point cloud   :", out_metric_pcd)
            print("Saved visible metric point cloud:", out_metric_vis)

        with self._timer("step5/export_native_metric_mesh"):
            print("\n[5/7] Apply solved transform to SAM3D native mesh...")
            metric_mesh = self.apply_alignment_to_trimesh(raw_mesh, best)
            mesh_iso_corr = 1.0

            self.export_trimesh(metric_mesh, out_metric_mesh_ply)
            self.export_trimesh(metric_mesh, out_metric_mesh_obj)
            print("Saved native metric mesh (PLY):", out_metric_mesh_ply)
            print("Saved native metric mesh (OBJ):", out_metric_mesh_obj)

            axis_mesh, mesh_align_axes, mesh_align_bbox_center = self.pca_align_trimesh_to_origin(
                metric_mesh,
                longest_axis="y",
            )
            self.export_trimesh(axis_mesh, out_metric_mesh_axis_ply)
            self.export_trimesh(axis_mesh, out_metric_mesh_axis_obj)
            print("Saved axis-aligned metric mesh (PLY):", out_metric_mesh_axis_ply)
            print("Saved axis-aligned metric mesh (OBJ):", out_metric_mesh_axis_obj)
            print("Mesh alignment axes:\n", mesh_align_axes)
            print("Mesh bbox center moved to origin from:", mesh_align_bbox_center)

        with self._timer("step6/save_metrics"):
            print("\n[6/7] Save metrics...")
            metrics_dict = {
                "mesh_source": "sam3d_native_glb",
                "mesh_pipeline": "apply_pointcloud_alignment_to_native_mesh_no_poisson",
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
                "target_px_wh": np.asarray(best["target_px_wh"]).tolist(),
                "projected_px_wh": np.asarray(best["projected_px_wh"]).tolist(),
                "size_rel_err_full_wh": np.asarray(best["size_rel_err_full_wh"]).tolist(),
                "raw_full_wh_before_scale": np.asarray(best["raw_full_wh_before_scale"]).tolist(),
                "raw_full_wh_after_icp_pre_corr": np.asarray(best["raw_full_wh_after_icp_pre_corr"]).tolist(),
                "anchor_scale_targets": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in anchor_data["scale_targets"].items()
                    if k != "inner_mask"
                },
                "anchor_xy_extent": np.asarray(anchor_data["anchor_xy_extent"]).tolist(),
                "depth_mode": anchor_data["depth_mode"],
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
            "metric_mesh_axis_aligned_ply_path": out_metric_mesh_axis_ply,
            "metric_mesh_axis_aligned_obj_path": out_metric_mesh_axis_obj,
            "metrics_npy_path": out_metrics_npy if save_metrics_npy else None,
            "metrics_json_path": out_metrics_json if save_metrics_json else None,
            "timings_json_path": out_timings_json,
            "metrics": metrics_dict,
            "timings": self.total_timings.copy(),
            "init_timings": self.init_timings.copy(),
        }
