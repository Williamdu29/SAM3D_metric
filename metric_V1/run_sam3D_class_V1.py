import os

from sam3D_class_V1 import SAM3DReconstructor

# ============================================================
# Input paths
# ============================================================
RGB_PATH = "/mnt/ws_shard/dct/SKU/ice_tea/image/1775209506830.jpg"
DEPTH_PATH = "/mnt/ws_shard/dct/SKU/ice_tea/depth/1775209506830.png"
MASK_PATH = "/mnt/ws_shard/dct/SKU/ice_tea/masks/maskdata/20260407_112555.png"
K_INPUT = [
    384.897491455078, 0.0, 326.897308349609,
    0.0, 384.471374511719, 240.425384521484,
    0.0, 0.0, 1.0,
]

# ============================================================
# Config
# ============================================================
SAM3D_ROOT = "/home/dct/work/sam-3d-objects"
SAM3D_CONFIG = "/home/user/datas/hc/data/ckpts/sam3d-obj/models/checkpoints/pipeline.yaml"
# OUT_DIR = os.path.join(SAM3D_ROOT, "cookie_output")

# ============================================================
# Build reconstructor
# ============================================================
reconstructor = SAM3DReconstructor(
    sam3d_root=SAM3D_ROOT,
    sam3d_config=SAM3D_CONFIG,
    out_dir=None,
    voxel_size=0.0025,
    icp_max_iter=40,
    mesh_poisson_depth=8,
    mesh_density_q=0.02,
    sam_compile=False,   # 如需 compile=True 可自行打开
    verbose=True,
)

# ============================================================
# Run reconstruction
# ============================================================

this_out_dir = os.path.join(SAM3D_ROOT, "Outputs", "ice_tea_output") # 输出目录示例，这里改成了 pejoy_output/case_001，避免和之前的 cookie_output 混淆，实际使用时可以改成任意路径

result = reconstructor.recon(
    rgb_path=RGB_PATH,
    depth_path=DEPTH_PATH,
    mask_path=MASK_PATH,
    k_input=K_INPUT,
    out_dir=this_out_dir,
    run_sam3d=True,                     # 如果已有 raw sam ply，也可以传 False
    output_prefix="sam3d_metric_general",
    raw_sam_name="sam3d_raw_v8.ply",
    anchor_name="anchor_metric_visible_surface_v8.ply",
    save_metrics_npy=True,
    save_metrics_json=True,
    seed=42,
)

# ============================================================
# Print outputs
# ============================================================
print("\n========== Recon Output ==========")
print("anchor_pcd_path         :", result["anchor_pcd_path"])
print("raw_sam_pcd_path        :", result["raw_sam_pcd_path"])
print("metric_full_pcd_path    :", result["metric_full_pcd_path"])
print("metric_visible_pcd_path :", result["metric_visible_pcd_path"])
print("metric_mesh_ply_path    :", result["metric_mesh_ply_path"])
print("metric_mesh_obj_path    :", result["metric_mesh_obj_path"])
print("metrics_npy_path        :", result["metrics_npy_path"])
print("metrics_json_path       :", result["metrics_json_path"])
print("timings_json_path       :", result["timings_json_path"])

print("\n========== Key Metrics ==========")
for k, v in result["metrics"].items():
    if k in ["anchor_scale_targets"]:
        continue
    print(f"{k}: {v}")

print("\n========== Timings ==========")
for k, v in result["timings"].items():
    print(f"{k}: {v:.4f}s")