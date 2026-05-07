from __future__ import annotations

import json

import requests


SERVICE_URL = "http://127.0.0.1:58860/reconstruct"

payload = {
    "rgb_path": "/home/dct/work/sam-3d-objects/test_data/1776255100302.jpg",
    "depth_path": "/home/dct/work/sam-3d-objects/test_data/1776255100302_depth.png",
    "mask_path": "/home/dct/work/sam-3d-objects/test_data/1776255100302.png",
    "k_input": [
        384.897491455078, 0.0, 326.897308349609,
        0.0, 384.471374511719, 240.425384521484,
        0.0, 0.0, 1.0,
    ],
    "out_dir": "/home/dct/work/sam-3d-objects/Outputs/service_outputs/test_case_001",
    "run_sam3d": True,
    "output_prefix": "sam3d_metric_general",
    "raw_sam_name": "sam3d_raw_v8.ply",
    "anchor_name": "anchor_metric_visible_surface_v8.ply",
    "save_metrics_npy": True,
    "save_metrics_json": True,
    "seed": 42,
    "release_after_recon": False,
}

response = requests.post(SERVICE_URL, json=payload, timeout=None)
print("status_code:", response.status_code)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
