# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import sys

sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

CKPT_ROOT = "/home/user/datas/hc/data/ckpts/sam3d-obj/models/checkpoints"
config_path = os.path.join(CKPT_ROOT, "pipeline.yaml")

print("Using config:", config_path)
print("Exists:", os.path.exists(config_path))

inference = Inference(config_path, compile=False)

image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

output = inference(image, mask, seed=42)
output["gs"].save_ply("splat.ply")
print("Your reconstruction has been saved to splat.ply")
