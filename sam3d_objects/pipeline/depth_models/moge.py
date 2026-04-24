# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch

from .base import DepthModel


class MoGe(DepthModel):
    def __call__(self, image):
        x = image.to(self.device, non_blocking=self.device.type == "cuda")
        param_dtype = next(self.model.parameters()).dtype
        if param_dtype in (torch.float16, torch.bfloat16):
            x = x.to(dtype=param_dtype)
        else:
            x = x.float()
        output = self.model.infer(
            x, force_projection=False, use_fp16=True
        )
        pointmaps = output["points"]
        output["pointmaps"] = pointmaps
        return output
