# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch


def _resolve_depth_dtype(dtype):
    """None = keep FP32; 支持 str / torch.dtype。"""
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    s = str(dtype).strip().lower()
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32", "none", "null"):
        return None
    return torch.float16


class DepthModel:
    def __init__(self, model, device="cuda", dtype=torch.float16):
        self.model = model
        self.device = torch.device(device)
        tdtype = _resolve_depth_dtype(dtype)
        # 先在 CPU 上转 dtype，再 .to(device)，降低峰值并避免 FP32 整模上显存翻倍
        if tdtype is not None:
            self.model = self.model.to(dtype=tdtype)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.to(self.device)
        self.model.eval()

    def promote_to_pipeline_device(self, pipeline_device: torch.device) -> None:
        """在 SAM3D 主模型加载完成后再把深度模型迁到 GPU（初始化时可用 device=cpu）。"""
        if self.device == pipeline_device:
            return
        if pipeline_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.to(pipeline_device)
        self.device = pipeline_device

    def __call__(self, image):
        pass
