# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Attention 后端环境配置（独立于 inference_pipeline，避免 import 顺序问题）。

必须在「首次 import sam3d_objects.model...attention」之前调用一次，例如
notebook/inference.py 里在 ``import sam3d_objects`` 之前::

    from sam3d_objects.pipeline.attention_env import configure_attention_backend
    configure_attention_backend()

``inference_pipeline`` 模块加载时也会再调用一次（幂等）。
"""
from __future__ import annotations

import os

import torch
from loguru import logger


def configure_attention_backend() -> None:
    """根据环境变量 / GPU / flash_attn 是否可装，设置 ATTN_BACKEND 与 SPARSE_ATTN_BACKEND。"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    logger.info(f"GPU name is {gpu_name}")

    user_attn = os.environ.get("ATTN_BACKEND")
    user_sparse_attn = os.environ.get("SPARSE_ATTN_BACKEND")
    if user_attn:
        logger.info(f"[ATTN] Respecting pre-set ATTN_BACKEND={user_attn}")
        if not user_sparse_attn:
            os.environ["SPARSE_ATTN_BACKEND"] = user_attn
        return

    try:
        import flash_attn  # noqa: F401

        os.environ["ATTN_BACKEND"] = "flash_attn"
        os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"
        logger.info("[ATTN] flash_attn detected, using flash_attn backend")
        return
    except Exception as e:
        logger.info(f"[ATTN] flash_attn not available ({type(e).__name__}: {e}), falling back")

    if "A100" in gpu_name or "H100" in gpu_name or "H200" in gpu_name:
        os.environ["ATTN_BACKEND"] = "flash_attn"
        os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"
        logger.info(f"[ATTN] GPU {gpu_name} matches whitelist, using flash_attn")
    else:
        os.environ["ATTN_BACKEND"] = "sdpa"
        os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
        logger.info("[ATTN] Using default SDPA backend (explicit ATTN_BACKEND=sdpa)")

    # 若 attention 已因错误 import 顺序提前加载，刷新内部 _state；未加载则首次 import 会读 env
    import sys

    _attn_key = "sam3d_objects.model.backbone.tdfy_dit.modules.attention"
    if _attn_key in sys.modules:
        try:
            sys.modules[_attn_key].refresh_backend_from_env()
        except Exception as e:
            logger.info(f"[ATTN] refresh_backend_from_env skipped: {e}")

    _sparse_key = "sam3d_objects.model.backbone.tdfy_dit.modules.sparse"
    if _sparse_key in sys.modules:
        try:
            sys.modules[_sparse_key].refresh_sparse_attn_from_env()
        except Exception as e:
            logger.info(f"[SPARSE] refresh_sparse_attn_from_env skipped: {e}")
