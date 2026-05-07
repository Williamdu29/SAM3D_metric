# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Cross-attention K/V cache helpers for DiT generators.

During diffusion / flow sampling the image/mask/pointmap conditioning
tokens (``context``) are constant across all sampling steps, but the
default implementation recomputes ``to_kv(context)`` inside every
cross-attention block on every step (and again for each CFG branch).
For the SAM3D DiTs this accounts for a non-trivial amount of memory
traffic and matmul work.

This module provides an explicit-residency cache: a context manager
that (a) flips a flag on every cross-attention module and (b) lets the
module memoize ``to_kv(context)`` keyed by ``id(context)``. Wrappers
such as ``SparseStructureFlowTdfyWrapper`` also keep a persistent
zero-tensor for the CFG-uncond branch, so both ``cond`` and
``uncond-zero`` branches benefit from caching.

Usage::

    from sam3d_objects.model.backbone.tdfy_dit.modules.attention.ca_cache \
        import cross_attn_kv_cache

    with cross_attn_kv_cache(pipeline._pipeline):
        result = pipeline.run(image=..., mask=...)

The manager clears all caches on exit, so the residency is bounded to
the lifetime of the ``with`` block.

**显存：** 若在 ``with`` 内调用会跑满整段 ``pipeline.run()``（含 mesh decode），
K/V 张量会一直驻留到 run 结束，可能推高峰值显存导致 OOM。因此 pipeline 在
``sample_sparse_structure`` / ``sample_slat`` 之后、``decode_slat`` 之前会调用
``release_cross_attn_kv_cache()`` 主动释放（仍保留 ``with`` 在 step 内的加速）。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, List

import torch
from loguru import logger


def _resolve_nn_subtree(root: Any) -> torch.nn.Module:
    """``InferencePipeline`` / ``InferencePipelinePointMap`` 不是 ``nn.Module``，子网络在 ``.models`` 里。"""
    if isinstance(root, torch.nn.Module):
        return root
    models = getattr(root, "models", None)
    if isinstance(models, torch.nn.Module):
        return models
    raise TypeError(
        "cross_attn_kv_cache(root): root 必须是 torch.nn.Module，或带有 .models (ModuleDict) 的 pipeline，"
        f"当前为 {type(root).__name__}"
    )


def _iter_cross_attn_modules(root: Any) -> List[torch.nn.Module]:
    """Return every cross-type attention module under ``root``."""
    # 惰性导入：避免 dense attention/__init__ 加载 ca_cache 时 sparse.attention 尚未初始化（循环依赖）。
    from .modules import MultiHeadAttention
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.modules import (
        SparseMultiHeadAttention,
    )

    nn_root = _resolve_nn_subtree(root)
    out: List[torch.nn.Module] = []
    for m in nn_root.modules():
        if isinstance(m, (MultiHeadAttention, SparseMultiHeadAttention)):
            if getattr(m, "_type", None) == "cross":
                out.append(m)
    return out


def _iter_zero_cond_owners(root: Any) -> List[torch.nn.Module]:
    """Modules that may hold a cached ``_cached_zero_cond`` tensor."""
    nn_root = _resolve_nn_subtree(root)
    out: List[torch.nn.Module] = []
    for m in nn_root.modules():
        if hasattr(m, "_cached_zero_cond"):
            out.append(m)
    return out


def enable_cross_attn_cache(modules: Iterable[torch.nn.Module]) -> int:
    n = 0
    for m in modules:
        m._ca_cache_enabled = True
        m._ca_cache = {}
        n += 1
    return n


def disable_cross_attn_cache(modules: Iterable[torch.nn.Module]) -> None:
    for m in modules:
        m._ca_cache_enabled = False
        m._ca_cache = {}


def clear_zero_cond(owners: Iterable[torch.nn.Module]) -> None:
    for m in owners:
        if hasattr(m, "_cached_zero_cond"):
            try:
                del m._cached_zero_cond
            except AttributeError:
                pass


def release_cross_attn_kv_cache(root: Any, verbose: bool = False) -> None:
    """关掉 cross-attn 缓存并清空 K/V / zero-cond 张量，降低峰值显存。

    应在每段 DiT 采样（ss / slat）结束后、decode 与大张量分配前调用。
    与 ``cross_attn_kv_cache`` 的 ``with`` 兼容：释放后下一步采样会重新填充缓存。
    """
    try:
        cross_mods = _iter_cross_attn_modules(root)
        zc_owners = _iter_zero_cond_owners(root)
        disable_cross_attn_cache(cross_mods)
        clear_zero_cond(zc_owners)
        if verbose:
            logger.info("[CA-CACHE] Released K/V tensors between DiT sampling and decode")
    except TypeError:
        pass
    except Exception as e:
        if verbose:
            logger.warning("[CA-CACHE] release_cross_attn_kv_cache failed: {}", e)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def cross_attn_kv_cache(root: Any, verbose: bool = False):
    """Enable cross-attention K/V caching on all cross-attn modules.

    ``root`` 可以是 ``torch.nn.Module``，也可以是 ``InferencePipeline*``（会用 ``root.models`` 遍历子模块）。

    The cache is cleared both on entry and on exit, so stale tensors from
    prior runs can never be reused. Safe to nest (inner scope behaves as
    a no-op because caches are already enabled and get cleared at exit).
    """
    cross_mods = _iter_cross_attn_modules(root)
    zc_owners = _iter_zero_cond_owners(root)

    # Start clean (guard against stale caches from aborted previous runs).
    disable_cross_attn_cache(cross_mods)
    clear_zero_cond(zc_owners)

    n = enable_cross_attn_cache(cross_mods)
    if verbose:
        logger.info(f"[CA-CACHE] Enabled cross-attn K/V cache on {n} modules")

    try:
        yield
    finally:
        disable_cross_attn_cache(cross_mods)
        clear_zero_cond(zc_owners)
        if verbose:
            logger.info("[CA-CACHE] Cleared cross-attn K/V cache")
