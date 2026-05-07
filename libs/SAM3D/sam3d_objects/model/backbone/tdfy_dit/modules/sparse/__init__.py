# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import *
from loguru import logger

# 与 dense attention 相同：避免子模块 ``from .. import ATTN`` 拿到导入时快照
_state = {"backend": "spconv", "debug": False, "attn": "sdpa"}


def get_sparse_attn() -> str:
    return _state["attn"]


def get_sparse_debug() -> bool:
    return _state["debug"]


def __from_env():
    import os

    env_sparse_backend = os.environ.get("SPARSE_BACKEND")
    env_sparse_debug = os.environ.get("SPARSE_DEBUG")
    env_sparse_attn = os.environ.get("SPARSE_ATTN_BACKEND")
    if env_sparse_attn is None:
        env_sparse_attn = os.environ.get("ATTN_BACKEND")

    if env_sparse_backend is not None and env_sparse_backend in [
        "spconv",
        "torchsparse",
    ]:
        _state["backend"] = env_sparse_backend
    if env_sparse_debug is not None:
        _state["debug"] = env_sparse_debug == "1"
    if env_sparse_attn is not None and env_sparse_attn in [
        "xformers",
        "flash_attn",
        "torch_flash_attn",
        "sdpa",
    ]:
        _state["attn"] = env_sparse_attn

    logger.info(
        f"[SPARSE] Backend: {_state['backend']}, Attention: {_state['attn']}"
    )


def refresh_sparse_attn_from_env():
    """在 configure_attention_backend 之后调用，修正过早 import sparse 时的默认值。"""
    __from_env()


__from_env()


def set_backend(backend: Literal["spconv", "torchsparse"]):
    _state["backend"] = backend


def set_debug(debug: bool):
    _state["debug"] = debug


def set_attn(attn: Literal["xformers", "flash_attn", "torch_flash_attn", "sdpa"]):
    _state["attn"] = attn


def __getattr__(name: str):
    if name == "BACKEND":
        return _state["backend"]
    if name == "DEBUG":
        return _state["debug"]
    if name == "ATTN":
        return _state["attn"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .basic import *
from .norm import *
from .nonlinearity import *
from .linear import *
from .attention import *
from .conv import *
from .spatial import *
from . import transformer
