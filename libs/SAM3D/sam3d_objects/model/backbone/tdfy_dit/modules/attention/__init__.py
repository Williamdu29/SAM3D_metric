# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import *
from loguru import logger

# 可变状态：子模块必须用 get_backend()，不要用 ``from . import BACKEND``（会得到旧快照）。
_state = {"backend": "sdpa", "debug": False}


def get_backend() -> str:
    return _state["backend"]


def get_debug() -> bool:
    return _state["debug"]


def __from_env():
    import os

    env_attn_backend = os.environ.get("ATTN_BACKEND")
    env_sttn_debug = os.environ.get("ATTN_DEBUG")

    if env_attn_backend is not None and env_attn_backend in [
        "xformers",
        "flash_attn",
        "torch_flash_attn",
        "sdpa",
        "naive",
    ]:
        _state["backend"] = env_attn_backend
    if env_sttn_debug is not None:
        _state["debug"] = env_sttn_debug == "1"

    logger.info(f"[ATTENTION] Using backend: {_state['backend']}")


def refresh_backend_from_env():
    """在已设置 os.environ['ATTN_BACKEND'] 之后调用，修正过早 import 时的默认值。"""
    __from_env()


__from_env()


def set_backend(
    backend: Literal["xformers", "flash_attn", "torch_flash_attn", "sdpa", "naive"],
):
    _state["backend"] = backend


def set_debug(debug: bool):
    _state["debug"] = debug


def __getattr__(name: str):
    """兼容 ``from ...attention import BACKEND``，始终返回当前后端。"""
    if name == "BACKEND":
        return _state["backend"]
    if name == "DEBUG":
        return _state["debug"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .full_attn import *
from .modules import *
from .ca_cache import cross_attn_kv_cache  # noqa: F401
