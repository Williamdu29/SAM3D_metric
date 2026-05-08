from __future__ import annotations

import base64
import binascii
import gc
import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request



'''
本地的 SAM3DReconstructor 类封装成一个 HTTP 服务。核心模型/算法逻辑不在这个文件里，而是动态从：

src/reconstrcution/metric_V1/sam3D_class_V1.py

导入 SAM3DReconstructor。这个 Flask 服务主要负责：

启动服务；
加载配置、日志、环境变量；
创建并保存 SAM3DReconstructor 实例；
接收 HTTP JSON 请求；
校验输入图片、深度图、mask、相机内参；
调用 reconstructor.load_model()、reconstructor.recon()、reconstructor.release()；
把结果统一包装成 JSON 返回
'''



_PROJECT_ROOT_BOOTSTRAP = Path(
    os.environ.get(
        "SAM3D_PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
).expanduser().resolve()

_SRC_ROOT_BOOTSTRAP = _PROJECT_ROOT_BOOTSTRAP / "src"

if str(_SRC_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT_BOOTSTRAP))

from utils.config import load_config
from utils.logger import bridge_external_logs, create_logger, logger


LOG_VALUE_PREVIEW_LIMIT = 160
LOG_BODY_PREVIEW_LIMIT = 600
IMAGE_FIELD_NAMES = {
    "rgb_image",
    "rgb_image_base64",
    "depth_image",
    "depth_image_base64",
    "mask_image",
    "mask_image_base64",
}

RECON_CODE_OK = 0
RECON_CODE_INVALID_INPUT = -100
RECON_CODE_INPUT_FILE_NOT_FOUND = -101
RECON_CODE_MODEL_LOAD_FAILED = -102
RECON_CODE_INTERNAL_ERROR = -1

RECON_MESSAGE_OK = "ok"
RECON_MESSAGE_INVALID_INPUT = "输入解析错误"
RECON_MESSAGE_INPUT_FILE_NOT_FOUND = "输入文件不存在或不可读"
RECON_MESSAGE_MODEL_LOAD_FAILED = "SAM3D模型加载失败"
RECON_MESSAGE_INTERNAL_ERROR = "SAM3D重建失败"


class PayloadValidationError(ValueError):
    """Raised when the /reconstruct request body cannot be parsed."""


class InputFileNotFoundError(FileNotFoundError):
    """Raised when a user-provided input path does not exist."""

'''
def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("sam3d_reconstruction_service")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger
'''


'''
logger = _setup_logger()
'''


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be a float, got {value!r}") from exc
    if not np.isfinite(parsed):
        raise RuntimeError(f"Environment variable {name} must be finite, got {value!r}")
    return parsed


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an int, got {value!r}") from exc
    return parsed


def _resolve_project_root() -> Path:
    env_project_root = os.environ.get("SAM3D_PROJECT_ROOT", "").strip()
    if env_project_root:
        return Path(env_project_root).expanduser().resolve()
    # Expected location: <project_root>/src/service/sam3d_service.py
    return Path(__file__).resolve().parents[2]


def _truncate_log_text(value: str, limit: int = LOG_VALUE_PREVIEW_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...(truncated, total_chars={len(value)})"


def _summarize_payload_value(value: Any, *, field_name: str | None = None, depth: int = 0) -> Any:
    if isinstance(value, str):
        if field_name in IMAGE_FIELD_NAMES:
            return {
                "type": "base64-string",
                "length": len(value),
                "preview": _truncate_log_text(value, limit=80),
            }
        return _truncate_log_text(value)

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, list):
        if depth >= 1:
            return f"<list length={len(value)}>"
        summarized_items = [_summarize_payload_value(item, depth=depth + 1) for item in value[:3]]
        if len(value) > 3:
            summarized_items.append(f"...({len(value) - 3} more items)")
        return summarized_items

    if isinstance(value, dict):
        if depth >= 2:
            return f"<object keys={list(value.keys())}>"
        return {
            str(key): _summarize_payload_value(item_value, field_name=str(key), depth=depth + 1)
            for key, item_value in value.items()
        }

    return f"<{type(value).__name__}>"


def _build_request_body_summary(payload: Any, raw_body_text: str) -> Any:
    if payload is not None:
        return _summarize_payload_value(payload)
    stripped_body = raw_body_text.strip()
    if not stripped_body:
        return "<empty body>"
    return _truncate_log_text(stripped_body, limit=LOG_BODY_PREVIEW_LIMIT)


def _require_json_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PayloadValidationError("Request body must be a JSON object")
    return payload


def _normalize_optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set)):
        raise PayloadValidationError(f"`{field_name}` must be a string-compatible value")
    normalized = str(value).strip()
    return normalized if normalized else None


def _parse_bool(value: Any, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise PayloadValidationError(f"`{field_name}` must be a boolean")


def _parse_int(value: Any, field_name: str, default: int | None = None, *, min_value: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        raise PayloadValidationError(f"`{field_name}` must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise PayloadValidationError(f"`{field_name}` must be an integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise PayloadValidationError(f"`{field_name}` must be an integer")
    parsed = int(numeric)
    if min_value is not None and parsed < min_value:
        raise PayloadValidationError(f"`{field_name}` must be >= {min_value}")
    return parsed


def _parse_float(value: Any, field_name: str, default: float | None = None, *, min_value: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        raise PayloadValidationError(f"`{field_name}` must be a float")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PayloadValidationError(f"`{field_name}` must be a float") from exc
    if not np.isfinite(parsed):
        raise PayloadValidationError(f"`{field_name}` must be finite")
    if min_value is not None and parsed < min_value:
        raise PayloadValidationError(f"`{field_name}` must be >= {min_value}")
    return parsed


def _parse_k_input(value: Any) -> list[float]:
    if value is None:
        raise PayloadValidationError("`k_input` must be provided")
    try:
        k_array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PayloadValidationError("`k_input` must be numeric") from exc
    if k_array.size != 9:
        raise PayloadValidationError(f"`k_input` must contain exactly 9 elements, got {k_array.size}")
    if not np.isfinite(k_array).all():
        raise PayloadValidationError("`k_input` contains NaN or Inf")
    return k_array.reshape(-1).astype(float).tolist()


def _strip_data_url_prefix(value: str) -> tuple[str, str | None]:
    text = value.strip()
    if not text.startswith("data:"):
        return text, None

    header, sep, data = text.partition(",")
    if not sep:
        raise PayloadValidationError("Invalid data URL base64 image")

    mime = header[5:].split(";", 1)[0].strip().lower()
    suffix = None
    if mime in {"image/jpeg", "image/jpg"}:
        suffix = ".jpg"
    elif mime == "image/png":
        suffix = ".png"
    elif mime == "image/tiff":
        suffix = ".tiff"
    return data, suffix


def _decode_base64_to_file(value: Any, output_path: Path, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PayloadValidationError(f"`{field_name}` must be a non-empty base64 string")

    base64_text, inferred_suffix = _strip_data_url_prefix(value)
    if inferred_suffix and output_path.suffix.lower() != inferred_suffix:
        output_path = output_path.with_suffix(inferred_suffix)

    try:
        image_bytes = base64.b64decode(base64_text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PayloadValidationError(f"`{field_name}` is not valid base64") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
    return output_path


def _require_existing_file(path_text: str, field_name: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise InputFileNotFoundError(f"`{field_name}` file not found: {path}")
    return path.resolve()


def _resolve_input_file(
    body: dict[str, Any],
    *,
    path_field: str,
    base64_fields: tuple[str, ...],
    default_filename: str,
    temp_input_dir: Path,
) -> Path:
    path_value = _normalize_optional_string(body.get(path_field), path_field)
    provided_base64_fields = [field for field in base64_fields if body.get(field) is not None]

    if path_value and provided_base64_fields:
        raise PayloadValidationError(
            f"Provide either `{path_field}` or one of {provided_base64_fields}, not both"
        )

    if path_value:
        return _require_existing_file(path_value, path_field)

    if provided_base64_fields:
        field_name = provided_base64_fields[0]
        return _decode_base64_to_file(
            body[field_name],
            temp_input_dir / default_filename,
            field_name,
        )

    raise PayloadValidationError(f"`{path_field}` or `{base64_fields[0]}` must be provided")


def _build_task_id(operation: str) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    milliseconds = int((time.time() % 1) * 1000)
    return f"task-{timestamp}{milliseconds:03d}-{operation}-1"


def _json_response(task_id: str, code: int, message: str, result: dict[str, Any] | None = None, http_status: int = 200):
    return (
        jsonify(
            {
                "task_id": task_id,
                "code": code,
                "message": message,
                "result": {} if result is None else result,
            }
        ),
        http_status,
    )


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    return value


def _parse_reconstruct_request(body: dict[str, Any], temp_input_dir: Path) -> dict[str, Any]:
    rgb_path = _resolve_input_file(
        body,
        path_field="rgb_path",
        base64_fields=("rgb_image", "rgb_image_base64"),
        default_filename="rgb.png",
        temp_input_dir=temp_input_dir,
    )
    depth_path = _resolve_input_file(
        body,
        path_field="depth_path",
        base64_fields=("depth_image", "depth_image_base64"),
        default_filename="depth.png",
        temp_input_dir=temp_input_dir,
    )
    mask_path = _resolve_input_file(
        body,
        path_field="mask_path",
        base64_fields=("mask_image", "mask_image_base64"),
        default_filename="mask.png",
        temp_input_dir=temp_input_dir,
    )

    k_input = _parse_k_input(body.get("k_input"))

    out_dir = _normalize_optional_string(body.get("out_dir"), "out_dir")
    case_id = _normalize_optional_string(body.get("case_id"), "case_id")
    run_sam3d = _parse_bool(body.get("run_sam3d"), "run_sam3d", default=True)
    save_metrics_npy = _parse_bool(body.get("save_metrics_npy"), "save_metrics_npy", default=True)
    save_metrics_json = _parse_bool(body.get("save_metrics_json"), "save_metrics_json", default=True)
    release_after_recon = _parse_bool(
        body.get("release_after_recon"),
        "release_after_recon",
        default=_truthy_env("SAM3D_RELEASE_AFTER_RECON", False),
    )

    seed = _parse_int(body.get("seed"), "seed", default=42, min_value=0)
    output_prefix = _normalize_optional_string(body.get("output_prefix"), "output_prefix") or "sam3d_metric_general"
    raw_sam_name = _normalize_optional_string(body.get("raw_sam_name"), "raw_sam_name") or "sam3d_raw_v8.ply"
    anchor_name = _normalize_optional_string(body.get("anchor_name"), "anchor_name") or "anchor_metric_visible_surface_v8.ply"

    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "mask_path": str(mask_path),
        "k_input": k_input,
        "out_dir": out_dir,
        "case_id": case_id,
        "run_sam3d": run_sam3d,
        "output_prefix": output_prefix,
        "raw_sam_name": raw_sam_name,
        "anchor_name": anchor_name,
        "save_metrics_npy": save_metrics_npy,
        "save_metrics_json": save_metrics_json,
        "seed": seed,
        "release_after_recon": release_after_recon,
    }


def _load_reconstructor_class(project_root: Path):
    metric_class_dir = project_root / "src" / "reconstrcution" / "metric_V1"
    if not metric_class_dir.is_dir():
        raise RuntimeError(f"Cannot find metric_V1 directory: {metric_class_dir}")
    if str(metric_class_dir) not in sys.path:
        sys.path.insert(0, str(metric_class_dir))
    from sam3D_class_V1 import SAM3DReconstructor  # noqa: E402

    return SAM3DReconstructor


def create_app() -> Flask:

    project_root = _resolve_project_root()

    config = load_config(project_root / "config.yaml")
    project_logger = create_logger(
        config,
        module_name=Path(__file__).stem,
    )

    logger.info(f"Project logger initialized, log_path={project_logger.log_path}")

    sam3d_root = Path(os.environ.get("SAM3D_ROOT", str(project_root / "libs" / "SAM3D"))).expanduser().resolve()
    sam3d_config = os.environ.get(
        "SAM3D_CONFIG",
        "/home/user/datas/hc/data/ckpts/sam3d-obj/models/checkpoints/pipeline.yaml",
    ).strip()
    output_root = Path(os.environ.get("SAM3D_OUTPUT_ROOT", str(project_root / "Outputs" / "service_outputs"))).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Initializing SAM3D reconstruction service, "
        f"project_root={project_root}, sam3d_root={sam3d_root}, "
        f"sam3d_config={sam3d_config}, output_root={output_root}"
    )

    SAM3DReconstructor = _load_reconstructor_class(project_root)
    reconstructor = SAM3DReconstructor(
        sam3d_root=str(sam3d_root),
        sam3d_config=sam3d_config,
        out_dir=None,
        voxel_size=_float_env("SAM3D_VOXEL_SIZE", 0.003),
        icp_max_iter=_int_env("SAM3D_ICP_MAX_ITER", 40),
        mesh_poisson_depth=_int_env("SAM3D_MESH_POISSON_DEPTH", 8),
        mesh_density_q=_float_env("SAM3D_MESH_DENSITY_Q", 0.02),
        sam_compile=_truthy_env("SAM3D_COMPILE_MODEL", False),
        verbose=_truthy_env("SAM3D_VERBOSE", True),
    )

    if _truthy_env("SAM3D_PRELOAD_MODEL", False):
        logger.info("SAM3D_PRELOAD_MODEL=1, loading model during service startup")
        try:
            reconstructor.load_model(force_reload=False)
        except Exception as exc:
            logger.error(f"Failed to preload SAM3D model: {exc}")
            raise

    app = Flask(__name__)
    app.config["PROJECT_ROOT"] = str(project_root)
    app.config["PROJECT_LOGGER"] = project_logger
    app.config["SAM3D_ROOT"] = str(sam3d_root)
    app.config["SAM3D_CONFIG"] = sam3d_config
    app.config["OUTPUT_ROOT"] = str(output_root)
    app.config["SERVICE_PORT"] = _int_env("SAM3D_SERVICE_PORT", 58860)
    app.config["RECONSTRUCTOR"] = reconstructor
    app.config["RECON_LOCK"] = threading.Lock()

    @app.route("/health", methods=["GET"])
    def health_check():
        reconstructor_obj = app.config["RECONSTRUCTOR"] # 类实例对象，包含属性 model_loaded 来表示模型是否已加载
        return (
            jsonify(
                {
                    "status": "healthy",
                    "project_root": app.config["PROJECT_ROOT"],
                    "sam3d_root": app.config["SAM3D_ROOT"],
                    "sam3d_config": app.config["SAM3D_CONFIG"],
                    "output_root": app.config["OUTPUT_ROOT"],
                    "service_port": app.config["SERVICE_PORT"],
                    "model_loaded": bool(getattr(reconstructor_obj, "model_loaded", False)),
                }
            ),
            200,
        )

    @app.route("/load_model", methods=["POST"])
    def load_model():
        task_id = _build_task_id("load-model")
        raw_payload = request.get_json(silent=True) or {}
        force_reload = _parse_bool(raw_payload.get("force_reload"), "force_reload", default=False)
        start_time = time.perf_counter()
        try:
            with app.config["RECON_LOCK"]:
                with bridge_external_logs(app.config["PROJECT_LOGGER"]):
                    app.config["RECONSTRUCTOR"].load_model(force_reload=force_reload) # 调用了 SAM3DReconstructor 类的 load_model 方法加载模型，加载后会将模型保存在类实例的属性中，供后续重建调用使用
        except Exception as exc:
            logger.error(f"/load_model failed: task_id={task_id}, reason={exc}")
            logger.error(traceback.format_exc())
            return _json_response(
                task_id,
                RECON_CODE_MODEL_LOAD_FAILED,
                RECON_MESSAGE_MODEL_LOAD_FAILED,
                result={"detail": str(exc)},
            )

        elapsed = time.perf_counter() - start_time
        return _json_response(
            task_id,
            RECON_CODE_OK,
            RECON_MESSAGE_OK,
            result={"model_loaded": True, "elapsed_seconds": elapsed},
        )

    @app.route("/release", methods=["POST"])
    def release_model():
        task_id = _build_task_id("release")
        raw_payload = request.get_json(silent=True) or {}
        clear_imports = _parse_bool(raw_payload.get("clear_imports"), "clear_imports", default=False)
        reset_timings = _parse_bool(raw_payload.get("reset_timings"), "reset_timings", default=False)
        try:
            with app.config["RECON_LOCK"]:
                with bridge_external_logs(app.config["PROJECT_LOGGER"]):
                    release_result = app.config["RECONSTRUCTOR"].release(
                        clear_imports=clear_imports,
                        reset_timings=reset_timings,
                    )
                    gc.collect()
        except Exception as exc:
            logger.error(f"/release failed: task_id={task_id}, reason={exc}")
            logger.error(traceback.format_exc())
            return _json_response(
                task_id,
                RECON_CODE_INTERNAL_ERROR,
                "模型释放失败",
                result={"detail": str(exc)},
            )
        return _json_response(task_id, RECON_CODE_OK, RECON_MESSAGE_OK, result=_make_json_safe(release_result))

    @app.route("/reconstruct", methods=["POST"])
    def reconstruct():
        task_id = _build_task_id("reconstruct")
        raw_body_text = request.get_data(cache=True, as_text=True)
        raw_payload = request.get_json(silent=True)
        request_body_summary = _build_request_body_summary(raw_payload, raw_body_text)

        logger.info(
            "Incoming /reconstruct request "
            f"remote_addr={request.remote_addr}, "
            f"content_type={request.content_type}, "
            f"content_length={request.content_length}, "
            f"body={request_body_summary}"
        )

        start_time = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="sam3d_service_input_") as temp_dir_text:
            temp_input_dir = Path(temp_dir_text)
            try:
                body = _require_json_object(raw_payload)
                parsed = _parse_reconstruct_request(body, temp_input_dir=temp_input_dir)
            except InputFileNotFoundError as exc:
                logger.error(f"Rejected /reconstruct request task_id={task_id}, reason={exc}")
                return _json_response(
                    task_id,
                    RECON_CODE_INPUT_FILE_NOT_FOUND,
                    RECON_MESSAGE_INPUT_FILE_NOT_FOUND,
                    result={"detail": str(exc)},
                )
            except PayloadValidationError as exc:
                logger.error(f"Rejected /reconstruct request task_id={task_id}, reason={exc}")
                return _json_response(
                    task_id,
                    RECON_CODE_INVALID_INPUT,
                    RECON_MESSAGE_INVALID_INPUT,
                    result={"detail": str(exc)},
                )

            out_dir = parsed["out_dir"]
            if out_dir is None:
                case_id = parsed["case_id"] or task_id
                safe_case_id = case_id.replace("/", "_").replace("\\", "_").strip()
                out_dir = str(Path(app.config["OUTPUT_ROOT"]) / safe_case_id)

            logger.info(
                "Received /reconstruct request "
                f"task_id={task_id}, out_dir={out_dir}, "
                f"run_sam3d={parsed['run_sam3d']}, seed={parsed['seed']}"
            )

            try:
                with app.config["RECON_LOCK"]:
                    with bridge_external_logs(app.config["PROJECT_LOGGER"]):
                        result = app.config["RECONSTRUCTOR"].recon( # 调用了类的 recon 方法进行重建，传入了之前解析的输入参数，重建完成后会返回一个包含各种输出路径和指标的结果字典
                            rgb_path=parsed["rgb_path"],
                            depth_path=parsed["depth_path"],
                            mask_path=parsed["mask_path"],
                            k_input=parsed["k_input"],
                            out_dir=out_dir,
                            run_sam3d=parsed["run_sam3d"],
                            output_prefix=parsed["output_prefix"],
                            raw_sam_name=parsed["raw_sam_name"],
                            anchor_name=parsed["anchor_name"],
                            save_metrics_npy=parsed["save_metrics_npy"],
                            save_metrics_json=parsed["save_metrics_json"],
                            seed=parsed["seed"],
                        )
                        if parsed["release_after_recon"]:
                            release_result = app.config["RECONSTRUCTOR"].release(
                                clear_imports=False,
                                reset_timings=False,
                            )
                        else:
                            release_result = None
            except Exception as exc:
                logger.error(f"/reconstruct failed: task_id={task_id}, reason={exc}")
                logger.error(traceback.format_exc())
                return _json_response(
                    task_id,
                    RECON_CODE_INTERNAL_ERROR,
                    RECON_MESSAGE_INTERNAL_ERROR,
                    result={"detail": str(exc)},
                )

        elapsed = time.perf_counter() - start_time
        response_result = _make_json_safe(result)
        response_result["elapsed_seconds"] = elapsed
        response_result["release_after_recon"] = bool(parsed["release_after_recon"])
        response_result["log_path"] = str(app.config["PROJECT_LOGGER"].log_path)
        if release_result is not None:
            response_result["release_result"] = _make_json_safe(release_result)

        logger.info(
            "Completed /reconstruct request "
            f"task_id={task_id}, elapsed_seconds={elapsed:.4f}, code={RECON_CODE_OK}"
        )
        return _json_response(task_id, RECON_CODE_OK, RECON_MESSAGE_OK, result=response_result)

    return app


def run_app(app: Flask | None = None) -> None:
    app = create_app() if app is None else app
    host = os.environ.get("SAM3D_SERVICE_HOST", "0.0.0.0")
    port = int(app.config.get("SERVICE_PORT", 58860))
    logger.info(
        "Starting SAM3D reconstruction service in single-process, single-thread mode "
        f"(host={host}, port={port})"
    )
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=False,
        use_reloader=False,
    )
