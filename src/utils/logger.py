from __future__ import annotations

import io
import logging
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.config import resolve_project_path


LEVEL_ORDER = {
    "NONE": 100,
    "DEBUG": 10,
    "INFO": 20,
    "WARN": 30,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


def normalize_level(level: str | None) -> str:
    level_name = str(level or "INFO").strip().upper()
    if level_name not in LEVEL_ORDER:
        raise ValueError(f"Unknown log level: {level_name}")
    return level_name


class ProjectLogger:
    def __init__(
        self,
        level: str = "INFO",
        output_console: bool = False,
        output_log: bool = True,
        output_log_color: bool = False,
        log_path: str | Path = "logs",
        log_name: str = "sam3d_service.log",
    ) -> None:
        self.level = normalize_level(level)
        self.output_console = bool(output_console)
        self.output_log = bool(output_log)
        self.output_log_color = bool(output_log_color)
        self.log_dir = resolve_project_path(log_path)
        self.log_path = self.log_dir / log_name

        if self.output_log:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def set_level(self, level: str) -> None:
        self.level = normalize_level(level)

    def debug(self, message: Any) -> None:
        self.log("DEBUG", message)

    def info(self, message: Any) -> None:
        self.log("INFO", message)

    def warn(self, message: Any) -> None:
        self.log("WARN", message)

    def warning(self, message: Any) -> None:
        self.log("WARNING", message)

    def error(self, message: Any) -> None:
        self.log("ERROR", message)

    def critical(self, message: Any) -> None:
        self.log("CRITICAL", message)

    def exception(self, message: Any) -> None:
        self.error(f"{message}\n{traceback.format_exc()}")

    def log(self, level: str, message: Any) -> None:
        level_name = normalize_level(level)

        if self.level == "NONE":
            return

        if LEVEL_ORDER[level_name] < LEVEL_ORDER[self.level]:
            return

        line = self._format_line(level_name, str(message))

        if self.output_console:
            stream = sys.stderr if level_name in {"ERROR", "CRITICAL"} else sys.stdout
            print(line, file=stream, flush=True)

        if self.output_log:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _format_line(self, level: str, message: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return f"{timestamp} | {level:<8} | {message}"


class ProjectLoggerProxy:
    def __init__(self) -> None:
        self.current_logger = ProjectLogger(output_console=True, output_log=False)

    def set_logger(self, project_logger: ProjectLogger) -> None:
        self.current_logger = project_logger

    def debug(self, message: Any, **kwargs: Any) -> None:
        self.current_logger.debug(message)

    def info(self, message: Any, **kwargs: Any) -> None:
        self.current_logger.info(message)

    def warn(self, message: Any, **kwargs: Any) -> None:
        self.current_logger.warn(message)

    def warning(self, message: Any, **kwargs: Any) -> None:
        self.current_logger.warning(message)

    def error(self, message: Any, **kwargs: Any) -> None:
        self.current_logger.error(message)

    def critical(self, message: Any, **kwargs: Any) -> None:
        self.current_logger.critical(message)

    def exception(self, message: Any, **kwargs: Any) -> None:
        self.current_logger.exception(message)


logger = ProjectLoggerProxy()


class LoggerWriter(io.TextIOBase):
    def __init__(self, emit_fn, level: str) -> None:
        self.emit_fn = emit_fn
        self.level = level
        self.buffer = ""

    def write(self, text: str) -> int:
        self.buffer += text

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._log(line)

        return len(text)

    def flush(self) -> None:
        if self.buffer:
            self._log(self.buffer)
            self.buffer = ""

    def _log(self, text: str) -> None:
        message = text.strip()
        if not message:
            return
        self.emit_fn(self.level, message)


class ProjectLogHandler(logging.Handler):
    def __init__(self, emit_fn) -> None:
        super().__init__()
        self.emit_fn = emit_fn

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        level_name = record.levelname.upper()

        if level_name in {"WARNING", "WARN"}:
            self.emit_fn("WARN", message)
        elif level_name in {"ERROR", "CRITICAL"}:
            self.emit_fn("ERROR", message)
        elif level_name == "DEBUG":
            self.emit_fn("DEBUG", message)
        else:
            self.emit_fn("INFO", message)


@contextmanager
def bridge_external_logs(project_logger: ProjectLogger):
    """
    Capture print/stdout/stderr/root logging into ProjectLogger.

    用途：
    with bridge_external_logs(project_logger):
        reconstructor.recon(...)
    """

    def emit(level: str, message: str) -> None:
        project_logger.log(level, message)

    stdout_writer = LoggerWriter(emit, "INFO")
    stderr_writer = LoggerWriter(emit, "ERROR")

    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level

    handler = ProjectLogHandler(emit)
    handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger.handlers = [handler]
    root_logger.setLevel(logging.DEBUG)

    try:
        with redirect_stdout(stdout_writer), redirect_stderr(stderr_writer):
            yield
    finally:
        stdout_writer.flush()
        stderr_writer.flush()
        root_logger.handlers = previous_handlers
        root_logger.setLevel(previous_level)


def create_logger(
    config: dict,
    log_name: Optional[str] = None,
    module_name: str = "sam3d_service",
) -> ProjectLogger:
    logger_config = config.get("logger", {})
    paths_config = config.get("paths", {})

    logs_dir = (
        logger_config.get("log_path")
        or paths_config.get("logs_dir")
        or "logs"
    )

    startup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_log_name = f"{module_name}_{startup_time}.log"

    project_logger = ProjectLogger(
        level=str(logger_config.get("level", "INFO")),
        output_console=bool(logger_config.get("output_console", False)),
        output_log=bool(logger_config.get("output_log", True)),
        output_log_color=bool(logger_config.get("output_log_color", False)),
        log_path=str(logs_dir),
        log_name=str(log_name or default_log_name),
    )

    logger.set_logger(project_logger)
    return project_logger