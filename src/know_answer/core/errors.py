"""应用异常体系与 FastAPI 统一错误响应。"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """所有可预期业务异常的基类。"""

    status_code: int = 500
    code: str = "INTERNAL_ERROR"
    default_message: str = "服务内部错误"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        self.message = message or self.default_message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(self.message)


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"
    default_message = "资源不存在"


class InvalidArgumentError(AppError):
    status_code = 400
    code = "INVALID_ARGUMENT"
    default_message = "请求参数不合法"


class InvalidFileTypeError(AppError):
    status_code = 400
    code = "INVALID_FILE_TYPE"
    default_message = "不支持的文件类型"


class InvalidDocumentError(AppError):
    status_code = 400
    code = "INVALID_DOCUMENT"
    default_message = "文档解析失败"


class FileTooLargeError(AppError):
    status_code = 413
    code = "FILE_TOO_LARGE"
    default_message = "文件超过大小限制"


class FileBusyError(AppError):
    status_code = 409
    code = "FILE_BUSY"
    default_message = "文件正在处理中，请稍后再试"


class MilvusUnavailableError(AppError):
    status_code = 503
    code = "MILVUS_UNAVAILABLE"
    default_message = "向量检索服务暂不可用"


class EmbeddingUnavailableError(AppError):
    status_code = 503
    code = "EMBEDDING_UNAVAILABLE"
    default_message = "向量化服务暂不可用"


class LLMUnavailableError(AppError):
    status_code = 503
    code = "LLM_UNAVAILABLE"
    default_message = "对话模型暂不可用"


def error_payload(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(exc.code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        if errors:
            first = errors[0]
            location = ".".join(str(item) for item in first.get("loc", []))
            message = f"{location}: {first.get('msg', '请求参数不合法')}"
        else:
            message = "请求参数不合法"
        return JSONResponse(
            status_code=400,
            content=error_payload("INVALID_ARGUMENT", message),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理的异常", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=error_payload("INTERNAL_ERROR", "服务内部错误"),
        )
