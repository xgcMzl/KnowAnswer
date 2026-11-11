"""对话模型（DeepSeek）异步流式封装。"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator

from langchain.chat_models import init_chat_model

from know_answer.core.config import get_settings
from know_answer.core.errors import LLMUnavailableError

_lock = threading.Lock()
_chat_model = None


def is_llm_configured() -> bool:
    return bool(get_settings().deepseek_api_key.strip())


def get_chat_model():
    global _chat_model
    if _chat_model is None:
        with _lock:
            if _chat_model is None:
                settings = get_settings()
                if not settings.deepseek_api_key:
                    raise LLMUnavailableError("未配置 DEEPSEEK_API_KEY")
                kwargs: dict[str, object] = {
                    "model": settings.deepseek_model_name,
                    "model_provider": "deepseek",
                    "api_key": settings.deepseek_api_key,
                    "temperature": settings.llm_temperature,
                }
                if settings.deepseek_base_url:
                    kwargs["api_base"] = settings.deepseek_base_url
                _chat_model = init_chat_model(**kwargs)  # type: ignore[arg-type]
    return _chat_model


def reset_chat_model() -> None:
    """测试辅助：清空单例。"""
    global _chat_model
    with _lock:
        _chat_model = None


async def astream_answer(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    model = get_chat_model()
    try:
        async for chunk in model.astream(messages):
            content = chunk.content
            if isinstance(content, str):
                if content:
                    yield content
            elif isinstance(content, list):
                # 兼容多模态 content block：只取文本块。
                text = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                if text:
                    yield text
    except LLMUnavailableError:
        raise
    except Exception as exc:
        raise LLMUnavailableError(f"对话模型调用失败：{exc}") from exc
