"""Embedding 模型单例。

入库与检索共用同一个模型实例；HuggingFace 模型首次加载较慢，因此加锁初始化。
"""

from __future__ import annotations

import threading

from llama_index.core.embeddings import BaseEmbedding

from know_answer.core.config import get_settings

_lock = threading.Lock()
_embed_model: BaseEmbedding | None = None


def _build_embed_model() -> BaseEmbedding:
    settings = get_settings()
    provider = settings.embedding_provider.strip().lower()

    if provider in {"huggingface", "hf"}:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        kwargs: dict[str, object] = {
            "model_name": settings.embedding_model_name,
            "device": settings.embedding_device,
        }
        if settings.embedding_cache_folder:
            kwargs["cache_folder"] = settings.embedding_cache_folder
        if settings.embedding_query_instruction:
            kwargs["query_instruction"] = settings.embedding_query_instruction
        return HuggingFaceEmbedding(**kwargs)  # type: ignore[arg-type]

    if provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        if not settings.embedding_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai 时必须配置 EMBEDDING_API_KEY")
        return OpenAIEmbedding(
            model=settings.embedding_model_name,
            api_key=settings.embedding_api_key,
            api_base=settings.embedding_api_base or None,
        )

    raise ValueError(f"不支持的 EMBEDDING_PROVIDER：{settings.embedding_provider}")


def get_embed_model() -> BaseEmbedding:
    global _embed_model
    if _embed_model is None:
        with _lock:
            if _embed_model is None:
                _embed_model = _build_embed_model()
    return _embed_model


def is_embedding_loaded() -> bool:
    return _embed_model is not None


def reset_embed_model() -> None:
    """测试辅助：清空单例。"""
    global _embed_model
    with _lock:
        _embed_model = None


def embed_query(text: str) -> list[float]:
    return get_embed_model().get_query_embedding(text)
