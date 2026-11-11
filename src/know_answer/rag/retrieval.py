"""向量检索与来源片段构造。"""

from __future__ import annotations

import logging
from typing import Any

from know_answer.core.config import get_settings
from know_answer.core.errors import EmbeddingUnavailableError, MilvusUnavailableError
from know_answer.rag.embedding import embed_query
from know_answer.rag.vector_store import search

logger = logging.getLogger(__name__)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _to_source(node: Any, score: float, index: int) -> dict[str, Any]:
    metadata = node.metadata or {}
    raw_file_id = metadata.get("file_id")
    try:
        file_id = int(raw_file_id) if raw_file_id is not None else None
    except (TypeError, ValueError):
        file_id = None

    settings = get_settings()
    return {
        "index": index,
        "node_id": node.node_id,
        "file_id": file_id,
        "file_name": metadata.get("file_name") or "",
        "knowledge_base_id": metadata.get("knowledge_base_id"),
        "page_label": metadata.get("page_label"),
        "chunk_index": metadata.get("chunk_index"),
        "score": float(score) if score is not None else None,
        "text": _truncate(node.get_content(), settings.rag_chunk_max_chars),
    }


def retrieve(question: str, knowledge_base_id: int | None) -> list[dict[str, Any]]:
    """同步检索；调用方应放到线程池中执行。"""
    settings = get_settings()

    try:
        query_embedding = embed_query(question)
    except Exception as exc:
        raise EmbeddingUnavailableError(f"生成查询向量失败：{exc}") from exc

    try:
        nodes, similarities = search(
            query_embedding,
            knowledge_base_id=knowledge_base_id,
            top_k=settings.rag_top_k,
        )
    except Exception as exc:
        raise MilvusUnavailableError(f"向量检索失败：{exc}") from exc

    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    used_chars = 0

    for node, score in zip(nodes, similarities):
        if score is not None and score < settings.rag_min_score:
            continue
        text = (node.get_content() or "").strip()
        if not text:
            continue
        dedupe_key = text[:200]
        if dedupe_key in seen:
            continue
        truncated = _truncate(text, settings.rag_chunk_max_chars)
        if used_chars + len(truncated) > settings.rag_context_max_chars:
            break
        seen.add(dedupe_key)
        used_chars += len(truncated)
        sources.append(_to_source(node, float(score or 0.0), len(sources) + 1))

    return sources
