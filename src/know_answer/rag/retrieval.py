"""检索与来源片段构造。

流程：向量/BM25 双路召回 → 按 ``parent_id`` 折叠子块 → 批量取父块 →
字符预算裁剪 → 组装 sources（父块作为 prompt 上下文，子块作为命中片段）。
"""

from __future__ import annotations

import logging
from typing import Any

from know_answer.core.config import get_settings
from know_answer.core.errors import EmbeddingUnavailableError, MilvusUnavailableError
from know_answer.db.parent_store import ParentRecord, fetch_parents
from know_answer.rag.embedding import embed_query
from know_answer.rag.vector_store import search

logger = logging.getLogger(__name__)

# 前端「命中子块」展示的最大字符数
MATCH_TEXT_MAX_CHARS = 200


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _to_source(
    node: Any,
    score: float,
    index: int,
    *,
    text: str,
    page_label: Any,
    parent_index: Any,
    score_type: str,
    matched_text: str | None,
) -> dict[str, Any]:
    metadata = node.metadata or {}
    raw_file_id = metadata.get("file_id")
    try:
        file_id = int(raw_file_id) if raw_file_id is not None else None
    except (TypeError, ValueError):
        file_id = None

    return {
        "index": index,
        "node_id": node.node_id,
        "parent_id": metadata.get("parent_id"),
        "parent_index": parent_index,
        "file_id": file_id,
        "file_name": metadata.get("file_name") or "",
        "knowledge_base_id": metadata.get("knowledge_base_id"),
        "page_label": page_label,
        "chunk_index": metadata.get("chunk_index"),
        "score": float(score) if score is not None else None,
        "score_type": score_type,
        "matched_text": matched_text,
        "text": text,
    }


def _unique_parent_ids(nodes: list[Any]) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for node in nodes:
        parent_id = (node.metadata or {}).get("parent_id")
        if not parent_id:
            continue
        key = str(parent_id)
        if key in seen:
            continue
        seen.add(key)
        ids.append(key)
    return ids


def _load_parents(parent_ids: list[str]) -> dict[str, ParentRecord]:
    if not parent_ids:
        return {}
    try:
        return fetch_parents(parent_ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning("批量读取父块失败，将退回子块文本：%s", exc)
        return {}


def retrieve(question: str, knowledge_base_id: int | None) -> list[dict[str, Any]]:
    """同步检索；调用方应放到线程池中执行。"""
    settings = get_settings()

    try:
        query_embedding = embed_query(question)
    except Exception as exc:
        raise EmbeddingUnavailableError(f"生成查询向量失败：{exc}") from exc

    child_top_k = min(
        max(1, settings.rag_top_k) * max(1, settings.rag_child_top_k_multiplier),
        50,
    )

    try:
        outcome = search(
            query_embedding,
            query_str=question,
            knowledge_base_id=knowledge_base_id,
            top_k=child_top_k,
        )
    except Exception as exc:
        raise MilvusUnavailableError(f"向量检索失败：{exc}") from exc

    parents = _load_parents(_unique_parent_ids(outcome.nodes))
    score_type = "rrf" if outcome.mode == "hybrid" else "dense"

    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    used_chars = 0

    for node, score in zip(outcome.nodes, outcome.similarities):
        if len(sources) >= settings.rag_top_k:
            break

        # 混合检索返回的是 RRF 融合分（量级约 0.0x），与余弦阈值语义不同。
        if outcome.mode == "hybrid":
            if (
                settings.rag_hybrid_min_score > 0
                and score is not None
                and score < settings.rag_hybrid_min_score
            ):
                continue
        elif score is not None and score < settings.rag_min_score:
            continue

        metadata = node.metadata or {}
        child_text = (node.get_content() or "").strip()
        if not child_text:
            continue

        parent_id = metadata.get("parent_id")
        parent = parents.get(str(parent_id)) if parent_id else None

        if parent_id:
            dedupe_key = str(parent_id)
            text = parent.text if parent else child_text
            page_label = parent.page_label if parent else metadata.get("page_label")
            parent_index = (
                parent.parent_index if parent else metadata.get("parent_index")
            )
            char_limit = settings.rag_parent_max_chars
            if parent is not None and child_text != parent.text.strip():
                matched_text = _truncate(child_text, MATCH_TEXT_MAX_CHARS)
            else:
                matched_text = None
        else:
            # 旧数据或未启用父子切片：沿用扁平逻辑。
            dedupe_key = child_text[:200]
            text = child_text
            page_label = metadata.get("page_label")
            parent_index = None
            char_limit = settings.rag_chunk_max_chars
            matched_text = None

        if dedupe_key in seen:
            continue

        text = text.strip()
        if not text:
            continue

        truncated = _truncate(text, char_limit)
        if used_chars + len(truncated) > settings.rag_context_max_chars:
            continue

        seen.add(dedupe_key)
        used_chars += len(truncated)
        sources.append(
            _to_source(
                node,
                float(score or 0.0),
                len(sources) + 1,
                text=truncated,
                page_label=page_label,
                parent_index=parent_index,
                score_type=score_type,
                matched_text=matched_text,
            )
        )

    logger.debug(
        "检索完成：mode=%s 候选=%s 来源=%s 预算占用=%s/%s",
        outcome.mode,
        len(outcome.nodes),
        len(sources),
        used_chars,
        settings.rag_context_max_chars,
    )
    return sources
