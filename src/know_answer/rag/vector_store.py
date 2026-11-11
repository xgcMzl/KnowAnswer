"""Milvus 向量库适配。

collection 在建库时声明三个业务标量字段，用于按知识库/文件过滤与删除。
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from know_answer.core.config import get_settings
from know_answer.core.errors import MilvusUnavailableError

logger = logging.getLogger(__name__)

SCALAR_FIELD_NAMES = ["knowledge_base_id", "file_id", "file_name"]

_lock = threading.Lock()
_vector_store: Any | None = None


def _build_vector_store() -> Any:
    from llama_index.vector_stores.milvus import MilvusVectorStore
    from pymilvus import DataType

    settings = get_settings()
    return MilvusVectorStore(
        uri=settings.milvus_uri,
        token=settings.milvus_token,
        collection_name=settings.milvus_collection,
        dim=settings.embedding_dim,
        overwrite=False,
        enable_dense=True,
        enable_sparse=False,
        similarity_metric="COSINE",
        scalar_field_names=SCALAR_FIELD_NAMES,
        scalar_field_types=[DataType.INT64, DataType.INT64, DataType.VARCHAR],
    )


def get_vector_store() -> Any:
    global _vector_store
    if _vector_store is None:
        with _lock:
            if _vector_store is None:
                try:
                    _vector_store = _build_vector_store()
                except Exception as exc:
                    raise MilvusUnavailableError(
                        f"连接 Milvus 或创建 collection 失败：{exc}"
                    ) from exc
    return _vector_store


def reset_vector_store() -> None:
    """测试辅助：清空单例。"""
    global _vector_store
    with _lock:
        _vector_store = None


def search(
    query_embedding: list[float],
    *,
    knowledge_base_id: int | None,
    top_k: int,
) -> tuple[list[BaseNode], list[float]]:
    filters = None
    if knowledge_base_id is not None:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="knowledge_base_id", value=int(knowledge_base_id)
                )
            ]
        )

    result = get_vector_store().query(
        VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
            filters=filters,
        )
    )
    return list(result.nodes or []), list(result.similarities or [])


def delete_by_file(file_id: int) -> None:
    get_vector_store().delete_nodes(
        filters=MetadataFilters(
            filters=[MetadataFilter(key="file_id", value=int(file_id))]
        )
    )


def delete_by_knowledge_base(knowledge_base_id: int) -> None:
    get_vector_store().delete_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="knowledge_base_id", value=int(knowledge_base_id)
                )
            ]
        )
    )


def health() -> dict[str, Any]:
    """返回 Milvus 与 collection 的状态；不抛异常。"""
    settings = get_settings()
    try:
        store = get_vector_store()
        fields = store.client.describe_collection(settings.milvus_collection)["fields"]
        by_name = {field["name"]: field for field in fields}
        missing = [name for name in SCALAR_FIELD_NAMES if name not in by_name]
        dim = by_name.get("embedding", {}).get("params", {}).get("dim")
        if missing:
            return {
                "status": "degraded",
                "detail": f"collection 缺少标量字段：{', '.join(missing)}",
            }
        if dim is not None and int(dim) != settings.embedding_dim:
            return {
                "status": "degraded",
                "detail": f"collection 维度 {dim} 与 EMBEDDING_DIM={settings.embedding_dim} 不一致",
            }
        return {"status": "ok"}
    except Exception as exc:
        logger.warning("Milvus 健康检查失败：%s", exc)
        return {"status": "unavailable", "detail": str(exc)[:200]}
