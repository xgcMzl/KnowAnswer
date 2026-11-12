"""Milvus 向量库适配。

collection 在建库时声明三个业务标量字段，用于按知识库/文件过滤与删除；
开启混合检索后，同一 collection 还会带 ``sparse_embedding`` 字段与 BM25 内置函数。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Literal

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from know_answer.core.config import get_settings
from know_answer.core.errors import MilvusUnavailableError

logger = logging.getLogger(__name__)

SCALAR_FIELD_NAMES = ["knowledge_base_id", "file_id", "file_name"]
SPARSE_FIELD_NAME = "sparse_embedding"
TEXT_FIELD_NAME = "text"

SearchMode = Literal["hybrid", "dense", "sparse"]
RequestedSearchMode = Literal["auto", "hybrid", "dense", "sparse"]

_lock = threading.Lock()
_vector_store: Any | None = None


@dataclass(frozen=True)
class SearchOutcome:
    """一次检索的结果；``mode`` 为实际生效的模式（可能因降级而变化）。"""

    nodes: list[BaseNode]
    similarities: list[float]
    mode: SearchMode


def _build_vector_store() -> Any:
    from llama_index.vector_stores.milvus import MilvusVectorStore
    from pymilvus import DataType

    settings = get_settings()
    kwargs: dict[str, Any] = {
        "uri": settings.milvus_uri,
        "token": settings.milvus_token,
        "collection_name": settings.milvus_collection,
        "dim": settings.embedding_dim,
        "overwrite": False,
        "enable_dense": True,
        "similarity_metric": "COSINE",
        "scalar_field_names": SCALAR_FIELD_NAMES,
        "scalar_field_types": [DataType.INT64, DataType.INT64, DataType.VARCHAR],
    }

    if settings.hybrid_retrieval_enabled:
        # 必须显式传入 BM25BuiltInFunction：复用旧 collection 时适配器会回退到
        # BGEM3SparseEmbeddingFunction（依赖 FlagEmbedding + BGE-M3，不可用）。
        from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction

        kwargs.update(
            enable_sparse=True,
            sparse_embedding_function=BM25BuiltInFunction(
                input_field_names=TEXT_FIELD_NAME,
                output_field_names=SPARSE_FIELD_NAME,
                analyzer_params={"type": settings.rag_sparse_analyzer},
            ),
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": settings.rag_rrf_k},
        )
    else:
        kwargs["enable_sparse"] = False

    store = MilvusVectorStore(**kwargs)
    _validate_collection_schema(store, settings)
    return store


def get_vector_store() -> Any:
    global _vector_store
    if _vector_store is None:
        with _lock:
            if _vector_store is None:
                try:
                    _vector_store = _build_vector_store()
                except MilvusUnavailableError:
                    raise
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


# --------------------------------------------------------------------------- #
# collection 能力探测
# --------------------------------------------------------------------------- #
def _as_name_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _function_type_name(value: Any) -> str:
    """``describe_collection`` 返回的 type 可能是 ``FunctionType`` 枚举或字符串。"""
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.upper()
    return str(value).upper()


def _has_bm25_function(functions: list[dict[str, Any]] | None) -> bool:
    for func in functions or []:
        if "BM25" not in _function_type_name(func.get("type")):
            continue
        inputs = _as_name_list(func.get("input_field_names"))
        outputs = _as_name_list(func.get("output_field_names"))
        if TEXT_FIELD_NAME in inputs and SPARSE_FIELD_NAME in outputs:
            return True
    return False


def collection_capabilities(
    store: Any | None = None,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """返回 collection 的字段与 BM25 能力，供校验、健康检查与诊断脚本使用。"""
    settings = get_settings()
    store = store or get_vector_store()
    name = collection_name or settings.milvus_collection
    info = store.client.describe_collection(name)

    fields = {field["name"]: field for field in info.get("fields") or []}
    functions = info.get("functions") or []
    embedding_params = (fields.get("embedding") or {}).get("params") or {}
    text_params = (fields.get(TEXT_FIELD_NAME) or {}).get("params") or {}
    raw_dim = embedding_params.get("dim")

    return {
        "collection": name,
        "fields": sorted(fields),
        "dim": int(raw_dim) if raw_dim is not None else None,
        "has_sparse_field": SPARSE_FIELD_NAME in fields,
        "has_bm25_function": _has_bm25_function(functions),
        "text_analyzer_enabled": bool(text_params.get("enable_analyzer")),
        "functions": functions,
    }


def _validate_collection_schema(store: Any, settings: Any) -> None:
    capabilities = collection_capabilities(store, settings.milvus_collection)
    missing = [name for name in SCALAR_FIELD_NAMES if name not in capabilities["fields"]]
    if "embedding" not in capabilities["fields"]:
        missing.append("embedding")
    if missing:
        raise MilvusUnavailableError(
            f"collection '{capabilities['collection']}' 缺少字段：{', '.join(missing)}。"
            "请检查 MILVUS_COLLECTION 是否指向本项目的 collection。"
        )

    if capabilities["dim"] is not None and int(capabilities["dim"]) != settings.embedding_dim:
        raise MilvusUnavailableError(
            f"collection '{capabilities['collection']}' 的向量维度"
            f"（{capabilities['dim']}）与 EMBEDDING_DIM={settings.embedding_dim} 不一致。"
        )

    if settings.hybrid_retrieval_enabled and not (
        capabilities["has_sparse_field"] and capabilities["has_bm25_function"]
    ):
        raise MilvusUnavailableError(
            f"collection '{capabilities['collection']}' 不支持混合检索："
            "缺少 sparse_embedding 字段或 BM25 内置函数（该 schema 只在新建 collection 时创建）。"
            "请设置 MILVUS_COLLECTION=knowanswer_chunks_v2 使用新 collection，"
            "或设置 HYBRID_RETRIEVAL_ENABLED=false 仅使用稠密向量检索。"
        )


# --------------------------------------------------------------------------- #
# 检索与删除
# --------------------------------------------------------------------------- #
def _build_filters(knowledge_base_id: int | None) -> MetadataFilters | None:
    if knowledge_base_id is None:
        return None
    return MetadataFilters(
        filters=[
            MetadataFilter(key="knowledge_base_id", value=int(knowledge_base_id))
        ]
    )


def _run_search(
    query_embedding: list[float] | None,
    *,
    query_str: str | None,
    knowledge_base_id: int | None,
    top_k: int,
    mode: SearchMode,
) -> SearchOutcome:
    query_mode = {
        "hybrid": VectorStoreQueryMode.HYBRID,
        "sparse": VectorStoreQueryMode.SPARSE,
        "dense": VectorStoreQueryMode.DEFAULT,
    }[mode]

    result = get_vector_store().query(
        VectorStoreQuery(
            mode=query_mode,
            query_embedding=query_embedding,
            query_str=query_str,
            similarity_top_k=top_k,
            filters=_build_filters(knowledge_base_id),
        )
    )

    nodes = list(result.nodes or [])
    similarities = list(result.similarities or [])
    logger.debug(
        "检索完成：mode=%s top_k=%s 命中=%s 首条分数=%s",
        mode,
        top_k,
        len(nodes),
        similarities[0] if similarities else None,
    )
    return SearchOutcome(nodes=nodes, similarities=similarities, mode=mode)


def search(
    query_embedding: list[float] | None = None,
    *,
    query_str: str | None = None,
    knowledge_base_id: int | None,
    top_k: int,
    mode: RequestedSearchMode = "auto",
) -> SearchOutcome:
    """检索入口。

    ``mode="auto"`` 时：混合检索开启且提供了原文 query 就走 hybrid，否则走 dense。
    hybrid 运行期失败（例如服务端不支持）会自动降级为 dense，并在日志中告警。
    """
    settings = get_settings()
    resolved: SearchMode
    if mode == "auto":
        resolved = "hybrid" if settings.hybrid_retrieval_enabled and query_str else "dense"
    else:
        resolved = mode  # type: ignore[assignment]

    if resolved == "sparse" and not settings.hybrid_retrieval_enabled:
        raise MilvusUnavailableError(
            "当前未开启混合检索，无法执行稀疏检索。请设置 HYBRID_RETRIEVAL_ENABLED=true。"
        )

    if resolved == "hybrid":
        try:
            return _run_search(
                query_embedding,
                query_str=query_str,
                knowledge_base_id=knowledge_base_id,
                top_k=top_k,
                mode="hybrid",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("混合检索失败，降级为稠密向量检索：%s", exc)

    return _run_search(
        query_embedding,
        query_str=query_str,
        knowledge_base_id=knowledge_base_id,
        top_k=top_k,
        mode="dense" if resolved == "hybrid" else resolved,
    )


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
        capabilities = collection_capabilities(store, settings.milvus_collection)
        base = {
            "collection": capabilities["collection"],
            "hybrid_enabled": settings.hybrid_retrieval_enabled,
            "sparse_field": capabilities["has_sparse_field"],
            "bm25_function": capabilities["has_bm25_function"],
            "text_analyzer": capabilities["text_analyzer_enabled"],
        }

        missing = [name for name in SCALAR_FIELD_NAMES if name not in capabilities["fields"]]
        if "embedding" not in capabilities["fields"]:
            missing.append("embedding")
        if missing:
            return {
                **base,
                "status": "degraded",
                "detail": f"collection 缺少字段：{', '.join(missing)}",
            }
        if (
            capabilities["dim"] is not None
            and int(capabilities["dim"]) != settings.embedding_dim
        ):
            return {
                **base,
                "status": "degraded",
                "detail": (
                    f"collection 维度 {capabilities['dim']} 与 "
                    f"EMBEDDING_DIM={settings.embedding_dim} 不一致"
                ),
            }
        if settings.hybrid_retrieval_enabled and not (
            capabilities["has_sparse_field"] and capabilities["has_bm25_function"]
        ):
            return {
                **base,
                "status": "degraded",
                "detail": (
                    "混合检索已开启，但 collection 缺少 sparse_embedding 或 BM25 函数；"
                    "请改用 knowanswer_chunks_v2 或关闭 HYBRID_RETRIEVAL_ENABLED"
                ),
            }
        return {**base, "status": "ok"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Milvus 健康检查失败：%s", exc)
        return {
            "status": "unavailable",
            "detail": str(exc)[:200],
            "hybrid_enabled": settings.hybrid_retrieval_enabled,
        }
