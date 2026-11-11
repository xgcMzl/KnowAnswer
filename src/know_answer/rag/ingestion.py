"""把业务文件交给 DocumentIngestor 入库。"""

from __future__ import annotations

import threading

from know_answer.core.config import get_settings
from know_answer.data_index.doc_loading import (
    DocumentIngestor,
    EmbeddingConfig,
    IngestResult,
    MilvusConfig,
    SplitterConfig,
)
from know_answer.rag.embedding import get_embed_model
from know_answer.rag.vector_store import get_vector_store

_lock = threading.Lock()
_ingestor: DocumentIngestor | None = None


def get_ingestor() -> DocumentIngestor:
    global _ingestor
    if _ingestor is None:
        with _lock:
            if _ingestor is None:
                settings = get_settings()
                config = EmbeddingConfig(
                    provider=settings.embedding_provider.lower(),  # type: ignore[arg-type]
                    model_name=settings.embedding_model_name,
                    api_key=settings.embedding_api_key or None,
                    api_base=settings.embedding_api_base or None,
                    cache_folder=settings.embedding_cache_folder or None,
                    device=settings.embedding_device or None,
                )
                _ingestor = DocumentIngestor(
                    milvus_config=MilvusConfig(
                        uri=settings.milvus_uri, token=settings.milvus_token
                    ),
                    embedding_config=config,
                    splitter_config=SplitterConfig(
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                    ),
                    embed_model=get_embed_model(),
                    vector_store=get_vector_store(),
                    collection_name=settings.milvus_collection,
                    scalar_field_names=[
                        "knowledge_base_id",
                        "file_id",
                        "file_name",
                    ],
                )
    return _ingestor


def reset_ingestor() -> None:
    """测试辅助：清空单例。"""
    global _ingestor
    with _lock:
        _ingestor = None


def ingest_document(
    storage_path: str,
    *,
    file_id: int,
    knowledge_base_id: int,
    file_name: str,
) -> IngestResult:
    return get_ingestor().ingest(
        storage_path,
        extra_metadata={
            "knowledge_base_id": int(knowledge_base_id),
            "file_id": int(file_id),
            "file_name": str(file_name),
        },
    )
