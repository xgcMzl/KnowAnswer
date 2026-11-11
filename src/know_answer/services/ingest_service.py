"""后台入库任务：单线程串行执行，避免多个上传同时占满 CPU。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select

from know_answer.core.db import session_scope
from know_answer.db.models import FileStatus, KnowledgeFile
from know_answer.rag import vector_store
from know_answer.rag.ingestion import ingest_document

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knowanswer-ingest")


def submit(file_id: int) -> None:
    _executor.submit(process_file, file_id)


def shutdown() -> None:
    _executor.shutdown(wait=False, cancel_futures=True)


def process_file(file_id: int) -> None:
    with session_scope() as db:
        row = db.get(KnowledgeFile, file_id)
        if row is None:
            return
        row.status = FileStatus.PROCESSING
        row.error_message = None
        snapshot = (row.knowledge_base_id, row.storage_path, row.file_name)

    knowledge_base_id, storage_path, file_name = snapshot

    try:
        result = ingest_document(
            storage_path,
            file_id=file_id,
            knowledge_base_id=knowledge_base_id,
            file_name=file_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("文件入库失败：file_id=%s", file_id)
        try:
            vector_store.delete_by_file(file_id)
        except Exception as cleanup_exc:  # noqa: BLE001
            logger.warning("清理 file_id=%s 的残留向量失败：%s", file_id, cleanup_exc)
        with session_scope() as db:
            row = db.get(KnowledgeFile, file_id)
            if row is not None:
                row.status = FileStatus.FAILED
                row.error_message = str(exc)[:1000]
        return

    with session_scope() as db:
        row = db.get(KnowledgeFile, file_id)
        if row is not None:
            row.status = FileStatus.READY
            row.chunk_count = result.node_count
            row.error_message = None


def recover_stale_files() -> None:
    """启动时把上次进程中断留下的 pending / processing 记录标记为 failed。"""
    with session_scope() as db:
        rows = db.scalars(
            select(KnowledgeFile).where(
                KnowledgeFile.status.in_(
                    [FileStatus.PENDING, FileStatus.PROCESSING]
                )
            )
        ).all()
        for row in rows:
            row.status = FileStatus.FAILED
            row.error_message = "服务重启导致入库中断，请重新上传文件"
        if rows:
            logger.warning("已将 %s 个中断的入库任务标记为 failed", len(rows))
