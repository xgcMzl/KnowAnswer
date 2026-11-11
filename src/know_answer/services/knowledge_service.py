"""知识库与文件的业务编排。"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from know_answer.core.config import get_settings
from know_answer.core.errors import (
    FileBusyError,
    FileTooLargeError,
    InvalidArgumentError,
    InvalidDocumentError,
    InvalidFileTypeError,
    NotFoundError,
)
from know_answer.db.models import (
    Conversation,
    FileStatus,
    KnowledgeBase,
    KnowledgeFile,
)
from know_answer.rag import vector_store
from know_answer.schemas.knowledge import KnowledgeBaseOut

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


# --------------------------------------------------------------------------- #
# 知识库
# --------------------------------------------------------------------------- #
def _get_kb_or_404(db: Session, kb_id: int) -> KnowledgeBase:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise NotFoundError(f"知识库不存在：{kb_id}")
    return kb


def _file_count(db: Session, kb_id: int) -> int:
    return int(
        db.scalar(
            select(func.count(KnowledgeFile.id)).where(
                KnowledgeFile.knowledge_base_id == kb_id
            )
        )
        or 0
    )


def _kb_out(db: Session, kb: KnowledgeBase) -> KnowledgeBaseOut:
    return KnowledgeBaseOut(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        file_count=_file_count(db, kb.id),
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


def list_knowledge_bases(db: Session) -> list[KnowledgeBaseOut]:
    count_subquery = (
        select(
            KnowledgeFile.knowledge_base_id,
            func.count(KnowledgeFile.id).label("file_count"),
        )
        .group_by(KnowledgeFile.knowledge_base_id)
        .subquery()
    )
    rows = db.execute(
        select(KnowledgeBase, func.coalesce(count_subquery.c.file_count, 0))
        .outerjoin(count_subquery, count_subquery.c.knowledge_base_id == KnowledgeBase.id)
        .order_by(KnowledgeBase.id)
    ).all()
    return [
        KnowledgeBaseOut(
            id=kb.id,
            name=kb.name,
            description=kb.description,
            file_count=int(count),
            created_at=kb.created_at,
            updated_at=kb.updated_at,
        )
        for kb, count in rows
    ]


def get_knowledge_base(db: Session, kb_id: int) -> KnowledgeBaseOut:
    return _kb_out(db, _get_kb_or_404(db, kb_id))


def create_knowledge_base(db: Session, name: str, description: str = "") -> KnowledgeBaseOut:
    name = name.strip()
    if not name:
        raise InvalidArgumentError("知识库名称不能为空")
    kb = KnowledgeBase(name=name, description=description or "")
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return _kb_out(db, kb)


def update_knowledge_base(
    db: Session,
    kb_id: int,
    *,
    name: str | None = None,
    description: str | None = None,
) -> KnowledgeBaseOut:
    kb = _get_kb_or_404(db, kb_id)
    if name is not None:
        name = name.strip()
        if not name:
            raise InvalidArgumentError("知识库名称不能为空")
        kb.name = name
    if description is not None:
        kb.description = description
    db.commit()
    db.refresh(kb)
    return _kb_out(db, kb)


def delete_knowledge_base(db: Session, kb_id: int) -> None:
    kb = _get_kb_or_404(db, kb_id)
    files = db.scalars(
        select(KnowledgeFile).where(KnowledgeFile.knowledge_base_id == kb_id)
    ).all()
    if any(item.status == FileStatus.PROCESSING for item in files):
        raise FileBusyError("知识库中有文件正在处理，请稍后再删除")

    # 向量清理是尽力而为；失败不应阻止元数据删除。
    try:
        vector_store.delete_by_knowledge_base(kb_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("删除知识库 %s 的向量失败：%s", kb_id, exc)

    storage_paths = [item.storage_path for item in files]

    # 保留会话历史，但解除知识库绑定。
    db.execute(
        update(Conversation)
        .where(Conversation.knowledge_base_id == kb_id)
        .values(knowledge_base_id=None)
    )
    db.delete(kb)
    db.commit()

    for raw_path in storage_paths:
        _unlink(raw_path)


# --------------------------------------------------------------------------- #
# 文件
# --------------------------------------------------------------------------- #
def _validate_filename(filename: str | None) -> tuple[str, str]:
    if not filename:
        raise InvalidFileTypeError("缺少文件名")
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = "、".join(sorted(ALLOWED_EXTENSIONS))
        raise InvalidFileTypeError(
            f"不支持的文件类型 '{suffix or '无扩展名'}'，仅支持：{allowed}"
        )
    return safe_name, suffix


async def save_upload(db: Session, kb_id: int, upload: UploadFile) -> KnowledgeFile:
    _get_kb_or_404(db, kb_id)
    safe_name, suffix = _validate_filename(upload.filename)

    settings = get_settings()
    settings.ensure_dirs()
    storage_path = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    max_bytes = settings.max_upload_bytes
    written = 0

    try:
        with storage_path.open("wb") as target:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise FileTooLargeError(
                        f"文件超过 {settings.max_upload_mb}MB 限制"
                    )
                target.write(chunk)
    except FileTooLargeError:
        _unlink(str(storage_path))
        raise
    except Exception:
        _unlink(str(storage_path))
        raise

    if written == 0:
        _unlink(str(storage_path))
        raise InvalidDocumentError("上传文件为空")

    row = KnowledgeFile(
        knowledge_base_id=kb_id,
        file_name=safe_name,
        file_ext=suffix.lstrip("."),
        file_size=written,
        storage_path=str(storage_path),
        status=FileStatus.PENDING,
        chunk_count=0,
    )
    try:
        db.add(row)
        db.commit()
        db.refresh(row)
    except Exception:
        db.rollback()
        _unlink(str(storage_path))
        raise
    return row


def list_files(db: Session, kb_id: int) -> list[KnowledgeFile]:
    _get_kb_or_404(db, kb_id)
    return list(
        db.scalars(
            select(KnowledgeFile)
            .where(KnowledgeFile.knowledge_base_id == kb_id)
            .order_by(KnowledgeFile.id)
        ).all()
    )


def get_file_or_404(db: Session, file_id: int) -> KnowledgeFile:
    row = db.get(KnowledgeFile, file_id)
    if row is None:
        raise NotFoundError(f"文件不存在：{file_id}")
    return row


def delete_file(db: Session, file_id: int) -> None:
    row = get_file_or_404(db, file_id)
    if row.status == FileStatus.PROCESSING:
        raise FileBusyError("文件正在处理中，请稍后再删除")

    try:
        vector_store.delete_by_file(file_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("删除文件 %s 的向量失败：%s", file_id, exc)

    storage_path = row.storage_path
    db.delete(row)
    db.commit()
    _unlink(storage_path)


def _unlink(raw_path: str) -> None:
    try:
        path = Path(raw_path)
        if path.exists():
            path.unlink()
    except OSError as exc:
        logger.warning("删除磁盘文件失败：%s（%s）", raw_path, exc)
