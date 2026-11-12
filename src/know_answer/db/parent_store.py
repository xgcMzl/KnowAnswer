"""父块（parent chunk）的 SQLite 读写。

父块正文只存在这里，不写入 Milvus；向量库里的子节点通过 ``parent_id`` 关联。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from know_answer.core.db import SessionLocal
from know_answer.db.models import ParentChunk

if TYPE_CHECKING:
    from know_answer.data_index.doc_loading import ParentChunkDraft


@dataclass(frozen=True)
class ParentRecord:
    """检索侧使用的父块视图。"""

    parent_id: str
    file_id: int
    parent_index: int
    page_label: str | None
    text: str


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def replace_for_file(
    db: Session,
    *,
    file_id: int,
    knowledge_base_id: int,
    drafts: Sequence["ParentChunkDraft"],
) -> None:
    """用本次切分结果整体替换某个文件的父块，保证重复入库幂等。"""
    db.execute(delete(ParentChunk).where(ParentChunk.file_id == file_id))
    if not drafts:
        return

    db.add_all(
        [
            ParentChunk(
                parent_id=draft.parent_id,
                file_id=int(file_id),
                knowledge_base_id=int(knowledge_base_id),
                parent_index=int(draft.parent_index),
                page_label=_optional_str(draft.metadata.get("page_label")),
                text=draft.text,
                char_count=len(draft.text),
            )
            for draft in drafts
        ]
    )
    db.flush()


def fetch_parents(
    parent_ids: Iterable[str],
    *,
    db: Session | None = None,
) -> dict[str, ParentRecord]:
    """一次 ``IN`` 查询取回父块；缺失的 ID 直接忽略。"""
    ids = [str(item) for item in parent_ids if item]
    if not ids:
        return {}

    own_session = db is None
    session = db or SessionLocal()
    try:
        rows = session.scalars(
            select(ParentChunk).where(ParentChunk.parent_id.in_(ids))
        ).all()
        return {
            row.parent_id: ParentRecord(
                parent_id=row.parent_id,
                file_id=row.file_id,
                parent_index=row.parent_index,
                page_label=row.page_label,
                text=row.text,
            )
            for row in rows
        }
    finally:
        if own_session:
            session.close()


def delete_by_file(db: Session, file_id: int) -> None:
    db.execute(delete(ParentChunk).where(ParentChunk.file_id == int(file_id)))


def delete_by_knowledge_base(db: Session, knowledge_base_id: int) -> None:
    db.execute(
        delete(ParentChunk).where(
            ParentChunk.knowledge_base_id == int(knowledge_base_id)
        )
    )
