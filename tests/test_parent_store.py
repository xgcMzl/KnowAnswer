from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from know_answer.core.db import Base
from know_answer.data_index.doc_loading import ParentChunkDraft
from know_answer.db import parent_store
from know_answer.db.models import FileStatus, KnowledgeBase, KnowledgeFile, ParentChunk


@pytest.fixture()
def db(tmp_path: Path) -> Session:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_pragma(connection, _record) -> None:  # type: ignore[no-untyped-def]
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _seed_file(db: Session) -> KnowledgeFile:
    kb = KnowledgeBase(name="测试库")
    db.add(kb)
    db.flush()

    row = KnowledgeFile(
        knowledge_base_id=kb.id,
        file_name="manual.pdf",
        file_ext="pdf",
        file_size=10,
        storage_path="data/uploads/x.pdf",
        status=FileStatus.READY,
    )
    db.add(row)
    db.commit()
    return row


def _draft(index: int, text: str) -> ParentChunkDraft:
    return ParentChunkDraft(
        parent_id=f"7:{index}",
        parent_index=index,
        text=text,
        metadata={"page_label": str(index + 1)},
    )


def test_replace_for_file_is_idempotent(db: Session) -> None:
    file = _seed_file(db)
    drafts = (_draft(0, "父块零"), _draft(1, "父块一"))

    parent_store.replace_for_file(
        db, file_id=file.id, knowledge_base_id=file.knowledge_base_id, drafts=drafts
    )
    parent_store.replace_for_file(
        db, file_id=file.id, knowledge_base_id=file.knowledge_base_id, drafts=drafts
    )
    db.commit()

    rows = db.scalars(select(ParentChunk).where(ParentChunk.file_id == file.id)).all()
    assert len(rows) == 2
    assert {row.parent_id for row in rows} == {"7:0", "7:1"}


def test_fetch_parents_in_batch_ignores_missing_ids(db: Session) -> None:
    file = _seed_file(db)
    parent_store.replace_for_file(
        db,
        file_id=file.id,
        knowledge_base_id=file.knowledge_base_id,
        drafts=(_draft(0, "父块零"), _draft(1, "父块一")),
    )
    db.commit()

    parents = parent_store.fetch_parents(["7:0", "7:1", "7:404"], db=db)

    assert set(parents) == {"7:0", "7:1"}
    assert parents["7:0"].text == "父块零"
    assert parents["7:0"].page_label == "1"


def test_fetch_parents_with_empty_ids_returns_empty(db: Session) -> None:
    assert parent_store.fetch_parents([], db=db) == {}


def test_parent_chunks_are_deleted_with_file(db: Session) -> None:
    file = _seed_file(db)
    parent_store.replace_for_file(
        db,
        file_id=file.id,
        knowledge_base_id=file.knowledge_base_id,
        drafts=(_draft(0, "父块零"),),
    )
    db.commit()

    db.delete(file)
    db.commit()

    assert db.scalars(select(ParentChunk)).all() == []
