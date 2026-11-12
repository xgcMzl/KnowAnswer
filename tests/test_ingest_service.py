from contextlib import contextmanager
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from know_answer.core.db import Base
from know_answer.data_index.doc_loading import IngestResult, ParentChunkDraft
from know_answer.db.models import FileStatus, KnowledgeBase, KnowledgeFile, ParentChunk
from know_answer.services import ingest_service


@pytest.fixture()
def session_factory(tmp_path: Path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'ingest.db'}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_pragma(connection, _record) -> None:  # type: ignore[no-untyped-def]
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    engine.dispose()


@pytest.fixture()
def file_id(session_factory) -> int:
    with session_factory() as db:
        kb = KnowledgeBase(name="测试库")
        db.add(kb)
        db.flush()
        row = KnowledgeFile(
            knowledge_base_id=kb.id,
            file_name="a.txt",
            file_ext="txt",
            file_size=10,
            storage_path="data/uploads/a.txt",
            status=FileStatus.PENDING,
        )
        db.add(row)
        db.commit()
        return row.id


@pytest.fixture()
def patched_scope(monkeypatch, session_factory):
    @contextmanager
    def fake_scope():
        db: Session = session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    monkeypatch.setattr(ingest_service, "session_scope", fake_scope)
    return fake_scope


def _result() -> IngestResult:
    return IngestResult(
        file_name="a.txt",
        file_type="txt",
        node_count=2,
        node_ids=("n1", "n2"),
        collection_name="test",
        parent_count=1,
        parent_chunks=(
            ParentChunkDraft(
                parent_id="1:0",
                parent_index=0,
                text="父块正文",
                metadata={"page_label": "1"},
            ),
        ),
    )


def test_process_file_persists_parents_and_marks_ready(
    monkeypatch, session_factory, file_id, patched_scope
) -> None:
    monkeypatch.setattr(
        ingest_service, "ingest_document", lambda *args, **kwargs: _result()
    )

    ingest_service.process_file(file_id)

    with session_factory() as db:
        row = db.get(KnowledgeFile, file_id)
        assert row.status == FileStatus.READY
        assert row.chunk_count == 2

        parents = db.scalars(
            select(ParentChunk).where(ParentChunk.file_id == file_id)
        ).all()
        assert [item.text for item in parents] == ["父块正文"]


def test_process_file_cleans_up_on_failure(
    monkeypatch, session_factory, file_id, patched_scope
) -> None:
    deleted: list[int] = []

    def fake_ingest(*args, **kwargs):
        raise RuntimeError("milvus 写入失败")

    monkeypatch.setattr(ingest_service, "ingest_document", fake_ingest)
    monkeypatch.setattr(
        ingest_service.vector_store, "delete_by_file", lambda fid: deleted.append(fid)
    )

    ingest_service.process_file(file_id)

    assert deleted == [file_id]
    with session_factory() as db:
        row = db.get(KnowledgeFile, file_id)
        assert row.status == FileStatus.FAILED
        assert "milvus 写入失败" in (row.error_message or "")
        assert db.scalars(select(ParentChunk)).all() == []
