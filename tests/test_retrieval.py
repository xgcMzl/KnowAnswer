from llama_index.core.schema import TextNode

import know_answer.rag.retrieval as retrieval
from know_answer.core.config import get_settings
from know_answer.db.parent_store import ParentRecord
from know_answer.rag.vector_store import SearchOutcome


def _node(
    text: str,
    *,
    file_id: int = 5,
    parent_id: str | None = None,
    parent_index: int | None = None,
    chunk_index: int = 1,
) -> TextNode:
    metadata = {
        "file_id": file_id,
        "file_name": "手册.pdf",
        "knowledge_base_id": 2,
        "page_label": "3",
        "chunk_index": chunk_index,
    }
    if parent_id is not None:
        metadata["parent_id"] = parent_id
        metadata["parent_index"] = parent_index
    return TextNode(text=text, metadata=metadata)


def _parent(parent_id: str, text: str, *, index: int = 0) -> ParentRecord:
    return ParentRecord(
        parent_id=parent_id,
        file_id=5,
        parent_index=index,
        page_label="3",
        text=text,
    )


def _install(monkeypatch, outcome: SearchOutcome, parents=None) -> None:
    monkeypatch.setattr(retrieval, "embed_query", lambda question: [0.1] * 8)
    monkeypatch.setattr(retrieval, "search", lambda *args, **kwargs: outcome)
    monkeypatch.setattr(retrieval, "fetch_parents", lambda ids, **kwargs: parents or {})


def test_retrieve_dedupes_parent_and_uses_parent_text(monkeypatch) -> None:
    outcome = SearchOutcome(
        nodes=[
            _node("子块A1", parent_id="5:0", parent_index=0),
            _node("子块A2", parent_id="5:0", parent_index=0, chunk_index=2),
            _node("子块B1", parent_id="5:1", parent_index=1, chunk_index=3),
        ],
        similarities=[0.031, 0.028, 0.020],
        mode="hybrid",
    )
    _install(
        monkeypatch,
        outcome,
        {
            "5:0": _parent("5:0", "父块零", index=0),
            "5:1": _parent("5:1", "父块一", index=1),
        },
    )

    sources = retrieval.retrieve("测试问题", 2)

    assert [item["index"] for item in sources] == [1, 2]
    assert sources[0]["text"] == "父块零"
    assert sources[0]["parent_id"] == "5:0"
    assert sources[0]["parent_index"] == 0
    assert sources[0]["matched_text"] == "子块A1"
    assert sources[0]["score_type"] == "rrf"
    assert sources[0]["file_id"] == 5
    assert sources[1]["text"] == "父块一"


def test_retrieve_falls_back_to_child_when_parent_missing(monkeypatch) -> None:
    outcome = SearchOutcome(
        nodes=[_node("子块内容", parent_id="5:0", parent_index=0)],
        similarities=[0.02],
        mode="hybrid",
    )
    _install(monkeypatch, outcome, parents={})

    sources = retrieval.retrieve("测试问题", 2)

    assert len(sources) == 1
    assert sources[0]["text"] == "子块内容"
    assert sources[0]["matched_text"] is None


def test_hybrid_ignores_dense_min_score_but_uses_hybrid_threshold(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_min_score", 0.9)
    monkeypatch.setattr(settings, "rag_hybrid_min_score", 0.0)
    outcome = SearchOutcome(
        nodes=[_node("子块", parent_id="5:0", parent_index=0)],
        similarities=[0.02],
        mode="hybrid",
    )
    _install(monkeypatch, outcome, {"5:0": _parent("5:0", "父块")})

    assert len(retrieval.retrieve("问题", 2)) == 1

    monkeypatch.setattr(settings, "rag_hybrid_min_score", 0.05)
    assert retrieval.retrieve("问题", 2) == []


def test_dense_still_applies_min_score(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_min_score", 0.5)
    outcome = SearchOutcome(
        nodes=[_node("低分片段")],
        similarities=[0.1],
        mode="dense",
    )
    _install(monkeypatch, outcome)

    assert retrieval.retrieve("问题", 2) == []


def test_legacy_nodes_without_parent_id_are_deduplicated(monkeypatch) -> None:
    outcome = SearchOutcome(
        nodes=[_node("重复内容"), _node("重复内容", file_id=9)],
        similarities=[0.9, 0.8],
        mode="dense",
    )
    _install(monkeypatch, outcome)

    sources = retrieval.retrieve("问题", 2)

    assert len(sources) == 1
    assert sources[0]["file_id"] == 5
    assert sources[0]["score_type"] == "dense"
    assert sources[0]["parent_id"] is None


def test_sources_are_capped_by_rag_top_k(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "rag_top_k", 1)
    outcome = SearchOutcome(
        nodes=[
            _node("一", parent_id="5:0", parent_index=0),
            _node("二", parent_id="5:1", parent_index=1),
        ],
        similarities=[0.03, 0.02],
        mode="hybrid",
    )
    _install(
        monkeypatch,
        outcome,
        {"5:0": _parent("5:0", "父块零"), "5:1": _parent("5:1", "父块一", index=1)},
    )

    assert len(retrieval.retrieve("问题", 2)) == 1
