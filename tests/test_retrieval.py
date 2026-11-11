from llama_index.core.schema import TextNode

import know_answer.rag.retrieval as retrieval


def _node(text: str, file_id: int = 5) -> TextNode:
    return TextNode(
        text=text,
        metadata={
            "file_id": file_id,
            "file_name": "手册.pdf",
            "knowledge_base_id": 2,
            "page_label": "3",
            "chunk_index": 1,
        },
    )


def test_retrieve_builds_sources(monkeypatch) -> None:
    monkeypatch.setattr(retrieval, "embed_query", lambda question: [0.1] * 8)
    monkeypatch.setattr(
        retrieval,
        "search",
        lambda embedding, *, knowledge_base_id, top_k: (
            [_node("第一段"), _node("第二段", file_id=6)],
            [0.9, 0.7],
        ),
    )

    sources = retrieval.retrieve("测试问题", 2)

    assert [item["index"] for item in sources] == [1, 2]
    assert sources[0]["file_id"] == 5
    assert sources[0]["file_name"] == "手册.pdf"
    assert sources[0]["page_label"] == "3"
    assert sources[0]["score"] == 0.9
    assert sources[1]["file_id"] == 6


def test_retrieve_deduplicates_same_text(monkeypatch) -> None:
    monkeypatch.setattr(retrieval, "embed_query", lambda question: [0.1] * 8)
    monkeypatch.setattr(
        retrieval,
        "search",
        lambda embedding, *, knowledge_base_id, top_k: (
            [_node("重复内容"), _node("重复内容", file_id=9)],
            [0.9, 0.8],
        ),
    )

    sources = retrieval.retrieve("测试问题", 2)

    assert len(sources) == 1
    assert sources[0]["file_id"] == 5
