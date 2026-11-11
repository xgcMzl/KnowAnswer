from know_answer.rag.prompt import build_messages


def test_prompt_contains_sources_and_history() -> None:
    sources = [
        {
            "index": 1,
            "file_name": "手册.pdf",
            "page_label": "2",
            "text": "支持向量检索。",
        }
    ]
    history = [
        {"role": "user", "content": "之前的问题"},
        {"role": "assistant", "content": "之前的回答"},
    ]

    messages = build_messages("现在的问题", sources, history)

    assert messages[0]["role"] == "system"
    assert "[1] 手册.pdf（第 2 页）" in messages[0]["content"]
    assert "支持向量检索。" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "之前的问题"}
    assert messages[2] == {"role": "assistant", "content": "之前的回答"}
    assert messages[-1] == {"role": "user", "content": "现在的问题"}


def test_prompt_without_sources() -> None:
    messages = build_messages("你好", [], None)
    assert "本轮未检索到知识库内容" in messages[0]["content"]
