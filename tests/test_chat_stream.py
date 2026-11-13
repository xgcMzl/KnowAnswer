"""流式问答编排（意图分流）测试：不启动 FastAPI、不连数据库。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import know_answer.app.routes.chat as chat_module
from know_answer.rag.router import IntentDecision
from know_answer.schemas.chat import ChatStreamRequest
from know_answer.services.chat_service import ChatContext


class FakeRequest:
    """模拟 Starlette Request 的断开检测。"""

    def __init__(self, disconnect_after: int | None = None) -> None:
        self.calls = 0
        self.disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return self.disconnect_after is not None and self.calls > self.disconnect_after


class Recorder:
    def __init__(self) -> None:
        self.retrieve_calls: list[tuple[str, int | None]] = []
        self.saved: list[tuple[int, str, Any]] = []
        self.prompt_system: str | None = None


def _install(
    monkeypatch,
    recorder: Recorder,
    *,
    decision: IntentDecision,
    sources=None,
    deltas=("回答",),
    stream_error: Exception | None = None,
) -> None:
    def fake_prepare(req: ChatStreamRequest) -> ChatContext:
        return ChatContext(
            conversation_id=7,
            knowledge_base_id=3,
            question=req.question.strip(),
            history=[{"role": "user", "content": "上一轮"}],
        )

    async def fake_classify(question, history, *, knowledge_base_id):
        return decision

    def fake_retrieve(question: str, knowledge_base_id: int | None):
        recorder.retrieve_calls.append((question, knowledge_base_id))
        return sources if sources is not None else []

    async def fake_stream(messages):
        recorder.prompt_system = messages[0]["content"]
        if stream_error is not None:
            raise stream_error
        for delta in deltas:
            yield delta

    def fake_save(conversation_id: int, content: str, saved_sources):
        recorder.saved.append((conversation_id, content, saved_sources))
        return 101

    monkeypatch.setattr(chat_module.chat_service, "prepare_stream", fake_prepare)
    monkeypatch.setattr(chat_module.chat_service, "save_assistant", fake_save)
    monkeypatch.setattr(chat_module, "classify_intent", fake_classify)
    monkeypatch.setattr(chat_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_module, "astream_answer", fake_stream)


def _run(payload: ChatStreamRequest, request: FakeRequest) -> list[Any]:
    async def collect():
        response = await chat_module.chat_stream(payload, request)
        return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(collect())
    events: list[Any] = []
    for chunk in chunks:
        if not isinstance(chunk, str):
            continue
        raw = chunk.strip()
        if raw == "data: [DONE]":
            events.append("[DONE]")
        elif raw.startswith("data: "):
            events.append(json.loads(raw[len("data: ") :]))
    return events


def _payload(question: str = "你好") -> ChatStreamRequest:
    return ChatStreamRequest(conversation_id=7, question=question)


def test_rag_emits_intent_sources_and_saves_sources(monkeypatch) -> None:
    recorder = Recorder()
    sources = [{"index": 1, "text": "父块", "file_name": "手册.pdf"}]
    _install(
        monkeypatch,
        recorder,
        decision=IntentDecision(intent="rag", query="改写后的检索问句"),
        sources=sources,
    )

    events = _run(_payload("那第二条呢？"), FakeRequest())

    assert events[0] == {"type": "intent", "intent": "rag"}
    assert events[1]["type"] == "sources"
    assert events[1]["sources"] == sources
    assert events[-1] == "[DONE]"
    # 检索使用改写后的 query
    assert recorder.retrieve_calls == [("改写后的检索问句", 3)]
    assert recorder.saved == [(7, "回答", sources)]
    assert "【知识库上下文】" in (recorder.prompt_system or "")


def test_rag_without_rewritten_query_uses_original(monkeypatch) -> None:
    recorder = Recorder()
    _install(monkeypatch, recorder, decision=IntentDecision(intent="rag", query=None))

    _run(_payload("文档讲了什么"), FakeRequest())

    assert recorder.retrieve_calls == [("文档讲了什么", 3)]


def test_general_skips_retrieval_and_saves_none(monkeypatch) -> None:
    recorder = Recorder()
    _install(monkeypatch, recorder, decision=IntentDecision(intent="general"))

    def forbidden(*args, **kwargs):
        raise AssertionError("general 分支不应调用检索")

    monkeypatch.setattr(chat_module, "retrieve", forbidden)

    events = _run(_payload("你好"), FakeRequest())

    assert events[0] == {"type": "intent", "intent": "general"}
    assert events[1] == {"type": "sources", "sources": []}
    assert recorder.saved == [(7, "回答", None)]
    assert "【知识库上下文】" not in (recorder.prompt_system or "")


def test_stream_error_emits_error_and_done_marker(monkeypatch) -> None:
    recorder = Recorder()
    _install(
        monkeypatch,
        recorder,
        decision=IntentDecision(intent="rag"),
        stream_error=RuntimeError("boom"),
    )

    events = _run(_payload(), FakeRequest())

    assert any(item.get("type") == "error" for item in events if isinstance(item, dict))
    assert events[-1] == "[DONE]"
    assert recorder.saved == []


def test_empty_response_does_not_save(monkeypatch) -> None:
    recorder = Recorder()
    _install(monkeypatch, recorder, decision=IntentDecision(intent="rag"), deltas=())

    events = _run(_payload(), FakeRequest())

    assert any(
        isinstance(item, dict) and item.get("code") == "EMPTY_RESPONSE"
        for item in events
    )
    assert recorder.saved == []


def test_disconnect_skips_save(monkeypatch) -> None:
    recorder = Recorder()
    _install(
        monkeypatch,
        recorder,
        decision=IntentDecision(intent="rag"),
        deltas=("第一段", "第二段"),
    )

    events = _run(_payload(), FakeRequest(disconnect_after=0))

    assert recorder.saved == []
    assert "[DONE]" not in events
