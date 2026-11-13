"""意图路由（LangChain Tool 轻量路由）单元测试。"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage

import know_answer.rag.router as router
from know_answer.core.config import get_settings
from know_answer.rag.router import GENERAL_TOOL_NAME, RAG_TOOL_NAME, classify_intent


class FakeToolModel:
    """模拟 bind_tools + ainvoke 的 LangChain 聊天模型。"""

    def __init__(self, message=None, error=None, delay: float = 0.0) -> None:
        self.message = message
        self.error = error
        self.delay = delay
        self.bound_tools = None
        self.tool_choice = None
        self.called = False
        self.messages = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        self.tool_choice = kwargs.get("tool_choice")
        return self

    async def ainvoke(self, messages, **kwargs):
        self.called = True
        self.messages = messages
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.message


def _tool_call(name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1"}])


def _install(monkeypatch, fake: FakeToolModel) -> None:
    monkeypatch.setattr(router, "get_chat_model", lambda: fake)
    monkeypatch.setattr(router, "is_llm_configured", lambda: True)


def _classify(question: str = "问题", *, knowledge_base_id: int | None = 1):
    return asyncio.run(
        classify_intent(question, [], knowledge_base_id=knowledge_base_id)
    )


def test_rag_tool_call_returns_rewritten_query(monkeypatch) -> None:
    fake = FakeToolModel(_tool_call(RAG_TOOL_NAME, {"query": "第二条的生效时间"}))
    _install(monkeypatch, fake)

    decision = _classify("那第二条呢？")

    assert decision.intent == "rag"
    assert decision.query == "第二条的生效时间"
    assert decision.decided_by == "model"
    assert fake.called


def test_general_tool_call(monkeypatch) -> None:
    fake = FakeToolModel(_tool_call(GENERAL_TOOL_NAME, {"question": "你好"}))
    _install(monkeypatch, fake)

    decision = _classify("你好")

    assert decision.intent == "general"
    assert decision.query is None


def test_no_tool_calls_falls_back_to_rag(monkeypatch) -> None:
    fake = FakeToolModel(AIMessage(content="你好"))
    _install(monkeypatch, fake)

    decision = _classify()

    assert decision.intent == "rag"
    assert decision.decided_by == "fallback"


def test_unknown_tool_falls_back_to_rag(monkeypatch) -> None:
    fake = FakeToolModel(_tool_call("some_unknown_tool", {}))
    _install(monkeypatch, fake)

    decision = _classify()

    assert decision.intent == "rag"
    assert decision.decided_by == "fallback"


def test_llm_error_falls_back_to_rag(monkeypatch) -> None:
    fake = FakeToolModel(error=RuntimeError("boom"))
    _install(monkeypatch, fake)

    decision = _classify()

    assert decision.intent == "rag"
    assert decision.decided_by == "fallback"


def test_timeout_falls_back_to_rag(monkeypatch) -> None:
    fake = FakeToolModel(_tool_call(GENERAL_TOOL_NAME, {}), delay=10)
    _install(monkeypatch, fake)
    monkeypatch.setattr(get_settings(), "intent_router_timeout_seconds", 0.01)

    decision = _classify()

    assert decision.intent == "rag"
    assert decision.decided_by == "fallback"


def test_none_knowledge_base_still_routes(monkeypatch) -> None:
    fake = FakeToolModel(_tool_call(GENERAL_TOOL_NAME, {}))
    _install(monkeypatch, fake)

    decision = _classify(knowledge_base_id=None)

    assert decision.intent == "general"
    assert fake.called


def test_router_disabled_returns_rag_without_call(monkeypatch) -> None:
    fake = FakeToolModel(_tool_call(GENERAL_TOOL_NAME, {}))
    _install(monkeypatch, fake)
    monkeypatch.setattr(get_settings(), "intent_routing_enabled", False)

    decision = _classify()

    assert decision.intent == "rag"
    assert decision.decided_by == "disabled"
    assert not fake.called


def test_unconfigured_llm_returns_rag_without_call(monkeypatch) -> None:
    fake = FakeToolModel(_tool_call(GENERAL_TOOL_NAME, {}))
    monkeypatch.setattr(router, "get_chat_model", lambda: fake)
    monkeypatch.setattr(router, "is_llm_configured", lambda: False)

    decision = _classify()

    assert decision.intent == "rag"
    assert decision.decided_by == "disabled"
    assert not fake.called


def test_query_is_truncated(monkeypatch) -> None:
    fake = FakeToolModel(_tool_call(RAG_TOOL_NAME, {"query": "很长的检索问句" * 10}))
    _install(monkeypatch, fake)
    monkeypatch.setattr(get_settings(), "intent_router_query_max_chars", 5)

    decision = _classify()

    assert decision.intent == "rag"
    assert decision.query is not None
    assert len(decision.query) <= 5


def test_empty_query_returns_none(monkeypatch) -> None:
    fake = FakeToolModel(_tool_call(RAG_TOOL_NAME, {"query": "   "}))
    _install(monkeypatch, fake)

    decision = _classify()

    assert decision.intent == "rag"
    assert decision.query is None


def test_tool_choice_is_passed_through(monkeypatch) -> None:
    fake = FakeToolModel(_tool_call(GENERAL_TOOL_NAME, {}))
    _install(monkeypatch, fake)

    _classify()
    assert fake.tool_choice == "auto"

    monkeypatch.setattr(get_settings(), "intent_router_tool_choice", "required")
    _classify()
    assert fake.tool_choice == "required"

    monkeypatch.setattr(get_settings(), "intent_router_tool_choice", "invalid")
    _classify()
    assert fake.tool_choice == "auto"


def test_raw_additional_kwargs_tool_calls_are_parsed(monkeypatch) -> None:
    message = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": RAG_TOOL_NAME,
                        "arguments": '{"query": "制度原文中的生效时间"}',
                    },
                }
            ]
        },
    )
    fake = FakeToolModel(message)
    _install(monkeypatch, fake)

    decision = _classify()

    assert decision.intent == "rag"
    assert decision.query == "制度原文中的生效时间"


def test_history_is_truncated_for_router(monkeypatch) -> None:
    fake = FakeToolModel(_tool_call(GENERAL_TOOL_NAME, {}))
    _install(monkeypatch, fake)
    monkeypatch.setattr(get_settings(), "intent_router_history_messages", 2)
    monkeypatch.setattr(get_settings(), "intent_router_history_chars", 2000)

    history = [
        {"role": "user", "content": "第一轮"},
        {"role": "assistant", "content": "第二轮"},
        {"role": "user", "content": "第三轮"},
        {"role": "assistant", "content": "第四轮"},
    ]
    asyncio.run(classify_intent("第五轮", history, knowledge_base_id=1))

    messages = fake.messages
    # system + 最近 2 条历史 + 当前问题
    assert len(messages) == 4
    assert messages[1]["content"] == "第三轮"
    assert messages[2]["content"] == "第四轮"
    assert messages[3]["content"] == "第五轮"
