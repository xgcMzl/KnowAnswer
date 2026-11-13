"""会话、消息与流式问答接口。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from know_answer.core.db import get_db
from know_answer.core.errors import AppError
from know_answer.rag.llm import astream_answer
from know_answer.rag.prompt import build_general_messages, build_messages
from know_answer.rag.retrieval import retrieve
from know_answer.rag.router import classify_intent
from know_answer.schemas.chat import (
    ChatStreamRequest,
    ConversationCreate,
    ConversationList,
    ConversationOut,
    ConversationUpdate,
    MessageList,
    MessageOut,
)
from know_answer.services import chat_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/conversations", response_model=ConversationOut, status_code=201)
def create_conversation(
    payload: ConversationCreate, db: Session = Depends(get_db)
) -> ConversationOut:
    conversation = chat_service.create_conversation(
        db,
        knowledge_base_id=payload.knowledge_base_id,
        title=payload.title,
    )
    return ConversationOut.model_validate(conversation)


@router.get("/conversations", response_model=ConversationList)
def list_conversations(db: Session = Depends(get_db)) -> ConversationList:
    rows = chat_service.list_conversations(db)
    return ConversationList(
        items=[ConversationOut.model_validate(row) for row in rows]
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationOut)
def update_conversation(
    conversation_id: int,
    payload: ConversationUpdate,
    db: Session = Depends(get_db),
) -> ConversationOut:
    conversation = chat_service.update_conversation(
        db, conversation_id, payload.title
    )
    return ConversationOut.model_validate(conversation)


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: int, db: Session = Depends(get_db)
) -> None:
    chat_service.delete_conversation(db, conversation_id)


@router.get("/conversations/{conversation_id}/messages", response_model=MessageList)
def list_messages(
    conversation_id: int, db: Session = Depends(get_db)
) -> MessageList:
    rows = chat_service.list_messages(db, conversation_id)
    return MessageList(items=[MessageOut.model_validate(row) for row in rows])


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatStreamRequest, request: Request
) -> StreamingResponse:
    # 阶段 A：短事务中校验会话、读历史、保存 user 消息
    ctx = await run_in_threadpool(chat_service.prepare_stream, payload)

    # 阶段 B：意图识别（非流式单次工具选择；失败时内部兜底，不抛业务异常）
    decision = await classify_intent(
        ctx.question,
        ctx.history,
        knowledge_base_id=ctx.knowledge_base_id,
    )

    sources: list[dict[str, Any]] = []
    if decision.intent == "rag":
        # 阶段 C：线程池中完成 embedding + Milvus 检索；失败时走统一 JSON 错误响应。
        # 检索用模型改写后的 query（补全“它/这个”等指代），回答仍用用户原问题。
        sources = await run_in_threadpool(
            retrieve, decision.query or ctx.question, ctx.knowledge_base_id
        )
        messages = build_messages(ctx.question, sources, ctx.history)
    else:
        # 常规问题：不检索、不生成 sources，走通用 system prompt
        messages = build_general_messages(ctx.question, ctx.history)

    async def event_stream():
        stream = astream_answer(messages)
        try:
            yield _sse({"type": "intent", "intent": decision.intent})
            yield _sse({"type": "sources", "sources": sources})

            buffer: list[str] = []
            async for delta in stream:
                if await request.is_disconnected():
                    return  # 客户端已断开：不保存 assistant 消息
                buffer.append(delta)
                yield _sse({"type": "delta", "content": delta})

            content = "".join(buffer).strip()
            if not content:
                yield _sse(
                    {
                        "type": "error",
                        "code": "EMPTY_RESPONSE",
                        "message": "模型未返回内容，请稍后重试",
                    }
                )
            else:
                # 阶段 D：新开短事务保存 assistant 消息。
                # general 不写 sources（None），rag 保留来源列表（可能为空）。
                message_id = await run_in_threadpool(
                    chat_service.save_assistant,
                    ctx.conversation_id,
                    content,
                    sources if decision.intent == "rag" else None,
                )
                yield _sse(
                    {
                        "type": "done",
                        "conversation_id": ctx.conversation_id,
                        "message_id": message_id,
                    }
                )
        except asyncio.CancelledError:
            raise
        except AppError as exc:
            logger.exception("流式问答失败：%s", exc.code)
            yield _sse({"type": "error", "code": exc.code, "message": exc.message})
        except Exception:  # noqa: BLE001
            logger.exception("流式问答发生未处理异常")
            yield _sse(
                {
                    "type": "error",
                    "code": "INTERNAL_ERROR",
                    "message": "生成失败，请稍后重试",
                }
            )
        finally:
            # 不要在 finally 中 yield：客户端断开时 GeneratorExit 会因再次 yield 报错
            await stream.aclose()

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
