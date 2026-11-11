"""会话、消息与流式问答的前后置编排。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from know_answer.core.config import get_settings
from know_answer.core.db import session_scope
from know_answer.core.errors import InvalidArgumentError, NotFoundError
from know_answer.db.models import (
    Conversation,
    KnowledgeBase,
    Message,
    utcnow,
)
from know_answer.schemas.chat import ChatStreamRequest

logger = logging.getLogger(__name__)


@dataclass
class ChatContext:
    conversation_id: int
    knowledge_base_id: int | None
    question: str
    history: list[dict[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 会话
# --------------------------------------------------------------------------- #
def _get_conversation_or_404(db: Session, conversation_id: int) -> Conversation:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError(f"会话不存在：{conversation_id}")
    return conversation


def list_conversations(db: Session) -> list[Conversation]:
    return list(
        db.scalars(
            select(Conversation).order_by(
                Conversation.updated_at.desc(), Conversation.id.desc()
            )
        ).all()
    )


def create_conversation(
    db: Session, *, knowledge_base_id: int | None, title: str = ""
) -> Conversation:
    if knowledge_base_id is not None and db.get(KnowledgeBase, knowledge_base_id) is None:
        raise NotFoundError(f"知识库不存在：{knowledge_base_id}")
    conversation = Conversation(
        knowledge_base_id=knowledge_base_id,
        title=(title or "").strip(),
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def update_conversation(db: Session, conversation_id: int, title: str) -> Conversation:
    conversation = _get_conversation_or_404(db, conversation_id)
    conversation.title = title.strip()
    conversation.updated_at = utcnow()
    db.commit()
    db.refresh(conversation)
    return conversation


def delete_conversation(db: Session, conversation_id: int) -> None:
    conversation = _get_conversation_or_404(db, conversation_id)
    db.delete(conversation)
    db.commit()


def list_messages(db: Session, conversation_id: int) -> list[Message]:
    _get_conversation_or_404(db, conversation_id)
    return list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        ).all()
    )


# --------------------------------------------------------------------------- #
# 流式问答
# --------------------------------------------------------------------------- #
def _load_history(
    db: Session, conversation_id: int, *, limit: int, max_chars: int
) -> list[dict[str, str]]:
    rows = list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(limit)
        ).all()
    )
    if not rows:
        return []

    selected: list[Message] = []
    used = 0
    for message in rows:  # 从新到旧累加预算
        cost = len(message.content or "")
        if selected and used + cost > max_chars:
            break
        selected.append(message)
        used += cost
    selected.reverse()
    return [
        {"role": message.role, "content": message.content}
        for message in selected
        if message.content
    ]


def prepare_stream(req: ChatStreamRequest) -> ChatContext:
    """校验会话、读历史、写 user 消息。使用独立的短事务。"""
    settings = get_settings()
    question = req.question.strip()
    if not question:
        raise InvalidArgumentError("问题不能为空")

    with session_scope() as db:
        conversation = _get_conversation_or_404(db, req.conversation_id)
        knowledge_base_id = (
            req.knowledge_base_id
            if req.knowledge_base_id is not None
            else conversation.knowledge_base_id
        )
        if (
            knowledge_base_id is not None
            and db.get(KnowledgeBase, knowledge_base_id) is None
        ):
            raise NotFoundError(f"知识库不存在：{knowledge_base_id}")

        history = _load_history(
            db,
            conversation.id,
            limit=settings.memory_max_messages,
            max_chars=settings.memory_max_chars,
        )

        db.add(
            Message(
                conversation_id=conversation.id,
                role="user",
                content=question,
                sources=None,
            )
        )
        if not conversation.title:
            conversation.title = question.replace("\n", " ").strip()[:30]
        conversation.updated_at = utcnow()
        db.flush()

    return ChatContext(
        conversation_id=req.conversation_id,
        knowledge_base_id=knowledge_base_id,
        question=question,
        history=history,
    )


def save_assistant(
    conversation_id: int, content: str, sources: list[dict[str, Any]] | None
) -> int:
    """流式生成完成后保存 assistant 消息；使用独立的短事务。"""
    with session_scope() as db:
        conversation = db.get(Conversation, conversation_id)
        if conversation is None:
            raise NotFoundError(f"会话不存在：{conversation_id}")
        message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            sources=sources,
        )
        db.add(message)
        conversation.updated_at = utcnow()
        db.flush()
        return message.id
