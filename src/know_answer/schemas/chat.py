"""会话、消息与流式问答接口的请求/响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreate(BaseModel):
    knowledge_base_id: int | None = None
    title: str = Field(default="", max_length=255)


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=0, max_length=255)


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    knowledge_base_id: int | None = None
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationList(BaseModel):
    items: list[ConversationOut]


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: int
    role: str
    content: str
    sources: list[dict[str, Any]] | None = None
    created_at: datetime


class MessageList(BaseModel):
    items: list[MessageOut]


class ChatStreamRequest(BaseModel):
    conversation_id: int
    question: str = Field(min_length=1, max_length=8000)
    knowledge_base_id: int | None = None
