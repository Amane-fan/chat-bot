from typing import Literal

from pydantic import BaseModel, Field


class MessageRecord(BaseModel):
    """单条聊天消息。"""

    role: Literal["user", "assistant"]
    content: str
    created_at: str


class SessionSummary(BaseModel):
    """前端会话列表需要的摘要信息。"""

    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class SessionCreateRequest(BaseModel):
    """创建会话时允许传入一个可选标题。"""

    title: str | None = Field(default=None, max_length=80)


class SessionRenameRequest(BaseModel):
    """重命名会话。"""

    title: str = Field(min_length=1, max_length=80)


class ChatRequest(BaseModel):
    """用户发送给模型的消息。"""

    content: str = Field(min_length=1, max_length=4000)


class ChatExchangeResponse(BaseModel):
    """一次对话交换，前端收到后可直接更新消息区和会话列表。"""

    session: SessionSummary
    user_message: MessageRecord
    assistant_message: MessageRecord
