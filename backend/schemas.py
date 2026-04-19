from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class KnowledgeBaseConfig(BaseModel):
    """知识库的可扩展配置，允许透传额外字段。"""

    # 先固定一组常用字段，同时允许后续实验性配置直接透传进来。
    model_config = ConfigDict(extra="allow")

    embedding_model: str | None = Field(default=None, max_length=120)
    chunk_size: int | None = Field(default=None, ge=1, le=10000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=5000)
    separator: str | None = Field(default=None, max_length=200)


class KnowledgeBaseCreateRequest(BaseModel):
    """创建知识库时提交的基本信息。"""

    name: str
    description: str | None = None
    config: KnowledgeBaseConfig


class KnowledgeBaseSummary(BaseModel):
    """知识库摘要信息。"""

    id: str
    name: str
    description: str | None
    # 返回原始配置对象，前端可以直接展示或回填表单。
    config: dict[str, Any]
    document_count: int
    created_at: str
    updated_at: str


class KnowledgeBaseListResponse(BaseModel):
    """知识库分页结果。"""

    items: list[KnowledgeBaseSummary]
    page: int
    page_size: int
    total: int
