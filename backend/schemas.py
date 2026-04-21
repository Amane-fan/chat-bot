from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KnowledgeBaseRetrievedChunk(BaseModel):
    """一次问答中命中的知识库片段。"""

    knowledge_base_id: str
    knowledge_base_name: str
    document_id: str | None = None
    original_filename: str | None = None
    chunk_index: int | None = None
    score: float | None = None
    text: str


class MessageRecord(BaseModel):
    """单条聊天消息。"""

    role: Literal["user", "assistant"]
    content: str
    created_at: str
    retrieved_chunks: list[KnowledgeBaseRetrievedChunk] = Field(default_factory=list)


class KnowledgeBaseReference(BaseModel):
    """会话或选择器里使用的轻量知识库信息。"""

    id: str
    name: str


class SessionSummary(BaseModel):
    """前端会话列表需要的摘要信息。"""

    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    knowledge_bases: list[KnowledgeBaseReference]


class SessionCreateRequest(BaseModel):
    """创建会话时允许传入一个可选标题。"""

    title: str | None = Field(default=None, max_length=80)
    knowledge_base_ids: list[str] = Field(default_factory=list)


class SessionRenameRequest(BaseModel):
    """重命名会话。"""

    title: str = Field(min_length=1, max_length=80)


class SessionKnowledgeBaseUpdateRequest(BaseModel):
    """完整替换一个会话绑定的知识库列表。"""

    knowledge_base_ids: list[str] = Field(default_factory=list)


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

    @model_validator(mode="after")
    def validate_chunk_config(self) -> "KnowledgeBaseConfig":
        if (
            self.chunk_size is not None
            and self.chunk_overlap is not None
            and self.chunk_overlap >= self.chunk_size
        ):
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return self


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


class KnowledgeBaseDocumentSummary(BaseModel):
    """知识库文档摘要信息。"""

    id: str
    knowledge_base_id: str
    original_filename: str
    stored_filename: str
    content_type: str | None
    file_size: int
    status: Literal["processing", "ready", "failed"]
    chunk_count: int
    error_message: str | None
    created_at: str
    updated_at: str


class KnowledgeBaseDocumentListResponse(BaseModel):
    """某个知识库下的文档列表。"""

    knowledge_base_id: str
    items: list[KnowledgeBaseDocumentSummary]


class KnowledgeBaseDocumentUploadResponse(BaseModel):
    """上传文档后的聚合返回。"""

    document: KnowledgeBaseDocumentSummary
    knowledge_base: KnowledgeBaseSummary


class KnowledgeBaseDocumentDeleteResponse(BaseModel):
    """删除文档后的聚合返回。"""

    document: KnowledgeBaseDocumentSummary
    knowledge_base: KnowledgeBaseSummary
