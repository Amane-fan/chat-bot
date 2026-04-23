from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy.orm import Mapped, mapped_column

from backend.db import Base


class ChatSession(Base):
    """聊天会话元数据。"""

    __tablename__ = "chat_sessions"
    __table_args__ = (Index("ix_chat_sessions_updated_at", "updated_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)


class ChatMessage(Base):
    """聊天消息明细。"""

    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_session_created", "session_id", "created_at", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 保存本次回答命中的知识库片段快照，刷新页面后仍可展示“来源”。
    retrieved_chunks: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)


class SessionKnowledgeBase(Base):
    """会话与知识库的多对多关联，保留用户选择顺序。"""

    __tablename__ = "session_knowledge_bases"
    __table_args__ = (
        Index("ix_session_knowledge_bases_session_id", "session_id"),
        Index("ix_session_knowledge_bases_knowledge_base_id", "knowledge_base_id"),
    )

    session_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        primary_key=True,
    )
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)


class ChatSessionMemory(Base):
    """会话级摘要记忆。"""

    __tablename__ = "chat_session_memories"
    __table_args__ = (Index("ix_chat_session_memories_updated_at", "updated_at"),)

    session_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    summarized_message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)


class KnowledgeBase(Base):
    """知识库元数据表，当前只保存基础信息和预留配置。"""

    __tablename__ = "knowledge_bases"

    # 主键使用 32 位 UUID hex，方便和现有会话 id 风格保持一致。
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    # 名称要求全局唯一，便于后端直接按重名返回 409。
    name: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 配置先收敛到 JSON，后续接入 embedding/切分策略时不必立刻改表。
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    # 当前先预留文档计数，后续接文档上传时直接复用。
    document_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)


class KnowledgeBaseDocument(Base):
    """知识库上传文档及其处理状态。"""

    __tablename__ = "knowledge_base_documents"
    __table_args__ = (
        Index("ix_knowledge_base_documents_knowledge_base_id", "knowledge_base_id"),
        Index("ix_knowledge_base_documents_status", "status"),
        Index(
            "uq_knowledge_base_documents_name",
            "knowledge_base_id",
            "original_filename",
            unique=True,
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    knowledge_base_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
