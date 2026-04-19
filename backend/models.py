from datetime import datetime

from sqlalchemy import Integer, JSON, String, Text
from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy.orm import Mapped, mapped_column

from backend.db import Base


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
