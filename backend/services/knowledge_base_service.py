import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from backend import config
from backend.db import get_engine, get_session_factory, init_mysql
from backend.models import KnowledgeBase
from backend.schemas import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseListResponse,
    KnowledgeBaseSummary,
)


def utc_now() -> datetime:
    # MySQL DATETIME 不带时区，这里统一转成 UTC 的 naive datetime 存储。
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_iso_string(value: datetime) -> str:
    # 对外响应统一补回 UTC 时区，前端展示时就能按标准 ISO 处理。
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


class KnowledgeBaseService:
    def startup(self) -> None:
        # 启动阶段提前暴露配置或建表问题，避免请求进来后才报错。
        self._validate_settings()
        init_mysql()

        try:
            engine = get_engine()
            with engine.connect():
                pass
            # 本项目不做自动建表，明确要求先执行仓库里的 SQL 文件。
            if not inspect(engine).has_table("knowledge_bases"):
                raise RuntimeError(
                    "MySQL 中缺少 knowledge_bases 表，请先执行 backend/sql/mysql_schema.sql"
                )
        except SQLAlchemyError as exc:
            raise RuntimeError("无法连接 MySQL，请检查 MYSQL_* 配置") from exc

    def create_knowledge_base(
        self, payload: KnowledgeBaseCreateRequest
    ) -> KnowledgeBaseSummary:
        name = self._normalize_name(payload.name)
        description = self._normalize_description(payload.description)
        # 去掉 None，避免把未填写项也存成显式 null。
        config_payload = payload.config.model_dump(exclude_none=True)
        now = utc_now()

        knowledge_base = KnowledgeBase(
            id=uuid.uuid4().hex,
            name=name,
            description=description,
            config=config_payload,
            document_count=0,
            created_at=now,
            updated_at=now,
        )

        session_factory = get_session_factory()
        with session_factory() as session:
            session.add(knowledge_base)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                # 名称唯一约束最终以数据库为准，避免并发下只靠应用层校验。
                raise HTTPException(status_code=409, detail="知识库名称已存在") from exc
            return self._to_summary(knowledge_base)

    def list_knowledge_bases(self, page: int, page_size: int) -> KnowledgeBaseListResponse:
        if page < 1 or page_size < 1:
            raise HTTPException(status_code=400, detail="分页参数必须大于 0")
        if page_size > 100:
            raise HTTPException(status_code=400, detail="page_size 不能超过 100")

        session_factory = get_session_factory()
        with session_factory() as session:
            total = session.scalar(select(func.count()).select_from(KnowledgeBase)) or 0
            offset = (page - 1) * page_size
            # 当前按创建时间倒序分页，便于前端优先看到最新创建的知识库。
            items = session.scalars(
                select(KnowledgeBase)
                .order_by(KnowledgeBase.created_at.desc())
                .offset(offset)
                .limit(page_size)
            ).all()

        return KnowledgeBaseListResponse(
            items=[self._to_summary(item) for item in items],
            page=page,
            page_size=page_size,
            total=total,
        )

    def _validate_settings(self) -> None:
        missing = []
        if not config.MYSQL_HOST:
            missing.append("MYSQL_HOST")
        if not config.MYSQL_USER:
            missing.append("MYSQL_USER")
        if config.MYSQL_PASSWORD is None:
            missing.append("MYSQL_PASSWORD")
        if not config.MYSQL_DATABASE:
            missing.append("MYSQL_DATABASE")
        if missing:
            raise RuntimeError(f"缺少必要环境变量: {', '.join(missing)}")

    def _normalize_name(self, name: str) -> str:
        # 把连续空白压成一个空格，避免“看起来一样”的名称实际不同。
        normalized = " ".join(name.split())
        if not normalized:
            raise HTTPException(status_code=400, detail="知识库名称不能为空")
        if len(normalized) > 80:
            raise HTTPException(status_code=400, detail="知识库名称不能超过 80 个字符")
        return normalized

    def _normalize_description(self, description: str | None) -> str | None:
        if description is None:
            return None
        normalized = description.strip()
        # 空描述统一视为未填写，数据库里存 NULL。
        if not normalized:
            return None
        return normalized

    def _to_summary(self, knowledge_base: KnowledgeBase) -> KnowledgeBaseSummary:
        return KnowledgeBaseSummary(
            id=knowledge_base.id,
            name=knowledge_base.name,
            description=knowledge_base.description,
            config=knowledge_base.config,
            document_count=knowledge_base.document_count,
            created_at=to_iso_string(knowledge_base.created_at),
            updated_at=to_iso_string(knowledge_base.updated_at),
        )
