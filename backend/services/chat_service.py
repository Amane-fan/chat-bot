import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend import config
from backend.db import get_engine, get_session_factory, init_mysql
from backend.models import ChatMessage, ChatSession
from backend.schemas import ChatExchangeResponse, MessageRecord, SessionSummary


def utc_now() -> datetime:
    """MySQL DATETIME 不带时区，这里统一存 UTC naive datetime。"""

    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_iso_string(value: datetime) -> str:
    """对外响应统一补回 UTC 时区，方便前后端按 ISO 处理。"""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


class ChatService:
    """封装 MySQL 会话存储和模型链，供路由层直接调用。"""

    def __init__(self) -> None:
        self.chain = None

    def startup(self) -> None:
        """应用启动时初始化依赖，避免在模块导入阶段直接失败。"""

        self._validate_settings()
        init_mysql()
        try:
            engine = get_engine()
            with engine.connect():
                pass
            inspector = inspect(engine)
            missing_tables = [
                table_name
                for table_name in ("chat_sessions", "chat_messages")
                if not inspector.has_table(table_name)
            ]
            if missing_tables:
                raise RuntimeError(
                    "MySQL 中缺少聊天相关表，请先执行 backend/sql/mysql_schema.sql"
                )
        except SQLAlchemyError as exc:
            raise RuntimeError("无法连接 MySQL，请检查 MYSQL_* 配置") from exc

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个智能助手，热情耐心地回答用户的问题。"),
                MessagesPlaceholder("history", optional=True),
                ("human", "{question}"),
            ]
        )
        model = ChatOpenAI(
            model=config.DASHSCOPE_DEFAULT_MODEL,
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
        )
        self.chain = prompt | model | StrOutputParser()

    def list_sessions(self) -> list[SessionSummary]:
        """按最近活跃时间返回会话列表。"""

        session_factory = get_session_factory()
        message_counts = (
            select(
                ChatMessage.session_id.label("session_id"),
                func.count(ChatMessage.id).label("message_count"),
            )
            .group_by(ChatMessage.session_id)
            .subquery()
        )

        with session_factory() as session:
            rows = session.execute(
                select(
                    ChatSession,
                    func.coalesce(message_counts.c.message_count, 0).label("message_count"),
                )
                .outerjoin(message_counts, ChatSession.id == message_counts.c.session_id)
                .order_by(ChatSession.updated_at.desc())
            ).all()

        return [
            self._session_summary_from_model(chat_session, int(message_count))
            for chat_session, message_count in rows
        ]

    def create_session(self, title: str | None = None) -> SessionSummary:
        """创建新会话，并写入初始元数据。"""

        now = utc_now()
        chat_session = ChatSession(
            id=uuid.uuid4().hex,
            title=self._normalize_optional_title(title) or "新对话",
            created_at=now,
            updated_at=now,
        )

        session_factory = get_session_factory()
        with session_factory() as session:
            session.add(chat_session)
            session.commit()

        return self._session_summary_from_model(chat_session, 0)

    def rename_session(self, session_id: str, title: str) -> SessionSummary:
        """修改会话标题，并刷新会话排序时间。"""

        session_factory = get_session_factory()
        with session_factory() as session:
            chat_session = self._get_session_or_404(session, session_id)
            chat_session.title = title
            chat_session.updated_at = utc_now()
            message_count = self._count_messages(session, session_id)
            session.commit()
            return self._session_summary_from_model(chat_session, message_count)

    def delete_session(self, session_id: str) -> None:
        """删除会话元数据和关联消息。"""

        session_factory = get_session_factory()
        with session_factory() as session:
            chat_session = self._get_session_or_404(session, session_id)
            session.delete(chat_session)
            session.commit()

    def get_session_messages(self, session_id: str) -> list[MessageRecord]:
        """返回指定会话的完整消息历史。"""

        session_factory = get_session_factory()
        with session_factory() as session:
            self._get_session_or_404(session, session_id)
            return self._load_messages(session, session_id)

    def send_message(self, session_id: str, content: str) -> ChatExchangeResponse:
        """执行一次问答，并把用户消息和模型回复写入 MySQL。"""

        normalized_content = content.strip()
        if not normalized_content:
            raise HTTPException(status_code=400, detail="消息内容不能为空")

        session_factory = get_session_factory()
        with session_factory() as session:
            self._get_session_or_404(session, session_id)
            history = self._load_history(session, session_id)

        try:
            answer = self._get_chain().invoke(
                {"question": normalized_content, "history": history}
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail="模型调用失败") from exc

        user_message_created_at = utc_now()
        assistant_message_created_at = utc_now()

        with session_factory() as session:
            chat_session = self._get_session_or_404(session, session_id)
            if chat_session.title == "新对话" and self._count_messages(session, session_id) == 0:
                chat_session.title = self._build_session_title(normalized_content)
            chat_session.updated_at = assistant_message_created_at

            user_message_model = ChatMessage(
                session_id=session_id,
                role="user",
                content=normalized_content,
                created_at=user_message_created_at,
            )
            assistant_message_model = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=answer,
                created_at=assistant_message_created_at,
            )
            session.add_all([user_message_model, assistant_message_model])
            session.commit()

            summary = self._session_summary_from_model(
                chat_session,
                self._count_messages(session, session_id),
            )

        return ChatExchangeResponse(
            session=summary,
            user_message=self._message_record_from_model(user_message_model),
            assistant_message=self._message_record_from_model(assistant_message_model),
        )

    def _validate_settings(self) -> None:
        """启动前校验关键环境变量，避免请求阶段才暴露配置问题。"""

        missing = []
        if not config.DASHSCOPE_API_KEY:
            missing.append("DASHSCOPE_API_KEY")
        if not config.DASHSCOPE_BASE_URL:
            missing.append("DASHSCOPE_BASE_URL")
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

    def _get_chain(self):
        if self.chain is None:
            raise RuntimeError("模型链尚未初始化")
        return self.chain

    def _build_session_title(self, content: str) -> str:
        normalized = " ".join(content.split())
        if not normalized:
            return "新对话"
        return normalized[:20]

    def _normalize_optional_title(self, title: str | None) -> str | None:
        if title is None:
            return None
        normalized = " ".join(title.split())
        return normalized or None

    def _session_summary_from_model(
        self, chat_session: ChatSession, message_count: int
    ) -> SessionSummary:
        return SessionSummary(
            id=chat_session.id,
            title=chat_session.title,
            created_at=to_iso_string(chat_session.created_at),
            updated_at=to_iso_string(chat_session.updated_at),
            message_count=message_count,
        )

    def _message_record_from_model(self, message: ChatMessage) -> MessageRecord:
        return MessageRecord(
            role=message.role, content=message.content, created_at=to_iso_string(message.created_at)
        )

    def _load_messages(self, session: Session, session_id: str) -> list[MessageRecord]:
        messages = session.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        ).all()
        return [self._message_record_from_model(message) for message in messages]

    def _load_history(self, session: Session, session_id: str) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in self._load_messages(session, session_id)
        ]

    def _count_messages(self, session: Session, session_id: str) -> int:
        return session.scalar(
            select(func.count(ChatMessage.id)).where(ChatMessage.session_id == session_id)
        ) or 0

    def _get_session_or_404(self, session: Session, session_id: str) -> ChatSession:
        chat_session = session.get(ChatSession, session_id)
        if chat_session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return chat_session
