import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from sqlalchemy import delete, func, inspect, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend import config
from backend.db import get_engine, get_session_factory, init_mysql
from backend.models import ChatMessage, ChatSession, KnowledgeBase, SessionKnowledgeBase
from backend.schemas import (
    ChatExchangeResponse,
    KnowledgeBaseRetrievedChunk,
    KnowledgeBaseReference,
    MessageRecord,
    SessionSummary,
)
from backend.services.knowledge_base_service import KnowledgeBaseService


BASE_SYSTEM_PROMPT = "你是一个智能助手，热情耐心地回答用户的问题。"
KNOWLEDGE_SYSTEM_INSTRUCTION = (
    "仅在相关时参考以下知识库内容；如果知识库资料不足或没有覆盖问题，"
    "请明确说明无法从知识库确认，并基于通用知识谨慎回答。"
    "回答中引用知识库内容时，请使用 [来源 1] 这样的来源编号。"
)
QUESTION_REWRITE_SYSTEM_PROMPT = (
    "你负责把用户当前问题改写成一个独立、明确、适合检索和回答的问题。"
    "请结合已有对话历史补全指代、省略和上下文，但不要引入历史中不存在的信息。"
    "如果当前问题已经足够独立，请原样或近似原样返回。"
    "只输出改写后的问题，不要解释、不要添加前缀。"
)
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


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

    def __init__(self, knowledge_base_service: KnowledgeBaseService | None = None) -> None:
        self.chain = None
        self.rewrite_chain = None
        self.knowledge_base_service = knowledge_base_service

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
                for table_name in ("chat_sessions", "chat_messages", "session_knowledge_bases")
                if not inspector.has_table(table_name)
            ]
            if missing_tables:
                raise RuntimeError(
                    "MySQL 中缺少聊天相关表，请先执行 backend/sql/mysql_schema.sql"
                )
            self._ensure_chat_message_columns(inspector)
        except SQLAlchemyError as exc:
            raise RuntimeError("无法连接 MySQL，请检查 MYSQL_* 配置") from exc

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                MessagesPlaceholder("history", optional=True),
                ("human", "{question}"),
            ]
        )
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QUESTION_REWRITE_SYSTEM_PROMPT),
                MessagesPlaceholder("history", optional=True),
                ("human", "当前用户问题：{question}\n\n请输出改写后的问题："),
            ]
        )
        model = ChatOpenAI(
            model=config.DASHSCOPE_DEFAULT_MODEL,
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
        )
        self.chain = prompt | model | StrOutputParser()
        self.rewrite_chain = rewrite_prompt | model | StrOutputParser()

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

            session_ids = [chat_session.id for chat_session, _ in rows]
            knowledge_base_map = self._list_knowledge_bases_for_sessions(session, session_ids)

        return [
            self._session_summary_from_model(
                chat_session,
                int(message_count),
                knowledge_base_map.get(chat_session.id, []),
            )
            for chat_session, message_count in rows
        ]

    def create_session(
        self, title: str | None = None, knowledge_base_ids: list[str] | None = None
    ) -> SessionSummary:
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
            normalized_knowledge_base_ids = self._normalize_knowledge_base_ids(knowledge_base_ids)
            self._validate_knowledge_base_ids(session, normalized_knowledge_base_ids)
            session.add(chat_session)
            session.flush()
            self._replace_session_knowledge_base_links(
                session, chat_session.id, normalized_knowledge_base_ids, now
            )
            session.commit()

            return self._build_session_summary(session, chat_session, 0)

    def rename_session(self, session_id: str, title: str) -> SessionSummary:
        """修改会话标题，并刷新会话排序时间。"""

        session_factory = get_session_factory()
        with session_factory() as session:
            chat_session = self._get_session_or_404(session, session_id)
            chat_session.title = title
            chat_session.updated_at = utc_now()
            message_count = self._count_messages(session, session_id)
            session.commit()
            return self._build_session_summary(session, chat_session, message_count)

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

    def replace_session_knowledge_bases(
        self, session_id: str, knowledge_base_ids: list[str] | None = None
    ) -> SessionSummary:
        session_factory = get_session_factory()
        with session_factory() as session:
            chat_session = self._get_session_or_404(session, session_id)
            normalized_knowledge_base_ids = self._normalize_knowledge_base_ids(knowledge_base_ids)
            self._validate_knowledge_base_ids(session, normalized_knowledge_base_ids)
            now = utc_now()
            self._replace_session_knowledge_base_links(
                session, session_id, normalized_knowledge_base_ids, now
            )
            chat_session.updated_at = now
            message_count = self._count_messages(session, session_id)
            session.commit()
            return self._build_session_summary(session, chat_session, message_count)

    def send_message(self, session_id: str, content: str) -> ChatExchangeResponse:
        """执行一次问答，并把用户消息和模型回复写入 MySQL。"""

        normalized_content = content.strip()
        if not normalized_content:
            raise HTTPException(status_code=400, detail="消息内容不能为空")

        session_factory = get_session_factory()
        prepared_request = self._prepare_chat_request(
            session_id=session_id,
            original_question=normalized_content,
            request_mode="sync",
        )
        retrieved_chunks = prepared_request["retrieved_chunks"]
        llm_payload = prepared_request["llm_payload"]

        try:
            answer = self._get_chain().invoke(llm_payload)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="模型调用失败") from exc
        answer = self._append_reference_section(answer, retrieved_chunks)
        retrieved_chunk_payload = self._serialize_retrieved_chunks(retrieved_chunks)

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
                retrieved_chunks=retrieved_chunk_payload,
                created_at=assistant_message_created_at,
            )
            session.add_all([user_message_model, assistant_message_model])
            session.commit()

            summary = self._build_session_summary(
                session,
                chat_session,
                self._count_messages(session, session_id),
            )

        return ChatExchangeResponse(
            session=summary,
            user_message=self._message_record_from_model(user_message_model),
            assistant_message=self._message_record_from_model(assistant_message_model),
        )

    def stream_message(self, session_id: str, content: str):
        """流式执行一次问答，并在完整答案生成后持久化消息。"""

        normalized_content = content.strip()
        if not normalized_content:
            raise HTTPException(status_code=400, detail="消息内容不能为空")

        session_factory = get_session_factory()
        prepared_request = self._prepare_chat_request(
            session_id=session_id,
            original_question=normalized_content,
            request_mode="stream",
        )
        retrieved_chunks = prepared_request["retrieved_chunks"]
        llm_payload = prepared_request["llm_payload"]

        user_message_created_at = utc_now()
        assistant_message_created_at = utc_now()
        user_message = MessageRecord(
            role="user",
            content=normalized_content,
            created_at=to_iso_string(user_message_created_at),
        )
        retrieved_chunk_records = self._to_retrieved_chunk_records(retrieved_chunks)
        yield self._stream_event(
            "metadata",
            {
                "user_message": user_message.model_dump(),
                "retrieved_chunks": [
                    chunk.model_dump() for chunk in retrieved_chunk_records
                ],
            },
        )

        answer_parts: list[str] = []
        try:
            for chunk in self._get_chain().stream(llm_payload):
                if not chunk:
                    continue
                chunk_text = str(chunk)
                answer_parts.append(chunk_text)
                yield self._stream_event("token", {"content": chunk_text})
        except Exception:
            yield self._stream_event("error", {"detail": "模型调用失败"})
            return

        raw_answer = "".join(answer_parts)
        answer = self._append_reference_section(raw_answer, retrieved_chunks)
        reference_suffix = answer[len(raw_answer) :]
        if reference_suffix:
            yield self._stream_event("token", {"content": reference_suffix})

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
                retrieved_chunks=[
                    chunk.model_dump() for chunk in retrieved_chunk_records
                ],
                created_at=assistant_message_created_at,
            )
            session.add_all([user_message_model, assistant_message_model])
            session.commit()

            summary = self._build_session_summary(
                session,
                chat_session,
                self._count_messages(session, session_id),
            )

        exchange = ChatExchangeResponse(
            session=summary,
            user_message=self._message_record_from_model(user_message_model),
            assistant_message=self._message_record_from_model(assistant_message_model),
        )
        yield self._stream_event("done", {"exchange": exchange.model_dump()})

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

    def _get_rewrite_chain(self):
        if self.rewrite_chain is None:
            raise RuntimeError("问题重写链尚未初始化")
        return self.rewrite_chain

    def _prepare_chat_request(
        self,
        *,
        session_id: str,
        original_question: str,
        request_mode: str,
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        session_factory = get_session_factory()
        with session_factory() as session:
            self._get_session_or_404(session, session_id)
            history = self._load_history(session, session_id)
            knowledge_bases = self._list_knowledge_bases_for_sessions(
                session, [session_id]
            ).get(session_id, [])

        rewritten_question, rewrite_fallback_reason = self._rewrite_question(
            request_id=request_id,
            session_id=session_id,
            original_question=original_question,
            history=history,
        )
        retrieved_chunks = self._retrieve_knowledge_chunks(
            rewritten_question, knowledge_bases
        )
        system_prompt = self._build_system_prompt(retrieved_chunks)
        llm_payload = {
            "system_prompt": system_prompt,
            "question": rewritten_question,
            "history": history,
        }
        self._log_chat_request(
            request_id=request_id,
            request_mode=request_mode,
            session_id=session_id,
            original_question=original_question,
            rewritten_question=rewritten_question,
            rewrite_fallback_reason=rewrite_fallback_reason,
            history=history,
            system_prompt=system_prompt,
            llm_payload=llm_payload,
            knowledge_bases=knowledge_bases,
            retrieved_chunks=retrieved_chunks,
        )
        return {
            "request_id": request_id,
            "history": history,
            "knowledge_bases": knowledge_bases,
            "retrieved_chunks": retrieved_chunks,
            "llm_payload": llm_payload,
            "rewritten_question": rewritten_question,
            "rewrite_fallback_reason": rewrite_fallback_reason,
        }

    def _rewrite_question(
        self,
        *,
        request_id: str,
        session_id: str,
        original_question: str,
        history: list[dict[str, str]],
    ) -> tuple[str, str | None]:
        rewrite_payload = {
            "question": original_question,
            "history": history,
        }
        try:
            rewritten_question = str(
                self._get_rewrite_chain().invoke(rewrite_payload)
            ).strip()
            if not rewritten_question:
                raise ValueError("rewrite chain returned empty question")
            return rewritten_question, None
        except Exception as exc:
            fallback_reason = type(exc).__name__
            logger.warning(
                "Question rewrite failed; falling back to original question: %s",
                json.dumps(
                    {
                        "request_id": request_id,
                        "session_id": session_id,
                        "error_type": fallback_reason,
                        "error_message": str(exc),
                        "fallback_question": original_question,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
            )
            return original_question, fallback_reason

    def _ensure_chat_message_columns(self, inspector) -> None:
        column_names = {
            column["name"] for column in inspector.get_columns("chat_messages")
        }
        if "retrieved_chunks" in column_names:
            return

        # 兼容已有 MySQL 数据库：老表缺少来源片段列时启动阶段自动补齐。
        engine = get_engine()
        with engine.begin() as connection:
            connection.execute(
                text("ALTER TABLE chat_messages ADD COLUMN retrieved_chunks JSON NULL")
            )

    def _stream_event(self, event_type: str, payload: dict[str, Any]) -> str:
        return json.dumps(
            {"type": event_type, **payload},
            ensure_ascii=False,
        ) + "\n"

    def _retrieve_knowledge_chunks(
        self, question: str, knowledge_bases: list[KnowledgeBaseReference]
    ) -> list[dict[str, Any]]:
        if self.knowledge_base_service is None or not knowledge_bases:
            return []
        try:
            return self.knowledge_base_service.retrieve_relevant_chunks(
                question, knowledge_bases
            )
        except Exception:
            # 知识库检索失败不影响基础聊天，避免一次 Qdrant/Embedding 波动阻断问答。
            return []

    def _log_chat_request(
        self,
        *,
        request_id: str,
        request_mode: str,
        session_id: str,
        original_question: str,
        rewritten_question: str,
        rewrite_fallback_reason: str | None,
        history: list[dict[str, str]],
        system_prompt: str,
        llm_payload: dict[str, Any],
        knowledge_bases: list[KnowledgeBaseReference],
        retrieved_chunks: list[dict[str, Any]],
    ) -> None:
        question_was_rewritten = rewritten_question != original_question
        log_history = self._mask_ai_outputs_in_history(history)
        log_llm_payload = {
            **llm_payload,
            "history": log_history,
        }
        log_payload = {
            "request_id": request_id,
            "request_mode": request_mode,
            "session_id": session_id,
            "model": config.DASHSCOPE_DEFAULT_MODEL,
            "original_question": original_question,
            "rewritten_question": rewritten_question,
            "question_was_rewritten": question_was_rewritten,
            "rewrite_status": "fallback" if rewrite_fallback_reason else "success",
            "rewrite_fallback_reason": rewrite_fallback_reason,
            "rewrite_payload": {
                "question": original_question,
                "history": log_history,
            },
            "history": log_history,
            "system_prompt": system_prompt,
            "llm_payload": log_llm_payload,
            "knowledge_bases": [
                knowledge_base.model_dump() for knowledge_base in knowledge_bases
            ],
            "retrieved_chunk_count": len(retrieved_chunks),
            "retrieved_sources": self._build_retrieved_source_log_items(
                retrieved_chunks
            ),
        }
        logger.info(
            "Chat request payload: %s",
            json.dumps(log_payload, ensure_ascii=False, indent=2, default=str),
        )

    def _mask_ai_outputs_in_history(
        self, history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        return [
            {
                **message,
                "content": (
                    "..."
                    if message.get("role") == "assistant"
                    else message.get("content", "")
                ),
            }
            for message in history
        ]

    def _build_retrieved_source_log_items(
        self, retrieved_chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            {
                "knowledge_base_id": chunk.get("knowledge_base_id"),
                "knowledge_base_name": chunk.get("knowledge_base_name"),
                "document_id": chunk.get("document_id"),
                "original_filename": chunk.get("original_filename"),
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score"),
                "text_length": len(str(chunk.get("text") or "")),
            }
            for chunk in retrieved_chunks
        ]

    def _build_system_prompt(self, retrieved_chunks: list[dict[str, Any]]) -> str:
        if not retrieved_chunks:
            return BASE_SYSTEM_PROMPT

        context_blocks = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("original_filename") or "未知文档"
            knowledge_base_name = chunk.get("knowledge_base_name") or "未知知识库"
            chunk_index = chunk.get("chunk_index")
            context_blocks.append(
                "\n".join(
                    [
                        f"[来源 {index}]",
                        f"知识库：{knowledge_base_name}",
                        f"来源文档：{source}",
                        f"分块序号：{chunk_index}",
                        f"内容：{chunk.get('text') or ''}",
                    ]
                )
            )

        # 把检索结果作为动态系统上下文交给模型，避免污染持久化聊天历史。
        return "\n\n".join(
            [
                BASE_SYSTEM_PROMPT,
                KNOWLEDGE_SYSTEM_INSTRUCTION,
                "以下是从当前会话绑定知识库中检索到的参考内容：",
                "\n\n".join(context_blocks),
            ]
        )

    def _append_reference_section(
        self, answer: str, retrieved_chunks: list[dict[str, Any]]
    ) -> str:
        if not retrieved_chunks:
            return answer

        seen_sources: set[tuple[str, str, str]] = set()
        reference_lines: list[str] = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            knowledge_base_name = str(chunk.get("knowledge_base_name") or "未知知识库")
            filename = str(chunk.get("original_filename") or "未知文档")
            chunk_index = str(chunk.get("chunk_index"))
            source_key = (knowledge_base_name, filename, chunk_index)
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            reference_lines.append(
                f"[来源 {index}] {knowledge_base_name} / {filename} / chunk {chunk_index}"
            )

        if not reference_lines:
            return answer
        return f"{answer.rstrip()}\n\n参考来源：\n" + "\n".join(reference_lines)

    def _to_retrieved_chunk_records(
        self, retrieved_chunks: list[dict[str, Any]]
    ) -> list[KnowledgeBaseRetrievedChunk]:
        records = []
        for chunk in retrieved_chunks:
            records.append(
                KnowledgeBaseRetrievedChunk(
                    knowledge_base_id=str(chunk.get("knowledge_base_id") or ""),
                    knowledge_base_name=str(chunk.get("knowledge_base_name") or ""),
                    document_id=chunk.get("document_id"),
                    original_filename=chunk.get("original_filename"),
                    chunk_index=chunk.get("chunk_index"),
                    score=chunk.get("score"),
                    text=str(chunk.get("text") or ""),
                )
            )
        return records

    def _serialize_retrieved_chunks(
        self, retrieved_chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # JSON 字段只保存响应模型需要的字段，形成当次回答的来源快照。
        return [
            chunk.model_dump()
            for chunk in self._to_retrieved_chunk_records(retrieved_chunks)
        ]

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
        self,
        chat_session: ChatSession,
        message_count: int,
        knowledge_bases: list[KnowledgeBaseReference],
    ) -> SessionSummary:
        return SessionSummary(
            id=chat_session.id,
            title=chat_session.title,
            created_at=to_iso_string(chat_session.created_at),
            updated_at=to_iso_string(chat_session.updated_at),
            message_count=message_count,
            knowledge_bases=knowledge_bases,
        )

    def _message_record_from_model(
        self,
        message: ChatMessage,
        retrieved_chunks: list[dict[str, Any]] | None = None,
    ) -> MessageRecord:
        stored_retrieved_chunks = retrieved_chunks
        if stored_retrieved_chunks is None and message.role == "assistant":
            stored_retrieved_chunks = message.retrieved_chunks or []

        return MessageRecord(
            role=message.role,
            content=message.content,
            created_at=to_iso_string(message.created_at),
            retrieved_chunks=self._to_retrieved_chunk_records(
                stored_retrieved_chunks or []
            ),
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

    def _build_session_summary(
        self, session: Session, chat_session: ChatSession, message_count: int | None = None
    ) -> SessionSummary:
        if message_count is None:
            message_count = self._count_messages(session, chat_session.id)
        knowledge_bases = self._list_knowledge_bases_for_sessions(session, [chat_session.id]).get(
            chat_session.id, []
        )
        return self._session_summary_from_model(chat_session, message_count, knowledge_bases)

    def _normalize_knowledge_base_ids(
        self, knowledge_base_ids: list[str] | None
    ) -> list[str]:
        normalized_ids: list[str] = []
        seen_ids: set[str] = set()
        for knowledge_base_id in knowledge_base_ids or []:
            normalized_id = knowledge_base_id.strip()
            if normalized_id in seen_ids:
                continue
            seen_ids.add(normalized_id)
            normalized_ids.append(normalized_id)
        return normalized_ids

    def _validate_knowledge_base_ids(
        self, session: Session, knowledge_base_ids: list[str]
    ) -> None:
        if not knowledge_base_ids:
            return
        existing_ids = set(
            session.scalars(
                select(KnowledgeBase.id).where(KnowledgeBase.id.in_(knowledge_base_ids))
            ).all()
        )
        missing_ids = [knowledge_base_id for knowledge_base_id in knowledge_base_ids if knowledge_base_id not in existing_ids]
        if missing_ids:
            raise HTTPException(status_code=400, detail="存在无效的知识库 ID")

    def _replace_session_knowledge_base_links(
        self,
        session: Session,
        session_id: str,
        knowledge_base_ids: list[str],
        created_at: datetime,
    ) -> None:
        session.execute(
            delete(SessionKnowledgeBase).where(SessionKnowledgeBase.session_id == session_id)
        )
        session.add_all(
            [
                SessionKnowledgeBase(
                    session_id=session_id,
                    knowledge_base_id=knowledge_base_id,
                    sort_order=index,
                    created_at=created_at,
                )
                for index, knowledge_base_id in enumerate(knowledge_base_ids)
            ]
        )

    def _list_knowledge_bases_for_sessions(
        self, session: Session, session_ids: list[str]
    ) -> dict[str, list[KnowledgeBaseReference]]:
        if not session_ids:
            return {}

        rows = session.execute(
            select(
                SessionKnowledgeBase.session_id,
                KnowledgeBase.id,
                KnowledgeBase.name,
            )
            .join(KnowledgeBase, KnowledgeBase.id == SessionKnowledgeBase.knowledge_base_id)
            .where(SessionKnowledgeBase.session_id.in_(session_ids))
            .order_by(
                SessionKnowledgeBase.session_id.asc(),
                SessionKnowledgeBase.sort_order.asc(),
            )
        ).all()

        knowledge_base_map: dict[str, list[KnowledgeBaseReference]] = defaultdict(list)
        for session_id, knowledge_base_id, knowledge_base_name in rows:
            knowledge_base_map[session_id].append(
                KnowledgeBaseReference(id=knowledge_base_id, name=knowledge_base_name)
            )
        return dict(knowledge_base_map)

    def _get_session_or_404(self, session: Session, session_id: str) -> ChatSession:
        chat_session = session.get(ChatSession, session_id)
        if chat_session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return chat_session
