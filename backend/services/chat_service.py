import json
import uuid
from datetime import datetime, timezone

import redis
from fastapi import HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from backend import config
from backend.schemas import ChatExchangeResponse, MessageRecord, SessionSummary


def utc_now() -> str:
    """返回统一的 UTC 时间字符串，方便前后端展示和排序。"""

    return datetime.now(timezone.utc).isoformat()


def epoch_now() -> float:
    """Redis 的 sorted set 使用时间戳作为排序分值。"""

    return datetime.now(timezone.utc).timestamp()


class ChatService:
    """封装 Redis 和模型链，供路由层直接调用。"""

    def __init__(self) -> None:
        self.redis_client: redis.Redis | None = None
        self.chain = None

    def startup(self) -> None:
        """应用启动时初始化依赖，避免在模块导入阶段直接失败。"""

        self._validate_settings()
        self.redis_client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
        try:
            self.redis_client.ping()
        except redis.RedisError as exc:
            raise RuntimeError("无法连接 Redis，请检查 REDIS_URL 配置") from exc

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一个智能助手，冷静客观地回答用户的问题。"),
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

        client = self._get_redis()
        session_ids = client.zrevrange(self._sessions_index_key(), 0, -1)
        if not session_ids:
            return []

        pipeline = client.pipeline()
        for session_id in session_ids:
            pipeline.hgetall(self._session_meta_key(session_id))
            pipeline.llen(self._session_messages_key(session_id))
        results = pipeline.execute()

        summaries: list[SessionSummary] = []
        for index in range(0, len(results), 2):
            meta = results[index]
            message_count = results[index + 1]
            if not meta:
                continue
            summaries.append(self._session_summary_from_meta(meta, message_count))
        return summaries

    def create_session(self, title: str | None = None) -> SessionSummary:
        """创建新会话，并写入初始元数据。"""

        client = self._get_redis()
        session_id = uuid.uuid4().hex
        now = utc_now()
        meta = {
            "id": session_id,
            "title": title or "新对话",
            "created_at": now,
            "updated_at": now,
        }
        pipeline = client.pipeline()
        pipeline.hset(self._session_meta_key(session_id), mapping=meta)
        pipeline.zadd(self._sessions_index_key(), {session_id: epoch_now()})
        pipeline.execute()
        return self._session_summary_from_meta(meta, 0)

    def rename_session(self, session_id: str, title: str) -> SessionSummary:
        """修改会话标题，并刷新会话排序时间。"""

        client = self._get_redis()
        meta = self.get_session_meta(session_id)
        meta["title"] = title
        meta["updated_at"] = utc_now()
        pipeline = client.pipeline()
        pipeline.hset(self._session_meta_key(session_id), mapping=meta)
        pipeline.zadd(self._sessions_index_key(), {session_id: epoch_now()})
        pipeline.execute()
        message_count = client.llen(self._session_messages_key(session_id))
        return self._session_summary_from_meta(meta, message_count)

    def delete_session(self, session_id: str) -> None:
        """删除会话元数据、消息记录和索引。"""

        client = self._get_redis()
        self.get_session_meta(session_id)
        pipeline = client.pipeline()
        pipeline.delete(self._session_meta_key(session_id))
        pipeline.delete(self._session_messages_key(session_id))
        pipeline.zrem(self._sessions_index_key(), session_id)
        pipeline.execute()

    def get_session_messages(self, session_id: str) -> list[MessageRecord]:
        """返回指定会话的完整消息历史。"""

        self.get_session_meta(session_id)
        return self._load_messages(session_id)

    def send_message(self, session_id: str, content: str) -> ChatExchangeResponse:
        """执行一次问答，并把用户消息和模型回复写入 Redis。"""

        client = self._get_redis()
        meta = self.get_session_meta(session_id)
        normalized_content = content.strip()
        if not normalized_content:
            raise HTTPException(status_code=400, detail="消息内容不能为空")

        history = self._load_history(session_id)
        try:
            answer = self._get_chain().invoke(
                {"question": normalized_content, "history": history}
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail="模型调用失败") from exc

        # 首次发消息时，自动把标题替换成首句摘要，方便前端列表识别会话内容。
        if meta["title"] == "新对话" and client.llen(self._session_messages_key(session_id)) == 0:
            meta["title"] = self._build_session_title(normalized_content)
            meta["updated_at"] = utc_now()
            pipeline = client.pipeline()
            pipeline.hset(self._session_meta_key(session_id), mapping=meta)
            pipeline.zadd(self._sessions_index_key(), {session_id: epoch_now()})
            pipeline.execute()

        user_message = self._append_message(session_id, "user", normalized_content)
        assistant_message = self._append_message(session_id, "assistant", answer)
        summary = self._session_summary_from_meta(
            self.get_session_meta(session_id),
            client.llen(self._session_messages_key(session_id)),
        )
        return ChatExchangeResponse(
            session=summary,
            user_message=user_message,
            assistant_message=assistant_message,
        )

    def get_session_meta(self, session_id: str) -> dict[str, str]:
        """读取会话元数据，不存在时直接返回 404。"""

        meta = self._get_redis().hgetall(self._session_meta_key(session_id))
        if not meta:
            raise HTTPException(status_code=404, detail="会话不存在")
        return meta

    def _validate_settings(self) -> None:
        """启动前校验关键环境变量，避免请求阶段才暴露配置问题。"""

        missing = []
        if not config.DASHSCOPE_API_KEY:
            missing.append("DASHSCOPE_API_KEY")
        if not config.DASHSCOPE_BASE_URL:
            missing.append("DASHSCOPE_BASE_URL")
        if not config.REDIS_URL:
            missing.append("REDIS_URL")
        if missing:
            raise RuntimeError(f"缺少必要环境变量: {', '.join(missing)}")

    def _get_redis(self) -> redis.Redis:
        if self.redis_client is None:
            raise RuntimeError("Redis 客户端尚未初始化")
        return self.redis_client

    def _get_chain(self):
        if self.chain is None:
            raise RuntimeError("模型链尚未初始化")
        return self.chain

    def _sessions_index_key(self) -> str:
        # sorted set 只保存会话 id，用更新时间排序。
        return f"{config.REDIS_CHAT_HISTORY_PREFIX}:sessions"

    def _session_meta_key(self, session_id: str) -> str:
        # hash 保存会话标题、创建时间和更新时间。
        return f"{config.REDIS_CHAT_HISTORY_PREFIX}:session:{session_id}:meta"

    def _session_messages_key(self, session_id: str) -> str:
        # list 以追加方式保存消息序列，天然适合按时间顺序读取。
        return f"{config.REDIS_CHAT_HISTORY_PREFIX}:session:{session_id}:messages"

    def _build_session_title(self, content: str) -> str:
        normalized = " ".join(content.split())
        if not normalized:
            return "新对话"
        return normalized[:20]

    def _session_summary_from_meta(
        self, meta: dict[str, str], message_count: int
    ) -> SessionSummary:
        return SessionSummary(
            id=meta["id"],
            title=meta["title"],
            created_at=meta["created_at"],
            updated_at=meta["updated_at"],
            message_count=message_count,
        )

    def _load_messages(self, session_id: str) -> list[MessageRecord]:
        items = self._get_redis().lrange(self._session_messages_key(session_id), 0, -1)
        return [MessageRecord(**json.loads(item)) for item in items]

    def _load_history(self, session_id: str) -> list[dict[str, str]]:
        return [
            {"role": message.role, "content": message.content}
            for message in self._load_messages(session_id)
        ]

    def _append_message(self, session_id: str, role: str, content: str) -> MessageRecord:
        # 每次写入消息时顺便刷新更新时间，并裁剪历史条数。
        client = self._get_redis()
        message = MessageRecord(role=role, content=content, created_at=utc_now())
        pipeline = client.pipeline()
        pipeline.rpush(
            self._session_messages_key(session_id),
            json.dumps(message.model_dump(), ensure_ascii=False),
        )
        pipeline.ltrim(
            self._session_messages_key(session_id),
            -config.REDIS_CHAT_HISTORY_LIMIT,
            -1,
        )
        pipeline.hset(
            self._session_meta_key(session_id),
            mapping={"updated_at": utc_now()},
        )
        pipeline.zadd(self._sessions_index_key(), {session_id: epoch_now()})
        pipeline.execute()
        return message
