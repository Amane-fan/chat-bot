import json
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.db import Base
from backend.models import ChatMessage, ChatSession, ChatSessionMemory
from backend.services.chat_service import ChatService, utc_now


class StaticInvokeChain:
    def __init__(self, response: str) -> None:
        self.response = response
        self.last_payload = None

    def invoke(self, payload):
        self.last_payload = payload
        return self.response

    def stream(self, payload):
        self.last_payload = payload
        yield self.response


class InspectableChatService(ChatService):
    def __init__(self) -> None:
        super().__init__()
        self.retrieval_question = None
        self.summary_inputs = []

    def _rewrite_question(
        self,
        *,
        request_id: str,
        session_id: str,
        original_question: str,
        history: list[dict[str, str]],
        memory_summary: str | None,
    ) -> tuple[str, str | None]:
        return "改写后的检索问题", None

    def _retrieve_knowledge_chunks(self, question, knowledge_bases):
        self.retrieval_question = question
        return []

    def _summarize_messages(self, existing_summary, messages):
        self.summary_inputs.append((existing_summary, messages))
        return f"摘要:{messages[0]['content']}"

    def _log_chat_request(self, **kwargs) -> None:
        self.last_log_payload = kwargs


class SendMessageChatService(ChatService):
    def __init__(self) -> None:
        super().__init__()
        self.chain = StaticInvokeChain("干净的回答正文")
        self.memory_refresh_calls = []

    def _prepare_chat_request(
        self,
        *,
        session_id: str,
        original_question: str,
        request_mode: str,
    ) -> dict:
        return {
            "retrieved_chunks": [
                {
                    "knowledge_base_id": "kb-1",
                    "knowledge_base_name": "知识库",
                    "document_id": "doc-1",
                    "original_filename": "doc.md",
                    "chunk_index": 0,
                    "score": 0.9,
                    "text": "命中片段",
                }
            ],
            "llm_payload": {
                "system_prompt": "system",
                "question": original_question,
                "history": [],
            },
        }

    def _refresh_session_memory_best_effort(self, session_id: str) -> None:
        self.memory_refresh_calls.append(session_id)


class ChatServiceMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine, autoflush=False, expire_on_commit=False
        )
        self.get_session_factory_patcher = patch(
            "backend.services.chat_service.get_session_factory",
            return_value=self.session_factory,
        )
        self.get_session_factory_patcher.start()
        self.original_config = {
            "CHAT_MEMORY_RECENT_TURNS": config.CHAT_MEMORY_RECENT_TURNS,
            "CHAT_MEMORY_REWRITE_RECENT_TURNS": config.CHAT_MEMORY_REWRITE_RECENT_TURNS,
            "CHAT_MEMORY_SUMMARIZE_AFTER_MESSAGES": config.CHAT_MEMORY_SUMMARIZE_AFTER_MESSAGES,
            "CHAT_MEMORY_SUMMARY_MAX_CHARS": config.CHAT_MEMORY_SUMMARY_MAX_CHARS,
        }

    def tearDown(self) -> None:
        self.get_session_factory_patcher.stop()
        for key, value in self.original_config.items():
            setattr(config, key, value)
        self.engine.dispose()

    def test_prepare_chat_request_uses_rewritten_question_only_for_retrieval(self) -> None:
        config.CHAT_MEMORY_RECENT_TURNS = 4
        config.CHAT_MEMORY_REWRITE_RECENT_TURNS = 4
        config.CHAT_MEMORY_SUMMARIZE_AFTER_MESSAGES = 20
        service = InspectableChatService()
        session_id = self._seed_session(
            [
                ("user", "第一问"),
                ("assistant", "第一答"),
                ("user", "第二问"),
                ("assistant", "第二答"),
            ]
        )

        prepared = service._prepare_chat_request(
            session_id=session_id,
            original_question="原始追问",
            request_mode="sync",
        )

        self.assertEqual("改写后的检索问题", service.retrieval_question)
        self.assertEqual("原始追问", prepared["llm_payload"]["question"])
        self.assertEqual("改写后的检索问题", prepared["retrieval_question"])
        self.assertEqual(4, len(prepared["history"]))
        self.assertIsNone(prepared["memory_summary"])

    def test_prepare_chat_request_persists_summary_and_keeps_recent_turns(self) -> None:
        config.CHAT_MEMORY_RECENT_TURNS = 2
        config.CHAT_MEMORY_REWRITE_RECENT_TURNS = 2
        config.CHAT_MEMORY_SUMMARIZE_AFTER_MESSAGES = 4
        config.CHAT_MEMORY_SUMMARY_MAX_CHARS = 200
        service = InspectableChatService()
        session_id = self._seed_session(
            [
                ("user", "第一轮用户"),
                ("assistant", "第一轮助手"),
                ("user", "第二轮用户"),
                ("assistant", "第二轮助手"),
                ("user", "第三轮用户"),
                ("assistant", "第三轮助手"),
            ]
        )

        prepared = service._prepare_chat_request(
            session_id=session_id,
            original_question="继续追问",
            request_mode="sync",
        )

        self.assertEqual("摘要:第一轮用户", prepared["memory_summary"])
        self.assertEqual(
            ["第二轮用户", "第二轮助手", "第三轮用户", "第三轮助手"],
            [item["content"] for item in prepared["history"]],
        )
        with self.session_factory() as session:
            memory = session.get(ChatSessionMemory, session_id)
            self.assertIsNotNone(memory)
            self.assertEqual("摘要:第一轮用户", memory.summary_text)
            self.assertEqual(2, memory.summarized_message_count)

    def test_send_message_keeps_assistant_content_clean_and_stores_chunks(self) -> None:
        service = SendMessageChatService()
        session_id = self._seed_session([])

        response = service.send_message(session_id, "你好")

        self.assertEqual("干净的回答正文", response.assistant_message.content)
        self.assertNotIn("参考来源", response.assistant_message.content)
        self.assertEqual(1, len(response.assistant_message.retrieved_chunks))
        self.assertEqual([session_id], service.memory_refresh_calls)

        with self.session_factory() as session:
            assistant_message = session.execute(
                select(ChatMessage)
                .where(
                    ChatMessage.session_id == session_id,
                    ChatMessage.role == "assistant",
                )
                .order_by(ChatMessage.id.desc())
            ).scalar_one()
            self.assertEqual("干净的回答正文", assistant_message.content)
            self.assertEqual("命中片段", assistant_message.retrieved_chunks[0]["text"])

    def test_sanitize_context_message_strips_legacy_reference_suffix(self) -> None:
        service = ChatService()

        sanitized = service._sanitize_context_message(
            "assistant",
            "正文内容\n\n参考来源：\n[来源 1] 知识库 / 文档 / chunk 0\n[来源 2] 知识库 / 文档 / chunk 1",
        )

        self.assertEqual("正文内容", sanitized)

    def test_chat_request_log_only_keeps_rewrite_and_memory_fields(self) -> None:
        service = ChatService()
        history = [
            {"role": "user", "content": "第一句"},
            {"role": "assistant", "content": "第二句"},
        ]

        with self.assertLogs("uvicorn.error", level="INFO") as captured:
            service._log_chat_request(
                request_id="req-1",
                request_mode="sync",
                session_id="session-1",
                original_question="原始问题",
                retrieval_question="改写后的检索问题",
                history=history,
                rewrite_fallback_reason=None,
                memory_summary="这是会话摘要",
                memory_fallback_reason=None,
            )

        payload = json.loads(captured.output[0].split("Chat request payload: ", 1)[1])
        self.assertEqual("改写后的检索问题", payload["retrieval_question"])
        self.assertEqual(history, payload["history"])
        self.assertEqual("这是会话摘要", payload["memory_summary"])
        self.assertTrue(payload["memory_summary_present"])
        self.assertNotIn("system_prompt", payload)
        self.assertNotIn("llm_payload", payload)
        self.assertNotIn("retrieved_sources", payload)

    def _seed_session(self, messages: list[tuple[str, str]]) -> str:
        session_id = "s" * 32
        now = utc_now()
        with self.session_factory() as session:
            chat_session = ChatSession(
                id=session_id,
                title="新对话",
                created_at=now,
                updated_at=now,
            )
            session.add(chat_session)
            for index, (role, content) in enumerate(messages):
                session.add(
                    ChatMessage(
                        session_id=session_id,
                        role=role,
                        content=content,
                        created_at=now.replace(microsecond=index),
                    )
                )
            session.commit()
        return session_id


if __name__ == "__main__":
    unittest.main()
