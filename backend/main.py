from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from backend import config
from backend.schemas import (
    ChatExchangeResponse,
    ChatRequest,
    MessageRecord,
    SessionCreateRequest,
    SessionRenameRequest,
    SessionSummary,
)
from backend.services.chat_service import ChatService

# Service 层负责状态和业务逻辑，路由层只做请求分发和参数校验。
chat_service = ChatService()

app = FastAPI(title="Session Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """启动时初始化 Redis 和模型链。"""

    chat_service.startup()


@app.get("/api/health")
def health() -> dict[str, str]:
    """提供一个最简单的健康检查接口。"""

    return {"status": "ok"}


@app.get("/api/sessions", response_model=list[SessionSummary])
def get_sessions() -> list[SessionSummary]:
    """返回最近活跃的会话列表。"""

    return chat_service.list_sessions()


@app.post("/api/sessions", response_model=SessionSummary, status_code=201)
def create_session(payload: SessionCreateRequest) -> SessionSummary:
    """创建新会话。"""

    return chat_service.create_session(payload.title)


@app.patch("/api/sessions/{session_id}", response_model=SessionSummary)
def rename_session(
    session_id: str, payload: SessionRenameRequest
) -> SessionSummary:
    """更新会话标题。"""

    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="会话标题不能为空")
    return chat_service.rename_session(session_id, title)


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> Response:
    """删除指定会话。"""

    chat_service.delete_session(session_id)
    return Response(status_code=204)


@app.get("/api/sessions/{session_id}/messages", response_model=list[MessageRecord])
def get_session_messages(session_id: str) -> list[MessageRecord]:
    """读取指定会话的消息历史。"""

    return chat_service.get_session_messages(session_id)


@app.post("/api/sessions/{session_id}/messages", response_model=ChatExchangeResponse)
def send_message(session_id: str, payload: ChatRequest) -> ChatExchangeResponse:
    """发送消息并获取模型回复。"""

    return chat_service.send_message(session_id, payload.content)
