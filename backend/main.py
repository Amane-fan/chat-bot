from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend import config
from backend.schemas import (
    ChatExchangeResponse,
    ChatRequest,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseDocumentDeleteResponse,
    KnowledgeBaseDocumentListResponse,
    KnowledgeBaseDocumentUploadResponse,
    KnowledgeBaseListResponse,
    KnowledgeBaseReference,
    KnowledgeBaseSummary,
    MessageRecord,
    SessionCreateRequest,
    SessionKnowledgeBaseUpdateRequest,
    SessionRenameRequest,
    SessionSummary,
)
from backend.services.chat_service import ChatService
from backend.services.knowledge_base_service import KnowledgeBaseService

# Service 层负责状态和业务逻辑，路由层只做请求分发和参数校验。
knowledge_base_service = KnowledgeBaseService()
chat_service = ChatService(knowledge_base_service)

app = FastAPI(title="Session Chat & Knowledge Base API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """启动时初始化模型链和 MySQL 依赖。"""

    knowledge_base_service.startup()
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

    return chat_service.create_session(payload.title, payload.knowledge_base_ids)


@app.patch("/api/sessions/{session_id}", response_model=SessionSummary)
def rename_session(
    session_id: str, payload: SessionRenameRequest
) -> SessionSummary:
    """更新会话标题。"""

    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="会话标题不能为空")
    return chat_service.rename_session(session_id, title)


@app.put("/api/sessions/{session_id}/knowledge-bases", response_model=SessionSummary)
def replace_session_knowledge_bases(
    session_id: str, payload: SessionKnowledgeBaseUpdateRequest
) -> SessionSummary:
    """完整替换一个会话绑定的知识库列表。"""

    return chat_service.replace_session_knowledge_bases(
        session_id, payload.knowledge_base_ids
    )


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


@app.post("/api/sessions/{session_id}/messages/stream")
def stream_message(session_id: str, payload: ChatRequest) -> StreamingResponse:
    """流式发送消息并获取模型回复。"""

    return StreamingResponse(
        chat_service.stream_message(session_id, payload.content),
        media_type="application/x-ndjson",
    )


@app.post("/api/knowledge-bases", response_model=KnowledgeBaseSummary, status_code=201)
def create_knowledge_base(
    payload: KnowledgeBaseCreateRequest,
) -> KnowledgeBaseSummary:
    """创建一个新的知识库。"""

    # 具体的名称校验、重复检查和落库逻辑都下沉到 service 层。
    return knowledge_base_service.create_knowledge_base(payload)


@app.get("/api/knowledge-bases", response_model=KnowledgeBaseListResponse)
def list_knowledge_bases(page: int = 1, page_size: int = 10) -> KnowledgeBaseListResponse:
    """按分页返回知识库列表。"""

    # 路由层只负责接收分页参数，实际分页规则由 service 统一控制。
    return knowledge_base_service.list_knowledge_bases(page, page_size)


@app.get("/api/knowledge-bases/options", response_model=list[KnowledgeBaseReference])
def list_knowledge_base_options() -> list[KnowledgeBaseReference]:
    """返回聊天页选择器所需的全量轻量知识库列表。"""

    return knowledge_base_service.list_knowledge_base_options()


@app.get(
    "/api/knowledge-bases/{knowledge_base_id}/documents",
    response_model=KnowledgeBaseDocumentListResponse,
)
def list_knowledge_base_documents(
    knowledge_base_id: str,
) -> KnowledgeBaseDocumentListResponse:
    """返回指定知识库下的文档列表。"""

    return knowledge_base_service.list_knowledge_base_documents(knowledge_base_id)


@app.post(
    "/api/knowledge-bases/{knowledge_base_id}/documents",
    response_model=KnowledgeBaseDocumentUploadResponse,
    status_code=201,
)
def upload_knowledge_base_document(
    knowledge_base_id: str,
    file: UploadFile = File(...),
) -> KnowledgeBaseDocumentUploadResponse:
    """上传一个知识库文档并同步完成切分、向量化和入库。"""

    return knowledge_base_service.upload_knowledge_base_document(knowledge_base_id, file)


@app.delete(
    "/api/knowledge-bases/{knowledge_base_id}/documents/{document_id}",
    response_model=KnowledgeBaseDocumentDeleteResponse,
)
def delete_knowledge_base_document(
    knowledge_base_id: str,
    document_id: str,
) -> KnowledgeBaseDocumentDeleteResponse:
    """删除指定知识库文档，并同步清理本地文件和向量数据。"""

    return knowledge_base_service.delete_knowledge_base_document(
        knowledge_base_id, document_id
    )
