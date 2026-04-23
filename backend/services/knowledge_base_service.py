import json
import logging
import math
import re
import shutil
import time
import uuid
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import HTTPException, UploadFile
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from backend import config
from backend.db import get_engine, get_session_factory, init_mysql
from backend.models import KnowledgeBase, KnowledgeBaseDocument
from backend.schemas import (
    KnowledgeBaseCreateRequest,
    KnowledgeBaseDocumentDeleteResponse,
    KnowledgeBaseDocumentListResponse,
    KnowledgeBaseDocumentSummary,
    KnowledgeBaseDocumentUploadResponse,
    KnowledgeBaseListResponse,
    KnowledgeBaseReference,
    KnowledgeBaseSummary,
    KnowledgeBaseUpdateRequest,
)

DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_RETRIEVAL_CONTEXT_CHAR_LIMIT = 6000
DEFAULT_RETRIEVAL_SCORE_THRESHOLD = 0.25
DEFAULT_QDRANT_SCROLL_LIMIT = 256
PROCESSING_STATUS = "processing"
READY_STATUS = "ready"
FAILED_STATUS = "failed"
ALLOWED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".pdf"}
DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]
CJK_TOKEN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+|[A-Za-z0-9]+")
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


def utc_now():
    """返回适合写入 MySQL DATETIME 的当前 UTC 时间。"""

    from datetime import datetime, timezone

    # MySQL DATETIME 不带时区，这里统一转成 UTC 的 naive datetime 存储。
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_iso_string(value):
    """把数据库时间转换成带 UTC 时区标记的 ISO 字符串。"""

    from datetime import timezone

    # 对外响应统一补回 UTC 时区，前端展示时就能按标准 ISO 处理。
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


class RecursiveCharacterTextSplitter:
    """最小实现版本，按给定分隔符序列递归切分文本。"""

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str],
    ) -> None:
        """保存切分窗口、重叠长度和分隔符优先级。"""

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text: str) -> list[str]:
        """按递归分隔策略切分文本，并在相邻块之间补重叠上下文。"""

        chunks = self._split_recursive(text, self.separators)
        return self._apply_overlap(chunks)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """按分隔符优先级递归切分，尽量保留自然段落边界。"""

        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return self._split_fixed(text)

        separator = separators[0]
        if separator:
            pieces = [piece.strip() for piece in text.split(separator) if piece.strip()]
            joiner = separator
        else:
            pieces = [character for character in text]
            joiner = ""

        if len(pieces) <= 1:
            return self._split_recursive(text, separators[1:])

        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = piece if not current else f"{current}{joiner}{piece}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(piece) <= self.chunk_size:
                current = piece
            else:
                chunks.extend(self._split_recursive(piece, separators[1:]))
                current = ""

        if current:
            chunks.append(current)
        return chunks

    def _split_fixed(self, text: str) -> list[str]:
        """当没有可用分隔符时，按固定窗口强制切分文本。"""

        step = max(1, self.chunk_size - self.chunk_overlap)
        return [
            text[index : index + self.chunk_size].strip()
            for index in range(0, len(text), step)
            if text[index : index + self.chunk_size].strip()
        ]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """为切分后的块补充前一块尾部文本，减少边界信息丢失。"""

        if not chunks or self.chunk_overlap <= 0:
            return chunks

        overlapped = [chunks[0]]
        for chunk in chunks[1:]:
            prefix = overlapped[-1][-self.chunk_overlap :]
            merged = f"{prefix}{chunk}".strip()
            overlapped.append(merged[-self.chunk_size :])
        return overlapped


class KnowledgeBaseService:
    def __init__(self) -> None:
        """初始化知识库服务运行时依赖和本地文档存储目录。"""

        self.qdrant_client: Any | None = None
        self.storage_root = Path(config.DOCUMENT_STORAGE_ROOT).resolve()
        self.bm25_index_cache: dict[str, dict[str, Any]] = {}

    def startup(self) -> None:
        """启动知识库服务，校验依赖并初始化本地存储和 Qdrant 客户端。"""

        # 启动阶段提前暴露配置或建表问题，避免请求进来后才报错。
        self._validate_settings()
        init_mysql()

        try:
            engine = get_engine()
            with engine.connect():
                pass
            inspector = inspect(engine)
            missing_tables = [
                table_name
                for table_name in ("knowledge_bases", "knowledge_base_documents")
                if not inspector.has_table(table_name)
            ]
            if missing_tables:
                raise RuntimeError(
                    "MySQL 中缺少知识库相关表，请先执行 backend/sql/mysql_schema.sql"
                )
        except SQLAlchemyError as exc:
            raise RuntimeError("无法连接 MySQL，请检查 MYSQL_* 配置") from exc

        self.storage_root.mkdir(parents=True, exist_ok=True)
        qdrant_client_class, _ = self._import_qdrant()
        self.qdrant_client = qdrant_client_class(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
        try:
            self.qdrant_client.get_collections()
        except Exception as exc:
            raise RuntimeError("无法连接 Qdrant，请检查 QDRANT_* 配置") from exc

    def create_knowledge_base(
        self, payload: KnowledgeBaseCreateRequest
    ) -> KnowledgeBaseSummary:
        """创建知识库元数据，并保存创建时确定的切分和向量化配置。"""

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

    def update_knowledge_base(
        self, knowledge_base_id: str, payload: KnowledgeBaseUpdateRequest
    ) -> KnowledgeBaseSummary:
        """编辑知识库基础信息，配置项创建后保持不变。"""

        name = self._normalize_name(payload.name)
        description = self._normalize_description(payload.description)

        session_factory = get_session_factory()
        with session_factory() as session:
            knowledge_base = self._get_knowledge_base_or_404(session, knowledge_base_id)
            knowledge_base.name = name
            knowledge_base.description = description
            knowledge_base.updated_at = utc_now()

            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise HTTPException(status_code=409, detail="知识库名称已存在") from exc

            return self._to_summary(knowledge_base)

    def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        """硬删除知识库，并清理 Qdrant collection 和本地文件目录。"""

        session_factory = get_session_factory()
        with session_factory() as session:
            self._get_knowledge_base_or_404(session, knowledge_base_id)

        try:
            self._delete_knowledge_base_artifacts(knowledge_base_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="删除知识库资源失败") from exc

        with session_factory() as session:
            knowledge_base = self._get_knowledge_base_or_404(session, knowledge_base_id)
            session.delete(knowledge_base)
            session.commit()

    def list_knowledge_bases(self, page: int, page_size: int) -> KnowledgeBaseListResponse:
        """分页查询知识库列表，返回前端列表页需要的摘要信息。"""

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

    def list_knowledge_base_options(self) -> list[KnowledgeBaseReference]:
        """返回会话绑定知识库时使用的轻量选项列表。"""

        session_factory = get_session_factory()
        with session_factory() as session:
            items = session.scalars(
                select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc())
            ).all()
        return [KnowledgeBaseReference(id=item.id, name=item.name) for item in items]

    def list_knowledge_base_documents(
        self, knowledge_base_id: str
    ) -> KnowledgeBaseDocumentListResponse:
        """查询单个知识库下的文档及其处理状态。"""

        session_factory = get_session_factory()
        with session_factory() as session:
            self._get_knowledge_base_or_404(session, knowledge_base_id)
            items = session.scalars(
                select(KnowledgeBaseDocument)
                .where(KnowledgeBaseDocument.knowledge_base_id == knowledge_base_id)
                .order_by(
                    KnowledgeBaseDocument.created_at.desc(),
                    KnowledgeBaseDocument.id.desc(),
                )
            ).all()

        return KnowledgeBaseDocumentListResponse(
            knowledge_base_id=knowledge_base_id,
            items=[self._to_document_summary(item) for item in items],
        )

    def retrieve_relevant_chunks(
        self,
        question: str,
        knowledge_bases: list[KnowledgeBaseReference],
        *,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        context_char_limit: int = DEFAULT_RETRIEVAL_CONTEXT_CHAR_LIMIT,
        score_threshold: float | None = DEFAULT_RETRIEVAL_SCORE_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """按会话绑定知识库检索和问题最相关的文本块。"""

        normalized_question = question.strip()
        if not normalized_question or not knowledge_bases:
            return []

        session_factory = get_session_factory()
        knowledge_base_ids = [item.id for item in knowledge_bases]
        with session_factory() as session:
            items = session.scalars(
                select(KnowledgeBase).where(KnowledgeBase.id.in_(knowledge_base_ids))
            ).all()
            ready_documents = session.scalars(
                select(KnowledgeBaseDocument)
                .where(
                    KnowledgeBaseDocument.knowledge_base_id.in_(knowledge_base_ids),
                    KnowledgeBaseDocument.status == READY_STATUS,
                )
            ).all()

        knowledge_base_map = {item.id: item for item in items}
        ready_document_map: dict[str, list[KnowledgeBaseDocument]] = {}
        for document in ready_documents:
            ready_document_map.setdefault(document.knowledge_base_id, []).append(document)
        candidate_chunks: list[dict[str, Any]] = []
        retrieval_log_items: list[dict[str, Any]] = []
        started_at = time.perf_counter()
        rerank_enabled = config.RERANK_ENABLED
        query_top_k = top_k
        if rerank_enabled:
            query_top_k = max(
                top_k,
                self._normalize_positive_int(config.RERANK_CANDIDATE_TOP_K, top_k),
            )
        sparse_top_k = max(
            query_top_k,
            self._normalize_positive_int(config.HYBRID_SPARSE_TOP_K, query_top_k),
        )
        fused_top_k = max(query_top_k, sparse_top_k)

        for knowledge_base_ref in knowledge_bases:
            knowledge_base = knowledge_base_map.get(knowledge_base_ref.id)
            if knowledge_base is None:
                retrieval_log_items.append(
                    self._build_retrieval_log_item(
                        knowledge_base_ref,
                        status="missing",
                    )
                )
                continue

            ready_document_items = ready_document_map.get(knowledge_base.id, [])
            ready_document_ids = [document.id for document in ready_document_items]
            # document_count 是冗余汇总字段，检索前以文档表实时状态为准。
            if not ready_document_ids:
                retrieval_log_items.append(
                    self._build_retrieval_log_item(
                        knowledge_base_ref,
                        status="skipped_no_ready_documents",
                    )
                )
                continue

            try:
                knowledge_base_config = dict(knowledge_base.config or {})
                index_status = self._ensure_retrieval_index_ready(
                    knowledge_base.id,
                    knowledge_base_config,
                    ready_document_items,
                )
                effective_score_threshold = self._normalize_score_threshold(
                    knowledge_base_config.get("retrieval_score_threshold"),
                    score_threshold,
                )
                query_vector = self._embed_query(
                    knowledge_base_config, normalized_question
                )
                dense_hits = self._query_knowledge_base_chunks(
                    knowledge_base.id,
                    query_vector,
                    query_top_k,
                    effective_score_threshold,
                    ready_document_ids,
                )
            except Exception:
                # 检索增强是可选上下文，单个知识库失败时保留普通聊天能力。
                retrieval_log_items.append(
                    self._build_retrieval_log_item(
                        knowledge_base_ref,
                        status="failed",
                    )
                )
                continue

            sparse_hits: list[dict[str, Any]] = []
            sparse_cache_status = "disabled"
            sparse_status = "skipped"
            if config.HYBRID_RETRIEVAL_ENABLED:
                try:
                    sparse_hits, sparse_cache_status = self._query_sparse_knowledge_base_chunks(
                        knowledge_base.id,
                        knowledge_base_ref,
                        normalized_question,
                        ready_document_ids,
                        sparse_top_k,
                    )
                    sparse_status = "searched"
                except Exception as exc:
                    sparse_status = "failed"
                    sparse_cache_status = f"failed:{type(exc).__name__}"

            dense_candidates = self._build_candidate_chunks(
                knowledge_base, knowledge_base_ref, dense_hits
            )
            fused_candidates = self._fuse_retrieval_candidates(
                dense_candidates,
                sparse_hits,
                fused_top_k,
            )

            retrieval_log_items.append(
                self._build_retrieval_log_item(
                    knowledge_base_ref,
                    status="searched",
                    score_threshold=effective_score_threshold,
                    dense_hits=dense_hits,
                    sparse_hits=sparse_hits,
                    fused_candidate_count=len(fused_candidates),
                    sparse_cache_status=sparse_cache_status,
                    sparse_status=sparse_status,
                    index_status=index_status,
                )
            )

            candidate_chunks.extend(fused_candidates)

        selected_candidates, rerank_log = self._select_retrieval_candidates(
            normalized_question, candidate_chunks, top_k
        )
        retrieved_chunks = self._candidates_to_retrieved_chunks(
            selected_candidates, context_char_limit
        )
        self._log_retrieval_result(
            normalized_question,
            retrieval_log_items,
            started_at,
            retrieved_chunks,
            rerank_log=rerank_log,
        )
        return retrieved_chunks

    def upload_knowledge_base_document(
        self, knowledge_base_id: str, file: UploadFile
    ) -> KnowledgeBaseDocumentUploadResponse:
        """上传文档并完成解析、切分、向量化和 Qdrant 入库。"""

        original_filename = self._normalize_upload_filename(file.filename)
        self._validate_document_extension(original_filename)
        file_bytes = file.file.read()
        file_size = len(file_bytes)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="上传文件不能为空")
        if file_size > config.MAX_DOCUMENT_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小不能超过 {config.MAX_DOCUMENT_SIZE_BYTES // (1024 * 1024)}MB",
            )

        document_id = uuid.uuid4().hex
        stored_filename = self._build_stored_filename(document_id, original_filename)
        storage_path = str(self._build_document_path(knowledge_base_id, stored_filename))
        now = utc_now()

        session_factory = get_session_factory()
        with session_factory() as session:
            knowledge_base = self._get_knowledge_base_or_404(session, knowledge_base_id)
            replaced_document = session.scalar(
                select(KnowledgeBaseDocument).where(
                    KnowledgeBaseDocument.knowledge_base_id == knowledge_base_id,
                    KnowledgeBaseDocument.original_filename == original_filename,
                )
            )
            replaced_snapshot = (
                self._document_snapshot(replaced_document) if replaced_document is not None else None
            )

            if replaced_document is not None:
                session.delete(replaced_document)

            document = KnowledgeBaseDocument(
                id=document_id,
                knowledge_base_id=knowledge_base_id,
                original_filename=original_filename,
                stored_filename=stored_filename,
                content_type=file.content_type,
                file_size=file_size,
                storage_path=storage_path,
                status=PROCESSING_STATUS,
                chunk_count=0,
                error_message=None,
                created_at=now,
                updated_at=now,
            )
            session.add(document)

            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise HTTPException(
                    status_code=409, detail="同名文档正在处理中，请稍后重试"
                ) from exc

            knowledge_base_config = dict(knowledge_base.config or {})

        try:
            if replaced_snapshot is not None:
                self._delete_document_artifacts(replaced_snapshot)

            self._save_document_bytes(knowledge_base_id, stored_filename, file_bytes)
            text = self._extract_document_text(original_filename, file_bytes)
            chunks = self._split_document_text(knowledge_base_config, text)
            vectors = self._embed_document_chunks(knowledge_base_config, chunks)
            collection_name = self._collection_name(knowledge_base_id)
            self._ensure_collection(collection_name, len(vectors[0]))
            self._upsert_document_chunks(
                collection_name=collection_name,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                original_filename=original_filename,
                chunks=chunks,
                vectors=vectors,
            )
        except HTTPException as exc:
            self._safe_delete_document_vectors(knowledge_base_id, document_id)
            self._mark_document_failed(knowledge_base_id, document_id, exc.detail)
            raise
        except Exception as exc:
            self._safe_delete_document_vectors(knowledge_base_id, document_id)
            self._mark_document_failed(knowledge_base_id, document_id, str(exc))
            raise HTTPException(status_code=502, detail="文档处理失败") from exc

        with session_factory() as session:
            knowledge_base = self._get_knowledge_base_or_404(session, knowledge_base_id)
            document = self._get_document_or_404(session, document_id)
            document.status = READY_STATUS
            document.chunk_count = len(vectors)
            document.error_message = None
            document.updated_at = utc_now()
            # 先 flush 文档状态，否则 autoflush=False 时下面的 count 看不到 ready 状态。
            session.flush()
            knowledge_base.document_count = self._count_ready_documents(session, knowledge_base_id)
            knowledge_base.updated_at = utc_now()
            session.commit()

            return KnowledgeBaseDocumentUploadResponse(
                document=self._to_document_summary(document),
                knowledge_base=self._to_summary(knowledge_base),
            )

    def delete_knowledge_base_document(
        self, knowledge_base_id: str, document_id: str
    ) -> KnowledgeBaseDocumentDeleteResponse:
        """删除知识库文档，并同步清理向量和本地文件。"""

        session_factory = get_session_factory()
        with session_factory() as session:
            knowledge_base = self._get_knowledge_base_or_404(session, knowledge_base_id)
            document = self._get_knowledge_base_document_or_404(
                session, knowledge_base_id, document_id
            )
            document_summary = self._to_document_summary(document)
            document_snapshot = self._document_snapshot(document)

        try:
            # 先清理检索侧资源，避免 MySQL 删除成功后 Qdrant 中残留可检索片段。
            self._delete_document_artifacts(document_snapshot)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="删除文档资源失败") from exc

        with session_factory() as session:
            knowledge_base = self._get_knowledge_base_or_404(session, knowledge_base_id)
            document = self._get_knowledge_base_document_or_404(
                session, knowledge_base_id, document_id
            )
            session.delete(document)
            # 删除记录后再统计 ready 文档数，保证列表中的 document_count 同步变化。
            session.flush()
            knowledge_base.document_count = self._count_ready_documents(
                session, knowledge_base_id
            )
            knowledge_base.updated_at = utc_now()
            session.commit()

            return KnowledgeBaseDocumentDeleteResponse(
                document=document_summary,
                knowledge_base=self._to_summary(knowledge_base),
            )

    def _validate_settings(self) -> None:
        """校验知识库能力启动所需的数据库、模型和向量库配置。"""

        missing = []
        if not config.MYSQL_HOST:
            missing.append("MYSQL_HOST")
        if not config.MYSQL_USER:
            missing.append("MYSQL_USER")
        if config.MYSQL_PASSWORD is None:
            missing.append("MYSQL_PASSWORD")
        if not config.MYSQL_DATABASE:
            missing.append("MYSQL_DATABASE")
        if not config.DASHSCOPE_API_KEY:
            missing.append("DASHSCOPE_API_KEY")
        if not config.EMBEDDING_BASE_URL:
            missing.append("EMBEDDING_BASE_URL 或 DASHSCOPE_BASE_URL")
        if not config.QDRANT_URL:
            missing.append("QDRANT_URL")
        if missing:
            raise RuntimeError(f"缺少必要环境变量: {', '.join(missing)}")

    def _normalize_name(self, name: str) -> str:
        """标准化知识库名称，并执行非空和长度校验。"""

        # 把连续空白压成一个空格，避免“看起来一样”的名称实际不同。
        normalized = " ".join(name.split())
        if not normalized:
            raise HTTPException(status_code=400, detail="知识库名称不能为空")
        if len(normalized) > 80:
            raise HTTPException(status_code=400, detail="知识库名称不能超过 80 个字符")
        return normalized

    def _normalize_description(self, description: str | None) -> str | None:
        """标准化知识库描述，空字符串统一转换为 None。"""

        if description is None:
            return None
        normalized = description.strip()
        # 空描述统一视为未填写，数据库里存 NULL。
        if not normalized:
            return None
        return normalized

    def _normalize_upload_filename(self, filename: str | None) -> str:
        """提取并校验上传文件名，避免目录穿越和空文件名。"""

        normalized = Path(filename or "").name.strip()
        if not normalized:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        if len(normalized) > 255:
            raise HTTPException(status_code=400, detail="文件名不能超过 255 个字符")
        return normalized

    def _validate_document_extension(self, filename: str) -> None:
        """限制可入库文档类型，只允许当前解析链路支持的扩展名。"""

        if Path(filename).suffix.lower() not in ALLOWED_DOCUMENT_EXTENSIONS:
            raise HTTPException(status_code=400, detail="仅支持上传 TXT、MD、PDF 文件")

    def _build_stored_filename(self, document_id: str, original_filename: str) -> str:
        """生成带文档 ID 前缀的安全落盘文件名。"""

        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", original_filename).strip("._")
        if not sanitized:
            sanitized = f"document{Path(original_filename).suffix.lower()}"
        return f"{document_id}_{sanitized}"

    def _build_document_path(self, knowledge_base_id: str, stored_filename: str) -> Path:
        """生成文档在本地存储目录中的完整路径。"""

        return self.storage_root / knowledge_base_id / stored_filename

    def _save_document_bytes(
        self, knowledge_base_id: str, stored_filename: str, file_bytes: bytes
    ) -> None:
        """把上传文件内容写入知识库对应的本地目录。"""

        document_dir = self.storage_root / knowledge_base_id
        document_dir.mkdir(parents=True, exist_ok=True)
        document_path = document_dir / stored_filename
        document_path.write_bytes(file_bytes)

    def _extract_document_text(self, filename: str, file_bytes: bytes) -> str:
        """按文件类型抽取可切分和向量化的纯文本内容。"""

        suffix = Path(filename).suffix.lower()
        if suffix in {".txt", ".md"}:
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(
                    status_code=400, detail="TXT/MD 文件必须使用 UTF-8 编码"
                ) from exc
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ModuleNotFoundError as exc:
                raise HTTPException(
                    status_code=500, detail="服务端缺少 pypdf 依赖，暂时无法解析 PDF 文件"
                ) from exc
            reader = PdfReader(BytesIO(file_bytes))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            raise HTTPException(status_code=400, detail="不支持的文件类型")

        if not text.strip():
            raise HTTPException(status_code=400, detail="文档解析后没有可用文本内容")
        return text

    def _split_document_text(self, knowledge_base_config: dict[str, Any], text: str) -> list[str]:
        """按知识库配置切分文档文本，返回可入向量库的文本块。"""

        chunk_size = int(knowledge_base_config.get("chunk_size") or DEFAULT_CHUNK_SIZE)
        chunk_overlap = int(
            knowledge_base_config.get("chunk_overlap") or DEFAULT_CHUNK_OVERLAP
        )
        if chunk_overlap >= chunk_size:
            raise HTTPException(status_code=400, detail="chunk_overlap 必须小于 chunk_size")

        separators: list[str] = []
        separator = self._decode_separator(knowledge_base_config.get("separator"))
        if separator:
            separators.append(separator)
        for default_separator in DEFAULT_SEPARATORS:
            if default_separator not in separators:
                separators.append(default_separator)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
        chunks = [
            chunk.strip()
            for chunk in splitter.split_text(text)
            if chunk and chunk.strip()
        ]
        if not chunks:
            raise HTTPException(status_code=400, detail="文档切分后没有可入库的文本块")
        return chunks

    def _decode_separator(self, separator: Any) -> str | None:
        """把前端传入的转义分隔符还原成真实字符。"""

        if not isinstance(separator, str) or not separator:
            return None
        try:
            return bytes(separator, "utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            return separator

    def _embed_document_chunks(
        self, knowledge_base_config: dict[str, Any], chunks: list[str]
    ) -> list[list[float]]:
        """调用 embedding 模型把文档块批量转换为向量。"""

        embeddings = self._build_embeddings(knowledge_base_config)
        try:
            return embeddings.embed_documents(chunks)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"文档向量化失败: {exc}") from exc

    def _embed_query(
        self, knowledge_base_config: dict[str, Any], question: str
    ) -> list[float]:
        """使用知识库对应 embedding 配置生成查询向量。"""

        embeddings = self._build_embeddings(knowledge_base_config)
        return embeddings.embed_query(question)

    def _build_embeddings(self, knowledge_base_config: dict[str, Any]) -> OpenAIEmbeddings:
        """构造兼容 OpenAI 协议的 embedding 客户端。"""

        embedding_model = (
            knowledge_base_config.get("embedding_model") or DEFAULT_EMBEDDING_MODEL
        )
        return OpenAIEmbeddings(
            model=embedding_model,
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.EMBEDDING_BASE_URL,
            chunk_size=10,                   # 关键：限制每次最多 10 条
            check_embedding_ctx_length=False, # 对兼容 OpenAI 的非 OpenAI 提供方更稳妥
        )

    def _ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """确保 Qdrant collection 存在，不存在时按向量维度创建。"""

        client = self._get_qdrant_client()
        _, qdrant_models = self._import_qdrant()
        if self._collection_exists(collection_name):
            return
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail="创建 Qdrant collection 失败") from exc

    def _collection_exists(self, collection_name: str) -> bool:
        """兼容不同 qdrant-client 版本检查 collection 是否存在。"""

        client = self._get_qdrant_client()
        try:
            if hasattr(client, "collection_exists"):
                return bool(client.collection_exists(collection_name=collection_name))
            client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def _query_knowledge_base_chunks(
        self,
        knowledge_base_id: str,
        query_vector: list[float],
        top_k: int,
        score_threshold: float | None,
        ready_document_ids: list[str],
    ) -> list[Any]:
        """在指定知识库 collection 中按查询向量召回候选文本块。"""

        collection_name = self._collection_name(knowledge_base_id)
        if not self._collection_exists(collection_name):
            return []

        # 本项目锁定的 qdrant-client 使用 query_points 作为向量检索入口。
        _, qdrant_models = self._import_qdrant()
        query_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="document_id",
                    match=qdrant_models.MatchAny(any=ready_document_ids),
                )
            ]
        )
        response = self._get_qdrant_client().query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )
        return list(response.points or [])

    def _ensure_retrieval_index_ready(
        self,
        knowledge_base_id: str,
        knowledge_base_config: dict[str, Any],
        ready_documents: list[KnowledgeBaseDocument],
    ) -> str:
        """在检索前确保知识库向量索引存在，必要时从本地文档重建。"""

        collection_name = self._collection_name(knowledge_base_id)
        if self._collection_exists(collection_name):
            return "existing"
        if not ready_documents:
            return "missing_no_ready_documents"

        self._rebuild_knowledge_base_vectors_from_documents(
            knowledge_base_id,
            knowledge_base_config,
            ready_documents,
        )
        return "rebuilt_from_local_documents"

    def _rebuild_knowledge_base_vectors_from_documents(
        self,
        knowledge_base_id: str,
        knowledge_base_config: dict[str, Any],
        ready_documents: list[KnowledgeBaseDocument],
    ) -> None:
        """当 Qdrant 索引缺失时，基于本地文档和数据库元信息重建向量。"""

        collection_name = self._collection_name(knowledge_base_id)
        rebuilt_any_document = False
        for document in ready_documents:
            document_path = Path(document.storage_path)
            if not document_path.exists():
                logger.warning(
                    "知识库索引重建时缺少本地文件: knowledge_base_id=%s document_id=%s path=%s",
                    knowledge_base_id,
                    document.id,
                    document.storage_path,
                )
                continue

            file_bytes = document_path.read_bytes()
            text = self._extract_document_text(document.original_filename, file_bytes)
            chunks = self._split_document_text(knowledge_base_config, text)
            vectors = self._embed_document_chunks(knowledge_base_config, chunks)
            if not vectors:
                continue
            self._ensure_collection(collection_name, len(vectors[0]))
            self._upsert_document_chunks(
                collection_name=collection_name,
                knowledge_base_id=knowledge_base_id,
                document_id=document.id,
                original_filename=document.original_filename,
                chunks=chunks,
                vectors=vectors,
            )
            rebuilt_any_document = True

        self._invalidate_sparse_index_cache(knowledge_base_id)
        if not rebuilt_any_document:
            raise RuntimeError("知识库索引缺失，且无法从本地文档重建")

    def _query_sparse_knowledge_base_chunks(
        self,
        knowledge_base_id: str,
        knowledge_base_ref: KnowledgeBaseReference,
        question: str,
        ready_document_ids: list[str],
        top_k: int,
    ) -> tuple[list[dict[str, Any]], str]:
        """基于知识库 chunk 文本构建 BM25 索引并执行稀疏召回。"""

        bm25_index, cache_status = self._get_or_build_bm25_index(
            knowledge_base_id, ready_document_ids
        )
        return (
            self._search_bm25_index(
                bm25_index, knowledge_base_id, knowledge_base_ref, question, top_k
            ),
            cache_status,
        )

    def _get_or_build_bm25_index(
        self, knowledge_base_id: str, ready_document_ids: list[str]
    ) -> tuple[dict[str, Any], str]:
        """按知识库获取 BM25 缓存，若文档集合变化则重建。"""

        ready_document_signature = self._build_ready_document_signature(ready_document_ids)
        cached_index = self.bm25_index_cache.get(knowledge_base_id)
        if (
            cached_index is not None
            and cached_index.get("ready_document_signature") == ready_document_signature
        ):
            return cached_index, "hit"

        bm25_index = self._build_bm25_index(knowledge_base_id, ready_document_ids)
        self.bm25_index_cache[knowledge_base_id] = bm25_index
        return bm25_index, "rebuilt"

    def _build_bm25_index(
        self, knowledge_base_id: str, ready_document_ids: list[str]
    ) -> dict[str, Any]:
        """从 Qdrant payload 拉取全文块并构建 BM25 统计量。"""

        chunk_payloads = self._scroll_knowledge_base_chunk_payloads(
            knowledge_base_id, ready_document_ids
        )
        term_frequencies: list[Counter[str]] = []
        document_frequencies: Counter[str] = Counter()
        document_lengths: list[int] = []
        chunk_records: list[dict[str, Any]] = []

        for payload in chunk_payloads:
            text = str(payload.get("text") or "").strip()
            if not text:
                continue
            tokens = self._tokenize_for_sparse_retrieval(text)
            if not tokens:
                continue
            term_frequency = Counter(tokens)
            term_frequencies.append(term_frequency)
            document_lengths.append(sum(term_frequency.values()))
            document_frequencies.update(term_frequency.keys())
            chunk_records.append(
                {
                    "document_id": payload.get("document_id"),
                    "original_filename": payload.get("original_filename"),
                    "chunk_index": payload.get("chunk_index"),
                    "text": text,
                }
            )

        document_count = len(chunk_records)
        average_document_length = (
            sum(document_lengths) / document_count if document_count else 0.0
        )
        inverse_document_frequency = {
            token: math.log(1 + ((document_count - frequency + 0.5) / (frequency + 0.5)))
            for token, frequency in document_frequencies.items()
        }
        return {
            "ready_document_signature": self._build_ready_document_signature(
                ready_document_ids
            ),
            "chunks": chunk_records,
            "term_frequencies": term_frequencies,
            "document_lengths": document_lengths,
            "average_document_length": average_document_length,
            "inverse_document_frequency": inverse_document_frequency,
        }

    def _scroll_knowledge_base_chunk_payloads(
        self, knowledge_base_id: str, ready_document_ids: list[str]
    ) -> list[dict[str, Any]]:
        """滚动读取知识库中 ready 文档的全部 chunk payload。"""

        collection_name = self._collection_name(knowledge_base_id)
        if not self._collection_exists(collection_name):
            return []

        _, qdrant_models = self._import_qdrant()
        scroll_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="document_id",
                    match=qdrant_models.MatchAny(any=ready_document_ids),
                )
            ]
        )

        payloads: list[dict[str, Any]] = []
        next_offset: Any | None = None
        client = self._get_qdrant_client()
        while True:
            response = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=DEFAULT_QDRANT_SCROLL_LIMIT,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )
            if isinstance(response, tuple):
                points, next_offset = response
            else:
                points = getattr(response, "points", []) or []
                next_offset = getattr(response, "next_page_offset", None)

            for point in points:
                payload = point.payload or {}
                if payload:
                    payloads.append(payload)

            if next_offset is None:
                break
        return payloads

    def _search_bm25_index(
        self,
        bm25_index: dict[str, Any],
        knowledge_base_id: str,
        knowledge_base_ref: KnowledgeBaseReference,
        question: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """在构建好的 BM25 索引上执行查询并返回候选块。"""

        query_tokens = self._tokenize_for_sparse_retrieval(question)
        if not query_tokens:
            return []

        chunks = bm25_index.get("chunks") or []
        if not chunks:
            return []

        average_document_length = float(bm25_index.get("average_document_length") or 0.0)
        inverse_document_frequency = dict(bm25_index.get("inverse_document_frequency") or {})
        document_lengths = list(bm25_index.get("document_lengths") or [])
        term_frequencies = list(bm25_index.get("term_frequencies") or [])
        query_term_frequencies = Counter(query_tokens)
        scored_candidates: list[dict[str, Any]] = []
        k1 = max(0.01, float(config.HYBRID_BM25_K1))
        b = min(max(float(config.HYBRID_BM25_B), 0.0), 1.0)

        for index, chunk in enumerate(chunks):
            if index >= len(term_frequencies) or index >= len(document_lengths):
                continue
            term_frequency = term_frequencies[index]
            document_length = document_lengths[index]
            score = 0.0
            for token, query_term_frequency in query_term_frequencies.items():
                token_frequency = term_frequency.get(token, 0)
                if token_frequency <= 0:
                    continue
                inverse_document_frequency_value = inverse_document_frequency.get(token, 0.0)
                denominator = token_frequency + k1 * (
                    1 - b + b * (document_length / average_document_length)
                ) if average_document_length > 0 else token_frequency + k1
                if denominator <= 0:
                    continue
                score += (
                    inverse_document_frequency_value
                    * ((token_frequency * (k1 + 1)) / denominator)
                    * query_term_frequency
                )
            if score <= 0:
                continue
            scored_candidates.append(
                {
                    "knowledge_base_id": knowledge_base_id,
                    "knowledge_base_name": knowledge_base_ref.name,
                    "document_id": chunk.get("document_id"),
                    "original_filename": chunk.get("original_filename"),
                    "chunk_index": chunk.get("chunk_index"),
                    "score": score,
                    "sparse_score": score,
                    "text": chunk.get("text"),
                }
            )

        scored_candidates.sort(
            key=lambda item: (
                float(item.get("score") or 0.0),
                -int(item.get("chunk_index") or 0),
            ),
            reverse=True,
        )
        return scored_candidates[:top_k]

    def _tokenize_for_sparse_retrieval(self, text: str) -> list[str]:
        """对英文和中文文本做轻量切词，兼顾关键词与短语匹配。"""

        tokens: list[str] = []
        for match in CJK_TOKEN_PATTERN.finditer(text or ""):
            segment = match.group(0)
            if not segment:
                continue
            if segment.isascii():
                normalized = segment.lower().strip()
                if normalized:
                    tokens.append(normalized)
                continue

            characters = [character for character in segment.strip() if character.strip()]
            tokens.extend(characters)
            tokens.extend(
                f"{characters[index]}{characters[index + 1]}"
                for index in range(len(characters) - 1)
            )
        return [token for token in tokens if token]

    def _fuse_retrieval_candidates(
        self,
        dense_candidates: list[dict[str, Any]],
        sparse_candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """使用 Reciprocal Rank Fusion 融合向量召回和 BM25 召回结果。"""

        if not config.HYBRID_RETRIEVAL_ENABLED:
            return dense_candidates[:top_k]

        fused_candidates: dict[str, dict[str, Any]] = {}
        rrf_k = max(1, self._normalize_positive_int(config.HYBRID_RRF_K, 60))
        ranked_candidate_lists = [dense_candidates, sparse_candidates]
        for candidates in ranked_candidate_lists:
            for rank, candidate in enumerate(candidates, start=1):
                candidate_key = self._candidate_key(candidate)
                fused_candidate = fused_candidates.setdefault(
                    candidate_key,
                    {
                        **candidate,
                        "score": 0.0,
                    },
                )
                fused_candidate["score"] = float(fused_candidate.get("score") or 0.0) + (
                    1.0 / (rrf_k + rank)
                )
                for field_name in ("vector_score", "sparse_score"):
                    if field_name in candidate and candidate.get(field_name) is not None:
                        fused_candidate[field_name] = candidate.get(field_name)
                if not fused_candidate.get("text"):
                    fused_candidate["text"] = candidate.get("text")
                if fused_candidate.get("document_id") is None:
                    fused_candidate["document_id"] = candidate.get("document_id")
                if fused_candidate.get("original_filename") is None:
                    fused_candidate["original_filename"] = candidate.get("original_filename")
                if fused_candidate.get("chunk_index") is None:
                    fused_candidate["chunk_index"] = candidate.get("chunk_index")

        sorted_candidates = sorted(
            fused_candidates.values(),
            key=lambda item: (
                float(item.get("score") or 0.0),
                float(item.get("vector_score") or 0.0),
                float(item.get("sparse_score") or 0.0),
            ),
            reverse=True,
        )
        return sorted_candidates[:top_k]

    def _candidate_key(self, candidate: dict[str, Any]) -> str:
        """为跨召回路由的同一 chunk 生成稳定键。"""

        return "::".join(
            [
                str(candidate.get("knowledge_base_id") or ""),
                str(candidate.get("document_id") or ""),
                str(candidate.get("chunk_index") or ""),
                str(candidate.get("original_filename") or ""),
            ]
        )

    def _build_ready_document_signature(self, ready_document_ids: list[str]) -> str:
        """把 ready 文档集合转换成稳定签名，用于缓存比对。"""

        return "|".join(sorted(str(document_id) for document_id in ready_document_ids))

    def _invalidate_sparse_index_cache(self, knowledge_base_id: str) -> None:
        """使指定知识库的 BM25 缓存失效。"""

        self.bm25_index_cache.pop(knowledge_base_id, None)

    def _build_candidate_chunks(
        self,
        knowledge_base: KnowledgeBase,
        knowledge_base_ref: KnowledgeBaseReference,
        hits: list[Any],
    ) -> list[dict[str, Any]]:
        """把 Qdrant 命中结果转换成后续精排和上下文组装使用的候选结构。"""

        candidates: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            text = str(payload.get("text") or "").strip()
            if not text:
                continue
            vector_score = getattr(hit, "score", None)
            candidates.append(
                {
                    "knowledge_base_id": knowledge_base.id,
                    "knowledge_base_name": knowledge_base_ref.name,
                    "document_id": payload.get("document_id"),
                    "original_filename": payload.get("original_filename"),
                    "chunk_index": payload.get("chunk_index"),
                    "score": vector_score,
                    "vector_score": vector_score,
                    "text": text,
                }
            )
        return candidates

    def _select_retrieval_candidates(
        self, question: str, candidates: list[dict[str, Any]], top_k: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """根据 rerank 开关选择最终候选，并返回用于检索日志的精排状态。"""

        rerank_log = {
            "enabled": config.RERANK_ENABLED,
            "status": "disabled",
            "candidate_count": len(candidates),
            "selected_count": len(candidates),
        }
        if not config.RERANK_ENABLED:
            return candidates, rerank_log

        rerank_top_n = self._normalize_positive_int(config.RERANK_TOP_N, top_k)
        if not candidates:
            return [], {
                **rerank_log,
                "status": "skipped_no_candidates",
                "selected_count": 0,
            }

        started_at = time.perf_counter()
        try:
            selected_candidates = self._rerank_candidate_chunks(
                question, candidates, rerank_top_n
            )
            rerank_log.update(
                {
                    "status": "succeeded",
                    "selected_count": len(selected_candidates),
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                }
            )
            return selected_candidates, rerank_log
        except Exception as exc:
            fallback_candidates = self._limit_vector_candidates(candidates, top_k)
            rerank_log.update(
                {
                    "status": "failed",
                    "selected_count": len(fallback_candidates),
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            return fallback_candidates, rerank_log

    def _rerank_candidate_chunks(
        self, question: str, candidates: list[dict[str, Any]], top_n: int
    ) -> list[dict[str, Any]]:
        """调用百炼 rerank 对候选片段重新排序，并把精排分数写回候选。"""

        request_top_n = min(top_n, len(candidates))
        parameters: dict[str, Any] = {
            "return_documents": False,
            "top_n": request_top_n,
        }
        instruct = str(config.RERANK_INSTRUCT or "").strip()
        if instruct:
            parameters["instruct"] = instruct

        response_payload = self._call_rerank_api(
            {
                "model": config.RERANK_MODEL,
                "input": {
                    "query": {"text": question},
                    "documents": [
                        {"text": str(candidate.get("text") or "")}
                        for candidate in candidates
                    ],
                },
                "parameters": parameters,
            }
        )
        results = (response_payload.get("output") or {}).get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("rerank response has no results")

        selected_candidates: list[dict[str, Any]] = []
        seen_indexes: set[int] = set()
        for result in results:
            if not isinstance(result, dict):
                continue
            index = result.get("index")
            if not isinstance(index, int) or index in seen_indexes:
                continue
            if index < 0 or index >= len(candidates):
                continue
            try:
                rerank_score = float(result.get("relevance_score"))
            except (TypeError, ValueError):
                continue
            if (
                config.RERANK_SCORE_THRESHOLD is not None
                and rerank_score < config.RERANK_SCORE_THRESHOLD
            ):
                continue
            seen_indexes.add(index)
            selected_candidates.append(
                {
                    **candidates[index],
                    "score": rerank_score,
                    "rerank_score": rerank_score,
                }
            )

        if not selected_candidates:
            raise ValueError("rerank response has no usable results")
        return selected_candidates

    def _call_rerank_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        """通过 HTTP 调用阿里百炼 rerank 接口并返回 JSON 响应。"""

        if not config.DASHSCOPE_API_KEY:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY，无法调用 rerank")

        request = Request(
            config.RERANK_BASE_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {config.DASHSCOPE_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=config.RERANK_TIMEOUT_SECONDS) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"rerank API returned HTTP {exc.code}: {error_body[:300]}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"rerank API request failed: {exc.reason}") from exc

        response_payload = json.loads(response_body)
        if not isinstance(response_payload, dict):
            raise ValueError("rerank response is not a JSON object")
        if response_payload.get("code"):
            raise RuntimeError(
                f"rerank API returned {response_payload.get('code')}: "
                f"{response_payload.get('message')}"
            )
        return response_payload

    def _limit_vector_candidates(
        self, candidates: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """rerank 不可用时，按每个知识库保留原向量召回的前 top_k 个候选。"""

        limited_candidates: list[dict[str, Any]] = []
        knowledge_base_counts: dict[str, int] = {}
        for candidate in candidates:
            knowledge_base_id = str(candidate.get("knowledge_base_id") or "")
            count = knowledge_base_counts.get(knowledge_base_id, 0)
            if count >= top_k:
                continue
            limited_candidates.append(candidate)
            knowledge_base_counts[knowledge_base_id] = count + 1
        return limited_candidates

    def _candidates_to_retrieved_chunks(
        self, candidates: list[dict[str, Any]], context_char_limit: int
    ) -> list[dict[str, Any]]:
        """把最终候选转成聊天层来源片段，并按上下文总长度截断。"""

        retrieved_chunks: list[dict[str, Any]] = []
        used_chars = 0
        for candidate in candidates:
            text = str(candidate.get("text") or "").strip()
            if not text:
                continue

            remaining_chars = context_char_limit - used_chars
            if remaining_chars <= 0:
                break

            truncated_text = text[:remaining_chars]
            used_chars += len(truncated_text)
            retrieved_chunks.append(
                {
                    "knowledge_base_id": candidate.get("knowledge_base_id"),
                    "knowledge_base_name": candidate.get("knowledge_base_name"),
                    "document_id": candidate.get("document_id"),
                    "original_filename": candidate.get("original_filename"),
                    "chunk_index": candidate.get("chunk_index"),
                    "score": candidate.get("score"),
                    "text": truncated_text,
                }
            )

            if used_chars >= context_char_limit:
                break
        return retrieved_chunks

    def _normalize_positive_int(self, value: Any, default_value: int) -> int:
        """把配置值规范化为正整数，非法值回退到默认值。"""

        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return default_value
        if normalized <= 0:
            return default_value
        return normalized

    def _normalize_score_threshold(
        self, value: Any, default_score_threshold: float | None
    ) -> float | None:
        """规范化向量检索分数阈值，非正数表示不启用阈值。"""

        if value is None or value == "":
            return default_score_threshold
        try:
            score_threshold = float(value)
        except (TypeError, ValueError):
            return default_score_threshold
        if score_threshold <= 0:
            return None
        return min(score_threshold, 1.0)

    def _build_retrieval_log_item(
        self,
        knowledge_base: KnowledgeBaseReference,
        *,
        status: str,
        score_threshold: float | None = None,
        hits: list[Any] | None = None,
        dense_hits: list[Any] | None = None,
        sparse_hits: list[dict[str, Any]] | None = None,
        fused_candidate_count: int | None = None,
        sparse_cache_status: str | None = None,
        sparse_status: str | None = None,
        index_status: str | None = None,
    ) -> dict[str, Any]:
        """构造单个知识库的检索日志条目，便于排查召回情况。"""

        effective_dense_hits = dense_hits if dense_hits is not None else hits or []
        effective_sparse_hits = sparse_hits or []
        return {
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "status": status,
            "score_threshold": score_threshold,
            "dense_hit_count": len(effective_dense_hits),
            "sparse_hit_count": len(effective_sparse_hits),
            "fused_candidate_count": fused_candidate_count,
            "sparse_cache_status": sparse_cache_status,
            "sparse_status": sparse_status,
            "index_status": index_status,
            "hit_count": len(effective_dense_hits),
            "hits": [
                {
                    "score": getattr(hit, "score", None),
                    "document_id": (hit.payload or {}).get("document_id"),
                    "original_filename": (hit.payload or {}).get("original_filename"),
                    "chunk_index": (hit.payload or {}).get("chunk_index"),
                }
                for hit in effective_dense_hits
            ],
            "sparse_hits": [
                {
                    "score": hit.get("score"),
                    "document_id": hit.get("document_id"),
                    "original_filename": hit.get("original_filename"),
                    "chunk_index": hit.get("chunk_index"),
                }
                for hit in effective_sparse_hits
            ],
        }

    def _log_retrieval_result(
        self,
        question: str,
        retrieval_log_items: list[dict[str, Any]],
        started_at: float,
        retrieved_chunks: list[dict[str, Any]],
        *,
        rerank_log: dict[str, Any] | None = None,
    ) -> None:
        """输出一次知识库检索的聚合日志，包括召回、精排和最终来源数量。"""

        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        log_payload = {
            "question": question,
            "elapsed_ms": elapsed_ms,
            "retrieved_chunk_count": len(retrieved_chunks),
            "rerank": rerank_log,
            "knowledge_bases": retrieval_log_items,
        }
        logger.info(
            "知识库检索结果:\n%s",
            json.dumps(log_payload, ensure_ascii=False, indent=2),
        )

    def _upsert_document_chunks(
        self,
        collection_name: str,
        knowledge_base_id: str,
        document_id: str,
        original_filename: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        """把文档分块向量及其来源信息批量写入 Qdrant。"""

        client = self._get_qdrant_client()
        _, qdrant_models = self._import_qdrant()
        points = [
            qdrant_models.PointStruct(
                id=self._build_chunk_point_id(document_id, index),
                vector=vector,
                payload={
                    "knowledge_base_id": knowledge_base_id,
                    "document_id": document_id,
                    "original_filename": original_filename,
                    "chunk_index": index,
                    "text": chunk,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        try:
            client.upsert(collection_name=collection_name, points=points, wait=True)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"写入 Qdrant 失败: {exc}") from exc

    def _build_chunk_point_id(self, document_id: str, chunk_index: int) -> str:
        """根据文档 ID 和分块序号生成稳定的 Qdrant point ID。"""

        # Qdrant 只接受无符号整数或 UUID 作为 point ID，这里生成稳定 UUID。
        namespace = uuid.UUID(hex=document_id)
        return str(uuid.uuid5(namespace, str(chunk_index)))

    def _delete_document_artifacts(self, document_snapshot: dict[str, str]) -> None:
        """清理单个文档对应的向量和本地文件。"""

        self._delete_document_vectors(
            document_snapshot["knowledge_base_id"], document_snapshot["id"]
        )
        self._delete_local_document_file(document_snapshot["storage_path"])

    def _delete_knowledge_base_artifacts(self, knowledge_base_id: str) -> None:
        """清理知识库对应的 Qdrant collection 和本地目录。"""

        self._delete_knowledge_base_collection(knowledge_base_id)
        self._delete_local_knowledge_base_dir(knowledge_base_id)

    def _delete_knowledge_base_collection(self, knowledge_base_id: str) -> None:
        """删除知识库在 Qdrant 中的 collection。"""

        collection_name = self._collection_name(knowledge_base_id)
        if not self._collection_exists(collection_name):
            return
        self._get_qdrant_client().delete_collection(collection_name=collection_name)

    def _delete_document_vectors(self, knowledge_base_id: str, document_id: str) -> None:
        """按 document_id 删除 Qdrant 中属于该文档的全部向量点。"""

        collection_name = self._collection_name(knowledge_base_id)
        if not self._collection_exists(collection_name):
            return

        _, qdrant_models = self._import_qdrant()
        selector = qdrant_models.FilterSelector(
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchValue(value=document_id),
                    )
                ]
            )
        )
        self._get_qdrant_client().delete(
            collection_name=collection_name,
            points_selector=selector,
            wait=True,
        )

    def _safe_delete_document_vectors(self, knowledge_base_id: str, document_id: str) -> None:
        """尽力清理文档向量，失败时静默返回用于异常回滚路径。"""

        try:
            self._delete_document_vectors(knowledge_base_id, document_id)
        except Exception:
            return

    def _delete_local_document_file(self, storage_path: str) -> None:
        """删除本地存储中的单个文档文件。"""

        path = Path(storage_path)
        if path.exists():
            path.unlink()

    def _delete_local_knowledge_base_dir(self, knowledge_base_id: str) -> None:
        """删除知识库对应的本地文档目录。"""

        path = self.storage_root / knowledge_base_id
        if path.exists():
            shutil.rmtree(path)

    def _mark_document_failed(
        self, knowledge_base_id: str, document_id: str, error_message: str
    ) -> None:
        """把文档处理状态标记为失败，并同步刷新知识库 ready 文档数。"""

        session_factory = get_session_factory()
        with session_factory() as session:
            knowledge_base = self._get_knowledge_base_or_404(session, knowledge_base_id)
            document = self._get_document_or_404(session, document_id)
            document.status = FAILED_STATUS
            document.error_message = self._truncate_error_message(error_message)
            document.updated_at = utc_now()
            # 失败状态也要先落到当前事务，再重新统计 ready 文档数。
            session.flush()
            knowledge_base.document_count = self._count_ready_documents(session, knowledge_base_id)
            knowledge_base.updated_at = utc_now()
            session.commit()

    def rebuild_document_counts(self) -> int:
        """按文档表状态回填知识库 document_count，返回被修正的知识库数量。"""

        session_factory = get_session_factory()
        updated_count = 0
        with session_factory() as session:
            knowledge_bases = session.scalars(select(KnowledgeBase)).all()
            for knowledge_base in knowledge_bases:
                ready_document_count = self._count_ready_documents(session, knowledge_base.id)
                if knowledge_base.document_count == ready_document_count:
                    continue
                knowledge_base.document_count = ready_document_count
                knowledge_base.updated_at = utc_now()
                updated_count += 1
            session.commit()
        return updated_count

    def _truncate_error_message(self, error_message: str) -> str:
        """压缩并截断异常文本，避免过长错误写入数据库。"""

        normalized = " ".join((error_message or "未知错误").split())
        return normalized[:1000]

    def _count_ready_documents(self, session, knowledge_base_id: str) -> int:
        """统计指定知识库中状态为 ready 的文档数量。"""

        return session.scalar(
            select(func.count(KnowledgeBaseDocument.id)).where(
                KnowledgeBaseDocument.knowledge_base_id == knowledge_base_id,
                KnowledgeBaseDocument.status == READY_STATUS,
            )
        ) or 0

    def _collection_name(self, knowledge_base_id: str) -> str:
        """生成知识库对应的 Qdrant collection 名称。"""

        return f"{config.QDRANT_COLLECTION_PREFIX}_{knowledge_base_id}"

    def _get_qdrant_client(self) -> Any:
        """获取已初始化的 Qdrant 客户端。"""

        if self.qdrant_client is None:
            raise RuntimeError("Qdrant 客户端尚未初始化")
        return self.qdrant_client

    def _import_qdrant(self):
        """延迟导入 qdrant-client，并在缺依赖时给出明确错误。"""

        try:
            from qdrant_client import QdrantClient, models as qdrant_models
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 qdrant-client 依赖，请先执行 uv sync") from exc
        return QdrantClient, qdrant_models

    def _get_knowledge_base_or_404(self, session, knowledge_base_id: str) -> KnowledgeBase:
        """按 ID 获取知识库，不存在时抛出 404。"""

        knowledge_base = session.get(KnowledgeBase, knowledge_base_id)
        if knowledge_base is None:
            raise HTTPException(status_code=404, detail="知识库不存在")
        return knowledge_base

    def _get_document_or_404(self, session, document_id: str) -> KnowledgeBaseDocument:
        """按 ID 获取文档记录，不存在时抛出 404。"""

        document = session.get(KnowledgeBaseDocument, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        return document

    def _get_knowledge_base_document_or_404(
        self, session, knowledge_base_id: str, document_id: str
    ) -> KnowledgeBaseDocument:
        """获取指定知识库下的文档，避免跨知识库误操作。"""

        document = self._get_document_or_404(session, document_id)
        if document.knowledge_base_id != knowledge_base_id:
            raise HTTPException(status_code=404, detail="文档不存在")
        return document

    def _document_snapshot(self, document: KnowledgeBaseDocument) -> dict[str, str]:
        """提取删除资源所需的文档字段快照，避免会话关闭后访问 ORM 对象。"""

        return {
            "id": document.id,
            "knowledge_base_id": document.knowledge_base_id,
            "storage_path": document.storage_path,
        }

    def _to_summary(self, knowledge_base: KnowledgeBase) -> KnowledgeBaseSummary:
        """把知识库 ORM 对象转换为 API 摘要模型。"""

        return KnowledgeBaseSummary(
            id=knowledge_base.id,
            name=knowledge_base.name,
            description=knowledge_base.description,
            config=knowledge_base.config,
            document_count=knowledge_base.document_count,
            created_at=to_iso_string(knowledge_base.created_at),
            updated_at=to_iso_string(knowledge_base.updated_at),
        )

    def _to_document_summary(
        self, document: KnowledgeBaseDocument
    ) -> KnowledgeBaseDocumentSummary:
        """把文档 ORM 对象转换为 API 摘要模型。"""

        return KnowledgeBaseDocumentSummary(
            id=document.id,
            knowledge_base_id=document.knowledge_base_id,
            original_filename=document.original_filename,
            stored_filename=document.stored_filename,
            content_type=document.content_type,
            file_size=document.file_size,
            status=document.status,
            chunk_count=document.chunk_count,
            error_message=document.error_message,
            created_at=to_iso_string(document.created_at),
            updated_at=to_iso_string(document.updated_at),
        )
