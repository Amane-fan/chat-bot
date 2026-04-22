import json
import logging
import re
import shutil
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

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
PROCESSING_STATUS = "processing"
READY_STATUS = "ready"
FAILED_STATUS = "failed"
ALLOWED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".pdf"}
DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


def utc_now():
    from datetime import datetime, timezone

    # MySQL DATETIME 不带时区，这里统一转成 UTC 的 naive datetime 存储。
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_iso_string(value):
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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text: str) -> list[str]:
        chunks = self._split_recursive(text, self.separators)
        return self._apply_overlap(chunks)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
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
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [
            text[index : index + self.chunk_size].strip()
            for index in range(0, len(text), step)
            if text[index : index + self.chunk_size].strip()
        ]

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
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
        self.qdrant_client: Any | None = None
        self.storage_root = Path(config.DOCUMENT_STORAGE_ROOT).resolve()

    def startup(self) -> None:
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
        session_factory = get_session_factory()
        with session_factory() as session:
            items = session.scalars(
                select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc())
            ).all()
        return [KnowledgeBaseReference(id=item.id, name=item.name) for item in items]

    def list_knowledge_base_documents(
        self, knowledge_base_id: str
    ) -> KnowledgeBaseDocumentListResponse:
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
            ready_document_rows = session.execute(
                select(
                    KnowledgeBaseDocument.knowledge_base_id,
                    KnowledgeBaseDocument.id,
                )
                .where(
                    KnowledgeBaseDocument.knowledge_base_id.in_(knowledge_base_ids),
                    KnowledgeBaseDocument.status == READY_STATUS,
                )
            ).all()

        knowledge_base_map = {item.id: item for item in items}
        ready_document_id_map: dict[str, list[str]] = {}
        for knowledge_base_id, document_id in ready_document_rows:
            ready_document_id_map.setdefault(knowledge_base_id, []).append(document_id)
        retrieved_chunks: list[dict[str, Any]] = []
        used_chars = 0
        retrieval_log_items: list[dict[str, Any]] = []
        started_at = time.perf_counter()

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

            ready_document_ids = ready_document_id_map.get(knowledge_base.id, [])
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
                effective_score_threshold = self._normalize_score_threshold(
                    knowledge_base_config.get("retrieval_score_threshold"),
                    score_threshold,
                )
                query_vector = self._embed_query(
                    knowledge_base_config, normalized_question
                )
                hits = self._query_knowledge_base_chunks(
                    knowledge_base.id,
                    query_vector,
                    top_k,
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

            retrieval_log_items.append(
                self._build_retrieval_log_item(
                    knowledge_base_ref,
                    status="searched",
                    score_threshold=effective_score_threshold,
                    hits=hits,
                )
            )

            for hit in hits:
                payload = hit.payload or {}
                text = str(payload.get("text") or "").strip()
                if not text:
                    continue

                remaining_chars = context_char_limit - used_chars
                if remaining_chars <= 0:
                    self._log_retrieval_result(
                        normalized_question, retrieval_log_items, started_at, retrieved_chunks
                    )
                    return retrieved_chunks

                truncated_text = text[:remaining_chars]
                used_chars += len(truncated_text)
                retrieved_chunks.append(
                    {
                        "knowledge_base_id": knowledge_base.id,
                        "knowledge_base_name": knowledge_base_ref.name,
                        "document_id": payload.get("document_id"),
                        "original_filename": payload.get("original_filename"),
                        "chunk_index": payload.get("chunk_index"),
                        "score": getattr(hit, "score", None),
                        "text": truncated_text,
                    }
                )

                if used_chars >= context_char_limit:
                    self._log_retrieval_result(
                        normalized_question, retrieval_log_items, started_at, retrieved_chunks
                    )
                    return retrieved_chunks

        self._log_retrieval_result(
            normalized_question, retrieval_log_items, started_at, retrieved_chunks
        )
        return retrieved_chunks

    def upload_knowledge_base_document(
        self, knowledge_base_id: str, file: UploadFile
    ) -> KnowledgeBaseDocumentUploadResponse:
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

    def _normalize_upload_filename(self, filename: str | None) -> str:
        normalized = Path(filename or "").name.strip()
        if not normalized:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        if len(normalized) > 255:
            raise HTTPException(status_code=400, detail="文件名不能超过 255 个字符")
        return normalized

    def _validate_document_extension(self, filename: str) -> None:
        if Path(filename).suffix.lower() not in ALLOWED_DOCUMENT_EXTENSIONS:
            raise HTTPException(status_code=400, detail="仅支持上传 TXT、MD、PDF 文件")

    def _build_stored_filename(self, document_id: str, original_filename: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", original_filename).strip("._")
        if not sanitized:
            sanitized = f"document{Path(original_filename).suffix.lower()}"
        return f"{document_id}_{sanitized}"

    def _build_document_path(self, knowledge_base_id: str, stored_filename: str) -> Path:
        return self.storage_root / knowledge_base_id / stored_filename

    def _save_document_bytes(
        self, knowledge_base_id: str, stored_filename: str, file_bytes: bytes
    ) -> None:
        document_dir = self.storage_root / knowledge_base_id
        document_dir.mkdir(parents=True, exist_ok=True)
        document_path = document_dir / stored_filename
        document_path.write_bytes(file_bytes)

    def _extract_document_text(self, filename: str, file_bytes: bytes) -> str:
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
        if not isinstance(separator, str) or not separator:
            return None
        try:
            return bytes(separator, "utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            return separator

    def _embed_document_chunks(
        self, knowledge_base_config: dict[str, Any], chunks: list[str]
    ) -> list[list[float]]:
        embeddings = self._build_embeddings(knowledge_base_config)
        try:
            return embeddings.embed_documents(chunks)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"文档向量化失败: {exc}") from exc

    def _embed_query(
        self, knowledge_base_config: dict[str, Any], question: str
    ) -> list[float]:
        embeddings = self._build_embeddings(knowledge_base_config)
        return embeddings.embed_query(question)

    def _build_embeddings(self, knowledge_base_config: dict[str, Any]) -> OpenAIEmbeddings:
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

    def _normalize_score_threshold(
        self, value: Any, default_score_threshold: float | None
    ) -> float | None:
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
    ) -> dict[str, Any]:
        return {
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "status": status,
            "score_threshold": score_threshold,
            "hit_count": len(hits or []),
            "hits": [
                {
                    "score": getattr(hit, "score", None),
                    "document_id": (hit.payload or {}).get("document_id"),
                    "original_filename": (hit.payload or {}).get("original_filename"),
                    "chunk_index": (hit.payload or {}).get("chunk_index"),
                }
                for hit in hits or []
            ],
        }

    def _log_retrieval_result(
        self,
        question: str,
        retrieval_log_items: list[dict[str, Any]],
        started_at: float,
        retrieved_chunks: list[dict[str, Any]],
    ) -> None:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        log_payload = {
            "question": question,
            "elapsed_ms": elapsed_ms,
            "retrieved_chunk_count": len(retrieved_chunks),
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
        # Qdrant 只接受无符号整数或 UUID 作为 point ID，这里生成稳定 UUID。
        namespace = uuid.UUID(hex=document_id)
        return str(uuid.uuid5(namespace, str(chunk_index)))

    def _delete_document_artifacts(self, document_snapshot: dict[str, str]) -> None:
        self._delete_document_vectors(
            document_snapshot["knowledge_base_id"], document_snapshot["id"]
        )
        self._delete_local_document_file(document_snapshot["storage_path"])

    def _delete_knowledge_base_artifacts(self, knowledge_base_id: str) -> None:
        self._delete_knowledge_base_collection(knowledge_base_id)
        self._delete_local_knowledge_base_dir(knowledge_base_id)

    def _delete_knowledge_base_collection(self, knowledge_base_id: str) -> None:
        collection_name = self._collection_name(knowledge_base_id)
        if not self._collection_exists(collection_name):
            return
        self._get_qdrant_client().delete_collection(collection_name=collection_name)

    def _delete_document_vectors(self, knowledge_base_id: str, document_id: str) -> None:
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
        try:
            self._delete_document_vectors(knowledge_base_id, document_id)
        except Exception:
            return

    def _delete_local_document_file(self, storage_path: str) -> None:
        path = Path(storage_path)
        if path.exists():
            path.unlink()

    def _delete_local_knowledge_base_dir(self, knowledge_base_id: str) -> None:
        path = self.storage_root / knowledge_base_id
        if path.exists():
            shutil.rmtree(path)

    def _mark_document_failed(
        self, knowledge_base_id: str, document_id: str, error_message: str
    ) -> None:
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
        normalized = " ".join((error_message or "未知错误").split())
        return normalized[:1000]

    def _count_ready_documents(self, session, knowledge_base_id: str) -> int:
        return session.scalar(
            select(func.count(KnowledgeBaseDocument.id)).where(
                KnowledgeBaseDocument.knowledge_base_id == knowledge_base_id,
                KnowledgeBaseDocument.status == READY_STATUS,
            )
        ) or 0

    def _collection_name(self, knowledge_base_id: str) -> str:
        return f"{config.QDRANT_COLLECTION_PREFIX}_{knowledge_base_id}"

    def _get_qdrant_client(self) -> Any:
        if self.qdrant_client is None:
            raise RuntimeError("Qdrant 客户端尚未初始化")
        return self.qdrant_client

    def _import_qdrant(self):
        try:
            from qdrant_client import QdrantClient, models as qdrant_models
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 qdrant-client 依赖，请先执行 uv sync") from exc
        return QdrantClient, qdrant_models

    def _get_knowledge_base_or_404(self, session, knowledge_base_id: str) -> KnowledgeBase:
        knowledge_base = session.get(KnowledgeBase, knowledge_base_id)
        if knowledge_base is None:
            raise HTTPException(status_code=404, detail="知识库不存在")
        return knowledge_base

    def _get_document_or_404(self, session, document_id: str) -> KnowledgeBaseDocument:
        document = session.get(KnowledgeBaseDocument, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        return document

    def _get_knowledge_base_document_or_404(
        self, session, knowledge_base_id: str, document_id: str
    ) -> KnowledgeBaseDocument:
        document = self._get_document_or_404(session, document_id)
        if document.knowledge_base_id != knowledge_base_id:
            raise HTTPException(status_code=404, detail="文档不存在")
        return document

    def _document_snapshot(self, document: KnowledgeBaseDocument) -> dict[str, str]:
        return {
            "id": document.id,
            "knowledge_base_id": document.knowledge_base_id,
            "storage_path": document.storage_path,
        }

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

    def _to_document_summary(
        self, document: KnowledgeBaseDocument
    ) -> KnowledgeBaseDocumentSummary:
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
