import os

from dotenv import load_dotenv

load_dotenv()


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_optional_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default

# 阿里百炼模型配置。
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
DASHSCOPE_DEFAULT_MODEL = os.getenv("DASHSCOPE_DEFAULT_MODEL", "qwen-plus")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or DASHSCOPE_BASE_URL

# Rerank 用于在向量召回后做二阶段精排。默认关闭，避免额外调用成本。
RERANK_ENABLED = _get_bool_env("RERANK_ENABLED", False)
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3-vl-rerank")
RERANK_BASE_URL = os.getenv(
    "RERANK_BASE_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
)
RERANK_CANDIDATE_TOP_K = _get_int_env("RERANK_CANDIDATE_TOP_K", 10)
RERANK_TOP_N = _get_int_env("RERANK_TOP_N", 5)
RERANK_TIMEOUT_SECONDS = _get_float_env("RERANK_TIMEOUT_SECONDS", 15)
RERANK_SCORE_THRESHOLD = _get_optional_float_env("RERANK_SCORE_THRESHOLD")
RERANK_INSTRUCT = os.getenv(
    "RERANK_INSTRUCT",
    "Given a web search query, retrieve relevant passages that answer the query.",
)

# 混合检索默认同时执行向量召回和本地 BM25 召回，再通过 RRF 做融合。
HYBRID_RETRIEVAL_ENABLED = _get_bool_env("HYBRID_RETRIEVAL_ENABLED", True)
HYBRID_SPARSE_TOP_K = _get_int_env("HYBRID_SPARSE_TOP_K", 10)
HYBRID_RRF_K = _get_int_env("HYBRID_RRF_K", 60)
HYBRID_BM25_K1 = _get_float_env("HYBRID_BM25_K1", 1.5)
HYBRID_BM25_B = _get_float_env("HYBRID_BM25_B", 0.75)

# MySQL 用于持久化聊天会话、消息历史和知识库元数据。
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# Qdrant 和文档存储用于知识库文档的向量化入库。
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_PREFIX = os.getenv("QDRANT_COLLECTION_PREFIX", "kb")
DOCUMENT_STORAGE_ROOT = os.getenv(
    "DOCUMENT_STORAGE_ROOT", "backend/storage/knowledge_bases"
)
MAX_DOCUMENT_SIZE_BYTES = int(os.getenv("MAX_DOCUMENT_SIZE_BYTES", str(20 * 1024 * 1024)))

# FastAPI 服务配置。
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
BACKEND_CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("BACKEND_CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

# 会话记忆配置：摘要记忆 + 最近窗口。
CHAT_MEMORY_RECENT_TURNS = _get_int_env("CHAT_MEMORY_RECENT_TURNS", 4)
CHAT_MEMORY_REWRITE_RECENT_TURNS = _get_int_env(
    "CHAT_MEMORY_REWRITE_RECENT_TURNS", CHAT_MEMORY_RECENT_TURNS
)
CHAT_MEMORY_SUMMARIZE_AFTER_MESSAGES = _get_int_env(
    "CHAT_MEMORY_SUMMARIZE_AFTER_MESSAGES", 12
)
CHAT_MEMORY_SUMMARY_MAX_CHARS = _get_int_env(
    "CHAT_MEMORY_SUMMARY_MAX_CHARS", 2000
)
