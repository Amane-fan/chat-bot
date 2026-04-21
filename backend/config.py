import os

from dotenv import load_dotenv

load_dotenv()

# 阿里百炼模型配置。
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
DASHSCOPE_DEFAULT_MODEL = os.getenv("DASHSCOPE_DEFAULT_MODEL", "qwen-plus")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or DASHSCOPE_BASE_URL

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
