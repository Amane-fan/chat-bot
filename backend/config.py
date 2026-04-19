import os

from dotenv import load_dotenv

load_dotenv()

# 阿里百炼模型配置。
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
DASHSCOPE_DEFAULT_MODEL = os.getenv("DASHSCOPE_DEFAULT_MODEL", "qwen-plus")

# Redis 用于持久化会话元数据和消息历史。
REDIS_URL = os.getenv("REDIS_URL")
REDIS_CHAT_HISTORY_PREFIX = os.getenv("REDIS_CHAT_HISTORY_PREFIX", "chat")
REDIS_CHAT_HISTORY_LIMIT = int(os.getenv("REDIS_CHAT_HISTORY_LIMIT", "40"))

# MySQL 用于知识库元数据管理。
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# FastAPI 服务配置。
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
BACKEND_CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("BACKEND_CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
