# LangChain Chatbot

一个基于 FastAPI、Vue、Redis 和 MySQL 的聊天机器人示例项目。

当前项目包含两条独立的数据链路：

- 聊天会话和消息历史继续使用 Redis
- 知识库元数据使用 MySQL
- LLM 通过 `langchain_openai.ChatOpenAI` 接入阿里百炼兼容接口

## 功能概览

- 多会话聊天
- 会话创建、重命名、删除
- Redis 持久化聊天记录
- 知识库创建
- 知识库分页列表
- 前端统一管理会话与知识库

## 目录结构

```text
.
├── backend/
│   ├── config.py
│   ├── db.py
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   ├── sql/
│   │   └── mysql_schema.sql
│   └── services/
│       ├── chat_service.py
│       └── knowledge_base_service.py
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.vue
│       ├── api.js
│       └── style.css
├── .env.example
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.10+
- Node.js 18+
- Redis 6+
- MySQL 8.0+

## 环境变量

参考 [`.env.example`](./.env.example)：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_DEFAULT_MODEL=qwen-plus

REDIS_URL=redis://:your_redis_password@localhost:6379/0
REDIS_CHAT_HISTORY_PREFIX=chat
REDIS_CHAT_HISTORY_LIMIT=40

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=langchain_chatbot

API_HOST=0.0.0.0
API_PORT=8000
BACKEND_CORS_ORIGINS=http://localhost:5173
```

前端环境变量参考 [`frontend/.env.example`](./frontend/.env.example)：

```env
VITE_API_BASE_URL=http://localhost:8000/api
```

## 安装依赖

后端：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

前端：

```bash
cd frontend
npm install
```

## MySQL 建表

先创建数据库：

```sql
CREATE DATABASE IF NOT EXISTS langchain_chatbot
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;
```

然后执行 [backend/sql/mysql_schema.sql](./backend/sql/mysql_schema.sql)：

```bash
mysql -u root -p langchain_chatbot < backend/sql/mysql_schema.sql
```

## 启动项目

启动 Redis。

```bash
redis-server
```

启动后端。

```bash
uvicorn backend.main:app --reload
```

启动前端。

```bash
cd frontend
npm run dev
```

默认地址：

- 后端：`http://localhost:8000`
- 前端：`http://localhost:5173`

## 后端接口

### 健康检查

- `GET /api/health`

### 会话接口

- `GET /api/sessions`
- `POST /api/sessions`
- `PATCH /api/sessions/{session_id}`
- `DELETE /api/sessions/{session_id}`
- `GET /api/sessions/{session_id}/messages`
- `POST /api/sessions/{session_id}/messages`

### 知识库接口

- `POST /api/knowledge-bases`
- `GET /api/knowledge-bases?page=1&page_size=10`

创建知识库请求示例：

```json
{
  "name": "产品文档库",
  "description": "用于存放产品手册和 FAQ",
  "config": {
    "embedding_model": "text-embedding-v1",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "separator": "\\n\\n"
  }
}
```

列表响应示例：

```json
{
  "items": [
    {
      "id": "8f2b2f5c8c0a4f0b8f9c1c4f7c2d9a1e",
      "name": "产品文档库",
      "description": "用于存放产品手册和 FAQ",
      "config": {
        "embedding_model": "text-embedding-v1",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "separator": "\\n\\n"
      },
      "document_count": 0,
      "created_at": "2026-04-19T12:00:00+00:00",
      "updated_at": "2026-04-19T12:00:00+00:00"
    }
  ],
  "page": 1,
  "page_size": 10,
  "total": 1
}
```

## 存储说明

- Redis 继续保存聊天会话元数据和消息历史
- MySQL 目前只保存知识库元数据
- 文档上传、切分、向量化和检索尚未接入，先通过 `config` 和 `document_count` 预留结构

## 后续扩展方向

- 知识库文档上传
- 文本切分与向量化
- 检索增强问答
- 用户体系与权限隔离
- Docker 部署
