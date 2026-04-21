# LangChain Chatbot

一个基于 FastAPI、Vue、MySQL、Qdrant 和 LangChain 的聊天机器人示例项目。

当前项目包含三类核心数据：

- 聊天会话和消息历史使用 MySQL
- 知识库元数据和文档状态使用 MySQL
- 文档切分后的向量块使用 Qdrant
- LLM 通过 `langchain_openai.ChatOpenAI` 接入阿里百炼兼容接口

## 功能概览

- 多会话聊天
- 会话创建、重命名、删除
- MySQL 持久化聊天记录
- 知识库创建
- 知识库文档上传
- 按知识库配置进行文档切分与向量化
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
├── .python-version
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```

## 环境要求

- Python 3.10+
- Node.js 18+
- MySQL 8.0+
- Qdrant 1.8+

## 环境变量

参考 [`.env.example`](./.env.example)：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_DEFAULT_MODEL=qwen-plus
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=langchain_chatbot

QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_PREFIX=kb
DOCUMENT_STORAGE_ROOT=backend/storage/knowledge_bases
MAX_DOCUMENT_SIZE_BYTES=20971520

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
uv sync
```

如果本机还没有安装 `uv`，先执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
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

启动后端。

```bash
uv run uvicorn backend.main:app --reload
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
- `GET /api/knowledge-bases/{knowledge_base_id}/documents`
- `POST /api/knowledge-bases/{knowledge_base_id}/documents`
- `DELETE /api/knowledge-bases/{knowledge_base_id}/documents/{document_id}`

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

文档上传接口使用 `multipart/form-data`，字段名固定为 `file`。当前支持 `TXT / MD / PDF`，会按知识库配置中的 `embedding_model`、`chunk_size`、`chunk_overlap` 和 `separator` 同步完成切分和向量化。

## 存储说明

- MySQL 保存聊天会话元数据和消息历史
- MySQL 同时保存知识库元数据和文档处理状态
- 原始文档文件落盘到 `DOCUMENT_STORAGE_ROOT`
- Qdrant 保存文档切分后的向量块，collection 名称格式为 `{QDRANT_COLLECTION_PREFIX}_{knowledge_base_id}`

## 后续扩展方向

- 检索增强问答
- 用户体系与权限隔离
- Docker 部署
