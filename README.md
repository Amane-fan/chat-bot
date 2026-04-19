# LangChain Learning

一个基于 LangChain、FastAPI、Vue 和 Redis 的多会话聊天机器人示例项目。

项目采用前后端分离架构：

- 后端使用 FastAPI 提供会话管理和聊天接口
- 前端使用 Vue + Vite 提供单页聊天界面
- Redis 用于保存会话元数据和消息历史
- LLM 通过 `langchain_openai.ChatOpenAI` 接入阿里百炼兼容接口

## 功能特性

- 多会话管理
- 会话创建、重命名、删除
- 会话消息历史读取
- Redis 持久化聊天记录
- 新会话首条消息自动生成标题
- 前后端分离，便于后续扩展鉴权、流式输出、RAG 等能力

## 目录结构

```text
.
├── backend/
│   ├── config.py
│   ├── main.py
│   ├── schemas.py
│   └── services/
│       └── chat_service.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.vue
│       ├── api.js
│       ├── main.js
│       └── style.css
├── .env.example
├── requirements.txt
└── README.md
```

## 技术栈

### 后端

- Python
- FastAPI
- LangChain
- Redis
- Uvicorn

### 前端

- Vue 3
- Vite

## 环境要求

- Python 3.10+
- Node.js 18+
- Redis 6+

## 环境变量

后端环境变量示例见 [`.env.example`](/home/amane/project/langchain-learning/.env.example:1)。

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_DEFAULT_MODEL=qwen-plus

REDIS_URL=redis://:your_redis_password@localhost:6379/0
REDIS_CHAT_HISTORY_PREFIX=chat
REDIS_CHAT_HISTORY_LIMIT=40

API_HOST=0.0.0.0
API_PORT=8000
BACKEND_CORS_ORIGINS=http://localhost:5173
```

前端环境变量示例见 [`frontend/.env.example`](/home/amane/project/langchain-learning/frontend/.env.example:1)。

```env
VITE_API_BASE_URL=http://localhost:8000/api
```

## 安装依赖

### 1. 后端依赖

建议使用虚拟环境。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 前端依赖

```bash
cd frontend
npm install
```

## 启动方式

### 1. 启动 Redis

确保本地 Redis 已启动，并且 `REDIS_URL` 配置正确。

例如本地默认端口：

```bash
redis-server
```

如果 Redis 需要密码，请在 `.env` 中配置：

```env
REDIS_URL=redis://:your_password@localhost:6379/0
```

### 2. 启动后端

在项目根目录执行：

```bash
uvicorn backend.main:app --reload
```

默认地址：

```text
http://localhost:8000
```

健康检查接口：

```text
GET /api/health
```

### 3. 启动前端

```bash
cd frontend
npm run dev
```

默认地址：

```text
http://localhost:5173
```

## 后端接口

### 健康检查

- `GET /api/health`

### 会话管理

- `GET /api/sessions`
- `POST /api/sessions`
- `PATCH /api/sessions/{session_id}`
- `DELETE /api/sessions/{session_id}`

### 消息管理

- `GET /api/sessions/{session_id}/messages`
- `POST /api/sessions/{session_id}/messages`

## Redis 存储设计

当前实现使用了 3 类 key：

- `chat:sessions`
  保存会话 id 的有序集合，按最近更新时间排序
- `chat:session:{session_id}:meta`
  保存单个会话的标题、创建时间、更新时间
- `chat:session:{session_id}:messages`
  保存该会话的消息列表

其中 `chat` 是默认前缀，可通过 `REDIS_CHAT_HISTORY_PREFIX` 修改。

## 核心代码说明

- [backend/main.py](/home/amane/project/langchain-learning/backend/main.py:1)
  FastAPI 应用入口，定义路由和启动逻辑
- [backend/services/chat_service.py](/home/amane/project/langchain-learning/backend/services/chat_service.py:27)
  封装 Redis 和模型调用，是主要业务逻辑所在
- [backend/schemas.py](/home/amane/project/langchain-learning/backend/schemas.py:1)
  定义请求和响应的数据结构
- [backend/config.py](/home/amane/project/langchain-learning/backend/config.py:1)
  读取环境变量和服务配置

## 开发建议

- 如果你要接入用户体系，建议为会话增加 `user_id`
- 如果你要支持长对话，建议增加摘要记忆，而不是只保留最近消息
- 如果你要优化体验，建议把回复改成流式输出
- 如果你要上线部署，建议把前端 API 地址和后端 CORS 做环境区分

## 常见问题

### `.env` 不在 `backend/` 目录下也能加载吗

可以。当前配置文件使用 `python-dotenv` 加载环境变量，只要启动时能按查找规则找到项目根目录的 `.env`，通常就可以生效。

不过更稳的做法是显式指定 `.env` 路径，避免换启动目录后找不到配置文件。

### 为什么聊天记录会保存在 Redis

因为 Redis 适合保存会话列表和按顺序追加的消息记录，读写简单，开发成本低，适合作为当前这个项目的会话存储层。

## 后续可扩展方向

- 流式输出
- 用户登录与鉴权
- 长期记忆和摘要
- 接入工具调用
- RAG 知识库问答
- Docker 部署

