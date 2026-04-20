<script setup>
import { computed, onMounted, ref } from "vue";
import {
  createKnowledgeBase,
  createSession,
  deleteSession,
  getKnowledgeBases,
  getMessages,
  getSessions,
  renameSession,
  sendMessage,
} from "./api";

const sessions = ref([]);
const activeSessionId = ref("");
const messages = ref([]);
const draft = ref("");
const errorMessage = ref("");
const loadingMessages = ref(false);
const loadingSessions = ref(false);
const sending = ref(false);
const editingSessionId = ref("");
const editingTitle = ref("");

const knowledgeBases = ref([]);
const knowledgeBasePage = ref(1);
const knowledgeBasePageSize = ref(5);
const knowledgeBaseTotal = ref(0);
const loadingKnowledgeBases = ref(false);
const creatingKnowledgeBase = ref(false);
const knowledgeBaseForm = ref({
  name: "",
  description: "",
  embedding_model: "text-embedding-v1",
  chunk_size: 500,
  chunk_overlap: 50,
  separator: "\\n\\n",
});

const activeSession = computed(
  () => sessions.value.find((session) => session.id === activeSessionId.value) || null,
);
const knowledgeBaseTotalPages = computed(() =>
  Math.max(1, Math.ceil(knowledgeBaseTotal.value / knowledgeBasePageSize.value)),
);

async function loadSessions(preferredSessionId = "") {
  loadingSessions.value = true;
  errorMessage.value = "";
  try {
    const data = await getSessions();
    sessions.value = data;

    if (preferredSessionId && data.some((session) => session.id === preferredSessionId)) {
      activeSessionId.value = preferredSessionId;
    } else if (!activeSessionId.value && data.length > 0) {
      activeSessionId.value = data[0].id;
    } else if (
      activeSessionId.value &&
      !data.some((session) => session.id === activeSessionId.value)
    ) {
      activeSessionId.value = data[0]?.id || "";
    }
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    loadingSessions.value = false;
  }
}

async function loadMessagesForSession(sessionId) {
  if (!sessionId) {
    messages.value = [];
    return;
  }

  loadingMessages.value = true;
  errorMessage.value = "";
  try {
    messages.value = await getMessages(sessionId);
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    loadingMessages.value = false;
  }
}

async function loadKnowledgeBases(page = knowledgeBasePage.value) {
  loadingKnowledgeBases.value = true;
  errorMessage.value = "";
  try {
    const data = await getKnowledgeBases(page, knowledgeBasePageSize.value);
    knowledgeBases.value = data.items;
    knowledgeBasePage.value = data.page;
    knowledgeBaseTotal.value = data.total;
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    loadingKnowledgeBases.value = false;
  }
}

async function ensureSession() {
  if (activeSessionId.value) {
    return activeSessionId.value;
  }

  const session = await createSession();
  sessions.value = [session, ...sessions.value];
  activeSessionId.value = session.id;
  messages.value = [];
  return session.id;
}

async function handleCreateSession() {
  errorMessage.value = "";
  try {
    const session = await createSession();
    sessions.value = [session, ...sessions.value];
    activeSessionId.value = session.id;
    messages.value = [];
    draft.value = "";
  } catch (error) {
    errorMessage.value = error.message;
  }
}

async function handleSelectSession(sessionId) {
  if (sessionId === activeSessionId.value) {
    return;
  }
  activeSessionId.value = sessionId;
  await loadMessagesForSession(sessionId);
}

async function handleDeleteSession(sessionId) {
  errorMessage.value = "";
  try {
    await deleteSession(sessionId);
    const wasActive = sessionId === activeSessionId.value;
    await loadSessions();
    if (wasActive) {
      const nextSessionId = sessions.value[0]?.id || "";
      activeSessionId.value = nextSessionId;
      await loadMessagesForSession(nextSessionId);
    }
  } catch (error) {
    errorMessage.value = error.message;
  }
}

function startRename(session) {
  editingSessionId.value = session.id;
  editingTitle.value = session.title;
}

async function submitRename(sessionId) {
  const title = editingTitle.value.trim();
  if (!title) {
    editingSessionId.value = "";
    editingTitle.value = "";
    return;
  }

  errorMessage.value = "";
  try {
    const updated = await renameSession(sessionId, title);
    sessions.value = sessions.value.map((session) =>
      session.id === sessionId ? updated : session,
    );
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    editingSessionId.value = "";
    editingTitle.value = "";
  }
}

async function handleSendMessage() {
  const content = draft.value.trim();
  if (!content || sending.value) {
    return;
  }

  sending.value = true;
  errorMessage.value = "";
  try {
    const sessionId = await ensureSession();
    const exchange = await sendMessage(sessionId, content);
    draft.value = "";
    messages.value = [...messages.value, exchange.user_message, exchange.assistant_message];
    sessions.value = [
      exchange.session,
      ...sessions.value.filter((session) => session.id !== exchange.session.id),
    ];
    activeSessionId.value = exchange.session.id;
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    sending.value = false;
  }
}

async function handleCreateKnowledgeBase() {
  if (creatingKnowledgeBase.value) {
    return;
  }

  creatingKnowledgeBase.value = true;
  errorMessage.value = "";
  try {
    const payload = {
      name: knowledgeBaseForm.value.name,
      description: knowledgeBaseForm.value.description,
      config: {
        embedding_model: knowledgeBaseForm.value.embedding_model || undefined,
        chunk_size: Number(knowledgeBaseForm.value.chunk_size),
        chunk_overlap: Number(knowledgeBaseForm.value.chunk_overlap),
        separator: knowledgeBaseForm.value.separator || undefined,
      },
    };
    await createKnowledgeBase(payload);
    knowledgeBaseForm.value = {
      name: "",
      description: "",
      embedding_model: "text-embedding-v1",
      chunk_size: 500,
      chunk_overlap: 50,
      separator: "\\n\\n",
    };
    await loadKnowledgeBases(1);
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    creatingKnowledgeBase.value = false;
  }
}

async function changeKnowledgeBasePage(nextPage) {
  if (nextPage < 1 || nextPage > knowledgeBaseTotalPages.value) {
    return;
  }
  await loadKnowledgeBases(nextPage);
}

function formatTime(value) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function formatKnowledgeBaseConfig(config) {
  const parts = [];
  if (config.embedding_model) {
    parts.push(config.embedding_model);
  }
  if (config.chunk_size) {
    parts.push(`chunk ${config.chunk_size}`);
  }
  if (config.chunk_overlap !== undefined) {
    parts.push(`overlap ${config.chunk_overlap}`);
  }
  return parts.join(" · ") || "未设置配置";
}

onMounted(async () => {
  await Promise.all([loadSessions(), loadKnowledgeBases()]);
  if (!sessions.value.length) {
    await handleCreateSession();
    return;
  }
  await loadMessagesForSession(activeSessionId.value);
});
</script>

<template>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">
        <p class="eyebrow">Impression Study</p>
        <h1>对话与知识库控制台</h1>
        <p class="brand-copy">
          聊天会话继续保存在 Redis，知识库元数据通过 MySQL 管理；同一界面里分别处理对话和知识库准备工作。
        </p>
      </div>

      <div class="sidebar-sections">
        <section class="session-panel">
          <div class="panel-header">
            <span>会话列表</span>
            <span v-if="loadingSessions">同步中</span>
          </div>

          <button class="primary-button" @click="handleCreateSession">新建会话</button>

          <article
            v-for="session in sessions"
            :key="session.id"
            class="session-card"
            :class="{ active: session.id === activeSessionId }"
            @click="handleSelectSession(session.id)"
          >
            <div class="session-copy">
              <template v-if="editingSessionId === session.id">
                <input
                  v-model="editingTitle"
                  class="session-input"
                  maxlength="80"
                  @click.stop
                  @keydown.enter.prevent="submitRename(session.id)"
                  @blur="submitRename(session.id)"
                />
              </template>
              <template v-else>
                <strong>{{ session.title }}</strong>
                <span>{{ session.message_count }} 条消息</span>
              </template>
            </div>
            <div class="session-actions">
              <span>{{ formatTime(session.updated_at) }}</span>
              <div class="mini-actions">
                <button class="ghost-button" @click.stop="startRename(session)">重命名</button>
                <button class="ghost-button danger" @click.stop="handleDeleteSession(session.id)">
                  删除
                </button>
              </div>
            </div>
          </article>
        </section>

        <section class="knowledge-panel">
          <div class="panel-header">
            <span>知识库工坊</span>
            <span>{{ knowledgeBaseTotal }} 个</span>
          </div>

          <div class="knowledge-shell">
            <form class="knowledge-form" @submit.prevent="handleCreateKnowledgeBase">
              <label class="field-label">
                <span>知识库名称</span>
                <input
                  v-model="knowledgeBaseForm.name"
                  class="field-input"
                  maxlength="80"
                  placeholder="例如：产品手册库"
                />
              </label>

              <label class="field-label">
                <span>知识库描述</span>
                <textarea
                  v-model="knowledgeBaseForm.description"
                  class="field-input field-area"
                  rows="3"
                  placeholder="说明这个知识库的内容范围和用途"
                />
              </label>

              <div class="field-grid">
                <label class="field-label">
                  <span>Embedding 模型</span>
                  <input
                    v-model="knowledgeBaseForm.embedding_model"
                    class="field-input"
                    placeholder="text-embedding-v1"
                  />
                </label>

                <label class="field-label">
                  <span>切片大小</span>
                  <input
                    v-model.number="knowledgeBaseForm.chunk_size"
                    class="field-input"
                    min="1"
                    type="number"
                  />
                </label>

                <label class="field-label">
                  <span>重叠长度</span>
                  <input
                    v-model.number="knowledgeBaseForm.chunk_overlap"
                    class="field-input"
                    min="0"
                    type="number"
                  />
                </label>

                <label class="field-label">
                  <span>分隔符</span>
                  <input
                    v-model="knowledgeBaseForm.separator"
                    class="field-input"
                    placeholder="\n\n"
                  />
                </label>
              </div>

              <button class="secondary-button" :disabled="creatingKnowledgeBase" type="submit">
                {{ creatingKnowledgeBase ? "创建中..." : "创建知识库" }}
              </button>
            </form>

            <div class="knowledge-list">
              <div class="knowledge-list-header">
                <strong>分页列表</strong>
                <span v-if="loadingKnowledgeBases">加载中</span>
              </div>

              <div v-if="!knowledgeBases.length && !loadingKnowledgeBases" class="knowledge-empty">
                还没有知识库。先创建一个元数据入口，后续再接文档上传和检索。
              </div>

              <article v-for="item in knowledgeBases" :key="item.id" class="knowledge-card">
                <div class="knowledge-card-header">
                  <strong>{{ item.name }}</strong>
                  <span>{{ item.document_count }} 份文档</span>
                </div>
                <p>{{ item.description || "暂无描述" }}</p>
                <div class="knowledge-meta">
                  <span>{{ formatKnowledgeBaseConfig(item.config) }}</span>
                  <time>{{ formatTime(item.created_at) }}</time>
                </div>
              </article>

              <div class="pager">
                <button
                  class="pager-button"
                  :disabled="knowledgeBasePage <= 1"
                  @click="changeKnowledgeBasePage(knowledgeBasePage - 1)"
                >
                  上一页
                </button>
                <span>第 {{ knowledgeBasePage }} / {{ knowledgeBaseTotalPages }} 页</span>
                <button
                  class="pager-button"
                  :disabled="knowledgeBasePage >= knowledgeBaseTotalPages"
                  @click="changeKnowledgeBasePage(knowledgeBasePage + 1)"
                >
                  下一页
                </button>
              </div>
            </div>
          </div>
        </section>
      </div>
    </aside>

    <main class="chat-stage">
      <div class="floating-study study-a" aria-hidden="true"></div>
      <div class="floating-study study-b" aria-hidden="true"></div>
      <div class="floating-study study-c" aria-hidden="true"></div>

      <header class="chat-header">
        <div>
          <p class="eyebrow chat-eyebrow">Monet Light</p>
          <h2>{{ activeSession?.title || "未选择会话" }}</h2>
        </div>
        <p class="hint">像在展墙上整理画片一样组织会话、消息与知识库条目。</p>
      </header>

      <div class="message-wall">
        <div v-if="loadingMessages" class="placeholder">正在加载消息...</div>
        <div v-else-if="!messages.length" class="placeholder">
          这是一个空白会话。发第一条消息后，标题会自动取自首句内容。
        </div>
        <article
          v-for="message in messages"
          :key="`${message.created_at}-${message.role}-${message.content}`"
          class="message-card"
          :class="message.role"
        >
          <div class="message-meta">
            <time>{{ formatTime(message.created_at) }}</time>
          </div>
          <p>{{ message.content }}</p>
        </article>
      </div>

      <footer class="composer">
        <div v-if="errorMessage" class="error-banner">{{ errorMessage }}</div>
        <textarea
          v-model="draft"
          class="composer-input"
          placeholder="输入消息，回车发送，Shift + Enter 换行"
          rows="4"
          @keydown.enter.exact.prevent="handleSendMessage"
        />
        <div class="composer-actions">
          <span>FastAPI + Vue + Redis + MySQL</span>
          <button class="primary-button composer-button" :disabled="sending" @click="handleSendMessage">
            {{ sending ? "发送中..." : "发送消息" }}
          </button>
        </div>
      </footer>
    </main>
  </div>
</template>
