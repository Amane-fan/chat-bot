<script setup>
import { computed, onMounted, ref } from "vue";
import {
  createSession,
  deleteSession,
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

const activeSession = computed(
  () => sessions.value.find((session) => session.id === activeSessionId.value) || null,
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

function formatTime(value) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

onMounted(async () => {
  await loadSessions();
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
        <p class="eyebrow">Session Layer</p>
        <h1>对话控制台</h1>
        <p class="brand-copy">Redis 保存会话，FastAPI 提供接口，Vue 负责交互和状态切换。</p>
      </div>

      <button class="primary-button" @click="handleCreateSession">新建会话</button>

      <div class="session-panel">
        <div class="panel-header">
          <span>会话列表</span>
          <span v-if="loadingSessions">同步中</span>
        </div>

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
      </div>
    </aside>

    <main class="chat-stage">
      <header class="chat-header">
        <div>
          <p class="eyebrow">Active Session</p>
          <h2>{{ activeSession?.title || "未选择会话" }}</h2>
        </div>
        <p class="hint">右侧只展示当前会话消息，左侧负责切换和管理。</p>
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
            <span>{{ message.role === "user" ? "你" : "助手" }}</span>
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
          <span>FastAPI + Vue + Redis</span>
          <button class="primary-button" :disabled="sending" @click="handleSendMessage">
            {{ sending ? "发送中..." : "发送消息" }}
          </button>
        </div>
      </footer>
    </main>
  </div>
</template>
