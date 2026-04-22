<script setup>
import { computed, onMounted, ref } from "vue";
import {
  createKnowledgeBase,
  createSession,
  deleteKnowledgeBase,
  deleteKnowledgeBaseDocument,
  deleteSession,
  getKnowledgeBaseDocuments,
  getKnowledgeBaseOptions,
  getKnowledgeBases,
  getMessages,
  getSessions,
  renameSession,
  sendMessageStream,
  updateKnowledgeBase,
  updateSessionKnowledgeBases,
  uploadKnowledgeBaseDocument,
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
const currentView = ref(getInitialView());
const chatInitialized = ref(false);
const knowledgeInitialized = ref(false);
const sessionListCollapsed = ref(false);
const sessionModalVisible = ref(false);
const sessionModalMode = ref("create");
const savingSessionConfig = ref(false);
const pendingSendAfterSessionCreate = ref(false);
const knowledgeOptionsLoading = ref(false);
const knowledgeOptionsLoaded = ref(false);
const knowledgeBaseOptions = ref([]);
const sessionForm = ref({
  sessionId: "",
  title: "",
  selectedKnowledgeBaseIds: [],
});

const knowledgeBases = ref([]);
const knowledgeBasePage = ref(1);
const knowledgeBasePageSize = ref(5);
const knowledgeBaseTotal = ref(0);
const loadingKnowledgeBases = ref(false);
const creatingKnowledgeBase = ref(false);
const editingKnowledgeBaseId = ref("");
const deletingKnowledgeBaseIds = ref({});
const knowledgeBaseForm = ref(createDefaultKnowledgeBaseForm());
const knowledgeBaseDocuments = ref({});
const loadingKnowledgeBaseDocumentIds = ref({});
const uploadingKnowledgeBaseDocumentIds = ref({});
const deletingKnowledgeBaseDocumentIds = ref({});
const pendingKnowledgeBaseFiles = ref({});
const knowledgeBaseUploadInputKeys = ref({});
const expandedRetrievalMessageIds = ref({});
let messageClientIdSeed = 0;

const activeSession = computed(
  () => sessions.value.find((session) => session.id === activeSessionId.value) || null,
);
const knowledgeBaseTotalPages = computed(() =>
  Math.max(1, Math.ceil(knowledgeBaseTotal.value / knowledgeBasePageSize.value)),
);
const isEditingKnowledgeBase = computed(() => Boolean(editingKnowledgeBaseId.value));
const knowledgeBaseFormHeading = computed(() =>
  isEditingKnowledgeBase.value ? "编辑知识库" : "知识库工坊",
);
const knowledgeBaseFormSubmitLabel = computed(() => {
  if (creatingKnowledgeBase.value) {
    return isEditingKnowledgeBase.value ? "保存中..." : "创建中...";
  }
  return isEditingKnowledgeBase.value ? "保存修改" : "创建知识库";
});
const sessionModalHeading = computed(() =>
  sessionModalMode.value === "create" ? "新建会话" : "编辑会话知识库",
);
const sessionModalSubmitLabel = computed(() =>
  sessionModalMode.value === "create" ? "创建会话" : "保存知识库范围",
);
const sessionModalCopy = computed(() =>
  sessionModalMode.value === "create"
    ? "为新会话选择一个可选标题，并决定是否立刻绑定知识库。"
    : "调整当前会话绑定的知识库范围。当前阶段只保存关系，不会改变回答逻辑。",
);

function getInitialView() {
  if (typeof window === "undefined") {
    return "chat";
  }
  return window.location.hash === "#knowledge" ? "knowledge" : "chat";
}

function syncViewHash(view) {
  if (typeof window === "undefined") {
    return;
  }
  const nextHash = view === "knowledge" ? "#knowledge" : "#chat";
  if (window.location.hash !== nextHash) {
    window.history.replaceState(null, "", nextHash);
  }
}

function createDefaultKnowledgeBaseForm() {
  return {
    name: "",
    description: "",
    embedding_model: "text-embedding-v1",
    chunk_size: 500,
    chunk_overlap: 50,
    separator: "\\n\\n",
  };
}

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
    } else if (!data.length) {
      activeSessionId.value = "";
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

async function loadKnowledgeBaseOptions(force = false) {
  if (knowledgeOptionsLoaded.value && !force) {
    return knowledgeBaseOptions.value;
  }

  knowledgeOptionsLoading.value = true;
  errorMessage.value = "";
  try {
    knowledgeBaseOptions.value = await getKnowledgeBaseOptions();
    knowledgeOptionsLoaded.value = true;
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    knowledgeOptionsLoading.value = false;
  }

  return knowledgeBaseOptions.value;
}

async function loadKnowledgeBaseDocumentsForItem(knowledgeBaseId) {
  loadingKnowledgeBaseDocumentIds.value = {
    ...loadingKnowledgeBaseDocumentIds.value,
    [knowledgeBaseId]: true,
  };

  try {
    const data = await getKnowledgeBaseDocuments(knowledgeBaseId);
    knowledgeBaseDocuments.value = {
      ...knowledgeBaseDocuments.value,
      [knowledgeBaseId]: data.items,
    };
  } finally {
    loadingKnowledgeBaseDocumentIds.value = {
      ...loadingKnowledgeBaseDocumentIds.value,
      [knowledgeBaseId]: false,
    };
  }
}

async function loadDocumentsForVisibleKnowledgeBases(items) {
  const visibleIds = new Set(items.map((item) => item.id));
  const nextDocuments = { ...knowledgeBaseDocuments.value };

  Object.keys(nextDocuments).forEach((knowledgeBaseId) => {
    if (!visibleIds.has(knowledgeBaseId)) {
      delete nextDocuments[knowledgeBaseId];
    }
  });

  knowledgeBaseDocuments.value = nextDocuments;
  await Promise.all(items.map((item) => loadKnowledgeBaseDocumentsForItem(item.id)));
}

async function loadKnowledgeBases(page = knowledgeBasePage.value) {
  loadingKnowledgeBases.value = true;
  errorMessage.value = "";
  try {
    const data = await getKnowledgeBases(page, knowledgeBasePageSize.value);
    knowledgeBases.value = data.items;
    knowledgeBasePage.value = data.page;
    knowledgeBaseTotal.value = data.total;
    await loadDocumentsForVisibleKnowledgeBases(data.items);
    knowledgeInitialized.value = true;
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    loadingKnowledgeBases.value = false;
  }
}

function upsertSessionSummary(session, moveToTop = true) {
  const nextSessions = sessions.value.filter((item) => item.id !== session.id);
  sessions.value = moveToTop ? [session, ...nextSessions] : [...nextSessions, session];
}

async function openSessionModal(mode, session = null, { pendingSend = false } = {}) {
  await loadKnowledgeBaseOptions();
  sessionModalMode.value = mode;
  pendingSendAfterSessionCreate.value = pendingSend;
  sessionForm.value = {
    sessionId: session?.id || "",
    title: mode === "create" ? "" : session?.title || "",
    selectedKnowledgeBaseIds: session?.knowledge_bases?.map((item) => item.id) || [],
  };
  sessionModalVisible.value = true;
}

function closeSessionModal() {
  sessionModalVisible.value = false;
  savingSessionConfig.value = false;
  pendingSendAfterSessionCreate.value = false;
  sessionForm.value = {
    sessionId: "",
    title: "",
    selectedKnowledgeBaseIds: [],
  };
}

function toggleKnowledgeBaseSelection(knowledgeBaseId) {
  const selectedIds = new Set(sessionForm.value.selectedKnowledgeBaseIds);
  if (selectedIds.has(knowledgeBaseId)) {
    selectedIds.delete(knowledgeBaseId);
  } else {
    selectedIds.add(knowledgeBaseId);
  }
  sessionForm.value.selectedKnowledgeBaseIds = Array.from(selectedIds);
}

function clearSessionModalKnowledgeBases() {
  sessionForm.value.selectedKnowledgeBaseIds = [];
}

async function handleCreateSession() {
  errorMessage.value = "";
  await openSessionModal("create");
}

async function handleEditSessionKnowledgeBases(session) {
  errorMessage.value = "";
  await openSessionModal("edit", session);
}

async function submitSessionModal() {
  if (savingSessionConfig.value) {
    return;
  }

  savingSessionConfig.value = true;
  errorMessage.value = "";
  try {
    if (sessionModalMode.value === "create") {
      const title = sessionForm.value.title.trim();
      const createdSession = await createSession({
        title: title || undefined,
        knowledgeBaseIds: sessionForm.value.selectedKnowledgeBaseIds,
      });
      upsertSessionSummary(createdSession);
      activeSessionId.value = createdSession.id;
      messages.value = [];
      chatInitialized.value = true;

      const shouldSendPendingMessage =
        pendingSendAfterSessionCreate.value && draft.value.trim().length > 0;
      closeSessionModal();

      if (shouldSendPendingMessage) {
        await performSendMessage(createdSession.id, draft.value.trim());
      }
      return;
    }

    const updatedSession = await updateSessionKnowledgeBases(
      sessionForm.value.sessionId,
      sessionForm.value.selectedKnowledgeBaseIds,
    );
    upsertSessionSummary(updatedSession);
    activeSessionId.value = updatedSession.id;
    closeSessionModal();
  } catch (error) {
    errorMessage.value = error.message;
    savingSessionConfig.value = false;
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
    upsertSessionSummary(updated);
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    editingSessionId.value = "";
    editingTitle.value = "";
  }
}

async function performSendMessage(sessionId, content) {
  sending.value = true;
  errorMessage.value = "";
  const userClientId = buildMessageClientId("user");
  const assistantClientId = buildMessageClientId("assistant");
  let assistantStarted = false;

  try {
    const exchange = await sendMessageStream(sessionId, content, {
      metadata(event) {
        draft.value = "";
        messages.value = [
          ...messages.value,
          {
            ...event.user_message,
            client_id: userClientId,
          },
          {
            role: "assistant",
            content: "",
            created_at: new Date().toISOString(),
            retrieved_chunks: event.retrieved_chunks || [],
            client_id: assistantClientId,
            streaming: true,
          },
        ];
      },
      token(event) {
        assistantStarted = true;
        messages.value = messages.value.map((message) =>
          message.client_id === assistantClientId
            ? { ...message, content: `${message.content}${event.content || ""}` }
            : message,
        );
      },
      done(event) {
        const finalExchange = event.exchange;
        messages.value = messages.value.map((message) => {
          if (message.client_id === userClientId) {
            return { ...finalExchange.user_message, client_id: userClientId };
          }
          if (message.client_id === assistantClientId) {
            return {
              ...finalExchange.assistant_message,
              client_id: assistantClientId,
              streaming: false,
            };
          }
          return message;
        });
        upsertSessionSummary(finalExchange.session);
        activeSessionId.value = finalExchange.session.id;
      },
    });

    if (exchange && !assistantStarted) {
      upsertSessionSummary(exchange.session);
      activeSessionId.value = exchange.session.id;
    }
  } catch (error) {
    errorMessage.value = error.message;
    messages.value = messages.value.map((message) =>
      message.client_id === assistantClientId
        ? { ...message, streaming: false, content: message.content || "模型调用失败" }
        : message,
    );
  } finally {
    sending.value = false;
  }
}

async function handleSendMessage() {
  const content = draft.value.trim();
  if (!content || sending.value) {
    return;
  }

  if (!activeSessionId.value) {
    await openSessionModal("create", null, { pendingSend: true });
    return;
  }

  await performSendMessage(activeSessionId.value, content);
}

function resetKnowledgeBaseForm() {
  editingKnowledgeBaseId.value = "";
  knowledgeBaseForm.value = createDefaultKnowledgeBaseForm();
}

function startEditKnowledgeBase(item) {
  errorMessage.value = "";
  editingKnowledgeBaseId.value = item.id;
  knowledgeBaseForm.value = {
    ...createDefaultKnowledgeBaseForm(),
    name: item.name,
    description: item.description || "",
  };
}

function cancelEditKnowledgeBase() {
  resetKnowledgeBaseForm();
}

async function refreshKnowledgeBaseReferences() {
  knowledgeOptionsLoaded.value = false;
  if (sessionModalVisible.value) {
    await loadKnowledgeBaseOptions(true);
  }
  if (chatInitialized.value) {
    await loadSessions(activeSessionId.value);
  }
}

async function handleSubmitKnowledgeBaseForm() {
  if (creatingKnowledgeBase.value) {
    return;
  }

  creatingKnowledgeBase.value = true;
  errorMessage.value = "";
  try {
    if (isEditingKnowledgeBase.value) {
      await updateKnowledgeBase(editingKnowledgeBaseId.value, {
        name: knowledgeBaseForm.value.name,
        description: knowledgeBaseForm.value.description,
      });
      resetKnowledgeBaseForm();
      await Promise.all([
        loadKnowledgeBases(knowledgeBasePage.value),
        refreshKnowledgeBaseReferences(),
      ]);
      return;
    }

    await createKnowledgeBase({
      name: knowledgeBaseForm.value.name,
      description: knowledgeBaseForm.value.description,
      config: {
        embedding_model: knowledgeBaseForm.value.embedding_model || undefined,
        chunk_size: Number(knowledgeBaseForm.value.chunk_size),
        chunk_overlap: Number(knowledgeBaseForm.value.chunk_overlap),
        separator: knowledgeBaseForm.value.separator || undefined,
      },
    });
    resetKnowledgeBaseForm();
    await Promise.all([loadKnowledgeBases(1), refreshKnowledgeBaseReferences()]);
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    creatingKnowledgeBase.value = false;
  }
}

async function handleDeleteKnowledgeBase(item) {
  if (deletingKnowledgeBaseIds.value[item.id]) {
    return;
  }
  const confirmed = window.confirm(
    `确认删除知识库「${item.name}」吗？这会同时删除其中的文档、本地文件和向量索引。`,
  );
  if (!confirmed) {
    return;
  }

  deletingKnowledgeBaseIds.value = {
    ...deletingKnowledgeBaseIds.value,
    [item.id]: true,
  };
  errorMessage.value = "";

  try {
    await deleteKnowledgeBase(item.id);
    if (editingKnowledgeBaseId.value === item.id) {
      resetKnowledgeBaseForm();
    }

    const nextTotal = Math.max(0, knowledgeBaseTotal.value - 1);
    const nextTotalPages = Math.max(
      1,
      Math.ceil(nextTotal / knowledgeBasePageSize.value),
    );
    const nextPage = Math.min(knowledgeBasePage.value, nextTotalPages);
    await Promise.all([loadKnowledgeBases(nextPage), refreshKnowledgeBaseReferences()]);
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    deletingKnowledgeBaseIds.value = {
      ...deletingKnowledgeBaseIds.value,
      [item.id]: false,
    };
  }
}

function handleKnowledgeBaseFileChange(knowledgeBaseId, event) {
  pendingKnowledgeBaseFiles.value = {
    ...pendingKnowledgeBaseFiles.value,
    [knowledgeBaseId]: event.target.files?.[0] || null,
  };
}

function resetKnowledgeBaseUploadInput(knowledgeBaseId) {
  knowledgeBaseUploadInputKeys.value = {
    ...knowledgeBaseUploadInputKeys.value,
    [knowledgeBaseId]: (knowledgeBaseUploadInputKeys.value[knowledgeBaseId] || 0) + 1,
  };
  pendingKnowledgeBaseFiles.value = {
    ...pendingKnowledgeBaseFiles.value,
    [knowledgeBaseId]: null,
  };
}

async function handleUploadKnowledgeBaseDocument(knowledgeBaseId) {
  const file = pendingKnowledgeBaseFiles.value[knowledgeBaseId];
  if (!file || uploadingKnowledgeBaseDocumentIds.value[knowledgeBaseId]) {
    return;
  }

  uploadingKnowledgeBaseDocumentIds.value = {
    ...uploadingKnowledgeBaseDocumentIds.value,
    [knowledgeBaseId]: true,
  };
  errorMessage.value = "";

  try {
    await uploadKnowledgeBaseDocument(knowledgeBaseId, file);
    resetKnowledgeBaseUploadInput(knowledgeBaseId);
    await loadKnowledgeBases(knowledgeBasePage.value);
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    uploadingKnowledgeBaseDocumentIds.value = {
      ...uploadingKnowledgeBaseDocumentIds.value,
      [knowledgeBaseId]: false,
    };
  }
}

async function handleDeleteKnowledgeBaseDocument(knowledgeBaseId, document) {
  if (deletingKnowledgeBaseDocumentIds.value[document.id]) {
    return;
  }
  const confirmed = window.confirm(`确认删除文档「${document.original_filename}」吗？`);
  if (!confirmed) {
    return;
  }

  deletingKnowledgeBaseDocumentIds.value = {
    ...deletingKnowledgeBaseDocumentIds.value,
    [document.id]: true,
  };
  errorMessage.value = "";

  try {
    await deleteKnowledgeBaseDocument(knowledgeBaseId, document.id);
    await Promise.all([
      loadKnowledgeBaseDocumentsForItem(knowledgeBaseId),
      loadKnowledgeBases(knowledgeBasePage.value),
    ]);
  } catch (error) {
    errorMessage.value = error.message;
  } finally {
    deletingKnowledgeBaseDocumentIds.value = {
      ...deletingKnowledgeBaseDocumentIds.value,
      [document.id]: false,
    };
  }
}

async function changeKnowledgeBasePage(nextPage) {
  if (nextPage < 1 || nextPage > knowledgeBaseTotalPages.value) {
    return;
  }
  await loadKnowledgeBases(nextPage);
}

async function initializeChatView() {
  if (chatInitialized.value) {
    return;
  }

  await loadSessions();
  await loadKnowledgeBaseOptions();
  if (activeSessionId.value) {
    await loadMessagesForSession(activeSessionId.value);
  } else {
    messages.value = [];
  }
  chatInitialized.value = true;
}

async function initializeKnowledgeView() {
  if (knowledgeInitialized.value) {
    return;
  }
  await loadKnowledgeBases();
}

async function switchView(view) {
  currentView.value = view;
  syncViewHash(view);
  errorMessage.value = "";

  if (view === "chat") {
    await initializeChatView();
    return;
  }

  await initializeKnowledgeView();
}

function toggleSessionList() {
  sessionListCollapsed.value = !sessionListCollapsed.value;
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

function formatSessionKnowledgeBases(knowledgeBases, limit = 2) {
  if (!knowledgeBases?.length) {
    return "未绑定知识库";
  }

  const names = knowledgeBases.map((item) => item.name);
  if (names.length <= limit) {
    return names.join(" · ");
  }
  return `${names.slice(0, limit).join(" · ")} +${names.length - limit}`;
}

function formatDocumentStatus(status) {
  if (status === "ready") {
    return "已入库";
  }
  if (status === "failed") {
    return "处理失败";
  }
  return "处理中";
}

function formatFileSize(bytes) {
  if (!bytes) {
    return "0 B";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatRetrievalScore(score) {
  if (score === null || score === undefined) {
    return "score -";
  }
  return `score ${Number(score).toFixed(3)}`;
}

function buildMessageClientId(role) {
  messageClientIdSeed += 1;
  return `${role}-${Date.now()}-${messageClientIdSeed}`;
}

function getMessageKey(message) {
  return (
    message.client_id ||
    `${message.created_at}-${message.role}-${String(message.content || "").slice(0, 24)}`
  );
}

function toggleMessageRetrieval(message) {
  const key = getMessageKey(message);
  expandedRetrievalMessageIds.value = {
    ...expandedRetrievalMessageIds.value,
    [key]: !expandedRetrievalMessageIds.value[key],
  };
}

function isMessageRetrievalExpanded(message) {
  return Boolean(expandedRetrievalMessageIds.value[getMessageKey(message)]);
}

onMounted(async () => {
  await switchView(currentView.value);
});
</script>

<template>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">
        <p class="eyebrow">Knowledge Cabin</p>
        <h1>知答舱</h1>
      </div>

      <nav class="view-switcher" aria-label="工作区切换">
        <button
          class="view-button"
          :class="{ active: currentView === 'chat' }"
          @click="switchView('chat')"
        >
          <span>问答舱</span>
          <small>会话、消息、模型回复</small>
        </button>
        <button
          class="view-button"
          :class="{ active: currentView === 'knowledge' }"
          @click="switchView('knowledge')"
        >
          <span>知识库</span>
          <small>元数据、配置、分页列表</small>
        </button>
      </nav>

      <div class="sidebar-cards">
        <article class="sidebar-card">
          <span>会话数</span>
          <strong>{{ sessions.length }}</strong>
          <p>当前聊天工作区会维护最近活跃会话。</p>
        </article>
        <article class="sidebar-card">
          <span>知识库数</span>
          <strong>{{ knowledgeBaseTotal }}</strong>
          <p>知识库独立负责元数据创建与管理。</p>
        </article>
      </div>
    </aside>

    <main class="workspace">
      <section v-if="currentView === 'chat'" class="view-stage chat-stage">
        <div class="floating-study study-a" aria-hidden="true"></div>
        <div class="floating-study study-b" aria-hidden="true"></div>

        <header class="stage-header">
          <div>
            <p class="eyebrow chat-eyebrow">Answer Cabin</p>
            <h2>{{ activeSession?.title || "问答舱" }}</h2>
          </div>
        </header>

        <div class="chat-layout" :class="{ 'sessions-collapsed': sessionListCollapsed }">
          <section class="session-panel session-stage" :class="{ collapsed: sessionListCollapsed }">
            <div class="panel-header">
              <span>{{ sessionListCollapsed ? "会话" : "会话列表" }}</span>
              <div class="panel-tools">
                <span v-if="loadingSessions">同步中</span>
                <button
                  class="panel-toggle"
                  :aria-expanded="(!sessionListCollapsed).toString()"
                  :aria-label="sessionListCollapsed ? '展开会话列表' : '收起会话列表'"
                  @click="toggleSessionList"
                >
                  {{ sessionListCollapsed ? "›" : "‹" }}
                </button>
              </div>
            </div>

            <div v-if="sessionListCollapsed" class="session-collapsed-summary">
              <strong>{{ sessions.length }}</strong>
              <span>条会话</span>
            </div>

            <div v-else class="session-panel-body">
              <button class="primary-button" @click="handleCreateSession">新建会话</button>

              <div class="session-list">
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
                      <div class="session-copy-meta">
                        <span>{{ session.message_count }} 条消息</span>
                        <span>{{ formatSessionKnowledgeBases(session.knowledge_bases) }}</span>
                      </div>
                    </template>
                  </div>
                  <div class="session-actions">
                    <span>{{ formatTime(session.updated_at) }}</span>
                    <div class="mini-actions">
                      <button
                        class="ghost-button"
                        @click.stop="handleEditSessionKnowledgeBases(session)"
                      >
                        知识库
                      </button>
                      <button class="ghost-button" @click.stop="startRename(session)">重命名</button>
                      <button class="ghost-button danger" @click.stop="handleDeleteSession(session.id)">
                        删除
                      </button>
                    </div>
                  </div>
                </article>
              </div>
            </div>
          </section>

          <section class="conversation-panel">
            <div class="panel-header">
              <span>消息窗口</span>
              <span v-if="loadingMessages">加载中</span>
            </div>

            <div class="message-wall">
              <div v-if="loadingMessages" class="placeholder">正在加载消息...</div>
              <div v-else-if="!activeSession" class="placeholder">
                还没有活动会话。点击“新建会话”，或直接输入消息后在弹层里完成创建。
              </div>
              <div v-else-if="!messages.length" class="placeholder">
                这是一个空白会话。发第一条消息后，标题会自动取自首句内容。
              </div>
              <article
                v-for="message in messages"
                :key="getMessageKey(message)"
                class="message-card"
                :class="[message.role, { streaming: message.streaming }]"
              >
                <div class="message-meta">
                  <time>{{ formatTime(message.created_at) }}</time>
                </div>
                <p>{{ message.content }}</p>
                <div
                  v-if="message.retrieved_chunks?.length"
                  class="message-retrieval"
                >
                  <button
                    class="message-retrieval-toggle"
                    @click="toggleMessageRetrieval(message)"
                  >
                    <span>本次命中的知识库片段</span>
                    <strong>{{ message.retrieved_chunks.length }}</strong>
                    <span>{{ isMessageRetrievalExpanded(message) ? "收起" : "展开" }}</span>
                  </button>
                  <div
                    v-if="isMessageRetrievalExpanded(message)"
                    class="message-retrieval-list"
                  >
                    <section
                      v-for="(chunk, index) in message.retrieved_chunks"
                      :key="`${chunk.document_id}-${chunk.chunk_index}-${index}`"
                      class="message-retrieval-item"
                    >
                      <div class="message-retrieval-source">
                        <strong>[来源 {{ index + 1 }}] {{ chunk.original_filename || "未知文档" }}</strong>
                        <span>{{ formatRetrievalScore(chunk.score) }}</span>
                      </div>
                      <div class="message-retrieval-meta">
                        {{ chunk.knowledge_base_name }} · chunk {{ chunk.chunk_index }}
                      </div>
                      <p>{{ chunk.text }}</p>
                    </section>
                  </div>
                </div>
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
                <span>知答舱 · RAG Workspace</span>
                <button
                  class="primary-button composer-button"
                  :disabled="sending"
                  @click="handleSendMessage"
                >
                  {{ sending ? "发送中..." : "发送消息" }}
                </button>
              </div>
            </footer>
          </section>
        </div>
      </section>

      <section v-else class="view-stage knowledge-stage">
        <div class="floating-study study-b" aria-hidden="true"></div>
        <div class="floating-study study-c" aria-hidden="true"></div>

        <header class="stage-header">
          <div>
            <p class="eyebrow chat-eyebrow">Knowledge Hold</p>
            <h2>知识库</h2>
          </div>
          <p class="hint">在这里创建知识库，并把文档按对应分块和 embedding 配置同步写入 Qdrant。</p>
        </header>

        <div class="knowledge-workspace">
          <section class="knowledge-panel knowledge-editor">
            <div class="panel-header">
              <span>{{ knowledgeBaseFormHeading }}</span>
              <span>{{ isEditingKnowledgeBase ? "配置锁定" : `${knowledgeBaseTotal} 个` }}</span>
            </div>

            <div v-if="errorMessage" class="error-banner">{{ errorMessage }}</div>

            <form class="knowledge-form" @submit.prevent="handleSubmitKnowledgeBaseForm">
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
                  rows="4"
                  placeholder="说明这个知识库的内容范围和用途"
                />
              </label>

              <div v-if="!isEditingKnowledgeBase" class="field-grid">
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

              <p v-else class="knowledge-form-note">
                配置项创建后不可修改，避免已有文档向量和切分策略不一致。
              </p>

              <div class="knowledge-form-actions">
                <button class="secondary-button" :disabled="creatingKnowledgeBase" type="submit">
                  {{ knowledgeBaseFormSubmitLabel }}
                </button>
                <button
                  v-if="isEditingKnowledgeBase"
                  class="ghost-button"
                  :disabled="creatingKnowledgeBase"
                  type="button"
                  @click="cancelEditKnowledgeBase"
                >
                  取消编辑
                </button>
              </div>
            </form>
          </section>

          <section class="knowledge-panel knowledge-library">
            <div class="knowledge-list-header">
              <strong>分页列表</strong>
              <span v-if="loadingKnowledgeBases">加载中</span>
            </div>

            <div class="knowledge-list-scroll">
              <div v-if="!knowledgeBases.length && !loadingKnowledgeBases" class="knowledge-empty">
                还没有知识库。先创建一个知识库，然后把 TXT、MD、PDF 文档上传进来。
              </div>

              <article v-for="item in knowledgeBases" :key="item.id" class="knowledge-card">
                <div class="knowledge-card-header">
                  <div class="knowledge-card-title">
                    <strong>{{ item.name }}</strong>
                    <span>{{ item.document_count }} 份文档</span>
                  </div>
                  <div class="knowledge-card-actions">
                    <button class="ghost-button" @click="startEditKnowledgeBase(item)">
                      编辑
                    </button>
                    <button
                      class="ghost-button danger"
                      :disabled="deletingKnowledgeBaseIds[item.id]"
                      @click="handleDeleteKnowledgeBase(item)"
                    >
                      {{ deletingKnowledgeBaseIds[item.id] ? "删除中..." : "删除" }}
                    </button>
                  </div>
                </div>
                <p>{{ item.description || "暂无描述" }}</p>
                <div class="knowledge-meta">
                  <span>{{ formatKnowledgeBaseConfig(item.config) }}</span>
                  <time>{{ formatTime(item.created_at) }}</time>
                </div>

                <div class="knowledge-document-panel">
                  <div class="knowledge-document-toolbar">
                    <input
                      :key="knowledgeBaseUploadInputKeys[item.id] || 0"
                      class="field-input knowledge-file-input"
                      accept=".txt,.md,.pdf"
                      type="file"
                      @change="handleKnowledgeBaseFileChange(item.id, $event)"
                    />
                    <button
                      class="secondary-button upload-button"
                      :disabled="
                        !pendingKnowledgeBaseFiles[item.id] ||
                        uploadingKnowledgeBaseDocumentIds[item.id]
                      "
                      @click="handleUploadKnowledgeBaseDocument(item.id)"
                    >
                      {{
                        uploadingKnowledgeBaseDocumentIds[item.id]
                          ? "处理中..."
                          : "上传文档"
                      }}
                    </button>
                  </div>

                  <div class="knowledge-upload-copy">
                    支持 TXT / MD / PDF，上传后将按当前知识库配置同步切分并写入向量库。
                  </div>

                  <div
                    v-if="loadingKnowledgeBaseDocumentIds[item.id]"
                    class="knowledge-empty knowledge-document-empty"
                  >
                    正在加载文档列表...
                  </div>
                  <div
                    v-else-if="!(knowledgeBaseDocuments[item.id] || []).length"
                    class="knowledge-empty knowledge-document-empty"
                  >
                    还没有上传文档。
                  </div>
                  <div v-else class="knowledge-document-list">
                    <article
                      v-for="document in knowledgeBaseDocuments[item.id]"
                      :key="document.id"
                      class="knowledge-document-item"
                    >
                      <div class="knowledge-document-header">
                        <strong>{{ document.original_filename }}</strong>
                        <div class="knowledge-document-actions">
                          <span
                            class="document-status"
                            :class="`status-${document.status}`"
                          >
                            {{ formatDocumentStatus(document.status) }}
                          </span>
                          <button
                            class="ghost-button danger"
                            :disabled="deletingKnowledgeBaseDocumentIds[document.id]"
                            @click="handleDeleteKnowledgeBaseDocument(item.id, document)"
                          >
                            {{
                              deletingKnowledgeBaseDocumentIds[document.id]
                                ? "删除中..."
                                : "删除"
                            }}
                          </button>
                        </div>
                      </div>
                      <div class="knowledge-document-meta">
                        <span>{{ formatFileSize(document.file_size) }}</span>
                        <span>{{ document.chunk_count }} chunks</span>
                        <span>{{ formatTime(document.updated_at) }}</span>
                      </div>
                      <p v-if="document.error_message" class="knowledge-document-error">
                        {{ document.error_message }}
                      </p>
                    </article>
                  </div>
                </div>
              </article>
            </div>

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
          </section>
        </div>
      </section>
    </main>

    <div
      v-if="sessionModalVisible"
      class="modal-scrim"
      role="presentation"
      @click.self="closeSessionModal"
    >
      <section class="session-modal" aria-modal="true" role="dialog">
        <div class="session-modal-header">
          <div>
            <p class="eyebrow chat-eyebrow">Session Setup</p>
            <h3>{{ sessionModalHeading }}</h3>
          </div>
          <button class="ghost-button" @click="closeSessionModal">关闭</button>
        </div>

        <p class="session-modal-copy">{{ sessionModalCopy }}</p>

        <label v-if="sessionModalMode === 'create'" class="field-label">
          <span>会话标题</span>
          <input
            v-model="sessionForm.title"
            class="field-input"
            maxlength="80"
            placeholder="默认会使用“新对话”"
          />
        </label>

        <div class="session-modal-section">
          <div class="session-modal-section-header">
            <strong>知识库范围</strong>
            <span>{{ sessionForm.selectedKnowledgeBaseIds.length }} 个已选</span>
          </div>

          <div v-if="sessionForm.selectedKnowledgeBaseIds.length" class="session-modal-selected">
            <span
              v-for="item in knowledgeBaseOptions.filter((option) =>
                sessionForm.selectedKnowledgeBaseIds.includes(option.id),
              )"
              :key="item.id"
              class="knowledge-chip"
            >
              {{ item.name }}
            </span>
          </div>

          <div v-if="knowledgeOptionsLoading" class="knowledge-option-empty">
            正在加载知识库选项...
          </div>
          <div v-else-if="!knowledgeBaseOptions.length" class="knowledge-option-empty">
            当前还没有可选知识库。你仍然可以先创建不绑定知识库的会话。
          </div>
          <div v-else class="knowledge-option-list">
            <label
              v-for="item in knowledgeBaseOptions"
              :key="item.id"
              class="knowledge-option-card"
              :class="{
                selected: sessionForm.selectedKnowledgeBaseIds.includes(item.id),
              }"
            >
              <input
                type="checkbox"
                :checked="sessionForm.selectedKnowledgeBaseIds.includes(item.id)"
                @change="toggleKnowledgeBaseSelection(item.id)"
              />
              <div class="knowledge-option-copy">
                <strong>{{ item.name }}</strong>
                <span>{{ item.id.slice(0, 8) }}</span>
              </div>
            </label>
          </div>
        </div>

        <div class="session-modal-actions">
          <button
            class="ghost-button"
            :disabled="savingSessionConfig || !sessionForm.selectedKnowledgeBaseIds.length"
            @click="clearSessionModalKnowledgeBases"
          >
            清空选择
          </button>
          <button class="secondary-button session-modal-secondary" @click="closeSessionModal">
            取消
          </button>
          <button
            class="primary-button session-modal-submit"
            :disabled="savingSessionConfig"
            @click="submitSessionModal"
          >
            {{ savingSessionConfig ? "处理中..." : sessionModalSubmitLabel }}
          </button>
        </div>
      </section>
    </div>
  </div>
</template>
