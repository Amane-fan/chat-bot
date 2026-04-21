const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (response.status === 204) {
    return null;
  }

  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    throw new Error(data?.detail || data || "请求失败");
  }
  return data;
}

export function getSessions() {
  return request("/sessions");
}

export function createSession(payload = {}) {
  return request("/sessions", {
    method: "POST",
    body: JSON.stringify({
      title: payload.title,
      knowledge_base_ids: payload.knowledgeBaseIds || [],
    }),
  });
}

export function renameSession(sessionId, title) {
  return request(`/sessions/${sessionId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export function deleteSession(sessionId) {
  return request(`/sessions/${sessionId}`, {
    method: "DELETE",
  });
}

export function getMessages(sessionId) {
  return request(`/sessions/${sessionId}/messages`);
}

export function updateSessionKnowledgeBases(sessionId, knowledgeBaseIds) {
  return request(`/sessions/${sessionId}/knowledge-bases`, {
    method: "PUT",
    body: JSON.stringify({ knowledge_base_ids: knowledgeBaseIds }),
  });
}

export function sendMessage(sessionId, content) {
  return request(`/sessions/${sessionId}/messages`, {
    method: "POST",
    body: JSON.stringify({ content }),
  });
}

export function getKnowledgeBaseOptions() {
  return request("/knowledge-bases/options");
}

export function getKnowledgeBases(page = 1, pageSize = 10) {
  return request(`/knowledge-bases?page=${page}&page_size=${pageSize}`);
}

export function createKnowledgeBase(payload) {
  return request("/knowledge-bases", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getKnowledgeBaseDocuments(knowledgeBaseId) {
  return request(`/knowledge-bases/${knowledgeBaseId}/documents`);
}

export function uploadKnowledgeBaseDocument(knowledgeBaseId, file) {
  const formData = new FormData();
  formData.append("file", file);

  return request(`/knowledge-bases/${knowledgeBaseId}/documents`, {
    method: "POST",
    body: formData,
  });
}

export function deleteKnowledgeBaseDocument(knowledgeBaseId, documentId) {
  return request(`/knowledge-bases/${knowledgeBaseId}/documents/${documentId}`, {
    method: "DELETE",
  });
}
