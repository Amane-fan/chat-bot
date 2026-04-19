const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (response.status === 204) {
    return null;
  }

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "请求失败");
  }
  return data;
}

export function getSessions() {
  return request("/sessions");
}

export function createSession(title) {
  return request("/sessions", {
    method: "POST",
    body: JSON.stringify({ title }),
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

export function sendMessage(sessionId, content) {
  return request(`/sessions/${sessionId}/messages`, {
    method: "POST",
    body: JSON.stringify({ content }),
  });
}
