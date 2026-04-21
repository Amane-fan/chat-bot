CREATE TABLE IF NOT EXISTS chat_sessions (
  id CHAR(32) NOT NULL,
  title VARCHAR(80) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  PRIMARY KEY (id),
  KEY ix_chat_sessions_updated_at (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS chat_messages (
  id INT NOT NULL AUTO_INCREMENT,
  session_id CHAR(32) NOT NULL,
  role VARCHAR(20) NOT NULL,
  content TEXT NOT NULL,
  created_at DATETIME(6) NOT NULL,
  PRIMARY KEY (id),
  KEY ix_chat_messages_session_created (session_id, created_at, id),
  CONSTRAINT fk_chat_messages_session_id
    FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS knowledge_bases (
  id CHAR(32) NOT NULL,
  name VARCHAR(80) NOT NULL,
  description TEXT NULL,
  config JSON NOT NULL,
  document_count INT NOT NULL DEFAULT 0,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_knowledge_bases_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS knowledge_base_documents (
  id CHAR(32) NOT NULL,
  knowledge_base_id CHAR(32) NOT NULL,
  original_filename VARCHAR(255) NOT NULL,
  stored_filename VARCHAR(255) NOT NULL,
  content_type VARCHAR(120) NULL,
  file_size INT NOT NULL,
  storage_path VARCHAR(1000) NOT NULL,
  status VARCHAR(20) NOT NULL,
  chunk_count INT NOT NULL DEFAULT 0,
  error_message TEXT NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_knowledge_base_documents_name (knowledge_base_id, original_filename),
  KEY ix_knowledge_base_documents_knowledge_base_id (knowledge_base_id),
  KEY ix_knowledge_base_documents_status (status),
  CONSTRAINT fk_knowledge_base_documents_knowledge_base_id
    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases (id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS session_knowledge_bases (
  session_id CHAR(32) NOT NULL,
  knowledge_base_id CHAR(32) NOT NULL,
  sort_order INT NOT NULL,
  created_at DATETIME(6) NOT NULL,
  PRIMARY KEY (session_id, knowledge_base_id),
  KEY ix_session_knowledge_bases_session_id (session_id),
  KEY ix_session_knowledge_bases_knowledge_base_id (knowledge_base_id),
  CONSTRAINT fk_session_knowledge_bases_session_id
    FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
    ON DELETE CASCADE,
  CONSTRAINT fk_session_knowledge_bases_knowledge_base_id
    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases (id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
