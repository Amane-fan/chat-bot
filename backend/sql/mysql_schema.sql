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
