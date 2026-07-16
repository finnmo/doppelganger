// src/db/schema.ts
import { Database } from 'better-sqlite3';

export function migrate(db: Database) {
  const stmts = [

    // Key-value store for pipeline metadata (e.g. which sentiment engine
    // produced the current scores). Used to auto-invalidate derived data.
    `CREATE TABLE IF NOT EXISTS meta (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );`,

    // Raw messages, platform-neutral. `source` identifies the importing
    // platform ('instagram', ...). (Instagram DM exports carry no reply/quote
    // links, so there is no reply_to_message_id column — threads are derived
    // from timing.)
    `CREATE TABLE IF NOT EXISTS messages (
      id                INTEGER PRIMARY KEY AUTOINCREMENT,
      conversation_id   TEXT    NOT NULL,
      sender            TEXT    NOT NULL,
      timestamp_ms      INTEGER NOT NULL,
      content           TEXT,
      has_photos        INTEGER DEFAULT 0,
      has_videos        INTEGER DEFAULT 0,
      has_audio         INTEGER DEFAULT 0,
      has_share         INTEGER DEFAULT 0,
      share_link        TEXT,
      source            TEXT    NOT NULL DEFAULT 'instagram',
      is_system         INTEGER NOT NULL DEFAULT 0
    );`,

    // Sentiment metrics
    `CREATE TABLE IF NOT EXISTS sentiment (
      message_id  INTEGER NOT NULL,
      compound    REAL    NOT NULL,
      positive    REAL    NOT NULL,
      negative    REAL    NOT NULL,
      neutral     REAL    NOT NULL,
      FOREIGN KEY(message_id) REFERENCES messages(id)
    );`,

    // Granular emotions (one row per detected emotion)
    `CREATE TABLE IF NOT EXISTS emotions (
      message_id  INTEGER NOT NULL,
      emotion     TEXT    NOT NULL,
      score       REAL    NOT NULL,
      FOREIGN KEY(message_id) REFERENCES messages(id)
    );`,

    // Conversation-level metrics
    `CREATE TABLE IF NOT EXISTS conversations (
      conversation_id   TEXT    PRIMARY KEY,
      participants      TEXT    NOT NULL,   -- JSON array of names
      first_message_ms  INTEGER NOT NULL,
      last_message_ms   INTEGER NOT NULL,
      total_messages    INTEGER NOT NULL
    );`,

    // Response latency between consecutive cross-sender messages. Derived data,
    // rebuilt on every run (see processors/responseTimes.ts). conversation_id is
    // stored for grouping but not foreign-keyed: the conversations table is not
    // populated in this pipeline.
    `CREATE TABLE IF NOT EXISTS response_times (
      id                INTEGER PRIMARY KEY AUTOINCREMENT,
      conversation_id   TEXT    NOT NULL,
      from_message_id   INTEGER NOT NULL,
      to_message_id     INTEGER NOT NULL,
      latency_ms        INTEGER NOT NULL,
      FOREIGN KEY(from_message_id) REFERENCES messages(id),
      FOREIGN KEY(to_message_id)   REFERENCES messages(id)
    );`,

    // Text‐based metrics
    `CREATE TABLE IF NOT EXISTS text_metrics (
        message_id  INTEGER PRIMARY KEY,
        word_count  INTEGER NOT NULL,
        emoji_count INTEGER NOT NULL,
        url_count   INTEGER NOT NULL,
        FOREIGN KEY(message_id) REFERENCES messages(id)
    );`,

    // Media tables
    `CREATE TABLE IF NOT EXISTS message_photos (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      message_id INTEGER NOT NULL,
      uri TEXT NOT NULL,
      creation_timestamp INTEGER,
      backup_uri TEXT,
      FOREIGN KEY (message_id) REFERENCES messages(id)
    );`,

    `CREATE TABLE IF NOT EXISTS message_videos (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      message_id INTEGER NOT NULL,
      uri TEXT NOT NULL,
      creation_timestamp INTEGER,
      backup_uri TEXT,
      FOREIGN KEY (message_id) REFERENCES messages(id)
    );`,

    `CREATE TABLE IF NOT EXISTS message_reactions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      message_id INTEGER NOT NULL,
      reaction TEXT NOT NULL,
      actor TEXT NOT NULL,
      timestamp INTEGER NOT NULL,
      FOREIGN KEY (message_id) REFERENCES messages(id)
    );`,

    `CREATE TABLE IF NOT EXISTS message_audio (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      message_id INTEGER NOT NULL,
      uri TEXT NOT NULL,
      creation_timestamp INTEGER,
      FOREIGN KEY (message_id) REFERENCES messages(id)
    );`,

    `CREATE INDEX IF NOT EXISTS idx_text_metrics_message
        ON text_metrics(message_id);`,

    // Simple index to speed up time-based queries
    `CREATE INDEX IF NOT EXISTS idx_messages_timestamp 
       ON messages(timestamp_ms);`,
    `CREATE INDEX IF NOT EXISTS idx_messages_conv
       ON messages(conversation_id);`,
    `CREATE INDEX IF NOT EXISTS idx_sentiment_message
       ON sentiment(message_id);`,
    `CREATE INDEX IF NOT EXISTS idx_emotions_message 
       ON emotions(message_id);`,
    `CREATE INDEX IF NOT EXISTS idx_response_times_conv 
       ON response_times(conversation_id);`,

    // Vector embeddings for persona RAG (generate-time; optional if no API key)
    `CREATE TABLE IF NOT EXISTS message_embeddings (
      message_id  INTEGER PRIMARY KEY,
      provider    TEXT    NOT NULL,
      model       TEXT    NOT NULL,
      dims        INTEGER NOT NULL,
      embedding   BLOB    NOT NULL,
      FOREIGN KEY(message_id) REFERENCES messages(id)
    );`,

    `CREATE INDEX IF NOT EXISTS idx_message_embeddings_model
       ON message_embeddings(model);`
  ];

  db.exec('BEGIN');
  for (const sql of stmts) {
    db.exec(sql);
  }
  db.exec('COMMIT');

  // Add columns introduced after the messages table already existed in the
  // wild (CREATE TABLE IF NOT EXISTS leaves an existing table untouched).
  addColumnIfMissing(db, 'messages', 'has_audio', 'INTEGER DEFAULT 0');
  addColumnIfMissing(db, 'messages', 'has_share', 'INTEGER DEFAULT 0');
  addColumnIfMissing(db, 'messages', 'share_link', 'TEXT');
  addColumnIfMissing(db, 'messages', 'source', `TEXT NOT NULL DEFAULT 'instagram'`);
  addColumnIfMissing(db, 'messages', 'is_system', 'INTEGER NOT NULL DEFAULT 0');
}

function addColumnIfMissing(db: Database, table: string, column: string, definition: string): void {
  const columns = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>;
  if (!columns.some(c => c.name === column)) {
    db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition}`);
  }
}
