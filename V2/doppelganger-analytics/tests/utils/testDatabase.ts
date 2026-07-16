import Database from 'better-sqlite3';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { migrate } from '../../src/db/schema.js';

const moduleDir = path.dirname(fileURLToPath(import.meta.url));

export class TestDatabase {
  private db: Database.Database;
  private dbPath: string;

  constructor(testName: string = 'test') {
    // Create unique database for each test
    this.dbPath = path.join(moduleDir, `../temp/${testName}_${Date.now()}.db`);
    
    // Ensure temp directory exists
    const tempDir = path.dirname(this.dbPath);
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }

    this.db = new Database(this.dbPath);
    this.setupDatabase();
  }

  private setupDatabase(): void {
    // Run migrations to set up schema
    migrate(this.db);
    
    // Enable foreign keys and WAL mode for testing
    this.db.pragma('foreign_keys = ON');
    this.db.pragma('journal_mode = WAL');
  }

  public getDatabase(): Database.Database {
    return this.db;
  }

  public insertTestMessages(messages: any[]): void {
    const insertStmt = this.db.prepare(`
      INSERT INTO messages (id, conversation_id, sender, timestamp_ms, content, has_photos, has_videos, has_audio, has_share, share_link)
      VALUES (@id, @conversation_id, @sender, @timestamp_ms, @content, @has_photos, @has_videos, @has_audio, @has_share, @share_link)
    `);

    const transaction = this.db.transaction((msgs: any[]) => {
      for (const msg of msgs) {
        insertStmt.run({
          id: msg.id,
          conversation_id: msg.conversation_id,
          sender: msg.sender,
          timestamp_ms: msg.timestamp_ms,
          content: msg.content ?? null,
          has_photos: msg.has_photos ?? 0,
          has_videos: msg.has_videos ?? 0,
          has_audio: msg.has_audio ?? 0,
          has_share: msg.has_share ?? 0,
          share_link: msg.share_link ?? null
        });
      }
    });

    transaction(messages);
  }

  public insertTestReactions(reactions: any[]): void {
    const insertStmt = this.db.prepare(`
      INSERT INTO message_reactions (message_id, reaction, actor, timestamp)
      VALUES (@message_id, @reaction, @actor, @timestamp)
    `);
    const transaction = this.db.transaction((rows: any[]) => {
      for (const r of rows) {
        insertStmt.run({ message_id: r.message_id, reaction: r.reaction, actor: r.actor, timestamp: r.timestamp ?? 0 });
      }
    });
    transaction(reactions);
  }

  public insertTestPhotos(photos: any[]): void {
    const insertStmt = this.db.prepare(`
      INSERT INTO message_photos (message_id, uri, creation_timestamp, backup_uri)
      VALUES (@message_id, @uri, @creation_timestamp, @backup_uri)
    `);
    const transaction = this.db.transaction((rows: any[]) => {
      for (const p of rows) {
        insertStmt.run({ message_id: p.message_id, uri: p.uri, creation_timestamp: p.creation_timestamp ?? null, backup_uri: p.backup_uri ?? null });
      }
    });
    transaction(photos);
  }

  public insertTestSentiment(sentimentData: any[]): void {
    const insertStmt = this.db.prepare(`
      INSERT INTO sentiment (message_id, compound, positive, negative, neutral)
      VALUES (?, ?, ?, ?, ?)
    `);

    const transaction = this.db.transaction((data: any[]) => {
      for (const sentiment of data) {
        insertStmt.run(
          sentiment.message_id,
          sentiment.compound,
          sentiment.positive,
          sentiment.negative,
          sentiment.neutral
        );
      }
    });

    transaction(sentimentData);
  }

  public insertTestEmotions(emotionData: any[]): void {
    const insertStmt = this.db.prepare(`
      INSERT INTO emotions (message_id, emotion, score)
      VALUES (?, ?, ?)
    `);

    const transaction = this.db.transaction((data: any[]) => {
      for (const emotion of data) {
        insertStmt.run(
          emotion.message_id,
          emotion.emotion,
          emotion.score
        );
      }
    });

    transaction(emotionData);
  }

  public getMessageCount(): number {
    const result = this.db.prepare('SELECT COUNT(*) as count FROM messages').get() as { count: number };
    return result.count;
  }

  public getConversationIds(): string[] {
    const results = this.db.prepare('SELECT DISTINCT conversation_id FROM messages').all() as { conversation_id: string }[];
    return results.map(r => r.conversation_id);
  }

  public getSenders(): string[] {
    const results = this.db.prepare('SELECT DISTINCT sender FROM messages').all() as { sender: string }[];
    return results.map(r => r.sender);
  }

  public validateTableExists(tableName: string): boolean {
    const result = this.db.prepare(`
      SELECT name FROM sqlite_master 
      WHERE type='table' AND name=?
    `).get(tableName);
    return result !== undefined;
  }

  public validateDataIntegrity(): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check for orphaned sentiment analysis
    const orphanedSentiment = this.db.prepare(`
      SELECT COUNT(*) as count 
      FROM sentiment s 
      LEFT JOIN messages m ON s.message_id = m.id 
      WHERE m.id IS NULL
    `).get() as { count: number };

    if (orphanedSentiment.count > 0) {
      errors.push(`Found ${orphanedSentiment.count} orphaned sentiment records`);
    }

    // Check for orphaned emotion analysis
    const orphanedEmotions = this.db.prepare(`
      SELECT COUNT(*) as count 
      FROM emotions e 
      LEFT JOIN messages m ON e.message_id = m.id 
      WHERE m.id IS NULL
    `).get() as { count: number };

    if (orphanedEmotions.count > 0) {
      errors.push(`Found ${orphanedEmotions.count} orphaned emotion records`);
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  public close(): void {
    this.db.close();
  }

  public cleanup(): void {
    this.close();
    
    // Delete the test database file
    if (fs.existsSync(this.dbPath)) {
      fs.unlinkSync(this.dbPath);
    }

    // Clean up WAL and SHM files
    const walPath = this.dbPath + '-wal';
    const shmPath = this.dbPath + '-shm';
    
    if (fs.existsSync(walPath)) {
      fs.unlinkSync(walPath);
    }
    
    if (fs.existsSync(shmPath)) {
      fs.unlinkSync(shmPath);
    }
  }
}

export function createTestDatabase(testName?: string): TestDatabase {
  return new TestDatabase(testName);
}

export function loadTestFixtureMessages(): any[] {
  const fixturePath = path.join(moduleDir, '../fixtures/basic-messages.json');
  const rawMessages = JSON.parse(fs.readFileSync(fixturePath, 'utf-8'));
  
  // Convert Instagram format to processed format with IDs
  return rawMessages.map((msg: any, index: number) => ({
    id: index + 1,
    conversation_id: 'test_conversation',
    sender: msg.sender_name,
    timestamp_ms: msg.timestamp_ms,
    content: msg.content || generateContentForMedia(msg)
  }));
}

function generateContentForMedia(msg: any): string | null {
  if (msg.photos && msg.photos.length > 0) {
    return `${msg.sender_name} sent ${msg.photos.length} photo${msg.photos.length > 1 ? 's' : ''}`;
  }
  if (msg.videos && msg.videos.length > 0) {
    return `${msg.sender_name} sent ${msg.videos.length} video${msg.videos.length > 1 ? 's' : ''}`;
  }
  return null;
}

export function createMockSentimentData(messageIds: number[]): any[] {
  return messageIds.map(id => ({
    message_id: id,
    compound: Math.random() * 2 - 1, // -1 to 1
    positive: Math.random(),
    negative: Math.random(),
    neutral: Math.random()
  }));
}

export function createMockEmotionData(messageIds: number[], senders: string[], conversationIds: string[]): any[] {
  const emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise'];
  const data: any[] = [];
  
  messageIds.forEach((messageId, index) => {
    emotions.forEach(emotion => {
      data.push({
        message_id: messageId,
        emotion,
        score: Math.random(),
        sender: senders[index % senders.length],
        conversation_id: conversationIds[index % conversationIds.length]
      });
    });
  });
  
  return data;
}

// Cleanup function for all test databases
export function cleanupTestDatabases(): void {
  const tempDir = path.join(moduleDir, '../temp');
  
  if (fs.existsSync(tempDir)) {
    const files = fs.readdirSync(tempDir);
    
    files.forEach(file => {
      if (file.endsWith('.db') || file.endsWith('.db-wal') || file.endsWith('.db-shm')) {
        const filePath = path.join(tempDir, file);
        try {
          fs.unlinkSync(filePath);
        } catch (error) {
          console.warn(`Failed to cleanup test database file: ${filePath}`, error);
        }
      }
    });
  }
} 