// src/processors/conversation.ts
import { getDb, closeDb } from '../db/client.js';
import chalk from 'chalk';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';
import { parseConversationId, sourceLabel } from '../utils/platformSource.js';

interface ConversationMetrics {
  conversation_id: string;
  source: string;
  turns: number;
  participants: string[];
  duration_ms: number;
  message_count: number;
  avg_response_time: number;
  messages_by_sender: Record<string, number>;
}

export async function analyzeConversations(): Promise<void> {
  progressReporter.start('Analyzing conversations...');
  const db = await getDb();

  try {
    const conversationSummaries = db.prepare(`
      SELECT 
        conversation_id,
        source,
        COUNT(*) as message_count,
        GROUP_CONCAT(DISTINCT sender) as participants,
        MIN(timestamp_ms) as first_message_ms,
        MAX(timestamp_ms) as last_message_ms
      FROM messages 
      GROUP BY conversation_id, source
      ORDER BY message_count DESC
    `).all() as Array<{
      conversation_id: string;
      source: string;
      message_count: number;
      participants: string;
      first_message_ms: number;
      last_message_ms: number;
    }>;

    progressReporter.update(`Found ${conversationSummaries.length} conversations`);

    const avgResponseByConversation = new Map<string, number>();
    for (const row of db.prepare(`
      SELECT conversation_id, AVG(latency_ms) as avg_latency
      FROM response_times
      GROUP BY conversation_id
    `).all() as Array<{ conversation_id: string; avg_latency: number }>) {
      avgResponseByConversation.set(row.conversation_id, row.avg_latency);
    }

    const turnsByConversation = new Map<string, number>();
    for (const row of db.prepare(`
      SELECT conversation_id, COUNT(*) - 1 as turns
      FROM messages
      GROUP BY conversation_id
    `).all() as Array<{ conversation_id: string; turns: number }>) {
      turnsByConversation.set(row.conversation_id, Math.max(0, row.turns));
    }

    const messagesBySender = new Map<string, Record<string, number>>();
    for (const row of db.prepare(`
      SELECT conversation_id, sender, COUNT(*) as count
      FROM messages
      GROUP BY conversation_id, sender
    `).all() as Array<{ conversation_id: string; sender: string; count: number }>) {
      let map = messagesBySender.get(row.conversation_id);
      if (!map) {
        map = {};
        messagesBySender.set(row.conversation_id, map);
      }
      map[row.sender] = row.count;
    }

    const results: Record<string, ConversationMetrics> = {};

    progressReporter.update('Initializing conversations...');
    for (const summary of conversationSummaries) {
      const participantsList = summary.participants.split(',').map(p => p.trim());
      const parsed = parseConversationId(summary.conversation_id);
      const source = summary.source || parsed.source || 'unknown';
      results[summary.conversation_id] = {
        conversation_id: summary.conversation_id,
        source,
        turns: turnsByConversation.get(summary.conversation_id) || 0,
        participants: participantsList,
        duration_ms: summary.last_message_ms - summary.first_message_ms,
        message_count: summary.message_count,
        avg_response_time: Math.round(avgResponseByConversation.get(summary.conversation_id) || 0),
        messages_by_sender: messagesBySender.get(summary.conversation_id) || {}
      };
    }

    progressReporter.update('Exporting metrics...');

    const allParticipants = new Set<string>();
    Object.values(results).forEach(conv => {
      conv.participants.forEach(participant => {
        allParticipants.add(participant);
      });
    });

    const conversationList = Object.values(results);
    const totalMessages = conversationList.reduce((sum, conv) => sum + conv.message_count, 0);

    const bySource = new Map<string, { conversations: number; messages: number }>();
    for (const conv of conversationList) {
      const entry = bySource.get(conv.source) ?? { conversations: 0, messages: 0 };
      entry.conversations += 1;
      entry.messages += conv.message_count;
      bySource.set(conv.source, entry);
    }

    const metrics = {
      conversations: conversationList.map(conv => ({
        conversation_id: conv.conversation_id,
        source: conv.source,
        source_label: sourceLabel(conv.source),
        participants: conv.participants,
        first_message_ms: 0,
        last_message_ms: 0,
        total_messages: conv.message_count,
        turns: conv.turns,
        duration_ms: conv.duration_ms,
        avg_response_time: conv.avg_response_time,
        messages_by_sender: conv.messages_by_sender
      })),
      summary: {
        totalConversations: conversationList.length,
        totalUniqueParticipants: allParticipants.size,
        averageTurns: conversationList.reduce((sum, conv) => sum + conv.turns, 0) / conversationList.length,
        averageDuration: conversationList.reduce((sum, conv) => sum + conv.duration_ms, 0) / conversationList.length,
        averageParticipants: conversationList.reduce((sum, conv) => sum + conv.participants.length, 0) / conversationList.length,
        messagesProcessed: totalMessages,
        platforms: [...bySource.entries()]
          .map(([source, stats]) => ({
            source,
            label: sourceLabel(source),
            conversations: stats.conversations,
            messages: stats.messages
          }))
          .sort((a, b) => b.messages - a.messages)
      }
    };

    writeDashData('conversationMetrics.json', metrics);

    progressReporter.success(`Conversation metrics computed for ${Object.keys(results).length} conversations.`);
    progressReporter.update(`Conversations: ${Object.keys(results).slice(0, 3).join(', ')}${Object.keys(results).length > 3 ? '...' : ''}`);
  } catch (error) {
    console.error(chalk.red('❌ Error computing conversation metrics:'), error);
    throw error;
  } finally {
    await closeDb(db);
  }
}
