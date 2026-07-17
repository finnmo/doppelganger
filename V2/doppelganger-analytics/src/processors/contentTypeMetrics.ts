import { getDb, closeDb } from '../db/client.js';
import { decodeInstagramUnicode, extractEmojis, detectReaction } from '../utils/unicodeDecoder.js';
import { trimMessageSnippet, isSystemMessage } from '../utils/messageFilters.js';
import chalk from 'chalk';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

interface ContentTypeData {
  type: string;
  count: number;
  totalLength: number;
  examples: string[];
  senders: Set<string>;
  representativeExample?: {
    text: string;
    sender: string;
    conversation_id: string;
  };
}

export function classifyMessage(content: string, hasMedia: boolean): string {
  if (detectReaction(content).isReaction) {
    return 'reaction';
  }
  if (hasMedia || content.includes('sent an attachment')) {
    return 'media_notification';
  }
  if (content.includes('started a call') || content.includes('ended the call') || content.includes('missed call')) {
    return 'call_event';
  }
  if (content.includes('You are now connected on Messenger') || content.includes('joined the group')) {
    return 'system_event';
  }
  if (content.includes('http://') || content.includes('https://')) {
    return 'link_share';
  }

  const emojiInfo = extractEmojis(content);
  const wordCount = content.split(/\s+/).filter(word => word.length > 0).length;

  if (emojiInfo.count > 0 && wordCount === 0) return 'emoji_only';
  if (content.match(/^\s*[!@#$%^&*()_+\-=[\]{}|;:'"<>?,./]+\s*$/)) return 'symbols_only';
  if (wordCount === 1) return 'single_word';
  if (wordCount <= 5) return 'short_text';
  if (wordCount <= 20) return 'medium_text';
  return 'long_text';
}

const NON_EXAMPLE_TYPES = new Set(['reaction', 'media_notification', 'call_event', 'system_event']);

interface ContentTypeResult {
  conversation_id: string;
  type: string;
  count: number;
  percentage: number;
  examples: string[];
  avgLength: number;
  uniqueSenders: number;
  representativeExample?: {
    text: string;
    sender: string;
    conversation_id: string;
  };
}

export async function computeContentTypeMetrics(): Promise<void> {
  progressReporter.start('Computing content type metrics...');
  const db = await getDb();

  try {
    const messages = db.prepare(`
      SELECT
        id,
        content,
        sender,
        conversation_id,
        timestamp_ms,
        has_photos,
        has_videos,
        is_system
      FROM messages
      WHERE content IS NOT NULL
      ORDER BY conversation_id, id DESC
    `).all() as Array<{
      id: number;
      content: string;
      sender: string;
      conversation_id: string;
      timestamp_ms: number;
      has_photos: number;
      has_videos: number;
      is_system: number;
    }>;

    progressReporter.update(`Analyzing ${messages.length.toLocaleString()} messages for content types...`);

    const conversationContentTypes = new Map<string, Map<string, ContentTypeData>>();
    const conversationMessageCounts = new Map<string, number>();
    const representativeExamples: Array<{
      type: string;
      text: string;
      sender: string;
      conversation_id: string;
    }> = [];

    for (const message of messages) {
      if (isSystemMessage(message.content, message.is_system)) continue;

      const content = decodeInstagramUnicode(message.content.trim());
      const hasMedia = message.has_photos === 1 || message.has_videos === 1;
      const messageType = classifyMessage(content, hasMedia);

      conversationMessageCounts.set(
        message.conversation_id,
        (conversationMessageCounts.get(message.conversation_id) || 0) + 1
      );

      if (!conversationContentTypes.has(message.conversation_id)) {
        conversationContentTypes.set(message.conversation_id, new Map<string, ContentTypeData>());
      }
      const conversationTypes = conversationContentTypes.get(message.conversation_id)!;

      if (!conversationTypes.has(messageType)) {
        conversationTypes.set(messageType, {
          type: messageType,
          count: 0,
          totalLength: 0,
          examples: [],
          senders: new Set<string>()
        });
      }

      const typeData = conversationTypes.get(messageType)!;
      typeData.count++;
      typeData.totalLength += message.content.length;
      typeData.senders.add(message.sender);

      if (typeData.examples.length < 3) {
        typeData.examples.push(content.length > 100 ? content.substring(0, 100) + '...' : content);
      }

      if (!typeData.representativeExample && !NON_EXAMPLE_TYPES.has(messageType)) {
        const example = {
          text: trimMessageSnippet(content, 120),
          sender: message.sender,
          conversation_id: message.conversation_id
        };
        typeData.representativeExample = example;
        representativeExamples.push({ type: messageType, ...example });
      }
    }

    const results: ContentTypeResult[] = [];

    for (const [conversationId, contentTypes] of conversationContentTypes.entries()) {
      const totalMessages = conversationMessageCounts.get(conversationId) || 0;

      for (const [type, data] of contentTypes.entries()) {
        results.push({
          conversation_id: conversationId,
          type,
          count: data.count,
          percentage: totalMessages > 0 ? (data.count / totalMessages) * 100 : 0,
          examples: data.examples,
          avgLength: data.count > 0 ? Math.round(data.totalLength / data.count) : 0,
          uniqueSenders: data.senders.size,
          representativeExample: data.representativeExample
        });
      }
    }

    results.sort((a, b) => {
      if (a.conversation_id !== b.conversation_id) {
        return a.conversation_id.localeCompare(b.conversation_id);
      }
      return b.count - a.count;
    });

    const totalMessages = messages.filter(m => !isSystemMessage(m.content, m.is_system)).length;
    const totalLength = messages.reduce((sum, msg) => sum + msg.content.length, 0);
    const uniqueTypes = new Set(results.map(r => r.type)).size;

    const contentTypeData = {
      summary: {
        totalMessages,
        totalTypes: uniqueTypes,
        avgMessageLength: totalMessages > 0 ? Math.round(totalLength / totalMessages) : 0,
        totalConversations: conversationContentTypes.size
      },
      contentTypes: results,
      representativeExamples
    };

    writeDashData('contentTypeMetrics.json', contentTypeData);

    progressReporter.success(`Content type metrics computed and exported (${totalMessages.toLocaleString()} messages analyzed).`);
    progressReporter.update(`Found ${uniqueTypes} different content types across ${conversationContentTypes.size} conversations`);
    progressReporter.update(`Total records: ${results.length}`);

  } catch (error) {
    console.error(chalk.red('❌ Error computing content type metrics:'), error);
    throw error;
  } finally {
    closeDb(db);
  }
}
