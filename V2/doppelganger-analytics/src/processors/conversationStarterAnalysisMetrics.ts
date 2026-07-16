import { getDb, closeDb } from '../db/client.js';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

// A "starter" is the first message of a conversation, or the first message
// after a quiet gap of at least SESSION_GAP_MS. That makes the metric useful
// both globally and when a single long DM thread is selected.
// 4h gap: overnight pauses count as new sessions; 24h undercounted restarts badly.
const SESSION_GAP_MS = 4 * 60 * 60 * 1000;
const RESPONSE_WINDOW_MS = 24 * 60 * 60 * 1000;

interface MessageRow {
  id: number;
  conversation_id: string;
  sender: string;
  timestamp_ms: number;
  content: string | null;
}

interface ConversationStarter {
  conversation_id: string;
  starter_sender: string;
  first_message_id: number;
  first_message_content: string;
  timestamp_ms: number;
  response_count: number;
  avg_response_time: number;
  topic_keywords: string[];
  starter_type: 'question' | 'greeting' | 'announcement' | 'media' | 'other';
  engagement_score: number;
}

interface StarterPattern {
  starter_sender: string;
  total_conversations_started: number;
  avg_engagement_score: number;
  preferred_starter_types: { type: string; count: number }[];
  common_topics: string[];
  avg_response_time: number;
  success_rate: number;
  time_patterns: { hour: number; count: number }[];
  starter_tier: 'prolific' | 'active' | 'occasional' | 'rare';
}

interface TopicCorrelation {
  topic: string;
  keyword: string;
  frequency: number;
  avg_engagement: number;
  top_starters: { sender: string; count: number }[];
  success_rate: number;
}

function extractKeywords(content: string): string[] {
  if (!content) return [];

  const cleanContent = content
    .replace(/https?:\/\/[^\s]+/g, '')
    .replace(/@\w+/g, '')
    .replace(/[^\w\s'’]/g, ' ')
    .toLowerCase();

  const stopWords = new Set([
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her',
    'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man',
    'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let',
    'put', 'say', 'she', 'too', 'use'
  ]);

  return cleanContent
    .split(/\s+/)
    .filter(word => word.length >= 3 && !stopWords.has(word))
    .slice(0, 5);
}

function classifyStarterType(content: string): 'question' | 'greeting' | 'announcement' | 'media' | 'other' {
  if (!content) return 'other';

  const lowerContent = content.toLowerCase();

  if (
    lowerContent.includes('?') ||
    lowerContent.startsWith('what') ||
    lowerContent.startsWith('how') ||
    lowerContent.startsWith('when') ||
    lowerContent.startsWith('where') ||
    lowerContent.startsWith('why') ||
    lowerContent.startsWith('who') ||
    lowerContent.startsWith('can you') ||
    lowerContent.startsWith('do you') ||
    lowerContent.startsWith('did you') ||
    lowerContent.startsWith('have you') ||
    lowerContent.startsWith('are you') ||
    lowerContent.startsWith('will you')
  ) {
    return 'question';
  }

  if (
    lowerContent.includes('hello') ||
    lowerContent.includes('hi ') ||
    lowerContent.includes('hey') ||
    lowerContent.includes('good morning') ||
    lowerContent.includes('good afternoon') ||
    lowerContent.includes('good evening') ||
    lowerContent.startsWith('hi') ||
    lowerContent.startsWith('hey')
  ) {
    return 'greeting';
  }

  if (
    lowerContent.includes('photo') ||
    lowerContent.includes('image') ||
    lowerContent.includes('video') ||
    lowerContent.includes('attachment') ||
    lowerContent.includes('shared') ||
    lowerContent.includes('sent') ||
    content.includes('📷') ||
    content.includes('🎥') ||
    content.includes('📎')
  ) {
    return 'media';
  }

  if (
    lowerContent.includes('everyone') ||
    lowerContent.includes('announcement') ||
    lowerContent.includes('news') ||
    lowerContent.includes('update') ||
    lowerContent.includes('fyi') ||
    lowerContent.includes('btw') ||
    lowerContent.startsWith('just wanted to') ||
    lowerContent.startsWith('wanted to let')
  ) {
    return 'announcement';
  }

  return 'other';
}

function calculateEngagementScore(responseCount: number, avgResponseTime: number): number {
  const responseScore = Math.min(responseCount * 1, 50);
  let timeScore = 5;
  if (avgResponseTime < 5 * 60 * 1000) timeScore = 30;
  else if (avgResponseTime < 30 * 60 * 1000) timeScore = 20;
  else if (avgResponseTime < 2 * 60 * 60 * 1000) timeScore = 10;
  return Math.min(responseScore + timeScore + 20, 100);
}

function getStarterTier(sessionsStarted: number): 'prolific' | 'active' | 'occasional' | 'rare' {
  if (sessionsStarted >= 20) return 'prolific';
  if (sessionsStarted >= 10) return 'active';
  if (sessionsStarted >= 5) return 'occasional';
  return 'rare';
}

/**
 * Returns session-start messages: the first message in each conversation, plus
 * any message that follows a gap of SESSION_GAP_MS or more.
 */
export function findSessionStarters(messages: MessageRow[]): MessageRow[] {
  const starters: MessageRow[] = [];
  let previousTimestamp = -Infinity;

  for (const message of messages) {
    const gap = message.timestamp_ms - previousTimestamp;
    if (previousTimestamp === -Infinity || gap >= SESSION_GAP_MS) {
      starters.push(message);
    }
    previousTimestamp = message.timestamp_ms;
  }

  return starters;
}

function scoreSession(
  starter: MessageRow,
  conversationMessages: MessageRow[]
): { responseCount: number; avgResponseTime: number } {
  const windowEnd = starter.timestamp_ms + RESPONSE_WINDOW_MS;
  const responses = conversationMessages.filter(
    m =>
      m.id !== starter.id &&
      m.timestamp_ms > starter.timestamp_ms &&
      m.timestamp_ms <= windowEnd &&
      m.sender !== starter.sender
  );

  if (responses.length === 0) {
    return { responseCount: 0, avgResponseTime: 0 };
  }

  const avgResponseTime =
    responses.reduce((sum, m) => sum + (m.timestamp_ms - starter.timestamp_ms), 0) /
    responses.length;

  return { responseCount: responses.length, avgResponseTime };
}

export async function computeConversationStarterAnalysisMetrics(): Promise<void> {
  progressReporter.start('Computing conversation starter analysis metrics...');

  const db = await getDb();

  try {
    const conversations = db.prepare(`
      SELECT DISTINCT conversation_id FROM messages ORDER BY conversation_id
    `).all() as { conversation_id: string }[];

    progressReporter.update(`Scanning ${conversations.length} conversations for session starters...`);

    const starterMetrics: ConversationStarter[] = [];

    for (const { conversation_id } of conversations) {
      const messages = db.prepare(`
        SELECT id, conversation_id, sender, timestamp_ms, content
        FROM messages
        WHERE conversation_id = ?
        ORDER BY timestamp_ms ASC, id ASC
      `).all(conversation_id) as MessageRow[];

      if (messages.length === 0) continue;

      const sessionStarts = findSessionStarters(messages);

      for (const starter of sessionStarts) {
        // Skip empty system-ish openers when content exists elsewhere; keep
        // media-only / empty content so photo opens still count.
        const { responseCount, avgResponseTime } = scoreSession(starter, messages);
        const content = starter.content || '';
        const keywords = extractKeywords(content);
        const starterType = classifyStarterType(content);
        const engagementScore = calculateEngagementScore(responseCount, avgResponseTime);

        starterMetrics.push({
          conversation_id: starter.conversation_id,
          starter_sender: starter.sender,
          first_message_id: starter.id,
          first_message_content: content.substring(0, 200),
          timestamp_ms: starter.timestamp_ms,
          response_count: responseCount,
          avg_response_time: avgResponseTime,
          topic_keywords: keywords,
          starter_type: starterType,
          engagement_score: engagementScore
        });
      }
    }

    progressReporter.update(`Found ${starterMetrics.length.toLocaleString()} session starters`);

    const starterPatternMap = new Map<string, {
      conversations: ConversationStarter[];
      totalEngagement: number;
      typeCount: Map<string, number>;
      topicKeywords: string[];
      timePatterns: Map<number, number>;
    }>();

    for (const starter of starterMetrics) {
      if (!starterPatternMap.has(starter.starter_sender)) {
        starterPatternMap.set(starter.starter_sender, {
          conversations: [],
          totalEngagement: 0,
          typeCount: new Map(),
          topicKeywords: [],
          timePatterns: new Map()
        });
      }

      const pattern = starterPatternMap.get(starter.starter_sender)!;
      pattern.conversations.push(starter);
      pattern.totalEngagement += starter.engagement_score;
      pattern.typeCount.set(
        starter.starter_type,
        (pattern.typeCount.get(starter.starter_type) || 0) + 1
      );
      pattern.topicKeywords.push(...starter.topic_keywords);
      const hour = new Date(starter.timestamp_ms).getHours();
      pattern.timePatterns.set(hour, (pattern.timePatterns.get(hour) || 0) + 1);
    }

    const starterPatterns: StarterPattern[] = Array.from(starterPatternMap.entries())
      .map(([sender, data]) => {
        const avgEngagement = data.totalEngagement / data.conversations.length;
        const avgResponseTime =
          data.conversations.reduce((sum, c) => sum + c.avg_response_time, 0) /
          data.conversations.length;
        const successRate =
          (data.conversations.filter(c => c.response_count > 0).length /
            data.conversations.length) *
          100;

        return {
          starter_sender: sender,
          total_conversations_started: data.conversations.length,
          avg_engagement_score: Math.round(avgEngagement * 10) / 10,
          preferred_starter_types: Array.from(data.typeCount.entries())
            .map(([type, count]) => ({ type, count }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 3),
          common_topics: (() => {
            const keywordCount = new Map<string, number>();
            data.topicKeywords.forEach(keyword => {
              keywordCount.set(keyword, (keywordCount.get(keyword) || 0) + 1);
            });
            return Array.from(keywordCount.entries())
              .sort((a, b) => b[1] - a[1])
              .slice(0, 5)
              .map(([keyword]) => keyword);
          })(),
          avg_response_time: Math.round(avgResponseTime),
          success_rate: Math.round(successRate * 10) / 10,
          time_patterns: Array.from(data.timePatterns.entries())
            .map(([hour, count]) => ({ hour, count }))
            .sort((a, b) => b.count - a.count),
          starter_tier: getStarterTier(data.conversations.length)
        };
      })
      .sort((a, b) => b.total_conversations_started - a.total_conversations_started);

    const topicMap = new Map<string, {
      frequency: number;
      totalEngagement: number;
      starters: Map<string, number>;
      successCount: number;
    }>();

    for (const starter of starterMetrics) {
      for (const keyword of starter.topic_keywords) {
        if (!topicMap.has(keyword)) {
          topicMap.set(keyword, {
            frequency: 0,
            totalEngagement: 0,
            starters: new Map(),
            successCount: 0
          });
        }
        const topic = topicMap.get(keyword)!;
        topic.frequency++;
        topic.totalEngagement += starter.engagement_score;
        topic.starters.set(
          starter.starter_sender,
          (topic.starters.get(starter.starter_sender) || 0) + 1
        );
        if (starter.response_count > 0) topic.successCount++;
      }
    }

    const topicCorrelations: TopicCorrelation[] = Array.from(topicMap.entries())
      .filter(([, data]) => data.frequency >= 3)
      .map(([keyword, data]) => ({
        topic: keyword,
        keyword,
        frequency: data.frequency,
        avg_engagement: Math.round((data.totalEngagement / data.frequency) * 10) / 10,
        top_starters: Array.from(data.starters.entries())
          .map(([sender, count]) => ({ sender, count }))
          .sort((a, b) => b.count - a.count)
          .slice(0, 3),
        success_rate: Math.round((data.successCount / data.frequency) * 1000) / 10
      }))
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 20);

    const hourlyPatterns = new Map<number, {
      count: number;
      totalEngagement: number;
      starterCount: Map<string, number>;
    }>();

    for (const starter of starterMetrics) {
      const hour = new Date(starter.timestamp_ms).getHours();
      if (!hourlyPatterns.has(hour)) {
        hourlyPatterns.set(hour, {
          count: 0,
          totalEngagement: 0,
          starterCount: new Map()
        });
      }
      const pattern = hourlyPatterns.get(hour)!;
      pattern.count++;
      pattern.totalEngagement += starter.engagement_score;
      pattern.starterCount.set(
        starter.starter_sender,
        (pattern.starterCount.get(starter.starter_sender) || 0) + 1
      );
    }

    const temporalPatterns = Array.from(hourlyPatterns.entries())
      .map(([hour, data]) => {
        const topStarter =
          Array.from(data.starterCount.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] || 'N/A';
        return {
          hour,
          starters_count: data.count,
          avg_engagement: Math.round((data.totalEngagement / data.count) * 10) / 10,
          top_starter: topStarter
        };
      })
      .sort((a, b) => a.hour - b.hour);

    const avgEngagement =
      starterMetrics.length > 0
        ? starterMetrics.reduce((sum, s) => sum + s.engagement_score, 0) / starterMetrics.length
        : 0;
    const avgResponseTime =
      starterMetrics.length > 0
        ? starterMetrics.reduce((sum, s) => sum + s.avg_response_time, 0) / starterMetrics.length
        : 0;

    // Export EVERY session starter — filtering by conversation_id must work.
    // Cap only the example content list if extremely large; 50k is still fine.
    const MAX_STARTERS = 50000;
    const exportedStarters =
      starterMetrics.length > MAX_STARTERS
        ? starterMetrics
            .slice()
            .sort((a, b) => b.engagement_score - a.engagement_score)
            .slice(0, MAX_STARTERS)
        : starterMetrics;

    writeDashData('conversationStarterAnalysis.json', {
      summary: {
        total_conversations: conversations.length,
        total_starters: starterMetrics.length,
        avg_engagement_per_starter: Math.round(avgEngagement),
        most_prolific_starter: starterPatterns[0]?.starter_sender || 'N/A',
        most_engaging_topic: topicCorrelations[0]?.topic || 'N/A',
        avg_response_time: Math.round(avgResponseTime)
      },
      conversation_starters: exportedStarters,
      starter_patterns: starterPatterns,
      topic_correlations: topicCorrelations,
      temporal_patterns: temporalPatterns
    });

    progressReporter.success('Conversation starter analysis metrics computed and exported.');
    progressReporter.update(`${starterMetrics.length.toLocaleString()} session starters across ${conversations.length} conversations`);
    progressReporter.update(`Most prolific starter: ${starterPatterns[0]?.starter_sender || 'N/A'}`);
  } catch (error) {
    console.error('Error computing conversation starter analysis metrics:', error);
    throw error;
  } finally {
    await closeDb(db);
  }
}
