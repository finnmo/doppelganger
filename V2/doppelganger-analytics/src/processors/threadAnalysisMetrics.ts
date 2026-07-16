import { getDb, closeDb } from '../db/client.js';
import chalk from 'chalk';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

// Instagram DM exports contain no reply/quote links, so threads can't be
// reconstructed from explicit references. Instead a "thread" is a burst: a run
// of messages in one conversation where each consecutive gap is under this
// threshold. Its depth is the number of messages in the burst.
const THREAD_GAP_MS = 5 * 60 * 1000;
const MIN_THREAD_LENGTH = 2;

interface MessageRow {
  id: number;
  conversation_id: string;
  sender: string;
  timestamp_ms: number;
}

interface ThreadData {
  conversation_id: string;
  thread_depth: number;
  thread_count: number;
  avg_thread_length: number;
  max_thread_depth: number;
  participants_in_threads: number;
  depth_distribution: ThreadDepthDistribution[];
  thread_starters: Array<{
    sender: string;
    threads_started: number;
  }>;
}

interface ThreadDepthDistribution {
  depth: number;
  count: number;
  percentage: number;
}

interface ThreadMetrics {
  summary: {
    total_threads: number;
    avg_depth: number;
    max_depth: number;
    total_conversations: number;
    threaded_conversations: number;
  };
  depth_distribution: ThreadDepthDistribution[];
  conversation_threads: ThreadData[];
  top_thread_starters: Array<{
    sender: string;
    threads_started: number;
    avg_thread_depth: number;
  }>;
}

interface Burst {
  starter: string;
  senders: Set<string>;
  length: number;
}

/**
 * Segments a conversation's chronological messages into bursts separated by
 * gaps larger than THREAD_GAP_MS. Only bursts of at least MIN_THREAD_LENGTH
 * messages count as threads.
 */
function findBursts(messages: MessageRow[]): Burst[] {
  const bursts: Burst[] = [];
  let current: { starter: string; senders: Set<string>; length: number } | null = null;
  let previousTimestamp = 0;

  for (const message of messages) {
    const gap = current ? message.timestamp_ms - previousTimestamp : Infinity;

    if (!current || gap > THREAD_GAP_MS) {
      if (current && current.length >= MIN_THREAD_LENGTH) {
        bursts.push(current);
      }
      current = { starter: message.sender, senders: new Set([message.sender]), length: 1 };
    } else {
      current.length++;
      current.senders.add(message.sender);
    }
    previousTimestamp = message.timestamp_ms;
  }

  if (current && current.length >= MIN_THREAD_LENGTH) {
    bursts.push(current);
  }

  return bursts;
}

export async function computeThreadAnalysisMetrics(): Promise<void> {
  progressReporter.start('Computing thread analysis metrics...');
  const db = await getDb();

  try {
    const messages = db.prepare(`
      SELECT id, conversation_id, sender, timestamp_ms
      FROM messages
      ORDER BY conversation_id, timestamp_ms
    `).all() as MessageRow[];

    progressReporter.update(`Analyzing ${messages.length.toLocaleString()} messages for thread patterns...`);

    // Group messages by conversation (already ordered by conversation, time)
    const conversationGroups = new Map<string, MessageRow[]>();
    for (const message of messages) {
      let group = conversationGroups.get(message.conversation_id);
      if (!group) {
        group = [];
        conversationGroups.set(message.conversation_id, group);
      }
      group.push(message);
    }

    const conversationThreads: ThreadData[] = [];
    const allThreadDepths: number[] = [];
    const threadStarterMap = new Map<string, { count: number; totalDepth: number }>();

    for (const [conversationId, convMessages] of conversationGroups.entries()) {
      const bursts = findBursts(convMessages);

      const depths = bursts.map(b => b.length);
      const maxThreadDepth = depths.length > 0 ? Math.max(...depths) : 0;
      const avgThreadLength = depths.length > 0
        ? depths.reduce((a, b) => a + b, 0) / depths.length
        : 0;

      const participantsInThreads = new Set<string>();
      const convThreadStarters = new Map<string, number>();
      const convDepthCounts = new Map<number, number>();

      for (const burst of bursts) {
        allThreadDepths.push(burst.length);
        convDepthCounts.set(burst.length, (convDepthCounts.get(burst.length) || 0) + 1);
        burst.senders.forEach(s => participantsInThreads.add(s));
        convThreadStarters.set(burst.starter, (convThreadStarters.get(burst.starter) || 0) + 1);

        const starterData = threadStarterMap.get(burst.starter) || { count: 0, totalDepth: 0 };
        starterData.count++;
        starterData.totalDepth += burst.length;
        threadStarterMap.set(burst.starter, starterData);
      }

      const convDepthDistribution: ThreadDepthDistribution[] = Array.from(convDepthCounts.entries())
        .map(([depth, count]) => ({
          depth,
          count,
          percentage: bursts.length > 0 ? (count / bursts.length) * 100 : 0
        }))
        .sort((a, b) => a.depth - b.depth);

      conversationThreads.push({
        conversation_id: conversationId,
        thread_depth: avgThreadLength,
        thread_count: bursts.length,
        avg_thread_length: avgThreadLength,
        max_thread_depth: maxThreadDepth,
        participants_in_threads: participantsInThreads.size,
        depth_distribution: convDepthDistribution,
        thread_starters: Array.from(convThreadStarters.entries()).map(([sender, count]) => ({
          sender,
          threads_started: count
        }))
      });
    }

    // Depth distribution across all threads
    const depthCounts = new Map<number, number>();
    allThreadDepths.forEach(depth => {
      depthCounts.set(depth, (depthCounts.get(depth) || 0) + 1);
    });
    const totalThreads = allThreadDepths.length;
    const depthDistribution: ThreadDepthDistribution[] = Array.from(depthCounts.entries())
      .map(([depth, count]) => ({
        depth,
        count,
        percentage: totalThreads > 0 ? (count / totalThreads) * 100 : 0
      }))
      .sort((a, b) => a.depth - b.depth);

    const topThreadStarters = Array.from(threadStarterMap.entries())
      .map(([sender, data]) => ({
        sender,
        threads_started: data.count,
        avg_thread_depth: data.count > 0 ? data.totalDepth / data.count : 0
      }))
      .sort((a, b) => b.threads_started - a.threads_started)
      .slice(0, 20);

    const avgDepth = totalThreads > 0 ? allThreadDepths.reduce((a, b) => a + b, 0) / totalThreads : 0;
    const maxDepth = totalThreads > 0 ? Math.max(...allThreadDepths) : 0;
    const threadedConversations = conversationThreads.filter(conv => conv.thread_count > 0).length;


    const threadMetrics: ThreadMetrics = {
      summary: {
        total_threads: totalThreads,
        avg_depth: Math.round(avgDepth * 10) / 10,
        max_depth: maxDepth,
        total_conversations: conversationGroups.size,
        threaded_conversations: threadedConversations
      },
      depth_distribution: depthDistribution,
      conversation_threads: conversationThreads,
      top_thread_starters: topThreadStarters
    };

    writeDashData('threadAnalysis.json', threadMetrics);

    progressReporter.success('Thread analysis metrics computed and exported.');
    progressReporter.update(`Analyzed ${conversationGroups.size} conversations`);
    progressReporter.update(`Found ${totalThreads} bursts (threads)`);
    progressReporter.update(`${threadedConversations} conversations have threads`);
    progressReporter.update(`Maximum thread depth: ${maxDepth}`);
    progressReporter.update(`Average thread depth: ${avgDepth.toFixed(2)}`);

  } catch (error) {
    console.error(chalk.red('❌ Error computing thread analysis metrics:'), error);
    throw error;
  } finally {
    closeDb(db);
  }
}
