import { getDb } from '../db/client.js';
import { decodeInstagramUnicode } from '../utils/unicodeDecoder.js';
import { isSystemMessage } from '../utils/messageFilters.js';
import { writeDashData } from '../utils/output.js';
import chalk from 'chalk';
import { progressReporter } from '../utils/progressReporter.js';

interface MessageRow {
  id: number;
  content: string;
  sender: string;
  timestamp_ms: number;
  conversation_id: string;
  compound: number;
  positive: number;
  negative: number;
  neutral: number;
  has_media: number;
}

interface ImportantMessage {
  message_id: number;
  content: string;
  sender: string;
  importance_score: number;
  factors: string[];
  timestamp_ms: number;
  conversation_id: string;
}

export function calculateImportanceScore(message: MessageRow): { score: number; factors: string[] } {
  const factors: string[] = [];
  let score = 0;

  // Content length factor (0-0.3)
  const contentLength = message.content?.length || 0;
  let lengthScore = 0;
  if (contentLength > 200) {
    lengthScore = 0.3;
    factors.push('Long Message');
  } else if (contentLength > 100) {
    lengthScore = 0.2;
    factors.push('Medium Message');
  } else if (contentLength > 50) {
    lengthScore = 0.1;
    factors.push('Short Message');
  } else {
    lengthScore = 0.05;
  }
  score += lengthScore;

  // Sentiment strength factor (0-0.25)
  const sentimentStrength = Math.abs(message.compound || 0);
  if (sentimentStrength > 0.5) {
    score += 0.25;
    factors.push(message.compound > 0 ? 'Very Positive' : 'Very Negative');
  } else if (sentimentStrength > 0.3) {
    score += 0.15;
    factors.push(message.compound > 0 ? 'Positive' : 'Negative');
  } else if (sentimentStrength > 0.1) {
    score += 0.1;
    factors.push('Mild Sentiment');
  }

  // Media presence factor (0-0.15)
  if (message.has_media) {
    score += 0.15;
    factors.push('Contains Media');
  }

  // Content analysis factors (0-0.3)
  const content = message.content?.toLowerCase() || '';
  
  // Questions increase importance
  if (content.includes('?') || content.match(/\b(what|how|why|when|where|who)\b/)) {
    score += 0.1;
    factors.push('Question');
  }

  // Exclamations indicate excitement/importance
  if (content.includes('!')) {
    const exclamationCount = (content.match(/!/g) || []).length;
    score += Math.min(exclamationCount * 0.05, 0.15);
    factors.push('Exclamation');
  }

  // URLs suggest information sharing
  if (content.match(/https?:\/\//) || content.includes('www.')) {
    score += 0.1;
    factors.push('Link Sharing');
  }

  // Mentions suggest direct communication
  if (content.includes('@')) {
    score += 0.08;
    factors.push('Mentions');
  }

  // Emotional keywords
  const emotionalWords = ['love', 'hate', 'amazing', 'terrible', 'fantastic', 'awful', 'excited', 'worried', 'happy', 'sad', 'angry', 'surprised'];
  if (emotionalWords.some(word => content.includes(word))) {
    score += 0.1;
    factors.push('Emotional Content');
  }

  // All caps (but not too much) — check the original content, not the lowercased copy
  const capsWords = (message.content || '').match(/\b[A-Z]{3,}\b/g);
  if (capsWords && capsWords.length > 0 && capsWords.length < 5) {
    score += 0.08;
    factors.push('Emphasis');
  }

  // Time-based factors
  const hour = new Date(message.timestamp_ms).getHours();
  if (hour < 6 || hour > 22) {
    score += 0.05;
    factors.push('Late Night Message');
  }

  // Length-based bonus for very long messages
  if (contentLength > 500) {
    score += 0.1;
    factors.push('Very Long Message');
  }

  // Ensure minimum score for all messages
  if (score < 0.1) {
    score = 0.1;
    if (factors.length === 0) {
      factors.push('Regular Message');
    }
  }

  // Cap maximum score
  score = Math.min(score, 1.0);

  return { score: Math.round(score * 1000) / 1000, factors };
}

export async function computeInsightMetrics(): Promise<void> {
  progressReporter.start('Computing insight metrics...');
  
  try {
    const db = await getDb();
    
    // Get messages with sentiment and response data
    const messages = db.prepare(`
      SELECT 
        m.id,
        m.content,
        m.sender,
        m.timestamp_ms,
        m.conversation_id,
        COALESCE(s.compound, 0) as compound,
        COALESCE(s.positive, 0) as positive,
        COALESCE(s.negative, 0) as negative,
        COALESCE(s.neutral, 0) as neutral,
        CASE
          WHEN m.has_photos = 1
            OR m.has_videos = 1
            OR m.content LIKE '%sent an attachment%'
          THEN 1
          ELSE 0
        END as has_media
      FROM messages m
      LEFT JOIN sentiment s ON m.id = s.message_id
      WHERE m.content IS NOT NULL 
      AND m.content != ''
      AND m.conversation_id IS NOT NULL
      ORDER BY m.timestamp_ms DESC
      LIMIT 50000
    `).all() as MessageRow[];

    progressReporter.update(`Processing ${messages.length} messages for importance analysis...`);

    const importantMessages: ImportantMessage[] = [];
    
    messages.forEach((message, index) => {
      if (index % 5000 === 0) {
        progressReporter.update(`Processed ${index}/${messages.length} messages...`);
      }

      // Skip system messages
      if (isSystemMessage(message.content)) {
        return;
      }

      const { score, factors } = calculateImportanceScore(message);
      
      // Decode Instagram's malformed Unicode encoding before storing
      const decodedContent = decodeInstagramUnicode(message.content);
      
      importantMessages.push({
        message_id: message.id,
        content: decodedContent.substring(0, 500), // Limit content length
        sender: message.sender,
        importance_score: score,
        factors,
        timestamp_ms: message.timestamp_ms,
        conversation_id: message.conversation_id
      });
    });

    // Sort by importance and take top 1000
    const topMessages = importantMessages
      .sort((a, b) => b.importance_score - a.importance_score)
      .slice(0, 1000);

    // Export data
    progressReporter.update('Exporting insight metrics...');
    writeDashData('messageImportance.json', topMessages);

    // Log statistics
    const avgScore = topMessages.reduce((sum, msg) => sum + msg.importance_score, 0) / topMessages.length;
    const factorCounts = topMessages.reduce((counts, msg) => {
      msg.factors.forEach(factor => {
        counts[factor] = (counts[factor] || 0) + 1;
      });
      return counts;
    }, {} as Record<string, number>);

    progressReporter.success(`Insight metrics computed: ${topMessages.length} important messages`);
    progressReporter.update(`Average importance score: ${avgScore.toFixed(3)}`);
    progressReporter.update(`Top factors: ${Object.entries(factorCounts).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([factor, count]) => `${factor} (${count})`).join(', ')}`);

  } catch (error) {
    console.error(chalk.red('❌ Error computing insight metrics:'), error);
    throw error;
  } finally {
    // closeDb() is handled automatically by the database client
  }
}
 