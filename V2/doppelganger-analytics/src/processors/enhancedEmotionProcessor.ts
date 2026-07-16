import { getDb, closeDb } from '../db/client.js';
import { decodeInstagramUnicode } from '../utils/unicodeDecoder.js';
import chalk from 'chalk';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

interface MessageRow {
  id: number;
  content: string;
  sender: string;
  timestamp_ms: number;
  conversation_id: string;
}

interface EmotionData {
  message_id: number;
  conversation_id: string;
  emotions: {
    joy: number;
    sadness: number;
    anger: number;
    fear: number;
    surprise: number;
  };
}

// Keyword patterns use \b word boundaries; emoji patterns must not, because
// \b never matches next to non-word characters like emoji.
const EMOTION_PATTERNS = {
  joy: [
    /\b(happy|joy|excited|great|awesome|amazing|wonderful|fantastic|love|perfect|best|excellent|brilliant|superb)\b/gi,
    /\b(haha|hehe|lol|lmao|rofl)\b/gi,
    /😂|😄|😊|😍|🥰|😘|💕|❤️|💖|🎉|🎊/g,
    /\b(yay|woohoo|hooray|celebrate|celebration|party|fun|enjoy|enjoying)\b/gi,
    /\b(good news|so good|feels good|makes me happy|thrilled|delighted|pleased)\b/gi
  ],
  sadness: [
    /\b(sad|cry|crying|tears|upset|disappointed|depressed|miserable)\b/gi,
    /😢|😭|😞|😔|💔|😿|😪|😫|😩/g,
    /\b(sorry|regret|miss you|missing you|hurt|pain|heartbreak|heartbroken)\b/gi,
    /\b(feel bad|so sad|makes me sad|feeling down|bummed|devastated)\b/gi
  ],
  anger: [
    /\b(angry|mad|furious|rage|pissed|annoyed|irritated|frustrated|hate|disgusted)\b/gi,
    /😠|😡|🤬|💢|😤|👿|😾/g,
    /\b(damn|shit|fuck|stupid|idiot|ridiculous|outrageous)\b/gi,
    /\b(so mad|makes me angry|pissed off|fed up|sick of)\b/gi
  ],
  fear: [
    /\b(scared|afraid|fear|frightened|terrified|nervous|worried|anxious|panic)\b/gi,
    /😨|😰|😱|😖|😟|😧|😬/g,
    /\b(dangerous|risky|unsafe|threatening|creepy|spooky)\b/gi,
    /\b(so scared|makes me nervous|freaking out|paranoid)\b/gi
  ],
  surprise: [
    /\b(wow|whoa|omg|oh my god|shocking|surprised|unexpected|incredible|unbelievable)\b/gi,
    /😮|😯|😲|🤯|😵|😳/g,
    /\b(can't believe|no way|plot twist|out of nowhere)\b/gi
  ]
};

// Context modifiers that can affect emotion intensity.
// No /g flag: global regexes keep lastIndex state across .test() calls.
const INTENSITY_MODIFIERS = {
  high: /\b(very|extremely|incredibly|absolutely|totally|completely|really|so|super|ultra)\b/i,
  medium: /\b(quite|pretty|fairly|somewhat|rather|kind of|sort of)\b/i,
  low: /\b(slightly|a bit|a little|barely|hardly|maybe|perhaps)\b/i
};

function detectEmotions(content: string): { joy: number; sadness: number; anger: number; fear: number; surprise: number } {
  const emotions = { joy: 0, sadness: 0, anger: 0, fear: 0, surprise: 0 };
  if (!content || content.trim().length === 0) {
    return emotions;
  }

  // Base score per matched signal, capped at 1.0 per emotion
  Object.entries(EMOTION_PATTERNS).forEach(([emotion, patterns]) => {
    let score = 0;
    patterns.forEach(pattern => {
      const matches = content.match(pattern) || [];
      score += matches.length * 0.3;
    });
    emotions[emotion as keyof typeof emotions] = Math.min(score, 1.0);
  });

  let intensityMultiplier = 1.0;
  if (INTENSITY_MODIFIERS.high.test(content)) intensityMultiplier = 1.5;
  else if (INTENSITY_MODIFIERS.medium.test(content)) intensityMultiplier = 1.2;
  else if (INTENSITY_MODIFIERS.low.test(content)) intensityMultiplier = 0.7;

  Object.keys(emotions).forEach(emotion => {
    const key = emotion as keyof typeof emotions;
    emotions[key] = Math.round(Math.min(emotions[key] * intensityMultiplier, 1.0) * 100) / 100;
  });

  return emotions;
}

export async function generateEnhancedEmotions(): Promise<void> {
  progressReporter.start('Generating enhanced emotion data...');
  const db = await getDb();

  try {
    const messages = db.prepare(`
      SELECT
        m.id,
        m.content,
        m.sender,
        m.timestamp_ms,
        m.conversation_id
      FROM messages m
      WHERE m.content IS NOT NULL
      AND m.content != ''
      AND m.conversation_id IS NOT NULL
      ORDER BY m.id
    `).all() as MessageRow[];

    progressReporter.update(`Processing ${messages.length} messages for emotion detection...`);

    // Only messages with a detected emotion are kept: consumers sum scores,
    // so all-zero rows carry no information and only bloat the payload.
    const emotionResults: EmotionData[] = [];

    for (const message of messages) {
      const decodedContent = decodeInstagramUnicode(message.content);
      const emotions = detectEmotions(decodedContent);

      if (emotions.joy > 0 || emotions.sadness > 0 || emotions.anger > 0 || emotions.fear > 0 || emotions.surprise > 0) {
        emotionResults.push({
          message_id: message.id,
          conversation_id: message.conversation_id,
          emotions
        });
      }
    }

    // Write emotion data (compact JSON: this file is machine-read only)
    writeDashData('emotionMetrics.json', emotionResults);
    
    // Calculate and log statistics
    const totalEmotions = emotionResults.reduce((sum, result) => {
      return sum + Object.values(result.emotions).reduce((emotionSum, value) => emotionSum + value, 0);
    }, 0);

    const emotionCounts = emotionResults.reduce((counts, result) => {
      Object.entries(result.emotions).forEach(([emotion, value]) => {
        if (value > 0) {
          counts[emotion] = (counts[emotion] || 0) + 1;
        }
      });
      return counts;
    }, {} as Record<string, number>);

    progressReporter.success('Enhanced emotions saved');
    progressReporter.update(`Total emotion instances: ${totalEmotions.toFixed(1)}`);
    progressReporter.update(`Emotion distribution:`);
    Object.entries(emotionCounts).forEach(([emotion, count]) => {
      progressReporter.update(`${emotion}: ${count} messages (${((count / emotionResults.length) * 100).toFixed(1)}%)`);
    });

  } catch (error) {
    console.error(chalk.red('❌ Error generating enhanced emotions:'), error);
    throw error;
  } finally {
    await closeDb(db);
  }
}
 