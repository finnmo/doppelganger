// src/generator.ts
import { getDb, closeDb, closeAllConnections } from './db/client.js';
import { analyzeSentiment } from './processors/sentiment.js';
import { computeTextMetrics } from './processors/textProcessor.js';
import { computeEmojiMetrics } from './processors/emojiMetrics.js';
import { analyzeConversations } from './processors/conversation.js';
import { computeResponseTimes } from './processors/responseTimes.js';
import { computeAdditionalMetrics } from './processors/additionalMetrics.js';
import { computeAdvancedMetrics } from './processors/advancedMetrics.js';
import { computeAttachmentTypeMetrics } from './processors/attachmentTypeMetrics.js';
import { computeMessageLengthMetrics } from './processors/messageLengthMetrics.js';
import { computeInsightMetrics } from './processors/insightMetrics.js';
import { logDevModeStatus } from './utils/devMode.js';
import { progressReporter } from './utils/progressReporter.js';
import { generateEnhancedEmotions } from './processors/enhancedEmotionProcessor.js';
import { generateEnhancedTimeMetrics } from './processors/enhancedTimeMetrics.js';
import { generateEnhancedMediaData } from './processors/enhancedMediaProcessor.js';
import { computeContentTypeMetrics } from './processors/contentTypeMetrics.js';
import { computeSentimentTimelineMetrics } from './processors/sentimentTimelineMetrics.js';
import { computeReactionMetrics } from './processors/reactionMetrics.js';
import { computeMoodCorrelationMetrics } from './processors/moodCorrelationMetrics.js';
import { computeMediaEngagementMetrics } from './processors/mediaEngagementMetrics.js';
import { computeThreadAnalysisMetrics } from './processors/threadAnalysisMetrics.js';
import { computeTurnTakingAnalysisMetrics } from './processors/turnTakingAnalysisMetrics.js';
import { computeEngagementScoringMetrics } from './processors/engagementScoringMetrics.js';
import { computeConversationStarterAnalysisMetrics } from './processors/conversationStarterAnalysisMetrics.js';
import { computeEmotionalPeaksMetrics } from './processors/emotionalPeaksMetrics.js';
import { computePersonaMetrics } from './processors/personaMetrics.js';
import { computePersonaEmbeddings } from './processors/personaEmbeddings.js';
import { computePersonaEval } from './processors/personaEval.js';
import chalk from 'chalk';

interface PipelineStep {
  label: string;
  run: () => Promise<void>;
}

/**
 * The full analytics pipeline, in execution order. Ordering constraints:
 *  - sentiment analysis must precede any sentiment-derived metric;
 *  - response times must precede conversation, additional, and engagement
 *    metrics (they read the response_times table).
 */
const PIPELINE: PipelineStep[] = [
  { label: 'Text metrics', run: computeTextMetrics },
  { label: 'Emoji metrics', run: computeEmojiMetrics },
  { label: 'Message length distribution', run: computeMessageLengthMetrics },
  { label: 'Sentiment analysis', run: analyzeSentiment },
  { label: 'Response times', run: computeResponseTimes },
  { label: 'Conversations', run: analyzeConversations },
  { label: 'Additional metrics', run: computeAdditionalMetrics },
  { label: 'Attachment type metrics', run: computeAttachmentTypeMetrics },
  { label: 'Advanced metrics', run: computeAdvancedMetrics },
  { label: 'Enhanced emotions', run: generateEnhancedEmotions },
  { label: 'Insight metrics', run: computeInsightMetrics },
  { label: 'Enhanced time metrics', run: generateEnhancedTimeMetrics },
  { label: 'Enhanced media data', run: generateEnhancedMediaData },
  { label: 'Content type metrics', run: computeContentTypeMetrics },
  { label: 'Sentiment timeline metrics', run: computeSentimentTimelineMetrics },
  { label: 'Reaction metrics', run: computeReactionMetrics },
  { label: 'Mood correlation metrics', run: computeMoodCorrelationMetrics },
  {
    label: 'Media engagement metrics',
    run: async () => {
      const db = await getDb();
      await computeMediaEngagementMetrics(db);
      await closeDb(db);
    }
  },
  { label: 'Thread analysis metrics', run: computeThreadAnalysisMetrics },
  { label: 'Turn-taking analysis metrics', run: computeTurnTakingAnalysisMetrics },
  { label: 'Engagement scoring metrics', run: computeEngagementScoringMetrics },
  { label: 'Conversation starter analysis metrics', run: computeConversationStarterAnalysisMetrics },
  { label: 'Emotional peaks & valleys metrics', run: computeEmotionalPeaksMetrics },
  { label: 'Persona style profiles', run: computePersonaMetrics },
  { label: 'Persona embeddings (vector RAG)', run: computePersonaEmbeddings },
  {
    label: 'Persona held-out eval set',
    run: async () => computePersonaEval({ live: false, maxSenders: 10, pairsPerSender: 20 }),
  },
];

export async function generateDashboards(): Promise<void> {
  console.log(chalk.blue('🚀 Running FAST MODE - optimized analytics generation'));
  const startTime = Date.now();

  try {
    logDevModeStatus();

    const overallBar = progressReporter.createProgressBar(PIPELINE.length, 'Overall Progress');

    for (const step of PIPELINE) {
      progressReporter.start(`${step.label}...`);
      try {
        await step.run();
        progressReporter.success(`${step.label} computed`);
      } catch (err) {
        progressReporter.error(`${step.label} failed`);
        throw err;
      }
      overallBar.tick(1);
    }

    const duration = (Date.now() - startTime) / 1000;
    console.log(chalk.green(`✅ All metrics computed successfully in ${duration.toFixed(1)} seconds!`));
  } catch (error) {
    console.error(chalk.red('❌ Error generating dashboards:'), error);
    throw error;
  } finally {
    await closeAllConnections();
  }
}
