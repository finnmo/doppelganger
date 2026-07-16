import vader from 'vader-sentiment';

interface SentimentScore {
  compound: number;
  positive: number;
  negative: number;
  neutral: number;
}

const NEUTRAL: SentimentScore = { compound: 0, positive: 0, negative: 0, neutral: 1 };

/**
 * Scores text with VADER, a rule-based sentiment model built for social
 * media text (handles negation, intensifiers, slang, emoticons, and emoji).
 *
 * compound is in [-1, 1]; positive/negative/neutral are proportions summing to ~1.
 */
export function analyzeSentiment(text: string): SentimentScore {
  if (!text || text.trim().length === 0) {
    return { ...NEUTRAL };
  }

  const scores = vader.SentimentIntensityAnalyzer.polarity_scores(text);

  return {
    compound: scores.compound,
    positive: scores.pos,
    negative: scores.neg,
    neutral: scores.neu
  };
}

export async function analyzeSentimentBatch(texts: string[]): Promise<SentimentScore[]> {
  return texts.map(text => analyzeSentiment(text));
}
