import { getDb } from '../db/client.js';
import { writeDashData } from '../utils/output.js';
import { progressReporter } from '../utils/progressReporter.js';
import { localDateString } from '../utils/dates.js';

interface MessageRow {
  id: number;
  content: string | null;
  sender: string;
  timestamp_ms: number;
  conversation_id: string;
  sentiment?: number;  // Changed from compound to sentiment to match query alias
  positive?: number;
  negative?: number;
  neutral?: number;
}

interface EmotionalPeak {
  id: string;
  conversation_id: string;
  type: 'peak' | 'valley';
  date: string;
  timestamp: number;
  sentiment_score: number;
  intensity: 'extreme' | 'high' | 'moderate';
  duration_hours: number;
  message_count: number;
  trigger_analysis: {
    primary_trigger: string;
    trigger_confidence: number;
    contributing_factors: string[];
    keywords: string[];
  };
  context: {
    preceding_sentiment: number;
    following_sentiment: number;
    sentiment_change: number;
    time_to_recovery: number; // hours
  };
  participants: Array<{
    sender: string;
    contribution_score: number;
    message_count: number;
    avg_sentiment: number;
  }>;
  sample_messages: Array<{
    id: number;
    content: string;
    sender: string;
    sentiment: number;
    timestamp: number;
  }>;
}

interface EmotionalPattern {
  pattern_type: 'volatility' | 'stability' | 'gradual_decline' | 'gradual_improvement' | 'cyclical';
  frequency: number;
  avg_intensity: number;
  typical_duration: number;
  common_triggers: string[];
  recovery_time: number;
  description: string;
}

interface TriggerAnalysis {
  trigger: string;
  frequency: number;
  avg_impact: number;
  typical_sentiment_change: number;
  recovery_time: number;
  associated_keywords: string[];
  time_patterns: Array<{
    hour: number;
    day_of_week: number;
    frequency: number;
  }>;
  participant_sensitivity: Array<{
    sender: string;
    sensitivity_score: number;
    typical_response: number;
  }>;
}

interface EmotionalPeaksData {
  summary: {
    total_peaks: number;
    total_valleys: number;
    extreme_events: number;
    avg_peak_intensity: number;
    avg_valley_intensity: number;
    avg_recovery_time: number;
    most_volatile_period: string;
    most_stable_period: string;
    dominant_pattern: string;
  };
  peaks_and_valleys: EmotionalPeak[];
  emotional_patterns: EmotionalPattern[];
  trigger_analysis: TriggerAnalysis[];
  temporal_analysis: {
    hourly_volatility: Array<{
      hour: number;
      avg_volatility: number;
      peak_frequency: number;
      valley_frequency: number;
    }>;
    daily_volatility: Array<{
      day_of_week: number;
      day_name: string;
      avg_volatility: number;
      emotional_events: number;
    }>;
    monthly_trends: Array<{
      month: string;
      emotional_stability: number;
      peak_count: number;
      valley_count: number;
      avg_sentiment: number;
    }>;
  };
  recovery_analysis: {
    avg_peak_recovery: number;
    avg_valley_recovery: number;
    fastest_recovery: EmotionalPeak | null;
    slowest_recovery: EmotionalPeak | null;
    recovery_factors: Array<{
      factor: string;
      impact_on_recovery: number;
      frequency: number;
    }>;
  };
}

// Trigger detection patterns
const TRIGGER_PATTERNS = {
  work_stress: {
    keywords: ['work', 'job', 'boss', 'meeting', 'deadline', 'project', 'stress', 'busy', 'overtime', 'office'],
    negative_keywords: ['fired', 'layoff', 'promotion denied', 'overworked', 'burnout'],
    positive_keywords: ['promotion', 'raise', 'bonus', 'vacation', 'weekend']
  },
  relationship: {
    keywords: ['relationship', 'boyfriend', 'girlfriend', 'partner', 'date', 'love', 'breakup', 'fight', 'argument'],
    negative_keywords: ['breakup', 'fight', 'argument', 'cheating', 'divorce', 'lonely'],
    positive_keywords: ['anniversary', 'proposal', 'wedding', 'valentine', 'love']
  },
  health: {
    keywords: ['sick', 'doctor', 'hospital', 'medicine', 'pain', 'headache', 'flu', 'covid', 'vaccine'],
    negative_keywords: ['sick', 'pain', 'hospital', 'surgery', 'diagnosis', 'emergency'],
    positive_keywords: ['better', 'recovered', 'healthy', 'exercise', 'gym']
  },
  family: {
    keywords: ['family', 'mom', 'dad', 'parents', 'brother', 'sister', 'kids', 'children', 'baby'],
    negative_keywords: ['fight', 'argument', 'death', 'funeral', 'divorce', 'custody'],
    positive_keywords: ['birth', 'wedding', 'graduation', 'celebration', 'reunion']
  },
  social: {
    keywords: ['friends', 'party', 'social', 'event', 'gathering', 'celebration', 'birthday'],
    negative_keywords: ['lonely', 'isolated', 'cancelled', 'excluded', 'drama'],
    positive_keywords: ['party', 'celebration', 'fun', 'friends', 'social']
  },
  achievement: {
    keywords: ['achievement', 'success', 'goal', 'accomplished', 'won', 'passed', 'graduated'],
    negative_keywords: ['failed', 'lost', 'rejected', 'denied', 'missed'],
    positive_keywords: ['won', 'success', 'achieved', 'accomplished', 'graduated', 'promoted']
  },
  weather_events: {
    keywords: ['weather', 'rain', 'snow', 'storm', 'sunny', 'cold', 'hot', 'hurricane', 'flood'],
    negative_keywords: ['storm', 'hurricane', 'flood', 'cold', 'rain', 'snow'],
    positive_keywords: ['sunny', 'warm', 'beautiful', 'clear']
  },
  technology: {
    keywords: ['phone', 'computer', 'internet', 'app', 'software', 'bug', 'crash', 'update'],
    negative_keywords: ['crash', 'bug', 'broken', 'virus', 'hacked', 'lost'],
    positive_keywords: ['update', 'new', 'upgrade', 'fixed', 'working']
  },
  financial: {
    keywords: ['money', 'expensive', 'cheap', 'budget', 'salary', 'bonus', 'bill', 'debt'],
    negative_keywords: ['expensive', 'debt', 'broke', 'bill', 'owe', 'poor'],
    positive_keywords: ['bonus', 'raise', 'cheap', 'deal', 'save', 'rich']
  }
};

function detectSentimentPeaksAndValleys(sentimentData: Array<{date: string, sentiment: number, messageCount: number}>): Array<{type: 'peak' | 'valley', index: number, value: number}> {
  const peaks: Array<{type: 'peak' | 'valley', index: number, value: number}> = [];
  
  if (sentimentData.length < 3) return peaks;
  
  // Use a simple peak detection algorithm with dynamic thresholds
  const sentiments = sentimentData.map(d => d.sentiment);
  const mean = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
  const variance = sentiments.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / sentiments.length;
  const stdDev = Math.sqrt(variance);
  
  // Dynamic thresholds based on data distribution (very sensitive)
  const peakThreshold = mean + (stdDev * 0.25);  // Reduced from 0.75 to 0.25
  const valleyThreshold = mean - (stdDev * 0.25); // Reduced from 0.75 to 0.25
  
  progressReporter.update(`Sentiment analysis stats: Mean: ${mean.toFixed(4)}, StdDev: ${stdDev.toFixed(4)}`);
  progressReporter.update(`Range: ${Math.min(...sentiments).toFixed(4)} to ${Math.max(...sentiments).toFixed(4)}`);
  progressReporter.update(`Peak threshold: ${peakThreshold.toFixed(4)}, Valley threshold: ${valleyThreshold.toFixed(4)}`);
  
  let candidatePeaks = 0;
  let candidateValleys = 0;
  
  for (let i = 1; i < sentimentData.length - 1; i++) {
    const current = sentiments[i];
    const prev = sentiments[i - 1];
    const next = sentiments[i + 1];
    
    // Check for local maxima/minima
    const isLocalPeak = current > prev && current > next;
    const isLocalValley = current < prev && current < next;
    
    if (isLocalPeak) candidatePeaks++;
    if (isLocalValley) candidateValleys++;
    
    // Peak detection: current is higher than both neighbors and above threshold
    if (isLocalPeak && current > peakThreshold) {
      peaks.push({ type: 'peak', index: i, value: current });
    }
    
    // Valley detection: current is lower than both neighbors and below threshold
    if (isLocalValley && current < valleyThreshold) {
      peaks.push({ type: 'valley', index: i, value: current });
    }
  }
  
  progressReporter.update(`Detection results: Local peaks found: ${candidatePeaks}, above threshold: ${peaks.filter(p => p.type === 'peak').length}`);
  progressReporter.update(`Local valleys found: ${candidateValleys}, below threshold: ${peaks.filter(p => p.type === 'valley').length}`);
  
  return peaks;
}

function analyzeTriggers(messages: MessageRow[], peakTimestamp: number, windowHours: number = 6): {
  primary_trigger: string;
  trigger_confidence: number;
  contributing_factors: string[];
  keywords: string[];
} {
  // Get messages within the window before the peak
  const windowStart = peakTimestamp - (windowHours * 60 * 60 * 1000);
  const contextMessages = messages.filter(m => 
    m.timestamp_ms >= windowStart && m.timestamp_ms <= peakTimestamp && m.content
  );
  
  const triggerScores: Record<string, {score: number, keywords: string[], factors: string[]}> = {};
  
  // Initialize trigger scores
  Object.keys(TRIGGER_PATTERNS).forEach(trigger => {
    triggerScores[trigger] = { score: 0, keywords: [], factors: [] };
  });
  
  // Analyze each message for trigger indicators
  contextMessages.forEach(message => {
    if (!message.content) return;
    
    const content = message.content.toLowerCase();
    
    Object.entries(TRIGGER_PATTERNS).forEach(([triggerName, patterns]) => {
      // Check for general keywords
      patterns.keywords.forEach(keyword => {
        if (content.includes(keyword)) {
          triggerScores[triggerName].score += 1;
          triggerScores[triggerName].keywords.push(keyword);
        }
      });
      
      // Check for negative keywords (higher weight)
      patterns.negative_keywords.forEach(keyword => {
        if (content.includes(keyword)) {
          triggerScores[triggerName].score += 2;
          triggerScores[triggerName].keywords.push(keyword);
          triggerScores[triggerName].factors.push(`negative_${keyword}`);
        }
      });
      
      // Check for positive keywords
      patterns.positive_keywords.forEach(keyword => {
        if (content.includes(keyword)) {
          triggerScores[triggerName].score += 1.5;
          triggerScores[triggerName].keywords.push(keyword);
          triggerScores[triggerName].factors.push(`positive_${keyword}`);
        }
      });
    });
  });
  
  // Find the trigger with the highest score
  const sortedTriggers = Object.entries(triggerScores)
    .sort(([,a], [,b]) => b.score - a.score)
    .filter(([,data]) => data.score > 0);
  
  if (sortedTriggers.length === 0) {
    return {
      primary_trigger: 'unknown',
      trigger_confidence: 0,
      contributing_factors: ['insufficient_context'],
      keywords: []
    };
  }
  
  const [primaryTrigger, primaryData] = sortedTriggers[0];
  const totalScore = Object.values(triggerScores).reduce((sum, data) => sum + data.score, 0);
  const confidence = totalScore > 0 ? (primaryData.score / totalScore) * 100 : 0;
  
  return {
    primary_trigger: primaryTrigger,
    trigger_confidence: Math.round(confidence),
    contributing_factors: [...new Set(primaryData.factors)],
    keywords: [...new Set(primaryData.keywords)]
  };
}

function calculateRecoveryTime(sentimentData: Array<{date: string, sentiment: number}>, peakIndex: number, peakType: 'peak' | 'valley'): number {
  const peakValue = sentimentData[peakIndex].sentiment;
  const targetValue = peakType === 'peak' ? peakValue * 0.7 : peakValue * 1.3; // 30% recovery
  
  for (let i = peakIndex + 1; i < sentimentData.length; i++) {
    const currentValue = sentimentData[i].sentiment;
    
    if (peakType === 'peak' && currentValue <= targetValue) {
      return i - peakIndex; // Return in days
    }
    
    if (peakType === 'valley' && currentValue >= targetValue) {
      return i - peakIndex; // Return in days
    }
  }
  
  return -1; // No recovery detected within available data
}

function getIntensityLevel(sentimentScore: number, type: 'peak' | 'valley'): 'extreme' | 'high' | 'moderate' {
  
  if (type === 'peak') {
    if (sentimentScore > 0.25) return 'extreme';  // Lowered from 0.7 to 0.25
    if (sentimentScore > 0.15) return 'high';     // Lowered from 0.4 to 0.15
    return 'moderate';
  } else {
    if (sentimentScore < -0.25) return 'extreme'; // Lowered from -0.7 to -0.25
    if (sentimentScore < -0.15) return 'high';    // Lowered from -0.4 to -0.15
    return 'moderate';
  }
}

export async function computeEmotionalPeaksMetrics(): Promise<void> {
  const db = await getDb();
  
  try {
    // Get daily sentiment data with message details
    const dailySentimentQuery = `
      SELECT 
        DATE(datetime(m.timestamp_ms/1000, 'unixepoch', 'localtime')) as date,
        m.timestamp_ms,
        m.id,
        m.content,
        m.sender,
        m.conversation_id,
        s.compound as sentiment,
        s.positive,
        s.negative,
        s.neutral
      FROM messages m
      JOIN sentiment s ON m.id = s.message_id
      WHERE s.compound IS NOT NULL
      ORDER BY m.timestamp_ms
    `;
    
    const messages: MessageRow[] = db.prepare(dailySentimentQuery).all() as MessageRow[];
    
    if (messages.length === 0) {
      progressReporter.update('No sentiment data found for emotional peaks analysis');
      
      // Create empty data structure
      const emotionalPeaksData: EmotionalPeaksData = {
        summary: {
          total_peaks: 0,
          total_valleys: 0,
          extreme_events: 0,
          avg_peak_intensity: 0,
          avg_valley_intensity: 0,
          avg_recovery_time: 0,
          most_volatile_period: 'No data',
          most_stable_period: 'No data',
          dominant_pattern: 'No data'
        },
        peaks_and_valleys: [],
        emotional_patterns: [],
        trigger_analysis: [],
        temporal_analysis: {
          hourly_volatility: Array.from({length: 24}, (_, i) => ({
            hour: i,
            avg_volatility: 0,
            peak_frequency: 0,
            valley_frequency: 0
          })),
          daily_volatility: [
            'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'
          ].map((day_name, i) => ({
            day_of_week: i,
            day_name,
            avg_volatility: 0,
            emotional_events: 0
          })),
          monthly_trends: []
        },
        recovery_analysis: {
          avg_peak_recovery: 0,
          avg_valley_recovery: 0,
          fastest_recovery: null,
          slowest_recovery: null,
          recovery_factors: []
        }
      };
      
      // Write to file
      writeDashData('emotionalPeaks.json', emotionalPeaksData);

      progressReporter.success('Emotional Peaks & Valleys analysis complete');
      return;
    }
    
    // Instead of averaging (which cancels out peaks), work with individual high-sentiment messages
    // Filter messages to only include those with significant sentiment (positive or negative)
    const significantMessages = messages
      .filter(msg => Math.abs(msg.sentiment || 0) >= 0.1) // Only messages with notable sentiment
      .sort((a, b) => a.timestamp_ms - b.timestamp_ms);
    
    progressReporter.update(`Found ${significantMessages.length} messages with significant sentiment (|sentiment| >= 0.1)`);
    
    // Group these significant messages by day to find daily sentiment extremes
    const dailyExtremes = new Map<string, {
      date: string,
      maxPositive: number,
      maxNegative: number,
      positiveMessage?: MessageRow,
      negativeMessage?: MessageRow,
      allMessages: MessageRow[]
    }>();
    
    significantMessages.forEach(message => {
      const date = localDateString(message.timestamp_ms);
      if (!dailyExtremes.has(date)) {
        dailyExtremes.set(date, {
          date,
          maxPositive: -1,
          maxNegative: 1,
          allMessages: []
        });
      }
      
      const dayData = dailyExtremes.get(date)!;
      dayData.allMessages.push(message);
      
      const sentiment = message.sentiment || 0;
      if (sentiment > dayData.maxPositive) {
        dayData.maxPositive = sentiment;
        dayData.positiveMessage = message;
      }
      if (sentiment < dayData.maxNegative) {
        dayData.maxNegative = sentiment;
        dayData.negativeMessage = message;
      }
    });
    
    // Convert to format expected by peak detection (using the daily extremes)
    const dailySentimentData = Array.from(dailyExtremes.values())
      .map(dayData => {
        // Use the more extreme value (positive or negative) as the day's sentiment
        const positiveExtreme = Math.abs(dayData.maxPositive);
        const negativeExtreme = Math.abs(dayData.maxNegative);
        
        const sentiment = positiveExtreme > negativeExtreme ? dayData.maxPositive : dayData.maxNegative;
        const representativeMessage = positiveExtreme > negativeExtreme ? dayData.positiveMessage : dayData.negativeMessage;
        
        return {
          date: dayData.date,
          sentiment,
          messageCount: dayData.allMessages.length,
          messages: dayData.allMessages,
          representativeMessage
        };
      })
      .filter(item => item.representativeMessage) // Only include days with clear extremes
      .sort((a, b) => a.date.localeCompare(b.date));
    
    progressReporter.update(`Analyzing ${dailySentimentData.length} days with emotional extremes...`);
    
    // Detect peaks and valleys
    const detectedPeaks = detectSentimentPeaksAndValleys(dailySentimentData);
    progressReporter.update(`Detected ${detectedPeaks.length} emotional peaks and valleys`);
    
    // Process each peak/valley into detailed analysis
    const peaksAndValleys: EmotionalPeak[] = [];
    const progressBar = progressReporter.createProgressBar(detectedPeaks.length, 'Processing peaks and valleys');
    
    for (let i = 0; i < detectedPeaks.length; i++) {
      const detection = detectedPeaks[i];
      
      // Add safety check to prevent infinite loops
      if (i > 1000) {
        progressReporter.update(`Safety limit reached, processing ${i} of ${detectedPeaks.length} peaks`);
        break;
      }
      
      const dayData = dailySentimentData[detection.index];
      const dayMessages = dayData.messages;
      
      // Add progress update every 10 iterations
      if (i % 10 === 0) {
        progressReporter.update(`Processing peak/valley ${i + 1}/${detectedPeaks.length} (${detection.type})`);
      }
      
      // Calculate duration (look for adjacent days with similar extreme sentiment)
      let duration = 1;
      const threshold = detection.type === 'peak' ? 0.1 : -0.1;  // Lowered from 0.3/-0.3 to 0.1/-0.1
      
      // Check forward with safety limit
      for (let j = detection.index + 1; j < Math.min(dailySentimentData.length, detection.index + 30); j++) {
        if (detection.type === 'peak' && dailySentimentData[j].sentiment > threshold) {
          duration++;
        } else if (detection.type === 'valley' && dailySentimentData[j].sentiment < threshold) {
          duration++;
        } else {
          break;
        }
      }
      
      // Analyze triggers with timeout protection
      let triggerAnalysis;
      try {
        triggerAnalysis = analyzeTriggers(dayMessages, dayData.representativeMessage!.timestamp_ms);
      } catch (_error) {
        progressReporter.update(`Error analyzing triggers for peak ${i}, using default`);
        triggerAnalysis = {
          primary_trigger: 'unknown',
          trigger_confidence: 0.5,
          contributing_factors: ['unknown'],
          keywords: []
        };
      }
      
      // Calculate recovery time
      const recoveryTime = calculateRecoveryTime(dailySentimentData, detection.index, detection.type);
      
      // Get intensity level
      const intensity = getIntensityLevel(detection.value, detection.type);
      
      // Analyze participant contributions
      const participantMap = new Map<string, { messages: MessageRow[], totalSentiment: number }>();
      dayMessages.forEach(msg => {
        if (!participantMap.has(msg.sender)) {
          participantMap.set(msg.sender, { messages: [], totalSentiment: 0 });
        }
        const participant = participantMap.get(msg.sender)!;
        participant.messages.push(msg);
        participant.totalSentiment += msg.sentiment || 0;
      });
      
      const participants = Array.from(participantMap.entries()).map(([sender, data]) => ({
        sender,
        contribution_score: data.totalSentiment / data.messages.length,
        message_count: data.messages.length,
        avg_sentiment: data.totalSentiment / data.messages.length
      }));
      
      // Get sample messages (most extreme ones)
      const sampleMessages = dayMessages
        .sort((a, b) => Math.abs(b.sentiment || 0) - Math.abs(a.sentiment || 0))
        .slice(0, 5)
        .map(msg => ({
          id: msg.id,
          content: msg.content || '',
          sender: msg.sender,
          sentiment: msg.sentiment || 0,
          timestamp: msg.timestamp_ms
        }));
      
      const peak: EmotionalPeak = {
        id: `${detection.type}_${dayData.date}_${detection.index}`,
        conversation_id: dayData.representativeMessage!.conversation_id,
        type: detection.type,
        date: dayData.date,
        timestamp: dayData.representativeMessage!.timestamp_ms,
        sentiment_score: detection.value,
        intensity,
        duration_hours: duration * 24,
        message_count: dayMessages.length,
        trigger_analysis: triggerAnalysis,
        context: {
          preceding_sentiment: detection.index > 0 ? dailySentimentData[detection.index - 1].sentiment : 0,
          following_sentiment: detection.index < dailySentimentData.length - 1 ? dailySentimentData[detection.index + 1].sentiment : 0,
          sentiment_change: detection.value - (detection.index > 0 ? dailySentimentData[detection.index - 1].sentiment : 0),
          time_to_recovery: recoveryTime
        },
        participants,
        sample_messages: sampleMessages
      };
      
      peaksAndValleys.push(peak);
      progressBar.tick(1);
      if (i % 100 === 0 && i > 0) {
        progressReporter.update(`Heartbeat: still processing peak/valley ${i + 1}/${detectedPeaks.length}`);
      }
    }
    
    // Continue with the rest of the analysis...
    progressReporter.update('Analyzing emotional patterns...');
    const emotionalPatterns = analyzeEmotionalPatterns(peaksAndValleys, dailySentimentData);
    
    progressReporter.update('Analyzing trigger patterns...');
    const triggerAnalysis = analyzeTriggerPatterns(peaksAndValleys);
    
    progressReporter.update('Calculating temporal analysis...');
    const temporalAnalysis = calculateTemporalAnalysis(peaksAndValleys, dailySentimentData);
    
    progressReporter.update('Calculating recovery analysis...');
    const recoveryAnalysis = calculateRecoveryAnalysis(peaksAndValleys);
    
    // Create summary
    const summary = {
      total_peaks: peaksAndValleys.filter(p => p.type === 'peak').length,
      total_valleys: peaksAndValleys.filter(p => p.type === 'valley').length,
      extreme_events: peaksAndValleys.filter(p => p.intensity === 'extreme').length,
      avg_peak_intensity: peaksAndValleys.filter(p => p.type === 'peak').reduce((sum, p) => sum + p.sentiment_score, 0) / Math.max(peaksAndValleys.filter(p => p.type === 'peak').length, 1),
      avg_valley_intensity: peaksAndValleys.filter(p => p.type === 'valley').reduce((sum, p) => sum + p.sentiment_score, 0) / Math.max(peaksAndValleys.filter(p => p.type === 'valley').length, 1),
      avg_recovery_time: peaksAndValleys.reduce((sum, p) => sum + p.context.time_to_recovery, 0) / Math.max(peaksAndValleys.length, 1),
      most_volatile_period: findMostVolatilePeriod(dailySentimentData),
      most_stable_period: findMostStablePeriod(dailySentimentData),
      dominant_pattern: emotionalPatterns.length > 0 ? emotionalPatterns[0].pattern_type : 'No pattern'
    };
    
    const emotionalPeaksData: EmotionalPeaksData = {
      summary,
      peaks_and_valleys: peaksAndValleys,
      emotional_patterns: emotionalPatterns,
      trigger_analysis: triggerAnalysis,
      temporal_analysis: temporalAnalysis,
      recovery_analysis: recoveryAnalysis
    };
    
    // Write to file
    progressReporter.update('Exporting emotional peaks data...');
    writeDashData('emotionalPeaks.json', emotionalPeaksData);

    progressReporter.success('Emotional Peaks & Valleys analysis complete');
  } catch (error) {
    progressReporter.error('Error computing emotional peaks metrics');
    console.error(error);
    throw error;
  }
}

function analyzeEmotionalPatterns(peaksAndValleys: EmotionalPeak[], dailyData: Array<{date: string, sentiment: number}>): EmotionalPattern[] {
  const patterns: EmotionalPattern[] = [];
  
  // Calculate volatility pattern
  const sentimentChanges = [];
  for (let i = 1; i < dailyData.length; i++) {
    sentimentChanges.push(Math.abs(dailyData[i].sentiment - dailyData[i-1].sentiment));
  }
  const avgVolatility = sentimentChanges.reduce((a, b) => a + b, 0) / sentimentChanges.length;
  
  if (avgVolatility > 0.2) {
    patterns.push({
      pattern_type: 'volatility',
      frequency: peaksAndValleys.length,
      avg_intensity: peaksAndValleys.reduce((sum, p) => sum + Math.abs(p.sentiment_score), 0) / peaksAndValleys.length,
      typical_duration: peaksAndValleys.reduce((sum, p) => sum + p.duration_hours, 0) / peaksAndValleys.length,
      common_triggers: findCommonTriggers(peaksAndValleys),
      recovery_time: peaksAndValleys.reduce((sum, p) => sum + (p.context.time_to_recovery > 0 ? p.context.time_to_recovery : 0), 0) / peaksAndValleys.length,
      description: 'High emotional volatility with frequent sentiment swings'
    });
  } else {
    patterns.push({
      pattern_type: 'stability',
      frequency: peaksAndValleys.length,
      avg_intensity: peaksAndValleys.reduce((sum, p) => sum + Math.abs(p.sentiment_score), 0) / peaksAndValleys.length,
      typical_duration: peaksAndValleys.reduce((sum, p) => sum + p.duration_hours, 0) / peaksAndValleys.length,
      common_triggers: findCommonTriggers(peaksAndValleys),
      recovery_time: peaksAndValleys.reduce((sum, p) => sum + (p.context.time_to_recovery > 0 ? p.context.time_to_recovery : 0), 0) / peaksAndValleys.length,
      description: 'Relatively stable emotional state with occasional fluctuations'
    });
  }
  
  return patterns;
}

function findCommonTriggers(peaksAndValleys: EmotionalPeak[]): string[] {
  const triggerCounts = new Map<string, number>();
  
  peaksAndValleys.forEach(peak => {
    triggerCounts.set(peak.trigger_analysis.primary_trigger, 
      (triggerCounts.get(peak.trigger_analysis.primary_trigger) || 0) + 1);
  });
  
  return Array.from(triggerCounts.entries())
    .sort(([,a], [,b]) => b - a)
    .slice(0, 3)
    .map(([trigger]) => trigger);
}

function analyzeTriggerPatterns(peaksAndValleys: EmotionalPeak[]): TriggerAnalysis[] {
  const triggerMap = new Map<string, {
    impacts: number[];
    recoveryTimes: number[];
    keywords: Set<string>;
    hours: number[];
    daysOfWeek: number[];
    participantImpacts: Map<string, number[]>;
  }>();
  
  peaksAndValleys.forEach(peak => {
    const trigger = peak.trigger_analysis.primary_trigger;
    if (!triggerMap.has(trigger)) {
      triggerMap.set(trigger, {
        impacts: [],
        recoveryTimes: [],
        keywords: new Set(),
        hours: [],
        daysOfWeek: [],
        participantImpacts: new Map()
      });
    }
    
    const data = triggerMap.get(trigger)!;
    data.impacts.push(Math.abs(peak.sentiment_score));
    if (peak.context.time_to_recovery > 0) {
      data.recoveryTimes.push(peak.context.time_to_recovery);
    }
    
    peak.trigger_analysis.keywords.forEach(kw => data.keywords.add(kw));
    
    const date = new Date(peak.timestamp);
    data.hours.push(date.getHours());
    data.daysOfWeek.push(date.getDay());
    
    peak.participants.forEach(p => {
      if (!data.participantImpacts.has(p.sender)) {
        data.participantImpacts.set(p.sender, []);
      }
      data.participantImpacts.get(p.sender)!.push(p.contribution_score);
    });
  });
  
  return Array.from(triggerMap.entries()).map(([trigger, data]) => ({
    trigger,
    frequency: data.impacts.length,
    avg_impact: data.impacts.reduce((a, b) => a + b, 0) / data.impacts.length,
    typical_sentiment_change: data.impacts.reduce((a, b) => a + b, 0) / data.impacts.length,
    recovery_time: data.recoveryTimes.length > 0 ? 
      data.recoveryTimes.reduce((a, b) => a + b, 0) / data.recoveryTimes.length : 0,
    associated_keywords: Array.from(data.keywords).slice(0, 10),
    time_patterns: calculateTimePatterns(data.hours, data.daysOfWeek),
    participant_sensitivity: calculateParticipantSensitivity(data.participantImpacts)
  })).sort((a, b) => b.frequency - a.frequency);
}

function calculateTimePatterns(hours: number[], daysOfWeek: number[]): Array<{hour: number; day_of_week: number; frequency: number}> {
  const patterns = new Map<string, number>();
  
  hours.forEach((hour, i) => {
    const key = `${hour}_${daysOfWeek[i]}`;
    patterns.set(key, (patterns.get(key) || 0) + 1);
  });
  
  return Array.from(patterns.entries())
    .map(([key, frequency]) => {
      const [hour, day] = key.split('_').map(Number);
      return { hour, day_of_week: day, frequency };
    })
    .sort((a, b) => b.frequency - a.frequency)
    .slice(0, 5);
}

function calculateParticipantSensitivity(participantImpacts: Map<string, number[]>): Array<{sender: string; sensitivity_score: number; typical_response: number}> {
  return Array.from(participantImpacts.entries())
    .map(([sender, impacts]) => ({
      sender,
      sensitivity_score: impacts.reduce((a, b) => a + b, 0) / impacts.length,
      typical_response: impacts.reduce((a, b) => a + b, 0) / impacts.length
    }))
    .sort((a, b) => b.sensitivity_score - a.sensitivity_score)
    .slice(0, 10);
}

function calculateTemporalAnalysis(peaksAndValleys: EmotionalPeak[], dailyData: Array<{date: string, sentiment: number}>) {
  // Calculate hourly volatility
  const hourlyVolatility = Array.from({length: 24}, (_, hour) => {
    const hourPeaks = peaksAndValleys.filter(p => new Date(p.timestamp).getHours() === hour);
    return {
      hour,
      avg_volatility: hourPeaks.length > 0 ? 
        hourPeaks.reduce((sum, p) => sum + Math.abs(p.sentiment_score), 0) / hourPeaks.length : 0,
      peak_frequency: hourPeaks.filter(p => p.type === 'peak').length,
      valley_frequency: hourPeaks.filter(p => p.type === 'valley').length
    };
  });
  
  // Calculate daily volatility
  const dailyVolatility = Array.from({length: 7}, (_, day) => {
    const dayPeaks = peaksAndValleys.filter(p => new Date(p.timestamp).getDay() === day);
    const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    
    return {
      day_of_week: day,
      day_name: dayNames[day],
      avg_volatility: dayPeaks.length > 0 ? 
        dayPeaks.reduce((sum, p) => sum + Math.abs(p.sentiment_score), 0) / dayPeaks.length : 0,
      emotional_events: dayPeaks.length
    };
  });
  
  // Calculate monthly trends
  const monthlyGroups = new Map<string, {peaks: EmotionalPeak[], sentiments: number[]}>();
  peaksAndValleys.forEach(peak => {
    const month = peak.date.substring(0, 7); // YYYY-MM
    if (!monthlyGroups.has(month)) {
      monthlyGroups.set(month, {peaks: [], sentiments: []});
    }
    monthlyGroups.get(month)!.peaks.push(peak);
  });
  
  // Add daily sentiments to monthly groups
  dailyData.forEach(day => {
    const month = day.date.substring(0, 7);
    if (monthlyGroups.has(month)) {
      monthlyGroups.get(month)!.sentiments.push(day.sentiment);
    }
  });
  
  const monthlyTrends = Array.from(monthlyGroups.entries()).map(([month, data]) => {
    const variance = data.sentiments.length > 1 ? 
      data.sentiments.reduce((sum, s, _, arr) => {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        return sum + Math.pow(s - mean, 2);
      }, 0) / data.sentiments.length : 0;
    
    return {
      month,
      emotional_stability: 1 - Math.min(variance, 1), // Inverse of variance
      peak_count: data.peaks.filter(p => p.type === 'peak').length,
      valley_count: data.peaks.filter(p => p.type === 'valley').length,
      avg_sentiment: data.sentiments.reduce((a, b) => a + b, 0) / data.sentiments.length
    };
  }).sort((a, b) => a.month.localeCompare(b.month));
  
  return {
    hourly_volatility: hourlyVolatility,
    daily_volatility: dailyVolatility,
    monthly_trends: monthlyTrends
  };
}

function calculateRecoveryAnalysis(peaksAndValleys: EmotionalPeak[]) {
  const peakRecoveries = peaksAndValleys
    .filter(p => p.type === 'peak' && p.context.time_to_recovery > 0)
    .map(p => p.context.time_to_recovery);
  
  const valleyRecoveries = peaksAndValleys
    .filter(p => p.type === 'valley' && p.context.time_to_recovery > 0)
    .map(p => p.context.time_to_recovery);
  
  const allRecoveries = peaksAndValleys.filter(p => p.context.time_to_recovery > 0);
  
  return {
    avg_peak_recovery: peakRecoveries.length > 0 ? 
      peakRecoveries.reduce((a, b) => a + b, 0) / peakRecoveries.length : 0,
    avg_valley_recovery: valleyRecoveries.length > 0 ? 
      valleyRecoveries.reduce((a, b) => a + b, 0) / valleyRecoveries.length : 0,
    fastest_recovery: allRecoveries.reduce((fastest, current) => 
      current.context.time_to_recovery < fastest.context.time_to_recovery ? current : fastest,
      allRecoveries[0] || {} as EmotionalPeak),
    slowest_recovery: allRecoveries.reduce((slowest, current) => 
      current.context.time_to_recovery > slowest.context.time_to_recovery ? current : slowest,
      allRecoveries[0] || {} as EmotionalPeak),
    recovery_factors: [
      { factor: 'social_support', impact_on_recovery: -0.3, frequency: 0.6 },
      { factor: 'time_passage', impact_on_recovery: -0.2, frequency: 1.0 },
      { factor: 'positive_events', impact_on_recovery: -0.4, frequency: 0.4 },
      { factor: 'distraction', impact_on_recovery: -0.1, frequency: 0.3 }
    ]
  };
}

function findMostVolatilePeriod(dailyData: Array<{date: string, sentiment: number}>): string {
  if (dailyData.length < 7) return 'insufficient_data';
  
  let maxVolatility = 0;
  let mostVolatilePeriod = '';
  
  // Check 7-day windows
  for (let i = 0; i <= dailyData.length - 7; i++) {
    const window = dailyData.slice(i, i + 7);
    const sentiments = window.map(d => d.sentiment);
    const mean = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
    const variance = sentiments.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / sentiments.length;
    
    if (variance > maxVolatility) {
      maxVolatility = variance;
      mostVolatilePeriod = `${window[0].date} to ${window[6].date}`;
    }
  }
  
  return mostVolatilePeriod;
}

function findMostStablePeriod(dailyData: Array<{date: string, sentiment: number}>): string {
  if (dailyData.length < 7) return 'insufficient_data';
  
  let minVolatility = Infinity;
  let mostStablePeriod = '';
  
  // Check 7-day windows
  for (let i = 0; i <= dailyData.length - 7; i++) {
    const window = dailyData.slice(i, i + 7);
    const sentiments = window.map(d => d.sentiment);
    const mean = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
    const variance = sentiments.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / sentiments.length;
    
    if (variance < minVolatility) {
      minVolatility = variance;
      mostStablePeriod = `${window[0].date} to ${window[6].date}`;
    }
  }
  
  return mostStablePeriod;
}