# Emotional Peaks & Valleys Analysis

## Overview

The Emotional Peaks & Valleys analysis identifies and analyzes extreme emotional moments in conversations, providing deep insights into sentiment extremes, their triggers, patterns, and recovery dynamics. This advanced sentiment analysis goes beyond basic sentiment tracking to understand the emotional journey of conversation participants.

## Key Features

### 🎯 Peak Detection
- **Sentiment Peaks**: Identifies periods of exceptionally positive sentiment
- **Sentiment Valleys**: Detects periods of significantly negative sentiment
- **Intensity Classification**: Categorizes events as extreme, high, or moderate intensity
- **Duration Analysis**: Measures how long emotional peaks and valleys last

### 🔍 Trigger Analysis
- **Primary Trigger Identification**: Determines what caused each emotional event
- **Confidence Scoring**: Provides confidence levels for trigger identification
- **Keyword Analysis**: Extracts relevant keywords associated with triggers
- **Contributing Factors**: Identifies secondary factors that influenced the event

### 📊 Pattern Recognition
- **Emotional Patterns**: Identifies recurring emotional patterns
- **Temporal Analysis**: Analyzes when emotional events typically occur
- **Frequency Patterns**: Tracks how often different types of events happen
- **Seasonal Trends**: Identifies time-based emotional patterns

### ⏱️ Recovery Analysis
- **Recovery Time Calculation**: Measures how long it takes to return to baseline
- **Recovery Factors**: Identifies what helps or hinders emotional recovery
- **Fastest/Slowest Cases**: Highlights extreme recovery examples
- **Recovery Patterns**: Analyzes typical recovery trajectories

## Data Structure

### EmotionalPeak Interface
```typescript
interface EmotionalPeak {
  id: string;
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
    time_to_recovery: number;
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
```

### Output Data Structure
```typescript
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
    hourly_volatility: Array<...>;
    daily_volatility: Array<...>;
    monthly_trends: Array<...>;
  };
  recovery_analysis: {
    avg_peak_recovery: number;
    avg_valley_recovery: number;
    fastest_recovery: EmotionalPeak;
    slowest_recovery: EmotionalPeak;
    recovery_factors: Array<...>;
  };
}
```

## Analysis Methodology

### 1. Peak Detection Algorithm

The system uses a multi-step approach to identify emotional peaks and valleys:

1. **Sentiment Smoothing**: Apply moving averages to reduce noise
2. **Threshold Analysis**: Identify sentiment values exceeding statistical thresholds
3. **Local Extrema Detection**: Find local maxima and minima in sentiment timelines
4. **Duration Validation**: Ensure peaks/valleys persist for meaningful durations
5. **Intensity Classification**: Categorize based on magnitude and statistical significance

#### Intensity Classification
- **Extreme**: |sentiment| ≥ 0.8 and >2 standard deviations from mean
- **High**: |sentiment| ≥ 0.6 and >1.5 standard deviations from mean
- **Moderate**: |sentiment| ≥ 0.4 and >1 standard deviation from mean

### 2. Trigger Identification

The trigger analysis system identifies what caused emotional events:

#### Trigger Categories
- **work_stress**: Work-related pressure, deadlines, conflicts
- **relationship**: Personal relationship issues or celebrations
- **health**: Health concerns or medical events
- **family**: Family-related events, good or bad
- **social**: Social gatherings, parties, social conflicts
- **achievement**: Accomplishments, promotions, successes
- **weather_events**: Weather-related mood impacts
- **technology**: Tech frustrations or breakthroughs
- **financial**: Money-related stress or windfalls
- **travel**: Travel experiences, delays, adventures

#### Trigger Detection Process
1. **Keyword Analysis**: Scan messages for trigger-related keywords
2. **Context Analysis**: Examine surrounding messages for context clues
3. **Temporal Correlation**: Look for timing patterns with known events
4. **Sentiment Correlation**: Analyze sentiment patterns typical of each trigger
5. **Confidence Scoring**: Calculate confidence based on multiple evidence sources

### 3. Pattern Recognition

The system identifies recurring emotional patterns:

#### Pattern Types
- **Stress Cycles**: Regular periods of stress followed by relief
- **Mood Swings**: Rapid alternation between positive and negative states
- **Seasonal Patterns**: Mood changes related to time of year
- **Weekly Patterns**: Consistent emotional patterns within weeks
- **Recovery Patterns**: How quickly and consistently recovery occurs

### 4. Recovery Analysis

Recovery analysis tracks how participants bounce back from emotional extremes:

#### Recovery Metrics
- **Time to Baseline**: Hours/days to return to neutral sentiment
- **Recovery Slope**: Rate of emotional recovery
- **Overshoot Analysis**: Whether recovery overshoots to opposite extreme
- **Stability Analysis**: How stable the recovery is over time

#### Recovery Factors
- **Social Support**: Impact of supportive messages from others
- **Time of Day**: Whether recovery is faster at certain times
- **Activity Type**: What types of activities aid recovery
- **Communication Frequency**: How message frequency affects recovery

## Dashboard Features

### Timeline View
- Interactive scatter plot showing peaks and valleys over time
- Color-coded intensity levels
- Clickable points for detailed information
- Trend lines and reference markers

### Triggers View
- Bar charts showing trigger frequency and impact
- Detailed trigger cards with keywords and patterns
- Trigger sensitivity analysis by participant
- Time-based trigger patterns

### Patterns View
- Emotional pattern identification and description
- Temporal volatility analysis (hourly, daily, monthly)
- Pattern frequency and characteristics
- Stability metrics over time

### Recovery View
- Recovery time statistics and comparisons
- Recovery factor analysis with impact scores
- Fastest and slowest recovery case studies
- Recovery pattern visualization

### Conversation Filtering
- **Automatic Filtering**: When conversations are selected via the conversation filter, all emotional peaks analysis is automatically filtered to show only data from the selected conversations
- **Recalculated Metrics**: Summary statistics, trigger analysis, and recovery metrics are dynamically recalculated based on filtered data
- **Empty State Handling**: When no emotional peaks exist in filtered conversations, appropriate empty states are displayed
- **Filtering Indicator**: Visual indicator shows when data is filtered and how many conversations are included

## Interpretation Guide

### Understanding Sentiment Scores
- **+0.8 to +1.0**: Extreme positive emotions (euphoria, celebration)
- **+0.5 to +0.8**: High positive emotions (happiness, excitement)
- **+0.2 to +0.5**: Moderate positive emotions (contentment, satisfaction)
- **-0.2 to +0.2**: Neutral emotions (balanced, calm)
- **-0.5 to -0.2**: Moderate negative emotions (disappointment, concern)
- **-0.8 to -0.5**: High negative emotions (sadness, frustration)
- **-1.0 to -0.8**: Extreme negative emotions (despair, rage)

### Trigger Confidence Levels
- **90-100%**: Very high confidence, multiple strong indicators
- **70-89%**: High confidence, clear evidence present
- **50-69%**: Moderate confidence, some indicators present
- **30-49%**: Low confidence, weak or conflicting evidence
- **0-29%**: Very low confidence, mostly speculative

### Recovery Time Interpretation
- **< 6 hours**: Very fast recovery, resilient response
- **6-24 hours**: Fast recovery, good emotional regulation
- **1-3 days**: Normal recovery, typical processing time
- **3-7 days**: Slow recovery, may need support
- **> 7 days**: Very slow recovery, potential concern

## Usage Examples

### Basic Analysis
```typescript
// Load emotional peaks data
const data = await fetch('/data/emotionalPeaks.json').then(r => r.json());

// Find extreme events
const extremeEvents = data.peaks_and_valleys.filter(
  event => event.intensity === 'extreme'
);

// Analyze most common triggers
const triggerFrequency = data.trigger_analysis.sort(
  (a, b) => b.frequency - a.frequency
);
```

### Advanced Filtering
```typescript
// Find work-related stress valleys
const workStress = data.peaks_and_valleys.filter(
  event => event.type === 'valley' && 
           event.trigger_analysis.primary_trigger === 'work_stress'
);

// Analyze recovery patterns for specific participant
const aliceRecovery = data.peaks_and_valleys
  .filter(event => event.participants.some(p => p.sender === 'Alice'))
  .map(event => event.context.time_to_recovery);
```

### Pattern Analysis
```typescript
// Find seasonal patterns
const seasonalPatterns = data.emotional_patterns.filter(
  pattern => pattern.pattern_type.includes('seasonal')
);

// Analyze volatility by day of week
const weekdayVolatility = data.temporal_analysis.daily_volatility
  .sort((a, b) => b.avg_volatility - a.avg_volatility);
```

## Data Processing Pipeline

### 1. Data Collection
```sql
-- Extract messages with sentiment data
SELECT 
  id, content, sender, timestamp_ms, conversation_id,
  compound, positive, negative, neutral
FROM messages 
WHERE compound IS NOT NULL
ORDER BY timestamp_ms;
```

### 2. Peak Detection
- Apply moving averages for smoothing
- Calculate statistical thresholds
- Identify local extrema
- Validate duration and intensity
- Classify intensity levels

### 3. Trigger Analysis
- Extract keywords from peak/valley periods
- Analyze temporal context
- Apply machine learning classification
- Calculate confidence scores
- Identify contributing factors

### 4. Pattern Recognition
- Group similar emotional events
- Analyze temporal distributions
- Identify recurring patterns
- Calculate pattern characteristics
- Generate pattern descriptions

### 5. Recovery Analysis
- Track sentiment progression after peaks/valleys
- Calculate recovery metrics
- Identify recovery factors
- Analyze recovery patterns
- Generate recovery insights

## Performance Considerations

### Optimization Strategies
- **Batched Processing**: Process messages in time-based chunks
- **Caching**: Cache intermediate calculations for reuse
- **Indexing**: Proper database indexing for timestamp queries
- **Parallel Processing**: Process different conversations in parallel
- **Memory Management**: Stream processing for large datasets

### Scalability
- **Time Complexity**: O(n log n) for peak detection
- **Space Complexity**: O(n) for data storage
- **Database Queries**: Optimized with proper indexing
- **Memory Usage**: Streaming approach for large datasets

## Testing

The system includes comprehensive tests covering:
- Peak detection accuracy
- Trigger identification correctness
- Pattern recognition validation
- Recovery calculation accuracy
- Edge case handling
- Performance benchmarks

### Test Categories
- **Unit Tests**: Individual function testing
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Speed and memory usage testing
- **Accuracy Tests**: Validation against known datasets
- **Edge Case Tests**: Handling of unusual scenarios

## Future Enhancements

### Planned Features
- **Machine Learning Enhancement**: Improved trigger classification
- **Predictive Analysis**: Predict future emotional events
- **Intervention Suggestions**: Recommend actions for recovery
- **Cross-Conversation Analysis**: Compare patterns across conversations
- **Real-time Monitoring**: Live emotional state tracking

### Research Areas
- **Emotion Contagion**: How emotions spread between participants
- **Trigger Sensitivity**: Individual differences in trigger response
- **Recovery Strategies**: What helps people recover faster
- **Pattern Evolution**: How emotional patterns change over time
- **Cultural Factors**: How culture affects emotional expression

## Troubleshooting

### Common Issues
1. **No Peaks Detected**: Check sentiment data quality and thresholds
2. **Incorrect Triggers**: Verify keyword lists and classification logic
3. **Missing Recovery Data**: Ensure sufficient follow-up messages
4. **Performance Issues**: Check database indexing and query optimization
5. **Memory Problems**: Implement streaming for large datasets

### Debug Information
The processor logs detailed information for debugging:
- Peak detection statistics
- Trigger classification confidence
- Pattern recognition results
- Recovery calculation details
- Performance metrics

### Data Quality Checks
- Sentiment score validation
- Message content verification
- Timestamp consistency
- Participant identification
- Conversation continuity 