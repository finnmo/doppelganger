'use client';

import React from 'react';
import { SentimentChart } from '@/components/SentimentChart';
import { EmotionChart } from '@/components/EmotionChart';
import { AdvancedEmotionChart } from '@/components/AdvancedEmotionChart';
import { EmotionalPeaksChart } from '@/components/EmotionalPeaksChart';
import SentimentTimelineChart from '@/components/SentimentTimelineChart';
import MoodCorrelationChart from '@/components/MoodCorrelationChart';
import { ChartCard } from '@/components/ui/ChartCard';
import { Heart, Smile, Brain, TrendingUp, Users, Zap } from 'lucide-react';
import { CHART_MD, CHART_LG, GRID_GAP, TAB_STACK } from '@/lib/layout';

/**
 * Sentiment & Emotions — single-screen grid.
 * Row 1: sentiment by sender · emotion distribution · advanced emotions.
 * Row 2: sentiment timeline · mood correlation · emotional peaks
 * (the richer analyses scroll internally so the page never grows).
 */
export function SentimentTab() {
  return (
    <div className={TAB_STACK}>
      <div className={`grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 ${GRID_GAP}`}>
        <ChartCard
          title="Sentiment by Sender"
          icon={Heart}
          accent="red"
          tooltip={{
            description:
              'Analyzes the average sentiment score for each conversation participant, showing who tends to communicate with more positive, negative, or neutral emotional tone.',
            calculation:
              'Average sentiment calculated using VADER sentiment analysis on all messages per sender. Scores range from -1 (most negative) to +1 (most positive), with compound scores representing overall emotional valence.',
            example:
              'A sender with 0.65 average sentiment indicates predominantly positive communication, while -0.32 suggests more negative or critical messaging patterns.',
          }}
          bodyClassName={`${CHART_MD} overflow-y-auto`}
        >
          <SentimentChart />
        </ChartCard>

        <ChartCard
          title="Emotion Distribution"
          icon={Smile}
          accent="pink"
          tooltip={{
            description:
              'Shows the distribution of basic emotions (joy, sadness, anger, fear, surprise) detected across all messages using natural language processing and emotion recognition algorithms.',
            calculation:
              'Each message is analyzed for emotional content using pre-trained emotion classification models. Scores are aggregated and normalized to show relative distribution of each emotion type.',
            example:
              '40% joy, 25% surprise, 20% sadness, 10% anger, 5% fear indicates predominantly positive emotional expression with occasional negative emotions.',
          }}
          bodyClassName={`${CHART_MD} overflow-y-auto`}
        >
          <EmotionChart />
        </ChartCard>

        <ChartCard
          title="Advanced Emotion Breakdown"
          icon={Brain}
          accent="purple"
          tooltip={{
            description:
              'Provides detailed categorization of emotions into broader psychological categories including positive, negative, anxiety, affection, and neutral states with intensity analysis.',
            calculation:
              'Advanced emotion classification using expanded emotion taxonomy. Messages are categorized into psychological emotion groups with intensity scoring based on linguistic markers and contextual analysis.',
            example:
              'Positive: 45% (high intensity), Negative: 25% (moderate), Anxiety: 15% (low), Affection: 10% (high), Neutral: 5% showing strong positive communication with some underlying concerns.',
          }}
          bodyClassName={`${CHART_MD} overflow-y-auto`}
        >
          <AdvancedEmotionChart />
        </ChartCard>
      </div>

      <div className={`grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 ${GRID_GAP}`}>
        <ChartCard
          title="Sentiment Trends Over Time"
          icon={TrendingUp}
          accent="blue"
          tooltip={{
            description:
              'Tracks how sentiment patterns change over time, showing daily sentiment averages and identifying periods of positive or negative emotional communication.',
            calculation:
              'Daily sentiment averages calculated from all messages per day. Trends identified using moving averages and statistical analysis to highlight significant sentiment shifts and patterns.',
            example:
              'Timeline might show increased positivity during holidays, sentiment dips during stressful periods, or gradual improvement in group dynamics over months.',
          }}
          bodyClassName={`${CHART_LG} overflow-y-auto`}
        >
          <SentimentTimelineChart />
        </ChartCard>

        <ChartCard
          title="Mood Correlation Analysis"
          icon={Users}
          accent="teal"
          tooltip={{
            description:
              "Analyzes correlations between different participants' moods and emotional states, identifying who influences group sentiment and how emotions spread through conversations.",
            calculation:
              "Statistical correlation analysis between participants' daily sentiment scores. Pearson correlation coefficients calculated to identify strong positive/negative mood relationships and influence patterns.",
            example:
              'Strong correlation (0.75) between Alice and Bob suggests their moods align, while negative correlation (-0.45) indicates contrasting emotional responses to similar events.',
          }}
          bodyClassName={`${CHART_LG} overflow-y-auto`}
        >
          <MoodCorrelationChart />
        </ChartCard>

        <ChartCard
          title="Emotional Peaks & Valleys"
          icon={Zap}
          accent="orange"
          tooltip={{
            description:
              'Identifies the most emotionally intense moments in conversations, highlighting peak positive and negative sentiment periods with context about what triggered these emotional extremes.',
            calculation:
              'Statistical analysis to identify sentiment outliers and extreme values. Peak detection algorithms find local maxima and minima in sentiment timelines, with contextual analysis of surrounding messages.',
            example:
              'Highest peak on Dec 25th (0.95 sentiment) during holiday celebrations, lowest valley on March 15th (-0.78 sentiment) during stressful work period, providing insights into emotional triggers.',
          }}
          bodyClassName={`${CHART_LG} overflow-y-auto`}
        >
          <EmotionalPeaksChart />
        </ChartCard>
      </div>
    </div>
  );
}
