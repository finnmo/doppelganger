'use client';

import React from 'react';
import { SentimentChart } from '@/components/SentimentChart';
import { EmotionChart } from '@/components/EmotionChart';
import { AdvancedEmotionChart } from '@/components/AdvancedEmotionChart';
import { EmotionalPeaksChart } from '@/components/EmotionalPeaksChart';
import SentimentTimelineChart from '@/components/SentimentTimelineChart';
import MoodCorrelationChart from '@/components/MoodCorrelationChart';
import { InfoTooltip } from '@/components/InfoTooltip';
import { Heart, TrendingUp, Smile, BarChart3, Zap, Users, Brain } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';

export function SentimentTab() {
  const { themeStyle, getThemeClasses } = useTheme();
  const themeClasses = getThemeClasses();

  return (
    <div className={themeClasses.spacingClass}>
      {/* Header */}
      <div className="relative">
        <div className={`absolute inset-0 ${themeClasses.headerGradientClass}`}></div>
        <div className={`relative ${themeStyle === 'modern' ? 'p-8' : 'p-6'}`}>
          <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 flex flex-wrap items-center gap-2 mb-1">
            <Heart className="w-8 h-8 mr-3 text-pink-600" />
          Sentiment & Emotional Analysis
        </h2>
          <p className="text-lg text-gray-600">
            Analyze emotional patterns, sentiment trends, and mood correlations across your conversations
        </p>
        </div>
      </div>

        {/* Sentiment by Sender */}
      <div className={themeClasses.sectionCardClass}>
        {themeStyle === 'modern' ? (
          <>
            <div className={themeClasses.sectionHeaderClass('purple')}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <BarChart3 className="w-6 h-6 mr-3" />
                  <h4 className="text-xl font-bold mr-2">Sentiment by Sender</h4>
                  <InfoTooltip
                    title="Sentiment by Sender"
                    description="Analyzes the average sentiment score for each conversation participant, showing who tends to communicate with more positive, negative, or neutral emotional tone."
                    calculation="Average sentiment calculated using VADER sentiment analysis on all messages per sender. Scores range from -1 (most negative) to +1 (most positive), with compound scores representing overall emotional valence."
                    example="A sender with 0.65 average sentiment indicates predominantly positive communication, while -0.32 suggests more negative or critical messaging patterns."
                    iconColor="white"
                  />
                </div>
              </div>
              <p className="text-purple-100 mt-2">
                Comparative analysis of emotional communication patterns across participants
              </p>
            </div>
            <div className={themeClasses.sectionContentClass}>
              <SentimentChart />
            </div>
          </>
        ) : (
          <>
            <h3 className={themeClasses.sectionTitleClass}>
              <BarChart3 className="w-5 h-5 mr-2 text-purple-500" />
            Sentiment by Sender
              <div className="ml-2">
                <InfoTooltip
                  title="Sentiment by Sender"
                  description="Analyzes the average sentiment score for each conversation participant, showing who tends to communicate with more positive, negative, or neutral emotional tone."
                  calculation="Average sentiment calculated using VADER sentiment analysis on all messages per sender. Scores range from -1 (most negative) to +1 (most positive), with compound scores representing overall emotional valence."
                  example="A sender with 0.65 average sentiment indicates predominantly positive communication, while -0.32 suggests more negative or critical messaging patterns."
                  iconColor="default"
                />
              </div>
          </h3>
          <p className="text-sm text-gray-600 mb-4">
              Comparative analysis of emotional communication patterns across participants
          </p>
          <SentimentChart />
          </>
        )}
        </div>

      {/* Emotion Distribution & Advanced Analysis */}
      <div className={`grid grid-cols-1 lg:grid-cols-2 ${themeStyle === 'modern' ? 'gap-10' : 'gap-8'}`}>
        {/* Basic Emotion Distribution */}
        <div className={themeClasses.sectionCardClass}>
          {themeStyle === 'modern' ? (
            <>
              <div className={themeClasses.sectionHeaderClass('yellow')}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Smile className="w-6 h-6 mr-3" />
                    <h4 className="text-xl font-bold mr-2">Emotion Distribution</h4>
                    <InfoTooltip
                      title="Emotion Distribution"
                      description="Shows the distribution of basic emotions (joy, sadness, anger, fear, surprise) detected across all messages using natural language processing and emotion recognition algorithms."
                      calculation="Each message is analyzed for emotional content using pre-trained emotion classification models. Scores are aggregated and normalized to show relative distribution of each emotion type."
                      example="40% joy, 25% surprise, 20% sadness, 10% anger, 5% fear indicates predominantly positive emotional expression with occasional negative emotions."
                      iconColor="white"
                    />
                  </div>
                </div>
                <p className="text-yellow-100 mt-2">
                  Distribution of basic emotional expressions detected in your messages
                </p>
              </div>
              <div className={themeClasses.sectionContentClass}>
                <EmotionChart />
              </div>
            </>
          ) : (
            <>
              <h3 className={themeClasses.sectionTitleClass}>
                <Smile className="w-5 h-5 mr-2 text-yellow-500" />
            Emotion Distribution
                <div className="ml-2">
                  <InfoTooltip
                    title="Emotion Distribution"
                    description="Shows the distribution of basic emotions (joy, sadness, anger, fear, surprise) detected across all messages using natural language processing and emotion recognition algorithms."
                    calculation="Each message is analyzed for emotional content using pre-trained emotion classification models. Scores are aggregated and normalized to show relative distribution of each emotion type."
                    example="40% joy, 25% surprise, 20% sadness, 10% anger, 5% fear indicates predominantly positive emotional expression with occasional negative emotions."
                    iconColor="default"
                  />
                </div>
          </h3>
          <p className="text-sm text-gray-600 mb-4">
                Distribution of basic emotional expressions detected in your messages
          </p>
          <EmotionChart />
            </>
          )}
      </div>

      {/* Advanced Emotion Breakdown */}
        <div className={themeClasses.sectionCardClass}>
          {themeStyle === 'modern' ? (
            <>
              <div className={themeClasses.sectionHeaderClass('pink')}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Brain className="w-6 h-6 mr-3" />
                    <h4 className="text-xl font-bold mr-2">Advanced Emotion Breakdown</h4>
                    <InfoTooltip
                      title="Advanced Emotion Breakdown"
                      description="Provides detailed categorization of emotions into broader psychological categories including positive, negative, anxiety, affection, and neutral states with intensity analysis."
                      calculation="Advanced emotion classification using expanded emotion taxonomy. Messages are categorized into psychological emotion groups with intensity scoring based on linguistic markers and contextual analysis."
                      example="Positive: 45% (high intensity), Negative: 25% (moderate), Anxiety: 15% (low), Affection: 10% (high), Neutral: 5% showing strong positive communication with some underlying concerns."
                      iconColor="white"
                    />
                  </div>
                </div>
                <p className="text-pink-100 mt-2">
                  Detailed psychological emotion categories with intensity analysis
                </p>
              </div>
              <div className={themeClasses.sectionContentClass}>
                <AdvancedEmotionChart />
              </div>
            </>
          ) : (
            <>
              <h3 className={themeClasses.sectionTitleClass}>
                <Brain className="w-5 h-5 mr-2 text-pink-500" />
          Advanced Emotion Breakdown
                <div className="ml-2">
                  <InfoTooltip
                    title="Advanced Emotion Breakdown"
                    description="Provides detailed categorization of emotions into broader psychological categories including positive, negative, anxiety, affection, and neutral states with intensity analysis."
                    calculation="Advanced emotion classification using expanded emotion taxonomy. Messages are categorized into psychological emotion groups with intensity scoring based on linguistic markers and contextual analysis."
                    example="Positive: 45% (high intensity), Negative: 25% (moderate), Anxiety: 15% (low), Affection: 10% (high), Neutral: 5% showing strong positive communication with some underlying concerns."
                    iconColor="default"
                  />
                </div>
        </h3>
        <p className="text-sm text-gray-600 mb-4">
                Detailed psychological emotion categories with intensity analysis
        </p>
        <AdvancedEmotionChart />
            </>
          )}
        </div>
      </div>

      {/* Sentiment Timeline */}
      <div className={themeClasses.sectionCardClass}>
        {themeStyle === 'modern' ? (
          <>
            <div className={themeClasses.sectionHeaderClass('blue')}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <TrendingUp className="w-6 h-6 mr-3" />
                  <h4 className="text-xl font-bold mr-2">Sentiment Trends Over Time</h4>
                  <InfoTooltip
                    title="Sentiment Trends Over Time"
                    description="Tracks how sentiment patterns change over time, showing daily sentiment averages and identifying periods of positive or negative emotional communication."
                    calculation="Daily sentiment averages calculated from all messages per day. Trends identified using moving averages and statistical analysis to highlight significant sentiment shifts and patterns."
                    example="Timeline might show increased positivity during holidays, sentiment dips during stressful periods, or gradual improvement in group dynamics over months."
                    iconColor="white"
                  />
                </div>
              </div>
              <p className="text-blue-100 mt-2">
                Temporal analysis of emotional communication patterns and sentiment evolution
              </p>
            </div>
            <div className={themeClasses.sectionContentClass}>
              <SentimentTimelineChart />
            </div>
          </>
        ) : (
          <>
            <h3 className={themeClasses.sectionTitleClass}>
              <TrendingUp className="w-5 h-5 mr-2 text-blue-500" />
          Sentiment Trends Over Time
              <div className="ml-2">
                <InfoTooltip
                  title="Sentiment Trends Over Time"
                  description="Tracks how sentiment patterns change over time, showing daily sentiment averages and identifying periods of positive or negative emotional communication."
                  calculation="Daily sentiment averages calculated from all messages per day. Trends identified using moving averages and statistical analysis to highlight significant sentiment shifts and patterns."
                  example="Timeline might show increased positivity during holidays, sentiment dips during stressful periods, or gradual improvement in group dynamics over months."
                  iconColor="default"
                />
              </div>
        </h3>
        <p className="text-sm text-gray-600 mb-4">
              Temporal analysis of emotional communication patterns and sentiment evolution
        </p>
        <SentimentTimelineChart />
          </>
        )}
      </div>

        {/* Mood Correlation Analysis */}
      <div className={themeClasses.sectionCardClass}>
        {themeStyle === 'modern' ? (
          <>
            <div className={themeClasses.sectionHeaderClass('red')}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Users className="w-6 h-6 mr-3" />
                  <h4 className="text-xl font-bold mr-2">Mood Correlation Analysis</h4>
                  <InfoTooltip
                    title="Mood Correlation Analysis"
                    description="Analyzes correlations between different participants' moods and emotional states, identifying who influences group sentiment and how emotions spread through conversations."
                    calculation="Statistical correlation analysis between participants' daily sentiment scores. Pearson correlation coefficients calculated to identify strong positive/negative mood relationships and influence patterns."
                    example="Strong correlation (0.75) between Alice and Bob suggests their moods align, while negative correlation (-0.45) indicates contrasting emotional responses to similar events."
                    iconColor="white"
                  />
                </div>
              </div>
              <p className="text-red-100 mt-2">
                Statistical analysis of emotional synchronization and influence patterns between participants
              </p>
            </div>
            <div className={themeClasses.sectionContentClass}>
          <MoodCorrelationChart />
        </div>
          </>
        ) : (
          <>
            <h3 className={themeClasses.sectionTitleClass}>
              <Users className="w-5 h-5 mr-2 text-red-500" />
              Mood Correlation Analysis
              <div className="ml-2">
                <InfoTooltip
                  title="Mood Correlation Analysis"
                  description="Analyzes correlations between different participants' moods and emotional states, identifying who influences group sentiment and how emotions spread through conversations."
                  calculation="Statistical correlation analysis between participants' daily sentiment scores. Pearson correlation coefficients calculated to identify strong positive/negative mood relationships and influence patterns."
                  example="Strong correlation (0.75) between Alice and Bob suggests their moods align, while negative correlation (-0.45) indicates contrasting emotional responses to similar events."
                  iconColor="default"
                />
              </div>
          </h3>
          <p className="text-sm text-gray-600 mb-4">
              Statistical analysis of emotional synchronization and influence patterns between participants
            </p>
            <MoodCorrelationChart />
          </>
        )}
      </div>

      {/* Emotional Peaks & Valleys */}
      <div className={themeClasses.sectionCardClass}>
        {themeStyle === 'modern' ? (
          <>
            <div className={themeClasses.sectionHeaderClass('orange')}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Zap className="w-6 h-6 mr-3" />
                  <h4 className="text-xl font-bold mr-2">Emotional Peaks & Valleys</h4>
                  <InfoTooltip
                    title="Emotional Peaks & Valleys"
                    description="Identifies the most emotionally intense moments in conversations, highlighting peak positive and negative sentiment periods with context about what triggered these emotional extremes."
                    calculation="Statistical analysis to identify sentiment outliers and extreme values. Peak detection algorithms find local maxima and minima in sentiment timelines, with contextual analysis of surrounding messages."
                    example="Highest peak on Dec 25th (0.95 sentiment) during holiday celebrations, lowest valley on March 15th (-0.78 sentiment) during stressful work period, providing insights into emotional triggers."
                    iconColor="white"
                  />
                </div>
          </div>
              <p className="text-orange-100 mt-2">
                Detection and analysis of extreme emotional moments and their contextual triggers
              </p>
          </div>
            <div className={themeClasses.sectionContentClass}>
              <EmotionalPeaksChart />
          </div>
          </>
        ) : (
          <>
            <h3 className={themeClasses.sectionTitleClass}>
              <Zap className="w-5 h-5 mr-2 text-orange-500" />
              Emotional Peaks & Valleys
              <div className="ml-2">
                <InfoTooltip
                  title="Emotional Peaks & Valleys"
                  description="Identifies the most emotionally intense moments in conversations, highlighting peak positive and negative sentiment periods with context about what triggered these emotional extremes."
                  calculation="Statistical analysis to identify sentiment outliers and extreme values. Peak detection algorithms find local maxima and minima in sentiment timelines, with contextual analysis of surrounding messages."
                  example="Highest peak on Dec 25th (0.95 sentiment) during holiday celebrations, lowest valley on March 15th (-0.78 sentiment) during stressful work period, providing insights into emotional triggers."
                  iconColor="default"
                />
        </div>
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              Detection and analysis of extreme emotional moments and their contextual triggers
            </p>
            <EmotionalPeaksChart />
          </>
        )}
      </div>
    </div>
  );
} 