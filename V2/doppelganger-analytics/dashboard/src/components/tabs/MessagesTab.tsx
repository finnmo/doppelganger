'use client';

import React from 'react';
import { MessageTrendChart } from '@/components/MessageTrendChart';
import { TopWordsChart } from '@/components/TopWordsChart';
import { URLDomainChart } from '@/components/URLDomainChart';
import { ImportantMessagesAnalysis } from '@/components/ImportantMessagesAnalysis';
import { MessageLengthChart } from '@/components/MessageLengthChart';
import ContentTypeChart from '@/components/ContentTypeChart';
import { InfoTooltip } from '@/components/InfoTooltip';
import { MessageCircle, Hash, Link, FileText } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';

export function MessagesTab() {
  const { themeStyle, getThemeClasses } = useTheme();
  const themeClasses = getThemeClasses();

  return (
    <div className={themeClasses.spacingClass}>
      {/* Header */}
      <div className="relative">
        <div className={`absolute inset-0 ${themeClasses.headerGradientClass}`}></div>
        <div className={`relative ${themeStyle === 'modern' ? 'p-8' : 'p-6'}`}>
          <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 flex flex-wrap items-center gap-2 mb-1">
            <MessageCircle className="w-8 h-8 mr-3 text-blue-600" />
          Messages & Content Analysis
        </h2>
          <p className="text-lg text-gray-600">
          Deep analysis of message content, patterns, and communication trends
        </p>
        </div>
      </div>

      {/* Message Volume & Content Analysis */}
      <div className={`grid grid-cols-1 lg:grid-cols-2 ${themeStyle === 'modern' ? 'gap-10' : 'gap-8'}`}>
        {/* Message Volume Trends */}
        <div className={themeClasses.sectionCardClass}>
          {themeStyle === 'modern' ? (
            <>
              <div className={themeClasses.sectionHeaderClass('blue')}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Hash className="w-6 h-6 mr-3" />
                    <h4 className="text-xl font-bold mr-2">Message Volume Over Time</h4>
                    <InfoTooltip
                      title="Message Volume Over Time"
                      description="Line chart showing how message volume changes over time across your selected conversations, helping identify busy periods and communication trends."
                      calculation="Messages are grouped by time period and plotted as a time series, with peaks indicating high activity periods"
                      example="You might see spikes during work hours, drops on weekends, or seasonal patterns like increased activity during holidays."
                      iconColor="white"
                    />
                  </div>
                </div>
                <p className="text-blue-100 mt-2">
                  Track communication frequency and identify peak activity periods
                </p>
              </div>
              <div className={themeClasses.sectionContentClass}>
                <div className="h-80">
                  <MessageTrendChart />
                </div>
              </div>
            </>
          ) : (
            <>
              <h3 className={themeClasses.sectionTitleClass}>
          <Hash className="w-5 h-5 mr-2 text-blue-500" />
                Message Volume Over Time
                <div className="ml-2">
                  <InfoTooltip
                    title="Message Volume Over Time"
                    description="Line chart showing how message volume changes over time across your selected conversations, helping identify busy periods and communication trends."
                    calculation="Messages are grouped by time period and plotted as a time series, with peaks indicating high activity periods"
                    example="You might see spikes during work hours, drops on weekends, or seasonal patterns like increased activity during holidays."
                    iconColor="default"
                  />
                </div>
        </h3>
            <p className="text-sm text-gray-600 mb-4">
              Track communication frequency and identify peak activity periods
            </p>
              <div className="h-80">
              <MessageTrendChart />
            </div>
            </>
          )}
          </div>

          {/* Content Type Breakdown */}
        <div className={themeClasses.sectionCardClass}>
          {themeStyle === 'modern' ? (
            <>
              <div className={themeClasses.sectionHeaderClass('green')}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <FileText className="w-6 h-6 mr-3" />
                    <h4 className="text-xl font-bold mr-2">Content Type Distribution</h4>
                    <InfoTooltip
                      title="Content Type Distribution"
                      description="Categorizes and analyzes different types of message content including text length variations, emoji-only messages, link sharing, and media notifications."
                      calculation="Messages are classified into types: short_text (1-10 words), medium_text (11-50 words), long_text (50+ words), emoji_only, link_share, media_notification, and system_event. Percentages calculated from total message count."
                      example="If you have 1,000 messages: 60% short_text, 25% medium_text, 10% emoji_only, 3% link_share, 2% media_notification shows your communication style preferences."
                      iconColor="white"
                    />
                  </div>
                </div>
                <p className="text-green-100 mt-2">
                  Breakdown of message types (text, media, reactions)
                </p>
              </div>
              <div className={themeClasses.sectionContentClass}>
                <div className="h-80">
                  <ContentTypeChart />
                </div>
              </div>
            </>
          ) : (
            <>
              <h3 className={themeClasses.sectionTitleClass}>
                <FileText className="w-5 h-5 mr-2 text-green-500" />
              Content Type Distribution
                <div className="ml-2">
                  <InfoTooltip
                    title="Content Type Distribution"
                    description="Categorizes and analyzes different types of message content including text length variations, emoji-only messages, link sharing, and media notifications."
                    calculation="Messages are classified into types: short_text (1-10 words), medium_text (11-50 words), long_text (50+ words), emoji_only, link_share, media_notification, and system_event. Percentages calculated from total message count."
                    example="If you have 1,000 messages: 60% short_text, 25% medium_text, 10% emoji_only, 3% link_share, 2% media_notification shows your communication style preferences."
                    iconColor="default"
                  />
                </div>
              </h3>
            <p className="text-sm text-gray-600 mb-4">
              Breakdown of message types (text, media, reactions)
            </p>
              <div className="h-80">
              <ContentTypeChart />
            </div>
            </>
          )}
        </div>
      </div>

      {/* Word Analysis */}
      <div className={themeClasses.sectionCardClass}>
        {themeStyle === 'modern' ? (
          <>
            <div className={themeClasses.sectionHeaderClass('purple')}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Hash className="w-6 h-6 mr-3" />
                  <h4 className="text-xl font-bold mr-2">Most Frequently Used Words</h4>
                  <InfoTooltip
                    title="Most Frequently Used Words"
                    description="Natural language processing analysis showing the most frequently used words across all your conversations. Word cloud visualization where size represents frequency."
                    calculation="Text content is tokenized, common stop words removed, and word frequencies calculated. Words with Unicode apostrophes (don't, can't) are preserved intact using enhanced regex patterns."
                    example="Common words might include names, greetings, expressions, and topic-specific terms. Size indicates relative frequency - larger words appear more often in your conversations."
                    iconColor="white"
                  />
                </div>
              </div>
              <p className="text-purple-100 mt-2">
                Most frequently used words across all conversations - word size represents frequency
              </p>
            </div>
            <div className={themeClasses.sectionContentClass}>
              <TopWordsChart />
            </div>
          </>
        ) : (
          <>
            <h3 className={themeClasses.sectionTitleClass}>
          <Hash className="w-5 h-5 mr-2 text-purple-500" />
              Most Frequently Used Words
              <div className="ml-2">
                <InfoTooltip
                  title="Most Frequently Used Words"
                  description="Natural language processing analysis showing the most frequently used words across all your conversations. Word cloud visualization where size represents frequency."
                  calculation="Text content is tokenized, common stop words removed, and word frequencies calculated. Words with Unicode apostrophes (don't, can't) are preserved intact using enhanced regex patterns."
                  example="Common words might include names, greetings, expressions, and topic-specific terms. Size indicates relative frequency - larger words appear more often in your conversations."
                  iconColor="default"
                />
              </div>
        </h3>
          <p className="text-sm text-gray-600 mb-4">
            Most frequently used words across all conversations - word size represents frequency
          </p>
          <TopWordsChart />
          </>
        )}
      </div>

      {/* Link & Message Analysis */}
      <div className={`grid grid-cols-1 lg:grid-cols-2 ${themeStyle === 'modern' ? 'gap-10' : 'gap-8'}`}>
          {/* URL Domain Analysis */}
        <div className={themeClasses.sectionCardClass}>
          {themeStyle === 'modern' ? (
            <>
              <div className={themeClasses.sectionHeaderClass('orange')}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Link className="w-6 h-6 mr-3" />
                    <h4 className="text-xl font-bold mr-2">URL Domain Analysis</h4>
                    <InfoTooltip
                      title="URL Domain Analysis"
                      description="Analyzes all URLs shared in your conversations, extracting and counting domain names to show which websites and services are most commonly shared."
                      calculation="URLs are detected using linkify, domains extracted and normalized (e.g., www.example.com → example.com), then counted and ranked by frequency."
                      example="Top domains might include social media (instagram.com, youtube.com), news sites, or work tools, revealing your group's interests and information sharing patterns."
                      iconColor="white"
                    />
                  </div>
                </div>
                <p className="text-orange-100 mt-2">
                  Most shared domains and link sharing patterns
                </p>
              </div>
              <div className={themeClasses.sectionContentClass}>
                <div className="h-80">
                  <URLDomainChart />
                </div>
              </div>
            </>
          ) : (
            <>
              <h3 className={themeClasses.sectionTitleClass}>
                <Link className="w-5 h-5 mr-2 text-orange-500" />
              URL Domain Analysis
                <div className="ml-2">
                  <InfoTooltip
                    title="URL Domain Analysis"
                    description="Analyzes all URLs shared in your conversations, extracting and counting domain names to show which websites and services are most commonly shared."
                    calculation="URLs are detected using linkify, domains extracted and normalized (e.g., www.example.com → example.com), then counted and ranked by frequency."
                    example="Top domains might include social media (instagram.com, youtube.com), news sites, or work tools, revealing your group's interests and information sharing patterns."
                    iconColor="default"
                  />
                </div>
              </h3>
            <p className="text-sm text-gray-600 mb-4">
              Most shared domains and link sharing patterns
            </p>
            <div className="h-80">
              <URLDomainChart />
            </div>
            </>
          )}
          </div>

          {/* Message Length Distribution */}
        <div className={themeClasses.sectionCardClass}>
          {themeStyle === 'modern' ? (
            <>
              <div className={themeClasses.sectionHeaderClass('blue')}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <FileText className="w-6 h-6 mr-3" />
                    <h4 className="text-xl font-bold mr-2">Message Length Distribution</h4>
                    <InfoTooltip
                      title="Message Length Distribution"
                      description="Analyzes the distribution of message lengths to understand communication styles and preferences across different word count ranges."
                      calculation="Messages are categorized into length buckets: Very short (1-5 words), Short (6-15 words), Medium (16-30 words), Long (31-50 words), and Very long (51+ words). Percentages calculated from total message count."
                      example="A distribution showing 40% short messages, 35% very short, 20% medium, and 5% long messages indicates preference for brief, quick communication."
                      iconColor="white"
                    />
                  </div>
                </div>
                <p className="text-blue-100 mt-2">
                  Distribution of message lengths and communication styles
                </p>
              </div>
              <div className={themeClasses.sectionContentClass}>
                <div className="h-80">
                  <MessageLengthChart />
                </div>
              </div>
            </>
          ) : (
            <>
              <h3 className={themeClasses.sectionTitleClass}>
                <FileText className="w-5 h-5 mr-2 text-blue-500" />
              Message Length Distribution
                <div className="ml-2">
                  <InfoTooltip
                    title="Message Length Distribution"
                    description="Analyzes the distribution of message lengths to understand communication styles and preferences across different word count ranges."
                    calculation="Messages are categorized into length buckets: Very short (1-5 words), Short (6-15 words), Medium (16-30 words), Long (31-50 words), and Very long (51+ words). Percentages calculated from total message count."
                    example="A distribution showing 40% short messages, 35% very short, 20% medium, and 5% long messages indicates preference for brief, quick communication."
                    iconColor="default"
                  />
                </div>
              </h3>
            <p className="text-sm text-gray-600 mb-4">
              Distribution of message lengths and communication styles
            </p>
            <div className="h-80">
              <MessageLengthChart />
            </div>
            </>
          )}
        </div>
      </div>

      {/* Important Messages Section */}
      <div className={themeClasses.sectionCardClass}>
        {themeStyle === 'modern' ? (
          <>
            <div className={themeClasses.sectionHeaderClass('yellow')}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <MessageCircle className="w-6 h-6 mr-3" />
                  <h4 className="text-xl font-bold mr-2">🌟 Important Messages Analysis</h4>
                  <InfoTooltip
                    title="Important Messages Analysis"
                    description="Identifies and scores messages by importance using multiple factors including content length, sentiment strength, engagement, media presence, and emotional content."
                    calculation="Multi-factor scoring system analyzing: content length (longer often more important), sentiment strength (high positive/negative), engagement (responses, reactions), media presence, question content, emotional indicators, and timing factors. Composite score normalized to 0-1 scale."
                    example="A long message with strong sentiment that received multiple responses and contains media might score 0.85, while a short acknowledgment might score 0.15."
                    iconColor="white"
                  />
                </div>
              </div>
              <p className="text-yellow-100 mt-2">
                Messages identified as important based on engagement, sentiment patterns, and content analysis
              </p>
            </div>
            <div className={themeClasses.sectionContentClass}>
              <ImportantMessagesAnalysis />
            </div>
          </>
        ) : (
          <>
            <h3 className={themeClasses.sectionTitleClass}>
              <MessageCircle className="w-5 h-5 mr-2 text-yellow-500" />
          🌟 Important Messages Analysis
              <div className="ml-2">
                <InfoTooltip
                  title="Important Messages Analysis"
                  description="Identifies and scores messages by importance using multiple factors including content length, sentiment strength, engagement, media presence, and emotional content."
                  calculation="Multi-factor scoring system analyzing: content length (longer often more important), sentiment strength (high positive/negative), engagement (responses, reactions), media presence, question content, emotional indicators, and timing factors. Composite score normalized to 0-1 scale."
                  example="A long message with strong sentiment that received multiple responses and contains media might score 0.85, while a short acknowledgment might score 0.15."
                  iconColor="default"
                />
              </div>
        </h3>
          <p className="text-sm text-gray-600 mb-4">
            Messages identified as important based on engagement, sentiment patterns, and content analysis
          </p>
          <ImportantMessagesAnalysis />
          </>
        )}
      </div>
    </div>
  );
} 