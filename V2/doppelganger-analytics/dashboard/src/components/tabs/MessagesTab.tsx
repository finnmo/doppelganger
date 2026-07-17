'use client';

import React from 'react';
import { MessageTrendChart } from '@/components/MessageTrendChart';
import { TopWordsChart, TopWordsFullscreen } from '@/components/TopWordsChart';
import { URLDomainChart, URLDomainFullscreen } from '@/components/URLDomainChart';
import { ImportantMessagesAnalysis, ImportantMessagesFullscreen } from '@/components/ImportantMessagesAnalysis';
import { MessageLengthChart } from '@/components/MessageLengthChart';
import ContentTypeChart, { ContentTypeFullscreen } from '@/components/ContentTypeChart';
import { ChartCard } from '@/components/ui/ChartCard';
import { Hash, Link, FileText, Star } from 'lucide-react';
import { TAB_VIEWPORT, CARD_GRID_ROW, CARD_FILL, BODY_FILL } from '@/lib/layout';

/**
 * Messages & Content — single-screen grid.
 * Row 1: volume trend · content types · message lengths.
 * Row 2: word cloud · URL domains · important messages.
 */
export function MessagesTab() {
  return (
    <div className={TAB_VIEWPORT}>
      <div className={CARD_GRID_ROW('grid-cols-1 lg:grid-cols-2 xl:grid-cols-3')}>
        <ChartCard
          title="Message Volume Over Time"
          icon={Hash}
          accent="blue"
          className={CARD_FILL}
          tooltip={{
            description:
              'Line chart showing how message volume changes over time across your selected conversations, helping identify busy periods and communication trends.',
            calculation:
              'Messages are grouped by time period and plotted as a time series, with peaks indicating high activity periods',
            example:
              'You might see spikes during work hours, drops on weekends, or seasonal patterns like increased activity during holidays.',
          }}
          bodyClassName={BODY_FILL}
        >
          <MessageTrendChart />
        </ChartCard>

        <ChartCard
          title="Content Type Distribution"
          icon={FileText}
          accent="green"
          className={CARD_FILL}
          tooltip={{
            description:
              'Categorizes and analyzes different types of message content including text length variations, emoji-only messages, link sharing, and media notifications.',
            calculation:
              'Messages are classified into types: short_text (1-10 words), medium_text (11-50 words), long_text (50+ words), emoji_only, link_share, media_notification, and system_event. Percentages calculated from total message count.',
            example:
              'If you have 1,000 messages: 60% short_text, 25% medium_text, 10% emoji_only, 3% link_share, 2% media_notification shows your communication style preferences.',
          }}
          bodyClassName={BODY_FILL}
          fullscreenChildren={<ContentTypeFullscreen />}
        >
          <ContentTypeChart />
        </ChartCard>

        <ChartCard
          title="Message Length Distribution"
          icon={FileText}
          accent="indigo"
          className={CARD_FILL}
          tooltip={{
            description:
              'Analyzes the distribution of message lengths to understand communication styles and preferences across different word count ranges.',
            calculation:
              'Messages are categorized into length buckets: Very short (1-5 words), Short (6-15 words), Medium (16-30 words), Long (31-50 words), and Very long (51+ words). Percentages calculated from total message count.',
            example:
              'A distribution showing 40% short messages, 35% very short, 20% medium, and 5% long messages indicates preference for brief, quick communication.',
          }}
          bodyClassName={BODY_FILL}
        >
          <MessageLengthChart />
        </ChartCard>
      </div>

      <div className={CARD_GRID_ROW('grid-cols-1 lg:grid-cols-2 xl:grid-cols-3')}>
        <ChartCard
          title="Most Frequently Used Words"
          icon={Hash}
          accent="purple"
          className={CARD_FILL}
          tooltip={{
            description:
              'Natural language processing analysis showing the most frequently used words across all your conversations. Word cloud visualization where size represents frequency.',
            calculation:
              "Text content is tokenized, common stop words removed, and word frequencies calculated. Words with Unicode apostrophes (don't, can't) are preserved intact using enhanced regex patterns.",
            example:
              'Common words might include names, greetings, expressions, and topic-specific terms. Size indicates relative frequency - larger words appear more often in your conversations.',
          }}
          bodyClassName={BODY_FILL}
          fullscreenChildren={<TopWordsFullscreen />}
        >
          <TopWordsChart />
        </ChartCard>

        <ChartCard
          title="URL Domain Analysis"
          icon={Link}
          accent="orange"
          className={CARD_FILL}
          tooltip={{
            description:
              'Analyzes all URLs shared in your conversations, extracting and counting domain names to show which websites and services are most commonly shared.',
            calculation:
              'URLs are detected using linkify, domains extracted and normalized (e.g., www.example.com → example.com), then counted and ranked by frequency.',
            example:
              "Top domains might include social media (instagram.com, youtube.com), news sites, or work tools, revealing your group's interests and information sharing patterns.",
          }}
          bodyClassName={BODY_FILL}
          fullscreenChildren={<URLDomainFullscreen />}
        >
          <URLDomainChart />
        </ChartCard>

        <ChartCard
          title="Important Messages"
          icon={Star}
          accent="yellow"
          className={CARD_FILL}
          tooltip={{
            description:
              'Identifies and scores messages by importance using multiple factors including content length, sentiment strength, engagement, media presence, and emotional content.',
            calculation:
              'Multi-factor scoring system analyzing: content length (longer often more important), sentiment strength (high positive/negative), engagement (responses, reactions), media presence, question content, emotional indicators, and timing factors. Composite score normalized to 0-1 scale.',
            example:
              'A long message with strong sentiment that received multiple responses and contains media might score 0.85, while a short acknowledgment might score 0.15.',
          }}
          bodyClassName={BODY_FILL}
          fullscreenChildren={<ImportantMessagesFullscreen />}
        >
          <ImportantMessagesAnalysis />
        </ChartCard>
      </div>
    </div>
  );
}
