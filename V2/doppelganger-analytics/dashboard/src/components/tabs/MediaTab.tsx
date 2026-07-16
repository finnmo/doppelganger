'use client';

import React from 'react';
import { MediaChart } from '@/components/MediaChart';
import { AttachmentTypeChart } from '@/components/AttachmentTypeChart';
import { MediaBySenderChart } from '@/components/MediaBySenderChart';
import ReactionMetricsChart from '@/components/ReactionMetricsChart';
import MediaEngagementChart from '@/components/MediaEngagementChart';
import { ReactionSummaryCards } from '@/components/ReactionSummaryCards';
import { MediaInsightsPanel } from '@/components/MediaInsightsPanel';
import { Image as ImageIcon, Video, ThumbsUp, Camera, FileImage } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';

export function MediaTab() {
  const { themeStyle, getThemeClasses } = useTheme();
  const themeClasses = getThemeClasses();

  return (
    <div className={themeClasses.spacingClass}>
      {/* Header */}
      <div className="relative">
        <div className={`absolute inset-0 ${themeClasses.headerGradientClass}`}></div>
        <div className={`relative ${themeStyle === 'modern' ? 'p-8' : 'p-6'}`}>
          <h2 className="text-3xl font-bold text-gray-900 flex items-center mb-1">
            <ImageIcon className="w-8 h-8 mr-3 text-orange-600" />
          Media & Engagement Analysis
        </h2>
          <p className="text-lg text-gray-600">
          Visual content sharing patterns, media trends, and reaction engagement metrics
        </p>
        </div>
      </div>

      {/* Media Sharing Overview */}
      <div className={`grid grid-cols-1 lg:grid-cols-2 ${themeStyle === 'modern' ? 'gap-10' : 'gap-8'}`}>
        {/* Media Sharing Trends */}
        <div className={themeClasses.cardClass}>
          <h3 className={`text-lg font-semibold text-gray-900 ${themeStyle === 'modern' ? 'mb-6' : 'mb-4'} flex items-center`}>
            <Camera className="w-5 h-5 mr-2 text-blue-500" />
            Media Sharing Trends
          </h3>
          <p className={`text-sm text-gray-600 ${themeStyle === 'modern' ? 'mb-6' : 'mb-4'}`}>
            Photos and videos shared over time showing content patterns
          </p>
          <MediaChart />
        </div>

        {/* Media Breakdown by Sender */}
        <div className={themeClasses.cardClass}>
          <h3 className={`text-lg font-semibold text-gray-900 ${themeStyle === 'modern' ? 'mb-6' : 'mb-4'} flex items-center`}>
            <FileImage className="w-5 h-5 mr-2 text-green-500" />
            Media Breakdown by Sender
          </h3>
          <p className={`text-sm text-gray-600 ${themeStyle === 'modern' ? 'mb-6' : 'mb-4'}`}>
            Who shares the most visual content and what types
          </p>
          
          {/* Placeholder for media breakdown chart */}
          <MediaBySenderChart />
        </div>
      </div>

      {/* Reaction Analysis */}
      <div className={themeClasses.cardClass}>
        <h3 className={`text-lg font-semibold text-gray-900 ${themeStyle === 'modern' ? 'mb-6' : 'mb-4'} flex items-center`}>
          <ThumbsUp className="w-5 h-5 mr-2 text-yellow-500" />
          Reaction Metrics Dashboard
        </h3>
        <p className={`text-sm text-gray-600 ${themeStyle === 'modern' ? 'mb-6' : 'mb-4'}`}>
          Message reactions, engagement patterns, and most reacted content
        </p>
        
        {/* Reaction metrics grid */}
        <ReactionSummaryCards />

        {/* Reaction Metrics Chart */}
        <ReactionMetricsChart />
      </div>

      {/* Media Engagement Correlation - Full Width */}
      <div className={themeClasses.cardClass}>
        <MediaEngagementChart />
      </div>

      {/* Detailed Media Analysis */}
      <div className={`grid grid-cols-1 lg:grid-cols-2 ${themeStyle === 'modern' ? 'gap-10' : 'gap-8'}`}>
        {/* Attachment Type Analysis */}
        <div className={themeClasses.cardClass}>
          <h3 className={`text-lg font-semibold text-gray-900 ${themeStyle === 'modern' ? 'mb-6' : 'mb-4'} flex items-center`}>
            <Video className="w-5 h-5 mr-2 text-red-500" />
            Attachment Type Analysis
          </h3>
          <p className={`text-sm text-gray-600 ${themeStyle === 'modern' ? 'mb-6' : 'mb-4'}`}>
            Breakdown of different media types and formats shared
          </p>
          
          {/* Placeholder for attachment types */}
          <AttachmentTypeChart />
        </div>

        {/* Media Insights Panel (computed from real data) */}
        <MediaInsightsPanel />
      </div>
    </div>
  );
} 