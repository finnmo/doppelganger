'use client';

import React from 'react';
import { ActivityHeatmap } from '@/components/ActivityHeatmap';
import { LatencyChart } from '@/components/LatencyChart';
import { DailyActivityPatterns } from '@/components/DailyActivityPatterns';
import { CommunicationFrequencyAnalysis } from '@/components/CommunicationFrequencyAnalysis';
import { CommunicationPatternsOverview } from '@/components/CommunicationPatternsOverview';
import { ActivityInsightsPanel } from '@/components/ActivityInsightsPanel';
import { Clock, Calendar, Zap } from 'lucide-react';

export function ActivityTab() {
  return (
    <div className="space-y-6 sm:space-y-8">
      {/* Header */}
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-to-r from-orange-50 via-yellow-50 to-red-50 rounded-2xl opacity-40"></div>
        <div className="relative p-4 sm:p-6">
          <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 flex flex-wrap items-center gap-2 mb-1">
            <Clock className="w-7 h-7 sm:w-8 sm:h-8 text-orange-600 shrink-0" />
          Activity Patterns & Communication Rhythms
        </h2>
          <p className="text-base sm:text-lg text-gray-600">
          Temporal analysis of communication patterns, activity rhythms, and response behaviors
        </p>
        </div>
      </div>

      {/* Primary Activity Analysis */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 sm:gap-6 lg:gap-8">
        {/* Activity Heatmap */}
        <div className="bg-white p-4 sm:p-6 rounded-lg shadow-sm border min-w-0">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex flex-wrap items-center gap-2">
            <Calendar className="w-5 h-5 text-blue-500 shrink-0" />
            Activity Heatmap by Hour
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            When people are most active during the day across different senders
          </p>
          <ActivityHeatmap />
        </div>

        {/* Response Time Distribution */}
        <div className="bg-white p-4 sm:p-6 rounded-lg shadow-sm border min-w-0">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex flex-wrap items-center gap-2">
            <Zap className="w-5 h-5 text-green-500 shrink-0" />
            Response Time Distribution
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            How quickly people respond to messages across different time buckets
          </p>
          <LatencyChart />
        </div>
      </div>

      {/* Daily Activity Patterns */}
      <DailyActivityPatterns />

      {/* Communication Patterns Overview */}
      <CommunicationPatternsOverview />

      {/* Communication Frequency Analysis */}
      <CommunicationFrequencyAnalysis />

      {/* Activity Insights Panel (computed from real data) */}
      <ActivityInsightsPanel />
    </div>
  );
} 