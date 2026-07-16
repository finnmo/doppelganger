'use client';

import React from 'react';
import { Activity, TrendingUp } from 'lucide-react';
import { InfoTooltip } from '@/components/InfoTooltip';
import { MessageTrendChart } from '@/components/MessageTrendChart';
import { PeakActivityChart } from '@/components/PeakActivityChart';
import { useTheme } from '@/contexts/ThemeContext';

/** The activity section: message trend and peak activity charts. */
export function ActivityAnalytics() {
  const { themeStyle, getThemeClasses } = useTheme();
  const themeClasses = getThemeClasses();

  return (
    <div className="mb-10">
      {themeStyle === 'modern' ? (
        <div className={themeClasses.sectionHeaderClass('blue')}>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Activity className="w-6 h-6 mr-3" />
              <h3 className="text-2xl font-bold mr-2">Activity Analytics</h3>
              <InfoTooltip
                title="Activity Analytics"
                description="Visual analysis of your communication patterns over time, showing message volume trends and peak activity periods to understand when you're most active."
                calculation="Messages are grouped by time periods and visualized to show activity concentration patterns and trends"
                example="You might see spikes during work hours, drops on weekends, or seasonal patterns revealing your natural communication rhythm."
                iconColor="white"
              />
            </div>
          </div>
          <p className="text-blue-100 mt-2">
            Visual analysis of communication patterns and activity trends over time
          </p>
        </div>
      ) : (
        <div className="flex items-center mb-6">
          <Activity className="w-6 h-6 mr-3 text-blue-600" />
          <h3 className="text-2xl font-bold text-gray-900">Activity Analytics</h3>
          <div className="ml-2">
            <InfoTooltip
              title="Activity Analytics"
              description="Visual analysis of your communication patterns over time, showing message volume trends and peak activity periods to understand when you're most active."
              calculation="Messages are grouped by time periods and visualized to show activity concentration patterns and trends"
              example="You might see spikes during work hours, drops on weekends, or seasonal patterns revealing your natural communication rhythm."
              iconColor="default"
            />
          </div>
        </div>
      )}

      <div className={`grid grid-cols-1 lg:grid-cols-2 ${themeStyle === 'modern' ? 'gap-10' : 'gap-8'}`}>
        <div className={themeClasses.sectionCardClass}>
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-blue-500" />
            Message Activity Over Time
            <div className="ml-2">
              <InfoTooltip
                title="Message Activity Over Time"
                description="Line chart showing how message volume changes over time across your selected conversations, helping identify busy periods and communication trends."
                calculation="Messages are grouped by time period and plotted as a time series, with peaks indicating high activity periods"
                example="You might see spikes during work hours, drops on weekends, or seasonal patterns like increased activity during holidays."
                iconColor="default"
              />
            </div>
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Message volume changes over time showing activity patterns
          </p>
          <div className="h-80">
            <MessageTrendChart />
          </div>
        </div>

        <div className={themeClasses.sectionCardClass}>
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-green-500" />
            Peak Activity Patterns
            <div className="ml-2">
              <InfoTooltip
                title="Peak Activity Patterns"
                description="Visual analysis showing when communication is most active during different hours and days, revealing your natural communication rhythm."
                calculation="Messages are analyzed by hour and day of week to identify peak activity periods and patterns"
                example="You might discover you're most active Tuesday-Thursday from 2-4 PM, or that weekends show completely different patterns."
                iconColor="default"
              />
            </div>
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            When communication is most active during different hours and days
          </p>
          <div className="h-80">
            <PeakActivityChart />
          </div>
        </div>
      </div>
    </div>
  );
}
