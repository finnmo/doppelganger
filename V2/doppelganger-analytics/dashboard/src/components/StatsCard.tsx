import React from 'react';
import { InfoTooltip } from './InfoTooltip';

interface StatsCardProps {
  title: string;
  value: string;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'purple' | 'orange';
  subtitle?: string;
  info?: {
    title: string;
    description: string;
    calculation?: string;
    example?: string;
  };
}

const colorStyles = {
  blue: {
    bg: 'bg-blue-50',
    icon: 'text-blue-600',
    text: 'text-blue-900'
  },
  green: {
    bg: 'bg-green-50',
    icon: 'text-green-600',
    text: 'text-green-900'
  },
  purple: {
    bg: 'bg-purple-50',
    icon: 'text-purple-600',
    text: 'text-purple-900'
  },
  orange: {
    bg: 'bg-orange-50',
    icon: 'text-orange-600',
    text: 'text-orange-900'
  }
};

export function StatsCard({ title, value, icon, color, subtitle, info }: StatsCardProps) {
  const styles = colorStyles[color];

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border">
      <div className="flex items-center">
        <div className={`p-3 rounded-lg ${styles.bg}`}>
          <div className={styles.icon}>
            {icon}
          </div>
        </div>
        <div className="ml-4 flex-1">
          <div className="flex items-center">
            <p className="text-sm font-medium text-gray-600">{title}</p>
            {info && (
              <InfoTooltip
                title={info.title}
                description={info.description}
                calculation={info.calculation}
                example={info.example}
              />
            )}
          </div>
          <p className={`text-2xl font-bold ${styles.text}`}>{value}</p>
          {subtitle && (
            <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
          )}
        </div>
      </div>
    </div>
  );
} 