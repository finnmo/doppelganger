'use client';

import React from 'react';
import { TrendingUp, MessageCircle, Heart, Image as ImageIcon, Clock, Users, Sparkles } from 'lucide-react';

export interface Tab {
  id: string;
  label: string;
  icon: React.ReactNode;
  description: string;
}

export const tabs: Tab[] = [
  {
    id: 'overview',
    label: 'Overview',
    icon: <TrendingUp className="w-4 h-4" />,
    description: 'High-level dashboard summary and KPIs'
  },
  {
    id: 'persona',
    label: 'Persona Chat',
    icon: <Sparkles className="w-4 h-4" />,
    description: 'Chat with an AI simulation of someone from your messages'
  },
  {
    id: 'messages',
    label: 'Messages & Content',
    icon: <MessageCircle className="w-4 h-4" />,
    description: 'Message content and pattern analysis'
  },
  {
    id: 'sentiment',
    label: 'Sentiment & Emotions',
    icon: <Heart className="w-4 h-4" />,
    description: 'Emotional analysis and mood tracking'
  },
  {
    id: 'media',
    label: 'Media & Reactions',
    icon: <ImageIcon className="w-4 h-4" />,
    description: 'Visual content and engagement analysis'
  },
  {
    id: 'activity',
    label: 'Activity Patterns',
    icon: <Clock className="w-4 h-4" />,
    description: 'Temporal analysis and communication rhythms'
  },
  {
    id: 'conversations',
    label: 'Conversations & Threads',
    icon: <Users className="w-4 h-4" />,
    description: 'Conversation structure and thread analysis'
  }
];

interface TabNavigationProps {
  activeTab: string;
  onTabChange: (tabId: string) => void;
}

export function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
  return (
    <div className="border-b border-gray-200 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <nav className="-mb-px flex space-x-8 overflow-x-auto" aria-label="Tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`
                group relative min-w-0 flex-1 overflow-hidden py-4 px-1 text-center text-sm font-medium hover:text-gray-700 focus:z-10 focus:outline-none whitespace-nowrap
                ${activeTab === tab.id
                  ? 'text-blue-600 border-b-2 border-blue-500'
                  : 'text-gray-500 border-b-2 border-transparent hover:border-gray-300'
                }
              `}
              aria-current={activeTab === tab.id ? 'page' : undefined}
              title={tab.description}
            >
              <div className="flex items-center justify-center space-x-2">
                {tab.icon}
                <span className="hidden sm:inline">{tab.label}</span>
                <span className="sm:hidden">{tab.label.split(' ')[0]}</span>
              </div>
            </button>
          ))}
        </nav>
      </div>
    </div>
  );
} 