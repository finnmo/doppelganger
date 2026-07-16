'use client';

import React from 'react';
import { TrendingUp, MessageCircle, Heart, Image as ImageIcon, Clock, Users, Sparkles } from 'lucide-react';
import { PAGE_SHELL } from '@/lib/layout';

export interface Tab {
  id: string;
  label: string;
  shortLabel: string;
  icon: React.ReactNode;
  description: string;
}

export const tabs: Tab[] = [
  {
    id: 'overview',
    label: 'Overview',
    shortLabel: 'Overview',
    icon: <TrendingUp className="w-4 h-4" />,
    description: 'High-level dashboard summary and KPIs',
  },
  {
    id: 'persona',
    label: 'Persona Chat',
    shortLabel: 'Persona',
    icon: <Sparkles className="w-4 h-4" />,
    description: 'Chat with an AI simulation of someone from your messages',
  },
  {
    id: 'messages',
    label: 'Messages & Content',
    shortLabel: 'Messages',
    icon: <MessageCircle className="w-4 h-4" />,
    description: 'Message content and pattern analysis',
  },
  {
    id: 'sentiment',
    label: 'Sentiment & Emotions',
    shortLabel: 'Sentiment',
    icon: <Heart className="w-4 h-4" />,
    description: 'Emotional analysis and mood tracking',
  },
  {
    id: 'media',
    label: 'Media & Reactions',
    shortLabel: 'Media',
    icon: <ImageIcon className="w-4 h-4" />,
    description: 'Visual content and engagement analysis',
  },
  {
    id: 'activity',
    label: 'Activity Patterns',
    shortLabel: 'Activity',
    icon: <Clock className="w-4 h-4" />,
    description: 'Temporal analysis and communication rhythms',
  },
  {
    id: 'conversations',
    label: 'Conversations & Threads',
    shortLabel: 'Threads',
    icon: <Users className="w-4 h-4" />,
    description: 'Conversation structure and thread analysis',
  },
];

interface TabNavigationProps {
  activeTab: string;
  onTabChange: (tabId: string) => void;
}

export function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
  return (
    <div className="border-b border-gray-200 bg-white">
      <div className={PAGE_SHELL}>
        <nav
          className="-mb-px flex gap-1 overflow-x-auto overscroll-x-contain pb-px [-ms-overflow-style:none] [scrollbar-width:none] sm:gap-2 [&::-webkit-scrollbar]:hidden"
          aria-label="Tabs"
        >
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => onTabChange(tab.id)}
              className={`
                group relative shrink-0 whitespace-nowrap px-3 py-3 text-center text-sm font-medium
                focus:z-10 focus:outline-none sm:px-4
                ${
                  activeTab === tab.id
                    ? 'border-b-2 border-blue-500 text-blue-600'
                    : 'border-b-2 border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                }
              `}
              aria-current={activeTab === tab.id ? 'page' : undefined}
              title={tab.description}
            >
              <div className="flex items-center justify-center gap-1.5 sm:gap-2">
                {tab.icon}
                <span className="hidden md:inline">{tab.label}</span>
                <span className="md:hidden">{tab.shortLabel}</span>
              </div>
            </button>
          ))}
        </nav>
      </div>
    </div>
  );
}
