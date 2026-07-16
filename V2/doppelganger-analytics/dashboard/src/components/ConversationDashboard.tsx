'use client';

import React, { useState } from 'react';
import { ArrowLeft, Users } from 'lucide-react';
import { ThemeSelector } from '@/components/ThemeSelector';
import { ApiKeySettingsButton } from '@/components/ApiKeySettings';
import { TabNavigation } from '@/components/TabNavigation';
import { OverviewTab } from '@/components/tabs/OverviewTab';
import { MessagesTab } from '@/components/tabs/MessagesTab';
import { SentimentTab } from '@/components/tabs/SentimentTab';
import { MediaTab } from '@/components/tabs/MediaTab';
import { ActivityTab } from '@/components/tabs/ActivityTab';
import { ConversationsTab } from '@/components/tabs/ConversationsTab';
import { PersonaTab } from '@/components/tabs/PersonaTab';
import { PlatformBadge } from '@/components/PlatformBadge';
import { useConversationFilter } from '@/contexts/ConversationContext';
import { parseConversationId } from '@/lib/platforms';

interface ConversationDashboardProps {
  conversationId: string;
  onBack: () => void;
}

export function ConversationDashboard({ conversationId, onBack }: ConversationDashboardProps) {
  const [activeTab, setActiveTab] = useState('overview');
  const { conversations } = useConversationFilter();
  const conversation = conversations.find((c) => c.conversation_id === conversationId);

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewTab />;
      case 'persona':
        return <PersonaTab />;
      case 'messages':
        return <MessagesTab />;
      case 'sentiment':
        return <SentimentTab />;
      case 'media':
        return <MediaTab />;
      case 'activity':
        return <ActivityTab />;
      case 'conversations':
        return <ConversationsTab />;
      default:
        return <OverviewTab />;
    }
  };

  const getConversationDisplayName = () => {
    if (conversation?.participants?.length) {
      if (conversation.participants.length === 1) return conversation.participants[0];
      if (conversation.participants.length === 2) {
        return conversation.participants.join(' & ');
      }
      return `${conversation.participants[0]} + ${conversation.participants.length - 1} others`;
    }

    const raw = parseConversationId(conversationId).rawId;
    if (raw === '.') return 'Mixed Conversations';
    const parts = raw.split('_');
    if (parts.length > 0 && parts[0]) {
      const username = parts[0].replace(/([A-Z])/g, ' $1').trim();
      return username.charAt(0).toUpperCase() + username.slice(1);
    }
    return raw;
  };

  const source = conversation?.source ?? parseConversationId(conversationId).source;

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center space-x-4 min-w-0">
              <button
                type="button"
                onClick={onBack}
                className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 shrink-0"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Conversations
              </button>

              <div className="border-l border-gray-300 pl-4 min-w-0">
                <div className="flex items-center space-x-2">
                  <Users className="w-5 h-5 text-gray-500 shrink-0" />
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <h1 className="text-xl font-bold text-gray-900 truncate">
                        {getConversationDisplayName()}
                      </h1>
                      {source && <PlatformBadge source={source} size="md" />}
                    </div>
                    <p className="text-sm text-gray-600">
                      {conversation
                        ? `${conversation.total_messages.toLocaleString()} messages · ${source ? conversation.source_label : 'Analytics'}`
                        : 'Analytics for this conversation'}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-2 shrink-0">
              <ApiKeySettingsButton />
              <ThemeSelector />
            </div>
          </div>
        </div>
      </div>

      <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderTabContent()}
      </div>
    </div>
  );
}
