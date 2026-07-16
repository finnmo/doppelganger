'use client';

import React, { useState } from 'react';
import { ArrowLeft, Users } from 'lucide-react';
import { ThemeSelector } from '@/components/ThemeSelector';
import { ApiKeySettingsButton } from '@/components/ApiKeySettings';
import { PrivacySettingsButton } from '@/components/PrivacySettings';
import { DataFreshnessBanner } from '@/components/DataFreshnessBanner';
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
import { PAGE_SHELL, TOOLBAR_ROW } from '@/lib/layout';

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
        <div className={PAGE_SHELL}>
          <div className="flex flex-col gap-3 py-3 sm:py-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex min-w-0 items-start gap-3 sm:items-center sm:gap-4">
              <button
                type="button"
                onClick={onBack}
                className="inline-flex shrink-0 items-center rounded-md border border-gray-300 bg-white px-2.5 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 sm:px-3"
              >
                <ArrowLeft className="h-4 w-4 sm:mr-2" />
                <span className="hidden sm:inline">Back to Conversations</span>
                <span className="sm:hidden">Back</span>
              </button>

              <div className="min-w-0 border-l border-gray-200 pl-3 sm:pl-4">
                <div className="flex min-w-0 items-start gap-2 sm:items-center">
                  <Users className="mt-0.5 h-5 w-5 shrink-0 text-gray-500 sm:mt-0" />
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h1 className="truncate text-lg font-bold text-gray-900 sm:text-xl">
                        {getConversationDisplayName()}
                      </h1>
                      {source && <PlatformBadge source={source} size="md" />}
                    </div>
                    <p className="text-xs text-gray-600 sm:text-sm">
                      {conversation
                        ? `${conversation.total_messages.toLocaleString()} messages · ${source ? conversation.source_label : 'Analytics'}`
                        : 'Analytics for this conversation'}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className={`${TOOLBAR_ROW} shrink-0`}>
              <PrivacySettingsButton />
              <ApiKeySettingsButton />
              <ThemeSelector />
            </div>
          </div>
        </div>
      </div>

      <div className={`${PAGE_SHELL} pt-3 sm:pt-4`}>
        <DataFreshnessBanner />
      </div>

      <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

      <div className={`${PAGE_SHELL} py-5 sm:py-8`}>
        {renderTabContent()}
      </div>
    </div>
  );
}
