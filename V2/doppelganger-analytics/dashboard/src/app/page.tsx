'use client';

import { useState } from 'react';
import { ConversationList } from '@/components/ConversationList';
import { ConversationDashboard } from '@/components/ConversationDashboard';
import { useConversationFilter } from '@/contexts/ConversationContext';
import { PAGE_SHELL } from '@/lib/layout';

export default function Home() {
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null);
  const { setSelectedConversations } = useConversationFilter();

  const handleConversationSelect = (conversationId: string) => {
    setSelectedConversations([conversationId]);
    setSelectedConversationId(conversationId);
  };

  const handleBackToList = () => {
    setSelectedConversationId(null);
    setSelectedConversations([]);
  };

  if (selectedConversationId) {
    return (
      <ConversationDashboard
        conversationId={selectedConversationId}
        onBack={handleBackToList}
      />
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className={`${PAGE_SHELL} py-6 sm:py-8`}>
        <ConversationList onConversationSelect={handleConversationSelect} />
      </div>
    </div>
  );
}
