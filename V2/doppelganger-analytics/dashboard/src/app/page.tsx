'use client';

import { useState } from 'react';
import { ConversationList } from '@/components/ConversationList';
import { ConversationDashboard } from '@/components/ConversationDashboard';
import { useConversationFilter } from '@/contexts/ConversationContext';

export default function Home() {
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null);
  const { setSelectedConversations } = useConversationFilter();

  const handleConversationSelect = (conversationId: string) => {
    // Filter the dashboard to this specific conversation
    setSelectedConversations([conversationId]);
    setSelectedConversationId(conversationId);
  };

  const handleBackToList = () => {
    setSelectedConversationId(null);
    setSelectedConversations([]); // Clear filter when going back to list
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
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <ConversationList onConversationSelect={handleConversationSelect} />
      </div>
    </div>
  );
}
