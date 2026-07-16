'use client';

import React from 'react';
import { Camera, FileImage, ThumbsUp } from 'lucide-react';
import { useDashData } from '@/hooks/useDashData';
import { useConversationFilter } from '@/contexts/ConversationContext';

interface MediaMetrics {
  conversation_metrics: Array<{
    conversation_id: string;
    photo_count: number;
    video_count: number;
    attachment_count: number;
    total_messages: number;
  }>;
  sender_media_data: Array<{
    conversation_id: string;
    sender: string;
    total_media: number;
  }>;
}

interface ReactionMetrics {
  reactionsByConversation: Array<{
    conversation_id: string;
    count: number;
    top_emoji: string;
  }>;
}

function pct(part: number, whole: number): number {
  return whole > 0 ? Math.round((part / whole) * 100) : 0;
}

export function MediaInsightsPanel() {
  const { data: media } = useDashData<MediaMetrics>('mediaMetrics.json');
  const { data: reactions } = useDashData<ReactionMetrics>('reactionMetrics.json');
  const { selectedConversations } = useConversationFilter();

  const selected = new Set(selectedConversations);

  // Media totals for the selected conversation(s)
  let photos = 0, videos = 0, attachments = 0, totalMessages = 0;
  for (const conv of media?.conversation_metrics || []) {
    if (!selected.has(conv.conversation_id)) continue;
    photos += conv.photo_count;
    videos += conv.video_count;
    attachments += conv.attachment_count;
    totalMessages += conv.total_messages;
  }
  const totalMedia = photos + videos + attachments;

  // Top media sharer within the selection
  const sharerTotals = new Map<string, number>();
  for (const row of media?.sender_media_data || []) {
    if (!selected.has(row.conversation_id)) continue;
    sharerTotals.set(row.sender, (sharerTotals.get(row.sender) || 0) + row.total_media);
  }
  const topSharer = [...sharerTotals.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || '—';

  const mediaShareRate = totalMessages > 0 ? (totalMedia / totalMessages) * 100 : 0;

  const preferredFormat = totalMedia === 0
    ? '—'
    : photos >= videos && photos >= attachments
      ? 'Photos'
      : videos >= attachments
        ? 'Videos'
        : 'Files';

  // Real per-conversation reaction totals for the selection
  const convReactions = (reactions?.reactionsByConversation || []).filter(r => selected.has(r.conversation_id));
  const totalReactions = convReactions.reduce((sum, r) => sum + r.count, 0);
  const topEmoji = convReactions.slice().sort((a, b) => b.count - a.count)[0]?.top_emoji || '—';
  const reactionRate = totalMessages > 0 ? (totalReactions / totalMessages) * 100 : 0;

  const ready = !!media;

  return (
    <div className="space-y-6">
      {/* Media Share Rate */}
      <div className="bg-gradient-to-br from-orange-400 to-orange-500 rounded-2xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow">
        <div className="flex items-center mb-4">
          <Camera className="w-6 h-6 mr-2" />
          <h3 className="text-xl font-bold">Media Share Rate</h3>
        </div>
        <div className="text-4xl font-bold mb-4">
          {ready ? `${mediaShareRate.toFixed(1)}%` : '—'}
        </div>
        <div className="space-y-3 mb-6">
          <div className="flex justify-between items-center">
            <span className="text-orange-100">🖼️ Media items</span>
            <span className="font-semibold">{ready ? totalMedia.toLocaleString() : '—'}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-orange-100">🏆 Top sharer</span>
            <span className="font-semibold truncate ml-2">{topSharer}</span>
          </div>
        </div>
        <div className="pt-4 border-t border-orange-300 border-opacity-40">
          <div className="text-orange-100 text-sm">Media items as a share of all messages</div>
        </div>
      </div>

      {/* Preferred Format */}
      <div className="bg-gradient-to-br from-blue-400 to-blue-500 rounded-2xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow">
        <div className="flex items-center mb-4">
          <FileImage className="w-6 h-6 mr-2" />
          <h3 className="text-xl font-bold">Preferred Format</h3>
        </div>
        <div className="text-4xl font-bold mb-4">{preferredFormat}</div>
        <div className="space-y-3 mb-6">
          <div className="flex justify-between items-center">
            <span className="text-blue-100">📸 Images</span>
            <span className="font-semibold">{ready ? `${pct(photos, totalMedia)}%` : '—'}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-blue-100">🎥 Videos</span>
            <span className="font-semibold">{ready ? `${pct(videos, totalMedia)}%` : '—'}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-blue-100">📎 Other</span>
            <span className="font-semibold">{ready ? `${pct(attachments, totalMedia)}%` : '—'}</span>
          </div>
        </div>
        <div className="pt-4 border-t border-blue-300 border-opacity-40">
          <div className="text-blue-100 text-sm">Breakdown of shared media types</div>
        </div>
      </div>

      {/* Reaction Rate */}
      <div className="bg-gradient-to-br from-green-400 to-green-500 rounded-2xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow">
        <div className="flex items-center mb-4">
          <ThumbsUp className="w-6 h-6 mr-2" />
          <h3 className="text-xl font-bold">Reaction Rate</h3>
        </div>
        <div className="text-4xl font-bold mb-4">
          {reactions ? `${reactionRate.toFixed(1)}%` : '—'}
        </div>
        <div className="space-y-3 mb-6">
          <div className="flex justify-between items-center">
            <span className="text-green-100">💬 Total reactions</span>
            <span className="font-semibold">{reactions ? totalReactions.toLocaleString() : '—'}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-green-100">⭐ Top reaction</span>
            <span className="font-semibold">{topEmoji}</span>
          </div>
        </div>
        <div className="pt-4 border-t border-green-300 border-opacity-40">
          <div className="text-green-100 text-sm">Reactions per 100 messages</div>
        </div>
      </div>
    </div>
  );
}
