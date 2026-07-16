'use client';

import React from 'react';
import { Camera, Heart, MessageSquare, Timer, Users } from 'lucide-react';
import { InfoTooltip } from '@/components/InfoTooltip';
import { useTheme } from '@/contexts/ThemeContext';
import type { ParticipantAnalyticsData } from './participants';

interface ParticipantAnalyticsProps {
  participants: ParticipantAnalyticsData;
  totalMessages: number;
}

function truncateName(name: string): string {
  return name.length > 12 ? `${name.substring(0, 12)}...` : name;
}

function formatResponseTime(ms: number): string {
  const seconds = ms / 1000;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h`;
}

/** The four participant cards: contributors, fast responders, emoji champions, media sharers. */
export function ParticipantAnalytics({ participants, totalMessages }: ParticipantAnalyticsProps) {
  const { themeStyle, getThemeClasses } = useTheme();
  const themeClasses = getThemeClasses();

  return (
    <div className="mb-10">
      {themeStyle === 'modern' ? (
        <div className={themeClasses.sectionHeaderClass('purple')}>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Users className="w-6 h-6 mr-3" />
              <h3 className="text-2xl font-bold mr-2">Participant Analytics</h3>
              <InfoTooltip
                title="Participant Analytics"
                description="Comprehensive analysis of how different participants engage with your conversations, showing message contributions, response patterns, and communication styles."
                calculation="Data is filtered to only show participants from your selected conversations and recalculates in real-time when filters change."
                example="If you filter to just work conversations, you'll only see colleagues' activity patterns, not personal chat participants."
                iconColor="white"
              />
            </div>
          </div>
          <p className="text-purple-100 mt-2">
            Comprehensive analysis of participant engagement and communication patterns
          </p>
        </div>
      ) : (
        <div className="flex items-center mb-6">
          <Users className="w-6 h-6 mr-3 text-blue-600" />
          <h3 className="text-2xl font-bold text-gray-900">Participant Analytics</h3>
          <InfoTooltip
            title="Participant Analytics"
            description="Comprehensive analysis of how different participants engage with your conversations, showing message contributions, response patterns, and communication styles."
            calculation="Data is filtered to only show participants from your selected conversations and recalculates in real-time when filters change."
            example="If you filter to just work conversations, you'll only see colleagues' activity patterns, not personal chat participants."
            iconColor="default"
          />
        </div>
      )}

      <div className={`grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 ${themeStyle === 'modern' ? 'gap-8' : 'gap-6'}`}>
        {/* Message Contributors */}
        <div className={themeClasses.sectionCardClass}>
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <MessageSquare className="w-5 h-5 mr-2 text-blue-500" />
            Top Contributors
            <div className="ml-2">
              <InfoTooltip
                title="Top Contributors"
                description="Participants who have sent the most messages across your selected conversations, ranked by total message count."
                calculation="Sum of all messages sent by each participant in selected conversations, then ranked in descending order"
                example="If Alice sent 1,500 messages, Bob sent 1,200, and Carol sent 800, they would appear in that order with percentages of total messages."
                iconColor="default"
              />
            </div>
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Participants who have sent the most messages in selected conversations
          </p>
          <div className="space-y-3">
            {participants.messageContributors.map((contributor, index) => (
              <div key={contributor.participant} className="flex items-center justify-between p-3 bg-blue-50 border-blue-100 rounded-md border">
                <div className="flex items-center">
                  <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">
                    {index + 1}
                  </div>
                  <span className="font-medium text-gray-800 truncate" title={contributor.participant}>
                    {truncateName(contributor.participant)}
                  </span>
                </div>
                <div className="text-right">
                  <div className="font-bold text-blue-600">{contributor.total_messages.toLocaleString()}</div>
                  <div className="text-xs text-gray-600">
                    {totalMessages > 0
                      ? `${Math.round((contributor.total_messages / totalMessages) * 100)}%`
                      : '—'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Fast Responders */}
        <div className={themeClasses.sectionCardClass}>
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Timer className="w-5 h-5 mr-2 text-orange-500" />
            Fast Responders
            <div className="ml-2">
              <InfoTooltip
                title="Fast Responders"
                description="Participants with the fastest average response times in conversations, showing who typically replies quickest to messages."
                calculation="Average response time calculated from time between receiving a message and sending the next message, converted from milliseconds to readable format"
                example="If someone typically responds within 2 minutes on average, they'll show as '2m avg response' and rank higher than someone who takes 10 minutes."
                iconColor="default"
              />
            </div>
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Participants with the lowest average response times
          </p>
          <div className="space-y-3">
            {participants.fastResponders.map((responder, index) => (
              <div key={responder.participant} className="flex items-center justify-between p-3 bg-orange-50 border-orange-100 rounded-md border">
                <div className="flex items-center">
                  <div className="w-8 h-8 bg-orange-600 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">
                    {index + 1}
                  </div>
                  <span className="font-medium text-gray-800 truncate" title={responder.participant}>
                    {truncateName(responder.participant)}
                  </span>
                </div>
                <div className="text-right">
                  <div className="font-bold text-orange-600">
                    {formatResponseTime(responder.avg_response_time)}
                  </div>
                  <div className="text-xs text-gray-600">avg response</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Top Emoji Users */}
        <div className={themeClasses.sectionCardClass}>
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Heart className="w-5 h-5 mr-2 text-purple-500" />
            Emoji Champions
            <div className="ml-2">
              <InfoTooltip
                title="Emoji Champions"
                description="Participants who use the most emojis in their messages, counted per conversation so filtering stays accurate."
                calculation="Sum of per-message emoji counts by sender within the selected conversation(s), from emojiMetrics data."
                example="If Alice used 120 emojis and Bob used 40 in this chat, Alice ranks first with 120."
                iconColor="default"
              />
            </div>
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Participants who use the most emojis in their messages
          </p>
          <div className="space-y-3">
            {participants.emojiUsers.length === 0 ? (
              <p className="text-sm text-gray-500 italic">
                Per-sender emoji counts are not available yet. Re-run “Generate analytics” to compute them.
              </p>
            ) : participants.emojiUsers.map((user, index) => (
              <div key={user.sender} className="flex items-center justify-between p-3 bg-purple-50 border-purple-100 rounded-md border">
                <div className="flex items-center">
                  <div className="w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">
                    {index + 1}
                  </div>
                  <span className="font-medium text-gray-800 truncate" title={user.sender}>
                    {truncateName(user.sender)}
                  </span>
                </div>
                <div className="text-right">
                  <div className="font-bold text-purple-600">{user.count.toLocaleString()}</div>
                  <div className="text-xs text-gray-600">emojis used</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Media Sharers */}
        <div className={themeClasses.sectionCardClass}>
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Camera className="w-5 h-5 mr-2 text-green-500" />
            Media Sharers
            <div className="ml-2">
              <InfoTooltip
                title="Media Sharers"
                description="Participants who shared the most photos, videos, and attachments in the selected conversation(s)."
                calculation="Sum of photo_count + video_count + attachment_count from mediaMetrics.sender_media_data filtered by conversation_id."
                example="If Tia shared 40 photos and 5 videos in this chat, she shows 45 total media."
                iconColor="default"
              />
            </div>
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Participants who have shared the most total media content
          </p>
          <div className="space-y-3">
            {participants.mediaSharers.length === 0 ? (
              <p className="text-sm text-gray-500 italic">No media shares in the selected conversation(s).</p>
            ) : participants.mediaSharers.map((sharer, index) => (
              <div key={sharer.sender} className="flex items-center justify-between p-3 bg-green-50 border-green-100 rounded-md border">
                <div className="flex items-center">
                  <div className="w-8 h-8 bg-green-600 text-white rounded-full flex items-center justify-center text-sm font-bold mr-3">
                    {index + 1}
                  </div>
                  <span className="font-medium text-gray-800 truncate" title={sharer.sender}>
                    {truncateName(sharer.sender)}
                  </span>
                </div>
                <div className="text-right">
                  <div className="font-bold text-green-600">{sharer.mediaShared.total.toLocaleString()}</div>
                  <div className="text-xs text-gray-600 flex space-x-1">
                    <span>📸{sharer.mediaShared.photos}</span>
                    <span>🎥{sharer.mediaShared.videos}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
