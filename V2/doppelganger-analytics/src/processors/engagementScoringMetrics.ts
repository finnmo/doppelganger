import { getDb, closeDb } from '../db/client.js';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

interface MessageRow {
  id: number;
  conversation_id: string;
  sender: string;
  timestamp_ms: number;
  content: string | null;
}

interface EngagementScore {
  participant: string;
  overall_score: number;
  message_frequency: number;
  response_speed: number;
  conversation_initiation: number;
  message_length: number;
  consistency: number;
  social_connectivity: number;
  conversations_count: number;
  total_messages: number;
  avg_response_time: number;
  engagement_tier: 'high' | 'medium' | 'low';
}

interface ConversationEngagement {
  conversation_id: string;
  participants: EngagementScore[];
  avg_engagement: number;
  engagement_distribution: {
    high: number;
    medium: number;
    low: number;
  };
}

interface EngagementMetrics {
  summary: {
    total_participants: number;
    avg_engagement_score: number;
    high_engagement_participants: number;
    most_engaged_participant: string;
    least_engaged_participant: string;
    engagement_variance: number;
  };
  participant_scores: EngagementScore[];
  conversation_engagement: ConversationEngagement[];
  engagement_trends: {
    participant: string;
    scores_over_time: { period: string; score: number }[];
  }[];
}

interface ParticipantStats {
  participant: string;
  conversations: Set<string>;
  total_messages: number;
  response_times: number[];
  message_lengths: number[];
  conversation_starts: number;
  daily_activity: Map<string, number>;
  first_message_time: number;
  last_message_time: number;
}

function calculateEngagementScore(stats: ParticipantStats, globalStats: {
  avgMessagesPerConversation: number;
  avgResponseTime: number;
  avgMessageLength: number;
  totalConversations: number;
}): EngagementScore {
  const conversationsCount = stats.conversations.size;
  const avgMessagesPerConversation = stats.total_messages / conversationsCount;
  const avgResponseTime = stats.response_times.length > 0 
    ? stats.response_times.reduce((sum, time) => sum + time, 0) / stats.response_times.length 
    : 0;
  const avgMessageLength = stats.message_lengths.length > 0
    ? stats.message_lengths.reduce((sum, len) => sum + len, 0) / stats.message_lengths.length
    : 0;

  // Calculate individual metrics (0-100 scale)
  
  // Message Frequency Score (based on messages per conversation)
  const messageFrequencyScore = Math.min(100, (avgMessagesPerConversation / globalStats.avgMessagesPerConversation) * 50);
  
  // Response Speed Score (inverse of response time, faster = higher score)
  const responseSpeedScore = avgResponseTime > 0 
    ? Math.min(100, (globalStats.avgResponseTime / avgResponseTime) * 50)
    : 50; // Neutral score if no response data
  
  // Conversation Initiation Score (based on starting conversations)
  const initiationRate = stats.conversation_starts / conversationsCount;
  const conversationInitiationScore = Math.min(100, initiationRate * 100);
  
  // Message Length Score (based on average message length)
  const messageLengthScore = avgMessageLength > 0 
    ? Math.min(100, (avgMessageLength / globalStats.avgMessageLength) * 50)
    : 25; // Low score for very short messages
  
  // Consistency Score (based on activity spread over time)
  const activityDays = stats.daily_activity.size;
  const totalDays = Math.max(1, Math.ceil((stats.last_message_time - stats.first_message_time) / (24 * 60 * 60 * 1000)));
  const consistencyScore = Math.min(100, (activityDays / totalDays) * 100);
  
  // Social Connectivity Score (based on number of conversations participated in)
  const connectivityRate = conversationsCount / globalStats.totalConversations;
  const socialConnectivityScore = Math.min(100, connectivityRate * 200); // Boost for high participation
  
  // Calculate weighted overall score
  const weights = {
    messageFrequency: 0.25,
    responseSpeed: 0.20,
    conversationInitiation: 0.15,
    messageLength: 0.10,
    consistency: 0.15,
    socialConnectivity: 0.15
  };
  
  const overallScore = 
    messageFrequencyScore * weights.messageFrequency +
    responseSpeedScore * weights.responseSpeed +
    conversationInitiationScore * weights.conversationInitiation +
    messageLengthScore * weights.messageLength +
    consistencyScore * weights.consistency +
    socialConnectivityScore * weights.socialConnectivity;
  
  // Determine engagement tier
  let engagementTier: 'high' | 'medium' | 'low';
  if (overallScore >= 70) {
    engagementTier = 'high';
  } else if (overallScore >= 40) {
    engagementTier = 'medium';
  } else {
    engagementTier = 'low';
  }
  
  return {
    participant: stats.participant,
    overall_score: Math.round(overallScore * 10) / 10,
    message_frequency: Math.round(messageFrequencyScore * 10) / 10,
    response_speed: Math.round(responseSpeedScore * 10) / 10,
    conversation_initiation: Math.round(conversationInitiationScore * 10) / 10,
    message_length: Math.round(messageLengthScore * 10) / 10,
    consistency: Math.round(consistencyScore * 10) / 10,
    social_connectivity: Math.round(socialConnectivityScore * 10) / 10,
    conversations_count: conversationsCount,
    total_messages: stats.total_messages,
    avg_response_time: Math.round(avgResponseTime),
    engagement_tier: engagementTier
  };
}

export async function computeEngagementScoringMetrics(): Promise<void> {
  progressReporter.start('Computing engagement scoring metrics...');
  
  const db = await getDb();
  
  try {
    // Get all conversations
    const conversations = db.prepare(`
      SELECT DISTINCT conversation_id 
      FROM messages 
      ORDER BY conversation_id
    `).all() as { conversation_id: string }[];
    
    progressReporter.update(`Analyzing engagement for ${conversations.length} conversations...`);
    
    // Per-participant reply latencies from the canonical response_times table
    // (the responder is the sender of the to-message). Replaces a bespoke
    // per-conversation response-time loop.
    const responderLatencies = new Map<string, number[]>();
    for (const row of db.prepare(`
      SELECT m.sender AS responder, rt.latency_ms AS latency
      FROM response_times rt
      JOIN messages m ON m.id = rt.to_message_id
    `).all() as Array<{ responder: string; latency: number }>) {
      const list = responderLatencies.get(row.responder);
      if (list) {
        list.push(row.latency);
      } else {
        responderLatencies.set(row.responder, [row.latency]);
      }
    }

    const participantStatsMap = new Map<string, ParticipantStats>();
    const conversationEngagement: ConversationEngagement[] = [];

    // Process each conversation
    for (const conv of conversations) {
      const messages = db.prepare(`
        SELECT id, conversation_id, sender, timestamp_ms, content
        FROM messages 
        WHERE conversation_id = ?
        ORDER BY timestamp_ms ASC
      `).all(conv.conversation_id) as MessageRow[];
      
      if (messages.length < 2) continue; // Skip conversations with too few messages
      
      const participants = [...new Set(messages.map(m => m.sender))];
      if (participants.length < 2) continue; // Skip single-participant conversations
      
      // Track conversation starter
      const conversationStarter = messages[0].sender;
      
      // Process messages for each participant
      const participantMessages = new Map<string, MessageRow[]>();
      participants.forEach(participant => {
        participantMessages.set(participant, messages.filter(m => m.sender === participant));
      });
      
      // Update participant statistics
      participants.forEach(participant => {
        if (!participantStatsMap.has(participant)) {
          participantStatsMap.set(participant, {
            participant,
            conversations: new Set(),
            total_messages: 0,
            response_times: [],
            message_lengths: [],
            conversation_starts: 0,
            daily_activity: new Map(),
            first_message_time: Infinity,
            last_message_time: 0
          });
        }
        
        const stats = participantStatsMap.get(participant)!;
        const participantMsgs = participantMessages.get(participant)!;
        
        stats.conversations.add(conv.conversation_id);
        stats.total_messages += participantMsgs.length;
        
        if (participant === conversationStarter) {
          stats.conversation_starts++;
        }

        // Calculate message lengths and daily activity
        participantMsgs.forEach(msg => {
          const messageLength = msg.content ? msg.content.length : 0;
          stats.message_lengths.push(messageLength);
          
          const day = new Date(msg.timestamp_ms).toDateString();
          stats.daily_activity.set(day, (stats.daily_activity.get(day) || 0) + 1);
          
          stats.first_message_time = Math.min(stats.first_message_time, msg.timestamp_ms);
          stats.last_message_time = Math.max(stats.last_message_time, msg.timestamp_ms);
        });
      });
    }
    
    // Attach each participant's canonical reply latencies
    for (const [participant, stats] of participantStatsMap.entries()) {
      stats.response_times = responderLatencies.get(participant) || [];
    }

    // Calculate global statistics for normalization
    const allStats = Array.from(participantStatsMap.values());
    const globalStats = {
      avgMessagesPerConversation: allStats.reduce((sum, stats) => 
        sum + (stats.total_messages / stats.conversations.size), 0) / allStats.length,
      avgResponseTime: allStats.reduce((sum, stats) => {
        const avgResponseTime = stats.response_times.length > 0 
          ? stats.response_times.reduce((s, t) => s + t, 0) / stats.response_times.length 
          : 0;
        return sum + avgResponseTime;
      }, 0) / allStats.length,
      avgMessageLength: allStats.reduce((sum, stats) => {
        const avgLength = stats.message_lengths.length > 0
          ? stats.message_lengths.reduce((s, l) => s + l, 0) / stats.message_lengths.length
          : 0;
        return sum + avgLength;
      }, 0) / allStats.length,
      totalConversations: conversations.length
    };
    
    // Calculate engagement scores for all participants
    const participantScores = allStats.map(stats => 
      calculateEngagementScore(stats, globalStats)
    ).sort((a, b) => b.overall_score - a.overall_score);
    
    // Calculate conversation-level engagement with PER-CONVERSATION message
    // counts (not the participant's global totals — those blow up filtered %).
    for (const conv of conversations) {
      const convMessages = db.prepare(`
        SELECT sender, COUNT(*) as count
        FROM messages WHERE conversation_id = ?
        GROUP BY sender
      `).all(conv.conversation_id) as { sender: string; count: number }[];
      
      if (convMessages.length < 2) continue;
      
      const conversationParticipants = convMessages
        .map(({ sender, count }) => {
          const globalScore = participantScores.find(p => p.participant === sender);
          if (!globalScore) return null;
          return {
            ...globalScore,
            total_messages: count
          };
        })
        .filter(Boolean) as EngagementScore[];
      
      if (conversationParticipants.length === 0) continue;
      
      const avgEngagement = conversationParticipants.reduce((sum, p) => sum + p.overall_score, 0) / conversationParticipants.length;
      
      const engagementDist = {
        high: conversationParticipants.filter(p => p.engagement_tier === 'high').length,
        medium: conversationParticipants.filter(p => p.engagement_tier === 'medium').length,
        low: conversationParticipants.filter(p => p.engagement_tier === 'low').length
      };
      
      conversationEngagement.push({
        conversation_id: conv.conversation_id,
        participants: conversationParticipants,
        avg_engagement: Math.round(avgEngagement * 10) / 10,
        engagement_distribution: engagementDist
      });
    }
    
    // Calculate summary statistics
    const highEngagementCount = participantScores.filter(p => p.engagement_tier === 'high').length;
    const avgEngagementScore = participantScores.length > 0 
      ? participantScores.reduce((sum, p) => sum + p.overall_score, 0) / participantScores.length 
      : 0;
    
    const engagementVariance = participantScores.length > 0
      ? participantScores.reduce((sum, p) => sum + Math.pow(p.overall_score - avgEngagementScore, 2), 0) / participantScores.length
      : 0;
    
    // Generate engagement trends (simplified - could be enhanced with temporal analysis)
    const engagementTrends = participantScores.slice(0, 10).map(participant => ({
      participant: participant.participant,
      scores_over_time: [
        { period: 'Overall', score: participant.overall_score }
      ]
    }));
    
    const metrics: EngagementMetrics = {
      summary: {
        total_participants: participantScores.length,
        avg_engagement_score: Math.round(avgEngagementScore * 10) / 10,
        high_engagement_participants: highEngagementCount,
        most_engaged_participant: participantScores[0]?.participant || '',
        least_engaged_participant: participantScores[participantScores.length - 1]?.participant || '',
        engagement_variance: Math.round(engagementVariance * 10) / 10
      },
      participant_scores: participantScores,
      conversation_engagement: conversationEngagement,
      engagement_trends: engagementTrends
    };
    
    // Export to JSON
    writeDashData('engagementScoring.json', metrics);
    
    progressReporter.success('Engagement scoring metrics computed and exported.');
    progressReporter.update(`Analyzed ${participantScores.length} participants`);
    progressReporter.update(`Found ${highEngagementCount} high-engagement participants`);
    progressReporter.update(`Average engagement score: ${metrics.summary.avg_engagement_score}`);
    progressReporter.update(`Most engaged: ${metrics.summary.most_engaged_participant}`);
    
  } catch (error) {
    console.error('Error computing engagement scoring metrics:', error);
    throw error;
  } finally {
    await closeDb(db);
  }
} 