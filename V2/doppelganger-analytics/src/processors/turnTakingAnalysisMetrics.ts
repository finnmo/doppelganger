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

interface TurnData {
  conversation_id: string;
  participant: string;
  turn_count: number;
  avg_turn_length: number;
  max_turn_length: number;
  turn_percentage: number;
  avg_response_time: number;
  interruption_rate: number;
}

interface TurnPattern {
  pattern_type: 'balanced' | 'dominant' | 'responsive' | 'sporadic';
  participants: string[];
  turn_ratio: number;
  conversation_health: number;
  description: string;
}

interface TurnSequence {
  participant: string;
  turn_number: number;
  message_count: number;
  duration_ms: number;
  start_time: number;
  end_time: number;
}

// turn_sequence is computed for the pattern analysis but deliberately not
// exported: no dashboard component reads it and it dominated the payload size
interface ConversationPattern {
  conversation_id: string;
  pattern: TurnPattern;
  participants: TurnData[];
}

interface ParticipantStats {
  participant: string;
  conversations: number;
  avg_turn_count: number;
  avg_turn_length: number;
  dominance_score: number;
  responsiveness_score: number;
}

interface TurnTakingMetrics {
  summary: {
    total_conversations: number;
    avg_participants: number;
    balanced_conversations: number;
    dominant_speaker_conversations: number;
    avg_turn_length: number;
    avg_response_time: number;
  };
  conversation_patterns: ConversationPattern[];
  participant_stats: ParticipantStats[];
}

function analyzeTurnSequence(messages: MessageRow[]): TurnSequence[] {
  if (messages.length === 0) return [];
  
  const turns: TurnSequence[] = [];
  
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    const lastTurn = turns[turns.length - 1];
    
    if (!lastTurn || lastTurn.participant !== message.sender) {
      // Start new turn
      const newTurn: TurnSequence = {
        participant: message.sender,
        turn_number: turns.length + 1,
        message_count: 1,
        duration_ms: 0,
        start_time: message.timestamp_ms,
        end_time: message.timestamp_ms
      };
      turns.push(newTurn);
    } else {
      // Continue current turn
      lastTurn.message_count++;
      lastTurn.end_time = message.timestamp_ms;
    }
  }
  
  // Calculate durations for all turns
  turns.forEach(turn => {
    turn.duration_ms = turn.end_time - turn.start_time;
  });
  
  return turns;
}

function calculateTurnPattern(turns: TurnSequence[], participants: string[]): TurnPattern {
  if (turns.length === 0 || participants.length === 0) {
    return {
      pattern_type: 'sporadic',
      participants,
      turn_ratio: 0,
      conversation_health: 0,
      description: 'No meaningful turn pattern detected'
    };
  }
  
  // Calculate turn distribution
  const turnCounts = participants.reduce((acc, p) => {
    acc[p] = turns.filter(t => t.participant === p).length;
    return acc;
  }, {} as Record<string, number>);
  
  const totalTurns = turns.length;
  const turnPercentages = participants.map(p => (turnCounts[p] || 0) / totalTurns);
  
  // Calculate dominance metrics. Dominance is measured by message share, not
  // turn share: adjacent turns always alternate participants, so one
  // participant's turn share is capped at 2/3 and could never exceed a
  // 0.7 threshold.
  const maxPercentage = Math.max(...turnPercentages);
  const minPercentage = Math.min(...turnPercentages);
  const turnRatio = maxPercentage / (minPercentage || 0.01);

  const totalMessages = turns.reduce((sum, t) => sum + t.message_count, 0);
  const maxMessageShare = totalMessages > 0
    ? Math.max(...participants.map(p =>
        turns.filter(t => t.participant === p).reduce((sum, t) => sum + t.message_count, 0) / totalMessages
      ))
    : 0;
  
  // Calculate response times
  const responseTimes = turns.slice(1).map((turn, index) => 
    turn.start_time - turns[index].end_time
  ).filter(time => time > 0 && time < 24 * 60 * 60 * 1000); // Filter out unrealistic times
  
  const avgResponseTime = responseTimes.length > 0 
    ? responseTimes.reduce((sum, time) => sum + time, 0) / responseTimes.length 
    : 0;
  
  // Determine pattern type
  let pattern_type: 'balanced' | 'dominant' | 'responsive' | 'sporadic';
  let conversation_health: number;
  let description: string;
  
  if (maxMessageShare > 0.7) {
    pattern_type = 'dominant';
    conversation_health = 0.4;
    description = 'One participant dominates the conversation';
  } else if (turnRatio <= 2 && participants.length > 1) {
    pattern_type = 'balanced';
    conversation_health = 0.9;
    description = 'Well-balanced conversation with equal participation';
  } else if (maxPercentage > 0.7) {
    pattern_type = 'dominant';
    conversation_health = 0.4;
    description = 'One participant dominates the conversation';
  } else if (avgResponseTime < 5 * 60 * 1000 && turns.length > 10) { // < 5 minutes
    pattern_type = 'responsive';
    conversation_health = 0.8;
    description = 'Quick back-and-forth exchanges';
  } else {
    pattern_type = 'sporadic';
    conversation_health = 0.6;
    description = 'Irregular participation patterns';
  }
  
  return {
    pattern_type,
    participants,
    turn_ratio: Math.round(turnRatio * 100) / 100,
    conversation_health: Math.round(conversation_health * 100) / 100,
    description
  };
}

function calculateParticipantTurnData(
  conversationId: string,
  participant: string,
  turns: TurnSequence[],
  _allMessages: MessageRow[]
): TurnData {
  const participantTurns = turns.filter(t => t.participant === participant);
  
  if (participantTurns.length === 0) {
    return {
      conversation_id: conversationId,
      participant,
      turn_count: 0,
      avg_turn_length: 0,
      max_turn_length: 0,
      turn_percentage: 0,
      avg_response_time: 0,
      interruption_rate: 0
    };
  }
  
  const turnLengths = participantTurns.map(t => t.message_count);
  const avgTurnLength = turnLengths.reduce((sum, len) => sum + len, 0) / turnLengths.length;
  const maxTurnLength = Math.max(...turnLengths);
  const turnPercentage = (participantTurns.length / turns.length) * 100;
  
  // Calculate response times for this participant
  const responseTimes: number[] = [];
  for (let i = 1; i < turns.length; i++) {
    if (turns[i].participant === participant && turns[i-1].participant !== participant) {
      const responseTime = turns[i].start_time - turns[i-1].end_time;
      if (responseTime > 0 && responseTime < 24 * 60 * 60 * 1000) {
        responseTimes.push(responseTime);
      }
    }
  }
  
  const avgResponseTime = responseTimes.length > 0 
    ? responseTimes.reduce((sum, time) => sum + time, 0) / responseTimes.length 
    : 0;
  
  // Calculate interruption rate (simplified)
  const interruptions = participantTurns.filter((turn, index) => {
    if (index === 0) return false;
    const prevTurn = turns.find(t => t.turn_number === turn.turn_number - 1);
    return prevTurn && prevTurn.participant !== participant && turn.message_count === 1;
  }).length;
  
  const interruptionRate = participantTurns.length > 0 ? (interruptions / participantTurns.length) * 100 : 0;
  
  return {
    conversation_id: conversationId,
    participant,
    turn_count: participantTurns.length,
    avg_turn_length: Math.round(avgTurnLength * 10) / 10,
    max_turn_length: maxTurnLength,
    turn_percentage: Math.round(turnPercentage * 10) / 10,
    avg_response_time: Math.round(avgResponseTime),
    interruption_rate: Math.round(interruptionRate * 10) / 10
  };
}

export async function computeTurnTakingAnalysisMetrics(): Promise<void> {
  progressReporter.start('Computing turn-taking analysis metrics...');
  
  const db = await getDb();
  
  try {
    // Get all conversations with their messages
    const conversations = db.prepare(`
      SELECT DISTINCT conversation_id 
      FROM messages 
      ORDER BY conversation_id
    `).all() as { conversation_id: string }[];
    
    progressReporter.update(`Analyzing turn-taking patterns for ${conversations.length} conversations...`);
    
    const conversationPatterns: ConversationPattern[] = [];
    const participantStatsMap = new Map<string, {
      conversations: number;
      totalTurns: number;
      totalTurnLength: number;
      totalDominance: number;
      totalResponsiveness: number;
    }>();
    
    let totalConversations = 0;
    let totalParticipants = 0;
    let balancedConversations = 0;
    let dominantConversations = 0;
    let totalTurnLength = 0;
    let totalResponseTime = 0;
    let totalResponseCount = 0;
    
    for (const conv of conversations) {
      // Get messages for this conversation
      const messages = db.prepare(`
        SELECT id, conversation_id, sender, timestamp_ms, content
        FROM messages 
        WHERE conversation_id = ?
        ORDER BY timestamp_ms ASC
      `).all(conv.conversation_id) as MessageRow[];
      
      if (messages.length < 3) continue; // Skip conversations with too few messages
      
      // Get unique participants
      const participants = [...new Set(messages.map(m => m.sender))];
      if (participants.length < 2) continue; // Skip single-participant conversations
      
      // Analyze turn sequence
      const turnSequence = analyzeTurnSequence(messages);
      if (turnSequence.length === 0) continue;
      
      // Calculate turn pattern
      const pattern = calculateTurnPattern(turnSequence, participants);
      
      // Calculate participant turn data
      const participantTurnData = participants.map(participant => 
        calculateParticipantTurnData(conv.conversation_id, participant, turnSequence, messages)
      );
      
      conversationPatterns.push({
        conversation_id: conv.conversation_id,
        pattern,
        participants: participantTurnData
      });
      
      // Update summary statistics
      totalConversations++;
      totalParticipants += participants.length;
      
      if (pattern.pattern_type === 'balanced') balancedConversations++;
      if (pattern.pattern_type === 'dominant') dominantConversations++;
      
      const avgTurnLengthForConv = participantTurnData.reduce((sum, p) => sum + p.avg_turn_length, 0) / participantTurnData.length;
      totalTurnLength += avgTurnLengthForConv;
      
      const avgResponseTimeForConv = participantTurnData.reduce((sum, p) => sum + p.avg_response_time, 0) / participantTurnData.length;
      if (avgResponseTimeForConv > 0) {
        totalResponseTime += avgResponseTimeForConv;
        totalResponseCount++;
      }
      
      // Update participant statistics
      participantTurnData.forEach(participant => {
        if (!participantStatsMap.has(participant.participant)) {
          participantStatsMap.set(participant.participant, {
            conversations: 0,
            totalTurns: 0,
            totalTurnLength: 0,
            totalDominance: 0,
            totalResponsiveness: 0
          });
        }
        
        const stats = participantStatsMap.get(participant.participant)!;
        stats.conversations++;
        stats.totalTurns += participant.turn_count;
        stats.totalTurnLength += participant.avg_turn_length;
        stats.totalDominance += participant.turn_percentage;
        stats.totalResponsiveness += participant.avg_response_time > 0 ? (1 / participant.avg_response_time) * 1000000 : 0;
      });
    }
    
    // Calculate participant statistics
    const participantStats: ParticipantStats[] = Array.from(participantStatsMap.entries()).map(([participant, stats]) => ({
      participant,
      conversations: stats.conversations,
      avg_turn_count: Math.round(stats.totalTurns / stats.conversations),
      avg_turn_length: Math.round((stats.totalTurnLength / stats.conversations) * 10) / 10,
      dominance_score: Math.round((stats.totalDominance / stats.conversations) * 10) / 10,
      responsiveness_score: Math.round((stats.totalResponsiveness / stats.conversations) * 10) / 10
    })).sort((a, b) => b.dominance_score - a.dominance_score);
    
    // Create final metrics object
    const metrics: TurnTakingMetrics = {
      summary: {
        total_conversations: totalConversations,
        avg_participants: totalConversations > 0 ? Math.round((totalParticipants / totalConversations) * 10) / 10 : 0,
        balanced_conversations: balancedConversations,
        dominant_speaker_conversations: dominantConversations,
        avg_turn_length: totalConversations > 0 ? Math.round((totalTurnLength / totalConversations) * 10) / 10 : 0,
        avg_response_time: totalResponseCount > 0 ? Math.round(totalResponseTime / totalResponseCount) : 0
      },
      conversation_patterns: conversationPatterns,
      participant_stats: participantStats
    };
    
    // Export to JSON
    writeDashData('turnTakingAnalysis.json', metrics);
    
    progressReporter.success('Turn-taking analysis metrics computed and exported.');
    progressReporter.update(`Analyzed ${totalConversations} conversations`);
    progressReporter.update(`Found ${balancedConversations} balanced conversations`);
    progressReporter.update(`Found ${dominantConversations} dominant-speaker conversations`);
    progressReporter.update(`Average participants per conversation: ${metrics.summary.avg_participants}`);
    progressReporter.update(`Average turn length: ${metrics.summary.avg_turn_length} messages`);
    
  } catch (error) {
    console.error('Error computing turn-taking analysis metrics:', error);
    throw error;
  } finally {
    await closeDb(db);
  }
} 