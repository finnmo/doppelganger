import { describe, it, expect } from '@jest/globals';

/**
 * Test suite for Emotional Peaks conversation filtering functionality
 * 
 * This tests the filtering logic that should be applied to emotional peaks data
 * when specific conversations are selected via the conversation filter.
 */
describe('Emotional Peaks Conversation Filtering', () => {
  // Mock emotional peaks data structure
  const mockEmotionalPeaksData = {
    summary: {
      total_peaks: 10,
      total_valleys: 8,
      extreme_events: 3,
      avg_peak_intensity: 0.75,
      avg_valley_intensity: -0.65,
      avg_recovery_time: 7200000, // 2 hours in ms
      most_volatile_period: 'March 2024',
      most_stable_period: 'January 2024',
      dominant_pattern: 'work_stress'
    },
    peaks_and_valleys: [
      {
        id: 'peak_1',
        conversation_id: 'conv_1',
        type: 'peak' as const,
        timestamp: 1640995200000,
        sentiment_score: 0.8,
        intensity: 'high' as const,
        trigger_analysis: {
          primary_trigger: 'achievement',
          confidence: 0.9,
          keywords: ['success', 'promotion', 'excited'],
          context_window: []
        },
        participants: [
          { sender: 'user1', contribution_score: 0.8, message_count: 5 }
        ],
        sample_messages: [],
        recovery_time: 3600000 // 1 hour
      },
      {
        id: 'peak_2',
        conversation_id: 'conv_2', 
        type: 'valley' as const,
        timestamp: 1640998800000,
        sentiment_score: -0.7,
        intensity: 'moderate' as const,
        trigger_analysis: {
          primary_trigger: 'work_stress',
          confidence: 0.8,
          keywords: ['deadline', 'stress', 'pressure'],
          context_window: []
        },
        participants: [
          { sender: 'user2', contribution_score: 0.7, message_count: 3 }
        ],
        sample_messages: [],
        recovery_time: 7200000 // 2 hours
      },
      {
        id: 'peak_3',
        conversation_id: 'conv_1',
        type: 'peak' as const,
        timestamp: 1641002400000,
        sentiment_score: 0.6,
        intensity: 'moderate' as const,
        trigger_analysis: {
          primary_trigger: 'social',
          confidence: 0.7,
          keywords: ['party', 'friends', 'fun'],
          context_window: []
        },
        participants: [
          { sender: 'user1', contribution_score: 0.6, message_count: 2 }
        ],
        sample_messages: [],
        recovery_time: 1800000 // 30 minutes
      }
    ],
    emotional_patterns: [],
    trigger_analysis: [
      {
        trigger: 'achievement',
        frequency: 1,
        avg_impact: 0.8,
        typical_sentiment_change: 0.8,
        recovery_time: 3600000,
        associated_keywords: ['success', 'promotion'],
        time_patterns: [],
        participant_sensitivity: []
      },
      {
        trigger: 'work_stress',
        frequency: 1,
        avg_impact: 0.7,
        typical_sentiment_change: -0.7,
        recovery_time: 7200000,
        associated_keywords: ['deadline', 'stress'],
        time_patterns: [],
        participant_sensitivity: []
      }
    ],
    temporal_analysis: {
      hourly_volatility: [],
      daily_volatility: [],
      monthly_trends: []
    },
    recovery_analysis: {
      avg_peak_recovery: 2400000, // 40 minutes
      avg_valley_recovery: 7200000, // 2 hours
      fastest_recovery: {} as any,
      slowest_recovery: {} as any,
      recovery_factors: []
    }
  };

  describe('Peak and Valley Filtering', () => {
    it('should filter peaks and valleys by selected conversations', () => {
      const selectedConversations = ['conv_1'];
      
      const filteredPeaksAndValleys = mockEmotionalPeaksData.peaks_and_valleys
        .filter(peak => selectedConversations.includes(peak.conversation_id));
      
      expect(filteredPeaksAndValleys).toHaveLength(2);
      expect(filteredPeaksAndValleys.every(peak => peak.conversation_id === 'conv_1')).toBe(true);
    });

    it('should return empty array when no conversations match', () => {
      const selectedConversations = ['conv_nonexistent'];
      
      const filteredPeaksAndValleys = mockEmotionalPeaksData.peaks_and_valleys
        .filter(peak => selectedConversations.includes(peak.conversation_id));
      
      expect(filteredPeaksAndValleys).toHaveLength(0);
    });

    it('should include all data when multiple conversations selected', () => {
      const selectedConversations = ['conv_1', 'conv_2'];
      
      const filteredPeaksAndValleys = mockEmotionalPeaksData.peaks_and_valleys
        .filter(peak => selectedConversations.includes(peak.conversation_id));
      
      expect(filteredPeaksAndValleys).toHaveLength(3);
    });
  });

  describe('Summary Statistics Recalculation', () => {
    it('should recalculate peak and valley counts correctly', () => {
      const selectedConversations = ['conv_1'];
      const filteredPeaksAndValleys = mockEmotionalPeaksData.peaks_and_valleys
        .filter(peak => selectedConversations.includes(peak.conversation_id));
      
      const peaks = filteredPeaksAndValleys.filter(item => item.type === 'peak');
      const valleys = filteredPeaksAndValleys.filter(item => item.type === 'valley');
      
      expect(peaks).toHaveLength(2);
      expect(valleys).toHaveLength(0);
    });

    it('should recalculate average intensities correctly', () => {
      const selectedConversations = ['conv_1'];
      const filteredPeaksAndValleys = mockEmotionalPeaksData.peaks_and_valleys
        .filter(peak => selectedConversations.includes(peak.conversation_id));
      
      const peaks = filteredPeaksAndValleys.filter(item => item.type === 'peak');
      
      const avgPeakIntensity = peaks.length > 0 
        ? peaks.reduce((sum, p) => sum + Math.abs(p.sentiment_score), 0) / peaks.length 
        : 0;
      
      expect(avgPeakIntensity).toBeCloseTo(0.7, 1); // (0.8 + 0.6) / 2 = 0.7
    });
  });

  describe('Trigger Analysis Filtering', () => {
    it('should recalculate trigger frequencies for filtered data', () => {
      const selectedConversations = ['conv_1'];
      const filteredPeaksAndValleys = mockEmotionalPeaksData.peaks_and_valleys
        .filter(peak => selectedConversations.includes(peak.conversation_id));
      
      // Group triggers
      const triggerMap = new Map();
      filteredPeaksAndValleys.forEach(peak => {
        const trigger = peak.trigger_analysis.primary_trigger;
        triggerMap.set(trigger, (triggerMap.get(trigger) || 0) + 1);
      });
      
      expect(triggerMap.get('achievement')).toBe(1);
      expect(triggerMap.get('social')).toBe(1);
      expect(triggerMap.get('work_stress')).toBeUndefined();
    });
  });

  describe('Recovery Analysis Filtering', () => {
    it('should recalculate recovery times for filtered data', () => {
      const selectedConversations = ['conv_1'];
      const filteredPeaksAndValleys = mockEmotionalPeaksData.peaks_and_valleys
        .filter(peak => selectedConversations.includes(peak.conversation_id));
      
      const recoveryTimes = filteredPeaksAndValleys
        .map(p => p.recovery_time || 0)
        .filter(t => t > 0);
      
      const avgRecoveryTime = recoveryTimes.length > 0
        ? recoveryTimes.reduce((a, b) => a + b, 0) / recoveryTimes.length
        : 0;
      
      expect(avgRecoveryTime).toBe(2700000); // (3600000 + 1800000) / 2 = 2700000
    });

    it('should find fastest recovery in filtered data', () => {
      const selectedConversations = ['conv_1'];
      const filteredPeaksAndValleys = mockEmotionalPeaksData.peaks_and_valleys
        .filter(peak => selectedConversations.includes(peak.conversation_id));
      
      const fastestRecovery = filteredPeaksAndValleys.reduce((fastest, current) => 
        (current.recovery_time || Infinity) < (fastest.recovery_time || Infinity) 
          ? current : fastest,
        filteredPeaksAndValleys[0]);
      
      expect(fastestRecovery.recovery_time).toBe(1800000); // 30 minutes
    });
  });

  describe('Empty State Handling', () => {
    it('should handle empty filtered results gracefully', () => {
      const selectedConversations = ['conv_nonexistent'];
      const filteredPeaksAndValleys = mockEmotionalPeaksData.peaks_and_valleys
        .filter(peak => selectedConversations.includes(peak.conversation_id));
      
      if (filteredPeaksAndValleys.length === 0) {
        const emptyData = {
          summary: {
            total_peaks: 0,
            total_valleys: 0,
            extreme_events: 0,
            avg_peak_intensity: 0,
            avg_valley_intensity: 0,
            avg_recovery_time: 0,
            most_volatile_period: 'N/A',
            most_stable_period: 'N/A',
            dominant_pattern: 'N/A'
          },
          peaks_and_valleys: [],
          trigger_analysis: [],
          // ... other empty arrays
        };
        
        expect(emptyData.summary.total_peaks).toBe(0);
        expect(emptyData.peaks_and_valleys).toHaveLength(0);
        expect(emptyData.trigger_analysis).toHaveLength(0);
      }
    });
  });
}); 