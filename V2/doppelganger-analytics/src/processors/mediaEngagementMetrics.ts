import { Database } from 'better-sqlite3';
import chalk from 'chalk';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

interface MediaMessage {
  id: number;
  sender: string;
  timestamp: number;
  conversationId: string;
  mediaType: 'photo' | 'video' | 'attachment';
  content: string;
}

interface ResponseMetric {
  messageId: number;
  responseCount: number;
  avgResponseTime: number;
  responseTypes: {
    text: number;
    media: number;
    reaction: number;
  };
  engagementScore: number;
}

interface MediaEngagementCorrelation {
  mediaType: 'photo' | 'video' | 'attachment';
  totalCount: number;
  avgResponseCount: number;
  avgResponseTime: number;
  engagementRate: number;
  responseDistribution: {
    immediate: number; // < 5 min
    quick: number;     // 5-30 min
    delayed: number;   // 30min-2hr
    late: number;      // > 2hr
  };
  topResponders: Array<{
    sender: string;
    responseCount: number;
    avgResponseTime: number;
  }>;
}

interface SenderMediaEngagement {
  sender: string;
  mediaShared: {
    photos: number;
    videos: number;
    attachments: number;
    total: number;
  };
  engagementReceived: {
    totalResponses: number;
    avgResponseTime: number;
    engagementScore: number;
  };
  engagementGiven: {
    responsesToMedia: number;
    avgResponseTime: number;
    preferredMediaType: string;
  };
  engagementRatio: number; // received/given
}

interface TimeBasedEngagement {
  hour: number;
  mediaCount: number;
  avgEngagement: number;
  responseRate: number;
}

export async function computeMediaEngagementMetrics(db: Database): Promise<void> {
  progressReporter.start('Computing media-engagement correlation metrics...');
  
  try {
    // Get all media messages from the real media flags; attachments are only
    // visible via Instagram's system message text
    const mediaMessages = db.prepare(`
      SELECT
        id,
        sender,
        timestamp_ms as timestamp,
        conversation_id as conversationId,
        content,
        CASE
          WHEN has_photos = 1 THEN 'photo'
          WHEN has_videos = 1 THEN 'video'
          ELSE 'attachment'
        END as mediaType
      FROM messages
      WHERE has_photos = 1
         OR has_videos = 1
         OR content LIKE '%sent an attachment%'
      ORDER BY timestamp_ms
    `).all() as Array<MediaMessage & { mediaType: string }>;

    progressReporter.update(`Analyzing ${mediaMessages.length.toLocaleString()} media messages for engagement patterns...`);

    // Get all messages for response analysis
    const allMessages = db.prepare(`
      SELECT id, sender, timestamp_ms, conversation_id, content, has_photos, has_videos
      FROM messages
      ORDER BY conversation_id, timestamp_ms
    `).all() as Array<{
      id: number;
      sender: string;
      timestamp_ms: number;
      conversation_id: string;
      content: string;
      has_photos: number;
      has_videos: number;
    }>;

    // Group messages by conversation for efficient lookup, and index each
    // message's position so responses can be found without scanning
    const conversationMessages = new Map<string, Array<{
      id: number;
      sender: string;
      timestamp_ms: number;
      content: string;
      has_media: boolean;
    }>>();
    const messageIndexById = new Map<number, number>();

    for (const msg of allMessages) {
      if (!conversationMessages.has(msg.conversation_id)) {
        conversationMessages.set(msg.conversation_id, []);
      }
      const conversation = conversationMessages.get(msg.conversation_id)!;
      messageIndexById.set(msg.id, conversation.length);
      conversation.push({
        id: msg.id,
        sender: msg.sender,
        timestamp_ms: msg.timestamp_ms,
        content: msg.content,
        has_media: msg.has_photos === 1 || msg.has_videos === 1
      });
    }

    // Analyze engagement for each media message
    const responseMetrics: ResponseMetric[] = [];
    const mediaEngagementMap = new Map<string, {
      count: number;
      totalResponses: number;
      totalResponseTime: number;
      responseTypes: { text: number; media: number; reaction: number };
      responseDistribution: { immediate: number; quick: number; delayed: number; late: number };
      responders: Map<string, { count: number; totalTime: number }>;
    }>();

    // Add progress bar for media message processing (always show if any messages)
    if (mediaMessages.length > 0) {
      const progressBar = progressReporter.createProgressBar(mediaMessages.length, 'Processing media messages');
      const responseWindow = 2 * 60 * 60 * 1000; // Look for responses in the next 2 hours
      for (let i = 0; i < mediaMessages.length; i++) {
        const mediaMsg = mediaMessages[i];
        const conversationMsgs = conversationMessages.get(mediaMsg.conversationId) || [];
        const mediaIndex = messageIndexById.get(mediaMsg.id);

        if (mediaIndex === undefined) continue;

        // Messages are sorted by timestamp, so stop at the first one outside the window
        const responses: typeof conversationMsgs = [];
        for (let j = mediaIndex + 1; j < conversationMsgs.length; j++) {
          const candidate = conversationMsgs[j];
          if (candidate.timestamp_ms > mediaMsg.timestamp + responseWindow) break;
          if (candidate.sender !== mediaMsg.sender) {
            responses.push(candidate);
          }
        }

        let responseCount = 0;
        let totalResponseTime = 0;
        const responseTypes = { text: 0, media: 0, reaction: 0 };
        const responseDistribution = { immediate: 0, quick: 0, delayed: 0, late: 0 };

        for (const response of responses) {
          const responseTime = response.timestamp_ms - mediaMsg.timestamp;
          responseCount++;
          totalResponseTime += responseTime;

          // Categorize response type
          if (response.content && (response.content.includes('reacted to') || response.content.includes('❤️'))) {
            responseTypes.reaction++;
          } else if (response.has_media) {
            responseTypes.media++;
          } else {
            responseTypes.text++;
          }

          // Categorize response timing
          const responseTimeMin = responseTime / (1000 * 60);
          if (responseTimeMin < 5) {
            responseDistribution.immediate++;
          } else if (responseTimeMin < 30) {
            responseDistribution.quick++;
          } else if (responseTimeMin < 120) {
            responseDistribution.delayed++;
          } else {
            responseDistribution.late++;
          }
        }

        const avgResponseTime = responseCount > 0 ? totalResponseTime / responseCount : 0;
        const engagementScore = responseCount * 10 + (responseTypes.reaction * 2) + (responseTypes.media * 5);

        responseMetrics.push({
          messageId: mediaMsg.id,
          responseCount,
          avgResponseTime,
          responseTypes,
          engagementScore
        });

        // Update media type aggregation
        const mediaType = mediaMsg.mediaType as 'photo' | 'video' | 'attachment';
        if (!mediaEngagementMap.has(mediaType)) {
          mediaEngagementMap.set(mediaType, {
            count: 0,
            totalResponses: 0,
            totalResponseTime: 0,
            responseTypes: { text: 0, media: 0, reaction: 0 },
            responseDistribution: { immediate: 0, quick: 0, delayed: 0, late: 0 },
            responders: new Map()
          });
        }

        const mediaData = mediaEngagementMap.get(mediaType)!;
        mediaData.count++;
        mediaData.totalResponses += responseCount;
        mediaData.totalResponseTime += avgResponseTime;
        mediaData.responseTypes.text += responseTypes.text;
        mediaData.responseTypes.media += responseTypes.media;
        mediaData.responseTypes.reaction += responseTypes.reaction;
        mediaData.responseDistribution.immediate += responseDistribution.immediate;
        mediaData.responseDistribution.quick += responseDistribution.quick;
        mediaData.responseDistribution.delayed += responseDistribution.delayed;
        mediaData.responseDistribution.late += responseDistribution.late;

        // Track responders
        for (const response of responses) {
          if (!mediaData.responders.has(response.sender)) {
            mediaData.responders.set(response.sender, { count: 0, totalTime: 0 });
          }
          const responder = mediaData.responders.get(response.sender)!;
          responder.count++;
          responder.totalTime += (response.timestamp_ms - mediaMsg.timestamp);
        }
        progressBar.tick(1);
        if (i % 1000 === 0 && i > 0) {
          progressReporter.update(`Processed ${i}/${mediaMessages.length} media messages...`);
        }
      }
    }

    // Process media engagement correlations
    const mediaEngagementCorrelations: MediaEngagementCorrelation[] = [];
    
    for (const [mediaType, data] of mediaEngagementMap.entries()) {
      const avgResponseCount = data.count > 0 ? data.totalResponses / data.count : 0;
      const avgResponseTime = data.totalResponses > 0 ? data.totalResponseTime / data.totalResponses : 0;
      const engagementRate = data.count > 0 ? (data.totalResponses / data.count) * 100 : 0;

      const topResponders = Array.from(data.responders.entries())
        .map(([sender, stats]) => ({
          sender,
          responseCount: stats.count,
          avgResponseTime: stats.count > 0 ? stats.totalTime / stats.count : 0
        }))
        .sort((a, b) => b.responseCount - a.responseCount)
        .slice(0, 5);

      mediaEngagementCorrelations.push({
        mediaType: mediaType as 'photo' | 'video' | 'attachment',
        totalCount: data.count,
        avgResponseCount,
        avgResponseTime,
        engagementRate,
        responseDistribution: data.responseDistribution,
        topResponders
      });
    }

    // Calculate sender-specific engagement patterns
    const senderEngagementMap = new Map<string, {
      mediaShared: { photos: number; videos: number; attachments: number };
      responsesReceived: number;
      totalResponseTime: number;
      responsesGiven: number;
      responseTimeGiven: number;
      mediaResponsesGiven: Map<string, number>;
    }>();

    // Initialize sender data
    for (const msg of mediaMessages) {
      if (!senderEngagementMap.has(msg.sender)) {
        senderEngagementMap.set(msg.sender, {
          mediaShared: { photos: 0, videos: 0, attachments: 0 },
          responsesReceived: 0,
          totalResponseTime: 0,
          responsesGiven: 0,
          responseTimeGiven: 0,
          mediaResponsesGiven: new Map()
        });
      }

      const senderData = senderEngagementMap.get(msg.sender)!;
      if (msg.mediaType === 'photo') senderData.mediaShared.photos++;
      else if (msg.mediaType === 'video') senderData.mediaShared.videos++;
      else if (msg.mediaType === 'attachment') senderData.mediaShared.attachments++;
    }

    // Calculate engagement metrics for each sender
    const mediaMessageById = new Map(mediaMessages.map(m => [m.id, m]));
    for (const metric of responseMetrics) {
      const mediaMsg = mediaMessageById.get(metric.messageId);
      if (mediaMsg) {
        const senderData = senderEngagementMap.get(mediaMsg.sender)!;
        senderData.responsesReceived += metric.responseCount;
        senderData.totalResponseTime += metric.avgResponseTime;
      }
    }

    const senderEngagementStats: SenderMediaEngagement[] = Array.from(senderEngagementMap.entries())
      .map(([sender, data]) => {
        const totalMedia = data.mediaShared.photos + data.mediaShared.videos + data.mediaShared.attachments;
        const avgResponseTime = data.responsesReceived > 0 ? data.totalResponseTime / data.responsesReceived : 0;
        const engagementScore = data.responsesReceived * 10 + totalMedia * 5;

        return {
          sender,
          mediaShared: {
            photos: data.mediaShared.photos,
            videos: data.mediaShared.videos,
            attachments: data.mediaShared.attachments,
            total: totalMedia
          },
          engagementReceived: {
            totalResponses: data.responsesReceived,
            avgResponseTime,
            engagementScore
          },
          engagementGiven: {
            responsesToMedia: data.responsesGiven,
            avgResponseTime: data.responseTimeGiven,
            preferredMediaType: data.mediaShared.photos > data.mediaShared.videos ? 'photos' : 'videos'
          },
          engagementRatio: data.responsesGiven > 0 ? data.responsesReceived / data.responsesGiven : data.responsesReceived
        };
      })
      .filter(s => s.mediaShared.total > 0)
      .sort((a, b) => b.engagementReceived.engagementScore - a.engagementReceived.engagementScore);

    // Time-based engagement analysis, derived from the response metrics
    // computed above (the response_times table is not populated)
    const hourBuckets = new Map<number, { mediaCount: number; totalResponses: number; responded: number }>();
    for (const metric of responseMetrics) {
      const mediaMsg = mediaMessageById.get(metric.messageId);
      if (!mediaMsg) continue;
      const hour = new Date(mediaMsg.timestamp).getHours();
      let bucket = hourBuckets.get(hour);
      if (!bucket) {
        bucket = { mediaCount: 0, totalResponses: 0, responded: 0 };
        hourBuckets.set(hour, bucket);
      }
      bucket.mediaCount++;
      bucket.totalResponses += metric.responseCount;
      if (metric.responseCount > 0) bucket.responded++;
    }

    const timeEngagement: TimeBasedEngagement[] = Array.from(hourBuckets.entries())
      .map(([hour, bucket]) => ({
        hour,
        mediaCount: bucket.mediaCount,
        avgEngagement: bucket.mediaCount > 0 ? bucket.totalResponses / bucket.mediaCount : 0,
        responseRate: bucket.mediaCount > 0 ? (bucket.responded / bucket.mediaCount) * 100 : 0
      }))
      .sort((a, b) => a.hour - b.hour);

    // Export data

    const mediaEngagementData = {
      summary: {
        totalMediaMessages: mediaMessages.length,
        avgEngagementRate: mediaEngagementCorrelations.reduce((sum, m) => sum + m.engagementRate, 0) / mediaEngagementCorrelations.length,
        mostEngagingType: mediaEngagementCorrelations.sort((a, b) => b.engagementRate - a.engagementRate)[0]?.mediaType || 'photo',
        totalSenders: senderEngagementStats.length,
        analysisWindow: '2 hours post-media'
      },
      mediaCorrelations: mediaEngagementCorrelations,
      senderEngagement: senderEngagementStats.slice(0, 20),
      timeBasedEngagement: timeEngagement,
      responseMetrics: responseMetrics.slice(0, 100) // Sample for detailed analysis
    };

    writeDashData('mediaEngagementMetrics.json', mediaEngagementData);

    progressReporter.success('Media-engagement correlation metrics computed and exported.');
    progressReporter.update(`Analyzed ${mediaMessages.length.toLocaleString()} media messages`);
    progressReporter.update(`Found ${senderEngagementStats.length} active media sharers`);
    progressReporter.update(`Most engaging type: ${mediaEngagementData.summary.mostEngagingType}`);
    progressReporter.update(`Average engagement rate: ${mediaEngagementData.summary.avgEngagementRate.toFixed(1)}%`);
    
  } catch (error) {
    console.error(chalk.red('❌ Error computing media-engagement metrics:'), error);
    throw error;
  }
} 