import { getDb, closeDb } from '../db/client.js';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

interface MediaMetrics {
  conversation_id: string;
  photo_count: number;
  video_count: number;
  attachment_count: number;
  total_messages: number;
  media_ratio: number;
}

interface SenderMediaData {
  conversation_id: string;
  sender: string;
  photo_count: number;
  video_count: number;
  attachment_count: number;
  total_media: number;
}

interface AttachmentTimeSeries {
  conversation_id: string;
  month: string;
  sender: string;
  photo_count: number;
  video_count: number;
}

interface EnhancedMediaData {
  summary: {
    total_media_messages: number;
    total_photos: number;
    total_videos: number;
    total_attachments: number;
    media_percentage: number;
    top_media_sender: string;
    most_active_month: string;
  };
  conversation_metrics: MediaMetrics[];
  sender_media_data: SenderMediaData[];
  attachment_time_series: AttachmentTimeSeries[];
  media_type_distribution: {
    photos: number;
    videos: number;
    attachments: number;
  };
}

// Instagram's system phrase for attachments that aren't exported as photo/video entries
const ATTACHMENT_PHRASE = '%sent an attachment%';

export async function generateEnhancedMediaData(): Promise<void> {
  progressReporter.start('Generating enhanced media data...');

  const db = await getDb();

  try {
    // Photos and videos come from the media tables the importer populates;
    // attachments are only visible via Instagram's system message text.
    const senderRows = db.prepare(`
      SELECT
        m.conversation_id,
        m.sender,
        SUM((SELECT COUNT(*) FROM message_photos p WHERE p.message_id = m.id)) AS photo_count,
        SUM((SELECT COUNT(*) FROM message_videos v WHERE v.message_id = m.id)) AS video_count,
        SUM(CASE WHEN m.content LIKE '${ATTACHMENT_PHRASE}' THEN 1 ELSE 0 END) AS attachment_count,
        COUNT(*) AS total_messages,
        SUM(CASE
          WHEN m.has_photos = 1 OR m.has_videos = 1 OR m.content LIKE '${ATTACHMENT_PHRASE}' THEN 1
          ELSE 0
        END) AS media_message_count
      FROM messages m
      GROUP BY m.conversation_id, m.sender
    `).all() as Array<{
      conversation_id: string;
      sender: string;
      photo_count: number;
      video_count: number;
      attachment_count: number;
      total_messages: number;
      media_message_count: number;
    }>;

    const monthlyRows = db.prepare(`
      SELECT
        m.conversation_id,
        strftime('%Y-%m', datetime(m.timestamp_ms / 1000, 'unixepoch', 'localtime')) AS month,
        m.sender,
        SUM((SELECT COUNT(*) FROM message_photos p WHERE p.message_id = m.id)) AS photo_count,
        SUM((SELECT COUNT(*) FROM message_videos v WHERE v.message_id = m.id)) AS video_count
      FROM messages m
      WHERE m.has_photos = 1 OR m.has_videos = 1
      GROUP BY m.conversation_id, month, m.sender
      ORDER BY month
    `).all() as AttachmentTimeSeries[];

    progressReporter.update(`Aggregated media for ${senderRows.length} participant-conversation pairs...`);

    // Roll sender-level rows up to conversations
    const conversationTotals = new Map<string, MediaMetrics & { media_message_count: number }>();
    let totalPhotos = 0;
    let totalVideos = 0;
    let totalAttachments = 0;
    let totalMediaMessages = 0;
    let totalMessages = 0;

    for (const row of senderRows) {
      let conv = conversationTotals.get(row.conversation_id);
      if (!conv) {
        conv = {
          conversation_id: row.conversation_id,
          photo_count: 0,
          video_count: 0,
          attachment_count: 0,
          total_messages: 0,
          media_ratio: 0,
          media_message_count: 0
        };
        conversationTotals.set(row.conversation_id, conv);
      }
      conv.photo_count += row.photo_count;
      conv.video_count += row.video_count;
      conv.attachment_count += row.attachment_count;
      conv.total_messages += row.total_messages;
      conv.media_message_count += row.media_message_count;

      totalPhotos += row.photo_count;
      totalVideos += row.video_count;
      totalAttachments += row.attachment_count;
      totalMediaMessages += row.media_message_count;
      totalMessages += row.total_messages;
    }

    const conversationMetrics: MediaMetrics[] = Array.from(conversationTotals.values()).map(conv => ({
      conversation_id: conv.conversation_id,
      photo_count: conv.photo_count,
      video_count: conv.video_count,
      attachment_count: conv.attachment_count,
      total_messages: conv.total_messages,
      media_ratio: conv.total_messages > 0
        ? (conv.photo_count + conv.video_count + conv.attachment_count) / conv.total_messages
        : 0
    }));

    const senderMediaData: SenderMediaData[] = senderRows
      .map(row => ({
        conversation_id: row.conversation_id,
        sender: row.sender,
        photo_count: row.photo_count,
        video_count: row.video_count,
        attachment_count: row.attachment_count,
        total_media: row.photo_count + row.video_count + row.attachment_count
      }))
      .filter(row => row.total_media > 0)
      .sort((a, b) => b.total_media - a.total_media);

    const topSender = senderMediaData[0] || { sender: 'Unknown', total_media: 0 };

    const monthlyTotals = new Map<string, number>();
    for (const row of monthlyRows) {
      monthlyTotals.set(row.month, (monthlyTotals.get(row.month) || 0) + row.photo_count + row.video_count);
    }
    let mostActiveMonth = 'Unknown';
    let mostActiveCount = 0;
    for (const [month, count] of monthlyTotals.entries()) {
      if (count > mostActiveCount) {
        mostActiveMonth = month;
        mostActiveCount = count;
      }
    }

    const enhancedData: EnhancedMediaData = {
      summary: {
        total_media_messages: totalMediaMessages,
        total_photos: totalPhotos,
        total_videos: totalVideos,
        total_attachments: totalAttachments,
        media_percentage: totalMessages > 0 ? (totalMediaMessages / totalMessages) * 100 : 0,
        top_media_sender: topSender.sender,
        most_active_month: mostActiveMonth
      },
      conversation_metrics: conversationMetrics,
      sender_media_data: senderMediaData,
      attachment_time_series: monthlyRows,
      media_type_distribution: {
        photos: totalPhotos,
        videos: totalVideos,
        attachments: totalAttachments
      }
    };

    writeDashData('mediaMetrics.json', enhancedData);

    progressReporter.success('Enhanced media data generated');
    progressReporter.update(`Media messages: ${totalMediaMessages}`);
    progressReporter.update(`Photos: ${totalPhotos}, Videos: ${totalVideos}, Attachments: ${totalAttachments}`);
    progressReporter.update(`Top sender: ${topSender.sender} (${topSender.total_media} items)`);
    progressReporter.update(`Most active month: ${mostActiveMonth}`);

  } catch (error) {
    progressReporter.error('Error generating enhanced media data');
    console.error(error);
    throw error;
  } finally {
    await closeDb(db);
  }
}
