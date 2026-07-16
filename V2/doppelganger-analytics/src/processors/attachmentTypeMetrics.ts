import { getDb } from '../db/client.js';
import { decodeInstagramUnicode } from '../utils/unicodeDecoder.js';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

interface AttachmentTypeData {
  conversation_id: string;
  type: string;
  count: number;
  percentage: number;
  examples: string[];
}

export async function computeAttachmentTypeMetrics(): Promise<void> {
  progressReporter.start('Computing attachment type metrics...');
  
  try {
    const db = await getDb();
    
    // Photos/videos/audio/shares come from the importer's flags; other
    // attachment kinds are only visible via Instagram's system message text
    const attachmentMessages = db.prepare(`
      SELECT
        conversation_id,
        content,
        sender,
        has_photos,
        has_videos,
        has_audio,
        has_share
      FROM messages
      WHERE has_photos = 1
         OR has_videos = 1
         OR has_audio = 1
         OR has_share = 1
         OR content LIKE '%sent an attachment%'
         OR content LIKE '%sent a gif%'
         OR content LIKE '%sent a sticker%'
         OR content LIKE '%sent a file%'
      ORDER BY conversation_id, sender
    `).all() as {
      conversation_id: string;
      content: string | null;
      sender: string;
      has_photos: number;
      has_videos: number;
      has_audio: number;
      has_share: number;
    }[];

    progressReporter.update(`Processing ${attachmentMessages.length} attachment messages...`);

    // Group by conversation and determine attachment types
    const conversationTypeMap = new Map<string, Map<string, {
      count: number;
      examples: Set<string>;
    }>>();

    attachmentMessages.forEach(msg => {
      if (!conversationTypeMap.has(msg.conversation_id)) {
        conversationTypeMap.set(msg.conversation_id, new Map());
      }

      const convTypeMap = conversationTypeMap.get(msg.conversation_id)!;
      const decodedContent = msg.content ? decodeInstagramUnicode(msg.content) : '';

      let attachmentType = 'other';

      if (msg.has_photos === 1) {
        attachmentType = 'image';
      } else if (msg.has_videos === 1) {
        attachmentType = 'video';
      } else if (msg.has_audio === 1) {
        attachmentType = 'audio';
      } else if (msg.has_share === 1) {
        attachmentType = 'link';
      } else if (decodedContent.includes('sent a gif')) {
        attachmentType = 'gif';
      } else if (decodedContent.includes('sent a sticker')) {
        attachmentType = 'sticker';
      } else if (decodedContent.includes('sent a file') || decodedContent.includes('sent an attachment')) {
        attachmentType = 'document';
      }
      
      if (!convTypeMap.has(attachmentType)) {
        convTypeMap.set(attachmentType, { count: 0, examples: new Set() });
      }
      
      const typeData = convTypeMap.get(attachmentType)!;
      typeData.count++;
      
      // Extract example content (sender name for privacy)
      if (typeData.examples.size < 3) {
        typeData.examples.add(`${msg.sender} shared ${attachmentType}`);
      }
    });

    // Convert to output format
    const results: AttachmentTypeData[] = [];
    
    for (const [conversationId, typeMap] of conversationTypeMap.entries()) {
      const totalAttachments = Array.from(typeMap.values()).reduce((sum, data) => sum + data.count, 0);
      
      for (const [type, data] of typeMap.entries()) {
        results.push({
          conversation_id: conversationId,
          type: type,
          count: data.count,
          percentage: (data.count / totalAttachments) * 100,
          examples: Array.from(data.examples)
        });
      }
    }

    progressReporter.update('Exporting attachment type metrics...');
    writeDashData('attachmentTypeMetrics.json', results);

    progressReporter.success(`Attachment type metrics computed: ${results.length} records across ${conversationTypeMap.size} conversations`);
    
  } catch (error) {
    progressReporter.error('Error computing attachment type metrics');
    console.error(error);
    throw error;
  }
} 