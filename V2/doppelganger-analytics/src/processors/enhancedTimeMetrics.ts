import { getDb, closeDb } from '../db/client.js';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

interface HourlyActivity {
  hour: number;
  message_count: number;
  media_count: number;
  avg_sentiment: number;
  unique_senders: number;
}

interface DailyActivity {
  day: string;
  day_number: number;
  message_count: number;
  media_count: number;
  avg_sentiment: number;
}

interface MonthlyActivity {
  month: string;
  message_count: number;
  media_count: number;
  avg_sentiment: number;
}

interface EnhancedTimeMetrics {
  hourly_activity: Record<string, number>;
  daily_activity: Record<string, number>;
  monthly_activity: Record<string, number>;
  peak_hours: number[];
  peak_days: string[];
  peak_months: string[];
  activity_patterns: {
    morning_peak: { hour: number; count: number };
    afternoon_peak: { hour: number; count: number };
    evening_peak: { hour: number; count: number };
    night_activity: number;
  };
  detailed_hourly: HourlyActivity[];
  detailed_daily: DailyActivity[];
  detailed_monthly: MonthlyActivity[];
  summary: {
    total_messages: number;
    total_media: number;
    most_active_hour: number;
    most_active_day: string;
    least_active_hour: number;
    activity_span_hours: number;
  };
}


export async function generateEnhancedTimeMetrics(): Promise<void> {
  progressReporter.start('Generating enhanced time metrics...');
  
  const db = await getDb();
  
  try {
    // Get hourly activity
    const hourlyData: HourlyActivity[] = db.prepare(`
      SELECT 
        CAST(strftime('%H', datetime(timestamp_ms/1000, 'unixepoch', 'localtime')) AS INTEGER) as hour,
        COUNT(*) as message_count,
        SUM(CASE WHEN has_photos = 1 OR has_videos = 1 OR content LIKE '%sent an attachment%' THEN 1 ELSE 0 END) as media_count,
        AVG(COALESCE(s.compound, 0)) as avg_sentiment,
        COUNT(DISTINCT sender) as unique_senders
      FROM messages m
      LEFT JOIN sentiment s ON m.id = s.message_id
      GROUP BY hour
      ORDER BY hour
    `).all() as HourlyActivity[];

    // Get daily activity
    const dailyData: DailyActivity[] = db.prepare(`
      SELECT 
        CASE CAST(strftime('%w', datetime(timestamp_ms/1000, 'unixepoch', 'localtime')) AS INTEGER)
          WHEN 0 THEN 'Sunday'
          WHEN 1 THEN 'Monday'
          WHEN 2 THEN 'Tuesday'
          WHEN 3 THEN 'Wednesday'
          WHEN 4 THEN 'Thursday'
          WHEN 5 THEN 'Friday'
          WHEN 6 THEN 'Saturday'
        END as day,
        CAST(strftime('%w', datetime(timestamp_ms/1000, 'unixepoch', 'localtime')) AS INTEGER) as day_number,
        COUNT(*) as message_count,
        SUM(CASE WHEN has_photos = 1 OR has_videos = 1 OR content LIKE '%sent an attachment%' THEN 1 ELSE 0 END) as media_count,
        AVG(COALESCE(s.compound, 0)) as avg_sentiment
      FROM messages m
      LEFT JOIN sentiment s ON m.id = s.message_id
      GROUP BY day_number
      ORDER BY day_number
    `).all() as DailyActivity[];

    // Get monthly activity
    const monthlyData: MonthlyActivity[] = db.prepare(`
      SELECT 
        strftime('%Y-%m', datetime(timestamp_ms/1000, 'unixepoch', 'localtime')) as month,
        COUNT(*) as message_count,
        SUM(CASE WHEN has_photos = 1 OR has_videos = 1 OR content LIKE '%sent an attachment%' THEN 1 ELSE 0 END) as media_count,
        AVG(COALESCE(s.compound, 0)) as avg_sentiment
      FROM messages m
      LEFT JOIN sentiment s ON m.id = s.message_id
      GROUP BY month
      ORDER BY month
    `).all() as MonthlyActivity[];

    progressReporter.update(`Processed ${hourlyData.length} hourly slots, ${dailyData.length} days, ${monthlyData.length} months`);

    // Calculate peak hours (top 3)
    const sortedHours = [...hourlyData].sort((a, b) => b.message_count - a.message_count);
    const peakHours = sortedHours.slice(0, 3).map(h => h.hour);

    // Calculate peak days (top 3)
    const sortedDays = [...dailyData].sort((a, b) => b.message_count - a.message_count);
    const peakDays = sortedDays.slice(0, 3).map(d => d.day);

    // Calculate peak months (top 3)
    const sortedMonths = [...monthlyData].sort((a, b) => b.message_count - a.message_count);
    const peakMonths = sortedMonths.slice(0, 3).map(m => m.month);

    // Calculate activity patterns
    const morningHours = hourlyData.filter(h => h.hour >= 6 && h.hour < 12);
    const afternoonHours = hourlyData.filter(h => h.hour >= 12 && h.hour < 18);
    const eveningHours = hourlyData.filter(h => h.hour >= 18 && h.hour < 24);
    const nightHours = hourlyData.filter(h => h.hour >= 0 && h.hour < 6);

    const morningPeak = morningHours.reduce((max, h) => h.message_count > max.message_count ? h : max, morningHours[0] || { hour: 9, message_count: 0 });
    const afternoonPeak = afternoonHours.reduce((max, h) => h.message_count > max.message_count ? h : max, afternoonHours[0] || { hour: 15, message_count: 0 });
    const eveningPeak = eveningHours.reduce((max, h) => h.message_count > max.message_count ? h : max, eveningHours[0] || { hour: 20, message_count: 0 });
    const nightActivity = nightHours.reduce((sum, h) => sum + h.message_count, 0);

    // Create simplified activity maps
    const hourlyActivity: Record<string, number> = {};
    hourlyData.forEach(h => {
      hourlyActivity[h.hour.toString()] = h.message_count;
    });

    const dailyActivity: Record<string, number> = {};
    dailyData.forEach(d => {
      dailyActivity[d.day] = d.message_count;
    });

    const monthlyActivity: Record<string, number> = {};
    monthlyData.forEach(m => {
      monthlyActivity[m.month] = m.message_count;
    });

    // Calculate summary stats
    const totalMessages = hourlyData.reduce((sum, h) => sum + h.message_count, 0);
    const totalMedia = hourlyData.reduce((sum, h) => sum + h.media_count, 0);
    const mostActiveHour = sortedHours[0]?.hour || 12;
    const mostActiveDay = sortedDays[0]?.day || 'Monday';
    const leastActiveHour = sortedHours[sortedHours.length - 1]?.hour || 3;
    const activeHours = hourlyData.filter(h => h.message_count > 0).length;

    // Build final metrics object
    const metrics: EnhancedTimeMetrics = {
      hourly_activity: hourlyActivity,
      daily_activity: dailyActivity,
      monthly_activity: monthlyActivity,
      peak_hours: peakHours,
      peak_days: peakDays,
      peak_months: peakMonths,
      activity_patterns: {
        morning_peak: { hour: morningPeak.hour, count: morningPeak.message_count },
        afternoon_peak: { hour: afternoonPeak.hour, count: afternoonPeak.message_count },
        evening_peak: { hour: eveningPeak.hour, count: eveningPeak.message_count },
        night_activity: nightActivity
      },
      detailed_hourly: hourlyData,
      detailed_daily: dailyData,
      detailed_monthly: monthlyData,
      summary: {
        total_messages: totalMessages,
        total_media: totalMedia,
        most_active_hour: mostActiveHour,
        most_active_day: mostActiveDay,
        least_active_hour: leastActiveHour,
        activity_span_hours: activeHours
      }
    };

    // Write to file
    writeDashData('timeMetrics.json', metrics);
    
    progressReporter.success('Enhanced time metrics generated');
    progressReporter.update(`Peak hours: ${peakHours.join(', ')}`);
    progressReporter.update(`Peak days: ${peakDays.join(', ')}`);

  } catch (error) {
    progressReporter.error('Error generating enhanced time metrics');
    console.error(error);
    throw error;
  } finally {
    await closeDb(db);
  }
}
