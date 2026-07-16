# Development Guide

## 🏗 Architecture Overview

Doppelgänger Analytics follows a pipeline architecture:

```
Instagram ZIP → Import → Process → Export → Dashboard
     ↓            ↓         ↓        ↓         ↓
   JSON files  → SQLite → Metrics → JSON → Next.js
```

## 📊 Metrics Processing Pipeline

### **Phase 1: Data Import**
```typescript
// src/importer.ts — fields parsed from Instagram's message_*.json
interface RawMessage {
  sender_name: string;
  timestamp_ms: number;
  content?: string;
  photos?: Array<{ uri: string; creation_timestamp: number; }>;
  videos?: Array<{ uri: string; creation_timestamp: number; }>;
  audio_files?: Array<{ uri: string; creation_timestamp?: number; }>;
  share?: { link?: string; share_text?: string };
  reactions?: Array<{ reaction: string; actor: string; timestamp?: number; }>;
}
```
The importer clears all tables, then inserts messages plus photos/videos/audio/reactions
into their own tables, setting `has_photos`/`has_videos`/`has_audio`/`has_share` flags on
each message. All text passes through `src/utils/unicodeDecoder.ts` to fix Instagram's
mangled UTF-8-as-Latin-1 encoding.

### **Phase 2: Metrics Computation**
`src/generator.ts` runs the processors in `src/processors/` as an ordered pipeline
(`PIPELINE` array). Notable steps and ordering constraints:

- **sentiment.ts** — VADER scoring via the `vader-sentiment` package
  (`src/services/sentimentService.ts`); versioned in the `meta` table so an engine
  change auto-rescoring. Must run before any sentiment-derived metric.
- **responseTimes.ts** — the single source of truth for reply latency (cross-sender,
  1s–24h), written to the `response_times` table. Must run before conversation,
  additional, and engagement metrics, which read that table.
- **textProcessor / messageLengthMetrics / additionalMetrics / advancedMetrics** —
  text, length, monthly/hourly/latency/sentiment-by-sender, word & URL frequencies.
- **enhancedMediaProcessor / attachmentTypeMetrics / mediaEngagementMetrics** — media
  from the real `message_photos`/`message_videos`/`has_*` flags (not text matching).
- **enhancedEmotionProcessor, sentimentTimelineMetrics, moodCorrelationMetrics,
  emotionalPeaksMetrics, reactionMetrics, threadAnalysisMetrics (timing bursts),
  turnTakingAnalysisMetrics, engagementScoringMetrics,
  conversationStarterAnalysisMetrics, enhancedTimeMetrics, contentTypeMetrics.**

All processors write via `writeDashData(filename, data)` (compact JSON into `dash-data/`).

### **Phase 3: Data Export**
Generated JSON files land in `dash-data/`; the CLI `dashboard` command syncs them into
`dashboard/public/data/` (removing stale files). Both directories are git-ignored — they
are generated artifacts, not source.

## 🗄 Database Schema

(Instagram DM exports carry no reply/quote links, so there is no `reply_to_message_id`.)

### **Core Tables**
```sql
CREATE TABLE messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL,
  sender TEXT NOT NULL,
  timestamp_ms INTEGER NOT NULL,
  content TEXT,
  has_photos INTEGER DEFAULT 0,
  has_videos INTEGER DEFAULT 0,
  has_audio INTEGER DEFAULT 0,
  has_share INTEGER DEFAULT 0,
  share_link TEXT
);

-- VADER sentiment results
CREATE TABLE sentiment (
  message_id INTEGER,
  compound REAL,
  positive REAL,
  negative REAL,
  neutral REAL
);

-- Media / reactions (populated by the importer)
CREATE TABLE message_photos    (message_id INTEGER, uri TEXT, creation_timestamp INTEGER, backup_uri TEXT);
CREATE TABLE message_videos    (message_id INTEGER, uri TEXT, creation_timestamp INTEGER, backup_uri TEXT);
CREATE TABLE message_audio     (message_id INTEGER, uri TEXT, creation_timestamp INTEGER);
CREATE TABLE message_reactions (message_id INTEGER, reaction TEXT, actor TEXT, timestamp INTEGER);

-- Canonical reply latencies (derived, rebuilt each run)
CREATE TABLE response_times (
  conversation_id TEXT, from_message_id INTEGER, to_message_id INTEGER, latency_ms INTEGER
);

-- Pipeline metadata (e.g. sentiment engine version)
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

-- Emotion detection results
CREATE TABLE emotions (
  message_id INTEGER,
  emotion TEXT,
  score REAL
);
```

### **Additional Tables**
```sql
-- Conversation metadata
CREATE TABLE conversations (
  id TEXT PRIMARY KEY,
  title TEXT,
  is_still_participant BOOLEAN,
  thread_type TEXT,
  thread_path TEXT
);

-- Response timing analysis
CREATE TABLE response_times (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT,
  sender TEXT,
  response_time_ms INTEGER,
  timestamp_ms INTEGER
);

-- Text processing results
CREATE TABLE text_metrics (
  message_id INTEGER PRIMARY KEY,
  word_count INTEGER,
  emoji_count INTEGER,
  url_count INTEGER
);
```

**Note**: Media content (photos, videos, reactions) is detected through content analysis rather than separate tables, as Instagram's export format embeds this information within message content.

## 🔧 Unicode Handling

### **Problem**: Instagram's Malformed Unicode
Instagram exports contain sequences like `\u00e2\u0080\u0099` instead of proper Unicode.

### **Solution**: Enhanced Unicode Decoder
```typescript
// src/utils/unicodeDecoder.ts
export function decodeUnicode(text: string): string {
  // Handle common Instagram encoding patterns
  return text
    .replace(/\\u00e2\\u0080\\u0099/g, ''')  // Right single quotation
    .replace(/\\u00e2\\u0080\\u0098/g, ''')  // Left single quotation
    .replace(/\\u00e2\\u0080\\u009c/g, '"')  // Left double quotation
    .replace(/\\u00e2\\u0080\\u009d/g, '"')  // Right double quotation
    // ... additional patterns
}
```

### **Implementation**: Applied at Import
```typescript
// src/importer.ts - Applied during data import
const decodedMessage = {
  sender: decodeUnicode(rawMessage.sender_name),
  content: rawMessage.content ? decodeUnicode(rawMessage.content) : null,
  // ... other fields
};
```

## 🎨 Dashboard Architecture

### **Component Structure**
```
dashboard/src/
├── app/
│   ├── layout.tsx          # Main layout
│   └── page.tsx           # Dashboard entry point
├── components/
│   ├── tabs/              # 6 specialized tabs
│   │   ├── OverviewTab.tsx    # Comprehensive overview with organized sections
│   │   ├── MessagesTab.tsx    # Message content analysis
│   │   ├── SentimentTab.tsx   # Sentiment analysis and trends
│   │   ├── MediaTab.tsx       # Media sharing patterns
│   │   ├── ActivityTab.tsx    # Time-based activity analysis
│   │   └── ConversationsTab.tsx # Conversation-level metrics
│   ├── charts/            # 28+ chart components
│   └── shared/            # Shared components (StatsCard, etc.)
└── contexts/
    └── ConversationContext.tsx  # Conversation filtering
```

### **Overview Tab Architecture**

The Overview tab follows a hierarchical information architecture:

```
Overview Tab Structure:
├── Core Metrics Cards (4)
│   ├── Total Messages
│   ├── Unique Participants  
│   ├── Avg Response Time (calculated from latency buckets)
│   └── Emojis Used (Unicode-enhanced detection)
├── User Activity Analysis Section
│   ├── Top Contributors (absolute message counts)
│   ├── Message Share % (percentage calculations)
│   ├── Fast Responders (time conversion fixed)
│   └── Top Emoji Users (emoji counts)
├── Media Sharing Analysis Section
│   ├── Top Photo Sharers (photo-specific sorting)
│   ├── Top Video Sharers (video-specific sorting)
│   ├── All Media Sharers (total media sorting)
│   └── Media Summary (consolidated stats box)
├── Activity Charts Section
│   ├── Message Activity Over Time
│   └── Peak Activity Patterns
└── Communication Insights Panel
    ├── Communication Health
    ├── Engagement Level
    ├── Media Activity
    └── Response Pattern
```

**Key Implementation Details:**
- Section headers with icons for visual hierarchy
- Consistent color theming across related metrics
- Fixed height containers prevent chart growth issues
- Memoized calculations for efficient filtering
- Real-time recalculation without server requests

## 🧪 Testing Framework

### **Test Structure**
```
tests/
├── fixtures/
│   ├── basic-messages.json      # Known test data
│   └── expected-outputs.json    # Pre-calculated results
├── processors/                  # Processor-specific tests
├── unicode-decoding.test.ts     # 22 Unicode tests
└── utils/
    └── testDatabase.ts          # Database testing utilities
```

### **Test Data Format**
```json
// tests/fixtures/basic-messages.json
{
  "messages": [
    {
      "sender_name": "Test User",
      "timestamp_ms": 1735409465834,
      "content": "Hello\\u00e2\\u0080\\u0099s world!",
      "reactions": [
        {
          "reaction": "\\u00f0\\u009f\\u0098\\u0080",
          "actor": "Another User",
          "timestamp": 1735409465835
        }
      ]
    }
  ]
}
```

### **Running Tests**
Tests run under Jest in ESM mode (the `npm test` script sets
`NODE_OPTIONS=--experimental-vm-modules`) and require **Node 23** — the compiled
`better-sqlite3` native module is built for it (see `.nvmrc`).
```bash
# Run all tests (uses the ESM flag automatically)
npm test

# Run specific test suites
npm test -- unicode-decoding.test.ts
npm test -- processors/

# Run with coverage
npm run test:coverage
```

## 🔄 Development Workflow

### **Adding New Metrics**

1. **Create Processor**
```typescript
// src/processors/newMetric.ts
export async function computeNewMetric(): Promise<void> {
  const db = await openDatabase();
  
  // Query data
  const messages = await db.all('SELECT * FROM messages');
  
  // Process data
  const results = messages.map(processMessage);
  
  // Export results
  await writeFile('dash-data/newMetric.json', JSON.stringify(results));
}
```

2. **Add to Generator**
```typescript
// src/generator.ts
import { computeNewMetric } from './processors/newMetric';

export async function generateAllMetrics(): Promise<void> {
  // ... existing processors
  await computeNewMetric();
}
```

3. **Create Dashboard Component**
```typescript
// dashboard/src/components/NewMetricChart.tsx
export function NewMetricChart() {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    fetch('/data/newMetric.json')
      .then(res => res.json())
      .then(setData);
  }, []);
  
  return <ResponsiveContainer>/* Chart implementation */</ResponsiveContainer>;
}
```

4. **Add Tests**
```typescript
// tests/processors/newMetric.test.ts
describe('New Metric Processor', () => {
  it('should compute metric correctly', async () => {
    // Test implementation
  });
});
```

### **Performance Optimization Guidelines**

1. **Database Queries**
   - Use indexes on timestamp_ms, sender, conversation_id
   - Batch operations for large datasets
   - Use prepared statements for repeated queries

2. **Memory Management**
   - Process data in chunks for large datasets
   - Clear intermediate results
   - Use streaming for file operations

3. **Dashboard Performance**
   - Lazy load chart components
   - Implement proper React memoization
   - Use efficient data structures for filtering

## 🚀 Deployment

### **Production Build**
```bash
# Build analytics engine
npm run build

# Build dashboard
cd dashboard
npm run build
npm run start
```

### **Data Processing Performance**
- **Test Dataset**: 950,000+ messages
- **Processing Time**: 53.2 seconds (all 12 processors)
- **Memory Usage**: ~500MB peak
- **Output Size**: ~50MB JSON data files

### **Dashboard Performance**
- **Load Time**: <2 seconds initial load
- **Chart Rendering**: <500ms per chart
- **Filtering**: Real-time client-side processing
- **Export**: ~3 seconds for full dashboard PNG

## 🔍 Debugging

### **Common Issues**

1. **Unicode Decoding Problems**
   - Check `unicodeDecoder.test.ts` for pattern coverage
   - Verify Unicode sequences in source data
   - Test with `decodeUnicode()` function directly

2. **Database Connection Issues**
   - Ensure SQLite file permissions
   - Check database schema migrations
   - Verify table creation in `src/db/schema.ts`

3. **Dashboard Data Loading**
   - Check JSON file generation in `dash-data/`
   - Verify file paths in dashboard components
   - Test API endpoints in browser network tab

### **Debug Commands**
```bash
# Check database contents
sqlite3 messages.db ".tables"
sqlite3 messages.db "SELECT COUNT(*) FROM messages;"

# Validate JSON output
cat dash-data/textMetrics.json | jq .

# Test Unicode decoding
node -e "console.log(require('./src/utils/unicodeDecoder').decodeUnicode('test\\u00e2\\u0080\\u0099s'))"
```

## 📝 Code Style

### **TypeScript Standards**
- Strict type checking enabled
- Interface definitions for all data structures
- Proper error handling with try/catch
- Async/await for asynchronous operations

### **Database Patterns**
- Use parameterized queries to prevent injection
- Implement proper transaction handling
- Close database connections in finally blocks
- Use foreign key constraints for data integrity

### **Component Patterns**
- Custom hooks for data fetching
- Proper loading and error states
- Responsive design with Tailwind CSS
- Accessible chart components with tooltips

## 🔄 Progress Reporting System

### **Standardized Progress Reporting**

The project uses a unified progress reporting system for all analytics processing:

```typescript
import { progressReporter } from '../utils/progressReporter.js';

// Start a process with spinner
progressReporter.start('Computing metrics...');

// Update progress with status
progressReporter.update('Processing data...');

// Create progress bar for batch operations
const progressBar = progressReporter.createProgressBar(total, 'Processing items');
progressBar.tick(1);

// Success/error reporting
progressReporter.success('Metrics computed successfully');
progressReporter.error('Error computing metrics');
```

### **Progress Reporter Features**

- **Spinners**: For indeterminate operations (database queries, file I/O)
- **Progress Bars**: For batch processing with known totals
- **Status Updates**: Real-time progress information
- **Error Handling**: Consistent error reporting across all processors
- **Silent Mode**: Available for testing environments

### **Usage in Processors**

All metric processors follow this pattern:

```typescript
export async function computeNewMetric(): Promise<void> {
  progressReporter.start('Computing new metric...');
  
  try {
    // Processing logic
    progressReporter.update('Processing data...');
    
    // Batch operations
    const progressBar = progressReporter.createProgressBar(items.length, 'Processing items');
    for (const item of items) {
      // Process item
      progressBar.tick(1);
    }
    
    // Export results
    progressReporter.update('Exporting results...');
    
    progressReporter.success('New metric computed successfully');
  } catch (error) {
    progressReporter.error('Error computing new metric');
    throw error;
  }
}
```

### **Benefits**

- **Consistent UX**: All processors show progress in the same format
- **Better Feedback**: Users see real-time progress instead of static messages
- **Error Recovery**: Clear error messages help with debugging
- **Performance Monitoring**: Progress bars help identify slow operations

---

**Last Updated**: Current with all implemented features and optimizations
**Coverage**: Complete development guide for all system components 