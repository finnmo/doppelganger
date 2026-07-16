# Doppelgänger Analytics

Turn your Instagram message export into an interactive analytics dashboard on your own computer. Your messages stay on your machine — nothing is uploaded.

## Just want to see your chats?

**Recommended path: Docker.** You do not need to install Node.js or know how to code. You only need Docker Desktop and your Instagram export ZIP.

### Step A — Get your Instagram messages (ZIP)

Instagram does not email the ZIP instantly. Request it, then wait (often a few hours, sometimes a day or two).

1. Open Instagram on your phone or [instagram.com](https://www.instagram.com) → your profile → **Settings and activity** (or Meta Accounts Center).
2. Go to **Your information and permissions** → **Download your information**.
3. Choose **Download or transfer information** → select your Instagram account.
4. Choose **Some of your information** → turn on **Messages** (DMs). You can leave other categories off.
5. Set **Format** to **JSON** (not HTML). Date range: **All time** is fine. Media quality: **Low** is enough.
6. Submit the request. When Meta emails you a download link, download the ZIP to your computer (e.g. Downloads).

Keep that ZIP file handy for the next step.

### Step B — Run with Docker (easiest)

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and open it. Wait until it says Docker is running (the whale icon is idle, not “starting”).
2. Get this project onto your computer (clone or download the folder). You should see a file named `docker-compose.yml` inside it.
3. In that same project folder, create a folder named `data` (if it is not already there).
4. Put your Instagram ZIP **inside** the `data` folder. One ZIP is enough.
5. Open Terminal (Mac) or PowerShell (Windows), go into the project folder, then run:

```bash
docker compose up --build
```

The first run can take several minutes while Docker downloads and builds. Leave the window open.
6. When the log says the dashboard is ready, open a browser to **http://localhost:3000**.

To stop: press `Ctrl+C` in that terminal window, or quit Docker Desktop.

**Privacy:** Import, analysis, and the dashboard all run locally in Docker on your computer.

**Updating with a new export:** Replace the ZIP in `data`, then run:

```bash
FORCE_REIMPORT=1 docker compose up
```

(On Windows PowerShell: `$env:FORCE_REIMPORT=1; docker compose up`)

#### If something goes wrong (Docker)

| What you see | What to try |
|---|---|
| “Cannot connect to the Docker daemon” | Open Docker Desktop and wait until it is fully started, then run the command again. |
| “No Instagram export found” | Confirm the ZIP is inside the project’s `data` folder (not next to it), then restart with `docker compose up`. |
| Browser can’t open localhost:3000 | Wait until the build finishes; the first run is slow. Check the terminal for error messages. |
| Old chats after replacing the ZIP | Run the `FORCE_REIMPORT=1` command above so it re-imports. |
| Port 3000 already in use | Quit the other app, or run `HOST_PORT=3001 docker compose up` and open http://localhost:3001. |

### Alternative — without Docker (needs Node.js)

Use this only if you already have Node.js, or prefer not to use Docker.

1. Install **Node.js 20 or newer** from [nodejs.org](https://nodejs.org/) (LTS is fine).
2. Get the Instagram ZIP (Step A above).
3. In Terminal / PowerShell, go into the project folder and run:

```bash
npm install
npm run analyze
```

4. In the menu: **Import** (paste the path to your ZIP) → **Generate** → **Dashboard**. Open the URL it prints (often http://localhost:3000 or 3001).

### Claude API key (for persona chat)

In the dashboard header, click **API key**. Paste a key from [console.anthropic.com](https://console.anthropic.com/settings/keys).

- The key is **verified with Anthropic**, then stored **encrypted** under `~/.doppelgaenger-analytics/` (not in the browser, not in git).
- After save you only ever see a masked hint (`sk-ant-…xxxx`).
- You can remove the saved key anytime from the same dialog.
- Optional: set `ANTHROPIC_API_KEY` in the environment instead (useful for Docker).

### Persona chat

1. Import + generate metrics (creates `personaProfiles.json`).
2. Open the dashboard → open a conversation → **Persona Chat** tab.
3. Pick someone from the list and message them. Replies use Claude with their style profile and real few-shot examples.

Same steps as separate commands:

```bash
npm run import -- /path/to/instagram-export.zip
npm run generate-metrics
npm run dashboard
```

---

## 🚀 Features

### **Core Analytics**
- **Message Processing**: Text analysis, sentiment scoring, emotion detection
- **Conversation Analysis**: Response patterns, turn-taking, engagement scoring
- **Media Analytics**: Photo/video sharing patterns, attachment analysis
- **Activity Patterns**: Temporal analysis, peak activity detection, communication rhythms
- **Content Analysis**: Word frequencies, URL sharing, content type classification
- **Social Dynamics**: Reaction analysis, mood correlation, conversation starters

### **Interactive Dashboard**
- **6 Specialized Tabs**: Overview, Messages, Sentiment, Media, Activity, Conversations
- **Chart Components**: Comprehensive visualizations with filtering
- **Conversation Filtering**: Focus analysis on specific conversations
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🛠 Tech Stack

- **Backend**: Node.js, TypeScript, SQLite (better-sqlite3)
- **Frontend**: Next.js 15, React 19, Tailwind CSS, Recharts
- **Data Processing**: VADER sentiment analysis (vader-sentiment), Unicode decoding
- **Testing**: Jest

## 📋 Prerequisites

**Non-technical / recommended:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) + Instagram messages ZIP (Step A above).

**Developers / without Docker:** Node.js 20 or newer (`.nvmrc` pins 23 for contributors; any Node ≥20 works) + the same Instagram ZIP.

## 🚀 Quick Start (developers)

```bash
npm install
npm run import -- <path-to-instagram-export.zip>
npm run generate-metrics
npm run dashboard
```

You can also run `npm run analyze` for the interactive menu.

### Troubleshooting a fresh setup

- **`better-sqlite3` fails to load after switching Node versions** — the native
  binding is compiled per Node ABI. Run `npm run rebuild:native` (or a full
  `npm install`) after changing Node majors.
- **Windows** — all npm scripts are cross-platform (`cross-env` handles env
  vars), and the CLI spawns the dashboard through a shell on Windows.
- **Custom locations** — set `DOPPELGANGER_DB_PATH` to relocate the SQLite
  database (default `~/.doppelgaenger-analytics/doppelgaenger-analytics.db`)
  and `DOPPELGANGER_DASH_DIR` to redirect generated JSON output.
- **Port already in use** — the CLI picks a free port and prints the real URL
  (do not assume `:3000`).
- **`postinstall` fails** — run `npm run build` then `npm ci --prefix dashboard` manually.
- **Dashboard build: `Cannot find module '../lightningcss…'`** — reinstall dashboard deps: `rm -rf dashboard/node_modules && npm ci --prefix dashboard`.
- **Dashboard says port is free but Next crashes with EADDRINUSE** — another app is on that port (e.g. something already on `:3000`). The CLI now probes `0.0.0.0` and will pick the next free port; open the URL it prints.

## 📊 Metrics Specification

The system computes **31 distinct metrics** across 10 categories:

### **Core Processing**
- **Text Metrics**: Word count, emoji detection, URL extraction
- **Sentiment Analysis**: VADER-based sentiment scoring with positive/negative/neutral classification
- **Emotion Detection**: Joy, sadness, anger, fear, surprise analysis

### **Communication Patterns**
- **Response Metrics**: Latency distribution, turn-taking analysis
- **Engagement Scoring**: Participant engagement levels and patterns
- **Conversation Analysis**: Thread depth, conversation starters, quiet periods

### **Content & Media**
- **Content Classification**: 9 content types (short text, emojis, media, etc.)
- **Media Analysis**: Photo/video sharing patterns with engagement correlation
- **URL Analysis**: Domain tracking and sharing patterns

### **Temporal Analysis**
- **Activity Patterns**: Hourly/daily/monthly activity heatmaps
- **Peak Detection**: Most active periods and communication rhythms
- **Timeline Analysis**: Sentiment trends over time

## 🗃 Data Architecture

### **Database Schema**
```sql
-- Core tables
messages (id, conversation_id, sender, timestamp_ms, content, has_photos, has_videos)
sentiment (message_id, compound, positive, negative, neutral)
text_metrics (message_id, word_count, emoji_count, url_count)
message_photos (message_id, uri, creation_timestamp)
message_videos (message_id, uri, creation_timestamp)
message_reactions (message_id, reaction, actor, timestamp)
```

### **Data Processing Pipeline**
1. **Import**: Parse Instagram JSON → SQLite database (messages, photos, videos, reactions)
2. **Process**: Run the specialized processors in `src/processors/` for metrics computation
3. **Export**: Generate JSON files into `dash-data/` for dashboard consumption
4. **Visualize**: Interactive dashboard with client-side filtering (the CLI `dashboard` command syncs `dash-data/` → `dashboard/public/data/`)

## 🎯 Key Achievements

### **Implementation Status: 100% Complete**
- ✅ **Phase 1**: 7 basic widgets implemented
- ✅ **Phase 2**: 5 complex widgets implemented  
- ✅ **Unicode Integration**: Proper handling of Instagram's malformed Unicode
- ✅ **Conversation Filtering**: All components now support filtering
- ✅ **Test Coverage**: Comprehensive test suite with 22 Unicode tests

### **Data Quality Improvements**
- **Unicode Decoding**: Fixed malformed sequences like `don\u00e2\u0080\u0099t` → `don't`
- **Media Detection**: Content-based analysis for photos, videos, and reactions
- **Response Time Calculation**: Accurate response time analysis between participants
- **Real Instagram Format**: Updated from simplified test data to actual Instagram export format
- **Comprehensive Testing**: 950k+ messages processed in 53.2 seconds

### **Dashboard Features**
- **31 Metrics**: All planned metrics implemented and visualized
- **Conversation Filtering**: Focus analysis on specific conversations
- **Export Functionality**: PNG export for presentations
- **Performance**: Fast loading with efficient data processing

## 📁 Project Structure

```
doppelganger-analytics/
├── src/
│   ├── cli/           # Command-line interface
│   ├── db/            # Database schema and client
│   ├── processors/    # 12 metrics processors
│   ├── services/      # External services (sentiment analysis)
│   └── utils/         # Unicode decoder, utilities
├── dashboard/         # Next.js dashboard application
├── tests/             # Comprehensive test suite
├── dash-data/         # Generated JSON data files
└── docs/              # Technical documentation
```

## 🔧 Development

### **Available Commands**
```bash
# One-shot / interactive
npm run analyze              # Interactive menu (import / generate / dashboard)

# Data processing
npm run import -- <zip-file>  # Import Instagram data
npm run generate-metrics      # Generate all analytics
npm run dashboard             # Sync data + start the dashboard
npm test                      # Run test suite
npm run lint                  # Lint backend + tests
npm run rebuild:native        # Rebuild better-sqlite3 after a Node change

# Development (caps imports at 2,000 messages)
npm run dev:import -- <zip-file>
npm run dev:generate

# Dashboard (run from dashboard/ if needed)
npm run dev                   # Development server
npm run build                 # Production build (lint-enforced)
npm run lint                  # Lint dashboard
```

### **Adding New Metrics**
1. Create processor in `src/processors/`
2. Add to `src/generator.ts`
3. Create dashboard component
4. Add tests in `tests/`

### **Testing**
- **Test Fixtures**: `tests/fixtures/basic-messages.json` with known data
- **Expected Outputs**: Pre-calculated results for validation
- **Unicode Tests**: 22 tests for Unicode decoding
- **Database Tests**: Full SQLite integration testing

## 📊 Dashboard Tabs

### **Overview Tab**
- **Hero Metrics Dashboard**: Large message analytics card with integrated sub-metrics, media overview with breakdown, and communication style assessment
- **Participant Analytics Grid**: Four sophisticated cards showing top contributors, fast responders, emoji champions, and media sharers with gradient headers and ranked displays
- **Message Trends**: Historical message volume visualization
- **Peak Activity Analysis**: Most active hours and communication patterns
- **Modern Design**: Gradient backgrounds, enhanced typography, hover effects, and improved visual hierarchy

### **Messages & Content Tab**  
- Message volume, word frequencies, URL analysis, content types, important messages

### **Sentiment & Emotions Tab**
- Sentiment by sender, emotion distribution, sentiment timeline, mood correlation

### **Media & Reactions Tab**
- Media sharing trends, reaction analysis, attachment types, engagement correlation

### **Activity Patterns Tab**
- Activity heatmaps, response times, communication frequency, peak detection

### **Conversations & Threads Tab**
- Thread visualization, turn-taking analysis, engagement scoring, conversation starters

## 🔮 Recent Updates

### **Overview Tab Redesign**
- **Hero Metrics Dashboard**: Modern 3-section layout with prominent message analytics, media overview, and communication style cards
- **Sophisticated Visual Design**: Gradient backgrounds, proper typography hierarchy, and enhanced spacing
- **Participant Analytics Grid**: Four redesigned cards with gradient headers, numbered rankings, and improved data presentation
- **Consolidated Information Architecture**: Eliminated redundancy by integrating insights and media analysis into main sections
- **Enhanced User Experience**: Hover effects, better responsive design, and cleaner visual flow

### **Database Schema Optimization**
- **Content-Based Media Detection**: Robust analysis of photos, videos, reactions through message content
- **Response Time Calculation**: Accurate inter-participant response time analysis
- **Enhanced Media Processor**: Fixed compatibility with actual database schema

### **Instagram Format Support** 
- Updated to handle real Instagram export format vs. simplified test data
- Added support for photos, videos, reactions through content analysis
- Enhanced Unicode decoding for proper emoji and text handling

### **Conversation Filtering**
- Implemented across 25/28 dashboard components
- Real-time filtering without server requests
- Clear UI indicators when filtering is active

### **Performance Optimization**
- 53.2 seconds to process 950k+ messages
- Efficient client-side data aggregation
- Optimized database queries and indexing

## 📝 License

Part of the Doppelgänger project for Instagram message analytics.

---

**Status**: Production-ready with comprehensive testing and documentation
**Coverage**: 31 metrics across 6 dashboard tabs with conversation filtering
**Performance**: Optimized for large datasets (1M+ messages) 

## 🔄 Progress Reporting

### **Standardized Terminal Output**

All analytics generation now uses a unified progress reporting system:

- **Progress Bars**: For batch operations (message processing, sentiment analysis)
- **Spinners**: For database operations and file I/O
- **Real-time Updates**: Live status messages during processing
- **Error Handling**: Consistent error reporting across all processors

### **Example Output**

```
🚀 Running FAST MODE - optimized analytics generation
Overall Progress [████████████████████████████████] 21/21 100% 0.0s
✅ Text metrics computed
✅ Message length distribution computed
✅ Sentiment analysis computed
✅ Conversations analyzed
✅ Additional metrics computed
✅ Attachment type metrics computed
✅ Advanced metrics computed
✅ Enhanced emotions generated
✅ Insight metrics computed
✅ Enhanced time metrics generated
✅ Enhanced media data generated
✅ Content type metrics computed
✅ Sentiment timeline metrics computed
✅ Reaction metrics computed
✅ Mood correlation metrics computed
✅ Media engagement metrics computed
✅ Thread analysis metrics computed
✅ Turn-taking analysis metrics computed
✅ Engagement scoring metrics computed
✅ Conversation starter analysis metrics computed
✅ Emotional peaks & valleys metrics computed
✅ All metrics computed successfully in 57.0 seconds!
```

This provides clear, consistent feedback during the entire analytics generation process. 