Below is a first-cut design document. I’ve broken it into:

* **1. Overview & Goals**
* **2. High-Level Architecture**
* **3. Feature & Module Breakdown**
* **4. Data Flow & Storage**
* **5. Dashboarding Options**
* **6. Code Organization**
* **7. Step-by-Step Roadmap**

---

## 1. Overview & Goals

You’ll ship a single-user, CLI-driven tool that:

1. **Imports** an Instagram zip (folders of `message_*.json`)
2. **Processes** text → metrics (word counts, sentiment, granular emotions, response times, etc.)
3. **Stores** everything in a local SQLite database for fast queries
4. **Generates** interactive dashboards (static site or self-hosted) covering:

   * Message frequency (daily/weekly/monthly/yearly)
   * Sentiment and emotion over time
   * Heatmaps (hour-of-day × day-of-week)
   * Conversation patterns (turns, initiations, response latency)
   * Message-type breakdowns (URLs, emojis, replies)

**Non-Goals**: multi-user auth, cloud hosting, real-time sync.

Quality priorities: reliability (robust parsing + retries), simplicity (straightforward CLI & clear folder structure), and high-quality UX (fast, responsive dashboards with export buttons).

---

## 2. High-Level Architecture

```
┌──────────────┐    unzip    ┌───────────────┐    ingest     ┌────────┐
│ Instagram    │ ─────────► │ CLI Importer  │ ────────────►│ SQLite │
│ ZIP (JSONs)  │            │ (Node.js)     │              │ DB     │
└──────────────┘            └───────────────┘              └────────┘
                                      │
                                      │ process metrics
                                      ▼
                           ┌────────────────────────┐
                           │ Metrics Generator      │
                           │ (JS + optional Python) │
                           └────────────────────────┘
                                      │
                  export JSON/CSV     │
                                      ▼
┌───────────────────────┐       ┌─────────────────────┐
│ Static Dashboard     │◄──────│ Data Artifact Dir   │
│ (Next.js / D3 / ECharts)      │ (JSON/CSV files)    │
└───────────────────────┘       └─────────────────────┘
```

Alternatively, you can push metrics to InfluxDB and point Grafana at it—see §5.

---

## 3. Feature & Module Breakdown

| Component               | Responsibilities                                                                  |
| ----------------------- | --------------------------------------------------------------------------------- |
| **CLI Importer**        | • Unzip & traverse `inbox/*/message_*.json`                                       |
|                         | • Parse timestamp\_ms → JS `Date`                                                 |
|                         | • Normalize & clean text (decode emojis, strip URLs)                              |
|                         | • Batch‐insert raw messages into SQLite                                           |
| **Metrics Processor**   | • Word/emoji/URL counts                                                           |
|                         | • Sentiment (compound, pos/neg/neutral via Hugging Face or JS library)            |
|                         | • Granular emotions (Ekman + a few more)                                          |
|                         | • Conversation metrics (response latency, turns, initiations, quiet periods)      |
|                         | • Caching/incremental re-runs if source unchanged                                 |
| **Storage Layer**       | • SQLite schema with tables: `messages`, `sentiment`, `emotions`, `conversations` |
|                         | • Indexes on `timestamp`, `sender`, `conversation_id`                             |
| **Dashboard Generator** | • Read aggregated data (SQL or pre-exported JSON)                                 |
|                         | • Build charts: time series, heatmap, network graph                               |
|                         | • “Export PNG/PDF” for each chart                                                 |
| **CLI Runner**          | • `insta-dash import <zip>`                                                       |
|                         | • `insta-dash generate-dashboards [--output dir]`                                 |

---

## 4. Data Flow & Storage

1. **Import**

   * CLI unzips into a temp dir
   * Recursively reads each conversation folder
   * For each `message_X.json`, parse JSON and insert messages

2. **Schema**

   ```sql
   CREATE TABLE messages (
     id INTEGER PRIMARY KEY,
     conversation_id TEXT,
     sender TEXT,
     timestamp INTEGER,
     content TEXT
   );
   CREATE TABLE sentiment (
     message_id INTEGER, compound REAL, pos REAL, neg REAL, neu REAL
   );
   CREATE TABLE emotions (
     message_id INTEGER, emotion TEXT, score REAL
   );
   -- plus tables for response_metrics, turn_metrics, etc.
   ```

3. **Processing**

   * Metrics processor queries `messages` in time order
   * Populates `sentiment`, `emotions`, and conversation tables

4. **Export**

   * Runs SQL aggregations (e.g. `GROUP BY strftime('%Y-%m', timestamp)`)
   * Dumps results as JSON in `./dash-data/`

---

## 5. Dashboarding Options

### A. Static Next.js Site

* **Pros**: zero-config, just open `index.html` or serve via `npm run start`.
* **Cons**: custom code for charts (we’ll use ECharts or Recharts).

### B. Grafana + InfluxDB

* CLI pushes metrics into local InfluxDB (via its HTTP API).
* User runs Grafana (Docker or binary) pointed at that DB.
* **Pros**: rich visualization library, panel export built in.
* **Cons**: heavier install, container or service needed.

*Recommendation*: start with **Static Next.js** for maximum simplicity. Provide an `export-to-grafana` script later if desired.

---

## 6. Code Organization

```
insta-dash/
├─ src/
│  ├─ cli/
│  │   └─ index.ts           # yargs commands: import, generate
│  ├─ db/
│  │   ├─ schema.ts          # SQLite DDL & migrations
│  │   └─ client.ts          # open/close connection
│  ├─ processors/
│  │   ├─ textProcessor.ts   # word count, emoji, URL
│  │   ├─ sentiment.ts       # wrap HF or JS lib
│  │   └─ conversation.ts    # response times, turns
│  ├─ exporters/
│  │   ├─ sqliteToJson.ts    # SQL → JSON dumps
│  │   └─ influxExporter.ts  # optional Influx push
│  └─ dashboard/
│      └─ nextjs/            # Next.js app reading ./dash-data
├─ tests/
│  └─ sample-data/           # small JSON zips for CI
├─ .eslintrc.js
├─ .prettierrc
├─ package.json
└─ README.md
```

---

## 7. Step-by-Step Roadmap

1. **Project Scaffold**

   * Initialize `npm init`, TypeScript, ESLint+Prettier
   * Set up `src/`, `tests/sample-data`

2. **SQLite Schema & Client**

   * Write migrations for `messages`, `sentiment`, `emotions`, etc.
   * Write a small wrapper to open/close DB.

3. **CLI Importer**

   * `insta-dash import <zip>`
   * Unzip, traverse, parse JSON, insert raw messages
   * Add basic logging + error handling

4. **Text & URL Metrics**

   * Implement `textProcessor.process(content)` → `{ wordCount, emojiCount, urlCount }`
   * Write unit tests against small samples

5. **Sentiment & Emotion**

   * Evaluate JS vs Python approach
   * Integrate chosen library, write tests for correctness

6. **Conversation Analysis**

   * Compute response times, conversation turns
   * Populate DB tables, write integration tests

7. **Exporters**

   * SQL → JSON exporter for each dashboard metric
   * CLI command `generate-dashboards`

8. **Next.js Dashboard**

   * Bootstrap with `create-next-app`
   * Load `dash-data/*.json`, render charts (ECharts/Recharts)
   * Add filters, drilldowns, heatmap components
   * “Export PNG/PDF” functionality via `html-to-image`

9. **Documentation & Packaging**

   * Fill `README.md` with install & usage
   * Publish to npm with `bin` entry for `insta-dash`

10. **CI & Tests**

    * Configure GitHub Actions to run `npm test` on pushes
    * Include sample JSON for end-to-end test of import→dashboard data dump

11. **Optional Grafana Integration**

    * Write `exporters/influxExporter.ts`
    * Provide `docker-compose.yml` to spin up InfluxDB + Grafana

---

With this in hand, you can start on the scaffold (step 1) and we’ll iterate as you bang out the CLI importer and DB schema. Let me know where you want to dive in first!
