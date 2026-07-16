# Project Design Document

## Overview & Goals

Doppelgänger Analytics is designed as a single-user, CLI-driven tool that transforms Instagram message exports into comprehensive analytics dashboards.

### Core Objectives
1. **Import** Instagram ZIP exports (folders of `message_*.json`)
2. **Process** text → metrics (sentiment, emotions, response times, engagement patterns)
3. **Store** everything in a local SQLite database for fast queries
4. **Generate** interactive dashboards covering all aspects of communication patterns

### Design Principles
- **Reliability**: Robust parsing with comprehensive error handling
- **Simplicity**: Straightforward CLI with clear folder structure
- **Performance**: Fast processing of large datasets (1M+ messages)
- **Quality**: High-quality UX with responsive dashboards and export capabilities

## Architecture Decision Rationale

### Database Choice: SQLite
**Decision**: Use SQLite for local storage
**Rationale**: 
- Zero-configuration setup for single-user application
- Excellent performance for read-heavy analytics workloads
- Built-in full-text search capabilities
- No external dependencies or server setup required
- Perfect for datasets up to several GB

### Processing Pipeline: Batch Processing
**Decision**: Process all metrics in batch operations rather than real-time
**Rationale**:
- Instagram exports are static datasets (not streaming)
- Batch processing allows for complex cross-message analysis
- Better performance for large datasets
- Simpler error handling and recovery

### Dashboard Technology: Static Generation
**Decision**: Generate static JSON files for dashboard consumption
**Rationale**:
- Fast loading times (no database queries during visualization)
- Easy deployment (just serve static files)
- Offline capability once data is generated
- Simple caching and CDN distribution

### Sentiment Analysis: VADER + Custom Emotion Detection
**Decision**: Use VADER for sentiment, custom logic for emotions
**Rationale**:
- VADER specifically designed for social media text
- Handles emojis, slang, and informal communication well
- Custom emotion detection allows for domain-specific tuning
- No external API dependencies (works offline)

## Data Flow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Instagram ZIP   │ -> │ CLI Importer    │ -> │ SQLite Database │
│ (JSON files)    │    │ (Unicode decode)│    │ (Normalized)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Static Dashboard│ <- │ JSON Exports    │ <- │ Metrics Engine  │
│ (Next.js)       │    │ (dash-data/)    │    │ (12 Processors) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Phase 1: Import & Normalization
- Parse Instagram's non-standard JSON format
- Handle malformed Unicode sequences
- Normalize timestamps and participant names
- Store in relational schema for efficient querying

### Phase 2: Metrics Computation
- Process data through specialized analytics processors
- Compute sentiment, emotions, engagement patterns
- Generate conversation-level and participant-level metrics
- Handle edge cases (empty messages, system notifications)

### Phase 3: Dashboard Generation
- Export processed metrics as JSON files
- Generate static dashboard with interactive visualizations
- Support conversation filtering and data export
- Optimize for mobile and desktop viewing

## Scalability Considerations

### Dataset Size Limits
- **Target**: 1-5 million messages (typical heavy Instagram user)
- **Processing Time**: <60 seconds for full analytics pipeline
- **Memory Usage**: <1GB peak during processing
- **Storage**: ~100MB for processed data + dashboard

### Performance Optimizations
- **Database Indexing**: Strategic indexes on timestamp, sender, conversation_id
- **Batch Processing**: Process messages in chunks to manage memory
- **Lazy Loading**: Dashboard components load data on-demand
- **Client-side Filtering**: Real-time filtering without server requests

## Security & Privacy

### Local-Only Processing
- **No Cloud Dependencies**: All processing happens locally
- **No Data Transmission**: Instagram data never leaves user's machine
- **No External APIs**: Sentiment analysis runs locally
- **No User Tracking**: Dashboard contains no analytics or tracking

### Data Handling
- **Original Data Preservation**: Instagram export remains unchanged
- **Secure Storage**: SQLite database stored locally with standard file permissions
- **Optional Cleanup**: Users can delete processed data at any time
- **Export Control**: Users control what data is exported from dashboard

## Extension Points

### Adding New Metrics
The system is designed for easy extension:
1. Create new processor in `src/processors/`
2. Add to metrics generation pipeline
3. Create corresponding dashboard component
4. Add test coverage

### Custom Visualizations
Dashboard supports additional chart types:
- Integration with Recharts library
- Custom D3.js visualizations
- Export to external tools (Grafana, Tableau)

### Alternative Data Sources
Architecture supports other messaging platforms:
- WhatsApp exports
- Telegram exports
- Discord exports
- Any JSON-based message format

## Technology Choices

### Backend Stack
- **Node.js + TypeScript**: Strong typing, excellent JSON handling
- **SQLite**: Zero-config database with excellent performance
- **Jest**: Comprehensive testing framework

### Frontend Stack  
- **Next.js 14**: Modern React framework with excellent performance
- **Tailwind CSS**: Utility-first styling for rapid development
- **Recharts**: React-based charting library with good accessibility

### Development Tools
- **ESLint + Prettier**: Code quality and formatting
- **TypeScript**: Type safety throughout the application
- **Git**: Version control with conventional commits

---

**Design Status**: Implemented and validated with real-world datasets
**Architecture**: Proven to handle 1M+ message datasets efficiently
