# Instagram Message Format Update

**⚠️ IMPLEMENTATION NOTE**: This document describes a planned database schema approach that was not fully implemented. The current system uses content analysis to detect media (photos, videos, reactions) rather than separate database tables. See `docs/DEVELOPMENT.md` for the actual current database schema and implementation details.

## Overview

This document outlines the planned updates to handle the actual Instagram message export format, including proper Unicode decoding, reactions, photos, videos, and improved test data.

## Changes Made

### 1. Updated Test Data Format (`tests/fixtures/basic-messages.json`)

**Previous Format (Incorrect):**
```json
{
  "id": 1,
  "sender": "Alice Johnson", 
  "content": "Hello world!"
}
```

**New Format (Correct Instagram Format):**
```json
{
  "sender_name": "Tia Shannon",
  "timestamp_ms": 1735409465834,
  "content": "I am just exhausted and I don\\u00e2\\u0080\\u0099t know what to do",
  "is_geoblocked_for_viewer": false,
  "is_unsent_image_by_messenger_kid_parent": false
}
```

### 2. Enhanced Importer (`src/importer.ts`)

#### Updated RawMessage Interface
```typescript
export interface RawMessage {
  sender_name: string;
  timestamp_ms: number;
  content?: string;
  photos?: Array<{
    uri: string;
    creation_timestamp: number;
    backup_uri?: string;
  }>;
  videos?: Array<{
    uri: string;
    creation_timestamp: number;
    backup_uri?: string;
  }>;
  reactions?: Array<{
    reaction: string;
    actor: string;
    timestamp: number;
  }>;
  is_geoblocked_for_viewer: boolean;
  is_unsent_image_by_messenger_kid_parent: boolean;
}
```

#### New Database Tables
- `message_photos`: Stores photo metadata
- `message_videos`: Stores video metadata  
- `message_reactions`: Stores reaction data
- Added `has_photos` and `has_videos` flags to messages table

#### Enhanced Processing Logic
- Processes photos, videos, and reactions from structured Instagram data
- Generates descriptive content for media messages (e.g., "User sent 1 photo")
- Applies Unicode decoding to all text fields during import
- Stores structured media data in dedicated tables

### 3. Updated Processors

#### Reaction Metrics (`src/processors/reactionMetrics.ts`)
- **Before**: Parsed reaction text from message content
- **After**: Reads from dedicated `message_reactions` table
- More accurate reaction counting and emoji handling
- Proper Unicode decoding of reaction emojis

#### Enhanced Media Processor (`src/processors/enhancedMediaProcessor.ts`)
- Added `getMediaFromDatabase()` function to get actual counts from database
- Updated to use `has_photos` and `has_videos` flags
- More accurate media statistics

### 4. Comprehensive Test Suite (`tests/instagram-preprocessing.test.ts`)

#### Unicode Decoding Tests
- Malformed Unicode apostrophes: `don\\u00e2\\u0080\\u0099t` → `don't`
- Emoji decoding: `\\u00f0\\u009f\\u0098\\u00ae` → `😮`
- Multiple sequences in one string
- Mixed normal and encoded text

#### Instagram Format Processing Tests
- Text message processing
- Photo message processing with metadata storage
- Video message processing with backup URLs
- Reaction processing with Unicode decoding
- Multiple media files in one message
- Edge cases (empty content, null values)

### 5. Updated Expected Outputs (`tests/fixtures/expected-outputs.json`)

Aligned test expectations with the new Instagram format:
- Proper Unicode-decoded text in expected results
- Updated message counts and content types
- Reaction and media statistics

## Real Instagram Message Examples

### Text with Unicode Encoding
```json
{
  "sender_name": "Finn Morris",
  "timestamp_ms": 1735476689085,
  "content": "that\\u00e2\\u0080\\u0099s my job",
  "is_geoblocked_for_viewer": false,
  "is_unsent_image_by_messenger_kid_parent": false
}
```

### Video Message
```json
{
  "sender_name": "Finn Morris",
  "timestamp_ms": 1735476194707,
  "videos": [
    {
      "uri": "your_instagram_activity/messages/inbox/user/videos/video.mp4",
      "creation_timestamp": 1735476188,
      "backup_uri": "https://video.fper9-1.fna.fbcdn.net/v/video.mp4?..."
    }
  ],
  "is_geoblocked_for_viewer": false,
  "is_unsent_image_by_messenger_kid_parent": false
}
```

### Message with Reaction
```json
{
  "sender_name": "Test User",
  "timestamp_ms": 1735409465834,
  "content": "I have a car",
  "reactions": [
    {
      "reaction": "\\u00f0\\u009f\\u0098\\u00ae",
      "actor": "Another User", 
      "timestamp": 1735409465835
    }
  ],
  "is_geoblocked_for_viewer": false,
  "is_unsent_image_by_messenger_kid_parent": false
}
```

## Database Schema Changes

### New Tables
```sql
CREATE TABLE message_photos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  message_id INTEGER NOT NULL,
  uri TEXT NOT NULL,
  creation_timestamp INTEGER,
  backup_uri TEXT,
  FOREIGN KEY (message_id) REFERENCES messages(id)
);

CREATE TABLE message_videos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  message_id INTEGER NOT NULL,
  uri TEXT NOT NULL,
  creation_timestamp INTEGER,
  backup_uri TEXT,
  FOREIGN KEY (message_id) REFERENCES messages(id)
);

CREATE TABLE message_reactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  message_id INTEGER NOT NULL,
  reaction TEXT NOT NULL,
  actor TEXT NOT NULL,
  timestamp INTEGER,
  FOREIGN KEY (message_id) REFERENCES messages(id)
);
```

### Updated Messages Table
```sql
ALTER TABLE messages ADD COLUMN has_photos INTEGER DEFAULT 0;
ALTER TABLE messages ADD COLUMN has_videos INTEGER DEFAULT 0;
```

## Benefits

1. **Accurate Data Processing**: Handles real Instagram export format
2. **Unicode Correctness**: Proper decoding of malformed Unicode sequences
3. **Rich Media Support**: Structured storage and processing of photos/videos
4. **Reaction Analytics**: Dedicated reaction tracking and analysis
5. **Better Testing**: Comprehensive test coverage with real data patterns
6. **Maintainability**: Clear separation of concerns and proper data modeling

## Migration Notes

- Existing databases will be automatically updated with new schema
- Unicode decoding is applied during import, so existing data remains unchanged
- New processors will work with both old and new data formats
- Test fixtures updated to reflect real Instagram message structure

## Testing

Run the comprehensive test suite:
```bash
npm test -- unicode-decoding.test.ts  # Unicode decoding tests
npm test                               # All tests
```

The system now properly handles:
- ✅ Unicode decoding (apostrophes, emojis, special characters)
- ✅ Photo messages with metadata
- ✅ Video messages with backup URLs
- ✅ Reactions with proper emoji handling
- ✅ Multiple media files per message
- ✅ Edge cases and error handling 