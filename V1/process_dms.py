#!/usr/bin/env python
"""
Instagram Message Analytics Processor

A user-friendly tool for analyzing Instagram message data and visualizing it in Grafana.
"""

import os
import json
import logging
import time
import hashlib
import sys
import secrets
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from questionary import Choice
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.exceptions import InfluxDBError
from influxdb_client.rest import ApiException
from dotenv import load_dotenv
from tqdm import tqdm
import re
import html
import unicodedata
import argparse
import questionary
from questionary import Style

# Configure custom style for interactive prompts
custom_style = Style([
    ('qmark', 'fg:#34ebde bold'),     # question mark color
    ('question', 'fg:#ffffff bold'),  # question text
    ('answer', 'fg:#34ebde bg:#000000'),  # answer text
    ('selected', 'fg:#cc5454'),       # selected item style
    ('pointer', 'fg:#ffd700 bold'),   # pointer style
    ('highlighted', 'fg:#ffffff bg:#000000 bold'),
])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('processing.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AppConfig:
    """Central configuration with user-friendly defaults"""
    def __init__(self):
        self.influx_url = os.getenv("INFLUX_URL", "http://localhost:8086")
        self.influx_token = os.getenv("INFLUXDB_ADMIN_TOKEN")
        self.influx_org = os.getenv("INFLUXDB_ORG", "instagram_analytics")
        self.influx_bucket = os.getenv("INFLUXDB_BUCKET", "instagram_data")
        self.batch_size = int(os.getenv("BATCH_SIZE", 500))
        self.test_mode = os.getenv("TEST_MODE", "false").lower() == "false"
        self.max_word_length = 128
        self.data_path = os.getenv("DATA_PATH", "./data/instagram_messages")

config = AppConfig()

def interactive_setup():
    """Guided interactive setup for first-time users"""
    print("\n🎉 Welcome to Instagram Message Analytics Setup! 🎉\n")
    
    # Data location discovery
    data_path = questionary.path(
        "📂 Where is your Instagram messages folder?",
        default=str(Path(config.data_path).resolve()),
        validate=lambda val: Path(val).exists() or "Path does not exist"
    ).ask()
    
    # Service configuration
    config.influx_org = questionary.text(
        "🏢 Organization name for InfluxDB:",
        default=config.influx_org
    ).ask()
    
    config.influx_bucket = questionary.text(
        "📦 Bucket name for message data:",
        default=config.influx_bucket
    ).ask()
    
    # Generate secure credentials
    password = questionary.password(
        "🔑 Set admin password for InfluxDB/Grafana:",
        validate=lambda val: len(val) >= 8 or "Minimum 8 characters"
    ).ask()
    
    # Save configuration
    env_content = f"""# Auto-generated configuration
INFLUX_URL={config.influx_url}
INFLUXDB_ADMIN_TOKEN={secrets.token_urlsafe(32)}
INFLUXDB_ORG={config.influx_org}
INFLUXDB_BUCKET={config.influx_bucket}
GRAFANA_PASSWORD={password}
DATA_PATH={data_path}
    """
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("\n✅ Setup complete! Configuration saved to .env")
    print("🚀 Run './run_with_docker.sh' to start the system\n")

def validate_data_path(path: str) -> Path:
    """Validate Instagram data folder structure with user feedback"""
    path = Path(path)
    required_structure = ["messages", "messages/inbox"]
    
    for folder in required_structure:
        if not (path / folder).exists():
            logger.error(f"❌ Missing required folder: {folder}")
            response = questionary.confirm(
                f"🔍 Couldn't find '{folder}'. Should we try another location?",
                default=False
            ).ask()
            
            if response:
                new_path = questionary.path(
                    "📂 Enter correct path to Instagram data folder:",
                    validate=lambda val: Path(val).exists()
                ).ask()
                return validate_data_path(new_path)
            
            logger.error("⚠️ Invalid data structure. Please ensure you unzipped the Instagram export correctly.")
            sys.exit(1)
    
    return path

class InfluxDBManager:
    """User-friendly InfluxDB connection manager"""
    def __init__(self):
        self.client = InfluxDBClient(
            url=config.influx_url,
            token=config.influx_token,
            org=config.influx_org,
            timeout=30_000
        )
        self._write_api = None
        self._health_check()

    def _health_check(self):
        """Perform user-friendly health check"""
        try:
            ready = self.client.ping()
            if not ready:
                logger.error("⏳ InfluxDB is not ready. Waiting 10 seconds...")
                time.sleep(10)
                self._health_check()
        except Exception as e:
            logger.error(f"❌ Failed to connect to InfluxDB: {str(e)}")
            logger.info("💡 Check if InfluxDB is running and .env is configured")
            sys.exit(1)

    def get_writer(self):
        """Get a configured InfluxWriter instance"""
        if not self._write_api:
            self._write_api = self.client.write_api(
                write_options=WriteOptions(
                    batch_size=config.batch_size,
                    flush_interval=10_000,
                    jitter_interval=2_000
                )
            )
        return InfluxWriter(self._write_api)

    def close(self):
        """Close the client connection"""
        if self._write_api:
            try:
                self._write_api.close()
            except Exception as e:
                logger.error(f"Error closing write API: {str(e)}")
        try:
            self.client.close()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error closing client: {str(e)}")

    def clear_bucket(self):
        """Clear all data from the bucket using correct predicate syntax"""
        try:
            delete_api = self.client.delete_api()
            delete_api.delete(
                start=datetime(1970, 1, 1),
                stop=datetime(2100, 1, 1),
                predicate='_measurement=""',
                bucket=config.influx_bucket,
                org=config.influx_org
            )
            time.sleep(2)
            logger.info(f"Successfully cleared bucket: {config.influx_bucket}")
            
            query_api = self.client.query_api()
            result = query_api.query(f'from(bucket:"{config.influx_bucket}") |> range(start:-1h) |> count()')
            if not any(result):
                logger.info("Bucket clearance verified")
            else:
                logger.warning("Bucket clearance verification failed")
                
        except ApiException as e:
            logger.error(f"Failed to clear bucket: {e.reason}")
            raise
        except Exception as e:
            logger.error(f"Error clearing bucket: {str(e)}")
            raise

class DataValidator:
    """Validates message structure and content"""
    @staticmethod
    def validate_message(msg: Dict) -> None:
        """Ensure message contains required fields with proper types"""
        required_fields = {
            'sender_name': str,
            'timestamp_ms': (int, float)
        }
        
        for field, field_type in required_fields.items():
            if field not in msg:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(msg[field], field_type):
                raise TypeError(f"Invalid type for {field}: {type(msg[field])}")

        if 'content' in msg and not isinstance(msg['content'], str):
            raise TypeError("Content must be string")

class InfluxWriter:
    """Handles all InfluxDB write operations with retries"""
    def __init__(self, write_api):
        self.write_api = write_api
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def write_point(self, point: Point) -> None:
        """Write single data point with retry logic"""
        try:
            self.write_api.write(
                bucket=config.influx_bucket,
                org=config.influx_org,
                record=point
            )
        except ApiException as e:
            logger.error(f"InfluxDB API error: {e.reason}")
            raise
        except InfluxDBError as e:
            logger.error(f"InfluxDB error: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def write_batch(self, points: List[Point]) -> None:
        """Write batch of points with retry logic"""
        try:
            self.write_api.write(
                bucket=config.influx_bucket,
                org=config.influx_org,
                record=points
            )
        except ApiException as e:
            logger.error(f"InfluxDB API error: {e.reason}")
            logger.debug(f"Failed batch size: {len(points)}")
            raise
        except InfluxDBError as e:
            logger.error(f"InfluxDB error: {str(e)}")
            raise

class MessageProcessor:
    def __init__(self):
        # Add timestamp validation ranges
        self.min_timestamp = int(datetime(2000, 1, 1).timestamp() * 1000)  # Jan 1, 2000
        self.max_timestamp = int(datetime(2038, 1, 1).timestamp() * 1000)  # Jan 1, 2038
        
        self._setup_nlp()
        self._prepare_regex()
        self._load_emotion_lexicon()

    def _prepare_regex(self):
        """Compile regex patterns for text processing"""
        # Updated emoji pattern
        self.emoji_pattern = re.compile(
            r"["
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F700-\U0001F77F"  # alchemical symbols
            u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251" 
            u"\U0001F004-\U0001F0CF"
            "]+", flags=re.UNICODE
        )
        
        # URL pattern remains the same
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    def _setup_nlp(self):
        """Initialize NLP components with progress tracking"""
        try:
            # Set up progress bar for NLP initialization
            with tqdm(total=3, desc="📚 Initializing language tools") as pbar:
                # Download required NLTK datasets
                nltk.download('vader_lexicon', quiet=True)
                pbar.update(1)
                nltk.download('opinion_lexicon', quiet=True)
                pbar.update(1)
                
                # Initialize sentiment analyzer
                self.sia = SentimentIntensityAnalyzer()
                pbar.update(1)
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize NLP tools: {str(e)}")
            logger.info("ℹ️ Try installing NLTK data with: python -m nltk.downloader all")
            sys.exit(1)

    def validate_timestamp(self, timestamp_ms: int) -> int:
            """Validate and convert timestamp with proper bounds checking"""
            try:
                ts = int(timestamp_ms)
                if not self.min_timestamp <= ts <= self.max_timestamp:
                    raise ValueError(f"Timestamp {ts} out of valid range (2000-2037)")
                return ts
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid timestamp {timestamp_ms}: {str(e)}")
                raise

    def process_text(self, text: str) -> Dict:
        if not text:
            return {
                'emoji_count': 0,
                'url_count': 0,
                'cleaned_text': '',
                'length_bucket': '0',
                'word_count': 0
            }

        try:
            # Convert JSON-encoded UTF-8 bytes to proper emojis
            decoded = text.encode('latin-1').decode('utf-8')
        except UnicodeEncodeError:
            decoded = text  # Fallback for non-latin-1 encodable text
        except Exception as e:
            logger.warning(f"Text decoding error: {str(e)}")
            decoded = text

        # Normalize and clean the text
        decoded = unicodedata.normalize('NFKC', decoded)
        decoded = html.unescape(decoded)
        
        # Remove remaining non-printable characters
        decoded = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', decoded)
        
        words = decoded.split()
        return {
            'emoji_count': len(self.emoji_pattern.findall(decoded)),
            'url_count': len(self.url_pattern.findall(decoded)),
            'cleaned_text': decoded,
            'length_bucket': self._get_length_bucket(len(words)),
            'word_count': len(words)
        }

    def _get_length_bucket(self, word_count: int) -> str:
        """Bucket message lengths for heatmap visualization"""
        if word_count == 0: return "0"
        if word_count <= 5: return "1-5"
        if word_count <= 10: return "6-10"
        if word_count <= 20: return "11-20"
        return "20+"

    def process_media(self, media_list: List, media_type: str) -> Dict:
        """Extract media metadata"""
        return {
            'count': len(media_list),
            'timestamps': [m.get('creation_timestamp') for m in media_list],
            'uris': [m.get('uri') for m in media_list]
        }
    
    def _load_emotion_lexicon(self):
        """Load NRC Emotion Lexicon with validation"""
        self.emotion_lexicon = defaultdict(list)  # Initialize even if loading fails
        try:
            with open('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    # Validate line structure
                    if len(parts) != 3:
                        logger.warning(f"Skipping invalid lexicon entry: {line}")
                        continue
                    word, emotion, value = parts
                    if value == '1':
                        self.emotion_lexicon[word].append(emotion)
        except FileNotFoundError:
            logger.warning("Emotion lexicon file not found. Emotion detection disabled.")
        return self.emotion_lexicon
    
    def analyze_sentiment(self, text: str) -> dict:
        """Get detailed sentiment scores"""
        scores = self.sia.polarity_scores(text)
        return {
            'sentiment': scores['compound'],
            'positive': scores['pos'],
            'neutral': scores['neu'],
            'negative': scores['neg'],
            'emotions': self._detect_emotions(text)
        }
    
    def _detect_emotions(self, text: str) -> dict:
        """Detect emotional content using lexicon"""
        emotion_counts = defaultdict(int)
        for word in text.split():
            # Access the emotion_lexicon dictionary, not the loading method
            for emotion in self.emotion_lexicon.get(word.lower(), []):
                emotion_counts[emotion] += 1
        return dict(emotion_counts)
    

class ConversationAnalyzer:
    """Analyzes conversation dynamics and patterns"""
    def __init__(self, writer: InfluxWriter):
        self.writer = writer
        self.active_threads = []
        self.current_thread = []
        self.last_timestamp = None
        self.last_sender = None
        self.response_times = []
        self.turn_transitions = defaultdict(int)
        self.initiations = []
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.total_words_processed = 0

        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.total_words = 0


    def process_message(self, msg: Dict, dt: datetime) -> List[Point]:
        """Track conversation patterns and aggregate words"""
        points = []
        sender = msg['sender_name']
        content = msg.get('content', '')
        
        # Existing conversation analysis logic
        if self.last_timestamp is not None:
            time_diff = (msg['timestamp_ms'] - self.last_timestamp) / 1000
            
            if time_diff > 86400:
                points.append(Point("conversation_gaps")
                    .tag("initiator", sender)
                    .field("duration_hours", time_diff/3600)
                    .time(dt))
                
            if self.last_sender != sender:
                self.response_times.append(time_diff)
                self.turn_transitions[(self.last_sender, sender)] += 1
                points.extend([
                    Point("turn_metrics")
                        .tag("previous_sender", self.last_sender or "None")
                        .tag("current_sender", sender)
                        .field("count", 1)
                        .time(dt),
                    Point("response_metrics")
                        .tag("responder", sender)
                        .field("response_time_seconds", float(time_diff))
                        .time(dt)
                ])

        # Thread management
        if self.last_timestamp is None or (msg['timestamp_ms'] - self.last_timestamp) > 3600 * 1000:
            if self.current_thread:
                self.active_threads.append({
                    'start': self.current_thread[0]['timestamp_ms'],
                    'end': self.current_thread[-1]['timestamp_ms'],
                    'participants': list({m['sender_name'] for m in self.current_thread}),
                    'message_count': len(self.current_thread)
                })
                self.current_thread = []
                
        self.current_thread.append(msg)
        self.last_timestamp = msg['timestamp_ms']
        self.last_sender = sender

        # Word processing with aggregation
        words = self.clean_and_extract_words(content)
        for word in words:
            self.word_counts[sender][word] += 1
            self.total_words_processed += 1

        return points

    def clean_and_extract_words(self, text: str) -> List[str]:
        """Robust text cleaning pipeline"""
        if not text:
            return []
            
        try:
            # Multi-stage cleaning
            decoded = text.encode('latin-1', 'ignore').decode('utf-8', 'ignore')
            decoded = html.unescape(decoded)
            decoded = unicodedata.normalize('NFKC', decoded)
            decoded = re.sub(r'http\S+', '', decoded)  # Remove URLs
        except Exception as e:
            logger.warning(f"Text cleaning failed: {str(e)}")
            decoded = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

        # Advanced cleaning
        cleaned = re.sub(r'[^\w\'\-áéíóúñÁÉÍÓÚÑ]', ' ', decoded)
        return [
            word[:config.max_word_length].lower()
            for word in cleaned.split()
            if len(word) > 1 and not word.isnumeric()
        ]

    def flush_word_counts(self, force=False):
        """Write word counts with frequency threshold"""
        if self.total_words < 10000 and not force:
            return
            
        batch = []
        for sender, words in self.word_counts.items():
            for word, count in words.items():
                batch.append(
                    Point("word_metrics")  # ← Must match measurement name
                        .tag("sender", sender)
                        .tag("word", word)
                        .field("count", count)
                )
        if batch:
            self.writer.write_batch(batch)
            logger.info(f"Flushed {len(batch)} word metrics (count >=10)")
        self.word_counts.clear()
        self.total_words = 0

def load_messages(folder_path: str, test_mode: bool) -> List[Dict]:
    """Load and validate messages from JSON files with deduplication"""
    logger.info(f"Loading messages from {folder_path}")
    
    try:
        matches = []
        for root, _, files in os.walk(folder_path):
            matches.extend(
                Path(root) / f for f in files 
                if f.startswith('message_') and f.endswith('.json')
            )

        if not matches:
            logger.error("❌ No message files found")
            sys.exit(1)

        if test_mode:
            matches = matches[:2]
            print("🔬 Test mode: Processing first 2 files")

        # Interactive file selection
        if len(matches) > 10 and not test_mode:
            choices = [questionary.Choice(title=str(f), value=f) for f in matches]
            selected = questionary.checkbox(
                "📄 Select message files to process:",
                choices=choices,
                style=custom_style
            ).ask()
            matches = selected

        seen_messages = set()
        messages = []
        duplicate_count = 0
        
        for file in tqdm(files, desc="Loading files"):
            file_path = os.path.join(folder_path, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in {file}: {str(e)}")
                        continue
                        
                    for msg in data.get('messages', []):
                        # Create unique message hash
                        content = msg.get('content', '')[:256]  # Truncate long content
                        msg_hash = hashlib.sha256(
                            f"{msg['timestamp_ms']}-{msg['sender_name']}-{content}".encode()
                        ).hexdigest()
                        
                        if msg_hash not in seen_messages:
                            seen_messages.add(msg_hash)
                            messages.append(msg)
                        else:
                            duplicate_count += 1
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
                continue
        
        if messages:
            messages.sort(key=lambda x: x['timestamp_ms'])
            logger.info(f"Loaded {len(messages)} unique messages (dropped {duplicate_count} duplicates)")
            return messages
        else:
            logger.error("No valid messages found in loaded files")
            return []

    except Exception as e:
        logger.error(f"Folder processing failed: {str(e)}")
        raise

def process_conversation(messages: List[Dict], writer: InfluxWriter) -> None:
    """Enhanced processing pipeline with user feedback and progress tracking"""
    # Initialize components with visual feedback
    console = Console()
    processor = MessageProcessor()
    conv_analyzer = ConversationAnalyzer(writer)
    
    with console.status("[bold green]🚀 Initializing processing components...") as status:
        conv_analyzer.last_timestamp = None
        conv_analyzer.last_sender = None
        conv_analyzer.current_thread = []
        message_counts = defaultdict(int)
        invalid_messages = 0
        processed_hashes = set()
        last_timestamp = None
        valid_messages = 0

        # Track additional metrics for user feedback
        earliest_date = datetime.max.replace(tzinfo=timezone.utc)
        latest_date = datetime.min.replace(tzinfo=timezone.utc)
        participant_counts = defaultdict(int)
        media_stats = defaultdict(int)

    # Phase 1: Initial message validation and counts
    with console.status("[bold cyan]🔍 Validating message structure...") as status:
        for msg in messages:
            try:
                DataValidator.validate_message(msg)
                sender = msg['sender_name']
                message_counts[sender] += 1
                participant_counts[sender] += 1
            except (ValueError, TypeError) as e:
                invalid_messages += 1
                logger.warning(f"⚠️ Invalid message format: {str(e)}")

    # Phase 2: Write initial metrics with progress tracking
    with tqdm(total=len(message_counts)+2, desc="📊 Initial metrics setup") as pbar:
        # Write individual participant counts
        for sender, count in message_counts.items():
            count_point = Point("conversation_stats") \
                .tag("type", "message_count") \
                .tag("sender", sender) \
                .field("value", count) \
                .time(datetime.now(timezone.utc))
            writer.write_point(count_point)
            pbar.update(1)
            pbar.set_postfix_str(f"Updated {sender}'s count")

        # Write total messages
        total_messages = sum(message_counts.values())
        total_point = Point("conversation_stats") \
            .tag("type", "total_messages") \
            .field("value", total_messages) \
            .time(datetime.now(timezone.utc))
        writer.write_point(total_point)
        pbar.update(1)
        pbar.set_postfix_str("Total messages recorded")

        pbar.set_description("✅ Initial metrics complete")
        pbar.update(1)

    # Phase 3: Main processing with enhanced feedback
    with tqdm(total=len(messages), desc="📨 Processing messages", 
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed} Remaining: {remaining}]") as main_bar:
        
        # Initialize metrics board
        metrics_table = Table(show_edge=False, show_header=False)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="bold green")
        
        for batch_idx, batch in enumerate(batched(messages, config.batch_size)):
            batch_points = []
            
            for msg in batch:
                try:
                    # Message validation and hashing
                    msg_hash = hashlib.sha256(
                        f"{msg['timestamp_ms']}-{msg['sender_name']}-{msg.get('content','')}".encode()
                    ).hexdigest()
                    if msg_hash in processed_hashes:
                        continue
                    processed_hashes.add(msg_hash)

                    # Track message timeline
                    timestamp_ms = processor.validate_timestamp(msg['timestamp_ms'])
                    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                    if dt < earliest_date:
                        earliest_date = dt
                    if dt > latest_date:
                        latest_date = dt

                    # Update metrics display
                    if main_bar.n % 50 == 0:
                        metrics_table.rows = [
                            ("Processed", f"{main_bar.n} messages"),
                            ("Current Date", dt.strftime("%Y-%m-%d %H:%M")),
                            ("Top Participant", max(participant_counts, key=participant_counts.get)),
                            ("Media Files", sum(media_stats.values()))
                        ]
                        console.print(metrics_table)

                    # Process message content
                    content = msg.get('content', '')
                    text_metrics = processor.process_text(content)
                    
                    # Handle media attachments
                    photos = processor.process_media(msg.get('photos', []), 'photo')
                    videos = processor.process_media(msg.get('videos', []), 'video')
                    media_count = photos['count'] + videos['count']
                    media_stats['total'] += media_count
                    media_stats['photos'] += photos['count']
                    media_stats['videos'] += videos['count']

                    # Generate data points
                    batch_points.extend(create_message_points(msg, dt, text_metrics, photos, videos))
                    batch_points.extend(create_sentiment_points(msg, dt, processor, text_metrics))
                    batch_points.extend(conv_analyzer.process_message(msg, dt))

                    valid_messages += 1
                    main_bar.update(1)

                except (ValueError, TypeError) as e:
                    invalid_messages += 1
                    logger.error(f"[red]✖️ Skipping invalid message: {str(e)}[/red]")
                    continue
                except Exception as e:
                    logger.error(f"[bold red]⚠️ Critical error processing message: {str(e)}[/bold red]")
                    raise

            # Batch writing with retry feedback
            if batch_points:
                try:
                    writer.write_batch(batch_points)
                    media_stats['batches'] += 1
                except Exception as e:
                    logger.error(f"[bold red]❌ Failed to write batch {batch_idx+1}: {str(e)}[/bold red]")
                    logger.info("🔄 Retrying with smaller batch size...")
                    for point in batch_points:
                        writer.write_point(point)
                    logger.info("✅ Recovery completed")

            # Periodic cleanup
            if batch_idx % 5 == 0:
                conv_analyzer.flush_word_counts()
                logger.debug("🧹 Flushed intermediate word counts")

    # Finalization phase
    with console.status("[bold blue]🎉 Finalizing processing...") as status:
        # Flush remaining data
        conv_analyzer.flush_word_counts(force=True)
        time.sleep(2)

        # Write conversation threads
        for thread in conv_analyzer.active_threads:
            try:
                thread_point = Point("conversation_threads") \
                    .tag("participants", ','.join(thread['participants'])) \
                    .field("duration_hours", (thread['end'] - thread['start']) / 3600000) \
                    .field("message_count", thread['message_count']) \
                    .time(datetime.fromtimestamp(thread['start']/1000, tz=timezone.utc))
                writer.write_point(thread_point)
            except Exception as e:
                logger.error(f"[yellow]⚠️ Couldn't write thread metric: {str(e)}[/yellow]")

        # Final health check
        writer.write_api.flush()
        time.sleep(5)

    # User-friendly summary
    console.print(Panel.fit(
        f"""✅ [bold green]Processing Complete![/bold green]
        
        📅 [cyan]Time Range:[/cyan] {earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}
        📨 [cyan]Total Messages:[/cyan] {valid_messages} ({invalid_messages} skipped)
        👥 [cyan]Participants:[/cyan] {', '.join(sorted(participant_counts.keys()))}
        📷 [cyan]Media Files:[/cyan] {media_stats['photos']} photos, {media_stats['videos']} videos
        
        [dim]Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]""",
        title="Final Report",
        border_style="green"
    ))

def create_message_points(msg, dt, text_metrics, photos, videos):
    """Generate standard message metrics points"""
    points = []
    
    # Core message counter
    points.append(
        Point("message_count")
        .tag("sender", msg['sender_name'])
        .tag("day_of_week", dt.strftime('%A'))
        .tag("hour", f"{dt.hour:02d}")
        .field("count", 1)
        .time(dt)
    )

    if 'reactions' in msg:
    # Track total reactions per message
        points.append(
            Point("message_metrics")
            .tag("sender", msg['sender_name'])
            .field("reaction_count", len(msg['reactions']))
            .time(dt)
        )
    
        # Track individual reactions
        for reaction in msg['reactions']:
            points.append(
                Point("reaction_metrics")
                .tag("sender", msg['sender_name'])
                .tag("reactor", sanitize_tag(reaction['actor']))
                .tag("emoji", reaction['reaction'])
                .field("count", 1)
                .time(dt)
            )
    
    # Main metrics
    points.append(
        Point("message_metrics")
        .tag("sender", msg['sender_name'])
        .field("word_count", int(text_metrics['word_count']))
        .field("emoji_count", int(text_metrics['emoji_count']))
        .field("url_count", int(text_metrics['url_count']))
        .field("photo_count", int(photos['count']))
        .field("video_count", int(videos['count']))
        .time(dt)
    )
    
    return points

def create_sentiment_points(msg, dt, processor, text_metrics):
    """Generate sentiment analysis points"""
    points = []
    sentiment = processor.analyze_sentiment(text_metrics['cleaned_text'])
    
    # Main sentiment
    points.append(
        Point("sentiment_metrics")
        .tag("sender", msg['sender_name'])
        .field("score", sentiment['sentiment'])
        .time(dt)
    )
    
    # Emotion breakdown
    for emotion, count in sentiment['emotions'].items():
        points.append(
            Point("emotion_metrics")
            .tag("sender", msg['sender_name'])
            .tag("emotion", emotion)
            .field("count", count)
            .time(dt)
        )
    
    return points
    
def batched(iterable, n):
    """Batch processing generator"""
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

def detect_instagram_folder() -> Path:
    """Automatically find Instagram data folder with confirmation"""
    console = Console()
    base_dir = Path.cwd()
    
    # Find all folders starting with "instagram-"
    candidates = list(base_dir.glob("instagram-*"))
    
    if not candidates:
        console.print("\n[bold red]❌ No Instagram data folder found![/bold red]")
        console.print(f"• Expected folder name format: [cyan]instagram-username[/cyan]")
        console.print(f"• Please place your Instagram data folder in: [yellow]{base_dir}[/yellow]")
        sys.exit(1)

    # If multiple candidates found, let user choose
    if len(candidates) > 1:
        choices = [Choice(title=f.name, value=f) for f in candidates]
        selected = questionary.select(
            "🔍 Multiple Instagram folders found - which one should we use?",
            choices=choices,
            style=custom_style
        ).ask()
        return validate_data_path(selected)

    # Single candidate found - confirm with user
    folder = candidates[0]
    console.print(f"\n🔍 Found Instagram data folder: [cyan]{folder.name}[/cyan]")
    if questionary.confirm("Is this the correct folder?", default=True).ask():
        return validate_data_path(folder)
    
    console.print("\n[bold yellow]⚠️ Please ensure you have the correct Instagram data folder:")
    console.print(f"• Should start with 'instagram-' followed by your username")
    console.print(f"• Should contain 'your_instagram_activity' subfolder[/bold yellow]")
    sys.exit(0)

def validate_data_path(folder: Path) -> Path:
    """Validate Instagram folder structure"""
    required_path = folder / "your_instagram_activity" / "messages" / "inbox"
    
    if not required_path.exists():
        console = Console()
        console.print("\n[bold red]⚠️ Invalid Instagram data structure![/bold red]")
        console.print(f"Missing expected path: {required_path}")
        console.print("\nPlease ensure you're using an unmodified Instagram data export")
        sys.exit(1)
    
    return folder

def load_messages(conversation_folder: Path, test_mode: bool) -> List[Dict]:
    """Load messages from a specific conversation folder"""
    json_files = list(conversation_folder.glob("message_*.json"))
    
    if not json_files:
        logger.warning(f"No message files found in {conversation_folder.name}")
        return []

    if test_mode:
        json_files = json_files[:1]
        logger.info(f"Test mode - processing first file in {conversation_folder.name}")

    messages = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                messages.extend(data.get('messages', []))
        except Exception as e:
            logger.error(f"Error loading {json_file.name}: {str(e)}")
    
    logger.info(f"Loaded {len(messages)} messages from {conversation_folder.name}")
    return messages

def get_conversation_name(folder_path: Path) -> str:
    """Extract human-readable conversation name from folder path"""
    # Split on first occurrence of underscore followed by numbers
    base_name = re.split(r'_\d+$', folder_path.name)[0]
    # Replace underscores with spaces and title case
    return base_name.replace('_', ' ').title()

def sanitize_tag(value: str) -> str:
    """Clean tags for InfluxDB compatibility"""
    return re.sub(r'[^\w]', '_', value.strip().lower())[:50]

def main():
    """Main execution flow with single conversation selection"""
    parser = argparse.ArgumentParser(description='Process Instagram conversation data')
    parser.add_argument('--test', action='store_true', help='Enable test mode')
    parser.add_argument('--clear', action='store_true', help='Clear bucket before processing')
    args = parser.parse_args()

    db_manager = None
    try:
        # Auto-detect and validate Instagram folder
        instagram_folder = detect_instagram_folder()
        inbox_path = instagram_folder / "your_instagram_activity" / "messages" / "inbox"

    # Get list of conversation folders
        conversation_folders = [f for f in inbox_path.iterdir() if f.is_dir()]
        if not conversation_folders:
            logger.error("❌ No conversations found in inbox folder")
            sys.exit(1)

        # Create a mapping between display names and paths
        folder_map = {
            get_conversation_name(f): f
            for f in conversation_folders
        }
        
    # Interactive conversation selection with search
        selected_name = questionary.autocomplete(
            "💬 Search and select a conversation:",
            choices=list(folder_map.keys()),
            style=custom_style,
            match_middle=True,
            validate=lambda text: len(text) >= 2 or "Type at least 2 characters"
        ).ask()

        # Handle cancellation or empty selection
        if not selected_name:
            logger.info("🚫 No conversation selected. Exiting.")
            sys.exit()

        # Get the actual folder path from our mapping
        try:
            selected_folder = folder_map[selected_name]
        except KeyError:
            logger.error(f"❌ Invalid selection: '{selected_name}' not found")
            sys.exit(1)

        # Initialize database connection
        db_manager = InfluxDBManager()
        writer = db_manager.get_writer()

        if args.clear:
            logger.info("🧹 Clearing existing bucket data...")
            db_manager.clear_bucket()

        logger.info(f"📂 Processing conversation: {selected_folder.name}")
        messages = load_messages(selected_folder, args.test)

        if not messages:
            logger.error("❌ No valid messages found in selected conversation")
            sys.exit(1)

        # Sort messages chronologically and process
        messages.sort(key=lambda x: x['timestamp_ms'])
        process_conversation(messages, writer)
        logger.info("✅ Processing completed successfully")

    except Exception as e:
        logger.error(f"💥 Critical error: {str(e)}", exc_info=True)
    finally:
        if db_manager:
            db_manager.close()

if __name__ == "__main__":
    main()