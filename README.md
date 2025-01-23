# Doppelganger: An Instagram Message Analytics Platform

📊 Visualize your Instagram message history with dashboards using InfluxDB & Grafana

## 🌟 Features
- Message frequency analysis
- Sentiment tracking
- Emotion heatmaps
- Conversation patterns
- Message type breakdowns

## 🛠️ Prerequisites
- Docker & Docker Compose
- Python 3.9+
- Instagram data export (instructions below)
- Basic terminal knowledge

## 📥 Getting Your Instagram Data
1. **Request Your Data**
   - Go to Instagram Settings → Security → Download Data
   - Enter your email and request "Complete copy"
   - You'll receive a download link within 48 hours

2. **Prepare the Data**
   ```bash
   unzip instagram-data.zip
   cp "messages/inbox" data/instagram_messages -r
   ```

## 🚀 Quick Start Guide

### 1. Clone Repository
```bash
git clone https://github.com/finnmo/doppelganger.git
cd doppelganger
```

### 2. Configure Environment
```bash
cp .env.example .env
```
Edit `.env` with your credentials:
```ini
# InfluxDB
INFLUXDB_USERNAME=admin
INFLUXDB_PASSWORD=securepassword
INFLUXDB_ORG=myorg
INFLUXDB_BUCKET=instagram
INFLUXDB_ADMIN_TOKEN=mytoken

# Grafana
GRAFANA_USERNAME=admin
GRAFANA_PASSWORD=secret
```

### 3. Start Services
```bash
chmod +x run_with_docker.sh
./run_with_docker.sh
```

### 4. Import Instagram Data
```bash
docker-compose run process_dms
```
Follow the prompts to locate your message data.

### 5. Access Dashboards
1. Open Grafana: http://localhost:3001
2. Login with credentials from `.env`
3. Explore pre-configured dashboards under "Instagram Analytics"

## 📊 Available Dashboards
- **Message Overview**: Daily message volume, peak activity times
- **Emotion Analysis**: Sentiment distribution, emotion heatmaps
- **User Insights**: Top contacts, response times
- **Conversation Flow**: Thread durations, message types

## 🔧 Troubleshooting

### Common Issues
**Data Not Appearing?**
- Verify message path: `data/instagram_messages/your_chats`
- Check container logs:
  ```bash
  docker-compose logs influxdb
  docker-compose logs process_dms
  ```

**Port Conflicts?**
- Modify ports in `docker-compose.yml`
  ```yaml
  ports:
    - '3002:3000'  # Grafana
    - '8087:8086'  # InfluxDB
  ```

**Reset Everything**
```bash
./run_with_docker.sh --reset
```

## 🔒 Security Best Practices
1. Change default credentials in `.env`
2. Regularly update Docker images
3. Use HTTPS in production
4. Restrict access to port 3001

## 📈 Example Insights
- Busiest conversation hours
- Most emotional contacts
- Message response patterns
- Media vs text ratios

## 🛑 Cleanup
```bash
docker-compose down -v
```

## TODO
- Fix Time remaining to account for batch processing
- Fix the remaining Grafana panels with "No Data"
- Add chat doppelganger

## 🤝 Contributing
PRs welcome!

---
