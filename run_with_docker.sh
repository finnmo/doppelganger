#!/bin/bash

# Check for required tools
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 could not be found. Please install $1 and try again."
        exit 1
    fi
}

check_command docker
check_command docker-compose
check_command python3

# Create .env if missing
if [ ! -f .env ]; then
    echo "No .env file found. Starting initial setup..."
    python3 process_dms.py --setup
fi

# Start Docker services
echo "Starting Docker containers..."
docker-compose up -d

# Wait for services to initialize
echo "Waiting for services to become ready..."
sleep 15

# Process Instagram data
echo "Starting data processing..."
python3 process_dms.py

echo -e "\n✅ Setup complete! Services are running in the background."
echo -e "Access Grafana at: http://localhost:3000 (admin:admin)"
echo -e "Stop services with: docker-compose down\n"