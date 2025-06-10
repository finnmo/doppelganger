import os
import subprocess
import sys
import socket

def install_dependencies():
    """Install Python dependencies from requirements.txt."""
    print("Installing Python dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

def is_grafana_running():
    """Check if Grafana is running on the default port."""
    try:
        with socket.create_connection(("localhost", 3000), timeout=2):
            return True
    except OSError:
        return False

def start_grafana():
    """Start Grafana Docker container."""
    print("Starting Grafana...")
    subprocess.run(["docker-compose", "up", "-d"], check=True)

def run_data_processing():
    """Run the data processing script."""
    print("Processing data...")
    subprocess.run([sys.executable, "process_dms.py"], check=True)

def main():
    try:
        install_dependencies()

        # Check if Grafana is running
        if not is_grafana_running():
            start_grafana()
            print("Grafana started. Access it at http://localhost:3000 (admin/admin).")
        else:
            print("Grafana is already running.")

        # Process the data
        run_data_processing()

        print("Setup complete! Access Grafana at http://localhost:3000 to view your insights.")
    except subprocess.CalledProcessError as e:
        print(f"Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
