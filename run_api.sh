#!/bin/bash

# Load environment variables if .env file exists
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Create directories if they don't exist
mkdir -p static/css static/js static/img templates saved_chats

# Check if the required packages are installed
echo "Checking and installing required packages..."
pip3 install -r requirements.txt

# Run the FastAPI application
echo "Starting MedCalc-Agent API server..."
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload 