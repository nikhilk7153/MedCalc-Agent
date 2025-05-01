#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 to run this application."
    exit 1
fi

echo "🏥 Starting Medical Calculator Assistant..."

# Check if requirements are installed, install if needed
echo "Installing required dependencies..."
pip install -r requirements.txt

# Run the Streamlit app
echo "Launching the Medical Calculator Assistant..."
streamlit run llm_chat_app.py 