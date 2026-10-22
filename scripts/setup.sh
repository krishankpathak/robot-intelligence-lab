#!/usr/bin/env bash
# Setup script for Linux / macOS
set -e

echo "Setting up ROBOT INTELLIGENCE LAB..."

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Running automated tests..."
pytest tests -v

echo "Setup complete! Run './scripts/run.sh' to launch."
