#!/usr/bin/env bash
# Run script for Linux / macOS
if [ -d "venv" ]; then
    source venv/bin/activate
fi

streamlit run app.py
