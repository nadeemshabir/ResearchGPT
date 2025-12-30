#!/bin/bash

echo "========================================"
echo "Starting ResearchGPT Web Interface"
echo "========================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found"
    echo "Please create one with: python -m venv venv"
    echo ""
fi

# Check if streamlit is installed
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the Streamlit app
echo ""
echo "Starting Streamlit server..."
echo ""
streamlit run app.py
