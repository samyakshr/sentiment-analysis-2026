#!/bin/bash
# Simple script to start Jupyter Notebook for this project

echo "Starting Jupyter Notebook..."
echo "The notebook will open in your browser automatically."
echo "Press Ctrl+C (or Cmd+C on Mac) to stop the server when done."
echo ""

# Check if jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "Jupyter not found. Installing requirements..."
    pip install -r requirements.txt
fi

# Start Jupyter
jupyter notebook

