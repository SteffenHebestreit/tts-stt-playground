#!/bin/bash

# STT Service startup script

echo "Starting STT Service..."

# Set environment variables
export PYTHONPATH=/app:$PYTHONPATH

# Check if models directory exists
if [ ! -d "/app/models" ]; then
    mkdir -p /app/models
fi

# Start the FastAPI application
echo "Starting FastAPI server on port 8000..."
exec python -m uvicorn app:app --host 0.0.0.0 --port 8000
