#!/bin/bash

# STT Service startup script

echo "Starting STT Service..."

# Set environment variables
export PYTHONPATH=/app:$PYTHONPATH

# Check if models directory exists
if [ ! -d "/app/models" ]; then
    mkdir -p /app/models
fi

# Start the FastAPI application.
#   --ws-max-queue    default is 32 frames (~2.7 s of audio); a larger queue
#                     absorbs a decode burst instead of dropping the connection
#   --timeout-keep-alive  must be >= the gateway's httpx keepalive_expiry, or the
#                     server FINs first and pooled sockets come back dead
#   single worker: the Whisper model is process-global GPU state
echo "Starting FastAPI server on port 8000..."
exec python -m uvicorn app:app \
    --host 0.0.0.0 --port 8000 \
    --workers 1 \
    --ws-max-queue "${WS_MAX_QUEUE:-128}" \
    --timeout-keep-alive "${KEEPALIVE_TIMEOUT:-120}" \
    --no-access-log
