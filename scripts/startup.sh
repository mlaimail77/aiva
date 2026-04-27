#!/bin/bash

# AIVA Startup Script
# This script runs on VM boot to start all required services

set -e

echo "[STARTUP] Starting AIVA services..."

# Change to aiva directory
cd /home/tomoyoukilai_gmail_com/aiva

# Set PYTHONPATH to include flash_head and user's site-packages
export PYTHONPATH=/home/tomoyoukilai_gmail_com/aiva/models/flash_head:/home/tomoyoukilai_gmail_com/aiva/models/flash_head/src:/home/tomoyoukilai_gmail_com/.local/lib/python3.10/site-packages

# Create symlinks for flash_head submodules if they don't exist
if [ ! -L /home/tomoyoukilai_gmail_com/aiva/models/flash_head/flash_head/ltx_video ]; then
    echo "[STARTUP] Creating symlinks for flash_head submodules..."
    cd /home/tomoyoukilai_gmail_com/aiva/models/flash_head/flash_head
    ln -sf ../ltx_video ltx_video
    ln -sf ../audio_analysis audio_analysis
    ln -sf ../wan wan
    cd /home/tomoyoukilai_gmail_com/aiva
fi

# Ensure nginx is running
echo "[STARTUP] Ensuring nginx is running..."
sudo systemctl enable nginx || true
sudo systemctl restart nginx || true

# Start Python inference server
echo "[STARTUP] Starting Python inference server on port 50051..."
export PYTHONPATH=/home/tomoyoukilai_gmail_com/aiva/models/flash_head:/home/tomoyoukilai_gmail_com/aiva/models/flash_head/src:/home/tomoyoukilai_gmail_com/.local/lib/python3.10/site-packages
cd /home/tomoyoukilai_gmail_com/aiva
nohup sudo -u tomoyoukilai_gmail_com -E bash -c 'cd /home/tomoyoukilai_gmail_com/aiva && exec python3 -m inference.server' > /home/tomoyoukilai_gmail_com/inference.log 2>&1 &
INFERENCE_PID=$!
echo "[STARTUP] Inference server started with PID: $INFERENCE_PID"
INFERENCE_PID=$!
echo "[STARTUP] Inference server started with PID: $INFERENCE_PID"

# Wait for inference server to be ready
echo "[STARTUP] Waiting for inference server to initialize..."
sleep 30

# Start Go API server
echo "[STARTUP] Starting Go API server on port 8080..."
nohup ./server/aiva-server -config ./aiva_config.yaml > /tmp/aiva-server.log 2>&1 &
GO_PID=$!
echo "[STARTUP] Go API server started with PID: $GO_PID"

# Wait for services to initialize
sleep 5

echo "[STARTUP] Checking services..."

# Check inference server
if ss -tlnp | grep -q ':50051'; then
    echo "[STARTUP] ✅ Inference server is running on port 50051"
else
    echo "[STARTUP] ❌ Inference server NOT running on port 50051"
    tail -20 /tmp/inference.log
fi

# Check Go API server
if ss -tlnp | grep -q ':8080'; then
    echo "[STARTUP] ✅ Go API server is running on port 8080"
else
    echo "[STARTUP] ❌ Go API server NOT running on port 8080"
    tail -20 /tmp/aiva-server.log
fi

# Check nginx
if ss -tlnp | grep -q ':80'; then
    echo "[STARTUP] ✅ Nginx is running on port 80"
else
    echo "[STARTUP] ❌ Nginx NOT running on port 80"
fi

echo "[STARTUP] AIVA startup complete!"