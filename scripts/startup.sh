#!/bin/bash

# AIVA Startup Script
# This script runs on VM boot to start all required services

set -e

echo "[STARTUP] Starting AIVA services..."

# Change to aiva directory
cd /home/tomoyoukilai_gmail_com/aiva

# Set PYTHONPATH to include flash_head src and user's site-packages
export PYTHONPATH=/home/tomoyoukilai_gmail_com/aiva/models/flash_head/src:/home/tomoyoukilai_gmail_com/.local/lib/python3.10/site-packages:$PYTHONPATH

# Ensure nginx is running
echo "[STARTUP] Ensuring nginx is running..."
sudo systemctl enable nginx || true
sudo systemctl restart nginx || true

# Start Python inference server
echo "[STARTUP] Starting Python inference server on port 50051..."
PYTHONPATH=/home/tomoyoukilai_gmail_com/aiva/models/flash_head/src:/home/tomoyoukilai_gmail_com/.local/lib/python3.10/site-packages nohup sudo -u tomoyoukilai_gmail_com python3 -m inference.server > /tmp/inference.log 2>&1 &
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