#!/bin/bash
# Stop the backend (Python) process
pkill -f "python backend/main.py"

# Stop the frontend (Node.js) process
pkill -f "node.*real-time-object-detection/frontend"