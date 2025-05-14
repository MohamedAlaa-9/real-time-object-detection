#!/bin/bash
cd /home/sci/WSL_Space/real-time-object-detection
export PYTHONPATH=/home/sci/WSL_Space/real-time-object-detection
nohup python backend/main.py > backend.log 2>&1 &
cd /home/sci/WSL_Space/real-time-object-detection/frontend
nohup npm run dev > frontend.log 2>&1 &