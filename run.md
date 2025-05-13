# prepare and runing the app

cd /home/sci/WSL_Space/real-time-object-detection
python ml_models/prepare_models.py --all

## runing backend

cd /home/sci/WSL_Space/real-time-object-detection
PYTHONPATH=/home/sci/WSL_Space/real-time-object-detection python backend/main.py

## runing frontend

cd /home/sci/WSL_Space/real-time-object-detection/frontend && npm run dev
