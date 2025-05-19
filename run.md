cd ~/real-time-object-detection && source venv/bin/activate &&
export PYTHONPATH=/home/aa30301161801636/real-time-object-detection && cd backend/

# Start the backend server

nohup uvicorn main:app --host 0.0.0.0 --port 8080 --reload > output.log 2>&1 &

# Navigate to frontend directory

cd ~/real-time-object-detection/frontend

# Install dependencies (only if you made changes or first time)

npm install

# Build the frontend

npm run build

# Copy the built files to Nginx directory

sudo rm -rf /usr/share/nginx/html/*
sudo cp -r dist/* /usr/share/nginx/html/

# Set proper permissions

sudo chown -R www-data:www-data /usr/share/nginx/html
sudo chmod -R 755 /usr/share/nginx/html

# Check if Nginx configuration is valid

sudo nginx -t

# Start/Restart Nginx

sudo systemctl restart nginx

# Verify Nginx is running

sudo systemctl status nginx

# Check Nginx errors

sudo tail -f /var/log/nginx/error.log

# Check backend logs (from the backend terminal)

# They will show in the terminal where you run uvicorn

# Check if ports are in use

sudo lsof -i :80
sudo lsof -i :443
sudo lsof -i :8080

# Restart services if needed

sudo systemctl restart nginx

# prepare and runing the app

cd /home/sci/WSL_Space/real-time-object-detection
python ml_models/prepare_models.py --all

## runing backend

cd /home/sci/WSL_Space/real-time-object-detection
PYTHONPATH=/home/sci/WSL_Space/real-time-object-detection python backend/main.py
