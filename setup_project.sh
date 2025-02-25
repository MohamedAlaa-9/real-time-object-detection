#!/bin/bash

echo "Setting up the project..."

# Ensure script runs from the project root directory
cd "$(dirname "$0")"

# Define the virtual environment name
VENV_NAME="object-detection-venv"

# -------------------------------
# Setup Datasets (raw & processed folders)
# -------------------------------
cd datasets
mkdir -p raw processed
cd ..

# -------------------------------
# Install Dependencies (Backend & ML)
# -------------------------------
echo "Installing Python Dependencies..."

# Create & activate virtual environment with a clean name
python3 -m venv $VENV_NAME
source $VENV_NAME/bin/activate

# Install dependencies from root `requirements.txt`
pip install --upgrade pip
pip install -r requirements.txt

# -------------------------------
# Frontend Setup
# -------------------------------
echo "Setting up Frontend..."

# Navigate to frontend folder
cd frontend

# Install frontend dependencies
npm install

# Go back to the root directory
cd ..

# -------------------------------
# Environment Setup
# -------------------------------
echo "Creating .env files..."

# Create a default backend .env file if not exists
if [ ! -f backend/.env ]; then
  cat <<EOL > backend/.env
DEBUG=
BACKEND_PORT=
DB_HOST=localhost
DB_PORT=
DB_USER=
DB_PASS=
EOL
  echo "Created backend/.env"
fi

# Create a default frontend .env file if not exists
if [ ! -f frontend/.env ]; then
  cat <<EOL > frontend/.env
REACT_APP_BACKEND_URL=http://localhost:$PORT
EOL
  echo "Created frontend/.env"
fi

# -------------------------------
# Run Backend & Frontend
# -------------------------------
echo "Starting Backend & Frontend..."

# Start backend in a new terminal tab
gnome-terminal -- bash -c "source $VENV_NAME/bin/activate && uvicorn backend.main:app --host 0.0.0.0 --port $PORT --reload; exec bash"

# Start frontend in a new terminal tab
gnome-terminal -- bash -c "cd frontend && npm start; exec bash"

echo "Setup complete!"