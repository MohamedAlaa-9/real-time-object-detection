import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

from backend.core.config import logger, API_HOST, API_PORT
from backend.api.websocket import websocket_endpoint
from backend.api.video import router as video_router

# Initialize FastAPI application
app = FastAPI(
    title="Real-Time Object Detection API",
    description="API for real-time object detection in videos and live streams",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add websocket endpoint
app.add_websocket_route("/ws", websocket_endpoint)

# Include routers
app.include_router(video_router)

@app.get("/api/health", tags=["Health Check"])
def health_check():
    """API health check endpoint"""
    return {"status": "online", "message": "Real-Time Object Detection API is running"}

# Serve static frontend files if the directory exists
static_dir = Path(__file__).parent.parent / "frontend" / "dist"
if static_dir.exists() and static_dir.is_dir():
    logger.info(f"Serving frontend from {static_dir}")
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")


if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    logger.info(f"Starting server at {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
