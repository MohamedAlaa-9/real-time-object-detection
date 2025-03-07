from fastapi import APIRouter, WebSocket
import cv2
import base64
import numpy as np
from inference import detect_objects
router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detect_objects(frame)
        
        _, buffer = cv2.imencode(".jpg", frame)
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        await websocket.send_json({"image": frame_base64, "detections": detections.tolist()})

    cap.release()
