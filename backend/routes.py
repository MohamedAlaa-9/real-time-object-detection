from fastapi import APIRouter, WebSocket, Query
import cv2
import base64
import numpy as np
from inference import detect_objects
router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, camera_id: int = Query(0)):
    try:
        await websocket.accept()
        print(f"WebSocket connection accepted. Camera ID: {camera_id}")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Could not open video device with id {camera_id}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            detections = detect_objects(frame)
            
            _, buffer = cv2.imencode(".jpg", frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

            await websocket.send_json({"image": frame_base64, "detections": detections.tolist()})

        cap.release()
    except Exception as e:
        print(f"Error in websocket_endpoint: {e}")
