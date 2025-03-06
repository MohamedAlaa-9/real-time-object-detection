import cv2
import torch
from ultralytics import YOLO
from config import MODEL_PATH

model = YOLO(MODEL_PATH)

def detect_objects(frame):
    """ Runs YOLOv11 on a webcam frame and returns detections. """
    results = model(frame)
    return results[0].boxes.data.cpu().numpy()