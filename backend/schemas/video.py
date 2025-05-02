from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Detection(BaseModel):
    """Schema for a single object detection result"""
    box: List[int]  # [x_min, y_min, x_max, y_max]
    score: float
    class_id: int
    label: str


class ProcessedFrame(BaseModel):
    """Schema for a processed frame with detections"""
    processed_frame: Optional[str] = None  # Base64 encoded image
    detections: List[Detection] = []
    processing_time: Optional[float] = None
    error: Optional[str] = None


class VideoStatus(BaseModel):
    """Schema for video processing status"""
    status: str  # "uploaded", "processing", "completed", "failed"
    progress: Optional[int] = None
    error: Optional[str] = None
    output_path: Optional[str] = None
    thumbnail_path: Optional[str] = None


class VideoUploadResponse(BaseModel):
    """Schema for video upload response"""
    video_id: str
    filename: str
    status: str


class VideoResult(BaseModel):
    """Schema for video result response"""
    status: str
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None