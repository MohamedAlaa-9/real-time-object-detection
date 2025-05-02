import os
import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from backend.core.config import logger, UPLOAD_DIR
from backend.services.video_processor import process_video, video_processing_status

router = APIRouter(prefix="/video", tags=["Video Processing"])


@router.post("/upload/")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a video file for processing"""
    try:
        # Generate a unique ID for this video
        video_id = str(uuid.uuid4())
        
        # Ensure file is a video
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
            
        # Create file path
        file_extension = os.path.splitext(file.filename)[1]
        video_path = UPLOAD_DIR / f"{video_id}{file_extension}"
        
        # Save the uploaded file
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Start processing the video in the background
        background_tasks.add_task(process_video, video_id, video_path)
        
        return {"video_id": video_id, "filename": file.filename, "status": "uploaded"}
        
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{video_id}")
async def get_video_status(video_id: str):
    """Get the processing status of a video"""
    if video_id not in video_processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
        
    return video_processing_status[video_id]


@router.get("/result/{video_id}")
async def get_video_result(video_id: str):
    """Get the processing result information for a video"""
    if video_id not in video_processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
        
    status = video_processing_status[video_id]
    
    if status["status"] != "completed":
        return status
        
    return {
        "status": "completed",
        "video_url": f"/video/stream/{video_id}",
        "thumbnail_url": f"/video/thumbnail/{video_id}"
    }


@router.get("/stream/{video_id}")
async def stream_video(video_id: str):
    """Stream the processed video"""
    if video_id not in video_processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
        
    status = video_processing_status[video_id]
    
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Video processing not completed")
        
    output_path = Path(status["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed video file not found")
        
    def iterfile():
        with open(output_path, "rb") as file:
            yield from file
            
    return StreamingResponse(
        iterfile(),
        media_type="video/mp4"
    )


@router.get("/thumbnail/{video_id}")
async def get_thumbnail(video_id: str):
    """Get the thumbnail of the processed video"""
    if video_id not in video_processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
        
    status = video_processing_status[video_id]
    
    if status["status"] != "completed" or "thumbnail_path" not in status:
        raise HTTPException(status_code=400, detail="Thumbnail not available")
        
    thumbnail_path = Path(status["thumbnail_path"])
    if not thumbnail_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail file not found")
        
    def iterfile():
        with open(thumbnail_path, "rb") as file:
            yield from file
            
    return StreamingResponse(
        iterfile(),
        media_type="image/jpeg"
    )