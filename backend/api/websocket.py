import time
from fastapi import WebSocket, WebSocketDisconnect

from backend.core.config import logger
from backend.services.video_processor import process_frame


async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time frame processing"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        # Process incoming frames
        while True:
            # Receive the frame
            data = await websocket.receive_text()
            
            # Skip if the data is not a valid image
            if not data.startswith('data:image'):
                await websocket.send_json({"error": "Invalid image data"})
                continue
            
            # Process the frame asynchronously
            start_time = time.time()
            result = await process_frame(data)
            processing_time = time.time() - start_time
            
            # Add processing time to the result
            result["processing_time"] = processing_time
            
            # Send the processed frame back
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=f"Server error: {str(e)}")