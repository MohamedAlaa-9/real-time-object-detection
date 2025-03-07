from fastapi import FastAPI
from routes import router
from inference import detect_objects

app = FastAPI()

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Real-time Object Detection API is running"}
