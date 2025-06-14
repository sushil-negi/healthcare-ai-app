import time
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Healthcare AI Service")


class ChatRequest(BaseModel):
    message: str
    user_id: str = "user123"


class HealthResponse(BaseModel):
    status: str
    timestamp: float


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", timestamp=time.time())


@app.get("/ready", response_model=HealthResponse)
async def ready():
    return HealthResponse(status="ready", timestamp=time.time())


@app.post("/chat")
async def chat(request: ChatRequest) -> Dict:
    return {
        "response": f"Healthcare AI received: {request.message}",
        "user_id": request.user_id,
        "timestamp": time.time(),
    }


@app.get("/")
async def root():
    return {"message": "Healthcare AI Service is running"}
