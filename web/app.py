#!/usr/bin/env python3
"""
Healthcare AI Web Interface
Serves the healthcare chat UI and proxies API requests
"""

import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import httpx
import structlog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Configuration from environment
HEALTHCARE_AI_URL = os.getenv("HEALTHCARE_AI_URL", "http://healthcare-ai:8001")
PORT = int(os.getenv("PORT", "8080"))
HIPAA_COMPLIANCE_MODE = os.getenv("HIPAA_COMPLIANCE_MODE", "true").lower() == "true"

# Create FastAPI app
app = FastAPI(
    title="Healthcare AI Web Interface",
    description="Web interface for Healthcare AI chatbot",
    version="1.0.0",
    docs_url="/docs" if not HIPAA_COMPLIANCE_MODE else None,
    redoc_url="/redoc" if not HIPAA_COMPLIANCE_MODE else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_path = Path(__file__).parent / "web-ui"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# HTTP client for backend communication
http_client = httpx.AsyncClient(timeout=30.0)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main healthcare chat interface"""
    html_file = static_path / "healthcare-chat.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    else:
        return HTMLResponse(
            content="<h1>Healthcare AI</h1><p>Chat interface not found</p>",
            status_code=200
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if healthcare AI backend is reachable
        response = await http_client.get(f"{HEALTHCARE_AI_URL}/health")
        backend_healthy = response.status_code == 200
        
        return {
            "status": "healthy" if backend_healthy else "degraded",
            "service": "healthcare-web",
            "backend_status": "healthy" if backend_healthy else "unhealthy",
            "backend_url": HEALTHCARE_AI_URL,
            "hipaa_compliance": HIPAA_COMPLIANCE_MODE,
            "timestamp": "2025-06-15T22:48:00Z"
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "healthcare-web",
                "error": "Cannot reach healthcare AI backend",
                "timestamp": "2025-06-15T22:48:00Z"
            }
        )

@app.post("/api/chat")
async def chat_proxy(request: Request):
    """Proxy chat requests to healthcare AI backend"""
    try:
        body = await request.body()
        headers = dict(request.headers)
        
        # Remove host header to avoid conflicts
        headers.pop("host", None)
        
        # Add HIPAA compliance headers
        if HIPAA_COMPLIANCE_MODE:
            headers["X-HIPAA-Compliance"] = "true"
            headers["X-PHI-Anonymization"] = "enabled"
        
        # Forward request to healthcare AI backend
        response = await http_client.post(
            f"{HEALTHCARE_AI_URL}/chat",
            content=body,
            headers=headers
        )
        
        # Return response with HIPAA headers
        response_headers = {}
        if HIPAA_COMPLIANCE_MODE:
            response_headers["X-HIPAA-Compliance"] = "true"
            response_headers["X-PHI-Protected"] = "true"
        
        return JSONResponse(
            content=response.json(),
            status_code=response.status_code,
            headers=response_headers
        )
        
    except httpx.RequestError as e:
        logger.error("Failed to proxy chat request", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Healthcare AI service unavailable"
        )
    except Exception as e:
        logger.error("Chat proxy error", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.get("/api/status")
async def status():
    """Get service status"""
    try:
        # Check backend status
        response = await http_client.get(f"{HEALTHCARE_AI_URL}/stats")
        backend_status = response.json() if response.status_code == 200 else None
        
        return {
            "web_interface": "healthy",
            "backend_service": backend_status,
            "hipaa_compliance": HIPAA_COMPLIANCE_MODE,
            "features": {
                "crisis_detection": True,
                "phi_anonymization": HIPAA_COMPLIANCE_MODE,
                "audit_logging": True,
                "emergency_escalation": True
            }
        }
    except Exception as e:
        logger.error("Status check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"error": "Cannot reach healthcare AI backend"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )