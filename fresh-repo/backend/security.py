from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
import time
from typing import Dict
from datetime import datetime, timedelta

# Rate limiting setup
RATE_LIMIT_DURATION = timedelta(minutes=1)
MAX_REQUESTS = 60  # requests per minute
rate_limit_store: Dict[str, Dict] = {}

# API key setup
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-production-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    
    # Check if IP exists in store and clean old entries
    now = datetime.now()
    if client_ip in rate_limit_store:
        if now - rate_limit_store[client_ip]["start_time"] >= RATE_LIMIT_DURATION:
            rate_limit_store[client_ip] = {"count": 0, "start_time": now}
    else:
        rate_limit_store[client_ip] = {"count": 0, "start_time": now}
    
    # Increment request count
    rate_limit_store[client_ip]["count"] += 1
    
    # Check if rate limit exceeded
    if rate_limit_store[client_ip]["count"] > MAX_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Please try again later."}
        )
    
    return await call_next(request)

# API key validation
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != "your-secure-api-key":  # Replace with secure key management
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}