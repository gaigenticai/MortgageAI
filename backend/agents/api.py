"""
Main API server for AI Agents microservices.

This module provides a unified FastAPI application that serves both
the Compliance & Plain-Language Advisor Agent and the Mortgage Application
Quality Control Agent as RESTful microservices.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import agent routers
from .compliance.api import router as compliance_router
from .quality_control.api import router as quality_control_router

# Create FastAPI application
app = FastAPI(
    title="MortgageAI Agents API",
    description="AI-powered agents for mortgage advice compliance and quality control",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include agent routers
app.include_router(compliance_router)
app.include_router(quality_control_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MortgageAI Agents API",
        "description": "AI-powered agents for mortgage advice compliance and quality control",
        "version": "1.0.0",
        "agents": {
            "compliance": "/api/compliance",
            "quality_control": "/api/quality-control"
        },
        "docs": "/docs"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Overall health check for all agents."""
    return {
        "status": "healthy",
        "service": "agents-api",
        "version": "1.0.0",
        "agents": {
            "compliance": "available",
            "quality_control": "available"
        }
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )