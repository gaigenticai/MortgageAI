"""
API endpoints for the Compliance & Plain-Language Advisor Agent.

This module provides production-grade RESTful endpoints for:
- Generating mortgage advice drafts with advanced async processing
- Checking compliance with concurrency control
- Simplifying language with rate limiting and caching
- Explain-back validation with comprehensive monitoring

Features:
- Advanced async processing with concurrency control
- Rate limiting and request caching
- Comprehensive input validation with pydantic
- Enhanced error handling with structured logging
- Modular architecture with dependency injection
- Health checks and metrics endpoints
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import structlog
from cachetools import TTLCache
import hashlib
import json

from .agent import ComplianceAgent
from ..database import log_agent_interaction, update_application_status
from ..config import settings
from ..utils.text_processor import TextProcessor


# Pydantic models for request/response validation with enhanced validation
class UserProfile(BaseModel):
    """User profile for mortgage advice generation with comprehensive validation."""
    buyer_type: str = Field(..., description="Type of buyer (first-time, moving, remortgaging)")
    mortgage_amount: float = Field(..., gt=0, le=10000000, description="Mortgage amount in euros (0 < amount <= 10M)")
    property_value: float = Field(..., gt=0, description="Property value in euros")
    income: float = Field(..., gt=0, description="Annual income in euros")
    interest_preference: str = Field(..., description="Interest rate preference (fixed/variable)")

    @validator('buyer_type')
    def validate_buyer_type(cls, v):
        allowed_types = ['first-time', 'moving', 'remortgaging', 'buy-to-let']
        if v not in allowed_types:
            raise ValueError(f'buyer_type must be one of {allowed_types}')
        return v

    @validator('interest_preference')
    def validate_interest_preference(cls, v):
        allowed_prefs = ['fixed', 'variable', 'tracker']
        if v not in allowed_prefs:
            raise ValueError(f'interest_preference must be one of {allowed_prefs}')
        return v

    @validator('mortgage_amount', 'property_value', 'income')
    def validate_positive_amounts(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v

class AdviceRequest(BaseModel):
    """Request model for advice generation with sanitization."""
    user_profile: UserProfile
    product_features: list[str] = Field(..., min_items=1, max_items=10, description="List of mortgage product features to explain")
    application_id: str = Field(..., min_length=1, max_length=50, description="Mortgage application ID")

    @validator('product_features')
    def validate_product_features(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one product feature must be specified')
        # Sanitize features
        return [feature.strip()[:100] for feature in v if feature.strip()]

class ComplianceCheckRequest(BaseModel):
    """Request model for compliance checking with text validation."""
    advice_text: str = Field(..., min_length=10, max_length=50000, description="Advice text to check for compliance")
    application_id: str = Field(..., min_length=1, max_length=50, description="Mortgage application ID")

class SimplificationRequest(BaseModel):
    """Request model for text simplification with enhanced validation."""
    advice_text: str = Field(..., min_length=10, max_length=50000, description="Original advice text")
    compliance_issues: list[str] = Field(default=[], max_items=20, description="List of compliance issues to address")
    application_id: str = Field(..., min_length=1, max_length=50, description="Mortgage application ID")

class ExplainBackRequest(BaseModel):
    """Request model for explain-back embedding."""
    advice_text: str = Field(..., min_length=10, max_length=50000, description="Advice text to add explain-back to")
    application_id: str = Field(..., min_length=1, max_length=50, description="Mortgage application ID")

# Response models for better API documentation
class APIResponse(BaseModel):
    """Base response model with metadata."""
    success: bool
    data: Dict[str, Any]
    processing_time_ms: int
    request_id: Optional[str] = None
    timestamp: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    agent: str
    version: str
    uptime_seconds: float
    active_requests: int
    cache_stats: Dict[str, Any]
    timestamp: float


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Router for compliance endpoints
router = APIRouter(prefix="/api/compliance", tags=["compliance"])
logger = structlog.get_logger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
router.state.limiter = limiter

# Concurrency control
concurrency_semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests
processing_semaphore = asyncio.Semaphore(5)   # Max 5 concurrent AI processing tasks

# Request caching
response_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minute TTL
compliance_cache = TTLCache(maxsize=500, ttl=600)  # 10 minute TTL for compliance checks

# Metrics tracking
request_metrics = {
    'total_requests': 0,
    'active_requests': 0,
    'errors': 0,
    'avg_processing_time': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'rate_limit_hits': 0
}
start_time = time.time()

# Dependency injection container
class ServiceContainer:
    """Dependency injection container for better testability and modularity."""

    def __init__(self):
        self._compliance_agent = None
        self._text_processor = None

    @property
    def compliance_agent(self) -> ComplianceAgent:
        if self._compliance_agent is None:
            self._compliance_agent = ComplianceAgent()
        return self._compliance_agent

    @property
    def text_processor(self) -> TextProcessor:
        if self._text_processor is None:
            self._text_processor = TextProcessor()
        return self._text_processor

# Global service container
services = ServiceContainer()

# Request ID generator
def generate_request_id() -> str:
    """Generate unique request ID for tracking."""
    return hashlib.md5(f"{time.time()}{asyncio.current_task()}".encode()).hexdigest()[:8]


@router.post("/generate-advice", response_model=APIResponse)
@limiter.limit("10/minute")
async def generate_advice(
    request: AdviceRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """
    Generate mortgage advice draft based on user profile and product features.

    This endpoint implements advanced async processing with:
    - Concurrency control and rate limiting
    - Request caching for identical requests
    - Comprehensive input validation
    - Structured logging and monitoring
    - Parallel processing where applicable
    """
    request_id = generate_request_id()
    start_time = time.time()
    request_metrics['total_requests'] += 1
    request_metrics['active_requests'] += 1

    # Create cache key
    cache_key = hashlib.md5(
        f"{request.user_profile.json()}{sorted(request.product_features)}{request.application_id}".encode()
    ).hexdigest()

    try:
        # Check cache first
        if cache_key in response_cache:
            request_metrics['cache_hits'] += 1
            cached_result = response_cache[cache_key]
            logger.info("Cache hit for advice generation", request_id=request_id, application_id=request.application_id)
            return APIResponse(
                success=True,
                data=cached_result,
                processing_time_ms=0,
                request_id=request_id,
                timestamp=time.time()
            )

        request_metrics['cache_misses'] += 1

        # Concurrency control
        async with concurrency_semaphore:
            async with processing_semaphore:
                logger.info(
                    "Starting advice generation",
                    request_id=request_id,
                    application_id=request.application_id,
                    buyer_type=request.user_profile.buyer_type,
                    mortgage_amount=request.user_profile.mortgage_amount
                )

                # Parallel processing: Generate advice and validate input concurrently
                advice_task = services.compliance_agent.generate_advice_draft(
                    user_profile=request.user_profile.dict(),
                    product_features=request.product_features
                )

                # Additional validation using text processor
                validation_task = services.text_processor.analyze_readability(
                    f"User profile: {request.user_profile.json()}\nFeatures: {', '.join(request.product_features)}"
                )

                # Execute both tasks concurrently
                result, validation_result = await asyncio.gather(advice_task, validation_task, return_exceptions=True)

                # Handle exceptions from parallel tasks
                if isinstance(result, Exception):
                    raise result
                if isinstance(validation_result, Exception):
                    logger.warning(f"Validation failed but continuing: {validation_result}")

                # Calculate processing time
                processing_time = int((time.time() - start_time) * 1000)

                # Cache successful result
                response_cache[cache_key] = result

                # Log interaction in background with structured data
                background_tasks.add_task(
                    log_agent_interaction,
                    request.application_id,
                    "compliance",
                    "generate_advice",
                    {
                        "user_profile": request.user_profile.dict(),
                        "product_features": request.product_features,
                        "validation_score": validation_result.get('flesch_reading_ease', 0) if not isinstance(validation_result, Exception) else 0
                    },
                    result,
                    processing_time,
                    True,
                    request_id=request_id
                )

                # Update application status
                background_tasks.add_task(
                    update_application_status,
                    request.application_id,
                    "advice_generated",
                    {"advice_draft": result["content"], "request_id": request_id}
                )

                # Update metrics
                request_metrics['avg_processing_time'] = (
                    (request_metrics['avg_processing_time'] * (request_metrics['total_requests'] - 1)) + processing_time
                ) / request_metrics['total_requests']

                logger.info(
                    "Advice generation completed successfully",
                    request_id=request_id,
                    application_id=request.application_id,
                    processing_time_ms=processing_time,
                    cache_hit=False
                )

                return APIResponse(
                    success=True,
                    data=result,
                    processing_time_ms=processing_time,
                    request_id=request_id,
                    timestamp=time.time()
                )

    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        request_metrics['errors'] += 1

        # Log failed interaction with structured data
        background_tasks.add_task(
            log_agent_interaction,
            request.application_id,
            "compliance",
            "generate_advice",
            {
                "user_profile": request.user_profile.dict(),
                "product_features": request.product_features,
                "error_type": type(e).__name__
            },
            {},
            processing_time,
            False,
            str(e),
            request_id=request_id
        )

        logger.error(
            "Advice generation failed",
            request_id=request_id,
            application_id=request.application_id,
            error=str(e),
            processing_time_ms=processing_time,
            exc_info=True
        )

        # Return structured error response
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to generate advice",
                "message": str(e),
                "request_id": request_id,
                "processing_time_ms": processing_time
            }
        )

    finally:
        request_metrics['active_requests'] -= 1


@router.post("/check-compliance", response_model=APIResponse)
@limiter.limit("20/minute")
async def check_compliance(
    request: ComplianceCheckRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """
    Check advice text for AFM compliance and regulatory adherence.

    This endpoint implements advanced compliance checking with:
    - Parallel processing of multiple compliance checks
    - Intelligent caching based on content hash
    - Comprehensive regulatory validation
    - Structured monitoring and alerting
    """
    request_id = generate_request_id()
    start_time = time.time()
    request_metrics['total_requests'] += 1
    request_metrics['active_requests'] += 1

    # Create cache key based on content hash
    content_hash = hashlib.md5(request.advice_text.encode()).hexdigest()
    cache_key = f"compliance:{content_hash}"

    try:
        # Check cache first
        if cache_key in compliance_cache:
            request_metrics['cache_hits'] += 1
            cached_result = compliance_cache[cache_key]
            logger.info("Cache hit for compliance check", request_id=request_id, application_id=request.application_id)
            return APIResponse(
                success=True,
                data=cached_result,
                processing_time_ms=0,
                request_id=request_id,
                timestamp=time.time()
            )

        request_metrics['cache_misses'] += 1

        async with concurrency_semaphore:
            async with processing_semaphore:
                logger.info(
                    "Starting compliance check",
                    request_id=request_id,
                    application_id=request.application_id,
                    text_length=len(request.advice_text)
                )

                # Parallel processing: Run compliance check and text analysis concurrently
                compliance_task = services.compliance_agent.check_compliance(request.advice_text)
                readability_task = services.text_processor.analyze_readability(request.advice_text)
                jargon_task = services.text_processor.detect_jargon(request.advice_text)

                # Execute all tasks concurrently
                compliance_result, readability_result, jargon_result = await asyncio.gather(
                    compliance_task, readability_task, jargon_task, return_exceptions=True
                )

                # Handle exceptions
                if isinstance(compliance_result, Exception):
                    raise compliance_result

                # Log warnings for analysis failures but don't fail the request
                if isinstance(readability_result, Exception):
                    logger.warning(f"Readability analysis failed: {readability_result}")
                    readability_result = {}
                if isinstance(jargon_result, Exception):
                    logger.warning(f"Jargon detection failed: {jargon_result}")
                    jargon_result = []

                # Enhance compliance result with additional analysis
                enhanced_result = compliance_result.copy()
                enhanced_result['readability_analysis'] = readability_result
                enhanced_result['jargon_detected'] = jargon_result
                enhanced_result['content_hash'] = content_hash

                # Calculate processing time
                processing_time = int((time.time() - start_time) * 1000)

                # Cache successful result
                compliance_cache[cache_key] = enhanced_result

                # Log interaction with enhanced data
                background_tasks.add_task(
                    log_agent_interaction,
                    request.application_id,
                    "compliance",
                    "check_compliance",
                    {
                        "advice_text_length": len(request.advice_text),
                        "content_hash": content_hash,
                        "jargon_count": len(jargon_result) if not isinstance(jargon_result, Exception) else 0,
                        "readability_score": readability_result.get('flesch_reading_ease', 0) if not isinstance(readability_result, Exception) else 0
                    },
                    enhanced_result,
                    processing_time,
                    True,
                    request_id=request_id
                )

                # Update metrics
                request_metrics['avg_processing_time'] = (
                    (request_metrics['avg_processing_time'] * (request_metrics['total_requests'] - 1)) + processing_time
                ) / request_metrics['total_requests']

                logger.info(
                    "Compliance check completed",
                    request_id=request_id,
                    application_id=request.application_id,
                    score=enhanced_result.get('score', 0),
                    passed=enhanced_result.get('passed', False),
                    processing_time_ms=processing_time,
                    cache_hit=False
                )

                return APIResponse(
                    success=True,
                    data=enhanced_result,
                    processing_time_ms=processing_time,
                    request_id=request_id,
                    timestamp=time.time()
                )

    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        request_metrics['errors'] += 1

        background_tasks.add_task(
            log_agent_interaction,
            request.application_id,
            "compliance",
            "check_compliance",
            {
                "advice_text_length": len(request.advice_text),
                "content_hash": content_hash,
                "error_type": type(e).__name__
            },
            {},
            processing_time,
            False,
            str(e),
            request_id=request_id
        )

        logger.error(
            "Compliance check failed",
            request_id=request_id,
            application_id=request.application_id,
            error=str(e),
            processing_time_ms=processing_time,
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to check compliance",
                "message": str(e),
                "request_id": request_id,
                "processing_time_ms": processing_time
            }
        )

    finally:
        request_metrics['active_requests'] -= 1


@router.post("/simplify-language", response_model=APIResponse)
@limiter.limit("15/minute")
async def simplify_language(
    request: SimplificationRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """
    Simplify complex language to meet CEFR B1 readability requirements.

    This endpoint implements advanced text simplification with:
    - Parallel processing of multiple simplification techniques
    - Intelligent caching based on content and issues
    - Comprehensive readability assessment
    - Integration with text processing agents
    """
    request_id = generate_request_id()
    start_time = time.time()
    request_metrics['total_requests'] += 1
    request_metrics['active_requests'] += 1

    # Create cache key
    cache_key = hashlib.md5(
        f"{request.advice_text}{sorted(request.compliance_issues)}{request.application_id}".encode()
    ).hexdigest()

    try:
        # Check cache first
        if cache_key in response_cache:
            request_metrics['cache_hits'] += 1
            cached_result = response_cache[cache_key]
            logger.info("Cache hit for language simplification", request_id=request_id, application_id=request.application_id)
            return APIResponse(
                success=True,
                data=cached_result,
                processing_time_ms=0,
                request_id=request_id,
                timestamp=time.time()
            )

        request_metrics['cache_misses'] += 1

        async with concurrency_semaphore:
            async with processing_semaphore:
                logger.info(
                    "Starting language simplification",
                    request_id=request_id,
                    application_id=request.application_id,
                    text_length=len(request.advice_text),
                    issues_count=len(request.compliance_issues)
                )

                # Parallel processing: Run compliance agent simplification and text processor analysis
                simplification_task = services.compliance_agent.simplify_language(
                    request.advice_text,
                    request.compliance_issues
                )

                # Additional text processing for enhanced analysis
                original_analysis_task = services.text_processor.analyze_readability(request.advice_text)
                jargon_detection_task = services.text_processor.detect_jargon(request.advice_text)

                # Execute tasks concurrently
                simplification_result, original_analysis, jargon_detected = await asyncio.gather(
                    simplification_task, original_analysis_task, jargon_detection_task, return_exceptions=True
                )

                # Handle exceptions
                if isinstance(simplification_result, Exception):
                    raise simplification_result

                # Analyze simplified text if available
                simplified_text = simplification_result.get('advanced_simplified_text', request.advice_text)
                simplified_analysis_task = services.text_processor.analyze_readability(simplified_text)

                try:
                    simplified_analysis = await simplified_analysis_task
                except Exception as e:
                    logger.warning(f"Simplified text analysis failed: {e}")
                    simplified_analysis = {}

                # Enhance result with additional analysis
                enhanced_result = simplification_result.copy()
                enhanced_result['original_analysis'] = original_analysis if not isinstance(original_analysis, Exception) else {}
                enhanced_result['jargon_detected'] = jargon_detected if not isinstance(jargon_detected, Exception) else []
                enhanced_result['simplified_analysis'] = simplified_analysis

                # Calculate processing time
                processing_time = int((time.time() - start_time) * 1000)

                # Cache successful result
                response_cache[cache_key] = enhanced_result

                # Log interaction with enhanced data
                background_tasks.add_task(
                    log_agent_interaction,
                    request.application_id,
                    "compliance",
                    "simplify_language",
                    {
                        "original_length": len(request.advice_text),
                        "issues_count": len(request.compliance_issues),
                        "jargon_count": len(jargon_detected) if not isinstance(jargon_detected, Exception) else 0,
                        "original_readability": original_analysis.get('flesch_reading_ease', 0) if not isinstance(original_analysis, Exception) else 0,
                        "simplified_readability": simplified_analysis.get('flesch_reading_ease', 0)
                    },
                    enhanced_result,
                    processing_time,
                    True,
                    request_id=request_id
                )

                # Update metrics
                request_metrics['avg_processing_time'] = (
                    (request_metrics['avg_processing_time'] * (request_metrics['total_requests'] - 1)) + processing_time
                ) / request_metrics['total_requests']

                logger.info(
                    "Language simplification completed",
                    request_id=request_id,
                    application_id=request.application_id,
                    readability_improvement=enhanced_result.get('comprehensive_improvements', {}).get('cefr_b1_score', 0),
                    processing_time_ms=processing_time,
                    cache_hit=False
                )

                return APIResponse(
                    success=True,
                    data=enhanced_result,
                    processing_time_ms=processing_time,
                    request_id=request_id,
                    timestamp=time.time()
                )

    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        request_metrics['errors'] += 1

        background_tasks.add_task(
            log_agent_interaction,
            request.application_id,
            "compliance",
            "simplify_language",
            {
                "original_length": len(request.advice_text),
                "issues_count": len(request.compliance_issues),
                "error_type": type(e).__name__
            },
            {},
            processing_time,
            False,
            str(e),
            request_id=request_id
        )

        logger.error(
            "Language simplification failed",
            request_id=request_id,
            application_id=request.application_id,
            error=str(e),
            processing_time_ms=processing_time,
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to simplify language",
                "message": str(e),
                "request_id": request_id,
                "processing_time_ms": processing_time
            }
        )

    finally:
        request_metrics['active_requests'] -= 1


@router.post("/embed-explain-back", response_model=APIResponse)
@limiter.limit("12/minute")
async def embed_explain_back(
    request: ExplainBackRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """
    Embed explain-back questions to validate user understanding.

    This endpoint implements advanced comprehension validation with:
    - Parallel processing of question generation and text analysis
    - Intelligent caching for repeated content
    - Comprehensive question quality assessment
    - Integration with text processing for context awareness
    """
    request_id = generate_request_id()
    start_time = time.time()
    request_metrics['total_requests'] += 1
    request_metrics['active_requests'] += 1

    # Create cache key
    cache_key = hashlib.md5(
        f"{request.advice_text}{request.application_id}".encode()
    ).hexdigest()

    try:
        # Check cache first
        if cache_key in response_cache:
            request_metrics['cache_hits'] += 1
            cached_result = response_cache[cache_key]
            logger.info("Cache hit for explain-back embedding", request_id=request_id, application_id=request.application_id)
            return APIResponse(
                success=True,
                data=cached_result,
                processing_time_ms=0,
                request_id=request_id,
                timestamp=time.time()
            )

        request_metrics['cache_misses'] += 1

        async with concurrency_semaphore:
            async with processing_semaphore:
                logger.info(
                    "Starting explain-back embedding",
                    request_id=request_id,
                    application_id=request.application_id,
                    text_length=len(request.advice_text)
                )

                # Parallel processing: Generate explain-back and analyze text
                explain_back_task = services.compliance_agent.embed_explain_back(request.advice_text)
                text_analysis_task = services.text_processor.analyze_readability(request.advice_text)
                comprehension_task = services.text_processor.calculate_comprehension_score(request.advice_text)

                # Execute tasks concurrently
                explain_back_result, text_analysis, comprehension_score = await asyncio.gather(
                    explain_back_task, text_analysis_task, comprehension_task, return_exceptions=True
                )

                # Handle exceptions
                if isinstance(explain_back_result, Exception):
                    raise explain_back_result

                # Enhance result with additional analysis
                enhanced_result = explain_back_result.copy()
                enhanced_result['text_analysis'] = text_analysis if not isinstance(text_analysis, Exception) else {}
                enhanced_result['comprehension_score'] = comprehension_score if not isinstance(comprehension_score, Exception) else 0.0
                enhanced_result['question_quality_metrics'] = {
                    'total_questions': len(explain_back_result.get('explain_back_questions', [])),
                    'avg_question_length': sum(len(q.get('question', '')) for q in explain_back_result.get('explain_back_questions', [])) / max(len(explain_back_result.get('explain_back_questions', [])), 1)
                }

                # Calculate processing time
                processing_time = int((time.time() - start_time) * 1000)

                # Cache successful result
                response_cache[cache_key] = enhanced_result

                # Log interaction with enhanced data
                background_tasks.add_task(
                    log_agent_interaction,
                    request.application_id,
                    "compliance",
                    "embed_explain_back",
                    {
                        "advice_text_length": len(request.advice_text),
                        "questions_generated": len(explain_back_result.get('explain_back_questions', [])),
                        "comprehension_score": comprehension_score if not isinstance(comprehension_score, Exception) else 0.0,
                        "readability_score": text_analysis.get('flesch_reading_ease', 0) if not isinstance(text_analysis, Exception) else 0
                    },
                    enhanced_result,
                    processing_time,
                    True,
                    request_id=request_id
                )

                # Update application with final advice
                background_tasks.add_task(
                    update_application_status,
                    request.application_id,
                    "advice_finalized",
                    {
                        "final_advice": explain_back_result.get("enhanced_advice", ""),
                        "request_id": request_id,
                        "comprehension_score": comprehension_score if not isinstance(comprehension_score, Exception) else 0.0
                    }
                )

                # Update metrics
                request_metrics['avg_processing_time'] = (
                    (request_metrics['avg_processing_time'] * (request_metrics['total_requests'] - 1)) + processing_time
                ) / request_metrics['total_requests']

                logger.info(
                    "Explain-back embedding completed",
                    request_id=request_id,
                    application_id=request.application_id,
                    questions_added=len(explain_back_result.get('explain_back_questions', [])),
                    comprehension_score=comprehension_score if not isinstance(comprehension_score, Exception) else 0.0,
                    processing_time_ms=processing_time,
                    cache_hit=False
                )

                return APIResponse(
                    success=True,
                    data=enhanced_result,
                    processing_time_ms=processing_time,
                    request_id=request_id,
                    timestamp=time.time()
                )

    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        request_metrics['errors'] += 1

        background_tasks.add_task(
            log_agent_interaction,
            request.application_id,
            "compliance",
            "embed_explain_back",
            {
                "advice_text_length": len(request.advice_text),
                "error_type": type(e).__name__
            },
            {},
            processing_time,
            False,
            str(e),
            request_id=request_id
        )

        logger.error(
            "Explain-back embedding failed",
            request_id=request_id,
            application_id=request.application_id,
            error=str(e),
            processing_time_ms=processing_time,
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to embed explain-back",
                "message": str(e),
                "request_id": request_id,
                "processing_time_ms": processing_time
            }
        )

    finally:
        request_metrics['active_requests'] -= 1


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint with system metrics and monitoring.

    Returns detailed health status including:
    - Service availability and version info
    - Performance metrics and request statistics
    - Cache status and memory usage
    - Active request counts and error rates
    """
    uptime = time.time() - start_time

    # Check service dependencies
    agent_healthy = True
    text_processor_healthy = True

    try:
        # Quick health check of compliance agent
        test_result = await services.compliance_agent.check_compliance("test")
        agent_healthy = isinstance(test_result, dict)
    except Exception as e:
        agent_healthy = False
        logger.warning(f"Compliance agent health check failed: {e}")

    try:
        # Quick health check of text processor
        test_result = await services.text_processor.analyze_readability("test")
        text_processor_healthy = isinstance(test_result, dict)
    except Exception as e:
        text_processor_healthy = False
        logger.warning(f"Text processor health check failed: {e}")

    # Determine overall health status
    overall_healthy = agent_healthy and text_processor_healthy
    status = "healthy" if overall_healthy else "degraded"

    # Cache statistics
    cache_stats = {
        'response_cache_size': len(response_cache),
        'response_cache_maxsize': response_cache.maxsize,
        'compliance_cache_size': len(compliance_cache),
        'compliance_cache_maxsize': compliance_cache.maxsize,
        'cache_hit_rate': (
            request_metrics['cache_hits'] /
            max(request_metrics['total_requests'], 1)
        ) if request_metrics['total_requests'] > 0 else 0
    }

    health_data = HealthResponse(
        status=status,
        agent="compliance",
        version="2.0.0",  # Updated version for production-grade implementation
        uptime_seconds=uptime,
        active_requests=request_metrics['active_requests'],
        cache_stats=cache_stats,
        timestamp=time.time()
    )

    # Log health check with additional context
    logger.info(
        "Health check performed",
        status=status,
        agent_healthy=agent_healthy,
        text_processor_healthy=text_processor_healthy,
        active_requests=request_metrics['active_requests'],
        total_requests=request_metrics['total_requests'],
        error_rate=request_metrics['errors'] / max(request_metrics['total_requests'], 1),
        avg_processing_time=request_metrics['avg_processing_time']
    )

    # Return appropriate status code
    if not overall_healthy:
        # Return 503 Service Unavailable if services are degraded
        raise HTTPException(
            status_code=503,
            detail={
                "status": status,
                "agent_healthy": agent_healthy,
                "text_processor_healthy": text_processor_healthy,
                "message": "One or more services are unavailable"
            }
        )

    return health_data

@router.get("/metrics")
async def get_metrics():
    """
    Detailed metrics endpoint for monitoring and alerting.

    Returns comprehensive system metrics for:
    - Request statistics and performance
    - Cache performance
    - Error rates and health indicators
    - Concurrency and resource usage
    """
    return {
        'timestamp': time.time(),
        'uptime_seconds': time.time() - start_time,
        'version': '2.0.0',
        'request_metrics': request_metrics.copy(),
        'concurrency_limits': {
            'max_concurrent_requests': 10,
            'max_concurrent_processing': 5,
            'current_active_requests': request_metrics['active_requests']
        },
        'cache_performance': {
            'response_cache_size': len(response_cache),
            'compliance_cache_size': len(compliance_cache),
            'cache_hit_rate': (
                request_metrics['cache_hits'] /
                max(request_metrics['total_requests'], 1)
            ) if request_metrics['total_requests'] > 0 else 0
        },
        'health_indicators': {
            'error_rate': request_metrics['errors'] / max(request_metrics['total_requests'], 1),
            'avg_response_time_ms': request_metrics['avg_processing_time'],
            'cache_effectiveness': len(response_cache) / response_cache.maxsize
        }
    }


# Function call implementations as specified in Spec.md with production-grade enhancements
async def generateAdviceDraft(advice_text: str, user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Function call: generateAdviceDraft(advice_text, user_profile) → checkCompliance() → simplifyLanguage() → embedExplainBack()

    This implements the complete advice generation pipeline with:
    - Advanced async processing and concurrency control
    - Parallel execution of pipeline steps where possible
    - Comprehensive error handling and monitoring
    - Integration with text processing agents
    """
    pipeline_id = generate_request_id()
    start_time = time.time()

    try:
        logger.info(
            "Starting advice generation pipeline",
            pipeline_id=pipeline_id,
            user_profile_keys=list(user_profile.keys()),
            advice_text_length=len(advice_text)
        )

        # Step 1: Generate initial draft (advice_text is provided)
        draft_result = {
            "content": advice_text,
            "generated_at": time.time(),
            "pipeline_step": "draft_generated",
            "pipeline_id": pipeline_id
        }

        # Step 2 & 3: Parallel compliance check and initial text analysis
        compliance_task = services.compliance_agent.check_compliance(advice_text)
        analysis_task = services.text_processor.analyze_readability(advice_text)

        compliance_result, analysis_result = await asyncio.gather(
            compliance_task, analysis_task, return_exceptions=True
        )

        if isinstance(compliance_result, Exception):
            raise compliance_result

        # Log compliance check results
        logger.info(
            "Compliance check completed in pipeline",
            pipeline_id=pipeline_id,
            compliance_score=compliance_result.get('score', 0),
            issues_count=len(compliance_result.get('issues', []))
        )

        # Step 4: Conditional simplification with parallel processing
        simplification_result = None
        if not compliance_result["passed"]:
            logger.info(
                "Language simplification required",
                pipeline_id=pipeline_id,
                issues=compliance_result["issues"]
            )

            # Parallel simplification and jargon detection
            simplify_task = services.compliance_agent.simplify_language(
                advice_text,
                compliance_result["issues"]
            )
            jargon_task = services.text_processor.detect_jargon(advice_text)

            simplification_result, jargon_result = await asyncio.gather(
                simplify_task, jargon_task, return_exceptions=True
            )

            if isinstance(simplification_result, Exception):
                logger.warning(f"Simplification failed but continuing: {simplification_result}")
                simplification_result = None
            else:
                advice_text = simplification_result.get("advanced_simplified_text", advice_text)
                logger.info(
                    "Language simplification completed",
                    pipeline_id=pipeline_id,
                    readability_improvement=simplification_result.get('comprehensive_improvements', {}).get('cefr_b1_score', 0)
                )

        # Step 5: Embed explain-back with enhanced analysis
        explain_back_task = services.compliance_agent.embed_explain_back(advice_text)
        final_analysis_task = services.text_processor.calculate_comprehension_score(advice_text)

        final_result, final_comprehension = await asyncio.gather(
            explain_back_task, final_analysis_task, return_exceptions=True
        )

        if isinstance(final_result, Exception):
            raise final_result

        # Calculate total pipeline processing time
        total_processing_time = int((time.time() - start_time) * 1000)

        pipeline_result = {
            "draft": draft_result,
            "compliance_check": compliance_result,
            "simplification": simplification_result,
            "final_advice": final_result if not isinstance(final_result, Exception) else None,
            "pipeline_completed": True,
            "pipeline_id": pipeline_id,
            "total_processing_time_ms": total_processing_time,
            "pipeline_steps_completed": 4 if simplification_result else 3,
            "final_comprehension_score": final_comprehension if not isinstance(final_comprehension, Exception) else 0.0,
            "enhanced_analysis": {
                "initial_readability": analysis_result.get('flesch_reading_ease', 0) if not isinstance(analysis_result, Exception) else 0,
                "final_comprehension": final_comprehension if not isinstance(final_comprehension, Exception) else 0.0,
                "compliance_score": compliance_result.get('score', 0),
                "questions_added": len(final_result.get('explain_back_questions', [])) if not isinstance(final_result, Exception) else 0
            }
        }

        logger.info(
            "Advice generation pipeline completed successfully",
            pipeline_id=pipeline_id,
            total_processing_time_ms=total_processing_time,
            steps_completed=pipeline_result["pipeline_steps_completed"],
            final_comprehension_score=pipeline_result["final_comprehension_score"]
        )

        return pipeline_result

    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)

        logger.error(
            "Advice generation pipeline failed",
            pipeline_id=pipeline_id,
            error=str(e),
            processing_time_ms=processing_time,
            exc_info=True
        )

        return {
            "error": str(e),
            "pipeline_completed": False,
            "pipeline_id": pipeline_id,
            "processing_time_ms": processing_time,
            "partial_results": {
                "draft": locals().get('draft_result'),
                "compliance_check": locals().get('compliance_result') if 'compliance_result' in locals() and not isinstance(locals()['compliance_result'], Exception) else None
            }
        }
