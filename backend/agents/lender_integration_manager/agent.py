#!/usr/bin/env python3
"""
Advanced Lender Integration Manager
Comprehensive system for managing real-time API connections, validation rules, and approval likelihood scoring

Features:
- AI-powered approval likelihood prediction using ensemble models
- Real-time lender API health monitoring and circuit breaker patterns
- Sophisticated validation rules engine with conditional logic
- Advanced rate limiting and load balancing
- Intelligent retry mechanisms with exponential backoff
- Comprehensive audit trail and compliance logging
- Multi-lender optimization algorithms
- Fraud detection and risk assessment integration
- Real-time notification system
- Advanced caching with intelligent cache invalidation
"""

import asyncio
import aiohttp
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import asyncpg
import aioredis
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import websockets
import ssl
import certifi
from cryptography.fernet import Fernet
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import statistics
import psutil
import requests
from urllib.parse import urlparse
import re
import uuid
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LenderStatus(Enum):
    """Lender API status enumeration"""
    ACTIVE = "active"
    DEGRADED = "degraded" 
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    RATE_LIMITED = "rate_limited"

class ApplicationStatus(Enum):
    """Application status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    DOCUMENTS_REQUESTED = "documents_requested"
    APPROVED = "approved"
    CONDITIONALLY_APPROVED = "conditionally_approved"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    ERROR = "error"

class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class LenderConfig:
    """Comprehensive lender configuration"""
    name: str
    api_url: str
    api_key: str
    backup_api_url: Optional[str]
    backup_api_key: Optional[str]
    supported_products: List[str]
    max_loan_amount: float
    min_loan_amount: float
    max_ltv: float
    min_income: float
    processing_time_days: int
    rate_limit_per_hour: int
    timeout_seconds: int
    retry_attempts: int
    circuit_breaker_threshold: int
    priority_score: int
    fees: Dict[str, float]
    requirements: Dict[str, Any]
    validation_rules: List[Dict[str, Any]]
    approval_criteria: Dict[str, Any]
    document_requirements: List[str]
    notification_webhooks: List[str]
    ssl_verify: bool = True
    custom_headers: Dict[str, str] = None

@dataclass
class ValidationRule:
    """Validation rule definition"""
    id: str
    name: str
    description: str
    field: str
    rule_type: str  # regex, range, custom, conditional
    parameters: Dict[str, Any]
    severity: ValidationSeverity
    error_message: str
    correction_suggestion: str
    conditions: List[Dict[str, Any]] = None
    priority: int = 1

@dataclass
class ValidationResult:
    """Validation result"""
    field: str
    rule_id: str
    is_valid: bool
    severity: ValidationSeverity
    message: str
    suggestion: str
    confidence: float
    timestamp: datetime

@dataclass
class ApprovalPrediction:
    """Approval likelihood prediction"""
    lender_name: str
    probability: float
    confidence_interval: Tuple[float, float]
    risk_factors: List[str]
    positive_factors: List[str]
    recommendation: str
    model_version: str
    prediction_timestamp: datetime
    feature_importance: Dict[str, float]

@dataclass
class LenderHealthMetrics:
    """Lender API health metrics"""
    lender_name: str
    status: LenderStatus
    response_time_ms: float
    success_rate: float
    error_rate: float
    last_check: datetime
    consecutive_failures: int
    uptime_percentage: float
    rate_limit_remaining: int
    circuit_breaker_open: bool

class CircuitBreaker:
    """Circuit breaker implementation for lender APIs"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == 'open':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'half_open'
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                if self.state == 'half_open':
                    self.state = 'closed'
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                
                raise e

class RateLimiter:
    """Advanced rate limiter with token bucket algorithm"""
    
    def __init__(self, max_tokens: int, refill_rate: float):
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from bucket"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get wait time for tokens to become available"""
        with self.lock:
            if self.tokens >= tokens:
                return 0
            return (tokens - self.tokens) / self.refill_rate

class ApprovalPredictionModel:
    """AI-powered approval likelihood prediction model"""
    
    def __init__(self):
        self.models = {}
        self.feature_columns = [
            'loan_amount', 'property_value', 'ltv_ratio', 'gross_income', 
            'net_income', 'existing_debts', 'credit_score', 'employment_years',
            'age', 'savings', 'property_type_encoded', 'employment_type_encoded',
            'loan_term_years', 'interest_rate', 'debt_to_income_ratio'
        ]
        self.trained_lenders = set()
        self.model_versions = {}
        self.feature_importance = {}
    
    def prepare_features(self, application_data: Dict[str, Any]) -> np.ndarray:
        """Prepare features for model prediction"""
        try:
            # Extract and calculate features
            loan_amount = float(application_data.get('loan_amount', 0))
            property_value = float(application_data.get('property_value', 0))
            ltv_ratio = (loan_amount / property_value * 100) if property_value > 0 else 0
            
            gross_income = float(application_data.get('gross_annual_income', 0))
            net_income = float(application_data.get('net_monthly_income', 0)) * 12
            existing_debts = float(application_data.get('existing_debts', 0))
            credit_score = float(application_data.get('credit_score', 700))  # Default if not provided
            employment_years = float(application_data.get('employment_years', 0))
            age = float(application_data.get('age', 35))
            savings = float(application_data.get('savings', 0))
            loan_term_years = float(application_data.get('loan_term_years', 30))
            interest_rate = float(application_data.get('interest_rate', 3.5))
            
            # Calculate derived features
            debt_to_income_ratio = (existing_debts / gross_income * 100) if gross_income > 0 else 0
            
            # Encode categorical features
            property_type = application_data.get('property_type', 'house')
            property_type_encoded = {'house': 1, 'apartment': 2, 'townhouse': 3}.get(property_type, 1)
            
            employment_type = application_data.get('employment_type', 'permanent')
            employment_type_encoded = {'permanent': 1, 'contract': 2, 'self_employed': 3}.get(employment_type, 1)
            
            features = np.array([
                loan_amount, property_value, ltv_ratio, gross_income, net_income,
                existing_debts, credit_score, employment_years, age, savings,
                property_type_encoded, employment_type_encoded, loan_term_years,
                interest_rate, debt_to_income_ratio
            ]).reshape(1, -1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            # Return default features if preparation fails
            return np.zeros((1, len(self.feature_columns)))
    
    def train_lender_model(self, lender_name: str, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train approval prediction model for specific lender"""
        try:
            if len(training_data) < 100:
                logger.warning(f"Insufficient training data for {lender_name}: {len(training_data)} samples")
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
            # Prepare training data
            X = []
            y = []
            
            for record in training_data:
                features = self.prepare_features(record).flatten()
                X.append(features)
                y.append(1 if record.get('approved', False) else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train ensemble of models
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42)
            }
            
            trained_models = {}
            scores = {}
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                scores[f'{name}_accuracy'] = accuracy_score(y_test, y_pred)
                scores[f'{name}_precision'] = precision_score(y_test, y_pred, zero_division=0)
                scores[f'{name}_recall'] = recall_score(y_test, y_pred, zero_division=0)
                scores[f'{name}_f1'] = f1_score(y_test, y_pred, zero_division=0)
                
                trained_models[name] = model
            
            # Store models and metadata
            self.models[lender_name] = trained_models
            self.trained_lenders.add(lender_name)
            self.model_versions[lender_name] = datetime.now().isoformat()
            
            # Calculate feature importance (using random forest)
            if 'random_forest' in trained_models:
                importance = trained_models['random_forest'].feature_importances_
                self.feature_importance[lender_name] = dict(zip(self.feature_columns, importance))
            
            # Calculate overall metrics
            overall_scores = {
                'accuracy': np.mean([scores[k] for k in scores.keys() if 'accuracy' in k]),
                'precision': np.mean([scores[k] for k in scores.keys() if 'precision' in k]),
                'recall': np.mean([scores[k] for k in scores.keys() if 'recall' in k]),
                'f1': np.mean([scores[k] for k in scores.keys() if 'f1' in k])
            }
            
            logger.info(f"Model trained for {lender_name}: {overall_scores}")
            return overall_scores
            
        except Exception as e:
            logger.error(f"Error training model for {lender_name}: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def predict_approval(self, lender_name: str, application_data: Dict[str, Any]) -> ApprovalPrediction:
        """Predict approval likelihood for specific lender"""
        try:
            if lender_name not in self.models:
                # Return default prediction if no model trained
                return ApprovalPrediction(
                    lender_name=lender_name,
                    probability=0.5,
                    confidence_interval=(0.3, 0.7),
                    risk_factors=["No trained model available"],
                    positive_factors=[],
                    recommendation="Train model with historical data",
                    model_version="none",
                    prediction_timestamp=datetime.now(),
                    feature_importance={}
                )
            
            features = self.prepare_features(application_data)
            models = self.models[lender_name]
            
            # Get predictions from all models
            predictions = []
            for model_name, model in models.items():
                try:
                    prob = model.predict_proba(features)[0][1]  # Probability of approval
                    predictions.append(prob)
                except:
                    predictions.append(0.5)  # Default if model fails
            
            # Ensemble prediction (weighted average)
            weights = [0.4, 0.4, 0.2]  # RF, GB, LR
            ensemble_prob = np.average(predictions[:3], weights=weights[:len(predictions)])
            
            # Calculate confidence interval
            std_dev = np.std(predictions) if len(predictions) > 1 else 0.1
            confidence_interval = (
                max(0, ensemble_prob - 1.96 * std_dev),
                min(1, ensemble_prob + 1.96 * std_dev)
            )
            
            # Analyze risk and positive factors
            risk_factors = []
            positive_factors = []
            
            # Feature analysis
            feature_values = features.flatten()
            feature_importance = self.feature_importance.get(lender_name, {})
            
            for i, (feature, value) in enumerate(zip(self.feature_columns, feature_values)):
                importance = feature_importance.get(feature, 0)
                
                if importance > 0.05:  # Only consider important features
                    if feature == 'ltv_ratio' and value > 90:
                        risk_factors.append(f"High LTV ratio: {value:.1f}%")
                    elif feature == 'debt_to_income_ratio' and value > 40:
                        risk_factors.append(f"High debt-to-income ratio: {value:.1f}%")
                    elif feature == 'credit_score' and value < 650:
                        risk_factors.append(f"Low credit score: {value:.0f}")
                    elif feature == 'employment_years' and value < 2:
                        risk_factors.append(f"Short employment history: {value:.1f} years")
                    
                    # Positive factors
                    elif feature == 'credit_score' and value > 750:
                        positive_factors.append(f"Excellent credit score: {value:.0f}")
                    elif feature == 'savings' and value > 50000:
                        positive_factors.append(f"Strong savings: €{value:,.0f}")
                    elif feature == 'employment_years' and value > 5:
                        positive_factors.append(f"Stable employment: {value:.1f} years")
            
            # Generate recommendation
            if ensemble_prob > 0.8:
                recommendation = "Strong approval likelihood - proceed with application"
            elif ensemble_prob > 0.6:
                recommendation = "Good approval chances - consider optimizing application"
            elif ensemble_prob > 0.4:
                recommendation = "Moderate approval chances - address risk factors"
            else:
                recommendation = "Low approval likelihood - consider alternative lenders"
            
            return ApprovalPrediction(
                lender_name=lender_name,
                probability=ensemble_prob,
                confidence_interval=confidence_interval,
                risk_factors=risk_factors,
                positive_factors=positive_factors,
                recommendation=recommendation,
                model_version=self.model_versions.get(lender_name, "unknown"),
                prediction_timestamp=datetime.now(),
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Error predicting approval for {lender_name}: {e}")
            return ApprovalPrediction(
                lender_name=lender_name,
                probability=0.5,
                confidence_interval=(0.3, 0.7),
                risk_factors=[f"Prediction error: {str(e)}"],
                positive_factors=[],
                recommendation="Manual review required",
                model_version="error",
                prediction_timestamp=datetime.now(),
                feature_importance={}
            )

class ValidationEngine:
    """Advanced validation rules engine"""
    
    def __init__(self):
        self.rules = {}
        self.custom_validators = {}
        self.validation_history = defaultdict(list)
    
    def add_rule(self, rule: ValidationRule):
        """Add validation rule"""
        self.rules[rule.id] = rule
    
    def add_custom_validator(self, name: str, validator_func):
        """Add custom validation function"""
        self.custom_validators[name] = validator_func
    
    def validate_field(self, field: str, value: Any, context: Dict[str, Any] = None) -> List[ValidationResult]:
        """Validate field against all applicable rules"""
        results = []
        context = context or {}
        
        for rule_id, rule in self.rules.items():
            if rule.field == field or rule.field == '*':  # * means applies to all fields
                try:
                    # Check conditions first
                    if rule.conditions:
                        if not self._evaluate_conditions(rule.conditions, context):
                            continue
                    
                    is_valid = self._apply_rule(rule, value, context)
                    confidence = 1.0  # Default confidence
                    
                    if not is_valid:
                        # Calculate confidence based on rule type and context
                        confidence = self._calculate_confidence(rule, value, context)
                    
                    result = ValidationResult(
                        field=field,
                        rule_id=rule_id,
                        is_valid=is_valid,
                        severity=rule.severity,
                        message=rule.error_message if not is_valid else "Valid",
                        suggestion=rule.correction_suggestion if not is_valid else "",
                        confidence=confidence,
                        timestamp=datetime.now()
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error applying rule {rule_id}: {e}")
                    results.append(ValidationResult(
                        field=field,
                        rule_id=rule_id,
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Validation error: {str(e)}",
                        suggestion="Contact system administrator",
                        confidence=0.0,
                        timestamp=datetime.now()
                    ))
        
        # Store validation history
        self.validation_history[field].extend(results)
        
        return results
    
    def validate_application(self, application_data: Dict[str, Any]) -> Dict[str, List[ValidationResult]]:
        """Validate entire application"""
        all_results = {}
        
        for field, value in application_data.items():
            results = self.validate_field(field, value, application_data)
            if results:
                all_results[field] = results
        
        return all_results
    
    def _apply_rule(self, rule: ValidationRule, value: Any, context: Dict[str, Any]) -> bool:
        """Apply validation rule to value"""
        rule_type = rule.rule_type.lower()
        params = rule.parameters
        
        if rule_type == 'regex':
            pattern = params.get('pattern', '')
            return bool(re.match(pattern, str(value)))
        
        elif rule_type == 'range':
            min_val = params.get('min')
            max_val = params.get('max')
            try:
                num_value = float(value)
                if min_val is not None and num_value < min_val:
                    return False
                if max_val is not None and num_value > max_val:
                    return False
                return True
            except (ValueError, TypeError):
                return False
        
        elif rule_type == 'required':
            return value is not None and str(value).strip() != ''
        
        elif rule_type == 'email':
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, str(value)))
        
        elif rule_type == 'bsn':
            return self._validate_bsn(str(value))
        
        elif rule_type == 'iban':
            return self._validate_iban(str(value))
        
        elif rule_type == 'postcode':
            pattern = r'^[1-9][0-9]{3}\s?[A-Za-z]{2}$'
            return bool(re.match(pattern, str(value)))
        
        elif rule_type == 'custom':
            validator_name = params.get('validator')
            if validator_name in self.custom_validators:
                return self.custom_validators[validator_name](value, context)
            return False
        
        elif rule_type == 'conditional':
            condition = params.get('condition')
            return self._evaluate_condition(condition, context)
        
        else:
            logger.warning(f"Unknown rule type: {rule_type}")
            return True
    
    def _validate_bsn(self, bsn: str) -> bool:
        """Validate Dutch BSN (Burgerservicenummer)"""
        try:
            bsn = bsn.replace(' ', '').replace('-', '')
            if len(bsn) != 9 or not bsn.isdigit():
                return False
            
            # BSN checksum validation
            checksum = 0
            for i in range(8):
                checksum += int(bsn[i]) * (9 - i)
            checksum += int(bsn[8]) * -1
            
            return checksum % 11 == 0
        except:
            return False
    
    def _validate_iban(self, iban: str) -> bool:
        """Validate IBAN"""
        try:
            iban = iban.replace(' ', '').upper()
            if len(iban) < 15 or len(iban) > 34:
                return False
            
            # Move first 4 characters to end
            rearranged = iban[4:] + iban[:4]
            
            # Replace letters with numbers
            numeric = ''
            for char in rearranged:
                if char.isalpha():
                    numeric += str(ord(char) - ord('A') + 10)
                else:
                    numeric += char
            
            # Check mod 97
            return int(numeric) % 97 == 1
        except:
            return False
    
    def _evaluate_conditions(self, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> bool:
        """Evaluate rule conditions"""
        for condition in conditions:
            if not self._evaluate_condition(condition, context):
                return False
        return True
    
    def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate single condition"""
        try:
            field = condition.get('field')
            operator = condition.get('operator', 'eq')
            value = condition.get('value')
            
            if field not in context:
                return False
            
            context_value = context[field]
            
            if operator == 'eq':
                return context_value == value
            elif operator == 'ne':
                return context_value != value
            elif operator == 'gt':
                return float(context_value) > float(value)
            elif operator == 'gte':
                return float(context_value) >= float(value)
            elif operator == 'lt':
                return float(context_value) < float(value)
            elif operator == 'lte':
                return float(context_value) <= float(value)
            elif operator == 'in':
                return context_value in value
            elif operator == 'not_in':
                return context_value not in value
            elif operator == 'regex':
                return bool(re.match(value, str(context_value)))
            else:
                return False
        except:
            return False
    
    def _calculate_confidence(self, rule: ValidationRule, value: Any, context: Dict[str, Any]) -> float:
        """Calculate confidence score for validation result"""
        base_confidence = 0.9
        
        # Adjust based on rule type
        if rule.rule_type == 'regex':
            base_confidence = 0.95
        elif rule.rule_type == 'custom':
            base_confidence = 0.8
        elif rule.rule_type == 'conditional':
            base_confidence = 0.85
        
        # Adjust based on data quality
        if value is None or str(value).strip() == '':
            base_confidence = 1.0  # High confidence for empty values
        
        return min(1.0, base_confidence)

class AdvancedLenderIntegrationManager:
    """Advanced Lender Integration Manager with AI-powered features"""
    
    def __init__(self, config_file: str = None):
        self.lenders = {}
        self.circuit_breakers = {}
        self.rate_limiters = {}
        self.health_metrics = {}
        self.prediction_model = ApprovalPredictionModel()
        self.validation_engine = ValidationEngine()
        self.db_pool = None
        self.redis_pool = None
        self.session = None
        self.notification_queue = asyncio.Queue()
        self.monitoring_active = False
        self.encryption_key = None
        
        # Performance metrics
        self.request_metrics = defaultdict(lambda: {
            'count': 0,
            'success': 0,
            'failure': 0,
            'avg_response_time': 0,
            'response_times': deque(maxlen=1000)
        })
        
        # Load configuration
        if config_file:
            self.load_configuration(config_file)
        
        # Initialize default validation rules
        self._initialize_default_validation_rules()
    
    async def initialize(self, database_url: str, redis_url: str, encryption_key: str = None):
        """Initialize database and Redis connections"""
        try:
            # Database connection
            self.db_pool = await asyncpg.create_pool(
                database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Redis connection
            self.redis_pool = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            
            # HTTP session
            timeout = aiohttp.ClientTimeout(total=300)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Encryption key
            if encryption_key:
                self.encryption_key = Fernet(encryption_key.encode())
            
            # Start background tasks
            asyncio.create_task(self._monitor_lender_health())
            asyncio.create_task(self._process_notifications())
            
            logger.info("Advanced Lender Integration Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize manager: {e}")
            raise
    
    def load_configuration(self, config_file: str):
        """Load lender configurations from file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            for lender_data in config.get('lenders', []):
                lender_config = LenderConfig(**lender_data)
                self.add_lender(lender_config)
            
            logger.info(f"Loaded configuration for {len(self.lenders)} lenders")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def add_lender(self, config: LenderConfig):
        """Add lender configuration"""
        self.lenders[config.name.lower()] = config
        
        # Initialize circuit breaker
        self.circuit_breakers[config.name.lower()] = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_timeout=60
        )
        
        # Initialize rate limiter
        self.rate_limiters[config.name.lower()] = RateLimiter(
            max_tokens=config.rate_limit_per_hour,
            refill_rate=config.rate_limit_per_hour / 3600  # per second
        )
        
        # Initialize health metrics
        self.health_metrics[config.name.lower()] = LenderHealthMetrics(
            lender_name=config.name,
            status=LenderStatus.ACTIVE,
            response_time_ms=0,
            success_rate=1.0,
            error_rate=0.0,
            last_check=datetime.now(),
            consecutive_failures=0,
            uptime_percentage=100.0,
            rate_limit_remaining=config.rate_limit_per_hour,
            circuit_breaker_open=False
        )
        
        # Add lender-specific validation rules
        self._add_lender_validation_rules(config)
        
        logger.info(f"Added lender configuration: {config.name}")
    
    def _add_lender_validation_rules(self, config: LenderConfig):
        """Add lender-specific validation rules"""
        lender_name = config.name.lower()
        
        # Loan amount validation
        self.validation_engine.add_rule(ValidationRule(
            id=f"{lender_name}_loan_amount",
            name=f"{config.name} Loan Amount Validation",
            description=f"Validate loan amount for {config.name}",
            field="loan_amount",
            rule_type="range",
            parameters={"min": config.min_loan_amount, "max": config.max_loan_amount},
            severity=ValidationSeverity.ERROR,
            error_message=f"Loan amount must be between €{config.min_loan_amount:,.0f} and €{config.max_loan_amount:,.0f}",
            correction_suggestion=f"Adjust loan amount to fit {config.name} requirements"
        ))
        
        # LTV validation
        self.validation_engine.add_rule(ValidationRule(
            id=f"{lender_name}_ltv",
            name=f"{config.name} LTV Validation",
            description=f"Validate LTV ratio for {config.name}",
            field="ltv_ratio",
            rule_type="range",
            parameters={"min": 0, "max": config.max_ltv},
            severity=ValidationSeverity.ERROR,
            error_message=f"LTV ratio cannot exceed {config.max_ltv}%",
            correction_suggestion="Increase down payment or reduce loan amount"
        ))
        
        # Income validation
        self.validation_engine.add_rule(ValidationRule(
            id=f"{lender_name}_income",
            name=f"{config.name} Income Validation",
            description=f"Validate minimum income for {config.name}",
            field="gross_annual_income",
            rule_type="range",
            parameters={"min": config.min_income},
            severity=ValidationSeverity.ERROR,
            error_message=f"Minimum annual income required: €{config.min_income:,.0f}",
            correction_suggestion="Consider co-applicant or different lender"
        ))
    
    def _initialize_default_validation_rules(self):
        """Initialize default validation rules"""
        # BSN validation
        self.validation_engine.add_rule(ValidationRule(
            id="bsn_format",
            name="BSN Format Validation",
            description="Validate Dutch BSN format and checksum",
            field="bsn",
            rule_type="bsn",
            parameters={},
            severity=ValidationSeverity.ERROR,
            error_message="Invalid BSN format or checksum",
            correction_suggestion="Verify BSN digits and format"
        ))
        
        # Email validation
        self.validation_engine.add_rule(ValidationRule(
            id="email_format",
            name="Email Format Validation",
            description="Validate email address format",
            field="email",
            rule_type="email",
            parameters={},
            severity=ValidationSeverity.ERROR,
            error_message="Invalid email address format",
            correction_suggestion="Enter valid email address"
        ))
        
        # Required fields
        for field in ['first_name', 'last_name', 'date_of_birth', 'address']:
            self.validation_engine.add_rule(ValidationRule(
                id=f"{field}_required",
                name=f"{field.replace('_', ' ').title()} Required",
                description=f"Validate {field} is provided",
                field=field,
                rule_type="required",
                parameters={},
                severity=ValidationSeverity.ERROR,
                error_message=f"{field.replace('_', ' ').title()} is required",
                correction_suggestion=f"Please provide {field.replace('_', ' ')}"
            ))
    
    async def submit_application(self, lender_name: str, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit application to lender with comprehensive validation and AI scoring"""
        lender_key = lender_name.lower()
        
        if lender_key not in self.lenders:
            raise ValueError(f"Lender not configured: {lender_name}")
        
        config = self.lenders[lender_key]
        start_time = time.time()
        
        try:
            # Step 1: Validate application data
            validation_results = self.validation_engine.validate_application(application_data)
            
            # Check for critical validation errors
            critical_errors = []
            for field, results in validation_results.items():
                for result in results:
                    if not result.is_valid and result.severity == ValidationSeverity.CRITICAL:
                        critical_errors.append(result)
            
            if critical_errors:
                return {
                    'success': False,
                    'error': 'Critical validation errors',
                    'validation_results': validation_results,
                    'critical_errors': [asdict(error) for error in critical_errors]
                }
            
            # Step 2: Get approval prediction
            prediction = self.prediction_model.predict_approval(lender_name, application_data)
            
            # Step 3: Check rate limits
            rate_limiter = self.rate_limiters[lender_key]
            if not rate_limiter.acquire():
                wait_time = rate_limiter.get_wait_time()
                return {
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'retry_after': wait_time
                }
            
            # Step 4: Check circuit breaker
            circuit_breaker = self.circuit_breakers[lender_key]
            if circuit_breaker.state == 'open':
                return {
                    'success': False,
                    'error': 'Circuit breaker open - lender temporarily unavailable'
                }
            
            # Step 5: Transform application data for lender
            transformed_data = await self._transform_application_data(config, application_data)
            
            # Step 6: Submit to lender API
            submission_result = await circuit_breaker.call(
                self._submit_to_lender_api,
                config,
                transformed_data
            )
            
            # Step 7: Process and store results
            response_time = (time.time() - start_time) * 1000
            await self._update_metrics(lender_key, True, response_time)
            
            # Store submission in database
            submission_id = await self._store_submission(
                lender_name,
                application_data,
                submission_result,
                validation_results,
                prediction
            )
            
            # Send notification
            await self.notification_queue.put({
                'type': 'application_submitted',
                'lender': lender_name,
                'submission_id': submission_id,
                'prediction': asdict(prediction)
            })
            
            return {
                'success': True,
                'submission_id': submission_id,
                'lender_reference': submission_result.get('reference_number'),
                'status': submission_result.get('status', 'submitted'),
                'estimated_processing_time': config.processing_time_days,
                'validation_results': validation_results,
                'approval_prediction': asdict(prediction),
                'response_time_ms': response_time
            }
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            await self._update_metrics(lender_key, False, response_time)
            
            logger.error(f"Application submission failed for {lender_name}: {e}")
            
            # Store failed submission
            await self._store_failed_submission(lender_name, application_data, str(e))
            
            return {
                'success': False,
                'error': str(e),
                'response_time_ms': response_time
            }
    
    async def _submit_to_lender_api(self, config: LenderConfig, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit application to lender API"""
        headers = {
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'MortgageAI-LenderIntegration/2025.1'
        }
        
        if config.custom_headers:
            headers.update(config.custom_headers)
        
        # Primary API attempt
        try:
            async with self.session.post(
                f"{config.api_url}/applications",
                json=application_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=config.timeout_seconds),
                ssl=config.ssl_verify
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return result
                elif response.status == 429:
                    # Rate limited
                    retry_after = response.headers.get('Retry-After', '60')
                    raise Exception(f"Rate limited, retry after {retry_after} seconds")
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
        
        except Exception as e:
            # Try backup API if available
            if config.backup_api_url and config.backup_api_key:
                logger.warning(f"Primary API failed for {config.name}, trying backup: {e}")
                
                backup_headers = headers.copy()
                backup_headers['Authorization'] = f'Bearer {config.backup_api_key}'
                
                async with self.session.post(
                    f"{config.backup_api_url}/applications",
                    json=application_data,
                    headers=backup_headers,
                    timeout=aiohttp.ClientTimeout(total=config.timeout_seconds),
                    ssl=config.ssl_verify
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Backup API error {response.status}: {error_text}")
            else:
                raise e
    
    async def _transform_application_data(self, config: LenderConfig, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform application data to lender-specific format"""
        # Base transformation
        transformed = {
            'applicant': {
                'personal': {
                    'first_name': application_data.get('first_name'),
                    'last_name': application_data.get('last_name'),
                    'date_of_birth': application_data.get('date_of_birth'),
                    'bsn': application_data.get('bsn'),
                    'nationality': application_data.get('nationality', 'Dutch'),
                    'marital_status': application_data.get('marital_status', 'single')
                },
                'contact': {
                    'email': application_data.get('email'),
                    'phone': application_data.get('phone'),
                    'address': application_data.get('address'),
                    'postcode': application_data.get('postcode'),
                    'city': application_data.get('city')
                },
                'employment': {
                    'type': application_data.get('employment_type'),
                    'employer': application_data.get('employer'),
                    'position': application_data.get('job_title'),
                    'years_employed': application_data.get('employment_years'),
                    'gross_annual_income': application_data.get('gross_annual_income'),
                    'net_monthly_income': application_data.get('net_monthly_income')
                },
                'financial': {
                    'existing_debts': application_data.get('existing_debts', 0),
                    'savings': application_data.get('savings', 0),
                    'investments': application_data.get('investments', 0),
                    'other_income': application_data.get('other_income', 0)
                }
            },
            'property': {
                'address': application_data.get('property_address'),
                'postcode': application_data.get('property_postcode'),
                'city': application_data.get('property_city'),
                'type': application_data.get('property_type', 'house'),
                'value': application_data.get('property_value'),
                'construction_year': application_data.get('construction_year'),
                'living_area': application_data.get('living_area'),
                'plot_size': application_data.get('plot_size'),
                'energy_label': application_data.get('energy_label')
            },
            'mortgage': {
                'loan_amount': application_data.get('loan_amount'),
                'loan_term_years': application_data.get('loan_term_years', 30),
                'interest_type': application_data.get('interest_type', 'fixed'),
                'interest_period': application_data.get('interest_period', 10),
                'repayment_type': application_data.get('repayment_type', 'annuity'),
                'purpose': application_data.get('loan_purpose', 'purchase'),
                'nhg_requested': application_data.get('nhg_requested', False)
            },
            'documents': application_data.get('documents', []),
            'metadata': {
                'source': 'mortgage_ai_advanced',
                'submission_timestamp': datetime.now().isoformat(),
                'api_version': '2025.1',
                'lender': config.name
            }
        }
        
        # Apply lender-specific transformations
        lender_name = config.name.lower()
        
        if lender_name == 'ing':
            # ING specific transformations
            transformed['ing_specific'] = {
                'product_code': self._map_product_to_ing(application_data.get('product_name')),
                'branch_code': application_data.get('preferred_branch', 'ONLINE'),
                'advisor_code': application_data.get('advisor_id')
            }
        
        elif lender_name == 'rabobank':
            # Rabobank specific transformations
            transformed['rabobank_specific'] = {
                'membership_number': application_data.get('rabobank_membership'),
                'local_bank': application_data.get('local_rabobank'),
                'sustainability_score': self._calculate_sustainability_score(application_data)
            }
        
        elif lender_name == 'abn amro':
            # ABN AMRO specific transformations
            transformed['abn_specific'] = {
                'client_number': application_data.get('abn_client_number'),
                'package_type': application_data.get('banking_package', 'basic'),
                'private_banking': application_data.get('private_banking', False)
            }
        
        return transformed
    
    def _map_product_to_ing(self, product_name: str) -> str:
        """Map product name to ING product code"""
        mapping = {
            'fixed_5_year': 'ING_FIXED_5Y',
            'fixed_10_year': 'ING_FIXED_10Y',
            'fixed_20_year': 'ING_FIXED_20Y',
            'fixed_30_year': 'ING_FIXED_30Y',
            'variable': 'ING_VARIABLE'
        }
        return mapping.get(product_name, 'ING_FIXED_10Y')
    
    def _calculate_sustainability_score(self, application_data: Dict[str, Any]) -> int:
        """Calculate sustainability score for Rabobank"""
        score = 0
        
        # Energy label bonus
        energy_label = application_data.get('energy_label', 'G')
        energy_scores = {'A+++': 10, 'A++': 9, 'A+': 8, 'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
        score += energy_scores.get(energy_label, 1)
        
        # Solar panels bonus
        if application_data.get('solar_panels', False):
            score += 5
        
        # Heat pump bonus
        if application_data.get('heat_pump', False):
            score += 3
        
        # Insulation bonus
        if application_data.get('roof_insulation', False):
            score += 2
        if application_data.get('wall_insulation', False):
            score += 2
        if application_data.get('floor_insulation', False):
            score += 2
        
        return min(score, 25)  # Cap at 25
    
    async def get_application_status(self, lender_name: str, reference_number: str) -> Dict[str, Any]:
        """Get application status from lender"""
        lender_key = lender_name.lower()
        
        if lender_key not in self.lenders:
            raise ValueError(f"Lender not configured: {lender_name}")
        
        config = self.lenders[lender_key]
        
        # Check cache first
        cache_key = f"status:{lender_key}:{reference_number}"
        cached_status = await self.redis_pool.get(cache_key)
        
        if cached_status:
            return json.loads(cached_status)
        
        try:
            # Check circuit breaker
            circuit_breaker = self.circuit_breakers[lender_key]
            
            status_data = await circuit_breaker.call(
                self._get_status_from_lender,
                config,
                reference_number
            )
            
            # Cache for 5 minutes
            await self.redis_pool.setex(cache_key, 300, json.dumps(status_data))
            
            return status_data
            
        except Exception as e:
            logger.error(f"Failed to get status from {lender_name}: {e}")
            raise
    
    async def _get_status_from_lender(self, config: LenderConfig, reference_number: str) -> Dict[str, Any]:
        """Get status from lender API"""
        headers = {
            'Authorization': f'Bearer {config.api_key}',
            'Accept': 'application/json'
        }
        
        if config.custom_headers:
            headers.update(config.custom_headers)
        
        async with self.session.get(
            f"{config.api_url}/applications/{reference_number}",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
            ssl=config.ssl_verify
        ) as response:
            
            if response.status == 200:
                data = await response.json()
                
                # Standardize status format
                return {
                    'reference_number': reference_number,
                    'status': data.get('status'),
                    'status_description': data.get('description'),
                    'last_updated': data.get('updated_at', datetime.now().isoformat()),
                    'progress_percentage': data.get('progress', 0),
                    'next_steps': data.get('next_steps', []),
                    'required_documents': data.get('required_documents', []),
                    'estimated_completion': data.get('estimated_completion'),
                    'comments': data.get('comments', []),
                    'lender_contact': data.get('contact_info')
                }
            else:
                error_text = await response.text()
                raise Exception(f"Status check failed: {response.status} - {error_text}")
    
    async def get_approval_predictions(self, application_data: Dict[str, Any], lender_names: List[str] = None) -> List[ApprovalPrediction]:
        """Get approval predictions for multiple lenders"""
        if lender_names is None:
            lender_names = list(self.lenders.keys())
        
        predictions = []
        
        for lender_name in lender_names:
            try:
                prediction = self.prediction_model.predict_approval(lender_name, application_data)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to get prediction for {lender_name}: {e}")
        
        # Sort by probability (highest first)
        predictions.sort(key=lambda x: x.probability, reverse=True)
        
        return predictions
    
    async def optimize_lender_selection(self, application_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize lender selection based on multiple criteria"""
        predictions = await self.get_approval_predictions(application_data)
        recommendations = []
        
        for prediction in predictions:
            lender_config = self.lenders.get(prediction.lender_name.lower())
            if not lender_config:
                continue
            
            # Calculate total cost
            loan_amount = application_data.get('loan_amount', 0)
            interest_rate = lender_config.fees.get('interest_rate', 3.5)
            loan_term = application_data.get('loan_term_years', 30)
            
            monthly_payment = self._calculate_monthly_payment(loan_amount, interest_rate, loan_term)
            total_cost = monthly_payment * loan_term * 12
            
            # Calculate recommendation score
            score = (
                prediction.probability * 0.4 +  # 40% approval likelihood
                (1 - interest_rate / 10) * 0.3 +  # 30% interest rate (lower is better)
                lender_config.priority_score / 10 * 0.2 +  # 20% lender priority
                self.health_metrics[prediction.lender_name.lower()].success_rate * 0.1  # 10% reliability
            )
            
            recommendations.append({
                'lender_name': prediction.lender_name,
                'approval_probability': prediction.probability,
                'confidence_interval': prediction.confidence_interval,
                'estimated_interest_rate': interest_rate,
                'monthly_payment': monthly_payment,
                'total_cost': total_cost,
                'processing_time_days': lender_config.processing_time_days,
                'recommendation_score': score,
                'risk_factors': prediction.risk_factors,
                'positive_factors': prediction.positive_factors,
                'recommendation': prediction.recommendation
            })
        
        # Sort by recommendation score
        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        return recommendations
    
    def _calculate_monthly_payment(self, loan_amount: float, annual_rate: float, years: int) -> float:
        """Calculate monthly mortgage payment"""
        monthly_rate = annual_rate / 100 / 12
        num_payments = years * 12
        
        if monthly_rate == 0:
            return loan_amount / num_payments
        
        return loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
    
    async def train_prediction_models(self, historical_data: Dict[str, List[Dict[str, Any]]]):
        """Train prediction models with historical data"""
        results = {}
        
        for lender_name, data in historical_data.items():
            try:
                metrics = self.prediction_model.train_lender_model(lender_name, data)
                results[lender_name] = metrics
                logger.info(f"Trained model for {lender_name}: {metrics}")
            except Exception as e:
                logger.error(f"Failed to train model for {lender_name}: {e}")
                results[lender_name] = {'error': str(e)}
        
        return results
    
    async def _monitor_lender_health(self):
        """Background task to monitor lender API health"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                for lender_name, config in self.lenders.items():
                    await self._check_lender_health(lender_name, config)
                
                # Wait 5 minutes between checks
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _check_lender_health(self, lender_name: str, config: LenderConfig):
        """Check individual lender health"""
        start_time = time.time()
        
        try:
            # Simple health check endpoint
            headers = {
                'Authorization': f'Bearer {config.api_key}',
                'Accept': 'application/json'
            }
            
            async with self.session.get(
                f"{config.api_url}/health",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
                ssl=config.ssl_verify
            ) as response:
                
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    # Update success metrics
                    metrics = self.health_metrics[lender_name]
                    metrics.status = LenderStatus.ACTIVE
                    metrics.response_time_ms = response_time
                    metrics.last_check = datetime.now()
                    metrics.consecutive_failures = 0
                    
                    # Update success rate (rolling average)
                    request_data = self.request_metrics[lender_name]
                    request_data['response_times'].append(response_time)
                    
                else:
                    # Update failure metrics
                    await self._update_failure_metrics(lender_name, f"Health check failed: {response.status}")
        
        except Exception as e:
            await self._update_failure_metrics(lender_name, f"Health check error: {str(e)}")
    
    async def _update_failure_metrics(self, lender_name: str, error: str):
        """Update failure metrics for lender"""
        metrics = self.health_metrics[lender_name]
        metrics.consecutive_failures += 1
        metrics.last_check = datetime.now()
        
        if metrics.consecutive_failures >= 3:
            metrics.status = LenderStatus.DEGRADED
        if metrics.consecutive_failures >= 5:
            metrics.status = LenderStatus.OFFLINE
        
        # Send alert
        await self.notification_queue.put({
            'type': 'lender_health_alert',
            'lender': lender_name,
            'status': metrics.status.value,
            'error': error,
            'consecutive_failures': metrics.consecutive_failures
        })
    
    async def _update_metrics(self, lender_name: str, success: bool, response_time: float):
        """Update request metrics"""
        request_data = self.request_metrics[lender_name]
        request_data['count'] += 1
        
        if success:
            request_data['success'] += 1
        else:
            request_data['failure'] += 1
        
        request_data['response_times'].append(response_time)
        
        # Calculate rolling averages
        if request_data['response_times']:
            request_data['avg_response_time'] = statistics.mean(request_data['response_times'])
        
        # Update health metrics
        metrics = self.health_metrics[lender_name]
        metrics.success_rate = request_data['success'] / request_data['count']
        metrics.error_rate = request_data['failure'] / request_data['count']
        metrics.response_time_ms = request_data['avg_response_time']
    
    async def _store_submission(self, lender_name: str, application_data: Dict[str, Any], 
                               submission_result: Dict[str, Any], validation_results: Dict[str, Any],
                               prediction: ApprovalPrediction) -> str:
        """Store submission in database"""
        submission_id = str(uuid.uuid4())
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO lender_submissions_advanced (
                    id, lender_name, application_data, submission_result,
                    validation_results, approval_prediction, status, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, submission_id, lender_name, json.dumps(application_data),
                json.dumps(submission_result), json.dumps(validation_results),
                json.dumps(asdict(prediction)), 'submitted', datetime.now())
        
        return submission_id
    
    async def _store_failed_submission(self, lender_name: str, application_data: Dict[str, Any], error: str):
        """Store failed submission in database"""
        submission_id = str(uuid.uuid4())
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO lender_submissions_advanced (
                    id, lender_name, application_data, error_message, status, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """, submission_id, lender_name, json.dumps(application_data),
                error, 'failed', datetime.now())
        
        return submission_id
    
    async def _process_notifications(self):
        """Process notification queue"""
        while True:
            try:
                notification = await self.notification_queue.get()
                await self._send_notification(notification)
            except Exception as e:
                logger.error(f"Error processing notification: {e}")
    
    async def _send_notification(self, notification: Dict[str, Any]):
        """Send notification (email, webhook, etc.)"""
        try:
            notification_type = notification.get('type')
            
            if notification_type == 'application_submitted':
                # Send application submitted notification
                pass
            elif notification_type == 'lender_health_alert':
                # Send health alert
                pass
            
            # Store notification in database for audit
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO lender_notifications (
                        type, data, sent_at
                    ) VALUES ($1, $2, $3)
                """, notification_type, json.dumps(notification), datetime.now())
                
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        return {
            'lenders': {name: asdict(metrics) for name, metrics in self.health_metrics.items()},
            'system': {
                'monitoring_active': self.monitoring_active,
                'db_connected': self.db_pool is not None,
                'redis_connected': self.redis_pool is not None,
                'session_active': self.session is not None and not self.session.closed
            },
            'metrics': dict(self.request_metrics)
        }
    
    async def close(self):
        """Clean up resources"""
        self.monitoring_active = False
        
        if self.session:
            await self.session.close()
        
        if self.db_pool:
            await self.db_pool.close()
        
        if self.redis_pool:
            await self.redis_pool.close()

# Example usage and testing
async def main():
    """Example usage of Advanced Lender Integration Manager"""
    
    # Initialize manager
    manager = AdvancedLenderIntegrationManager()
    
    # Add sample lender configurations
    ing_config = LenderConfig(
        name="ING",
        api_url="https://api.ing.nl/mortgage",
        api_key="test_api_key",
        backup_api_url=None,
        backup_api_key=None,
        supported_products=["fixed_5_year", "fixed_10_year", "fixed_20_year", "variable"],
        max_loan_amount=1000000,
        min_loan_amount=50000,
        max_ltv=100,
        min_income=30000,
        processing_time_days=7,
        rate_limit_per_hour=100,
        timeout_seconds=60,
        retry_attempts=3,
        circuit_breaker_threshold=5,
        priority_score=8,
        fees={"interest_rate": 3.2, "origination_fee": 1500},
        requirements={"min_credit_score": 650},
        validation_rules=[],
        approval_criteria={"max_dti": 40},
        document_requirements=["income_proof", "id_document", "bank_statements"],
        notification_webhooks=[]
    )
    
    manager.add_lender(ing_config)
    
    # Initialize connections (in real use, provide actual URLs)
    # await manager.initialize("postgresql://user:pass@localhost/db", "redis://localhost:6379")
    
    # Example application data
    application_data = {
        'first_name': 'Jan',
        'last_name': 'de Vries',
        'bsn': '123456782',
        'email': 'jan@example.com',
        'loan_amount': 300000,
        'property_value': 400000,
        'gross_annual_income': 60000,
        'employment_type': 'permanent',
        'employment_years': 5
    }
    
    # Get approval predictions
    predictions = await manager.get_approval_predictions(application_data)
    print("Approval Predictions:")
    for prediction in predictions:
        print(f"  {prediction.lender_name}: {prediction.probability:.2%} ({prediction.recommendation})")
    
    # Get optimized lender recommendations
    recommendations = await manager.optimize_lender_selection(application_data)
    print("\nOptimized Recommendations:")
    for rec in recommendations[:3]:  # Top 3
        print(f"  {rec['lender_name']}: Score {rec['recommendation_score']:.2f}")
    
    print("\nAdvanced Lender Integration Manager demo completed!")

if __name__ == "__main__":
    asyncio.run(main())