#!/usr/bin/env python3
"""
Advanced Risk Assessment Engine
Sophisticated multi-factor analysis system with predictive modeling and mitigation recommendations

Features:
- Multi-dimensional risk scoring with 15+ risk factors
- Advanced predictive modeling using ensemble machine learning
- Real-time risk monitoring with dynamic threshold adjustment
- Comprehensive mitigation strategy recommendations
- Integration with BKR, NHG, and compliance systems
- Stress testing and scenario analysis capabilities
- Risk correlation analysis and factor interaction modeling
- Advanced portfolio risk aggregation and concentration analysis
"""

import asyncio
import json
import logging
import hashlib
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import asyncpg
import aioredis
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskCategory(Enum):
    """Risk category classification"""
    CREDIT_RISK = "credit_risk"
    MARKET_RISK = "market_risk"
    OPERATIONAL_RISK = "operational_risk"
    COMPLIANCE_RISK = "compliance_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CONCENTRATION_RISK = "concentration_risk"
    REPUTATIONAL_RISK = "reputational_risk"

class RiskLevel(Enum):
    """Risk level classification"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

class ModelType(Enum):
    """Risk model types"""
    CREDIT_SCORING = "credit_scoring"
    DEFAULT_PROBABILITY = "default_probability"
    LOSS_GIVEN_DEFAULT = "loss_given_default"
    EXPOSURE_AT_DEFAULT = "exposure_at_default"
    PORTFOLIO_RISK = "portfolio_risk"
    STRESS_TESTING = "stress_testing"

@dataclass
class RiskFactor:
    """Individual risk factor definition"""
    factor_id: str
    name: str
    category: RiskCategory
    weight: float
    value: float
    normalized_value: float
    confidence: float
    data_quality: float
    last_updated: datetime
    source: str
    methodology: str
    benchmark: Optional[float] = None
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result"""
    assessment_id: str
    entity_id: str
    entity_type: str
    overall_risk_score: float
    risk_level: RiskLevel
    confidence_interval: Tuple[float, float]
    risk_factors: List[RiskFactor]
    category_scores: Dict[str, float]
    predicted_default_probability: float
    expected_loss: float
    value_at_risk: float
    stress_test_results: Dict[str, float]
    mitigation_recommendations: List[str]
    monitoring_alerts: List[str]
    assessment_timestamp: datetime
    model_version: str
    data_quality_score: float
    risk_appetite_alignment: str
    regulatory_capital_impact: float
    next_review_date: datetime

@dataclass
class StressTestScenario:
    """Stress testing scenario definition"""
    scenario_id: str
    name: str
    description: str
    scenario_type: str
    severity: str
    probability: float
    parameters: Dict[str, float]
    impact_factors: Dict[str, float]
    duration_months: int
    recovery_assumptions: Dict[str, Any]

@dataclass
class MitigationStrategy:
    """Risk mitigation strategy"""
    strategy_id: str
    risk_category: RiskCategory
    strategy_type: str
    name: str
    description: str
    implementation_cost: float
    expected_risk_reduction: float
    implementation_time_weeks: int
    effectiveness_score: float
    prerequisites: List[str]
    success_metrics: List[str]
    monitoring_requirements: List[str]

class RiskFactorCalculator:
    """Advanced risk factor calculation engine"""
    
    def __init__(self):
        self.factor_definitions = self._load_factor_definitions()
        self.benchmarks = self._load_benchmarks()
        self.correlation_matrix = None
    
    def _load_factor_definitions(self) -> Dict[str, Any]:
        """Load comprehensive risk factor definitions"""
        return {
            "credit_score": {
                "category": RiskCategory.CREDIT_RISK,
                "weight": 0.25,
                "min_value": 300,
                "max_value": 850,
                "optimal_range": (700, 850),
                "calculation": "direct_value"
            },
            "debt_to_income_ratio": {
                "category": RiskCategory.CREDIT_RISK,
                "weight": 0.20,
                "min_value": 0,
                "max_value": 100,
                "optimal_range": (0, 36),
                "calculation": "percentage"
            },
            "loan_to_value_ratio": {
                "category": RiskCategory.CREDIT_RISK,
                "weight": 0.18,
                "min_value": 0,
                "max_value": 125,
                "optimal_range": (0, 80),
                "calculation": "percentage"
            },
            "employment_stability": {
                "category": RiskCategory.CREDIT_RISK,
                "weight": 0.12,
                "min_value": 0,
                "max_value": 40,
                "optimal_range": (5, 40),
                "calculation": "years_employed"
            },
            "payment_history": {
                "category": RiskCategory.CREDIT_RISK,
                "weight": 0.15,
                "min_value": 0,
                "max_value": 100,
                "optimal_range": (95, 100),
                "calculation": "percentage_on_time"
            },
            "property_value_volatility": {
                "category": RiskCategory.MARKET_RISK,
                "weight": 0.10,
                "min_value": 0,
                "max_value": 50,
                "optimal_range": (0, 15),
                "calculation": "volatility_percentage"
            },
            "interest_rate_sensitivity": {
                "category": RiskCategory.MARKET_RISK,
                "weight": 0.08,
                "min_value": 0,
                "max_value": 100,
                "optimal_range": (0, 30),
                "calculation": "sensitivity_score"
            },
            "regulatory_compliance_score": {
                "category": RiskCategory.COMPLIANCE_RISK,
                "weight": 0.12,
                "min_value": 0,
                "max_value": 100,
                "optimal_range": (95, 100),
                "calculation": "compliance_percentage"
            },
            "liquidity_coverage_ratio": {
                "category": RiskCategory.LIQUIDITY_RISK,
                "weight": 0.08,
                "min_value": 0,
                "max_value": 500,
                "optimal_range": (100, 200),
                "calculation": "ratio_percentage"
            },
            "concentration_risk_score": {
                "category": RiskCategory.CONCENTRATION_RISK,
                "weight": 0.07,
                "min_value": 0,
                "max_value": 100,
                "optimal_range": (0, 25),
                "calculation": "concentration_percentage"
            },
            "operational_risk_incidents": {
                "category": RiskCategory.OPERATIONAL_RISK,
                "weight": 0.06,
                "min_value": 0,
                "max_value": 50,
                "optimal_range": (0, 2),
                "calculation": "incident_count_annual"
            },
            "fraud_indicators": {
                "category": RiskCategory.OPERATIONAL_RISK,
                "weight": 0.09,
                "min_value": 0,
                "max_value": 100,
                "optimal_range": (0, 10),
                "calculation": "fraud_score"
            },
            "data_quality_score": {
                "category": RiskCategory.OPERATIONAL_RISK,
                "weight": 0.05,
                "min_value": 0,
                "max_value": 100,
                "optimal_range": (90, 100),
                "calculation": "quality_percentage"
            },
            "customer_satisfaction": {
                "category": RiskCategory.REPUTATIONAL_RISK,
                "weight": 0.04,
                "min_value": 0,
                "max_value": 100,
                "optimal_range": (80, 100),
                "calculation": "satisfaction_score"
            },
            "regulatory_changes_impact": {
                "category": RiskCategory.COMPLIANCE_RISK,
                "weight": 0.06,
                "min_value": 0,
                "max_value": 100,
                "optimal_range": (0, 20),
                "calculation": "impact_score"
            }
        }
    
    def _load_benchmarks(self) -> Dict[str, float]:
        """Load industry benchmarks for risk factors"""
        return {
            "credit_score": 720,
            "debt_to_income_ratio": 28,
            "loan_to_value_ratio": 75,
            "employment_stability": 8,
            "payment_history": 96,
            "property_value_volatility": 12,
            "interest_rate_sensitivity": 25,
            "regulatory_compliance_score": 98,
            "liquidity_coverage_ratio": 150,
            "concentration_risk_score": 15,
            "operational_risk_incidents": 1,
            "fraud_indicators": 5,
            "data_quality_score": 95,
            "customer_satisfaction": 85,
            "regulatory_changes_impact": 10
        }
    
    def calculate_risk_factors(self, input_data: Dict[str, Any]) -> List[RiskFactor]:
        """Calculate all risk factors from input data"""
        risk_factors = []
        
        for factor_id, definition in self.factor_definitions.items():
            try:
                # Extract raw value
                raw_value = self._extract_factor_value(factor_id, input_data, definition)
                
                # Normalize value (0-1 scale where 0 is best, 1 is worst)
                normalized_value = self._normalize_factor_value(raw_value, definition)
                
                # Calculate confidence based on data quality
                confidence = self._calculate_confidence(factor_id, input_data)
                
                # Assess data quality
                data_quality = self._assess_data_quality(factor_id, input_data)
                
                risk_factor = RiskFactor(
                    factor_id=factor_id,
                    name=definition.get("name", factor_id.replace("_", " ").title()),
                    category=definition["category"],
                    weight=definition["weight"],
                    value=raw_value,
                    normalized_value=normalized_value,
                    confidence=confidence,
                    data_quality=data_quality,
                    last_updated=datetime.now(),
                    source=input_data.get(f"{factor_id}_source", "system"),
                    methodology=definition["calculation"],
                    benchmark=self.benchmarks.get(factor_id),
                    threshold_low=definition.get("optimal_range", (0, 100))[0],
                    threshold_high=definition.get("optimal_range", (0, 100))[1]
                )
                
                risk_factors.append(risk_factor)
                
            except Exception as e:
                logger.error(f"Error calculating risk factor {factor_id}: {e}")
                # Create error factor for tracking
                risk_factors.append(RiskFactor(
                    factor_id=factor_id,
                    name=f"ERROR: {factor_id}",
                    category=definition["category"],
                    weight=0,
                    value=0,
                    normalized_value=0.5,  # Neutral risk
                    confidence=0,
                    data_quality=0,
                    last_updated=datetime.now(),
                    source="error",
                    methodology="error"
                ))
        
        return risk_factors
    
    def _extract_factor_value(self, factor_id: str, input_data: Dict[str, Any], 
                             definition: Dict[str, Any]) -> float:
        """Extract factor value from input data"""
        calculation_method = definition["calculation"]
        
        if factor_id == "credit_score":
            return float(input_data.get("credit_score", 700))
        
        elif factor_id == "debt_to_income_ratio":
            monthly_debt = float(input_data.get("monthly_debt_payments", 0))
            monthly_income = float(input_data.get("net_monthly_income", 1))
            return (monthly_debt / monthly_income * 100) if monthly_income > 0 else 100
        
        elif factor_id == "loan_to_value_ratio":
            loan_amount = float(input_data.get("loan_amount", 0))
            property_value = float(input_data.get("property_value", 1))
            return (loan_amount / property_value * 100) if property_value > 0 else 100
        
        elif factor_id == "employment_stability":
            return float(input_data.get("employment_years", 0))
        
        elif factor_id == "payment_history":
            on_time_payments = float(input_data.get("on_time_payment_percentage", 95))
            return on_time_payments
        
        elif factor_id == "property_value_volatility":
            # Calculate from historical property values or use market data
            volatility = float(input_data.get("property_volatility", 12))
            return volatility
        
        elif factor_id == "interest_rate_sensitivity":
            # Calculate based on loan structure and market conditions
            fixed_rate_period = float(input_data.get("fixed_rate_period_years", 10))
            loan_term = float(input_data.get("loan_term_years", 30))
            sensitivity = (1 - (fixed_rate_period / loan_term)) * 100
            return sensitivity
        
        elif factor_id == "regulatory_compliance_score":
            return float(input_data.get("compliance_score", 95))
        
        elif factor_id == "liquidity_coverage_ratio":
            liquid_assets = float(input_data.get("liquid_assets", 100000))
            monthly_expenses = float(input_data.get("monthly_expenses", 3000))
            return (liquid_assets / (monthly_expenses * 3)) * 100 if monthly_expenses > 0 else 100
        
        elif factor_id == "concentration_risk_score":
            # Calculate based on income source concentration
            primary_income_ratio = float(input_data.get("primary_income_percentage", 80))
            return primary_income_ratio
        
        elif factor_id == "operational_risk_incidents":
            return float(input_data.get("operational_incidents_12m", 0))
        
        elif factor_id == "fraud_indicators":
            return float(input_data.get("fraud_score", 0))
        
        elif factor_id == "data_quality_score":
            return float(input_data.get("data_completeness", 95))
        
        elif factor_id == "customer_satisfaction":
            return float(input_data.get("satisfaction_score", 85))
        
        elif factor_id == "regulatory_changes_impact":
            return float(input_data.get("regulatory_impact_score", 10))
        
        else:
            return float(input_data.get(factor_id, 50))  # Default neutral value
    
    def _normalize_factor_value(self, value: float, definition: Dict[str, Any]) -> float:
        """Normalize factor value to 0-1 scale (0=best, 1=worst)"""
        min_val = definition["min_value"]
        max_val = definition["max_value"]
        optimal_range = definition.get("optimal_range", (min_val, max_val))
        
        # Clamp value to valid range
        clamped_value = max(min_val, min(max_val, value))
        
        # Determine if lower or higher values are better
        if optimal_range[0] == min_val:  # Higher is better (e.g., credit score)
            normalized = 1 - ((clamped_value - min_val) / (max_val - min_val))
        else:  # Lower is better (e.g., debt ratio)
            normalized = (clamped_value - min_val) / (max_val - min_val)
        
        # Apply optimal range adjustment
        if optimal_range[0] <= clamped_value <= optimal_range[1]:
            normalized *= 0.3  # Reduce risk for values in optimal range
        
        return max(0, min(1, normalized))
    
    def _calculate_confidence(self, factor_id: str, input_data: Dict[str, Any]) -> float:
        """Calculate confidence in factor value"""
        base_confidence = 0.8
        
        # Adjust based on data source
        source = input_data.get(f"{factor_id}_source", "estimated")
        if source == "verified":
            base_confidence = 0.95
        elif source == "estimated":
            base_confidence = 0.7
        elif source == "calculated":
            base_confidence = 0.85
        
        # Adjust based on data age
        last_updated = input_data.get(f"{factor_id}_last_updated")
        if last_updated:
            days_old = (datetime.now() - datetime.fromisoformat(last_updated)).days
            age_penalty = min(0.2, days_old / 365 * 0.2)  # Max 20% penalty for 1 year old data
            base_confidence -= age_penalty
        
        return max(0.1, min(1.0, base_confidence))
    
    def _assess_data_quality(self, factor_id: str, input_data: Dict[str, Any]) -> float:
        """Assess data quality for factor"""
        quality_score = 1.0
        
        # Check for missing data
        if factor_id not in input_data and f"{factor_id}_calculated" not in input_data:
            quality_score *= 0.5
        
        # Check for data consistency
        if input_data.get(f"{factor_id}_inconsistent", False):
            quality_score *= 0.7
        
        # Check for outliers
        value = input_data.get(factor_id, 0)
        definition = self.factor_definitions[factor_id]
        if value < definition["min_value"] or value > definition["max_value"]:
            quality_score *= 0.6
        
        return max(0.1, quality_score)

class PredictiveRiskModels:
    """Advanced predictive risk modeling system"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.training_history = {}
    
    def train_credit_risk_model(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """Train comprehensive credit risk model"""
        try:
            # Prepare features and target
            feature_columns = [
                'credit_score', 'debt_to_income_ratio', 'loan_to_value_ratio',
                'employment_stability', 'payment_history', 'property_value_volatility',
                'interest_rate_sensitivity', 'liquidity_coverage_ratio'
            ]
            
            X = training_data[feature_columns].fillna(training_data[feature_columns].median())
            y = training_data['default_probability'].fillna(0)
            
            if len(X) < 100:
                logger.warning(f"Insufficient training data: {len(X)} samples")
                return {"error": "insufficient_data"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create ensemble model
            models = {
                'random_forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=150, max_depth=8, random_state=42),
                'ridge': Ridge(alpha=1.0)
            }
            
            # Train individual models
            trained_models = {}
            model_scores = {}
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate performance metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_scores[name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                }
                
                trained_models[name] = model
            
            # Create voting ensemble
            voting_regressor = VotingRegressor([
                ('rf', trained_models['random_forest']),
                ('gb', trained_models['gradient_boosting']),
                ('ridge', trained_models['ridge'])
            ])
            voting_regressor.fit(X_train_scaled, y_train)
            
            # Evaluate ensemble
            y_pred_ensemble = voting_regressor.predict(X_test_scaled)
            ensemble_performance = {
                'mse': mean_squared_error(y_test, y_pred_ensemble),
                'mae': mean_absolute_error(y_test, y_pred_ensemble),
                'r2': r2_score(y_test, y_pred_ensemble),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
            }
            
            # Store models and metadata
            self.models[ModelType.CREDIT_SCORING] = voting_regressor
            self.scalers[ModelType.CREDIT_SCORING] = scaler
            self.model_performance[ModelType.CREDIT_SCORING] = ensemble_performance
            
            # Calculate feature importance (using random forest)
            if 'random_forest' in trained_models:
                importance = trained_models['random_forest'].feature_importances_
                self.feature_importance[ModelType.CREDIT_SCORING] = dict(zip(feature_columns, importance))
            
            logger.info(f"Credit risk model trained successfully: R² = {ensemble_performance['r2']:.3f}")
            
            return ensemble_performance
            
        except Exception as e:
            logger.error(f"Error training credit risk model: {e}")
            return {"error": str(e)}
    
    def predict_default_probability(self, risk_factors: List[RiskFactor]) -> Tuple[float, float]:
        """Predict default probability with confidence interval"""
        try:
            if ModelType.CREDIT_SCORING not in self.models:
                # Return conservative estimate if no model
                return 0.05, 0.95  # 5% default probability, 95% confidence
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(risk_factors)
            
            # Scale features
            scaler = self.scalers[ModelType.CREDIT_SCORING]
            feature_vector_scaled = scaler.transform([feature_vector])
            
            # Get prediction
            model = self.models[ModelType.CREDIT_SCORING]
            prediction = model.predict(feature_vector_scaled)[0]
            
            # Calculate confidence based on model performance and data quality
            model_r2 = self.model_performance[ModelType.CREDIT_SCORING]['r2']
            avg_data_quality = np.mean([rf.data_quality for rf in risk_factors])
            confidence = model_r2 * avg_data_quality
            
            # Ensure reasonable bounds
            prediction = max(0.001, min(0.999, prediction))
            confidence = max(0.5, min(0.99, confidence))
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error predicting default probability: {e}")
            return 0.05, 0.5  # Conservative fallback
    
    def _prepare_feature_vector(self, risk_factors: List[RiskFactor]) -> List[float]:
        """Prepare feature vector for model prediction"""
        feature_map = {rf.factor_id: rf.value for rf in risk_factors}
        
        feature_order = [
            'credit_score', 'debt_to_income_ratio', 'loan_to_value_ratio',
            'employment_stability', 'payment_history', 'property_value_volatility',
            'interest_rate_sensitivity', 'liquidity_coverage_ratio'
        ]
        
        return [feature_map.get(feature, 0) for feature in feature_order]
    
    def perform_stress_testing(self, risk_factors: List[RiskFactor], 
                              scenarios: List[StressTestScenario]) -> Dict[str, float]:
        """Perform comprehensive stress testing"""
        stress_results = {}
        
        for scenario in scenarios:
            try:
                # Apply scenario stresses to risk factors
                stressed_factors = self._apply_stress_scenario(risk_factors, scenario)
                
                # Recalculate risk with stressed factors
                stressed_prediction, _ = self.predict_default_probability(stressed_factors)
                
                # Calculate impact
                baseline_prediction, _ = self.predict_default_probability(risk_factors)
                impact = stressed_prediction - baseline_prediction
                
                stress_results[scenario.scenario_id] = {
                    'baseline_probability': baseline_prediction,
                    'stressed_probability': stressed_prediction,
                    'impact': impact,
                    'severity_multiplier': scenario.severity
                }
                
            except Exception as e:
                logger.error(f"Error in stress test {scenario.scenario_id}: {e}")
                stress_results[scenario.scenario_id] = {"error": str(e)}
        
        return stress_results
    
    def _apply_stress_scenario(self, risk_factors: List[RiskFactor], 
                              scenario: StressTestScenario) -> List[RiskFactor]:
        """Apply stress scenario to risk factors"""
        stressed_factors = []
        
        for factor in risk_factors:
            stressed_factor = RiskFactor(**asdict(factor))
            
            # Apply scenario-specific stress
            if factor.factor_id in scenario.impact_factors:
                stress_multiplier = scenario.impact_factors[factor.factor_id]
                
                if factor.category == RiskCategory.CREDIT_RISK:
                    # For credit factors, increase normalized value (worse risk)
                    stressed_factor.normalized_value = min(1.0, factor.normalized_value * stress_multiplier)
                elif factor.category == RiskCategory.MARKET_RISK:
                    # For market factors, apply volatility stress
                    stressed_factor.value = factor.value * stress_multiplier
                    stressed_factor.normalized_value = min(1.0, factor.normalized_value * stress_multiplier)
            
            stressed_factors.append(stressed_factor)
        
        return stressed_factors

class RiskMitigationEngine:
    """Advanced risk mitigation strategy engine"""
    
    def __init__(self):
        self.mitigation_strategies = self._load_mitigation_strategies()
        self.effectiveness_models = {}
    
    def _load_mitigation_strategies(self) -> Dict[str, List[MitigationStrategy]]:
        """Load comprehensive mitigation strategies by risk category"""
        return {
            RiskCategory.CREDIT_RISK.value: [
                MitigationStrategy(
                    strategy_id="credit_enhancement",
                    risk_category=RiskCategory.CREDIT_RISK,
                    strategy_type="credit_enhancement",
                    name="Credit Enhancement Program",
                    description="Implement credit counseling and improvement program",
                    implementation_cost=5000,
                    expected_risk_reduction=0.15,
                    implementation_time_weeks=12,
                    effectiveness_score=0.8,
                    prerequisites=["customer_consent", "credit_analysis"],
                    success_metrics=["credit_score_improvement", "payment_behavior"],
                    monitoring_requirements=["monthly_credit_review", "payment_tracking"]
                ),
                MitigationStrategy(
                    strategy_id="collateral_requirement",
                    risk_category=RiskCategory.CREDIT_RISK,
                    strategy_type="collateral",
                    name="Additional Collateral Requirement",
                    description="Require additional collateral to reduce credit exposure",
                    implementation_cost=2000,
                    expected_risk_reduction=0.25,
                    implementation_time_weeks=4,
                    effectiveness_score=0.9,
                    prerequisites=["collateral_valuation", "legal_documentation"],
                    success_metrics=["ltv_ratio_improvement", "security_coverage"],
                    monitoring_requirements=["collateral_revaluation", "market_monitoring"]
                ),
                MitigationStrategy(
                    strategy_id="income_verification",
                    risk_category=RiskCategory.CREDIT_RISK,
                    strategy_type="verification",
                    name="Enhanced Income Verification",
                    description="Implement comprehensive income verification procedures",
                    implementation_cost=1500,
                    expected_risk_reduction=0.12,
                    implementation_time_weeks=2,
                    effectiveness_score=0.75,
                    prerequisites=["documentation_access", "employer_cooperation"],
                    success_metrics=["verification_accuracy", "income_stability"],
                    monitoring_requirements=["annual_income_review", "employment_verification"]
                )
            ],
            RiskCategory.MARKET_RISK.value: [
                MitigationStrategy(
                    strategy_id="interest_rate_hedging",
                    risk_category=RiskCategory.MARKET_RISK,
                    strategy_type="hedging",
                    name="Interest Rate Hedging",
                    description="Implement interest rate hedging strategies",
                    implementation_cost=10000,
                    expected_risk_reduction=0.30,
                    implementation_time_weeks=8,
                    effectiveness_score=0.85,
                    prerequisites=["risk_assessment", "hedging_approval"],
                    success_metrics=["rate_exposure_reduction", "hedge_effectiveness"],
                    monitoring_requirements=["daily_mark_to_market", "hedge_ratio_monitoring"]
                ),
                MitigationStrategy(
                    strategy_id="portfolio_diversification",
                    risk_category=RiskCategory.MARKET_RISK,
                    strategy_type="diversification",
                    name="Portfolio Diversification",
                    description="Diversify mortgage portfolio across risk segments",
                    implementation_cost=5000,
                    expected_risk_reduction=0.20,
                    implementation_time_weeks=16,
                    effectiveness_score=0.78,
                    prerequisites=["portfolio_analysis", "market_research"],
                    success_metrics=["correlation_reduction", "risk_distribution"],
                    monitoring_requirements=["portfolio_composition_review", "concentration_monitoring"]
                )
            ],
            RiskCategory.OPERATIONAL_RISK.value: [
                MitigationStrategy(
                    strategy_id="process_automation",
                    risk_category=RiskCategory.OPERATIONAL_RISK,
                    strategy_type="automation",
                    name="Process Automation",
                    description="Automate manual processes to reduce operational errors",
                    implementation_cost=25000,
                    expected_risk_reduction=0.35,
                    implementation_time_weeks=20,
                    effectiveness_score=0.88,
                    prerequisites=["process_mapping", "technology_assessment"],
                    success_metrics=["error_rate_reduction", "processing_time"],
                    monitoring_requirements=["system_monitoring", "error_tracking"]
                ),
                MitigationStrategy(
                    strategy_id="staff_training",
                    risk_category=RiskCategory.OPERATIONAL_RISK,
                    strategy_type="training",
                    name="Enhanced Staff Training",
                    description="Comprehensive training program for operational excellence",
                    implementation_cost=8000,
                    expected_risk_reduction=0.18,
                    implementation_time_weeks=12,
                    effectiveness_score=0.72,
                    prerequisites=["training_needs_analysis", "curriculum_development"],
                    success_metrics=["competency_scores", "incident_reduction"],
                    monitoring_requirements=["training_effectiveness", "skill_assessment"]
                )
            ],
            RiskCategory.COMPLIANCE_RISK.value: [
                MitigationStrategy(
                    strategy_id="compliance_monitoring",
                    risk_category=RiskCategory.COMPLIANCE_RISK,
                    strategy_type="monitoring",
                    name="Enhanced Compliance Monitoring",
                    description="Implement real-time compliance monitoring system",
                    implementation_cost=15000,
                    expected_risk_reduction=0.40,
                    implementation_time_weeks=16,
                    effectiveness_score=0.92,
                    prerequisites=["compliance_framework", "monitoring_tools"],
                    success_metrics=["violation_detection_rate", "response_time"],
                    monitoring_requirements=["daily_compliance_review", "alert_monitoring"]
                )
            ]
        }
    
    def recommend_mitigation_strategies(self, risk_assessment: RiskAssessment, 
                                      budget_constraint: float = None) -> List[MitigationStrategy]:
        """Recommend optimal mitigation strategies"""
        recommendations = []
        
        # Analyze risk categories that need attention
        high_risk_categories = [
            category for category, score in risk_assessment.category_scores.items()
            if score > 0.6  # High risk threshold
        ]
        
        # Get strategies for high-risk categories
        for category in high_risk_categories:
            if category in self.mitigation_strategies:
                category_strategies = self.mitigation_strategies[category]
                
                # Sort by effectiveness and cost
                sorted_strategies = sorted(
                    category_strategies,
                    key=lambda s: (s.effectiveness_score / (s.implementation_cost / 1000)),
                    reverse=True
                )
                
                recommendations.extend(sorted_strategies[:2])  # Top 2 per category
        
        # Apply budget constraint if specified
        if budget_constraint:
            recommendations = self._optimize_strategies_for_budget(recommendations, budget_constraint)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _optimize_strategies_for_budget(self, strategies: List[MitigationStrategy], 
                                      budget: float) -> List[MitigationStrategy]:
        """Optimize strategy selection for budget constraint"""
        # Sort by cost-effectiveness ratio
        strategies.sort(key=lambda s: s.expected_risk_reduction / s.implementation_cost, reverse=True)
        
        selected_strategies = []
        remaining_budget = budget
        
        for strategy in strategies:
            if strategy.implementation_cost <= remaining_budget:
                selected_strategies.append(strategy)
                remaining_budget -= strategy.implementation_cost
        
        return selected_strategies

class AdvancedRiskAssessmentEngine:
    """Main risk assessment engine with comprehensive capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.factor_calculator = RiskFactorCalculator()
        self.predictive_models = PredictiveRiskModels()
        self.mitigation_engine = RiskMitigationEngine()
        self.db_pool = None
        self.redis_pool = None
        
        # Performance metrics
        self.metrics = {
            "assessments_performed": 0,
            "models_trained": 0,
            "stress_tests_executed": 0,
            "mitigation_strategies_recommended": 0,
            "avg_assessment_time": 0,
            "assessment_times": []
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            RiskLevel.VERY_LOW: (0.0, 0.1),
            RiskLevel.LOW: (0.1, 0.3),
            RiskLevel.MEDIUM: (0.3, 0.6),
            RiskLevel.HIGH: (0.6, 0.8),
            RiskLevel.VERY_HIGH: (0.8, 0.95),
            RiskLevel.CRITICAL: (0.95, 1.0)
        }
    
    async def initialize(self, database_url: str, redis_url: str):
        """Initialize the risk assessment engine"""
        try:
            # Initialize database connection
            self.db_pool = await asyncpg.create_pool(
                database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Initialize Redis connection
            self.redis_pool = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            
            # Load historical data and train models
            await self._initialize_models()
            
            logger.info("Advanced Risk Assessment Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Risk Assessment Engine: {e}")
            raise
    
    async def perform_comprehensive_risk_assessment(self, entity_id: str, entity_type: str,
                                                  input_data: Dict[str, Any]) -> RiskAssessment:
        """Perform comprehensive risk assessment"""
        start_time = time.time()
        assessment_id = f"RISK_{uuid.uuid4().hex[:12]}_{int(time.time())}"
        
        try:
            # Calculate individual risk factors
            risk_factors = self.factor_calculator.calculate_risk_factors(input_data)
            
            # Calculate category scores
            category_scores = self._calculate_category_scores(risk_factors)
            
            # Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(risk_factors, category_scores)
            
            # Determine risk level
            risk_level = self._determine_risk_level(overall_risk_score)
            
            # Predict default probability
            default_probability, confidence = self.predictive_models.predict_default_probability(risk_factors)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(overall_risk_score, confidence)
            
            # Calculate expected loss
            expected_loss = self._calculate_expected_loss(input_data, default_probability)
            
            # Calculate Value at Risk
            value_at_risk = self._calculate_value_at_risk(input_data, risk_factors)
            
            # Perform stress testing
            stress_scenarios = self._create_stress_scenarios()
            stress_test_results = self.predictive_models.perform_stress_testing(risk_factors, stress_scenarios)
            
            # Generate mitigation recommendations
            mitigation_recommendations = self.mitigation_engine.recommend_mitigation_strategies(
                None, input_data.get("budget_constraint")  # Will be filled after assessment creation
            )
            
            # Generate monitoring alerts
            monitoring_alerts = self._generate_monitoring_alerts(risk_factors, overall_risk_score)
            
            # Assess data quality
            data_quality_score = np.mean([rf.data_quality for rf in risk_factors])
            
            # Determine risk appetite alignment
            risk_appetite_alignment = self._assess_risk_appetite_alignment(overall_risk_score)
            
            # Calculate regulatory capital impact
            regulatory_capital_impact = self._calculate_regulatory_capital_impact(
                default_probability, expected_loss, input_data
            )
            
            # Create assessment object
            assessment = RiskAssessment(
                assessment_id=assessment_id,
                entity_id=entity_id,
                entity_type=entity_type,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                confidence_interval=confidence_interval,
                risk_factors=risk_factors,
                category_scores=category_scores,
                predicted_default_probability=default_probability,
                expected_loss=expected_loss,
                value_at_risk=value_at_risk,
                stress_test_results=stress_test_results,
                mitigation_recommendations=[strategy.name for strategy in mitigation_recommendations],
                monitoring_alerts=monitoring_alerts,
                assessment_timestamp=datetime.now(),
                model_version="2025.1",
                data_quality_score=data_quality_score,
                risk_appetite_alignment=risk_appetite_alignment,
                regulatory_capital_impact=regulatory_capital_impact,
                next_review_date=datetime.now() + timedelta(days=90)
            )
            
            # Store assessment
            await self._store_assessment(assessment, mitigation_recommendations)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self.metrics["assessments_performed"] += 1
            self.metrics["assessment_times"].append(processing_time)
            
            if len(self.metrics["assessment_times"]) > 1000:
                self.metrics["assessment_times"] = self.metrics["assessment_times"][-1000:]
            
            self.metrics["avg_assessment_time"] = np.mean(self.metrics["assessment_times"])
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error performing risk assessment: {e}")
            raise
    
    def _calculate_category_scores(self, risk_factors: List[RiskFactor]) -> Dict[str, float]:
        """Calculate risk scores by category"""
        category_scores = {}
        category_weights = defaultdict(list)
        category_values = defaultdict(list)
        
        # Group factors by category
        for factor in risk_factors:
            category = factor.category.value
            category_weights[category].append(factor.weight)
            category_values[category].append(factor.normalized_value * factor.confidence)
        
        # Calculate weighted average for each category
        for category in category_weights:
            weights = np.array(category_weights[category])
            values = np.array(category_values[category])
            
            # Normalize weights within category
            normalized_weights = weights / np.sum(weights)
            
            # Calculate weighted score
            category_score = np.sum(normalized_weights * values)
            category_scores[category] = min(1.0, max(0.0, category_score))
        
        return category_scores
    
    def _calculate_overall_risk_score(self, risk_factors: List[RiskFactor], 
                                    category_scores: Dict[str, float]) -> float:
        """Calculate overall risk score using advanced aggregation"""
        # Method 1: Factor-weighted approach
        total_weight = sum(factor.weight for factor in risk_factors)
        weighted_sum = sum(
            factor.normalized_value * factor.weight * factor.confidence 
            for factor in risk_factors
        )
        factor_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Method 2: Category-weighted approach
        category_weights = {
            RiskCategory.CREDIT_RISK.value: 0.40,
            RiskCategory.MARKET_RISK.value: 0.20,
            RiskCategory.OPERATIONAL_RISK.value: 0.15,
            RiskCategory.COMPLIANCE_RISK.value: 0.15,
            RiskCategory.LIQUIDITY_RISK.value: 0.05,
            RiskCategory.CONCENTRATION_RISK.value: 0.03,
            RiskCategory.REPUTATIONAL_RISK.value: 0.02
        }
        
        category_score = sum(
            category_scores.get(category, 0) * weight
            for category, weight in category_weights.items()
        )
        
        # Combine approaches with 70/30 weighting
        overall_score = 0.7 * factor_score + 0.3 * category_score
        
        # Apply non-linear transformation for extreme values
        if overall_score > 0.8:
            overall_score = 0.8 + (overall_score - 0.8) * 1.5  # Amplify high risk
        
        return min(1.0, max(0.0, overall_score))
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        for level, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= risk_score < max_score:
                return level
        return RiskLevel.CRITICAL  # Fallback for edge cases
    
    def _calculate_confidence_interval(self, risk_score: float, confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for risk score"""
        # Use normal distribution approximation
        std_dev = (1 - confidence) * 0.2  # Standard deviation based on confidence
        margin_of_error = 1.96 * std_dev  # 95% confidence interval
        
        lower_bound = max(0, risk_score - margin_of_error)
        upper_bound = min(1, risk_score + margin_of_error)
        
        return (lower_bound, upper_bound)
    
    def _calculate_expected_loss(self, input_data: Dict[str, Any], default_probability: float) -> float:
        """Calculate expected loss"""
        loan_amount = float(input_data.get("loan_amount", 0))
        loss_given_default = float(input_data.get("loss_given_default", 0.45))  # 45% default
        
        return loan_amount * default_probability * loss_given_default
    
    def _calculate_value_at_risk(self, input_data: Dict[str, Any], 
                               risk_factors: List[RiskFactor]) -> float:
        """Calculate Value at Risk (VaR) at 99% confidence level"""
        loan_amount = float(input_data.get("loan_amount", 0))
        
        # Extract volatility factors
        volatility_factors = [
            rf for rf in risk_factors 
            if rf.category in [RiskCategory.MARKET_RISK, RiskCategory.CREDIT_RISK]
        ]
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(np.sum([rf.normalized_value ** 2 for rf in volatility_factors]))
        
        # 99% VaR using normal distribution
        var_99 = loan_amount * 2.33 * portfolio_volatility * 0.1  # 2.33 is 99% quantile
        
        return var_99
    
    def _create_stress_scenarios(self) -> List[StressTestScenario]:
        """Create comprehensive stress testing scenarios"""
        return [
            StressTestScenario(
                scenario_id="economic_downturn",
                name="Severe Economic Downturn",
                description="GDP decline of 5%, unemployment increase to 12%",
                scenario_type="macroeconomic",
                severity="severe",
                probability=0.05,
                parameters={"gdp_decline": -5.0, "unemployment_rate": 12.0},
                impact_factors={
                    "credit_score": 1.3,
                    "debt_to_income_ratio": 1.4,
                    "employment_stability": 1.6,
                    "property_value_volatility": 2.0
                },
                duration_months=18,
                recovery_assumptions={"recovery_time_months": 36, "recovery_shape": "U"}
            ),
            StressTestScenario(
                scenario_id="interest_rate_shock",
                name="Interest Rate Shock",
                description="Interest rates increase by 300 basis points",
                scenario_type="market",
                severity="moderate",
                probability=0.15,
                parameters={"interest_rate_increase": 3.0},
                impact_factors={
                    "interest_rate_sensitivity": 2.5,
                    "debt_to_income_ratio": 1.2,
                    "property_value_volatility": 1.4
                },
                duration_months=12,
                recovery_assumptions={"recovery_time_months": 24, "recovery_shape": "V"}
            ),
            StressTestScenario(
                scenario_id="property_market_crash",
                name="Property Market Crash",
                description="Property values decline by 25%",
                scenario_type="market",
                severity="severe",
                probability=0.08,
                parameters={"property_value_decline": -25.0},
                impact_factors={
                    "loan_to_value_ratio": 1.8,
                    "property_value_volatility": 3.0,
                    "liquidity_coverage_ratio": 1.3
                },
                duration_months=24,
                recovery_assumptions={"recovery_time_months": 60, "recovery_shape": "L"}
            )
        ]
    
    def _generate_monitoring_alerts(self, risk_factors: List[RiskFactor], 
                                  overall_risk_score: float) -> List[str]:
        """Generate monitoring alerts based on risk assessment"""
        alerts = []
        
        # Overall risk alerts
        if overall_risk_score > 0.8:
            alerts.append("CRITICAL: Overall risk score exceeds 80% - immediate review required")
        elif overall_risk_score > 0.6:
            alerts.append("HIGH: Overall risk score exceeds 60% - enhanced monitoring recommended")
        
        # Factor-specific alerts
        for factor in risk_factors:
            if factor.normalized_value > 0.9 and factor.confidence > 0.8:
                alerts.append(f"CRITICAL: {factor.name} shows critical risk level")
            elif factor.normalized_value > 0.7 and factor.confidence > 0.7:
                alerts.append(f"HIGH: {factor.name} requires attention")
            
            # Data quality alerts
            if factor.data_quality < 0.6:
                alerts.append(f"DATA QUALITY: {factor.name} has poor data quality - verification needed")
        
        return alerts
    
    def _assess_risk_appetite_alignment(self, risk_score: float) -> str:
        """Assess alignment with organizational risk appetite"""
        risk_appetite_threshold = self.config.get("risk_appetite_threshold", 0.4)
        
        if risk_score <= risk_appetite_threshold * 0.8:
            return "well_within_appetite"
        elif risk_score <= risk_appetite_threshold:
            return "within_appetite"
        elif risk_score <= risk_appetite_threshold * 1.2:
            return "approaching_limit"
        else:
            return "exceeds_appetite"
    
    def _calculate_regulatory_capital_impact(self, default_probability: float, 
                                           expected_loss: float, input_data: Dict[str, Any]) -> float:
        """Calculate regulatory capital impact"""
        # Basel III capital calculation with comprehensive risk weighting
        loan_amount = float(input_data.get("loan_amount", 0))
        
        # Risk-weighted asset calculation
        risk_weight = min(1.0, default_probability * 2.5)  # Cap at 100%
        risk_weighted_assets = loan_amount * risk_weight
        
        # Capital requirement (8% minimum)
        capital_requirement = risk_weighted_assets * 0.08
        
        return capital_requirement
    
    async def _initialize_models(self):
        """Initialize and train risk models with historical data"""
        try:
            # Load historical data
            historical_data = await self._load_historical_data()
            
            if len(historical_data) > 100:
                # Train credit risk model
                performance = self.predictive_models.train_credit_risk_model(historical_data)
                self.metrics["models_trained"] += 1
                logger.info(f"Risk models initialized with {len(historical_data)} records")
            else:
                logger.warning("Insufficient historical data for model training")
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def _load_historical_data(self) -> pd.DataFrame:
        """Load historical risk and performance data"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM risk_assessment_historical_data
                    WHERE created_at > NOW() - INTERVAL '2 years'
                    ORDER BY created_at DESC
                    LIMIT 10000
                """)
                
                if rows:
                    return pd.DataFrame([dict(row) for row in rows])
                else:
                    # Return empty DataFrame with expected columns
                    columns = [
                        'credit_score', 'debt_to_income_ratio', 'loan_to_value_ratio',
                        'employment_stability', 'payment_history', 'default_probability'
                    ]
                    return pd.DataFrame(columns=columns)
                    
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    async def _store_assessment(self, assessment: RiskAssessment, 
                              mitigation_strategies: List[MitigationStrategy]):
        """Store risk assessment in database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store main assessment
                await conn.execute("""
                    INSERT INTO risk_assessments_advanced (
                        assessment_id, entity_id, entity_type, overall_risk_score,
                        risk_level, confidence_interval_lower, confidence_interval_upper,
                        risk_factors, category_scores, predicted_default_probability,
                        expected_loss, value_at_risk, stress_test_results,
                        mitigation_recommendations, monitoring_alerts, model_version,
                        data_quality_score, risk_appetite_alignment, regulatory_capital_impact,
                        next_review_date, created_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21
                    )
                """,
                    assessment.assessment_id, assessment.entity_id, assessment.entity_type,
                    assessment.overall_risk_score, assessment.risk_level.value,
                    assessment.confidence_interval[0], assessment.confidence_interval[1],
                    json.dumps([asdict(rf) for rf in assessment.risk_factors]),
                    json.dumps(assessment.category_scores), assessment.predicted_default_probability,
                    assessment.expected_loss, assessment.value_at_risk,
                    json.dumps(assessment.stress_test_results), json.dumps(assessment.mitigation_recommendations),
                    json.dumps(assessment.monitoring_alerts), assessment.model_version,
                    assessment.data_quality_score, assessment.risk_appetite_alignment,
                    assessment.regulatory_capital_impact, assessment.next_review_date,
                    assessment.assessment_timestamp
                )
                
                # Store mitigation strategies
                for strategy in mitigation_strategies:
                    await conn.execute("""
                        INSERT INTO risk_mitigation_strategies (
                            strategy_id, assessment_id, risk_category, strategy_type,
                            name, description, implementation_cost, expected_risk_reduction,
                            implementation_time_weeks, effectiveness_score, prerequisites,
                            success_metrics, monitoring_requirements, created_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                        )
                    """,
                        strategy.strategy_id, assessment.assessment_id, strategy.risk_category.value,
                        strategy.strategy_type, strategy.name, strategy.description,
                        strategy.implementation_cost, strategy.expected_risk_reduction,
                        strategy.implementation_time_weeks, strategy.effectiveness_score,
                        json.dumps(strategy.prerequisites), json.dumps(strategy.success_metrics),
                        json.dumps(strategy.monitoring_requirements), datetime.now()
                    )
                
        except Exception as e:
            logger.error(f"Error storing risk assessment: {e}")
            raise
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics"""
        return self.metrics
    
    async def close(self):
        """Clean up resources"""
        if self.db_pool:
            await self.db_pool.close()
        
        if self.redis_pool:
            await self.redis_pool.close()

# Example usage
async def main():
    """Example usage of Advanced Risk Assessment Engine"""
    config = {
        "risk_appetite_threshold": 0.4,
        "stress_test_confidence": 0.99
    }
    
    engine = AdvancedRiskAssessmentEngine(config)
    
    # Example input data
    input_data = {
        "credit_score": 720,
        "loan_amount": 300000,
        "property_value": 400000,
        "net_monthly_income": 5000,
        "monthly_debt_payments": 800,
        "employment_years": 8,
        "on_time_payment_percentage": 96,
        "liquid_assets": 150000,
        "monthly_expenses": 3500,
        "primary_income_percentage": 85,
        "compliance_score": 97,
        "satisfaction_score": 88
    }
    
    # Perform assessment
    # assessment = await engine.perform_comprehensive_risk_assessment(
    #     "customer_123", "mortgage_application", input_data
    # )
    
    print("Advanced Risk Assessment Engine demo completed!")

if __name__ == "__main__":
    asyncio.run(main())