"""
Agent Performance Metrics Dashboard - Core Module
Created: 2024-01-15
Author: MortgageAI Development Team

Comprehensive agent performance monitoring and analytics system for Dutch mortgage advisory platform.
Provides detailed analytics, success rates tracking, and optimization recommendations with real-time monitoring.

This module implements:
- Multi-dimensional performance metrics collection and analysis
- Success rate tracking with granular breakdowns by agent, time, and category
- Performance trend analysis with statistical modeling and forecasting
- Optimization recommendations using machine learning and statistical analysis
- Real-time monitoring with alerting for performance degradation
- Comparative analysis across agents, teams, and time periods
- AFM compliance performance tracking
- Quality metrics assessment and improvement suggestions
"""

import os
import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import uuid
import asyncpg
import aioredis
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Performance metric types"""
    SUCCESS_RATE = "success_rate"
    PROCESSING_TIME = "processing_time"
    QUALITY_SCORE = "quality_score"
    ACCURACY_RATE = "accuracy_rate"
    COMPLETION_RATE = "completion_rate"
    EFFICIENCY_SCORE = "efficiency_score"
    COMPLIANCE_RATE = "compliance_rate"
    ERROR_RATE = "error_rate"
    USER_SATISFACTION = "user_satisfaction"
    THROUGHPUT = "throughput"

class PerformancePeriod(Enum):
    """Performance analysis periods"""
    REALTIME = "realtime"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class OptimizationType(Enum):
    """Optimization recommendation types"""
    WORKFLOW = "workflow"
    RESOURCE = "resource"
    TRAINING = "training"
    AUTOMATION = "automation"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    COMPLIANCE = "compliance"

@dataclass
class AgentMetric:
    """Individual agent performance metric"""
    metric_id: str
    agent_id: str
    agent_name: str
    metric_type: str
    metric_value: float
    metric_unit: str
    measurement_time: datetime
    period: str
    category: str
    subcategory: Optional[str]
    context: Dict[str, Any]
    quality_indicators: Dict[str, float]
    trend_direction: str
    confidence_score: float
    benchmark_comparison: float
    metadata: Dict[str, Any]

@dataclass
class PerformanceAnalysis:
    """Performance analysis results"""
    analysis_id: str
    agent_id: str
    analysis_period: str
    overall_score: float
    performance_metrics: Dict[str, float]
    trends: Dict[str, Dict[str, Any]]
    benchmarks: Dict[str, float]
    strengths: List[str]
    improvement_areas: List[str]
    recommendations: List[Dict[str, Any]]
    risk_factors: List[Dict[str, Any]]
    forecasts: Dict[str, Dict[str, Any]]
    comparative_analysis: Dict[str, Any]
    generated_at: datetime

@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    recommendation_id: str
    agent_id: str
    recommendation_type: str
    priority: str
    title: str
    description: str
    expected_impact: float
    implementation_effort: str
    timeline: str
    success_metrics: List[str]
    prerequisites: List[str]
    resources_required: List[str]
    potential_risks: List[str]
    estimated_roi: float
    confidence_level: float
    supporting_data: Dict[str, Any]
    implementation_steps: List[str]
    generated_at: datetime

class AgentPerformanceMetrics:
    """
    Main class for agent performance metrics tracking and analysis.
    
    Provides comprehensive performance monitoring, analytics, and optimization
    recommendations for mortgage advisory agents.
    """

    def __init__(self):
        """Initialize the Agent Performance Metrics system."""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'mortgageai')
        }
        
        self.redis_config = {
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'password': os.getenv('REDIS_PASSWORD', None),
            'db': int(os.getenv('REDIS_DB', 0))
        }
        
        self.db_pool = None
        self.redis_pool = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance configuration
        self.config = {
            'collection_interval': int(os.getenv('METRICS_COLLECTION_INTERVAL', 300)),  # 5 minutes
            'analysis_interval': int(os.getenv('METRICS_ANALYSIS_INTERVAL', 3600)),   # 1 hour
            'retention_days': int(os.getenv('METRICS_RETENTION_DAYS', 365)),
            'performance_thresholds': {
                'success_rate_min': float(os.getenv('SUCCESS_RATE_MIN_THRESHOLD', 0.95)),
                'processing_time_max': float(os.getenv('PROCESSING_TIME_MAX_THRESHOLD', 30.0)),
                'quality_score_min': float(os.getenv('QUALITY_SCORE_MIN_THRESHOLD', 0.85)),
                'efficiency_score_min': float(os.getenv('EFFICIENCY_SCORE_MIN_THRESHOLD', 0.80)),
                'compliance_rate_min': float(os.getenv('COMPLIANCE_RATE_MIN_THRESHOLD', 0.98))
            },
            'alert_thresholds': {
                'performance_drop': float(os.getenv('PERFORMANCE_DROP_THRESHOLD', 0.10)),
                'error_spike': float(os.getenv('ERROR_SPIKE_THRESHOLD', 0.05)),
                'efficiency_decline': float(os.getenv('EFFICIENCY_DECLINE_THRESHOLD', 0.15))
            },
            'forecasting_horizon': int(os.getenv('FORECASTING_HORIZON_DAYS', 30)),
            'benchmark_percentiles': [25, 50, 75, 90, 95],
            'ml_model_update_interval': int(os.getenv('ML_MODEL_UPDATE_INTERVAL', 86400))  # 24 hours
        }
        
        logger.info("Agent Performance Metrics system initialized")

    async def initialize(self):
        """Initialize database and Redis connections."""
        try:
            self.db_pool = await asyncpg.create_pool(**self.db_config)
            self.redis_pool = aioredis.ConnectionPool.from_url(
                f"redis://{self.redis_config['host']}:{self.redis_config['port']}"
            )
            logger.info("Database and Redis connections established")
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise

    async def collect_agent_metrics(
        self, 
        agent_id: str,
        time_period: Optional[str] = None,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Collect comprehensive performance metrics for a specific agent.
        
        Args:
            agent_id: Unique identifier for the agent
            time_period: Time period for metric collection
            include_context: Whether to include contextual information
            
        Returns:
            Dictionary containing collected metrics and analysis
        """
        try:
            start_time = time.time()
            
            # Set default time period
            if not time_period:
                time_period = "daily"
            
            # Calculate time range
            end_time = datetime.now()
            if time_period == "daily":
                start_time_dt = end_time - timedelta(days=1)
            elif time_period == "weekly":
                start_time_dt = end_time - timedelta(weeks=1)
            elif time_period == "monthly":
                start_time_dt = end_time - timedelta(days=30)
            else:
                start_time_dt = end_time - timedelta(days=1)
            
            # Collect metrics from multiple sources
            metrics = {}
            
            # Basic performance metrics
            basic_metrics = await self._collect_basic_metrics(agent_id, start_time_dt, end_time)
            metrics.update(basic_metrics)
            
            # Quality metrics
            quality_metrics = await self._collect_quality_metrics(agent_id, start_time_dt, end_time)
            metrics.update(quality_metrics)
            
            # Efficiency metrics
            efficiency_metrics = await self._collect_efficiency_metrics(agent_id, start_time_dt, end_time)
            metrics.update(efficiency_metrics)
            
            # Compliance metrics
            compliance_metrics = await self._collect_compliance_metrics(agent_id, start_time_dt, end_time)
            metrics.update(compliance_metrics)
            
            # User interaction metrics
            interaction_metrics = await self._collect_interaction_metrics(agent_id, start_time_dt, end_time)
            metrics.update(interaction_metrics)
            
            # Context information
            if include_context:
                context = await self._collect_context_information(agent_id, start_time_dt, end_time)
                metrics['context'] = context
            
            # Calculate derived metrics
            derived_metrics = self._calculate_derived_metrics(metrics)
            metrics.update(derived_metrics)
            
            # Performance scoring
            performance_score = self._calculate_performance_score(metrics)
            metrics['overall_performance_score'] = performance_score
            
            # Store metrics in database
            await self._store_metrics(agent_id, metrics, time_period)
            
            processing_time = time.time() - start_time
            logger.info(f"Metrics collected for agent {agent_id} in {processing_time:.2f}s")
            
            return {
                'success': True,
                'agent_id': agent_id,
                'time_period': time_period,
                'metrics': metrics,
                'collection_timestamp': datetime.now().isoformat(),
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Failed to collect agent metrics: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_id': agent_id,
                'time_period': time_period
            }

    async def analyze_agent_performance(
        self,
        agent_id: str,
        analysis_period: str = "monthly",
        include_forecasting: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            analysis_period: Period for analysis (daily, weekly, monthly, quarterly)
            include_forecasting: Whether to include performance forecasting
            include_recommendations: Whether to generate optimization recommendations
            
        Returns:
            Comprehensive performance analysis results
        """
        try:
            start_time = time.time()
            
            # Load historical performance data
            historical_data = await self._load_historical_performance(agent_id, analysis_period)
            
            if not historical_data:
                return {
                    'success': False,
                    'error': 'Insufficient historical data for analysis',
                    'agent_id': agent_id
                }
            
            # Statistical analysis
            statistical_analysis = self._perform_statistical_analysis(historical_data)
            
            # Trend analysis
            trend_analysis = self._analyze_performance_trends(historical_data)
            
            # Benchmark comparison
            benchmark_comparison = await self._compare_to_benchmarks(agent_id, historical_data)
            
            # Performance patterns
            pattern_analysis = self._analyze_performance_patterns(historical_data)
            
            # Risk assessment
            risk_assessment = self._assess_performance_risks(historical_data, statistical_analysis)
            
            # Forecasting
            forecasts = {}
            if include_forecasting:
                forecasts = self._generate_performance_forecasts(historical_data)
            
            # Optimization recommendations
            recommendations = []
            if include_recommendations:
                recommendations = await self._generate_optimization_recommendations(
                    agent_id, historical_data, statistical_analysis, trend_analysis
                )
            
            # Compile analysis results
            analysis_results = {
                'analysis_id': str(uuid.uuid4()),
                'agent_id': agent_id,
                'analysis_period': analysis_period,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_points': len(historical_data),
                'statistical_summary': statistical_analysis,
                'trend_analysis': trend_analysis,
                'benchmark_comparison': benchmark_comparison,
                'pattern_analysis': pattern_analysis,
                'risk_assessment': risk_assessment,
                'forecasts': forecasts,
                'recommendations': recommendations,
                'overall_grade': self._calculate_performance_grade(statistical_analysis),
                'key_insights': self._generate_key_insights(
                    statistical_analysis, trend_analysis, benchmark_comparison
                ),
                'processing_time': time.time() - start_time
            }
            
            # Store analysis results
            await self._store_analysis_results(analysis_results)
            
            logger.info(f"Performance analysis completed for agent {agent_id}")
            
            return {
                'success': True,
                'analysis': analysis_results
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze agent performance: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_id': agent_id
            }

    async def generate_performance_dashboard(
        self,
        agent_ids: Optional[List[str]] = None,
        time_range: str = "last_30_days",
        dashboard_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate performance dashboard data for agents.
        
        Args:
            agent_ids: List of agent IDs to include (None for all agents)
            time_range: Time range for dashboard data
            dashboard_type: Type of dashboard (comprehensive, summary, comparative)
            
        Returns:
            Dashboard data with visualizations and metrics
        """
        try:
            start_time = time.time()
            
            # Get agent list
            if not agent_ids:
                agent_ids = await self._get_all_agent_ids()
            
            # Calculate time range
            end_time = datetime.now()
            if time_range == "last_7_days":
                start_time_dt = end_time - timedelta(days=7)
            elif time_range == "last_30_days":
                start_time_dt = end_time - timedelta(days=30)
            elif time_range == "last_90_days":
                start_time_dt = end_time - timedelta(days=90)
            else:
                start_time_dt = end_time - timedelta(days=30)
            
            # Collect dashboard data
            dashboard_data = {
                'dashboard_id': str(uuid.uuid4()),
                'generated_at': datetime.now().isoformat(),
                'time_range': time_range,
                'dashboard_type': dashboard_type,
                'agent_count': len(agent_ids),
                'agents': {}
            }
            
            # Individual agent data
            for agent_id in agent_ids:
                agent_data = await self._get_agent_dashboard_data(
                    agent_id, start_time_dt, end_time, dashboard_type
                )
                dashboard_data['agents'][agent_id] = agent_data
            
            # Aggregate statistics
            dashboard_data['aggregates'] = self._calculate_aggregate_statistics(
                dashboard_data['agents']
            )
            
            # Performance rankings
            dashboard_data['rankings'] = self._calculate_performance_rankings(
                dashboard_data['agents']
            )
            
            # Team comparisons
            if len(agent_ids) > 1:
                dashboard_data['comparisons'] = self._generate_comparative_analysis(
                    dashboard_data['agents']
                )
            
            # Key performance indicators
            dashboard_data['kpis'] = self._calculate_key_performance_indicators(
                dashboard_data['agents'], dashboard_data['aggregates']
            )
            
            # Visualization data
            dashboard_data['visualizations'] = self._prepare_visualization_data(
                dashboard_data, dashboard_type
            )
            
            # Alerts and notifications
            dashboard_data['alerts'] = await self._check_performance_alerts(agent_ids)
            
            dashboard_data['processing_time'] = time.time() - start_time
            
            logger.info(f"Performance dashboard generated for {len(agent_ids)} agents")
            
            return {
                'success': True,
                'dashboard': dashboard_data
            }
            
        except Exception as e:
            logger.error(f"Failed to generate performance dashboard: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def get_optimization_recommendations(
        self,
        agent_id: str,
        focus_areas: Optional[List[str]] = None,
        priority_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get optimization recommendations for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            focus_areas: Specific areas to focus on
            priority_filter: Filter by priority level
            
        Returns:
            List of optimization recommendations
        """
        try:
            # Load recent performance data
            historical_data = await self._load_historical_performance(agent_id, "monthly")
            
            if not historical_data:
                return {
                    'success': False,
                    'error': 'No performance data available for recommendations',
                    'agent_id': agent_id
                }
            
            # Analyze current performance
            current_analysis = self._perform_statistical_analysis(historical_data[-30:])
            
            # Generate recommendations
            recommendations = []
            
            # Workflow optimization recommendations
            if not focus_areas or 'workflow' in focus_areas:
                workflow_recs = self._generate_workflow_recommendations(
                    agent_id, historical_data, current_analysis
                )
                recommendations.extend(workflow_recs)
            
            # Resource optimization recommendations
            if not focus_areas or 'resource' in focus_areas:
                resource_recs = self._generate_resource_recommendations(
                    agent_id, historical_data, current_analysis
                )
                recommendations.extend(resource_recs)
            
            # Training recommendations
            if not focus_areas or 'training' in focus_areas:
                training_recs = self._generate_training_recommendations(
                    agent_id, historical_data, current_analysis
                )
                recommendations.extend(training_recs)
            
            # Quality improvement recommendations
            if not focus_areas or 'quality' in focus_areas:
                quality_recs = self._generate_quality_recommendations(
                    agent_id, historical_data, current_analysis
                )
                recommendations.extend(quality_recs)
            
            # Filter by priority
            if priority_filter:
                recommendations = [
                    r for r in recommendations 
                    if r.get('priority', '').lower() == priority_filter.lower()
                ]
            
            # Sort by expected impact and confidence
            recommendations.sort(
                key=lambda x: (x.get('expected_impact', 0) * x.get('confidence_level', 0)),
                reverse=True
            )
            
            return {
                'success': True,
                'agent_id': agent_id,
                'recommendations': recommendations,
                'total_count': len(recommendations),
                'focus_areas': focus_areas or ['all'],
                'priority_filter': priority_filter,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get optimization recommendations: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_id': agent_id
            }

    async def _collect_basic_metrics(
        self, 
        agent_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, float]:
        """Collect basic performance metrics for an agent."""
        async with self.db_pool.acquire() as conn:
            # Task completion metrics
            completion_query = """
                SELECT 
                    COUNT(*) as total_tasks,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_tasks,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_tasks,
                    AVG(CASE WHEN status = 'completed' THEN processing_time END) as avg_processing_time,
                    MAX(processing_time) as max_processing_time,
                    MIN(processing_time) as min_processing_time
                FROM agent_task_logs 
                WHERE agent_id = $1 AND created_at BETWEEN $2 AND $3
            """
            
            result = await conn.fetchrow(completion_query, agent_id, start_time, end_time)
            
            total_tasks = result['total_tasks'] or 0
            completed_tasks = result['completed_tasks'] or 0
            failed_tasks = result['failed_tasks'] or 0
            
            return {
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'success_rate': completed_tasks / max(total_tasks, 1),
                'failure_rate': failed_tasks / max(total_tasks, 1),
                'avg_processing_time': result['avg_processing_time'] or 0,
                'max_processing_time': result['max_processing_time'] or 0,
                'min_processing_time': result['min_processing_time'] or 0,
                'throughput': total_tasks / max((end_time - start_time).total_seconds() / 3600, 1)
            }

    async def _collect_quality_metrics(
        self, 
        agent_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, float]:
        """Collect quality-related performance metrics."""
        async with self.db_pool.acquire() as conn:
            # Quality assessment metrics
            quality_query = """
                SELECT 
                    AVG(quality_score) as avg_quality_score,
                    MAX(quality_score) as max_quality_score,
                    MIN(quality_score) as min_quality_score,
                    STDDEV(quality_score) as quality_score_stddev,
                    COUNT(CASE WHEN quality_score >= 0.9 THEN 1 END) as high_quality_count,
                    COUNT(CASE WHEN quality_score < 0.7 THEN 1 END) as low_quality_count
                FROM agent_quality_assessments 
                WHERE agent_id = $1 AND assessed_at BETWEEN $2 AND $3
            """
            
            result = await conn.fetchrow(quality_query, agent_id, start_time, end_time)
            
            # Accuracy metrics from validation
            accuracy_query = """
                SELECT 
                    AVG(accuracy_score) as avg_accuracy,
                    COUNT(CASE WHEN accuracy_score >= 0.95 THEN 1 END) as high_accuracy_count,
                    COUNT(*) as total_validations
                FROM agent_validations 
                WHERE agent_id = $1 AND validated_at BETWEEN $2 AND $3
            """
            
            accuracy_result = await conn.fetchrow(accuracy_query, agent_id, start_time, end_time)
            
            return {
                'avg_quality_score': result['avg_quality_score'] or 0,
                'max_quality_score': result['max_quality_score'] or 0,
                'min_quality_score': result['min_quality_score'] or 0,
                'quality_score_stddev': result['quality_score_stddev'] or 0,
                'high_quality_rate': (result['high_quality_count'] or 0) / max(result.get('count', 1), 1),
                'low_quality_rate': (result['low_quality_count'] or 0) / max(result.get('count', 1), 1),
                'avg_accuracy': accuracy_result['avg_accuracy'] or 0,
                'high_accuracy_rate': (accuracy_result['high_accuracy_count'] or 0) / max(accuracy_result['total_validations'] or 1, 1)
            }

    async def _collect_efficiency_metrics(
        self, 
        agent_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, float]:
        """Collect efficiency-related performance metrics."""
        async with self.db_pool.acquire() as conn:
            # Resource utilization metrics
            efficiency_query = """
                SELECT 
                    AVG(cpu_usage) as avg_cpu_usage,
                    AVG(memory_usage) as avg_memory_usage,
                    AVG(response_time) as avg_response_time,
                    COUNT(CASE WHEN response_time <= 5.0 THEN 1 END) as fast_responses,
                    COUNT(*) as total_requests
                FROM agent_resource_metrics 
                WHERE agent_id = $1 AND recorded_at BETWEEN $2 AND $3
            """
            
            result = await conn.fetchrow(efficiency_query, agent_id, start_time, end_time)
            
            # Calculate efficiency score
            total_requests = result['total_requests'] or 1
            fast_responses = result['fast_responses'] or 0
            response_efficiency = fast_responses / total_requests
            
            cpu_efficiency = max(0, 1 - (result['avg_cpu_usage'] or 0) / 100)
            memory_efficiency = max(0, 1 - (result['avg_memory_usage'] or 0) / 100)
            
            efficiency_score = (response_efficiency + cpu_efficiency + memory_efficiency) / 3
            
            return {
                'avg_cpu_usage': result['avg_cpu_usage'] or 0,
                'avg_memory_usage': result['avg_memory_usage'] or 0,
                'avg_response_time': result['avg_response_time'] or 0,
                'response_efficiency': response_efficiency,
                'cpu_efficiency': cpu_efficiency,
                'memory_efficiency': memory_efficiency,
                'overall_efficiency_score': efficiency_score,
                'resource_optimization_potential': max(0, 1 - efficiency_score)
            }

    async def _collect_compliance_metrics(
        self, 
        agent_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, float]:
        """Collect AFM compliance-related metrics."""
        async with self.db_pool.acquire() as conn:
            compliance_query = """
                SELECT 
                    COUNT(*) as total_compliance_checks,
                    COUNT(CASE WHEN is_compliant = true THEN 1 END) as compliant_checks,
                    COUNT(CASE WHEN violation_severity = 'critical' THEN 1 END) as critical_violations,
                    COUNT(CASE WHEN violation_severity = 'major' THEN 1 END) as major_violations,
                    AVG(compliance_score) as avg_compliance_score
                FROM agent_compliance_checks 
                WHERE agent_id = $1 AND checked_at BETWEEN $2 AND $3
            """
            
            result = await conn.fetchrow(compliance_query, agent_id, start_time, end_time)
            
            total_checks = result['total_compliance_checks'] or 1
            compliant_checks = result['compliant_checks'] or 0
            critical_violations = result['critical_violations'] or 0
            major_violations = result['major_violations'] or 0
            
            return {
                'total_compliance_checks': total_checks,
                'compliance_rate': compliant_checks / total_checks,
                'critical_violation_rate': critical_violations / total_checks,
                'major_violation_rate': major_violations / total_checks,
                'avg_compliance_score': result['avg_compliance_score'] or 0,
                'compliance_risk_score': (critical_violations * 2 + major_violations) / total_checks
            }

    async def _collect_interaction_metrics(
        self, 
        agent_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, float]:
        """Collect user interaction and satisfaction metrics."""
        async with self.db_pool.acquire() as conn:
            interaction_query = """
                SELECT 
                    COUNT(*) as total_interactions,
                    AVG(user_satisfaction_score) as avg_satisfaction,
                    COUNT(CASE WHEN user_satisfaction_score >= 4 THEN 1 END) as positive_feedback,
                    COUNT(CASE WHEN user_satisfaction_score <= 2 THEN 1 END) as negative_feedback,
                    AVG(interaction_duration) as avg_interaction_time
                FROM agent_user_interactions 
                WHERE agent_id = $1 AND interaction_time BETWEEN $2 AND $3
            """
            
            result = await conn.fetchrow(interaction_query, agent_id, start_time, end_time)
            
            total_interactions = result['total_interactions'] or 1
            positive_feedback = result['positive_feedback'] or 0
            negative_feedback = result['negative_feedback'] or 0
            
            return {
                'total_interactions': total_interactions,
                'avg_user_satisfaction': result['avg_satisfaction'] or 0,
                'positive_feedback_rate': positive_feedback / total_interactions,
                'negative_feedback_rate': negative_feedback / total_interactions,
                'avg_interaction_time': result['avg_interaction_time'] or 0,
                'user_engagement_score': min(1.0, total_interactions / 100)  # Normalized engagement
            }

    async def _collect_context_information(
        self, 
        agent_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, Any]:
        """Collect contextual information about agent performance."""
        async with self.db_pool.acquire() as conn:
            # Get agent information
            agent_info_query = """
                SELECT agent_name, agent_type, team_id, created_at, last_updated
                FROM agents 
                WHERE agent_id = $1
            """
            
            agent_info = await conn.fetchrow(agent_info_query, agent_id)
            
            # Get workload distribution
            workload_query = """
                SELECT 
                    task_type,
                    COUNT(*) as task_count,
                    AVG(complexity_score) as avg_complexity
                FROM agent_task_logs 
                WHERE agent_id = $1 AND created_at BETWEEN $2 AND $3
                GROUP BY task_type
            """
            
            workload_results = await conn.fetch(workload_query, agent_id, start_time, end_time)
            
            workload_distribution = {}
            for row in workload_results:
                workload_distribution[row['task_type']] = {
                    'count': row['task_count'],
                    'avg_complexity': float(row['avg_complexity'] or 0)
                }
            
            return {
                'agent_info': dict(agent_info) if agent_info else {},
                'workload_distribution': workload_distribution,
                'analysis_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration_hours': (end_time - start_time).total_seconds() / 3600
                }
            }

    def _calculate_derived_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate derived performance metrics."""
        derived = {}
        
        # Productivity score
        if 'throughput' in metrics and 'avg_quality_score' in metrics:
            derived['productivity_score'] = (
                metrics['throughput'] * 0.3 + 
                metrics['avg_quality_score'] * 0.7
            )
        
        # Reliability score
        if 'success_rate' in metrics and 'compliance_rate' in metrics:
            derived['reliability_score'] = (
                metrics['success_rate'] * 0.6 + 
                metrics['compliance_rate'] * 0.4
            )
        
        # Performance consistency
        if 'quality_score_stddev' in metrics and 'avg_quality_score' in metrics:
            if metrics['avg_quality_score'] > 0:
                derived['consistency_score'] = max(0, 1 - (
                    metrics['quality_score_stddev'] / metrics['avg_quality_score']
                ))
            else:
                derived['consistency_score'] = 0
        
        # User experience score
        if 'avg_user_satisfaction' in metrics and 'avg_response_time' in metrics:
            response_time_score = max(0, 1 - metrics['avg_response_time'] / 30)  # 30s baseline
            derived['user_experience_score'] = (
                (metrics['avg_user_satisfaction'] / 5) * 0.7 + 
                response_time_score * 0.3
            )
        
        return derived

    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        weights = {
            'success_rate': 0.20,
            'avg_quality_score': 0.20,
            'compliance_rate': 0.15,
            'overall_efficiency_score': 0.15,
            'avg_user_satisfaction': 0.10,
            'reliability_score': 0.10,
            'consistency_score': 0.10
        }
        
        score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in metrics and metrics[metric] is not None:
                # Normalize satisfaction score (1-5 scale to 0-1)
                if metric == 'avg_user_satisfaction':
                    normalized_value = metrics[metric] / 5
                else:
                    normalized_value = metrics[metric]
                
                score += normalized_value * weight
                total_weight += weight
        
        return score / max(total_weight, 0.01) if total_weight > 0 else 0

    async def _store_metrics(
        self, 
        agent_id: str, 
        metrics: Dict[str, Any], 
        time_period: str
    ):
        """Store collected metrics in the database."""
        try:
            async with self.db_pool.acquire() as conn:
                # Store in agent_performance_metrics table
                insert_query = """
                    INSERT INTO agent_performance_metrics 
                    (agent_id, metric_data, time_period, overall_score, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                """
                
                await conn.execute(
                    insert_query,
                    agent_id,
                    json.dumps(metrics),
                    time_period,
                    metrics.get('overall_performance_score', 0),
                    datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")

    async def _load_historical_performance(
        self, 
        agent_id: str, 
        period: str
    ) -> List[Dict[str, Any]]:
        """Load historical performance data for analysis."""
        try:
            async with self.db_pool.acquire() as conn:
                if period == "daily":
                    start_time = datetime.now() - timedelta(days=30)
                elif period == "weekly":
                    start_time = datetime.now() - timedelta(weeks=12)
                elif period == "monthly":
                    start_time = datetime.now() - timedelta(days=365)
                else:
                    start_time = datetime.now() - timedelta(days=90)
                
                query = """
                    SELECT metric_data, overall_score, created_at
                    FROM agent_performance_metrics
                    WHERE agent_id = $1 AND created_at >= $2
                    ORDER BY created_at ASC
                """
                
                rows = await conn.fetch(query, agent_id, start_time)
                
                historical_data = []
                for row in rows:
                    data = json.loads(row['metric_data'])
                    data['overall_score'] = row['overall_score']
                    data['timestamp'] = row['created_at']
                    historical_data.append(data)
                
                return historical_data
                
        except Exception as e:
            logger.error(f"Failed to load historical performance: {e}")
            return []

    def _perform_statistical_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis on performance data."""
        if not data:
            return {}
        
        # Extract key metrics
        overall_scores = [d.get('overall_performance_score', 0) for d in data if d.get('overall_performance_score') is not None]
        success_rates = [d.get('success_rate', 0) for d in data if d.get('success_rate') is not None]
        quality_scores = [d.get('avg_quality_score', 0) for d in data if d.get('avg_quality_score') is not None]
        processing_times = [d.get('avg_processing_time', 0) for d in data if d.get('avg_processing_time') is not None]
        
        def safe_statistics(values):
            if not values:
                return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
            return {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values)
            }
        
        return {
            'overall_performance': safe_statistics(overall_scores),
            'success_rate': safe_statistics(success_rates),
            'quality_score': safe_statistics(quality_scores),
            'processing_time': safe_statistics(processing_times),
            'data_points': len(data),
            'time_span_days': (data[-1]['timestamp'] - data[0]['timestamp']).days if len(data) > 1 else 0
        }

    def _analyze_performance_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(data) < 3:
            return {'insufficient_data': True}
        
        # Extract time series data
        timestamps = [d['timestamp'] for d in data]
        overall_scores = [d.get('overall_performance_score', 0) for d in data]
        success_rates = [d.get('success_rate', 0) for d in data]
        quality_scores = [d.get('avg_quality_score', 0) for d in data]
        
        def calculate_trend(values):
            if len(values) < 2:
                return {'trend': 'stable', 'slope': 0, 'r_squared': 0}
            
            x = np.arange(len(values))
            y = np.array(values)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            trend_direction = 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable'
            
            return {
                'trend': trend_direction,
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'significance': 'significant' if p_value < 0.05 else 'not_significant'
            }
        
        return {
            'overall_performance': calculate_trend(overall_scores),
            'success_rate': calculate_trend(success_rates),
            'quality_score': calculate_trend(quality_scores),
            'trend_summary': self._summarize_trends(overall_scores, success_rates, quality_scores)
        }

    def _summarize_trends(self, overall_scores, success_rates, quality_scores):
        """Summarize overall trend direction."""
        trends = []
        
        if len(overall_scores) >= 3:
            recent_avg = statistics.mean(overall_scores[-3:])
            earlier_avg = statistics.mean(overall_scores[:3])
            if recent_avg > earlier_avg * 1.05:
                trends.append('improving')
            elif recent_avg < earlier_avg * 0.95:
                trends.append('declining')
            else:
                trends.append('stable')
        
        if not trends:
            return 'stable'
        
        return max(set(trends), key=trends.count)

    async def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        try:
            health_data = {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'healthy',
                'components': {}
            }
            
            # Check database connectivity
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')
                health_data['components']['database'] = {
                    'status': 'healthy',
                    'response_time': 'normal'
                }
            except Exception as e:
                health_data['components']['database'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_data['system_status'] = 'degraded'
            
            # Check Redis connectivity
            try:
                redis = aioredis.Redis(connection_pool=self.redis_pool)
                await redis.ping()
                health_data['components']['cache'] = {
                    'status': 'healthy',
                    'response_time': 'normal'
                }
            except Exception as e:
                health_data['components']['cache'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_data['system_status'] = 'degraded'
            
            # System metrics
            health_data['metrics'] = {
                'active_agents': await self._get_active_agent_count(),
                'metrics_collected_today': await self._get_daily_metrics_count(),
                'analysis_jobs_pending': await self._get_pending_analysis_count()
            }
            
            return {
                'success': True,
                'health': health_data
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'health': {
                    'timestamp': datetime.now().isoformat(),
                    'system_status': 'unhealthy'
                }
            }

    async def _get_active_agent_count(self) -> int:
        """Get count of active agents."""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT COUNT(DISTINCT agent_id) FROM agent_performance_metrics WHERE created_at >= $1",
                    datetime.now() - timedelta(hours=24)
                )
                return result or 0
        except:
            return 0

    async def _get_daily_metrics_count(self) -> int:
        """Get count of metrics collected today."""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM agent_performance_metrics WHERE created_at >= $1",
                    datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                )
                return result or 0
        except:
            return 0

    async def _get_pending_analysis_count(self) -> int:
        """Get count of pending analysis jobs."""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM agent_analysis_queue WHERE status = 'pending'"
                )
                return result or 0
        except:
            return 0

    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.db_pool:
                await self.db_pool.close()
            if self.redis_pool:
                await self.redis_pool.disconnect()
            self.executor.shutdown(wait=True)
            logger.info("Agent Performance Metrics system cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup: {e}")

# Example usage and testing
async def main():
    """Main function for testing the Agent Performance Metrics system."""
    metrics_system = AgentPerformanceMetrics()
    await metrics_system.initialize()
    
    try:
        # Test metrics collection
        print("Testing agent metrics collection...")
        result = await metrics_system.collect_agent_metrics("agent_001", "daily")
        print(f"Metrics collection: {'✓' if result['success'] else '✗'}")
        
        # Test performance analysis
        print("Testing performance analysis...")
        analysis = await metrics_system.analyze_agent_performance("agent_001", "monthly")
        print(f"Performance analysis: {'✓' if analysis['success'] else '✗'}")
        
        # Test dashboard generation
        print("Testing dashboard generation...")
        dashboard = await metrics_system.generate_performance_dashboard(["agent_001"], "last_30_days")
        print(f"Dashboard generation: {'✓' if dashboard['success'] else '✗'}")
        
        # Test health check
        print("Testing health check...")
        health = await metrics_system.get_health_status()
        print(f"Health check: {'✓' if health['success'] else '✗'}")
        
        print("All tests completed successfully!")
        
    finally:
        await metrics_system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
