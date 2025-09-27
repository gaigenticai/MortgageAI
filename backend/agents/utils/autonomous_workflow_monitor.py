"""
Autonomous Workflow Monitor for MortgageAI

This module provides comprehensive workflow monitoring capabilities for tracking agent decisions,
visualizing learning patterns, and analyzing performance metrics in real-time.

Features:
- Real-time agent decision tracking and logging
- Learning pattern visualization and analysis
- Performance analytics with predictive insights
- Workflow bottleneck detection and optimization recommendations
- Decision tree visualization and path analysis
- Multi-agent coordination monitoring
- Automated workflow optimization suggestions
- Advanced metrics collection and reporting
- Machine learning-based pattern recognition
- Real-time alerts and notification system
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import threading
import time
import pickle
import math
import statistics

# Advanced analytics and machine learning
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Time series analysis
from scipy import stats, signal
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Data processing and visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Network analysis for workflow dependencies
import networkx as nx

from ..config import settings


class WorkflowStatus(Enum):
    """Workflow execution statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    RETRY = "retry"
    TIMEOUT = "timeout"


class DecisionType(Enum):
    """Types of agent decisions."""
    CLASSIFICATION = "classification"
    RECOMMENDATION = "recommendation"
    VALIDATION = "validation"
    ESCALATION = "escalation"
    APPROVAL = "approval"
    REJECTION = "rejection"
    ROUTING = "routing"
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    INTERVENTION = "intervention"


class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    PROCESSING_TIME = "processing_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_UTILIZATION = "resource_utilization"
    USER_SATISFACTION = "user_satisfaction"
    COST_PER_DECISION = "cost_per_decision"


class LearningPatternType(Enum):
    """Types of learning patterns."""
    IMPROVEMENT_TREND = "improvement_trend"
    PERFORMANCE_PLATEAU = "performance_plateau"
    DEGRADATION = "degradation"
    SEASONAL_PATTERN = "seasonal_pattern"
    CONCEPT_DRIFT = "concept_drift"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    ADAPTATION = "adaptation"
    SPECIALIZATION = "specialization"


@dataclass
class AgentDecision:
    """Represents a single agent decision."""
    decision_id: str
    agent_id: str
    workflow_id: str
    decision_type: DecisionType
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence_score: float
    processing_time: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Decision quality metrics
    correctness_score: Optional[float] = None
    user_feedback_score: Optional[float] = None
    downstream_impact_score: Optional[float] = None
    
    # Learning indicators
    model_version: str = "1.0"
    training_data_version: str = "1.0"
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Performance tracking
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_details: Optional[str] = None
    retry_count: int = 0


@dataclass
class WorkflowExecution:
    """Represents a complete workflow execution."""
    workflow_id: str
    workflow_name: str
    execution_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Workflow structure
    steps: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    parallel_branches: List[List[str]] = field(default_factory=list)
    
    # Execution details
    decisions: List[AgentDecision] = field(default_factory=list)
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_processing_time: float = 0.0
    total_cost: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    overall_accuracy: float = 0.0
    user_satisfaction: Optional[float] = None
    business_impact: Optional[float] = None
    
    # Context and metadata
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    output_results: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPattern:
    """Represents a detected learning pattern."""
    pattern_id: str
    pattern_type: LearningPatternType
    agent_id: str
    metric_type: PerformanceMetricType
    
    # Pattern characteristics
    start_time: datetime
    end_time: datetime
    confidence: float
    statistical_significance: float
    
    # Pattern details
    trend_direction: str  # "improving", "declining", "stable"
    trend_magnitude: float
    seasonality_period: Optional[int] = None
    change_points: List[datetime] = field(default_factory=list)
    
    # Supporting data
    data_points: List[Tuple[datetime, float]] = field(default_factory=list)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    
    # Insights and recommendations
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    detection_method: str = "statistical_analysis"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement."""
    metric_id: str
    agent_id: str
    metric_type: PerformanceMetricType
    value: float
    timestamp: datetime
    
    # Context
    workflow_id: Optional[str] = None
    decision_id: Optional[str] = None
    measurement_context: Dict[str, Any] = field(default_factory=dict)
    
    # Quality indicators
    measurement_confidence: float = 1.0
    data_quality_score: float = 1.0
    
    # Metadata
    collection_method: str = "automated"
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowOptimizer:
    """Advanced workflow optimization using machine learning."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Optimization models
        self.bottleneck_detector = None
        self.performance_predictor = None
        self.resource_optimizer = None
        
        # Learning algorithms
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.pattern_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Optimization history
        self.optimization_history = []
        self.model_performance = {}
        
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize machine learning models for optimization."""
        try:
            # Bottleneck detection model
            self.bottleneck_detector = BottleneckDetectionNN()
            
            # Performance prediction model
            self.performance_predictor = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            )
            
            # Resource optimization model
            self.resource_optimizer = ResourceOptimizationModel()
            
            self.logger.info("Initialized workflow optimization models")
            
        except Exception as e:
            self.logger.error(f"Error initializing optimization models: {str(e)}")
    
    def detect_bottlenecks(self, workflow_executions: List[WorkflowExecution]) -> List[Dict[str, Any]]:
        """Detect bottlenecks in workflow executions."""
        try:
            bottlenecks = []
            
            # Analyze step-level performance
            step_performance = defaultdict(list)
            
            for execution in workflow_executions:
                for decision in execution.decisions:
                    step_performance[decision.agent_id].append({
                        'processing_time': decision.processing_time,
                        'timestamp': decision.timestamp,
                        'workflow_id': execution.workflow_id
                    })
            
            # Statistical analysis for bottleneck detection
            for agent_id, performances in step_performance.items():
                if len(performances) < 10:  # Need sufficient data
                    continue
                
                processing_times = [p['processing_time'] for p in performances]
                
                # Statistical measures
                mean_time = statistics.mean(processing_times)
                median_time = statistics.median(processing_times)
                std_dev = statistics.stdev(processing_times) if len(processing_times) > 1 else 0
                
                # Detect outliers (potential bottlenecks)
                q75, q25 = np.percentile(processing_times, [75, 25])
                iqr = q75 - q25
                outlier_threshold = q75 + 1.5 * iqr
                
                outliers = [t for t in processing_times if t > outlier_threshold]
                
                if len(outliers) > len(processing_times) * 0.1:  # More than 10% outliers
                    severity = "high" if len(outliers) > len(processing_times) * 0.2 else "medium"
                    
                    bottlenecks.append({
                        'agent_id': agent_id,
                        'bottleneck_type': 'processing_time',
                        'severity': severity,
                        'mean_processing_time': mean_time,
                        'outlier_threshold': outlier_threshold,
                        'outlier_percentage': len(outliers) / len(processing_times) * 100,
                        'recommendations': self._generate_bottleneck_recommendations(
                            agent_id, mean_time, std_dev, outliers
                        )
                    })
            
            # Analyze workflow-level dependencies
            dependency_bottlenecks = self._analyze_dependency_bottlenecks(workflow_executions)
            bottlenecks.extend(dependency_bottlenecks)
            
            return bottlenecks
            
        except Exception as e:
            self.logger.error(f"Error detecting bottlenecks: {str(e)}")
            return []
    
    def _generate_bottleneck_recommendations(self, agent_id: str, mean_time: float, 
                                           std_dev: float, outliers: List[float]) -> List[str]:
        """Generate recommendations for addressing bottlenecks."""
        recommendations = []
        
        if mean_time > 5.0:  # More than 5 seconds average
            recommendations.append(f"Consider optimizing {agent_id} algorithm or infrastructure")
        
        if std_dev > mean_time * 0.5:  # High variability
            recommendations.append(f"High variability detected in {agent_id} - investigate input data quality")
        
        if len(outliers) > 0:
            max_outlier = max(outliers)
            if max_outlier > mean_time * 3:
                recommendations.append(f"Extreme processing times detected - implement timeout mechanisms")
        
        recommendations.append(f"Monitor {agent_id} resource utilization for optimization opportunities")
        
        return recommendations
    
    def _analyze_dependency_bottlenecks(self, workflow_executions: List[WorkflowExecution]) -> List[Dict[str, Any]]:
        """Analyze workflow dependencies for bottlenecks."""
        bottlenecks = []
        
        try:
            # Build dependency graph
            dependency_graph = nx.DiGraph()
            step_times = defaultdict(list)
            
            for execution in workflow_executions:
                # Add nodes and edges
                for step in execution.steps:
                    if step not in dependency_graph:
                        dependency_graph.add_node(step)
                
                for step, deps in execution.dependencies.items():
                    for dep in deps:
                        dependency_graph.add_edge(dep, step)
                
                # Collect step timing data
                step_start_times = {}
                for decision in execution.decisions:
                    agent_id = decision.agent_id
                    if agent_id in step_start_times:
                        # Calculate waiting time
                        waiting_time = decision.timestamp.timestamp() - step_start_times[agent_id]
                        step_times[agent_id].append({
                            'processing_time': decision.processing_time,
                            'waiting_time': waiting_time,
                            'total_time': decision.processing_time + waiting_time
                        })
                    step_start_times[agent_id] = decision.timestamp.timestamp()
            
            # Analyze critical path
            if dependency_graph.nodes():
                # Find longest paths (critical paths)
                try:
                    critical_paths = []
                    for source in dependency_graph.nodes():
                        if dependency_graph.in_degree(source) == 0:  # Source nodes
                            for target in dependency_graph.nodes():
                                if dependency_graph.out_degree(target) == 0:  # Sink nodes
                                    if nx.has_path(dependency_graph, source, target):
                                        path = nx.shortest_path(dependency_graph, source, target)
                                        critical_paths.append(path)
                    
                    # Identify bottlenecks in critical paths
                    for path in critical_paths:
                        path_bottlenecks = self._analyze_path_bottlenecks(path, step_times)
                        bottlenecks.extend(path_bottlenecks)
                
                except nx.NetworkXError:
                    self.logger.warning("Could not analyze critical paths due to graph structure")
            
            return bottlenecks
            
        except Exception as e:
            self.logger.error(f"Error analyzing dependency bottlenecks: {str(e)}")
            return []
    
    def _analyze_path_bottlenecks(self, path: List[str], step_times: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Analyze bottlenecks within a critical path."""
        bottlenecks = []
        
        try:
            path_total_times = []
            
            for step in path:
                if step in step_times and step_times[step]:
                    times = step_times[step]
                    avg_total_time = statistics.mean([t['total_time'] for t in times])
                    avg_waiting_time = statistics.mean([t['waiting_time'] for t in times])
                    
                    path_total_times.append(avg_total_time)
                    
                    # Check for excessive waiting times
                    if avg_waiting_time > avg_total_time * 0.3:  # More than 30% waiting
                        bottlenecks.append({
                            'agent_id': step,
                            'bottleneck_type': 'dependency_waiting',
                            'severity': 'high' if avg_waiting_time > avg_total_time * 0.5 else 'medium',
                            'average_waiting_time': avg_waiting_time,
                            'average_total_time': avg_total_time,
                            'recommendations': [
                                f"Optimize dependencies for {step}",
                                "Consider parallel execution where possible",
                                f"Review resource allocation for {step}"
                            ]
                        })
            
            return bottlenecks
            
        except Exception as e:
            self.logger.error(f"Error analyzing path bottlenecks: {str(e)}")
            return []
    
    def optimize_workflow(self, workflow_executions: List[WorkflowExecution]) -> Dict[str, Any]:
        """Provide optimization recommendations for workflows."""
        try:
            optimization_results = {
                'optimization_id': str(uuid.uuid4()),
                'timestamp': datetime.now(timezone.utc),
                'bottlenecks': [],
                'optimization_opportunities': [],
                'predicted_improvements': {},
                'implementation_recommendations': []
            }
            
            # Detect bottlenecks
            bottlenecks = self.detect_bottlenecks(workflow_executions)
            optimization_results['bottlenecks'] = bottlenecks
            
            # Identify optimization opportunities
            opportunities = self._identify_optimization_opportunities(workflow_executions)
            optimization_results['optimization_opportunities'] = opportunities
            
            # Predict improvements
            improvements = self._predict_optimization_improvements(workflow_executions, opportunities)
            optimization_results['predicted_improvements'] = improvements
            
            # Generate implementation recommendations
            recommendations = self._generate_implementation_recommendations(bottlenecks, opportunities)
            optimization_results['implementation_recommendations'] = recommendations
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing workflow: {str(e)}")
            return {'error': str(e)}
    
    def _identify_optimization_opportunities(self, workflow_executions: List[WorkflowExecution]) -> List[Dict[str, Any]]:
        """Identify potential optimization opportunities."""
        opportunities = []
        
        try:
            # Analyze parallelization opportunities
            parallel_opportunities = self._analyze_parallelization_opportunities(workflow_executions)
            opportunities.extend(parallel_opportunities)
            
            # Analyze caching opportunities
            caching_opportunities = self._analyze_caching_opportunities(workflow_executions)
            opportunities.extend(caching_opportunities)
            
            # Analyze resource optimization opportunities
            resource_opportunities = self._analyze_resource_optimization(workflow_executions)
            opportunities.extend(resource_opportunities)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error identifying optimization opportunities: {str(e)}")
            return []
    
    def _analyze_parallelization_opportunities(self, workflow_executions: List[WorkflowExecution]) -> List[Dict[str, Any]]:
        """Analyze opportunities for parallel execution."""
        opportunities = []
        
        try:
            # Analyze sequential steps that could be parallel
            for execution in workflow_executions:
                sequential_steps = []
                current_step_time = {}
                
                for decision in execution.decisions:
                    agent_id = decision.agent_id
                    if agent_id not in current_step_time:
                        current_step_time[agent_id] = []
                    
                    current_step_time[agent_id].append({
                        'timestamp': decision.timestamp,
                        'processing_time': decision.processing_time,
                        'dependencies': execution.dependencies.get(agent_id, [])
                    })
                
                # Find steps that don't have dependencies and could run in parallel
                independent_steps = []
                for step, times in current_step_time.items():
                    if len(times) > 0:
                        deps = times[0]['dependencies']
                        if len(deps) == 0 or all(dep in sequential_steps for dep in deps):
                            independent_steps.append({
                                'step': step,
                                'avg_processing_time': statistics.mean([t['processing_time'] for t in times]),
                                'execution_count': len(times)
                            })
                
                if len(independent_steps) > 1:
                    total_sequential_time = sum([s['avg_processing_time'] for s in independent_steps])
                    max_parallel_time = max([s['avg_processing_time'] for s in independent_steps])
                    potential_savings = total_sequential_time - max_parallel_time
                    
                    if potential_savings > 1.0:  # At least 1 second savings
                        opportunities.append({
                            'type': 'parallelization',
                            'description': f"Parallelize {len(independent_steps)} independent steps",
                            'affected_steps': [s['step'] for s in independent_steps],
                            'potential_time_savings': potential_savings,
                            'implementation_complexity': 'medium',
                            'recommendations': [
                                "Implement parallel execution for independent steps",
                                "Ensure thread safety and resource coordination",
                                "Monitor resource utilization during parallel execution"
                            ]
                        })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing parallelization opportunities: {str(e)}")
            return []
    
    def _analyze_caching_opportunities(self, workflow_executions: List[WorkflowExecution]) -> List[Dict[str, Any]]:
        """Analyze opportunities for result caching."""
        opportunities = []
        
        try:
            # Analyze repeated inputs/outputs
            input_output_cache = defaultdict(list)
            
            for execution in workflow_executions:
                for decision in execution.decisions:
                    # Create a hash of input data for caching analysis
                    input_hash = hash(json.dumps(decision.input_data, sort_keys=True))
                    input_output_cache[decision.agent_id].append({
                        'input_hash': input_hash,
                        'output_data': decision.output_data,
                        'processing_time': decision.processing_time,
                        'timestamp': decision.timestamp
                    })
            
            # Find repeated computations
            for agent_id, computations in input_output_cache.items():
                if len(computations) < 10:  # Need sufficient data
                    continue
                
                input_frequency = defaultdict(int)
                input_processing_times = defaultdict(list)
                
                for comp in computations:
                    input_frequency[comp['input_hash']] += 1
                    input_processing_times[comp['input_hash']].append(comp['processing_time'])
                
                # Find frequently repeated inputs
                repeated_inputs = {k: v for k, v in input_frequency.items() if v > 1}
                
                if repeated_inputs:
                    total_repeated_computations = sum(repeated_inputs.values()) - len(repeated_inputs)
                    avg_processing_time = statistics.mean([
                        statistics.mean(times) for times in input_processing_times.values()
                    ])
                    potential_savings = total_repeated_computations * avg_processing_time
                    
                    if potential_savings > 5.0:  # At least 5 seconds savings
                        cache_hit_rate = total_repeated_computations / len(computations) * 100
                        
                        opportunities.append({
                            'type': 'caching',
                            'description': f"Implement result caching for {agent_id}",
                            'affected_steps': [agent_id],
                            'potential_time_savings': potential_savings,
                            'cache_hit_rate': cache_hit_rate,
                            'repeated_computations': total_repeated_computations,
                            'implementation_complexity': 'low',
                            'recommendations': [
                                f"Implement LRU cache for {agent_id} results",
                                "Set appropriate cache TTL based on data freshness requirements",
                                "Monitor cache hit rates and adjust cache size accordingly"
                            ]
                        })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing caching opportunities: {str(e)}")
            return []
    
    def _analyze_resource_optimization(self, workflow_executions: List[WorkflowExecution]) -> List[Dict[str, Any]]:
        """Analyze resource optimization opportunities."""
        opportunities = []
        
        try:
            # Analyze resource utilization patterns
            resource_usage = defaultdict(list)
            
            for execution in workflow_executions:
                for decision in execution.decisions:
                    if decision.resource_usage:
                        for resource_type, usage in decision.resource_usage.items():
                            resource_usage[resource_type].append({
                                'agent_id': decision.agent_id,
                                'usage': usage,
                                'processing_time': decision.processing_time,
                                'timestamp': decision.timestamp
                            })
            
            # Analyze each resource type
            for resource_type, usage_data in resource_usage.items():
                if len(usage_data) < 10:
                    continue
                
                usage_values = [d['usage'] for d in usage_data]
                
                # Statistical analysis
                mean_usage = statistics.mean(usage_values)
                std_usage = statistics.stdev(usage_values) if len(usage_values) > 1 else 0
                max_usage = max(usage_values)
                
                # Detect over-provisioning (low average utilization)
                if mean_usage < 0.3:  # Less than 30% average utilization
                    opportunities.append({
                        'type': 'resource_optimization',
                        'description': f"Optimize {resource_type} allocation - low utilization detected",
                        'resource_type': resource_type,
                        'current_avg_utilization': mean_usage,
                        'optimization_potential': 'high',
                        'implementation_complexity': 'medium',
                        'recommendations': [
                            f"Reduce {resource_type} allocation to improve efficiency",
                            "Implement dynamic resource scaling",
                            "Monitor performance after resource optimization"
                        ]
                    })
                
                # Detect resource contention (high variability)
                if std_usage > mean_usage * 0.5 and max_usage > 0.9:
                    opportunities.append({
                        'type': 'resource_optimization',
                        'description': f"Address {resource_type} contention - high variability detected",
                        'resource_type': resource_type,
                        'utilization_variability': std_usage / mean_usage,
                        'optimization_potential': 'medium',
                        'implementation_complexity': 'high',
                        'recommendations': [
                            f"Implement load balancing for {resource_type}",
                            "Consider resource pooling or queueing strategies",
                            "Monitor resource contention patterns"
                        ]
                    })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error analyzing resource optimization: {str(e)}")
            return []
    
    def _predict_optimization_improvements(self, workflow_executions: List[WorkflowExecution], 
                                         opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict improvements from optimization opportunities."""
        try:
            improvements = {
                'total_time_savings': 0.0,
                'total_cost_savings': 0.0,
                'resource_efficiency_gain': 0.0,
                'throughput_improvement': 0.0,
                'quality_impact': 'neutral'
            }
            
            # Calculate total potential time savings
            time_savings = sum([
                opp.get('potential_time_savings', 0) 
                for opp in opportunities 
                if 'potential_time_savings' in opp
            ])
            improvements['total_time_savings'] = time_savings
            
            # Estimate cost savings (assuming $0.001 per second of processing time)
            cost_per_second = 0.001
            improvements['total_cost_savings'] = time_savings * cost_per_second
            
            # Calculate resource efficiency gains
            resource_ops = [opp for opp in opportunities if opp['type'] == 'resource_optimization']
            if resource_ops:
                avg_utilization_improvement = 0.2  # Assume 20% improvement
                improvements['resource_efficiency_gain'] = avg_utilization_improvement
            
            # Estimate throughput improvement
            if time_savings > 0:
                # Simplified throughput calculation
                current_avg_time = self._calculate_average_execution_time(workflow_executions)
                if current_avg_time > 0:
                    new_avg_time = current_avg_time - time_savings
                    throughput_improvement = (current_avg_time - new_avg_time) / current_avg_time
                    improvements['throughput_improvement'] = throughput_improvement
            
            return improvements
            
        except Exception as e:
            self.logger.error(f"Error predicting optimization improvements: {str(e)}")
            return {}
    
    def _calculate_average_execution_time(self, workflow_executions: List[WorkflowExecution]) -> float:
        """Calculate average workflow execution time."""
        try:
            execution_times = []
            
            for execution in workflow_executions:
                if execution.end_time and execution.start_time:
                    duration = (execution.end_time - execution.start_time).total_seconds()
                    execution_times.append(duration)
            
            return statistics.mean(execution_times) if execution_times else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating average execution time: {str(e)}")
            return 0.0
    
    def _generate_implementation_recommendations(self, bottlenecks: List[Dict[str, Any]], 
                                               opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate implementation recommendations for optimizations."""
        recommendations = []
        
        try:
            # Prioritize recommendations based on impact and complexity
            all_items = bottlenecks + opportunities
            
            # Sort by potential impact
            high_impact_items = [item for item in all_items 
                               if item.get('severity') == 'high' or 
                                  item.get('optimization_potential') == 'high']
            
            medium_impact_items = [item for item in all_items 
                                 if item.get('severity') == 'medium' or 
                                    item.get('optimization_potential') == 'medium']
            
            # Generate recommendations
            priority = 1
            for item in high_impact_items + medium_impact_items:
                if priority > 10:  # Limit to top 10 recommendations
                    break
                
                rec = {
                    'priority': priority,
                    'title': item.get('description', f"Optimize {item.get('agent_id', 'workflow')}"),
                    'type': item.get('bottleneck_type', item.get('type', 'optimization')),
                    'implementation_complexity': item.get('implementation_complexity', 'medium'),
                    'estimated_effort': self._estimate_implementation_effort(item),
                    'expected_impact': item.get('potential_time_savings', 0),
                    'detailed_recommendations': item.get('recommendations', []),
                    'affected_components': item.get('affected_steps', [item.get('agent_id')])
                }
                
                recommendations.append(rec)
                priority += 1
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating implementation recommendations: {str(e)}")
            return []
    
    def _estimate_implementation_effort(self, item: Dict[str, Any]) -> str:
        """Estimate implementation effort for an optimization."""
        complexity = item.get('implementation_complexity', 'medium')
        
        effort_mapping = {
            'low': '1-2 days',
            'medium': '1-2 weeks',
            'high': '2-4 weeks'
        }
        
        return effort_mapping.get(complexity, '1-2 weeks')


class LearningPatternAnalyzer:
    """Analyzes learning patterns in agent behavior and performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Pattern detection models
        self.trend_detector = None
        self.seasonality_detector = None
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        
        # Historical patterns
        self.known_patterns = {}
        self.pattern_history = []
        
        self.initialize_analyzers()
    
    def initialize_analyzers(self):
        """Initialize pattern analysis components."""
        try:
            self.logger.info("Initialized learning pattern analyzers")
            
        except Exception as e:
            self.logger.error(f"Error initializing pattern analyzers: {str(e)}")
    
    async def analyze_learning_patterns(self, performance_metrics: List[PerformanceMetric], 
                                      agent_decisions: List[AgentDecision]) -> List[LearningPattern]:
        """Analyze learning patterns from performance data and decisions."""
        try:
            patterns = []
            
            # Group metrics by agent and type
            agent_metrics = defaultdict(lambda: defaultdict(list))
            
            for metric in performance_metrics:
                agent_metrics[metric.agent_id][metric.metric_type].append({
                    'timestamp': metric.timestamp,
                    'value': metric.value,
                    'context': metric.measurement_context
                })
            
            # Analyze patterns for each agent and metric type
            for agent_id, metrics_by_type in agent_metrics.items():
                for metric_type, metric_data in metrics_by_type.items():
                    if len(metric_data) < 10:  # Need sufficient data
                        continue
                    
                    # Sort by timestamp
                    metric_data.sort(key=lambda x: x['timestamp'])
                    
                    # Extract time series data
                    timestamps = [d['timestamp'] for d in metric_data]
                    values = [d['value'] for d in metric_data]
                    
                    # Detect various pattern types
                    trend_patterns = self._detect_trend_patterns(agent_id, metric_type, timestamps, values)
                    patterns.extend(trend_patterns)
                    
                    seasonal_patterns = self._detect_seasonal_patterns(agent_id, metric_type, timestamps, values)
                    patterns.extend(seasonal_patterns)
                    
                    anomaly_patterns = self._detect_anomaly_patterns(agent_id, metric_type, timestamps, values)
                    patterns.extend(anomaly_patterns)
                    
                    drift_patterns = self._detect_concept_drift(agent_id, metric_type, timestamps, values)
                    patterns.extend(drift_patterns)
            
            # Analyze decision patterns
            decision_patterns = self._analyze_decision_patterns(agent_decisions)
            patterns.extend(decision_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing learning patterns: {str(e)}")
            return []
    
    def _detect_trend_patterns(self, agent_id: str, metric_type: PerformanceMetricType, 
                             timestamps: List[datetime], values: List[float]) -> List[LearningPattern]:
        """Detect trend patterns in performance metrics."""
        patterns = []
        
        try:
            if len(values) < 10:
                return patterns
            
            # Convert timestamps to numeric values for analysis
            time_numeric = [(ts.timestamp() - timestamps[0].timestamp()) / 3600 for ts in timestamps]  # Hours
            
            # Linear trend detection
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
            
            # Determine if trend is significant
            significance_threshold = 0.05
            if p_value < significance_threshold and abs(r_value) > 0.3:  # Significant correlation
                
                if slope > 0:
                    trend_direction = "improving"
                    pattern_type = LearningPatternType.IMPROVEMENT_TREND
                elif slope < 0:
                    trend_direction = "declining" 
                    pattern_type = LearningPatternType.DEGRADATION
                else:
                    trend_direction = "stable"
                    pattern_type = LearningPatternType.PERFORMANCE_PLATEAU
                
                # Calculate trend magnitude
                value_range = max(values) - min(values)
                trend_magnitude = abs(slope) * len(time_numeric) / value_range if value_range > 0 else 0
                
                pattern = LearningPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=pattern_type,
                    agent_id=agent_id,
                    metric_type=metric_type,
                    start_time=timestamps[0],
                    end_time=timestamps[-1],
                    confidence=abs(r_value),
                    statistical_significance=1 - p_value,
                    trend_direction=trend_direction,
                    trend_magnitude=trend_magnitude,
                    data_points=list(zip(timestamps, values)),
                    statistical_tests={
                        'linear_regression': {
                            'slope': slope,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'standard_error': std_err
                        }
                    },
                    detection_method='linear_regression'
                )
                
                # Generate insights and recommendations
                pattern.insights = self._generate_trend_insights(pattern_type, slope, r_value)
                pattern.recommendations = self._generate_trend_recommendations(pattern_type, agent_id, metric_type)
                
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting trend patterns: {str(e)}")
            return []
    
    def _detect_seasonal_patterns(self, agent_id: str, metric_type: PerformanceMetricType,
                                timestamps: List[datetime], values: List[float]) -> List[LearningPattern]:
        """Detect seasonal patterns in performance metrics."""
        patterns = []
        
        try:
            if len(values) < 24:  # Need at least 24 data points for seasonality
                return patterns
            
            # Create time series
            ts_data = pd.Series(values, index=pd.to_datetime(timestamps))
            
            # Detect periodicity using autocorrelation
            autocorr = [ts_data.autocorr(lag=i) for i in range(1, min(len(values)//2, 24))]
            
            # Find significant periods
            significant_periods = []
            for i, corr in enumerate(autocorr, 1):
                if corr > 0.3:  # Significant autocorrelation
                    significant_periods.append((i, corr))
            
            if significant_periods:
                # Get the most significant period
                best_period, best_correlation = max(significant_periods, key=lambda x: x[1])
                
                # Perform seasonal decomposition if we have enough data
                try:
                    if len(values) >= best_period * 2:
                        decomposition = seasonal_decompose(ts_data, period=best_period, extrapolate_trend='freq')
                        
                        # Calculate seasonality strength
                        seasonal_component = decomposition.seasonal.dropna()
                        seasonal_strength = seasonal_component.std() / ts_data.std() if ts_data.std() > 0 else 0
                        
                        if seasonal_strength > 0.1:  # Significant seasonality
                            pattern = LearningPattern(
                                pattern_id=str(uuid.uuid4()),
                                pattern_type=LearningPatternType.SEASONAL_PATTERN,
                                agent_id=agent_id,
                                metric_type=metric_type,
                                start_time=timestamps[0],
                                end_time=timestamps[-1],
                                confidence=best_correlation,
                                statistical_significance=seasonal_strength,
                                trend_direction="cyclical",
                                trend_magnitude=seasonal_strength,
                                seasonality_period=best_period,
                                data_points=list(zip(timestamps, values)),
                                statistical_tests={
                                    'seasonality': {
                                        'period': best_period,
                                        'autocorrelation': best_correlation,
                                        'seasonal_strength': seasonal_strength
                                    }
                                },
                                detection_method='seasonal_decomposition'
                            )
                            
                            # Generate insights and recommendations
                            pattern.insights = self._generate_seasonal_insights(best_period, seasonal_strength)
                            pattern.recommendations = self._generate_seasonal_recommendations(agent_id, metric_type, best_period)
                            
                            patterns.append(pattern)
                
                except Exception as decomp_error:
                    self.logger.warning(f"Could not perform seasonal decomposition: {str(decomp_error)}")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting seasonal patterns: {str(e)}")
            return []
    
    def _detect_anomaly_patterns(self, agent_id: str, metric_type: PerformanceMetricType,
                               timestamps: List[datetime], values: List[float]) -> List[LearningPattern]:
        """Detect anomalous behavior patterns."""
        patterns = []
        
        try:
            if len(values) < 10:
                return patterns
            
            # Prepare data for anomaly detection
            X = np.array(values).reshape(-1, 1)
            
            # Detect anomalies
            anomaly_labels = self.anomaly_detector.fit_predict(X)
            anomaly_scores = self.anomaly_detector.decision_function(X)
            
            # Find anomalous periods
            anomaly_indices = [i for i, label in enumerate(anomaly_labels) if label == -1]
            
            if len(anomaly_indices) > len(values) * 0.05:  # More than 5% anomalies
                # Group consecutive anomalies
                anomaly_groups = []
                current_group = []
                
                for i in range(len(anomaly_indices)):
                    if i == 0 or anomaly_indices[i] - anomaly_indices[i-1] <= 2:
                        current_group.append(anomaly_indices[i])
                    else:
                        if current_group:
                            anomaly_groups.append(current_group)
                        current_group = [anomaly_indices[i]]
                
                if current_group:
                    anomaly_groups.append(current_group)
                
                # Create patterns for significant anomaly groups
                for group in anomaly_groups:
                    if len(group) >= 2:  # At least 2 consecutive anomalies
                        start_idx, end_idx = min(group), max(group)
                        
                        pattern = LearningPattern(
                            pattern_id=str(uuid.uuid4()),
                            pattern_type=LearningPatternType.ANOMALOUS_BEHAVIOR,
                            agent_id=agent_id,
                            metric_type=metric_type,
                            start_time=timestamps[start_idx],
                            end_time=timestamps[end_idx],
                            confidence=abs(np.mean([anomaly_scores[i] for i in group])),
                            statistical_significance=len(group) / len(values),
                            trend_direction="anomalous",
                            trend_magnitude=np.std([values[i] for i in group]),
                            data_points=[(timestamps[i], values[i]) for i in group],
                            statistical_tests={
                                'anomaly_detection': {
                                    'method': 'isolation_forest',
                                    'anomaly_count': len(group),
                                    'total_points': len(values),
                                    'anomaly_percentage': len(group) / len(values) * 100
                                }
                            },
                            detection_method='isolation_forest'
                        )
                        
                        # Generate insights and recommendations
                        pattern.insights = self._generate_anomaly_insights(len(group), len(values))
                        pattern.recommendations = self._generate_anomaly_recommendations(agent_id, metric_type)
                        
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting anomaly patterns: {str(e)}")
            return []
    
    def _detect_concept_drift(self, agent_id: str, metric_type: PerformanceMetricType,
                            timestamps: List[datetime], values: List[float]) -> List[LearningPattern]:
        """Detect concept drift in performance patterns."""
        patterns = []
        
        try:
            if len(values) < 20:  # Need sufficient data for drift detection
                return patterns
            
            # Split data into segments and compare distributions
            segment_size = max(10, len(values) // 4)  # At least 10 points per segment
            segments = []
            
            for i in range(0, len(values) - segment_size + 1, segment_size):
                segment = values[i:i + segment_size]
                if len(segment) >= segment_size:
                    segments.append({
                        'start_idx': i,
                        'end_idx': i + segment_size - 1,
                        'values': segment,
                        'mean': statistics.mean(segment),
                        'std': statistics.stdev(segment) if len(segment) > 1 else 0
                    })
            
            if len(segments) >= 2:
                # Compare adjacent segments for significant changes
                drift_points = []
                
                for i in range(len(segments) - 1):
                    current_segment = segments[i]
                    next_segment = segments[i + 1]
                    
                    # Statistical test for significant difference
                    try:
                        t_stat, p_value = stats.ttest_ind(current_segment['values'], next_segment['values'])
                        
                        # Check for significant change
                        if p_value < 0.05 and abs(t_stat) > 2:  # Significant difference
                            mean_change = abs(next_segment['mean'] - current_segment['mean'])
                            relative_change = mean_change / current_segment['mean'] if current_segment['mean'] != 0 else 0
                            
                            if relative_change > 0.1:  # More than 10% change
                                drift_points.append({
                                    'timestamp': timestamps[next_segment['start_idx']],
                                    'index': next_segment['start_idx'],
                                    'magnitude': relative_change,
                                    'p_value': p_value,
                                    't_statistic': t_stat
                                })
                    
                    except Exception as test_error:
                        self.logger.debug(f"T-test failed for segments: {str(test_error)}")
                        continue
                
                # Create drift patterns
                if drift_points:
                    for drift in drift_points:
                        pattern = LearningPattern(
                            pattern_id=str(uuid.uuid4()),
                            pattern_type=LearningPatternType.CONCEPT_DRIFT,
                            agent_id=agent_id,
                            metric_type=metric_type,
                            start_time=timestamps[max(0, drift['index'] - segment_size)],
                            end_time=timestamps[min(len(timestamps) - 1, drift['index'] + segment_size)],
                            confidence=1 - drift['p_value'],
                            statistical_significance=1 - drift['p_value'],
                            trend_direction="drift_detected",
                            trend_magnitude=drift['magnitude'],
                            change_points=[drift['timestamp']],
                            data_points=list(zip(timestamps, values)),
                            statistical_tests={
                                'concept_drift': {
                                    'drift_point': drift['timestamp'].isoformat(),
                                    'magnitude': drift['magnitude'],
                                    'p_value': drift['p_value'],
                                    't_statistic': drift['t_statistic']
                                }
                            },
                            detection_method='statistical_test'
                        )
                        
                        # Generate insights and recommendations
                        pattern.insights = self._generate_drift_insights(drift['magnitude'])
                        pattern.recommendations = self._generate_drift_recommendations(agent_id, metric_type)
                        
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting concept drift: {str(e)}")
            return []
    
    def _analyze_decision_patterns(self, agent_decisions: List[AgentDecision]) -> List[LearningPattern]:
        """Analyze patterns in agent decision making."""
        patterns = []
        
        try:
            # Group decisions by agent
            agent_decisions_grouped = defaultdict(list)
            for decision in agent_decisions:
                agent_decisions_grouped[decision.agent_id].append(decision)
            
            # Analyze decision patterns for each agent
            for agent_id, decisions in agent_decisions_grouped.items():
                if len(decisions) < 10:
                    continue
                
                # Analyze confidence evolution
                confidence_pattern = self._analyze_confidence_evolution(agent_id, decisions)
                if confidence_pattern:
                    patterns.append(confidence_pattern)
                
                # Analyze decision accuracy improvement
                accuracy_pattern = self._analyze_accuracy_evolution(agent_id, decisions)
                if accuracy_pattern:
                    patterns.append(accuracy_pattern)
                
                # Analyze specialization patterns
                specialization_pattern = self._analyze_specialization(agent_id, decisions)
                if specialization_pattern:
                    patterns.append(specialization_pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing decision patterns: {str(e)}")
            return []
    
    def _analyze_confidence_evolution(self, agent_id: str, decisions: List[AgentDecision]) -> Optional[LearningPattern]:
        """Analyze evolution of decision confidence over time."""
        try:
            # Sort decisions by timestamp
            decisions.sort(key=lambda x: x.timestamp)
            
            # Extract confidence scores over time
            timestamps = [d.timestamp for d in decisions]
            confidences = [d.confidence_score for d in decisions]
            
            if len(confidences) < 10:
                return None
            
            # Detect trends in confidence
            time_numeric = [(ts.timestamp() - timestamps[0].timestamp()) / 3600 for ts in timestamps]
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, confidences)
            
            # Check for significant trend
            if p_value < 0.05 and abs(r_value) > 0.3:
                if slope > 0:
                    pattern_type = LearningPatternType.IMPROVEMENT_TREND
                    trend_direction = "improving"
                elif slope < 0:
                    pattern_type = LearningPatternType.DEGRADATION
                    trend_direction = "declining"
                else:
                    return None
                
                pattern = LearningPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=pattern_type,
                    agent_id=agent_id,
                    metric_type=PerformanceMetricType.ACCURACY,  # Using accuracy as proxy for confidence
                    start_time=timestamps[0],
                    end_time=timestamps[-1],
                    confidence=abs(r_value),
                    statistical_significance=1 - p_value,
                    trend_direction=trend_direction,
                    trend_magnitude=abs(slope),
                    data_points=list(zip(timestamps, confidences)),
                    detection_method='confidence_analysis'
                )
                
                pattern.insights = [f"Decision confidence is {trend_direction} over time"]
                pattern.recommendations = [f"Monitor confidence trends for {agent_id}"]
                
                return pattern
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing confidence evolution: {str(e)}")
            return None
    
    def _analyze_accuracy_evolution(self, agent_id: str, decisions: List[AgentDecision]) -> Optional[LearningPattern]:
        """Analyze evolution of decision accuracy over time."""
        try:
            # Filter decisions with correctness scores
            decisions_with_accuracy = [d for d in decisions if d.correctness_score is not None]
            
            if len(decisions_with_accuracy) < 10:
                return None
            
            # Sort by timestamp
            decisions_with_accuracy.sort(key=lambda x: x.timestamp)
            
            timestamps = [d.timestamp for d in decisions_with_accuracy]
            accuracies = [d.correctness_score for d in decisions_with_accuracy]
            
            # Detect trends in accuracy
            time_numeric = [(ts.timestamp() - timestamps[0].timestamp()) / 3600 for ts in timestamps]
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, accuracies)
            
            # Check for significant trend
            if p_value < 0.05 and abs(r_value) > 0.3:
                if slope > 0:
                    pattern_type = LearningPatternType.IMPROVEMENT_TREND
                    trend_direction = "improving"
                elif slope < 0:
                    pattern_type = LearningPatternType.DEGRADATION
                    trend_direction = "declining"
                else:
                    return None
                
                pattern = LearningPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=pattern_type,
                    agent_id=agent_id,
                    metric_type=PerformanceMetricType.ACCURACY,
                    start_time=timestamps[0],
                    end_time=timestamps[-1],
                    confidence=abs(r_value),
                    statistical_significance=1 - p_value,
                    trend_direction=trend_direction,
                    trend_magnitude=abs(slope),
                    data_points=list(zip(timestamps, accuracies)),
                    detection_method='accuracy_analysis'
                )
                
                pattern.insights = [f"Decision accuracy is {trend_direction} over time"]
                pattern.recommendations = [f"{'Continue current training approach' if slope > 0 else 'Review training strategy'} for {agent_id}"]
                
                return pattern
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing accuracy evolution: {str(e)}")
            return None
    
    def _analyze_specialization(self, agent_id: str, decisions: List[AgentDecision]) -> Optional[LearningPattern]:
        """Analyze agent specialization patterns."""
        try:
            # Analyze decision types over time
            decision_type_counts = defaultdict(int)
            recent_decisions = decisions[-50:] if len(decisions) > 50 else decisions  # Last 50 decisions
            
            for decision in recent_decisions:
                decision_type_counts[decision.decision_type] += 1
            
            if len(decision_type_counts) == 0:
                return None
            
            # Calculate specialization index (entropy-based)
            total_decisions = sum(decision_type_counts.values())
            probabilities = [count / total_decisions for count in decision_type_counts.values()]
            entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
            max_entropy = math.log2(len(decision_type_counts))
            specialization_index = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
            
            # Check if agent is becoming specialized (low entropy)
            if specialization_index > 0.7:  # High specialization
                dominant_type = max(decision_type_counts, key=decision_type_counts.get)
                
                pattern = LearningPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=LearningPatternType.SPECIALIZATION,
                    agent_id=agent_id,
                    metric_type=PerformanceMetricType.ACCURACY,  # Using accuracy as general metric
                    start_time=recent_decisions[0].timestamp,
                    end_time=recent_decisions[-1].timestamp,
                    confidence=specialization_index,
                    statistical_significance=specialization_index,
                    trend_direction="specializing",
                    trend_magnitude=specialization_index,
                    data_points=[(datetime.now(timezone.utc), specialization_index)],
                    detection_method='entropy_analysis'
                )
                
                pattern.insights = [
                    f"Agent is specializing in {dominant_type.value} decisions",
                    f"Specialization index: {specialization_index:.2f}"
                ]
                pattern.recommendations = [
                    f"Monitor performance in specialized area: {dominant_type.value}",
                    "Consider cross-training to maintain versatility"
                ]
                
                return pattern
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing specialization: {str(e)}")
            return None
    
    def _generate_trend_insights(self, pattern_type: LearningPatternType, slope: float, r_value: float) -> List[str]:
        """Generate insights for trend patterns."""
        insights = []
        
        if pattern_type == LearningPatternType.IMPROVEMENT_TREND:
            insights.append(f"Performance is improving with correlation strength {r_value:.2f}")
            if slope > 0.1:
                insights.append("Rate of improvement is significant")
        elif pattern_type == LearningPatternType.DEGRADATION:
            insights.append(f"Performance is declining with correlation strength {r_value:.2f}")
            if abs(slope) > 0.1:
                insights.append("Rate of decline requires immediate attention")
        
        return insights
    
    def _generate_trend_recommendations(self, pattern_type: LearningPatternType, 
                                      agent_id: str, metric_type: PerformanceMetricType) -> List[str]:
        """Generate recommendations for trend patterns."""
        recommendations = []
        
        if pattern_type == LearningPatternType.IMPROVEMENT_TREND:
            recommendations.append(f"Continue current approach for {agent_id}")
            recommendations.append(f"Document successful strategies for {metric_type.value}")
        elif pattern_type == LearningPatternType.DEGRADATION:
            recommendations.append(f"Investigate causes of performance decline in {agent_id}")
            recommendations.append(f"Consider retraining or model updates for {metric_type.value}")
            recommendations.append("Implement additional monitoring and alerting")
        
        return recommendations
    
    def _generate_seasonal_insights(self, period: int, strength: float) -> List[str]:
        """Generate insights for seasonal patterns."""
        return [
            f"Seasonal pattern detected with period of {period} time units",
            f"Seasonal strength: {strength:.2f}",
            "Performance varies predictably over time cycles"
        ]
    
    def _generate_seasonal_recommendations(self, agent_id: str, metric_type: PerformanceMetricType, 
                                         period: int) -> List[str]:
        """Generate recommendations for seasonal patterns."""
        return [
            f"Adjust resource allocation for {agent_id} based on {period}-unit cycles",
            f"Implement seasonal forecasting for {metric_type.value}",
            "Consider time-based model variants to handle seasonality"
        ]
    
    def _generate_anomaly_insights(self, anomaly_count: int, total_count: int) -> List[str]:
        """Generate insights for anomaly patterns."""
        percentage = anomaly_count / total_count * 100
        return [
            f"Anomalous behavior detected in {percentage:.1f}% of observations",
            f"{anomaly_count} out of {total_count} data points are anomalous",
            "Investigate root causes of anomalous behavior"
        ]
    
    def _generate_anomaly_recommendations(self, agent_id: str, metric_type: PerformanceMetricType) -> List[str]:
        """Generate recommendations for anomaly patterns."""
        return [
            f"Investigate anomalous behavior in {agent_id}",
            f"Review data quality and processing for {metric_type.value}",
            "Implement anomaly alerting and automatic investigation",
            "Consider robust modeling approaches"
        ]
    
    def _generate_drift_insights(self, magnitude: float) -> List[str]:
        """Generate insights for concept drift patterns."""
        return [
            f"Concept drift detected with magnitude {magnitude:.2f}",
            "Performance distribution has significantly changed",
            "Model may need retraining or adaptation"
        ]
    
    def _generate_drift_recommendations(self, agent_id: str, metric_type: PerformanceMetricType) -> List[str]:
        """Generate recommendations for concept drift patterns."""
        return [
            f"Retrain or update model for {agent_id}",
            f"Implement drift monitoring for {metric_type.value}",
            "Consider online learning or adaptive algorithms",
            "Review recent changes in data sources or environment"
        ]


class BottleneckDetectionNN(nn.Module):
    """Neural network for bottleneck detection."""
    
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super(BottleneckDetectionNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class ResourceOptimizationModel:
    """Model for resource optimization recommendations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.optimizer_model = None
        
    def optimize_resources(self, resource_usage_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize resource allocation based on usage patterns."""
        try:
            # Implementation would include advanced optimization algorithms
            # For now, return a placeholder
            return {
                'optimization_recommendations': [],
                'predicted_savings': 0.0,
                'confidence': 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing resources: {str(e)}")
            return {}


class AutonomousWorkflowMonitor:
    """Main autonomous workflow monitoring system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.workflow_optimizer = WorkflowOptimizer()
        self.pattern_analyzer = LearningPatternAnalyzer()
        
        # Data storage
        self.workflow_executions: Dict[str, WorkflowExecution] = {}
        self.agent_decisions: List[AgentDecision] = []
        self.performance_metrics: List[PerformanceMetric] = []
        self.learning_patterns: List[LearningPattern] = []
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_collection_interval = 60  # seconds
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.alert_thresholds = {
            PerformanceMetricType.ACCURACY: 0.8,
            PerformanceMetricType.PROCESSING_TIME: 10.0,
            PerformanceMetricType.ERROR_RATE: 0.1
        }
        
        # Alerting system
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
    async def start_monitoring(self):
        """Start real-time workflow monitoring."""
        try:
            if self.monitoring_active:
                self.logger.warning("Monitoring is already active")
                return
            
            self.monitoring_active = True
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            
            self.logger.info("Started autonomous workflow monitoring")
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {str(e)}")
            self.monitoring_active = False
    
    async def stop_monitoring(self):
        """Stop real-time workflow monitoring."""
        try:
            if not self.monitoring_active:
                return
            
            self.monitoring_active = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            self.logger.info("Stopped autonomous workflow monitoring")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {str(e)}")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.monitoring_active:
            try:
                # Collect metrics
                asyncio.run(self._collect_real_time_metrics())
                
                # Analyze patterns
                asyncio.run(self._analyze_current_patterns())
                
                # Check for alerts
                asyncio.run(self._check_performance_alerts())
                
                # Sleep until next collection
                time.sleep(self.metrics_collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.metrics_collection_interval)
    
    async def _collect_real_time_metrics(self):
        """Collect real-time performance metrics."""
        try:
            # This would typically collect metrics from various sources
            # For now, we'll simulate metric collection
            current_time = datetime.now(timezone.utc)
            
            # Simulate metric collection for active agents
            for agent_id in ['compliance_agent', 'quality_control_agent', 'network_analyzer']:
                # Simulate some metrics
                accuracy_metric = PerformanceMetric(
                    metric_id=str(uuid.uuid4()),
                    agent_id=agent_id,
                    metric_type=PerformanceMetricType.ACCURACY,
                    value=np.random.normal(0.85, 0.1),  # Simulated accuracy
                    timestamp=current_time
                )
                
                processing_time_metric = PerformanceMetric(
                    metric_id=str(uuid.uuid4()),
                    agent_id=agent_id,
                    metric_type=PerformanceMetricType.PROCESSING_TIME,
                    value=abs(np.random.normal(2.5, 1.0)),  # Simulated processing time
                    timestamp=current_time
                )
                
                self.performance_metrics.extend([accuracy_metric, processing_time_metric])
                
                # Keep only recent metrics (last 1000)
                if len(self.performance_metrics) > 1000:
                    self.performance_metrics = self.performance_metrics[-1000:]
            
        except Exception as e:
            self.logger.error(f"Error collecting real-time metrics: {str(e)}")
    
    async def _analyze_current_patterns(self):
        """Analyze current learning and performance patterns."""
        try:
            # Analyze patterns with recent data
            recent_metrics = [m for m in self.performance_metrics 
                            if m.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)]
            
            recent_decisions = [d for d in self.agent_decisions 
                              if d.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)]
            
            if recent_metrics and recent_decisions:
                new_patterns = await self.pattern_analyzer.analyze_learning_patterns(
                    recent_metrics, recent_decisions
                )
                
                # Add new patterns
                self.learning_patterns.extend(new_patterns)
                
                # Keep only recent patterns (last 100)
                if len(self.learning_patterns) > 100:
                    self.learning_patterns = self.learning_patterns[-100:]
            
        except Exception as e:
            self.logger.error(f"Error analyzing current patterns: {str(e)}")
    
    async def _check_performance_alerts(self):
        """Check for performance alerts and trigger notifications."""
        try:
            current_time = datetime.now(timezone.utc)
            recent_window = timedelta(minutes=30)
            
            # Group recent metrics by agent and type
            recent_metrics = [m for m in self.performance_metrics 
                            if current_time - m.timestamp <= recent_window]
            
            agent_metrics = defaultdict(lambda: defaultdict(list))
            for metric in recent_metrics:
                agent_metrics[metric.agent_id][metric.metric_type].append(metric.value)
            
            # Check thresholds
            for agent_id, metrics_by_type in agent_metrics.items():
                for metric_type, values in metrics_by_type.items():
                    if not values:
                        continue
                    
                    avg_value = statistics.mean(values)
                    threshold = self.alert_thresholds.get(metric_type)
                    
                    if threshold and self._check_threshold_violation(metric_type, avg_value, threshold):
                        alert_data = {
                            'agent_id': agent_id,
                            'metric_type': metric_type.value,
                            'current_value': avg_value,
                            'threshold': threshold,
                            'timestamp': current_time.isoformat(),
                            'severity': self._calculate_alert_severity(metric_type, avg_value, threshold)
                        }
                        
                        await self._trigger_alert(f"Performance threshold violation: {agent_id}", alert_data)
            
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {str(e)}")
    
    def _check_threshold_violation(self, metric_type: PerformanceMetricType, value: float, threshold: float) -> bool:
        """Check if a metric value violates the threshold."""
        if metric_type in [PerformanceMetricType.ACCURACY, PerformanceMetricType.PRECISION, PerformanceMetricType.RECALL]:
            return value < threshold  # Lower is worse
        else:
            return value > threshold  # Higher is worse
    
    def _calculate_alert_severity(self, metric_type: PerformanceMetricType, value: float, threshold: float) -> str:
        """Calculate alert severity based on how much the threshold is violated."""
        if metric_type in [PerformanceMetricType.ACCURACY, PerformanceMetricType.PRECISION, PerformanceMetricType.RECALL]:
            violation_ratio = (threshold - value) / threshold
        else:
            violation_ratio = (value - threshold) / threshold
        
        if violation_ratio > 0.5:
            return "critical"
        elif violation_ratio > 0.2:
            return "high"
        elif violation_ratio > 0.1:
            return "medium"
        else:
            return "low"
    
    async def _trigger_alert(self, message: str, alert_data: Dict[str, Any]):
        """Trigger an alert notification."""
        try:
            self.logger.warning(f"ALERT: {message} - {alert_data}")
            
            # Call registered alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(message, alert_data)
                except Exception as callback_error:
                    self.logger.error(f"Error in alert callback: {str(callback_error)}")
            
        except Exception as e:
            self.logger.error(f"Error triggering alert: {str(e)}")
    
    async def log_agent_decision(self, decision: AgentDecision):
        """Log an agent decision for monitoring."""
        try:
            self.agent_decisions.append(decision)
            
            # Keep only recent decisions (last 1000)
            if len(self.agent_decisions) > 1000:
                self.agent_decisions = self.agent_decisions[-1000:]
            
            # Update performance tracking
            self.performance_history[decision.agent_id].append({
                'timestamp': decision.timestamp,
                'processing_time': decision.processing_time,
                'confidence': decision.confidence_score,
                'correctness': decision.correctness_score
            })
            
        except Exception as e:
            self.logger.error(f"Error logging agent decision: {str(e)}")
    
    async def log_workflow_execution(self, execution: WorkflowExecution):
        """Log a workflow execution for monitoring."""
        try:
            self.workflow_executions[execution.execution_id] = execution
            
            # Keep only recent executions (last 100)
            if len(self.workflow_executions) > 100:
                oldest_executions = sorted(
                    self.workflow_executions.items(),
                    key=lambda x: x[1].start_time
                )[:len(self.workflow_executions) - 100]
                
                for exec_id, _ in oldest_executions:
                    del self.workflow_executions[exec_id]
            
        except Exception as e:
            self.logger.error(f"Error logging workflow execution: {str(e)}")
    
    async def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data."""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Recent performance metrics
            recent_metrics = [m for m in self.performance_metrics 
                            if current_time - m.timestamp <= timedelta(hours=24)]
            
            # Group metrics by agent
            agent_performance = defaultdict(lambda: defaultdict(list))
            for metric in recent_metrics:
                agent_performance[metric.agent_id][metric.metric_type.value].append(metric.value)
            
            # Calculate summary statistics
            performance_summary = {}
            for agent_id, metrics in agent_performance.items():
                performance_summary[agent_id] = {}
                for metric_type, values in metrics.items():
                    if values:
                        performance_summary[agent_id][metric_type] = {
                            'current': values[-1] if values else 0,
                            'average': statistics.mean(values),
                            'trend': 'improving' if len(values) > 1 and values[-1] > values[0] else 'stable'
                        }
            
            # Recent learning patterns
            recent_patterns = [p for p in self.learning_patterns 
                             if current_time - p.end_time <= timedelta(hours=24)]
            
            # Workflow performance
            recent_executions = [e for e in self.workflow_executions.values() 
                               if current_time - e.start_time <= timedelta(hours=24)]
            
            workflow_stats = {
                'total_executions': len(recent_executions),
                'successful_executions': len([e for e in recent_executions if e.status == WorkflowStatus.COMPLETED]),
                'average_processing_time': statistics.mean([
                    e.total_processing_time for e in recent_executions 
                    if e.total_processing_time > 0
                ]) if recent_executions else 0
            }
            
            # Optimization recommendations
            optimization_results = await self.get_optimization_recommendations()
            
            dashboard_data = {
                'timestamp': current_time.isoformat(),
                'performance_summary': performance_summary,
                'learning_patterns': [
                    {
                        'pattern_type': p.pattern_type.value,
                        'agent_id': p.agent_id,
                        'confidence': p.confidence,
                        'insights': p.insights[:3]  # Top 3 insights
                    }
                    for p in recent_patterns[:10]  # Top 10 recent patterns
                ],
                'workflow_statistics': workflow_stats,
                'optimization_recommendations': optimization_results.get('implementation_recommendations', [])[:5],
                'active_alerts': len([p for p in recent_patterns if p.pattern_type == LearningPatternType.ANOMALOUS_BEHAVIOR]),
                'system_health': self._calculate_system_health()
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health metrics."""
        try:
            current_time = datetime.now(timezone.utc)
            recent_window = timedelta(hours=1)
            
            # Recent metrics for health calculation
            recent_metrics = [m for m in self.performance_metrics 
                            if current_time - m.timestamp <= recent_window]
            
            if not recent_metrics:
                return {'status': 'unknown', 'score': 0.5}
            
            # Calculate health score based on various factors
            health_factors = []
            
            # Accuracy factor
            accuracy_metrics = [m.value for m in recent_metrics 
                              if m.metric_type == PerformanceMetricType.ACCURACY]
            if accuracy_metrics:
                avg_accuracy = statistics.mean(accuracy_metrics)
                health_factors.append(min(avg_accuracy / 0.9, 1.0))  # Normalize to 0.9 as perfect
            
            # Processing time factor (lower is better)
            processing_time_metrics = [m.value for m in recent_metrics 
                                     if m.metric_type == PerformanceMetricType.PROCESSING_TIME]
            if processing_time_metrics:
                avg_processing_time = statistics.mean(processing_time_metrics)
                health_factors.append(max(0, 1 - (avg_processing_time / 10.0)))  # Normalize to 10s as baseline
            
            # Error rate factor (lower is better)
            error_rate_metrics = [m.value for m in recent_metrics 
                                 if m.metric_type == PerformanceMetricType.ERROR_RATE]
            if error_rate_metrics:
                avg_error_rate = statistics.mean(error_rate_metrics)
                health_factors.append(max(0, 1 - (avg_error_rate / 0.1)))  # Normalize to 0.1 as baseline
            
            # Calculate overall health score
            if health_factors:
                health_score = statistics.mean(health_factors)
            else:
                health_score = 0.5  # Neutral if no data
            
            # Determine health status
            if health_score >= 0.8:
                status = 'excellent'
            elif health_score >= 0.6:
                status = 'good'
            elif health_score >= 0.4:
                status = 'fair'
            elif health_score >= 0.2:
                status = 'poor'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'score': health_score,
                'factors_evaluated': len(health_factors),
                'last_update': current_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating system health: {str(e)}")
            return {'status': 'error', 'score': 0.0}
    
    async def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get workflow optimization recommendations."""
        try:
            recent_executions = list(self.workflow_executions.values())
            
            if not recent_executions:
                return {'message': 'No workflow executions available for optimization analysis'}
            
            # Get optimization recommendations
            optimization_results = self.workflow_optimizer.optimize_workflow(recent_executions)
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error getting optimization recommendations: {str(e)}")
            return {'error': str(e)}
    
    async def generate_learning_insights_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning insights report."""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Analyze all learning patterns
            pattern_summary = defaultdict(int)
            agent_patterns = defaultdict(list)
            
            for pattern in self.learning_patterns:
                pattern_summary[pattern.pattern_type.value] += 1
                agent_patterns[pattern.agent_id].append(pattern)
            
            # Generate insights by agent
            agent_insights = {}
            for agent_id, patterns in agent_patterns.items():
                insights = []
                recommendations = []
                
                # Analyze pattern distribution
                agent_pattern_types = [p.pattern_type for p in patterns]
                
                if LearningPatternType.IMPROVEMENT_TREND in agent_pattern_types:
                    insights.append("Shows consistent improvement trends")
                    recommendations.append("Continue current training approach")
                
                if LearningPatternType.DEGRADATION in agent_pattern_types:
                    insights.append("Performance degradation detected")
                    recommendations.append("Immediate review and retraining required")
                
                if LearningPatternType.ANOMALOUS_BEHAVIOR in agent_pattern_types:
                    insights.append("Anomalous behavior patterns detected")
                    recommendations.append("Investigate data quality and processing pipeline")
                
                if LearningPatternType.SPECIALIZATION in agent_pattern_types:
                    insights.append("Agent is developing specialization")
                    recommendations.append("Monitor specialization effects on versatility")
                
                agent_insights[agent_id] = {
                    'insights': insights,
                    'recommendations': recommendations,
                    'pattern_count': len(patterns),
                    'last_analysis': patterns[-1].end_time.isoformat() if patterns else None
                }
            
            # Overall system insights
            system_insights = []
            if pattern_summary.get('improvement_trend', 0) > pattern_summary.get('degradation', 0):
                system_insights.append("Overall system is showing improvement trends")
            else:
                system_insights.append("System performance needs attention")
            
            if pattern_summary.get('anomalous_behavior', 0) > len(self.learning_patterns) * 0.1:
                system_insights.append("High rate of anomalous behavior detected")
            
            report = {
                'report_id': str(uuid.uuid4()),
                'generated_at': current_time.isoformat(),
                'analysis_period': '24 hours',
                'pattern_summary': dict(pattern_summary),
                'agent_insights': agent_insights,
                'system_insights': system_insights,
                'total_patterns_analyzed': len(self.learning_patterns),
                'agents_monitored': len(agent_patterns)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating learning insights report: {str(e)}")
            return {'error': str(e)}
    
    def register_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register a callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def set_alert_threshold(self, metric_type: PerformanceMetricType, threshold: float):
        """Set custom alert threshold for a metric type."""
        self.alert_thresholds[metric_type] = threshold
    
    async def export_monitoring_data(self, filepath: str) -> bool:
        """Export monitoring data to file."""
        try:
            export_data = {
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'workflow_executions': [
                    {
                        'workflow_id': exec.workflow_id,
                        'execution_id': exec.execution_id,
                        'status': exec.status.value,
                        'start_time': exec.start_time.isoformat(),
                        'end_time': exec.end_time.isoformat() if exec.end_time else None,
                        'total_processing_time': exec.total_processing_time,
                        'overall_accuracy': exec.overall_accuracy
                    }
                    for exec in self.workflow_executions.values()
                ],
                'performance_metrics': [
                    {
                        'metric_id': metric.metric_id,
                        'agent_id': metric.agent_id,
                        'metric_type': metric.metric_type.value,
                        'value': metric.value,
                        'timestamp': metric.timestamp.isoformat()
                    }
                    for metric in self.performance_metrics
                ],
                'learning_patterns': [
                    {
                        'pattern_id': pattern.pattern_id,
                        'pattern_type': pattern.pattern_type.value,
                        'agent_id': pattern.agent_id,
                        'confidence': pattern.confidence,
                        'insights': pattern.insights,
                        'recommendations': pattern.recommendations
                    }
                    for pattern in self.learning_patterns
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting monitoring data: {str(e)}")
            return False


# Factory function for creating autonomous workflow monitor
def create_autonomous_workflow_monitor() -> AutonomousWorkflowMonitor:
    """Factory function to create and initialize AutonomousWorkflowMonitor."""
    return AutonomousWorkflowMonitor()


# Convenience functions for integration with other systems
async def start_workflow_monitoring() -> AutonomousWorkflowMonitor:
    """
    Start autonomous workflow monitoring.
    
    Returns:
        AutonomousWorkflowMonitor instance that can be used for monitoring operations
    """
    monitor = create_autonomous_workflow_monitor()
    await monitor.start_monitoring()
    return monitor


async def log_decision_for_monitoring(monitor: AutonomousWorkflowMonitor, 
                                    agent_id: str, decision_type: str, 
                                    input_data: Dict[str, Any], output_data: Dict[str, Any],
                                    processing_time: float, confidence_score: float) -> str:
    """
    Log an agent decision for monitoring.
    
    Args:
        monitor: The workflow monitor instance
        agent_id: ID of the agent making the decision
        decision_type: Type of decision being made
        input_data: Input data for the decision
        output_data: Output/result of the decision
        processing_time: Time taken to make the decision
        confidence_score: Confidence in the decision
        
    Returns:
        Decision ID for tracking
    """
    decision = AgentDecision(
        decision_id=str(uuid.uuid4()),
        agent_id=agent_id,
        workflow_id="default_workflow",
        decision_type=DecisionType(decision_type.lower()),
        input_data=input_data,
        output_data=output_data,
        confidence_score=confidence_score,
        processing_time=processing_time,
        timestamp=datetime.now(timezone.utc)
    )
    
    await monitor.log_agent_decision(decision)
    return decision.decision_id


async def get_monitoring_dashboard_data(monitor: AutonomousWorkflowMonitor) -> Dict[str, Any]:
    """
    Get current monitoring dashboard data.
    
    Args:
        monitor: The workflow monitor instance
        
    Returns:
        Dictionary with comprehensive dashboard data
    """
    return await monitor.get_performance_dashboard_data()


async def get_optimization_insights(monitor: AutonomousWorkflowMonitor) -> Dict[str, Any]:
    """
    Get workflow optimization insights and recommendations.
    
    Args:
        monitor: The workflow monitor instance
        
    Returns:
        Dictionary with optimization recommendations and insights
    """
    return await monitor.get_optimization_recommendations()


async def generate_learning_report(monitor: AutonomousWorkflowMonitor) -> Dict[str, Any]:
    """
    Generate comprehensive learning insights report.
    
    Args:
        monitor: The workflow monitor instance
        
    Returns:
        Dictionary with learning patterns analysis and insights
    """
    return await monitor.generate_learning_insights_report()
