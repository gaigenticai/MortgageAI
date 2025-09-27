#!/usr/bin/env python3
"""
Autonomous Workflow Monitor Executor

This script serves as the execution interface for the autonomous workflow monitoring system.
It handles requests from the Node.js API and coordinates with the AutonomousWorkflowMonitor
classes to perform real-time monitoring, pattern analysis, and optimization operations.

Usage:
    python3 workflow_monitor_executor.py <operation> [options]

Operations:
    start_monitoring - Start autonomous workflow monitoring
    stop_monitoring - Stop active monitoring session
    log_decision - Log an agent decision for analysis
    log_workflow - Log a workflow execution
    log_metric - Log a performance metric
    get_dashboard_data - Get comprehensive dashboard data
    analyze_patterns - Analyze learning patterns
    optimize_workflow - Get workflow optimization recommendations
    generate_report - Generate comprehensive reports
    set_alert_thresholds - Configure alert thresholds
    export_data - Export monitoring data
    get_status - Get monitoring status
    health_check - Service health check

Input/Output:
    - Input data is provided via stdin as JSON
    - Output results are returned via stdout as JSON
    - Errors are logged to stderr
"""

import sys
import json
import logging
import asyncio
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import uuid
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import workflow monitoring components
from autonomous_workflow_monitor import (
    AutonomousWorkflowMonitor,
    AgentDecision, WorkflowExecution, PerformanceMetric, LearningPattern,
    DecisionType, WorkflowStatus, PerformanceMetricType, LearningPatternType,
    start_workflow_monitoring, log_decision_for_monitoring,
    get_monitoring_dashboard_data, get_optimization_insights, generate_learning_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('/app/logs/workflow_monitoring.log', mode='a') if os.path.exists('/app/logs') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


class WorkflowMonitorExecutor:
    """Main executor class for workflow monitoring operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitor = None
        self.active_sessions = {}
    
    async def initialize_monitor(self, session_id: str = None) -> AutonomousWorkflowMonitor:
        """Initialize or retrieve the workflow monitor instance."""
        if session_id and session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        if not self.monitor:
            self.monitor = AutonomousWorkflowMonitor()
        
        if session_id:
            self.active_sessions[session_id] = self.monitor
        
        return self.monitor
    
    def validate_input_data(self, data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate input data contains required fields."""
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        return True
    
    def convert_decision_type(self, decision_type_str: str) -> DecisionType:
        """Convert string to DecisionType enum."""
        try:
            return DecisionType(decision_type_str.lower())
        except ValueError:
            self.logger.warning(f"Unknown decision type: {decision_type_str}, using CLASSIFICATION as default")
            return DecisionType.CLASSIFICATION
    
    def convert_metric_type(self, metric_type_str: str) -> PerformanceMetricType:
        """Convert string to PerformanceMetricType enum."""
        try:
            return PerformanceMetricType(metric_type_str.lower())
        except ValueError:
            self.logger.warning(f"Unknown metric type: {metric_type_str}, using ACCURACY as default")
            return PerformanceMetricType.ACCURACY
    
    async def start_monitoring(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Start autonomous workflow monitoring session."""
        try:
            self.validate_input_data(input_data, ['session_id'])
            
            session_id = input_data['session_id']
            configuration = input_data.get('configuration', {})
            
            # Initialize monitor for this session
            monitor = await self.initialize_monitor(session_id)
            
            # Configure monitor if settings provided
            if 'alert_thresholds' in configuration:
                thresholds = configuration['alert_thresholds']
                for metric_name, threshold_value in thresholds.items():
                    try:
                        metric_type = self.convert_metric_type(metric_name)
                        monitor.set_alert_threshold(metric_type, threshold_value)
                    except Exception as e:
                        self.logger.warning(f"Failed to set threshold for {metric_name}: {str(e)}")
            
            # Start monitoring
            await monitor.start_monitoring()
            
            # Store session info
            self.active_sessions[session_id] = monitor
            
            return {
                'success': True,
                'session_id': session_id,
                'configuration': {
                    'metrics_collection_interval': monitor.metrics_collection_interval,
                    'alert_thresholds': {
                        metric_type.value: threshold for metric_type, threshold in monitor.alert_thresholds.items()
                    },
                    'monitoring_active': monitor.monitoring_active
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def stop_monitoring(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stop autonomous workflow monitoring session."""
        try:
            self.validate_input_data(input_data, ['session_id'])
            
            session_id = input_data['session_id']
            
            if session_id not in self.active_sessions:
                return {
                    'success': False,
                    'error': 'No active monitoring session found'
                }
            
            monitor = self.active_sessions[session_id]
            
            # Generate session summary before stopping
            session_summary = {
                'total_decisions': len(monitor.agent_decisions),
                'total_metrics': len(monitor.performance_metrics),
                'total_patterns': len(monitor.learning_patterns),
                'total_workflows': len(monitor.workflow_executions)
            }
            
            # Stop monitoring
            await monitor.stop_monitoring()
            
            # Clean up session
            del self.active_sessions[session_id]
            
            return {
                'success': True,
                'session_summary': session_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def log_decision(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log an agent decision for monitoring and analysis."""
        try:
            self.validate_input_data(input_data, [
                'decision_id', 'agent_id', 'decision_type', 'input_data', 
                'output_data', 'confidence_score', 'processing_time'
            ])
            
            # Create decision object
            decision = AgentDecision(
                decision_id=input_data['decision_id'],
                agent_id=input_data['agent_id'],
                workflow_id=input_data.get('workflow_id', 'default_workflow'),
                decision_type=self.convert_decision_type(input_data['decision_type']),
                input_data=input_data['input_data'],
                output_data=input_data['output_data'],
                confidence_score=input_data['confidence_score'],
                processing_time=input_data['processing_time'],
                timestamp=datetime.fromisoformat(input_data.get('logged_at', datetime.now(timezone.utc).isoformat())),
                context=input_data.get('context', {}),
                metadata=input_data.get('metadata', {}),
                correctness_score=input_data.get('correctness_score'),
                user_feedback_score=input_data.get('user_feedback_score')
            )
            
            # Log to monitor (use default monitor if no specific session)
            monitor = self.monitor or await self.initialize_monitor()
            await monitor.log_agent_decision(decision)
            
            # Simple analysis results
            analysis_results = {
                'decision_logged': True,
                'total_decisions_for_agent': len([
                    d for d in monitor.agent_decisions 
                    if d.agent_id == decision.agent_id
                ]),
                'average_confidence': sum([
                    d.confidence_score for d in monitor.agent_decisions 
                    if d.agent_id == decision.agent_id
                ]) / max(len([d for d in monitor.agent_decisions if d.agent_id == decision.agent_id]), 1)
            }
            
            return {
                'success': True,
                'decision_id': decision.decision_id,
                'analysis_results': analysis_results
            }
            
        except Exception as e:
            self.logger.error(f"Error logging decision: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def log_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log a workflow execution for monitoring."""
        try:
            self.validate_input_data(input_data, [
                'workflow_id', 'execution_id', 'workflow_name', 'steps'
            ])
            
            # Create workflow execution object
            workflow = WorkflowExecution(
                workflow_id=input_data['workflow_id'],
                workflow_name=input_data['workflow_name'],
                execution_id=input_data['execution_id'],
                status=WorkflowStatus.RUNNING,  # Default to running
                start_time=datetime.fromisoformat(input_data.get('logged_at', datetime.now(timezone.utc).isoformat())),
                steps=input_data['steps'],
                dependencies=input_data.get('dependencies', {}),
                parallel_branches=input_data.get('parallel_branches', []),
                input_parameters=input_data.get('input_parameters', {}),
                output_results=input_data.get('output_results', {}),
                metadata=input_data.get('metadata', {})
            )
            
            # Log to monitor
            monitor = self.monitor or await self.initialize_monitor()
            await monitor.log_workflow_execution(workflow)
            
            return {
                'success': True,
                'workflow_id': workflow.workflow_id,
                'execution_id': workflow.execution_id
            }
            
        except Exception as e:
            self.logger.error(f"Error logging workflow: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def log_metric(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log a performance metric for monitoring."""
        try:
            self.validate_input_data(input_data, [
                'metric_id', 'agent_id', 'metric_type', 'value'
            ])
            
            # Create performance metric object
            metric = PerformanceMetric(
                metric_id=input_data['metric_id'],
                agent_id=input_data['agent_id'],
                metric_type=self.convert_metric_type(input_data['metric_type']),
                value=input_data['value'],
                timestamp=datetime.fromisoformat(input_data.get('logged_at', datetime.now(timezone.utc).isoformat())),
                workflow_id=input_data.get('workflow_id'),
                decision_id=input_data.get('decision_id'),
                measurement_context=input_data.get('measurement_context', {}),
                measurement_confidence=input_data.get('measurement_confidence', 1.0),
                data_quality_score=input_data.get('data_quality_score', 1.0)
            )
            
            # Add metric to monitor's collection
            monitor = self.monitor or await self.initialize_monitor()
            monitor.performance_metrics.append(metric)
            
            # Keep only recent metrics (last 1000)
            if len(monitor.performance_metrics) > 1000:
                monitor.performance_metrics = monitor.performance_metrics[-1000:]
            
            return {
                'success': True,
                'metric_id': metric.metric_id
            }
            
        except Exception as e:
            self.logger.error(f"Error logging metric: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def get_dashboard_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        try:
            time_range = input_data.get('time_range', '24h')
            agent_filter = input_data.get('agent_filter')
            include_predictions = input_data.get('include_predictions', True)
            include_optimization = input_data.get('include_optimization', True)
            
            # Get monitor instance
            monitor = self.monitor or await self.initialize_monitor()
            
            # Get dashboard data
            dashboard_data = await monitor.get_performance_dashboard_data()
            
            # Apply filters if provided
            if agent_filter:
                # Filter performance summary
                if 'performance_summary' in dashboard_data:
                    dashboard_data['performance_summary'] = {
                        agent_id: data for agent_id, data in dashboard_data['performance_summary'].items()
                        if agent_id == agent_filter
                    }
                
                # Filter learning patterns
                if 'learning_patterns' in dashboard_data:
                    dashboard_data['learning_patterns'] = [
                        pattern for pattern in dashboard_data['learning_patterns']
                        if pattern.get('agent_id') == agent_filter
                    ]
            
            # Add predictions if requested
            if include_predictions:
                dashboard_data['predictions'] = await self._generate_performance_predictions(monitor)
            
            # Add optimization data if requested
            if include_optimization:
                optimization_data = await monitor.get_optimization_recommendations()
                dashboard_data['optimization_summary'] = {
                    'total_opportunities': len(optimization_data.get('optimization_opportunities', [])),
                    'total_bottlenecks': len(optimization_data.get('bottlenecks', [])),
                    'predicted_savings': optimization_data.get('predicted_improvements', {}).get('total_time_savings', 0)
                }
            
            return {
                'success': True,
                'dashboard_data': dashboard_data
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def _generate_performance_predictions(self, monitor: AutonomousWorkflowMonitor) -> Dict[str, Any]:
        """Generate simple performance predictions."""
        try:
            predictions = {}
            
            # Analyze recent trends for predictions
            current_time = datetime.now(timezone.utc)
            recent_window = timedelta(hours=6)
            
            # Get recent metrics
            recent_metrics = [
                m for m in monitor.performance_metrics
                if current_time - m.timestamp <= recent_window
            ]
            
            # Group by agent and metric type
            agent_metrics = {}
            for metric in recent_metrics:
                if metric.agent_id not in agent_metrics:
                    agent_metrics[metric.agent_id] = {}
                if metric.metric_type not in agent_metrics[metric.agent_id]:
                    agent_metrics[metric.agent_id][metric.metric_type] = []
                agent_metrics[metric.agent_id][metric.metric_type].append(metric.value)
            
            # Generate simple predictions
            for agent_id, metrics_by_type in agent_metrics.items():
                agent_predictions = {}
                
                for metric_type, values in metrics_by_type.items():
                    if len(values) >= 3:  # Need at least 3 points for trend
                        # Simple linear trend prediction
                        recent_avg = sum(values[-3:]) / 3
                        older_avg = sum(values[:-3]) / max(len(values[:-3]), 1) if len(values) > 3 else recent_avg
                        
                        trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
                        
                        agent_predictions[metric_type.value] = {
                            'trend': trend,
                            'confidence': min(0.8, len(values) / 10),  # Higher confidence with more data
                            'predicted_next_value': recent_avg,  # Simple prediction
                            'recommendation': self._get_metric_recommendation(metric_type, trend, recent_avg)
                        }
                
                if agent_predictions:
                    predictions[agent_id] = agent_predictions
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
            return {}
    
    def _get_metric_recommendation(self, metric_type: PerformanceMetricType, trend: str, current_value: float) -> str:
        """Get recommendation based on metric trend."""
        if metric_type == PerformanceMetricType.ACCURACY:
            if trend == "declining" or current_value < 0.8:
                return "Consider model retraining or parameter adjustment"
            else:
                return "Performance is satisfactory"
        elif metric_type == PerformanceMetricType.PROCESSING_TIME:
            if trend == "declining" and current_value > 5.0:  # Declining = getting worse (longer times)
                return "Investigate performance bottlenecks"
            else:
                return "Processing time is within acceptable range"
        else:
            return f"Monitor {metric_type.value} trend: {trend}"
    
    async def analyze_patterns(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze learning patterns in agent behavior."""
        try:
            time_range = input_data.get('time_range', '24h')
            agent_ids = input_data.get('agent_ids', [])
            pattern_types = input_data.get('pattern_types', [])
            minimum_confidence = input_data.get('minimum_confidence', 0.5)
            include_recommendations = input_data.get('include_recommendations', True)
            
            # Get monitor instance
            monitor = self.monitor or await self.initialize_monitor()
            
            # Get time window
            current_time = datetime.now(timezone.utc)
            time_delta_map = {
                '1h': timedelta(hours=1),
                '24h': timedelta(hours=24),
                '7d': timedelta(days=7),
                '30d': timedelta(days=30)
            }
            time_window = time_delta_map.get(time_range, timedelta(hours=24))
            
            # Filter metrics and decisions by time window
            cutoff_time = current_time - time_window
            
            recent_metrics = [
                m for m in monitor.performance_metrics
                if m.timestamp > cutoff_time
            ]
            
            recent_decisions = [
                d for d in monitor.agent_decisions
                if d.timestamp > cutoff_time
            ]
            
            # Apply agent filter if provided
            if agent_ids:
                recent_metrics = [m for m in recent_metrics if m.agent_id in agent_ids]
                recent_decisions = [d for d in recent_decisions if d.agent_id in agent_ids]
            
            # Analyze patterns
            detected_patterns = await monitor.pattern_analyzer.analyze_learning_patterns(
                recent_metrics, recent_decisions
            )
            
            # Filter by confidence and pattern types
            filtered_patterns = []
            for pattern in detected_patterns:
                if pattern.confidence >= minimum_confidence:
                    if not pattern_types or pattern.pattern_type.value in pattern_types:
                        filtered_patterns.append(pattern)
            
            # Create pattern summary
            pattern_summary = {}
            for pattern in filtered_patterns:
                pattern_type = pattern.pattern_type.value
                if pattern_type not in pattern_summary:
                    pattern_summary[pattern_type] = 0
                pattern_summary[pattern_type] += 1
            
            # Convert patterns to serializable format
            patterns_data = []
            for pattern in filtered_patterns:
                pattern_data = {
                    'pattern_id': pattern.pattern_id,
                    'pattern_type': pattern.pattern_type.value,
                    'agent_id': pattern.agent_id,
                    'metric_type': pattern.metric_type.value,
                    'confidence': pattern.confidence,
                    'statistical_significance': pattern.statistical_significance,
                    'trend_direction': pattern.trend_direction,
                    'trend_magnitude': pattern.trend_magnitude,
                    'start_time': pattern.start_time.isoformat(),
                    'end_time': pattern.end_time.isoformat(),
                    'insights': pattern.insights,
                    'recommendations': pattern.recommendations if include_recommendations else []
                }
                patterns_data.append(pattern_data)
            
            # Generate overall insights
            insights = []
            if pattern_summary:
                total_patterns = sum(pattern_summary.values())
                insights.append(f"Detected {total_patterns} learning patterns across {len(set(p.agent_id for p in filtered_patterns))} agents")
                
                most_common_pattern = max(pattern_summary, key=pattern_summary.get)
                insights.append(f"Most common pattern: {most_common_pattern} ({pattern_summary[most_common_pattern]} occurrences)")
            
            # Generate recommendations
            recommendations = []
            if include_recommendations:
                improvement_patterns = [p for p in filtered_patterns if p.pattern_type == LearningPatternType.IMPROVEMENT_TREND]
                degradation_patterns = [p for p in filtered_patterns if p.pattern_type == LearningPatternType.DEGRADATION]
                
                if improvement_patterns:
                    recommendations.append(f"Continue successful strategies for {len(improvement_patterns)} improving agents")
                
                if degradation_patterns:
                    recommendations.append(f"Urgent attention needed for {len(degradation_patterns)} declining agents")
                    for pattern in degradation_patterns[:3]:  # Top 3
                        recommendations.extend(pattern.recommendations[:2])  # Top 2 recommendations per pattern
            
            return {
                'success': True,
                'analysis_id': str(uuid.uuid4()),
                'patterns_detected': patterns_data,
                'pattern_summary': pattern_summary,
                'insights': insights,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def optimize_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get workflow optimization recommendations."""
        try:
            workflow_ids = input_data.get('workflow_ids', [])
            optimization_scope = input_data.get('optimization_scope', 'comprehensive')
            include_bottleneck_analysis = input_data.get('include_bottleneck_analysis', True)
            include_resource_optimization = input_data.get('include_resource_optimization', True)
            include_predictions = input_data.get('include_predictions', True)
            
            # Get monitor instance
            monitor = self.monitor or await self.initialize_monitor()
            
            # Filter workflows if specific IDs provided
            workflows_to_analyze = list(monitor.workflow_executions.values())
            if workflow_ids:
                workflows_to_analyze = [
                    w for w in workflows_to_analyze 
                    if w.workflow_id in workflow_ids
                ]
            
            if not workflows_to_analyze:
                return {
                    'success': True,
                    'optimization_id': str(uuid.uuid4()),
                    'bottlenecks': [],
                    'optimization_opportunities': [],
                    'predicted_improvements': {},
                    'implementation_recommendations': [],
                    'message': 'No workflows available for analysis'
                }
            
            # Get optimization recommendations
            optimization_results = await monitor.get_optimization_recommendations()
            
            return {
                'success': True,
                'optimization_id': str(uuid.uuid4()),
                'bottlenecks': optimization_results.get('bottlenecks', []),
                'optimization_opportunities': optimization_results.get('optimization_opportunities', []),
                'predicted_improvements': optimization_results.get('predicted_improvements', {}),
                'implementation_recommendations': optimization_results.get('implementation_recommendations', [])
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing workflow: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def generate_report(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive reports."""
        try:
            report_type = input_data.get('report_type', 'comprehensive')
            time_period = input_data.get('time_period', '24h')
            include_visualizations = input_data.get('include_visualizations', True)
            include_recommendations = input_data.get('include_recommendations', True)
            export_format = input_data.get('export_format', 'json')
            
            # Get monitor instance
            monitor = self.monitor or await self.initialize_monitor()
            
            report_id = str(uuid.uuid4())
            
            # Generate different types of reports
            if report_type in ['learning_insights', 'comprehensive']:
                report_data = await monitor.generate_learning_insights_report()
            elif report_type == 'performance_summary':
                report_data = await monitor.get_performance_dashboard_data()
            elif report_type == 'optimization_report':
                report_data = await monitor.get_optimization_recommendations()
            else:
                # Comprehensive report
                dashboard_data = await monitor.get_performance_dashboard_data()
                learning_insights = await monitor.generate_learning_insights_report()
                optimization_data = await monitor.get_optimization_recommendations()
                
                report_data = {
                    'report_type': 'comprehensive',
                    'generated_at': datetime.now(timezone.utc).isoformat(),
                    'time_period': time_period,
                    'dashboard_summary': dashboard_data,
                    'learning_insights': learning_insights,
                    'optimization_recommendations': optimization_data
                }
            
            # Add report metadata
            report_data['report_id'] = report_id
            report_data['export_format'] = export_format
            
            # Generate export URL (would be implemented based on file storage system)
            export_url = None
            if export_format in ['pdf', 'html']:
                # In a real implementation, this would generate and store the file
                export_url = f"/api/workflow-monitor/reports/{report_id}.{export_format}"
            
            return {
                'success': True,
                'report_id': report_id,
                'report_data': report_data,
                'export_url': export_url
            }
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def set_alert_thresholds(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Set custom alert thresholds."""
        try:
            thresholds = input_data.get('thresholds', {})
            agent_specific_thresholds = input_data.get('agent_specific_thresholds', {})
            notification_settings = input_data.get('notification_settings', {})
            
            # Get monitor instance
            monitor = self.monitor or await self.initialize_monitor()
            
            # Set global thresholds
            thresholds_updated = {}
            for metric_name, threshold_value in thresholds.items():
                try:
                    metric_type = self.convert_metric_type(metric_name)
                    monitor.set_alert_threshold(metric_type, threshold_value)
                    thresholds_updated[metric_name] = threshold_value
                except Exception as e:
                    self.logger.warning(f"Failed to set threshold for {metric_name}: {str(e)}")
            
            # Agent-specific thresholds would be implemented here
            # notification_settings would configure the alert system
            
            return {
                'success': True,
                'thresholds_updated': thresholds_updated
            }
            
        except Exception as e:
            self.logger.error(f"Error setting alert thresholds: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def export_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export monitoring data."""
        try:
            export_type = input_data.get('export_type', 'all')
            time_range = input_data.get('time_range', '24h')
            format = input_data.get('format', 'json')
            include_metadata = input_data.get('include_metadata', True)
            agent_filter = input_data.get('agent_filter', [])
            export_id = input_data.get('export_id', str(uuid.uuid4()))
            
            # Get monitor instance
            monitor = self.monitor or await self.initialize_monitor()
            
            # Prepare export data
            export_data = {
                'export_id': export_id,
                'export_type': export_type,
                'format': format,
                'exported_at': datetime.now(timezone.utc).isoformat(),
                'time_range': time_range
            }
            
            # Add data based on export type
            if export_type in ['decisions', 'all']:
                decisions_data = []
                for decision in monitor.agent_decisions:
                    if not agent_filter or decision.agent_id in agent_filter:
                        decision_data = {
                            'decision_id': decision.decision_id,
                            'agent_id': decision.agent_id,
                            'decision_type': decision.decision_type.value,
                            'confidence_score': decision.confidence_score,
                            'processing_time': decision.processing_time,
                            'timestamp': decision.timestamp.isoformat()
                        }
                        if include_metadata:
                            decision_data.update({
                                'input_data': decision.input_data,
                                'output_data': decision.output_data,
                                'context': decision.context,
                                'metadata': decision.metadata
                            })
                        decisions_data.append(decision_data)
                export_data['decisions'] = decisions_data
            
            if export_type in ['metrics', 'all']:
                metrics_data = []
                for metric in monitor.performance_metrics:
                    if not agent_filter or metric.agent_id in agent_filter:
                        metric_data = {
                            'metric_id': metric.metric_id,
                            'agent_id': metric.agent_id,
                            'metric_type': metric.metric_type.value,
                            'value': metric.value,
                            'timestamp': metric.timestamp.isoformat()
                        }
                        if include_metadata:
                            metric_data.update({
                                'measurement_context': metric.measurement_context,
                                'measurement_confidence': metric.measurement_confidence,
                                'data_quality_score': metric.data_quality_score
                            })
                        metrics_data.append(metric_data)
                export_data['metrics'] = metrics_data
            
            # Calculate export statistics
            record_count = sum([
                len(export_data.get('decisions', [])),
                len(export_data.get('metrics', [])),
                len(export_data.get('workflows', [])),
                len(export_data.get('patterns', []))
            ])
            
            # In a real implementation, this would save to file and provide download URL
            export_url = f"/api/workflow-monitor/exports/{export_id}.{format}"
            file_size = len(json.dumps(export_data))  # Approximate size
            expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
            
            return {
                'success': True,
                'export_id': export_id,
                'export_url': export_url,
                'file_size': file_size,
                'record_count': record_count,
                'expires_at': expires_at
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def get_status(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get monitoring status and configuration."""
        try:
            if input_data is None:
                input_data = {}
            
            # Get monitor instance if available
            monitor = self.monitor
            
            if monitor:
                system_health = monitor._calculate_system_health()
                configuration = {
                    'metrics_collection_interval': monitor.metrics_collection_interval,
                    'monitoring_active': monitor.monitoring_active,
                    'alert_thresholds': {
                        metric_type.value: threshold 
                        for metric_type, threshold in monitor.alert_thresholds.items()
                    }
                }
                statistics = {
                    'total_decisions': len(monitor.agent_decisions),
                    'total_metrics': len(monitor.performance_metrics),
                    'total_patterns': len(monitor.learning_patterns),
                    'total_workflows': len(monitor.workflow_executions),
                    'active_sessions': len(self.active_sessions)
                }
            else:
                system_health = {'status': 'not_initialized', 'score': 0.0}
                configuration = {}
                statistics = {'active_sessions': len(self.active_sessions)}
            
            return {
                'success': True,
                'system_health': system_health,
                'configuration': configuration,
                'statistics': statistics
            }
            
        except Exception as e:
            self.logger.error(f"Error getting status: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def health_check(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform health check."""
        try:
            # Test basic functionality
            test_monitor = AutonomousWorkflowMonitor()
            
            # Test decision creation
            test_decision = AgentDecision(
                decision_id="test_decision",
                agent_id="test_agent",
                workflow_id="test_workflow",
                decision_type=DecisionType.CLASSIFICATION,
                input_data={"test": True},
                output_data={"result": "test"},
                confidence_score=0.9,
                processing_time=0.1,
                timestamp=datetime.now(timezone.utc)
            )
            
            await test_monitor.log_agent_decision(test_decision)
            
            # Test dashboard data generation
            dashboard_data = await test_monitor.get_performance_dashboard_data()
            
            return {
                'success': True,
                'status': 'healthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'test_results': {
                    'decision_logging': True,
                    'dashboard_generation': True,
                    'monitor_creation': True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                'success': False,
                'status': 'unhealthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }


async def main():
    """Main execution function."""
    try:
        if len(sys.argv) < 2:
            print(json.dumps({
                'success': False,
                'error': 'No operation specified',
                'usage': 'python3 workflow_monitor_executor.py <operation>'
            }))
            sys.exit(1)
        
        operation = sys.argv[1].lower()
        
        # Read input data from stdin if available
        input_data = {}
        if not sys.stdin.isatty():
            try:
                input_data = json.load(sys.stdin)
            except json.JSONDecodeError as e:
                print(json.dumps({
                    'success': False,
                    'error': f'Invalid JSON input: {str(e)}'
                }))
                sys.exit(1)
        
        # Create executor
        executor = WorkflowMonitorExecutor()
        
        # Route to appropriate operation
        operations = {
            'start_monitoring': executor.start_monitoring,
            'stop_monitoring': executor.stop_monitoring,
            'log_decision': executor.log_decision,
            'log_workflow': executor.log_workflow,
            'log_metric': executor.log_metric,
            'get_dashboard_data': executor.get_dashboard_data,
            'analyze_patterns': executor.analyze_patterns,
            'optimize_workflow': executor.optimize_workflow,
            'generate_report': executor.generate_report,
            'set_alert_thresholds': executor.set_alert_thresholds,
            'export_data': executor.export_data,
            'get_status': executor.get_status,
            'health_check': executor.health_check
        }
        
        if operation not in operations:
            print(json.dumps({
                'success': False,
                'error': f'Unknown operation: {operation}',
                'available_operations': list(operations.keys())
            }))
            sys.exit(1)
        
        # Execute operation
        result = await operations[operation](input_data)
        
        # Output result
        print(json.dumps(result, indent=2))
        
        # Exit with appropriate code
        sys.exit(0 if result.get('success', False) else 1)
    
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print(json.dumps({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == '__main__':
    # Run async main
    asyncio.run(main())
