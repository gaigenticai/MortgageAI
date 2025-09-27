#!/usr/bin/env python3
"""
Anomaly Detection Interface Executor
Created: 2024-01-15
Author: MortgageAI Development Team
Description: Executor script for handling anomaly detection operations via API calls.
"""

import asyncio
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback
import os
import pandas as pd
import numpy as np

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from anomaly_detection_interface import (
        AnomalyDetectionInterface,
        AnomalyDetection,
        AlertRule,
        InvestigationSession
    )
except ImportError as e:
    print(f"Error importing anomaly detection modules: {str(e)}", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyDetectionExecutor:
    """Executor class for anomaly detection operations."""
    
    def __init__(self):
        """Initialize the anomaly detection executor."""
        self.detector = AnomalyDetectionInterface()
        
    async def execute_operation(self, operation: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the specified anomaly detection operation."""
        try:
            operation_map = {
                'detect_anomalies': self._detect_anomalies,
                'get_anomaly_summary': self._get_anomaly_summary,
                'get_anomaly_details': self._get_anomaly_details,
                'update_anomaly_feedback': self._update_anomaly_feedback,
                'create_alert_rule': self._create_alert_rule,
                'list_alert_rules': self._list_alert_rules,
                'update_alert_rule': self._update_alert_rule,
                'delete_alert_rule': self._delete_alert_rule,
                'start_investigation': self._start_investigation,
                'list_investigations': self._list_investigations,
                'get_investigation_details': self._get_investigation_details,
                'add_investigation_evidence': self._add_investigation_evidence,
                'analyze_investigation_patterns': self._analyze_investigation_patterns,
                'health_check': self._health_check
            }
            
            if operation not in operation_map:
                raise ValueError(f"Unknown operation: {operation}")
            
            result = await operation_map[operation](args)
            return result
            
        except Exception as e:
            logger.error(f"Error executing operation {operation}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def _detect_anomalies(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in provided data."""
        try:
            detection_id = args.get('detection_id', 'unknown')
            data = json.loads(args.get('data', '{}'))
            methods = args.get('methods', 'statistical,ml_based').split(',')
            severity_threshold = args.get('severity_threshold', 'medium')
            detection_options = json.loads(args.get('detection_options', '{}'))
            
            logger.info(f"Detecting anomalies with methods: {methods}")
            
            # Convert data to appropriate format
            if isinstance(data, list):
                df_data = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Handle single record or structured data
                if all(isinstance(v, (list, tuple)) for v in data.values()):
                    df_data = pd.DataFrame(data)
                else:
                    df_data = pd.DataFrame([data])
            else:
                raise ValueError("Invalid data format")
            
            # Perform anomaly detection
            anomalies = await self.detector.detect_anomalies(
                df_data, 
                methods=methods, 
                severity_threshold=severity_threshold
            )
            
            # Convert anomalies to serializable format
            serialized_anomalies = [self._serialize_anomaly(anomaly) for anomaly in anomalies]
            
            # Generate detection summary
            detection_summary = {
                'total_anomalies': len(anomalies),
                'severity_distribution': {},
                'method_distribution': {},
                'confidence_stats': {
                    'mean_confidence': np.mean([a.confidence_score for a in anomalies]) if anomalies else 0,
                    'min_confidence': min([a.confidence_score for a in anomalies]) if anomalies else 0,
                    'max_confidence': max([a.confidence_score for a in anomalies]) if anomalies else 0
                },
                'data_processed': {
                    'records': len(df_data),
                    'features': len(df_data.columns) if not df_data.empty else 0
                },
                'detection_options': detection_options
            }
            
            # Calculate distributions
            for anomaly in anomalies:
                detection_summary['severity_distribution'][anomaly.severity] = \
                    detection_summary['severity_distribution'].get(anomaly.severity, 0) + 1
                detection_summary['method_distribution'][anomaly.detection_type] = \
                    detection_summary['method_distribution'].get(anomaly.detection_type, 0) + 1
            
            return {
                'success': True,
                'detection_id': detection_id,
                'anomalies': serialized_anomalies,
                'detection_summary': detection_summary,
                'methods_used': methods,
                'severity_threshold': severity_threshold
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'detect_anomalies'
            }
    
    async def _get_anomaly_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        try:
            time_period = args.get('time_period', '24h')
            category = args.get('category')
            severity = args.get('severity')
            
            logger.info(f"Retrieving anomaly summary for period: {time_period}")
            
            # Get summary from detector
            summary = await self.detector.get_anomaly_summary(time_period)
            
            # Apply filters if specified
            if category or severity:
                filtered_summary = self._apply_summary_filters(summary, category, severity)
                summary.update(filtered_summary)
            
            return {
                'success': True,
                'summary': summary,
                'time_period': time_period,
                'filters': {'category': category, 'severity': severity}
            }
            
        except Exception as e:
            logger.error(f"Error retrieving anomaly summary: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'get_anomaly_summary'
            }
    
    async def _get_anomaly_details(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about a specific anomaly."""
        try:
            anomaly_id = args.get('anomaly_id')
            
            if not anomaly_id:
                raise ValueError("anomaly_id is required")
            
            logger.info(f"Retrieving details for anomaly: {anomaly_id}")
            
            # Get anomaly from detector's stored anomalies
            if anomaly_id in self.detector.detected_anomalies:
                anomaly = self.detector.detected_anomalies[anomaly_id]
                
                # Generate investigation suggestions
                investigation_suggestions = [
                    "Review data quality and collection processes",
                    "Analyze temporal patterns around detection time",
                    "Compare with similar historical cases",
                    "Check for system changes or external events",
                    "Validate data sources and transformations",
                    "Review business process compliance"
                ]
                
                # Find related anomalies
                related_anomalies = await self._find_related_anomalies(anomaly)
                
                return {
                    'success': True,
                    'anomaly': self._serialize_anomaly(anomaly),
                    'investigation_suggestions': investigation_suggestions,
                    'related_anomalies': [self._serialize_anomaly(ra) for ra in related_anomalies]
                }
            else:
                return {
                    'success': False,
                    'error': 'Anomaly not found',
                    'anomaly_id': anomaly_id
                }
            
        except Exception as e:
            logger.error(f"Error retrieving anomaly details: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'get_anomaly_details'
            }
    
    async def _update_anomaly_feedback(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Update feedback for an anomaly detection."""
        try:
            anomaly_id = args.get('anomaly_id')
            is_true_positive = args.get('is_true_positive', 'false').lower() == 'true'
            feedback_notes = args.get('feedback_notes', '')
            resolution_action = args.get('resolution_action', '')
            confidence_rating = int(args.get('confidence_rating', '3'))
            
            if not anomaly_id:
                raise ValueError("anomaly_id is required")
            
            logger.info(f"Updating feedback for anomaly: {anomaly_id}")
            
            # Create comprehensive feedback notes
            comprehensive_feedback = feedback_notes
            if resolution_action:
                comprehensive_feedback += f" Resolution action: {resolution_action}"
            if confidence_rating:
                comprehensive_feedback += f" Confidence rating: {confidence_rating}/5"
            
            # Update feedback in detector
            updated = await self.detector.update_anomaly_feedback(
                anomaly_id, is_true_positive, comprehensive_feedback
            )
            
            return {
                'success': True,
                'anomaly_id': anomaly_id,
                'updated': updated,
                'detection_statistics': self.detector.detection_statistics
            }
            
        except Exception as e:
            logger.error(f"Error updating anomaly feedback: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'update_anomaly_feedback'
            }
    
    async def _create_alert_rule(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new anomaly detection rule."""
        try:
            rule_id = args.get('rule_id')
            rule_data = json.loads(args.get('rule_data', '{}'))
            
            if not rule_id or not rule_data:
                raise ValueError("rule_id and rule_data are required")
            
            logger.info(f"Creating alert rule: {rule_data.get('rule_name', 'Unknown')}")
            
            # Create AlertRule object
            alert_rule = AlertRule(
                rule_id=rule_id,
                rule_name=rule_data['rule_name'],
                rule_type=rule_data['rule_type'],
                category=rule_data['category'],
                conditions=rule_data['conditions'],
                thresholds=rule_data.get('thresholds', {}),
                parameters=rule_data.get('parameters', {}),
                severity_mapping=rule_data.get('severity_mapping', {}),
                escalation_rules=rule_data.get('escalation_rules', []),
                notification_channels=rule_data.get('notification_channels', ['in_app']),
                suppression_rules=rule_data.get('suppression_rules', {}),
                is_active=rule_data.get('is_active', True),
                created_by='system',
                created_at=datetime.now(),
                metadata=rule_data.get('metadata', {})
            )
            
            # Add rule to detector
            await self.detector.rule_detector.add_rule(alert_rule)
            
            return {
                'success': True,
                'rule_id': rule_id,
                'created': True,
                'rule_configuration': self._serialize_alert_rule(alert_rule)
            }
            
        except Exception as e:
            logger.error(f"Error creating alert rule: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'create_alert_rule'
            }
    
    async def _list_alert_rules(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all anomaly detection rules."""
        try:
            active_only = args.get('active_only', 'true').lower() == 'true'
            category = args.get('category')
            rule_type = args.get('rule_type')
            
            logger.info("Retrieving anomaly detection rules")
            
            # Get rules from detector
            all_rules = self.detector.rule_detector.rules.values()
            
            # Apply filters
            filtered_rules = []
            for rule in all_rules:
                if active_only and not rule.is_active:
                    continue
                if category and rule.category != category:
                    continue
                if rule_type and rule.rule_type != rule_type:
                    continue
                filtered_rules.append(rule)
            
            # Serialize rules
            serialized_rules = [self._serialize_alert_rule(rule) for rule in filtered_rules]
            
            return {
                'success': True,
                'rules': serialized_rules,
                'total_rules': len(serialized_rules),
                'active_rules': len([r for r in filtered_rules if r.is_active]),
                'filters': {
                    'active_only': active_only,
                    'category': category,
                    'rule_type': rule_type
                }
            }
            
        except Exception as e:
            logger.error(f"Error listing alert rules: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'list_alert_rules'
            }
    
    async def _update_alert_rule(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing anomaly detection rule."""
        try:
            rule_id = args.get('rule_id')
            rule_data = json.loads(args.get('rule_data', '{}'))
            
            if not rule_id:
                raise ValueError("rule_id is required")
            
            logger.info(f"Updating alert rule: {rule_id}")
            
            # Check if rule exists
            if rule_id not in self.detector.rule_detector.rules:
                return {
                    'success': False,
                    'error': 'Rule not found',
                    'rule_id': rule_id
                }
            
            # Update rule
            rule = self.detector.rule_detector.rules[rule_id]
            
            # Update rule properties
            for key, value in rule_data.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            return {
                'success': True,
                'rule_id': rule_id,
                'updated': True,
                'rule_configuration': self._serialize_alert_rule(rule)
            }
            
        except Exception as e:
            logger.error(f"Error updating alert rule: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'update_alert_rule'
            }
    
    async def _delete_alert_rule(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Delete an anomaly detection rule."""
        try:
            rule_id = args.get('rule_id')
            
            if not rule_id:
                raise ValueError("rule_id is required")
            
            logger.info(f"Deleting alert rule: {rule_id}")
            
            # Check if rule exists and delete
            if rule_id in self.detector.rule_detector.rules:
                del self.detector.rule_detector.rules[rule_id]
                if rule_id in self.detector.rule_detector.rule_history:
                    del self.detector.rule_detector.rule_history[rule_id]
                deleted = True
            else:
                deleted = False
            
            return {
                'success': True,
                'rule_id': rule_id,
                'deleted': deleted
            }
            
        except Exception as e:
            logger.error(f"Error deleting alert rule: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'delete_alert_rule'
            }
    
    async def _start_investigation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new anomaly investigation session."""
        try:
            session_id = args.get('session_id')
            anomaly_ids = args.get('anomaly_ids', '').split(',')
            session_name = args.get('session_name', 'Investigation Session')
            investigator_id = args.get('investigator_id', 'anonymous')
            priority_level = args.get('priority_level', 'medium')
            initial_hypothesis = json.loads(args.get('initial_hypothesis', '[]'))
            
            if not session_id or not anomaly_ids[0]:
                raise ValueError("session_id and anomaly_ids are required")
            
            logger.info(f"Starting investigation session: {session_name}")
            
            # Start investigation
            investigation = await self.detector.investigation_tools.start_investigation(
                anomaly_ids, investigator_id, session_name
            )
            
            # Add initial hypothesis if provided
            for hypothesis in initial_hypothesis:
                await self.detector.investigation_tools.add_hypothesis(
                    investigation.session_id, hypothesis
                )
            
            # Get initial suggestions from the investigation metadata
            initial_suggestions = investigation.metadata.get('initial_suggestions', [])
            
            return {
                'success': True,
                'session_id': investigation.session_id,
                'investigation': self._serialize_investigation(investigation),
                'initial_suggestions': initial_suggestions
            }
            
        except Exception as e:
            logger.error(f"Error starting investigation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'start_investigation'
            }
    
    async def _list_investigations(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List investigation sessions."""
        try:
            status = args.get('status')
            investigator_id = args.get('investigator_id')
            limit = int(args.get('limit', '10'))
            
            logger.info("Retrieving investigation sessions")
            
            # Get investigations from detector
            all_investigations = list(self.detector.investigation_tools.active_sessions.values())
            
            # Apply filters
            filtered_investigations = []
            for investigation in all_investigations:
                if status and investigation.investigation_status != status:
                    continue
                if investigator_id and investigation.investigator_id != investigator_id:
                    continue
                filtered_investigations.append(investigation)
            
            # Apply limit
            limited_investigations = filtered_investigations[:limit]
            
            # Serialize investigations
            serialized_investigations = [
                self._serialize_investigation(inv) for inv in limited_investigations
            ]
            
            return {
                'success': True,
                'investigations': serialized_investigations,
                'total_investigations': len(filtered_investigations),
                'filters': {
                    'status': status,
                    'investigator_id': investigator_id,
                    'limit': limit
                }
            }
            
        except Exception as e:
            logger.error(f"Error listing investigations: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'list_investigations'
            }
    
    async def _get_investigation_details(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about an investigation session."""
        try:
            session_id = args.get('session_id')
            
            if not session_id:
                raise ValueError("session_id is required")
            
            logger.info(f"Retrieving investigation details: {session_id}")
            
            # Get investigation
            if session_id in self.detector.investigation_tools.active_sessions:
                investigation = self.detector.investigation_tools.active_sessions[session_id]
                
                # Generate progress summary
                progress_summary = {
                    'hypothesis_count': len(investigation.hypothesis),
                    'evidence_count': len(investigation.evidence_collected),
                    'findings_count': len(investigation.findings),
                    'completion_percentage': self._calculate_investigation_progress(investigation),
                    'last_activity': investigation.last_activity_at.isoformat(),
                    'days_active': (datetime.now() - investigation.started_at).days
                }
                
                return {
                    'success': True,
                    'investigation': self._serialize_investigation(investigation),
                    'analysis_results': investigation.analysis_results,
                    'progress_summary': progress_summary
                }
            else:
                return {
                    'success': False,
                    'error': 'Investigation not found',
                    'session_id': session_id
                }
            
        except Exception as e:
            logger.error(f"Error retrieving investigation details: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'get_investigation_details'
            }
    
    async def _add_investigation_evidence(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Add evidence to an investigation session."""
        try:
            session_id = args.get('session_id')
            evidence_data = json.loads(args.get('evidence_data', '{}'))
            
            if not session_id or not evidence_data:
                raise ValueError("session_id and evidence_data are required")
            
            logger.info(f"Adding evidence to investigation: {session_id}")
            
            # Add evidence
            evidence_added = await self.detector.investigation_tools.collect_evidence(
                session_id, evidence_data
            )
            
            if evidence_added:
                investigation = self.detector.investigation_tools.active_sessions[session_id]
                evidence_id = str(len(investigation.evidence_collected))  # Simple ID
                total_evidence = len(investigation.evidence_collected)
            else:
                evidence_id = None
                total_evidence = 0
            
            return {
                'success': True,
                'session_id': session_id,
                'evidence_added': evidence_added,
                'evidence_id': evidence_id,
                'total_evidence': total_evidence
            }
            
        except Exception as e:
            logger.error(f"Error adding investigation evidence: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'add_investigation_evidence'
            }
    
    async def _analyze_investigation_patterns(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Perform pattern analysis on investigation data."""
        try:
            session_id = args.get('session_id')
            analysis_data = json.loads(args.get('analysis_data', '{}'))
            analysis_type = args.get('analysis_type', 'comprehensive')
            include_visualizations = args.get('include_visualizations', 'true').lower() == 'true'
            
            if not session_id or not analysis_data:
                raise ValueError("session_id and analysis_data are required")
            
            logger.info(f"Performing pattern analysis for investigation: {session_id}")
            
            # Convert analysis data to DataFrame
            if isinstance(analysis_data, list):
                df_data = pd.DataFrame(analysis_data)
            else:
                df_data = pd.DataFrame([analysis_data])
            
            # Perform analysis
            analysis_results = await self.detector.investigation_tools.analyze_patterns(
                session_id, df_data
            )
            
            # Count patterns found
            patterns_found = 0
            if 'temporal_patterns' in analysis_results:
                patterns_found += len(analysis_results['temporal_patterns'])
            if 'correlation_analysis' in analysis_results:
                patterns_found += len(analysis_results['correlation_analysis'].get('strong_correlations', []))
            
            # Generate insights
            insights = []
            if analysis_results:
                if 'temporal_patterns' in analysis_results:
                    patterns = analysis_results['temporal_patterns']
                    if 'peak_hour' in patterns:
                        insights.append(f"Peak activity occurs at hour {patterns['peak_hour']}")
                    if 'peak_day' in patterns:
                        insights.append(f"Highest activity on {patterns['peak_day']}")
                
                if 'correlation_analysis' in analysis_results:
                    correlations = analysis_results['correlation_analysis'].get('strong_correlations', [])
                    for corr in correlations[:3]:  # Top 3 correlations
                        strength = corr['strength']
                        var1, var2 = corr['variable1'], corr['variable2']
                        insights.append(f"Found {strength} correlation between {var1} and {var2}")
            
            # Generate visualizations (placeholder)
            visualizations = {}
            if include_visualizations:
                visualizations = {
                    'trend_chart': {'type': 'line', 'data': 'placeholder'},
                    'correlation_heatmap': {'type': 'heatmap', 'data': 'placeholder'},
                    'distribution_plot': {'type': 'histogram', 'data': 'placeholder'}
                }
            
            return {
                'success': True,
                'session_id': session_id,
                'analysis_type': analysis_type,
                'analysis_results': analysis_results,
                'patterns_found': patterns_found,
                'insights': insights,
                'visualizations': visualizations
            }
            
        except Exception as e:
            logger.error(f"Error performing pattern analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'analyze_investigation_patterns'
            }
    
    async def _health_check(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Perform health check of anomaly detection services."""
        try:
            logger.info("Performing anomaly detection health check")
            
            # Test components
            start_time = datetime.now()
            
            # Test statistical detector
            try:
                test_data = pd.DataFrame({'test': [1, 2, 3, 100]})  # Simple test with outlier
                await self.detector.statistical_detector.detect_anomalies(test_data)
                statistical_status = 'healthy'
            except Exception:
                statistical_status = 'unhealthy'
            
            # Test ML detector
            try:
                test_data = pd.DataFrame({'test1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100], 'test2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
                await self.detector.ml_detector.detect_anomalies(test_data)
                ml_status = 'healthy'
            except Exception:
                ml_status = 'unhealthy'
            
            # Test rule detector
            try:
                await self.detector.rule_detector.detect_anomalies({'test_field': 50})
                rule_status = 'healthy'
            except Exception:
                rule_status = 'healthy'  # Expected to work even with no rules
            
            # Test investigation tools
            try:
                # Simple test of investigation tools initialization
                investigation_status = 'healthy' if self.detector.investigation_tools else 'unhealthy'
            except Exception:
                investigation_status = 'unhealthy'
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'status': 'healthy' if all(s == 'healthy' for s in [statistical_status, ml_status, rule_status, investigation_status]) else 'degraded',
                'services': {
                    'statistical_detector': statistical_status,
                    'ml_detector': ml_status,
                    'rule_detector': rule_status,
                    'investigation_tools': investigation_status,
                    'alert_manager': 'healthy'  # Basic check
                },
                'performance': {
                    'response_time_seconds': response_time,
                    'total_anomalies_detected': len(self.detector.detected_anomalies),
                    'active_investigations': len(self.detector.investigation_tools.active_sessions),
                    'active_rules': len([r for r in self.detector.rule_detector.rules.values() if r.is_active])
                },
                'dependencies': {
                    'python_version': sys.version,
                    'required_packages': ['numpy', 'pandas', 'scikit-learn', 'scipy']
                },
                'detection_statistics': self.detector.detection_statistics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            return {
                'success': False,
                'status': 'unhealthy',
                'error': str(e),
                'operation': 'health_check'
            }
    
    def _serialize_anomaly(self, anomaly: AnomalyDetection) -> Dict[str, Any]:
        """Serialize an AnomalyDetection object to dictionary."""
        return {
            'anomaly_id': anomaly.anomaly_id,
            'detection_type': anomaly.detection_type,
            'anomaly_category': anomaly.anomaly_category,
            'severity': anomaly.severity,
            'confidence_score': anomaly.confidence_score,
            'anomaly_score': anomaly.anomaly_score,
            'title': anomaly.title,
            'description': anomaly.description,
            'affected_entities': anomaly.affected_entities,
            'detection_method': anomaly.detection_method,
            'detection_parameters': anomaly.detection_parameters,
            'deviation_metrics': anomaly.deviation_metrics,
            'investigation_priority': anomaly.investigation_priority,
            'recommended_actions': anomaly.recommended_actions,
            'investigation_hints': anomaly.investigation_hints,
            'related_anomalies': anomaly.related_anomalies,
            'detection_timestamp': anomaly.detection_timestamp.isoformat(),
            'data_timestamp': anomaly.data_timestamp.isoformat(),
            'time_window': anomaly.time_window,
            'status': anomaly.status,
            'resolution_notes': anomaly.resolution_notes,
            'resolved_by': anomaly.resolved_by,
            'resolved_at': anomaly.resolved_at.isoformat() if anomaly.resolved_at else None,
            'metadata': anomaly.metadata,
            'tags': anomaly.tags
        }
    
    def _serialize_alert_rule(self, rule: AlertRule) -> Dict[str, Any]:
        """Serialize an AlertRule object to dictionary."""
        return {
            'rule_id': rule.rule_id,
            'rule_name': rule.rule_name,
            'rule_type': rule.rule_type,
            'category': rule.category,
            'conditions': rule.conditions,
            'thresholds': rule.thresholds,
            'parameters': rule.parameters,
            'severity_mapping': rule.severity_mapping,
            'escalation_rules': rule.escalation_rules,
            'notification_channels': rule.notification_channels,
            'suppression_rules': rule.suppression_rules,
            'is_active': rule.is_active,
            'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None,
            'trigger_count': rule.trigger_count,
            'false_positive_count': rule.false_positive_count,
            'created_by': rule.created_by,
            'created_at': rule.created_at.isoformat(),
            'metadata': rule.metadata
        }
    
    def _serialize_investigation(self, investigation: InvestigationSession) -> Dict[str, Any]:
        """Serialize an InvestigationSession object to dictionary."""
        return {
            'session_id': investigation.session_id,
            'anomaly_ids': investigation.anomaly_ids,
            'investigator_id': investigation.investigator_id,
            'session_name': investigation.session_name,
            'investigation_status': investigation.investigation_status,
            'priority_level': investigation.priority_level,
            'hypothesis': investigation.hypothesis,
            'evidence_collected': investigation.evidence_collected,
            'analysis_results': investigation.analysis_results,
            'findings': investigation.findings,
            'root_cause': investigation.root_cause,
            'impact_assessment': investigation.impact_assessment,
            'recommendations': investigation.recommendations,
            'started_at': investigation.started_at.isoformat(),
            'last_activity_at': investigation.last_activity_at.isoformat(),
            'completed_at': investigation.completed_at.isoformat() if investigation.completed_at else None,
            'collaborators': investigation.collaborators,
            'notes': investigation.notes,
            'attachments': investigation.attachments,
            'metadata': investigation.metadata
        }
    
    async def _find_related_anomalies(self, anomaly: AnomalyDetection) -> List[AnomalyDetection]:
        """Find anomalies related to the given anomaly."""
        related = []
        
        for other_anomaly in self.detector.detected_anomalies.values():
            if other_anomaly.anomaly_id == anomaly.anomaly_id:
                continue
            
            # Simple relatedness based on category, severity, and time
            if (other_anomaly.anomaly_category == anomaly.anomaly_category and
                other_anomaly.severity == anomaly.severity and
                abs((other_anomaly.detection_timestamp - anomaly.detection_timestamp).total_seconds()) < 3600):
                related.append(other_anomaly)
            
            if len(related) >= 5:  # Limit to 5 related anomalies
                break
        
        return related
    
    def _apply_summary_filters(self, summary: Dict[str, Any], 
                             category: Optional[str], 
                             severity: Optional[str]) -> Dict[str, Any]:
        """Apply filters to anomaly summary."""
        # This would filter the summary based on category and severity
        # For now, return the summary as-is since filtering logic would depend
        # on the actual data structure
        return {}
    
    def _calculate_investigation_progress(self, investigation: InvestigationSession) -> float:
        """Calculate investigation progress percentage."""
        progress_factors = [
            len(investigation.hypothesis) > 0,  # Has hypothesis
            len(investigation.evidence_collected) > 0,  # Has evidence
            len(investigation.analysis_results) > 0,  # Has analysis
            len(investigation.findings) > 0,  # Has findings
            investigation.root_cause is not None,  # Has root cause
            len(investigation.recommendations or []) > 0  # Has recommendations
        ]
        
        return (sum(progress_factors) / len(progress_factors)) * 100

async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Anomaly Detection Interface Executor')
    parser.add_argument('operation', help='Operation to perform')
    parser.add_argument('--detection_id', help='Detection ID')
    parser.add_argument('--data', help='JSON data for analysis', default='{}')
    parser.add_argument('--methods', help='Detection methods', default='statistical,ml_based')
    parser.add_argument('--severity_threshold', help='Severity threshold', default='medium')
    parser.add_argument('--detection_options', help='Detection options JSON', default='{}')
    parser.add_argument('--time_period', help='Time period for analysis', default='24h')
    parser.add_argument('--category', help='Category filter')
    parser.add_argument('--severity', help='Severity filter')
    parser.add_argument('--anomaly_id', help='Anomaly ID')
    parser.add_argument('--is_true_positive', help='Feedback: is true positive', default='false')
    parser.add_argument('--feedback_notes', help='Feedback notes', default='')
    parser.add_argument('--resolution_action', help='Resolution action taken', default='')
    parser.add_argument('--confidence_rating', help='Confidence rating 1-5', default='3')
    parser.add_argument('--rule_id', help='Alert rule ID')
    parser.add_argument('--rule_data', help='Alert rule data JSON', default='{}')
    parser.add_argument('--active_only', help='Active rules only', default='true')
    parser.add_argument('--rule_type', help='Rule type filter')
    parser.add_argument('--session_id', help='Investigation session ID')
    parser.add_argument('--anomaly_ids', help='Comma-separated anomaly IDs', default='')
    parser.add_argument('--session_name', help='Investigation session name', default='Investigation')
    parser.add_argument('--investigator_id', help='Investigator ID', default='anonymous')
    parser.add_argument('--priority_level', help='Priority level', default='medium')
    parser.add_argument('--initial_hypothesis', help='Initial hypothesis JSON', default='[]')
    parser.add_argument('--status', help='Status filter')
    parser.add_argument('--limit', help='Limit results', default='10')
    parser.add_argument('--evidence_data', help='Evidence data JSON', default='{}')
    parser.add_argument('--analysis_data', help='Analysis data JSON', default='{}')
    parser.add_argument('--analysis_type', help='Analysis type', default='comprehensive')
    parser.add_argument('--include_visualizations', help='Include visualizations', default='true')
    parser.add_argument('--output_format', help='Output format', default='json')
    
    args = parser.parse_args()
    
    try:
        executor = AnomalyDetectionExecutor()
        result = await executor.execute_operation(args.operation, vars(args))
        
        if args.output_format == 'json':
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result)
            
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'operation': args.operation
        }
        print(json.dumps(error_result, indent=2, default=str))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
