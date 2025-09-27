#!/usr/bin/env python3
"""
Anomaly Detection Interface - Real-time Pattern Recognition & Investigation Tools
Created: 2024-01-15
Author: MortgageAI Development Team
Description: Comprehensive anomaly detection system for Dutch mortgage advisory platform
            with real-time pattern recognition, alert management, and investigation tools.
"""

import asyncio
import logging
import json
import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from decimal import Decimal
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AnomalyDetection:
    """Represents a detected anomaly with comprehensive details."""
    anomaly_id: str
    detection_type: str  # 'statistical', 'ml_based', 'rule_based', 'hybrid'
    anomaly_category: str  # 'application', 'behavior', 'transaction', 'compliance', 'data_quality'
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence_score: float  # 0.0 to 1.0
    anomaly_score: float  # Normalized anomaly score
    
    # Anomaly description and context
    title: str
    description: str
    affected_entities: Dict[str, Any]  # Applications, users, processes affected
    
    # Detection details
    detection_method: str
    detection_parameters: Dict[str, Any]
    reference_patterns: Dict[str, Any]
    deviation_metrics: Dict[str, Any]
    
    # Data and evidence
    anomalous_data: Dict[str, Any]
    historical_baseline: Dict[str, Any]
    statistical_measures: Dict[str, Any]
    
    # Investigation support
    investigation_priority: str  # 'low', 'medium', 'high', 'urgent'
    recommended_actions: List[str]
    investigation_hints: List[str]
    related_anomalies: List[str]  # IDs of related anomalies
    
    # Temporal information
    detection_timestamp: datetime
    data_timestamp: datetime
    time_window: str  # '1h', '1d', '1w', etc.
    
    # Status and resolution
    status: str  # 'new', 'investigating', 'resolved', 'false_positive', 'suppressed'
    resolution_notes: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    tags: List[str] = None

@dataclass
class AlertRule:
    """Represents an alert rule configuration."""
    rule_id: str
    rule_name: str
    rule_type: str  # 'threshold', 'statistical', 'pattern', 'ml_based'
    category: str  # 'application', 'behavior', 'compliance', 'performance'
    
    # Rule configuration
    conditions: Dict[str, Any]
    thresholds: Dict[str, Any]
    parameters: Dict[str, Any]
    
    # Alert settings
    severity_mapping: Dict[str, str]
    escalation_rules: List[Dict[str, Any]]
    notification_channels: List[str]
    suppression_rules: Dict[str, Any]
    
    # Rule status
    is_active: bool
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    false_positive_count: int = 0
    
    # Metadata
    created_by: str
    created_at: datetime
    metadata: Dict[str, Any] = None

@dataclass
class InvestigationSession:
    """Represents an anomaly investigation session."""
    session_id: str
    anomaly_ids: List[str]
    investigator_id: str
    session_name: str
    
    # Investigation details
    investigation_status: str  # 'active', 'paused', 'completed', 'cancelled'
    priority_level: str  # 'low', 'medium', 'high', 'urgent'
    
    # Investigation data
    hypothesis: List[str]
    evidence_collected: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    
    # Findings and conclusions
    findings: List[str]
    root_cause: Optional[str] = None
    impact_assessment: Dict[str, Any] = None
    recommendations: List[str] = None
    
    # Temporal tracking
    started_at: datetime
    last_activity_at: datetime
    completed_at: Optional[datetime] = None
    
    # Collaboration
    collaborators: List[str] = None
    notes: List[Dict[str, Any]] = None
    attachments: List[Dict[str, Any]] = None
    
    # Metadata
    metadata: Dict[str, Any] = None

class StatisticalAnomalyDetector:
    """Statistical methods for anomaly detection."""
    
    def __init__(self):
        """Initialize the statistical anomaly detector."""
        self.methods = {
            'z_score': self._z_score_detection,
            'modified_z_score': self._modified_z_score_detection,
            'iqr': self._iqr_detection,
            'grubbs_test': self._grubbs_test,
            'dixon_test': self._dixon_test,
            'mahalanobis': self._mahalanobis_detection
        }
        
    async def detect_anomalies(self, data: pd.DataFrame, 
                             method: str = 'z_score',
                             threshold: float = 3.0,
                             target_columns: List[str] = None) -> List[Dict[str, Any]]:
        """Detect anomalies using statistical methods."""
        try:
            if method not in self.methods:
                raise ValueError(f"Unknown method: {method}")
            
            if target_columns is None:
                target_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            anomalies = []
            detection_method = self.methods[method]
            
            for column in target_columns:
                if column in data.columns:
                    column_anomalies = await detection_method(
                        data[column].dropna(), threshold, column
                    )
                    anomalies.extend(column_anomalies)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {str(e)}")
            raise
    
    async def _z_score_detection(self, series: pd.Series, 
                               threshold: float,
                               column_name: str) -> List[Dict[str, Any]]:
        """Z-score based anomaly detection."""
        mean_val = series.mean()
        std_val = series.std()
        z_scores = np.abs((series - mean_val) / std_val)
        
        anomalies = []
        for idx, z_score in z_scores.items():
            if z_score > threshold:
                anomalies.append({
                    'index': idx,
                    'value': series.iloc[idx],
                    'z_score': z_score,
                    'method': 'z_score',
                    'column': column_name,
                    'threshold': threshold,
                    'mean': mean_val,
                    'std': std_val,
                    'anomaly_score': min(z_score / threshold, 1.0)
                })
        
        return anomalies
    
    async def _modified_z_score_detection(self, series: pd.Series,
                                        threshold: float,
                                        column_name: str) -> List[Dict[str, Any]]:
        """Modified Z-score using median absolute deviation."""
        median_val = series.median()
        mad = np.median(np.abs(series - median_val))
        modified_z_scores = 0.6745 * (series - median_val) / mad
        
        anomalies = []
        for idx, mod_z_score in modified_z_scores.items():
            if abs(mod_z_score) > threshold:
                anomalies.append({
                    'index': idx,
                    'value': series.iloc[idx],
                    'modified_z_score': mod_z_score,
                    'method': 'modified_z_score',
                    'column': column_name,
                    'threshold': threshold,
                    'median': median_val,
                    'mad': mad,
                    'anomaly_score': min(abs(mod_z_score) / threshold, 1.0)
                })
        
        return anomalies
    
    async def _iqr_detection(self, series: pd.Series,
                           threshold: float,
                           column_name: str) -> List[Dict[str, Any]]:
        """Interquartile range based anomaly detection."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        anomalies = []
        for idx, value in series.items():
            if value < lower_bound or value > upper_bound:
                distance = max(lower_bound - value, value - upper_bound, 0)
                anomalies.append({
                    'index': idx,
                    'value': value,
                    'method': 'iqr',
                    'column': column_name,
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'distance': distance,
                    'anomaly_score': min(distance / iqr, 1.0) if iqr > 0 else 1.0
                })
        
        return anomalies
    
    async def _grubbs_test(self, series: pd.Series,
                         threshold: float,
                         column_name: str) -> List[Dict[str, Any]]:
        """Grubbs test for outlier detection."""
        n = len(series)
        if n < 3:
            return []
        
        mean_val = series.mean()
        std_val = series.std()
        
        # Calculate Grubbs statistic for each point
        grubbs_stats = np.abs(series - mean_val) / std_val
        max_grubbs = grubbs_stats.max()
        
        # Critical value for Grubbs test (approximation)
        alpha = 1 - threshold / 3.0  # Convert threshold to significance level
        t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        critical_value = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        
        anomalies = []
        if max_grubbs > critical_value:
            max_idx = grubbs_stats.idxmax()
            anomalies.append({
                'index': max_idx,
                'value': series[max_idx],
                'grubbs_statistic': max_grubbs,
                'critical_value': critical_value,
                'method': 'grubbs_test',
                'column': column_name,
                'anomaly_score': min(max_grubbs / critical_value, 1.0)
            })
        
        return anomalies
    
    async def _dixon_test(self, series: pd.Series,
                        threshold: float,
                        column_name: str) -> List[Dict[str, Any]]:
        """Dixon's Q test for outlier detection."""
        n = len(series)
        if n < 3:
            return []
        
        sorted_series = series.sort_values()
        values = sorted_series.values
        
        # Calculate Q statistics
        q_low = (values[1] - values[0]) / (values[-1] - values[0])
        q_high = (values[-1] - values[-2]) / (values[-1] - values[0])
        
        # Critical values (simplified)
        critical_values = {10: 0.412, 15: 0.338, 20: 0.300, 30: 0.254}
        critical_value = critical_values.get(min(30, n), 0.254)
        
        anomalies = []
        if q_low > critical_value:
            anomalies.append({
                'index': sorted_series.index[0],
                'value': values[0],
                'q_statistic': q_low,
                'critical_value': critical_value,
                'method': 'dixon_test',
                'column': column_name,
                'position': 'low',
                'anomaly_score': min(q_low / critical_value, 1.0)
            })
        
        if q_high > critical_value:
            anomalies.append({
                'index': sorted_series.index[-1],
                'value': values[-1],
                'q_statistic': q_high,
                'critical_value': critical_value,
                'method': 'dixon_test',
                'column': column_name,
                'position': 'high',
                'anomaly_score': min(q_high / critical_value, 1.0)
            })
        
        return anomalies
    
    async def _mahalanobis_detection(self, series: pd.Series,
                                   threshold: float,
                                   column_name: str) -> List[Dict[str, Any]]:
        """Mahalanobis distance based anomaly detection (requires multivariate data)."""
        # For univariate data, fall back to z-score
        return await self._z_score_detection(series, threshold, column_name)

class MachineLearningAnomalyDetector:
    """Machine learning based anomaly detection."""
    
    def __init__(self):
        """Initialize the ML anomaly detector."""
        self.models = {
            'isolation_forest': IsolationForest,
            'local_outlier_factor': LocalOutlierFactor,
            'elliptic_envelope': EllipticEnvelope,
            'dbscan': DBSCAN
        }
        self.scalers = {}
        self.trained_models = {}
        
    async def detect_anomalies(self, data: pd.DataFrame,
                             method: str = 'isolation_forest',
                             contamination: float = 0.1,
                             target_columns: List[str] = None) -> List[Dict[str, Any]]:
        """Detect anomalies using machine learning methods."""
        try:
            if target_columns is None:
                target_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Prepare data
            X = data[target_columns].fillna(data[target_columns].mean())
            
            if len(X) < 10:
                logger.warning("Insufficient data for ML anomaly detection")
                return []
            
            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply selected method
            if method == 'isolation_forest':
                anomalies = await self._isolation_forest_detection(
                    X, X_scaled, contamination, target_columns
                )
            elif method == 'local_outlier_factor':
                anomalies = await self._lof_detection(
                    X, X_scaled, contamination, target_columns
                )
            elif method == 'elliptic_envelope':
                anomalies = await self._elliptic_envelope_detection(
                    X, X_scaled, contamination, target_columns
                )
            elif method == 'dbscan':
                anomalies = await self._dbscan_detection(
                    X, X_scaled, target_columns
                )
            else:
                raise ValueError(f"Unknown ML method: {method}")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {str(e)}")
            raise
    
    async def _isolation_forest_detection(self, X: pd.DataFrame,
                                        X_scaled: np.ndarray,
                                        contamination: float,
                                        columns: List[str]) -> List[Dict[str, Any]]:
        """Isolation Forest based anomaly detection."""
        model = IsolationForest(contamination=contamination, random_state=42)
        predictions = model.fit_predict(X_scaled)
        scores = model.score_samples(X_scaled)
        
        anomalies = []
        for idx, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:  # Anomaly
                anomalies.append({
                    'index': X.index[idx],
                    'method': 'isolation_forest',
                    'prediction': pred,
                    'anomaly_score': abs(score),
                    'features': dict(zip(columns, X.iloc[idx])),
                    'scaled_features': dict(zip(columns, X_scaled[idx])),
                    'contamination': contamination
                })
        
        return anomalies
    
    async def _lof_detection(self, X: pd.DataFrame,
                           X_scaled: np.ndarray,
                           contamination: float,
                           columns: List[str]) -> List[Dict[str, Any]]:
        """Local Outlier Factor based anomaly detection."""
        n_neighbors = min(20, len(X) - 1)
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        predictions = model.fit_predict(X_scaled)
        scores = model.negative_outlier_factor_
        
        anomalies = []
        for idx, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:  # Anomaly
                anomalies.append({
                    'index': X.index[idx],
                    'method': 'local_outlier_factor',
                    'prediction': pred,
                    'anomaly_score': abs(score),
                    'features': dict(zip(columns, X.iloc[idx])),
                    'scaled_features': dict(zip(columns, X_scaled[idx])),
                    'n_neighbors': n_neighbors,
                    'contamination': contamination
                })
        
        return anomalies
    
    async def _elliptic_envelope_detection(self, X: pd.DataFrame,
                                         X_scaled: np.ndarray,
                                         contamination: float,
                                         columns: List[str]) -> List[Dict[str, Any]]:
        """Elliptic Envelope based anomaly detection."""
        model = EllipticEnvelope(contamination=contamination, random_state=42)
        predictions = model.fit_predict(X_scaled)
        
        anomalies = []
        for idx, pred in enumerate(predictions):
            if pred == -1:  # Anomaly
                anomalies.append({
                    'index': X.index[idx],
                    'method': 'elliptic_envelope',
                    'prediction': pred,
                    'features': dict(zip(columns, X.iloc[idx])),
                    'scaled_features': dict(zip(columns, X_scaled[idx])),
                    'contamination': contamination,
                    'anomaly_score': 0.8  # Placeholder score
                })
        
        return anomalies
    
    async def _dbscan_detection(self, X: pd.DataFrame,
                              X_scaled: np.ndarray,
                              columns: List[str]) -> List[Dict[str, Any]]:
        """DBSCAN clustering based anomaly detection."""
        model = DBSCAN(eps=0.5, min_samples=5)
        clusters = model.fit_predict(X_scaled)
        
        anomalies = []
        for idx, cluster in enumerate(clusters):
            if cluster == -1:  # Noise point (anomaly)
                anomalies.append({
                    'index': X.index[idx],
                    'method': 'dbscan',
                    'cluster': cluster,
                    'features': dict(zip(columns, X.iloc[idx])),
                    'scaled_features': dict(zip(columns, X_scaled[idx])),
                    'anomaly_score': 0.7  # Placeholder score
                })
        
        return anomalies

class RuleBasedAnomalyDetector:
    """Rule-based anomaly detection for specific business logic."""
    
    def __init__(self):
        """Initialize the rule-based detector."""
        self.rules = {}
        self.rule_history = {}
        
    async def add_rule(self, rule: AlertRule) -> None:
        """Add a new detection rule."""
        self.rules[rule.rule_id] = rule
        self.rule_history[rule.rule_id] = []
        logger.info(f"Added detection rule: {rule.rule_name}")
    
    async def detect_anomalies(self, data: Dict[str, Any],
                             active_rules_only: bool = True) -> List[Dict[str, Any]]:
        """Detect anomalies using configured rules."""
        try:
            anomalies = []
            
            for rule_id, rule in self.rules.items():
                if active_rules_only and not rule.is_active:
                    continue
                
                try:
                    rule_anomalies = await self._evaluate_rule(rule, data)
                    anomalies.extend(rule_anomalies)
                    
                    # Update rule statistics
                    if rule_anomalies:
                        rule.trigger_count += len(rule_anomalies)
                        rule.last_triggered = datetime.now()
                        
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule_id}: {str(e)}")
                    continue
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in rule-based anomaly detection: {str(e)}")
            raise
    
    async def _evaluate_rule(self, rule: AlertRule, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate a specific rule against data."""
        anomalies = []
        
        try:
            if rule.rule_type == 'threshold':
                anomalies = await self._evaluate_threshold_rule(rule, data)
            elif rule.rule_type == 'pattern':
                anomalies = await self._evaluate_pattern_rule(rule, data)
            elif rule.rule_type == 'statistical':
                anomalies = await self._evaluate_statistical_rule(rule, data)
            else:
                logger.warning(f"Unknown rule type: {rule.rule_type}")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.rule_id}: {str(e)}")
            return []
    
    async def _evaluate_threshold_rule(self, rule: AlertRule, 
                                     data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate threshold-based rules."""
        anomalies = []
        
        for field, threshold_config in rule.thresholds.items():
            if field not in data:
                continue
            
            value = data[field]
            threshold_type = threshold_config.get('type', 'greater_than')
            threshold_value = threshold_config.get('value')
            severity = threshold_config.get('severity', 'medium')
            
            is_anomaly = False
            if threshold_type == 'greater_than' and value > threshold_value:
                is_anomaly = True
            elif threshold_type == 'less_than' and value < threshold_value:
                is_anomaly = True
            elif threshold_type == 'equal' and value == threshold_value:
                is_anomaly = True
            elif threshold_type == 'not_equal' and value != threshold_value:
                is_anomaly = True
            elif threshold_type == 'between':
                min_val, max_val = threshold_value
                if min_val <= value <= max_val:
                    is_anomaly = True
            elif threshold_type == 'outside':
                min_val, max_val = threshold_value
                if value < min_val or value > max_val:
                    is_anomaly = True
            
            if is_anomaly:
                anomaly_score = self._calculate_threshold_score(
                    value, threshold_value, threshold_type
                )
                
                anomalies.append({
                    'rule_id': rule.rule_id,
                    'rule_name': rule.rule_name,
                    'rule_type': 'threshold',
                    'field': field,
                    'value': value,
                    'threshold_value': threshold_value,
                    'threshold_type': threshold_type,
                    'severity': severity,
                    'anomaly_score': anomaly_score,
                    'detection_timestamp': datetime.now()
                })
        
        return anomalies
    
    async def _evaluate_pattern_rule(self, rule: AlertRule, 
                                   data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate pattern-based rules."""
        # Implementation for pattern matching rules
        # This could include regex patterns, sequence patterns, etc.
        return []
    
    async def _evaluate_statistical_rule(self, rule: AlertRule,
                                        data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate statistical rules."""
        # Implementation for statistical rules (moving averages, trends, etc.)
        return []
    
    def _calculate_threshold_score(self, value: float, threshold: Union[float, List[float]], 
                                 threshold_type: str) -> float:
        """Calculate anomaly score for threshold violations."""
        if threshold_type in ['greater_than', 'less_than']:
            return min(abs(value - threshold) / max(abs(threshold), 1), 1.0)
        elif threshold_type in ['between', 'outside']:
            min_val, max_val = threshold
            range_size = max_val - min_val
            if threshold_type == 'between':
                return 1.0 if min_val <= value <= max_val else 0.0
            else:  # outside
                if value < min_val:
                    return min(abs(value - min_val) / range_size, 1.0)
                elif value > max_val:
                    return min(abs(value - max_val) / range_size, 1.0)
                else:
                    return 0.0
        else:
            return 1.0

class AlertManager:
    """Manages alerts and notifications for detected anomalies."""
    
    def __init__(self):
        """Initialize the alert manager."""
        self.active_alerts = {}
        self.alert_history = []
        self.notification_channels = {}
        self.suppression_rules = {}
        
    async def process_anomaly(self, anomaly: AnomalyDetection) -> Dict[str, Any]:
        """Process a detected anomaly and generate appropriate alerts."""
        try:
            alert_info = {
                'alert_id': str(uuid.uuid4()),
                'anomaly_id': anomaly.anomaly_id,
                'alert_timestamp': datetime.now(),
                'severity': anomaly.severity,
                'status': 'active',
                'notifications_sent': []
            }
            
            # Check suppression rules
            if await self._should_suppress_alert(anomaly):
                alert_info['status'] = 'suppressed'
                logger.info(f"Alert suppressed for anomaly {anomaly.anomaly_id}")
                return alert_info
            
            # Determine escalation level
            escalation_level = await self._determine_escalation_level(anomaly)
            alert_info['escalation_level'] = escalation_level
            
            # Send notifications
            notifications = await self._send_notifications(anomaly, escalation_level)
            alert_info['notifications_sent'] = notifications
            
            # Store active alert
            self.active_alerts[alert_info['alert_id']] = alert_info
            self.alert_history.append(alert_info)
            
            logger.info(f"Processed alert for anomaly {anomaly.anomaly_id}")
            return alert_info
            
        except Exception as e:
            logger.error(f"Error processing anomaly alert: {str(e)}")
            raise
    
    async def _should_suppress_alert(self, anomaly: AnomalyDetection) -> bool:
        """Check if an alert should be suppressed."""
        # Implement suppression logic based on:
        # - Recent similar alerts
        # - Maintenance windows
        # - User-defined suppression rules
        # - Alert frequency limits
        
        # Simple implementation: suppress if similar alert in last 10 minutes
        recent_cutoff = datetime.now() - timedelta(minutes=10)
        
        for alert in self.alert_history:
            if (alert['alert_timestamp'] > recent_cutoff and
                alert['severity'] == anomaly.severity and
                alert.get('anomaly_category') == anomaly.anomaly_category):
                return True
        
        return False
    
    async def _determine_escalation_level(self, anomaly: AnomalyDetection) -> str:
        """Determine the escalation level for an anomaly."""
        if anomaly.severity == 'critical':
            return 'immediate'
        elif anomaly.severity == 'high':
            return 'urgent'
        elif anomaly.severity == 'medium':
            return 'normal'
        else:
            return 'low'
    
    async def _send_notifications(self, anomaly: AnomalyDetection, 
                                escalation_level: str) -> List[str]:
        """Send notifications for the anomaly."""
        notifications_sent = []
        
        try:
            # Email notifications
            if escalation_level in ['immediate', 'urgent']:
                email_sent = await self._send_email_notification(anomaly)
                if email_sent:
                    notifications_sent.append('email')
            
            # SMS notifications for critical alerts
            if escalation_level == 'immediate':
                sms_sent = await self._send_sms_notification(anomaly)
                if sms_sent:
                    notifications_sent.append('sms')
            
            # Slack notifications
            slack_sent = await self._send_slack_notification(anomaly)
            if slack_sent:
                notifications_sent.append('slack')
            
            # In-app notifications
            in_app_sent = await self._send_in_app_notification(anomaly)
            if in_app_sent:
                notifications_sent.append('in_app')
            
        except Exception as e:
            logger.error(f"Error sending notifications: {str(e)}")
        
        return notifications_sent
    
    async def _send_email_notification(self, anomaly: AnomalyDetection) -> bool:
        """Send email notification."""
        # Implementation would integrate with email service
        logger.info(f"Email notification would be sent for anomaly {anomaly.anomaly_id}")
        return True
    
    async def _send_sms_notification(self, anomaly: AnomalyDetection) -> bool:
        """Send SMS notification."""
        # Implementation would integrate with SMS service
        logger.info(f"SMS notification would be sent for anomaly {anomaly.anomaly_id}")
        return True
    
    async def _send_slack_notification(self, anomaly: AnomalyDetection) -> bool:
        """Send Slack notification."""
        # Implementation would integrate with Slack API
        logger.info(f"Slack notification would be sent for anomaly {anomaly.anomaly_id}")
        return True
    
    async def _send_in_app_notification(self, anomaly: AnomalyDetection) -> bool:
        """Send in-app notification."""
        # Implementation would create in-app notification record
        logger.info(f"In-app notification created for anomaly {anomaly.anomaly_id}")
        return True

class InvestigationTools:
    """Tools for investigating detected anomalies."""
    
    def __init__(self):
        """Initialize investigation tools."""
        self.active_sessions = {}
        self.investigation_templates = {}
        
    async def start_investigation(self, anomaly_ids: List[str],
                                investigator_id: str,
                                session_name: str) -> InvestigationSession:
        """Start a new investigation session."""
        try:
            session = InvestigationSession(
                session_id=str(uuid.uuid4()),
                anomaly_ids=anomaly_ids,
                investigator_id=investigator_id,
                session_name=session_name,
                investigation_status='active',
                priority_level='medium',
                hypothesis=[],
                evidence_collected=[],
                analysis_results={},
                findings=[],
                started_at=datetime.now(),
                last_activity_at=datetime.now(),
                collaborators=[],
                notes=[],
                attachments=[],
                metadata={}
            )
            
            self.active_sessions[session.session_id] = session
            
            # Generate initial investigation suggestions
            initial_suggestions = await self._generate_investigation_suggestions(anomaly_ids)
            session.metadata['initial_suggestions'] = initial_suggestions
            
            logger.info(f"Started investigation session {session.session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error starting investigation: {str(e)}")
            raise
    
    async def _generate_investigation_suggestions(self, 
                                                anomaly_ids: List[str]) -> List[str]:
        """Generate investigation suggestions based on anomaly types."""
        suggestions = [
            "Review historical data patterns for similar anomalies",
            "Check system logs around the anomaly detection time",
            "Analyze correlation with external events or changes",
            "Verify data quality and collection processes",
            "Compare with peer institutions or benchmarks",
            "Review recent system or process changes",
            "Analyze user behavior patterns leading to the anomaly",
            "Check for potential data entry errors or system glitches"
        ]
        return suggestions
    
    async def add_hypothesis(self, session_id: str, hypothesis: str) -> bool:
        """Add a hypothesis to an investigation session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.hypothesis.append({
            'hypothesis': hypothesis,
            'added_at': datetime.now(),
            'added_by': session.investigator_id,
            'status': 'active'
        })
        session.last_activity_at = datetime.now()
        
        return True
    
    async def collect_evidence(self, session_id: str, 
                             evidence: Dict[str, Any]) -> bool:
        """Add evidence to an investigation session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        evidence_record = {
            **evidence,
            'collected_at': datetime.now(),
            'collected_by': session.investigator_id,
            'evidence_id': str(uuid.uuid4())
        }
        session.evidence_collected.append(evidence_record)
        session.last_activity_at = datetime.now()
        
        return True
    
    async def analyze_patterns(self, session_id: str,
                             data: pd.DataFrame) -> Dict[str, Any]:
        """Perform pattern analysis on investigation data."""
        try:
            if session_id not in self.active_sessions:
                return {}
            
            session = self.active_sessions[session_id]
            analysis_results = {}
            
            # Temporal pattern analysis
            if 'timestamp' in data.columns:
                temporal_patterns = await self._analyze_temporal_patterns(data)
                analysis_results['temporal_patterns'] = temporal_patterns
            
            # Statistical analysis
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                statistical_analysis = await self._perform_statistical_analysis(
                    data[numeric_columns]
                )
                analysis_results['statistical_analysis'] = statistical_analysis
            
            # Correlation analysis
            if len(numeric_columns) > 1:
                correlation_analysis = await self._analyze_correlations(
                    data[numeric_columns]
                )
                analysis_results['correlation_analysis'] = correlation_analysis
            
            # Update session
            session.analysis_results.update(analysis_results)
            session.last_activity_at = datetime.now()
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            return {}
    
    async def _analyze_temporal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in the data."""
        try:
            temporal_analysis = {}
            
            # Convert timestamp column
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Time-based aggregations
            hourly_counts = data.groupby(data['timestamp'].dt.hour).size()
            daily_counts = data.groupby(data['timestamp'].dt.day_name()).size()
            
            temporal_analysis['hourly_distribution'] = hourly_counts.to_dict()
            temporal_analysis['daily_distribution'] = daily_counts.to_dict()
            temporal_analysis['peak_hour'] = hourly_counts.idxmax()
            temporal_analysis['peak_day'] = daily_counts.idxmax()
            
            return temporal_analysis
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {str(e)}")
            return {}
    
    async def _perform_statistical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis on numeric data."""
        try:
            stats_analysis = {}
            
            for column in data.columns:
                column_stats = {
                    'mean': data[column].mean(),
                    'median': data[column].median(),
                    'std': data[column].std(),
                    'min': data[column].min(),
                    'max': data[column].max(),
                    'skewness': stats.skew(data[column].dropna()),
                    'kurtosis': stats.kurtosis(data[column].dropna()),
                    'outlier_count': len(data[column][
                        np.abs(stats.zscore(data[column].dropna())) > 3
                    ])
                }
                stats_analysis[column] = column_stats
            
            return stats_analysis
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {str(e)}")
            return {}
    
    async def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric variables."""
        try:
            correlation_matrix = data.corr()
            
            # Find strongest correlations
            correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    corr_value = correlation_matrix.iloc[i, j]
                    
                    if abs(corr_value) > 0.5:  # Strong correlation threshold
                        correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                        })
            
            correlation_analysis = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strong_correlations': correlations,
                'highest_correlation': max(correlations, 
                                         key=lambda x: abs(x['correlation'])) if correlations else None
            }
            
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return {}

class AnomalyDetectionInterface:
    """Main interface for anomaly detection system."""
    
    def __init__(self):
        """Initialize the anomaly detection interface."""
        self.statistical_detector = StatisticalAnomalyDetector()
        self.ml_detector = MachineLearningAnomalyDetector()
        self.rule_detector = RuleBasedAnomalyDetector()
        self.alert_manager = AlertManager()
        self.investigation_tools = InvestigationTools()
        
        # System state
        self.detected_anomalies = {}
        self.detection_statistics = {
            'total_detections': 0,
            'false_positives': 0,
            'true_positives': 0,
            'detection_accuracy': 0.0
        }
        
        logger.info("Anomaly Detection Interface initialized")
    
    async def detect_anomalies(self, data: Union[pd.DataFrame, Dict[str, Any]],
                             methods: List[str] = None,
                             severity_threshold: str = 'medium') -> List[AnomalyDetection]:
        """Main anomaly detection function."""
        try:
            if methods is None:
                methods = ['statistical', 'ml_based', 'rule_based']
            
            all_anomalies = []
            
            # Convert dict data to DataFrame if needed
            if isinstance(data, dict):
                df_data = pd.DataFrame([data])
            else:
                df_data = data
            
            # Statistical detection
            if 'statistical' in methods:
                stat_anomalies = await self.statistical_detector.detect_anomalies(
                    df_data, method='z_score', threshold=3.0
                )
                stat_anomaly_objects = await self._convert_to_anomaly_objects(
                    stat_anomalies, 'statistical', df_data
                )
                all_anomalies.extend(stat_anomaly_objects)
            
            # ML-based detection
            if 'ml_based' in methods and len(df_data) >= 10:
                ml_anomalies = await self.ml_detector.detect_anomalies(
                    df_data, method='isolation_forest', contamination=0.1
                )
                ml_anomaly_objects = await self._convert_to_anomaly_objects(
                    ml_anomalies, 'ml_based', df_data
                )
                all_anomalies.extend(ml_anomaly_objects)
            
            # Rule-based detection
            if 'rule_based' in methods and isinstance(data, dict):
                rule_anomalies = await self.rule_detector.detect_anomalies(data)
                rule_anomaly_objects = await self._convert_to_anomaly_objects(
                    rule_anomalies, 'rule_based', data
                )
                all_anomalies.extend(rule_anomaly_objects)
            
            # Filter by severity threshold
            filtered_anomalies = self._filter_by_severity(all_anomalies, severity_threshold)
            
            # Store detected anomalies
            for anomaly in filtered_anomalies:
                self.detected_anomalies[anomaly.anomaly_id] = anomaly
                
                # Process alerts
                await self.alert_manager.process_anomaly(anomaly)
            
            # Update statistics
            self.detection_statistics['total_detections'] += len(filtered_anomalies)
            
            logger.info(f"Detected {len(filtered_anomalies)} anomalies")
            return filtered_anomalies
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise
    
    async def _convert_to_anomaly_objects(self, raw_anomalies: List[Dict[str, Any]],
                                        detection_type: str,
                                        data: Union[pd.DataFrame, Dict[str, Any]]) -> List[AnomalyDetection]:
        """Convert raw anomaly detections to AnomalyDetection objects."""
        anomaly_objects = []
        
        for raw_anomaly in raw_anomalies:
            try:
                anomaly_id = str(uuid.uuid4())
                
                # Determine severity based on anomaly score
                anomaly_score = raw_anomaly.get('anomaly_score', 0.5)
                if anomaly_score >= 0.8:
                    severity = 'critical'
                elif anomaly_score >= 0.6:
                    severity = 'high'
                elif anomaly_score >= 0.4:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                # Create anomaly object
                anomaly = AnomalyDetection(
                    anomaly_id=anomaly_id,
                    detection_type=detection_type,
                    anomaly_category='application',  # Default category
                    severity=severity,
                    confidence_score=raw_anomaly.get('confidence', 0.8),
                    anomaly_score=anomaly_score,
                    title=f"Anomaly detected using {detection_type}",
                    description=f"Anomaly detected in {raw_anomaly.get('column', 'data')}",
                    affected_entities={'data_points': [raw_anomaly.get('index', 0)]},
                    detection_method=raw_anomaly.get('method', detection_type),
                    detection_parameters=raw_anomaly,
                    reference_patterns={},
                    deviation_metrics=raw_anomaly,
                    anomalous_data=raw_anomaly,
                    historical_baseline={},
                    statistical_measures=raw_anomaly,
                    investigation_priority=severity,
                    recommended_actions=[
                        "Review the flagged data point for accuracy",
                        "Compare with historical patterns",
                        "Investigate potential root causes"
                    ],
                    investigation_hints=[
                        "Check for data entry errors",
                        "Verify system processes",
                        "Analyze external factors"
                    ],
                    related_anomalies=[],
                    detection_timestamp=datetime.now(),
                    data_timestamp=datetime.now(),
                    time_window='current',
                    status='new',
                    metadata={'raw_detection': raw_anomaly},
                    tags=[detection_type, severity]
                )
                
                anomaly_objects.append(anomaly)
                
            except Exception as e:
                logger.error(f"Error converting anomaly: {str(e)}")
                continue
        
        return anomaly_objects
    
    def _filter_by_severity(self, anomalies: List[AnomalyDetection], 
                          threshold: str) -> List[AnomalyDetection]:
        """Filter anomalies by severity threshold."""
        severity_levels = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        threshold_level = severity_levels.get(threshold, 1)
        
        filtered = []
        for anomaly in anomalies:
            anomaly_level = severity_levels.get(anomaly.severity, 1)
            if anomaly_level >= threshold_level:
                filtered.append(anomaly)
        
        return filtered
    
    async def get_anomaly_summary(self, time_period: str = '24h') -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        try:
            # Calculate time cutoff
            hours = int(time_period.rstrip('h'))
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter recent anomalies
            recent_anomalies = [
                anomaly for anomaly in self.detected_anomalies.values()
                if anomaly.detection_timestamp > cutoff_time
            ]
            
            # Calculate statistics
            total_count = len(recent_anomalies)
            severity_counts = {}
            category_counts = {}
            method_counts = {}
            
            for anomaly in recent_anomalies:
                # Severity distribution
                severity_counts[anomaly.severity] = severity_counts.get(
                    anomaly.severity, 0
                ) + 1
                
                # Category distribution
                category_counts[anomaly.anomaly_category] = category_counts.get(
                    anomaly.anomaly_category, 0
                ) + 1
                
                # Method distribution
                method_counts[anomaly.detection_type] = method_counts.get(
                    anomaly.detection_type, 0
                ) + 1
            
            summary = {
                'time_period': time_period,
                'total_anomalies': total_count,
                'severity_distribution': severity_counts,
                'category_distribution': category_counts,
                'detection_method_distribution': method_counts,
                'average_confidence': np.mean([
                    a.confidence_score for a in recent_anomalies
                ]) if recent_anomalies else 0,
                'detection_statistics': self.detection_statistics,
                'active_investigations': len(self.investigation_tools.active_sessions)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating anomaly summary: {str(e)}")
            return {}
    
    async def update_anomaly_feedback(self, anomaly_id: str,
                                    is_true_positive: bool,
                                    feedback_notes: str = None) -> bool:
        """Update feedback for an anomaly detection."""
        try:
            if anomaly_id not in self.detected_anomalies:
                return False
            
            anomaly = self.detected_anomalies[anomaly_id]
            
            # Update anomaly status
            if is_true_positive:
                anomaly.status = 'confirmed'
                self.detection_statistics['true_positives'] += 1
            else:
                anomaly.status = 'false_positive'
                self.detection_statistics['false_positives'] += 1
            
            # Add feedback to metadata
            if anomaly.metadata is None:
                anomaly.metadata = {}
            
            anomaly.metadata['feedback'] = {
                'is_true_positive': is_true_positive,
                'feedback_notes': feedback_notes,
                'feedback_timestamp': datetime.now()
            }
            
            # Update detection accuracy
            total_feedback = (self.detection_statistics['true_positives'] + 
                            self.detection_statistics['false_positives'])
            if total_feedback > 0:
                self.detection_statistics['detection_accuracy'] = (
                    self.detection_statistics['true_positives'] / total_feedback
                )
            
            logger.info(f"Updated feedback for anomaly {anomaly_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating anomaly feedback: {str(e)}")
            return False

# Main execution function for testing
async def main():
    """Main function for testing the Anomaly Detection Interface."""
    try:
        detector = AnomalyDetectionInterface()
        
        # Generate sample data
        np.random.seed(42)
        normal_data = np.random.normal(100, 15, 1000)
        anomaly_data = np.concatenate([normal_data, [200, 250, -50]])  # Add anomalies
        
        df = pd.DataFrame({
            'value1': anomaly_data,
            'value2': np.random.normal(50, 10, len(anomaly_data)),
            'timestamp': pd.date_range('2024-01-01', periods=len(anomaly_data), freq='H')
        })
        
        print("Testing anomaly detection...")
        
        # Detect anomalies
        anomalies = await detector.detect_anomalies(df, methods=['statistical', 'ml_based'])
        
        print(f"✅ Detection Complete!")
        print(f"Detected {len(anomalies)} anomalies")
        
        for anomaly in anomalies[:3]:  # Show first 3 anomalies
            print(f"- {anomaly.title} (Severity: {anomaly.severity}, Score: {anomaly.anomaly_score:.2f})")
        
        # Get summary
        summary = await detector.get_anomaly_summary()
        print(f"\nSummary: {summary['total_anomalies']} total anomalies")
        print(f"Severity distribution: {summary['severity_distribution']}")
        
        # Test investigation tools
        if anomalies:
            investigation = await detector.investigation_tools.start_investigation(
                [anomalies[0].anomaly_id],
                "test_investigator",
                "Test Investigation"
            )
            print(f"Started investigation: {investigation.session_id}")
        
        return detector
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
