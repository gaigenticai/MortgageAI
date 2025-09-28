#!/usr/bin/env python3
"""
Comprehensive Compliance Audit Trail System
Advanced system for detailed logging, regulatory reporting, and investigation capabilities

Features:
- Real-time compliance event tracking with forensic-level detail
- Advanced regulatory reporting with automated compliance dashboards
- Sophisticated investigation tools with pattern recognition
- Immutable audit logs with blockchain-style verification
- Multi-dimensional compliance analytics and trend analysis
- Automated compliance violation detection and alerting
- Comprehensive data lineage and impact analysis
- Advanced search and filtering with natural language queries
- Regulatory change impact assessment and notification
- Forensic investigation capabilities with timeline reconstruction
"""

import asyncio
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import asyncpg
import aioredis
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import statistics
import re
import hmac
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import sys
import traceback
import ipaddress
from urllib.parse import urlparse
import geoip2.database
import geoip2.errors
from user_agents import parse as parse_user_agent
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import websockets
import ssl
import certifi
import requests
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceEventType(Enum):
    """Compliance event types"""
    USER_ACTION = "user_action"
    SYSTEM_ACTION = "system_action"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    REGULATION_CHECK = "regulation_check"
    VIOLATION_DETECTED = "violation_detected"
    REMEDIATION_ACTION = "remediation_action"
    AUDIT_REVIEW = "audit_review"
    REGULATORY_CHANGE = "regulatory_change"
    INVESTIGATION = "investigation"

class ComplianceSeverity(Enum):
    """Compliance severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class InvestigationStatus(Enum):
    """Investigation status levels"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"

class AuditTrailIntegrity(Enum):
    """Audit trail integrity status"""
    VERIFIED = "verified"
    COMPROMISED = "compromised"
    UNKNOWN = "unknown"

@dataclass
class ComplianceEvent:
    """Comprehensive compliance event record"""
    event_id: str
    event_type: ComplianceEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    geo_location: Optional[Dict[str, Any]]
    entity_type: str
    entity_id: str
    action: str
    details: Dict[str, Any]
    regulation: Optional[str]
    compliance_status: str
    severity: ComplianceSeverity
    risk_score: float
    data_classification: str
    retention_period: int
    encryption_key_id: Optional[str]
    hash_chain_previous: Optional[str]
    hash_chain_current: str
    digital_signature: Optional[str]
    investigation_id: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    event_id: str
    regulation: str
    article: str
    violation_type: str
    description: str
    severity: ComplianceSeverity
    risk_impact: str
    affected_entities: List[str]
    detection_method: str
    detection_timestamp: datetime
    remediation_actions: List[str]
    remediation_deadline: Optional[datetime]
    remediation_status: str
    investigation_required: bool
    notification_sent: bool
    escalation_level: int
    compliance_officer_assigned: Optional[str]
    resolution_timestamp: Optional[datetime]
    resolution_details: Optional[str]

@dataclass
class Investigation:
    """Compliance investigation record"""
    investigation_id: str
    title: str
    description: str
    investigation_type: str
    priority: str
    status: InvestigationStatus
    assigned_investigator: str
    created_timestamp: datetime
    updated_timestamp: datetime
    deadline: Optional[datetime]
    related_events: List[str]
    related_violations: List[str]
    evidence_collected: List[Dict[str, Any]]
    findings: List[str]
    conclusions: Optional[str]
    recommendations: List[str]
    actions_taken: List[str]
    timeline: List[Dict[str, Any]]
    stakeholders: List[str]
    confidentiality_level: str
    tags: List[str]

@dataclass
class RegulatoryChange:
    """Regulatory change tracking"""
    change_id: str
    regulation: str
    change_type: str
    title: str
    description: str
    effective_date: datetime
    impact_assessment: str
    affected_systems: List[str]
    implementation_required: bool
    implementation_deadline: Optional[datetime]
    implementation_status: str
    change_source: str
    change_document_url: Optional[str]
    stakeholders_notified: List[str]
    impact_score: float
    created_timestamp: datetime

class ComplianceAnalyzer:
    """Advanced compliance analytics and pattern recognition"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.pattern_cache = {}
        self.risk_models = {}
    
    def analyze_compliance_patterns(self, events: List[ComplianceEvent]) -> Dict[str, Any]:
        """Analyze compliance patterns and anomalies"""
        try:
            if not events:
                return {"patterns": [], "anomalies": [], "insights": []}
            
            # Convert events to feature matrix
            features = self._extract_features(events)
            
            if len(features) < 10:  # Need minimum data for analysis
                return {"patterns": [], "anomalies": [], "insights": ["Insufficient data for pattern analysis"]}
            
            # Normalize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Detect anomalies
            anomaly_scores = self.anomaly_detector.fit_predict(features_scaled)
            anomalies = [events[i] for i, score in enumerate(anomaly_scores) if score == -1]
            
            # Identify patterns
            patterns = self._identify_patterns(events, features_scaled)
            
            # Generate insights
            insights = self._generate_insights(events, patterns, anomalies)
            
            return {
                "patterns": patterns,
                "anomalies": [asdict(anomaly) for anomaly in anomalies],
                "insights": insights,
                "risk_trends": self._analyze_risk_trends(events),
                "compliance_scores": self._calculate_compliance_scores(events)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing compliance patterns: {e}")
            return {"error": str(e)}
    
    def _extract_features(self, events: List[ComplianceEvent]) -> np.ndarray:
        """Extract numerical features from compliance events"""
        features = []
        
        for event in events:
            feature_vector = [
                event.timestamp.hour,  # Hour of day
                event.timestamp.weekday(),  # Day of week
                len(event.details),  # Complexity measure
                event.risk_score,  # Risk score
                len(event.tags),  # Number of tags
                hash(event.entity_type) % 1000,  # Entity type hash
                hash(event.action) % 1000,  # Action hash
                1 if event.severity == ComplianceSeverity.CRITICAL else 0,  # Critical flag
                1 if event.regulation else 0,  # Regulation flag
                len(event.affected_entities) if hasattr(event, 'affected_entities') else 0
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _identify_patterns(self, events: List[ComplianceEvent], features: np.ndarray) -> List[Dict[str, Any]]:
        """Identify compliance patterns using clustering"""
        try:
            # Use DBSCAN for pattern identification
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(features)
            
            patterns = []
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                
                cluster_events = [events[i] for i, l in enumerate(cluster_labels) if l == label]
                
                if len(cluster_events) >= 5:  # Significant pattern
                    pattern = self._analyze_cluster_pattern(cluster_events)
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")
            return []
    
    def _analyze_cluster_pattern(self, cluster_events: List[ComplianceEvent]) -> Dict[str, Any]:
        """Analyze a cluster of events to identify patterns"""
        # Common attributes in cluster
        common_actions = defaultdict(int)
        common_entities = defaultdict(int)
        common_regulations = defaultdict(int)
        severity_distribution = defaultdict(int)
        
        for event in cluster_events:
            common_actions[event.action] += 1
            common_entities[event.entity_type] += 1
            if event.regulation:
                common_regulations[event.regulation] += 1
            severity_distribution[event.severity.value] += 1
        
        # Time pattern analysis
        timestamps = [event.timestamp for event in cluster_events]
        time_span = max(timestamps) - min(timestamps)
        
        # Risk pattern
        risk_scores = [event.risk_score for event in cluster_events]
        avg_risk = statistics.mean(risk_scores)
        
        return {
            "pattern_id": str(uuid.uuid4()),
            "event_count": len(cluster_events),
            "time_span_hours": time_span.total_seconds() / 3600,
            "most_common_action": max(common_actions.items(), key=lambda x: x[1])[0],
            "most_common_entity": max(common_entities.items(), key=lambda x: x[1])[0],
            "primary_regulation": max(common_regulations.items(), key=lambda x: x[1])[0] if common_regulations else None,
            "severity_distribution": dict(severity_distribution),
            "average_risk_score": avg_risk,
            "pattern_type": self._classify_pattern_type(cluster_events),
            "recommendations": self._generate_pattern_recommendations(cluster_events)
        }
    
    def _classify_pattern_type(self, events: List[ComplianceEvent]) -> str:
        """Classify the type of pattern identified"""
        # Analyze temporal patterns
        timestamps = [event.timestamp for event in events]
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                     for i in range(len(timestamps)-1)]
        
        if time_diffs and max(time_diffs) < 3600:  # Within 1 hour
            return "burst_activity"
        elif len(set(event.user_id for event in events if event.user_id)) == 1:
            return "user_specific"
        elif len(set(event.entity_type for event in events)) == 1:
            return "entity_specific"
        elif all(event.severity == ComplianceSeverity.CRITICAL for event in events):
            return "critical_sequence"
        else:
            return "general_pattern"
    
    def _generate_pattern_recommendations(self, events: List[ComplianceEvent]) -> List[str]:
        """Generate recommendations based on identified patterns"""
        recommendations = []
        
        # High-risk pattern recommendations
        avg_risk = statistics.mean(event.risk_score for event in events)
        if avg_risk > 0.7:
            recommendations.append("Implement enhanced monitoring for high-risk activities")
        
        # Frequency-based recommendations
        if len(events) > 50:
            recommendations.append("Consider automated controls to prevent excessive activity")
        
        # Time-based recommendations
        timestamps = [event.timestamp for event in events]
        hours = [ts.hour for ts in timestamps]
        if statistics.mode(hours) in [22, 23, 0, 1, 2, 3]:  # Late night activity
            recommendations.append("Review after-hours access controls")
        
        # User-based recommendations
        users = [event.user_id for event in events if event.user_id]
        if len(set(users)) == 1 and len(events) > 20:
            recommendations.append("Conduct user behavior analysis for potential insider threats")
        
        return recommendations
    
    def _generate_insights(self, events: List[ComplianceEvent], patterns: List[Dict], anomalies: List[ComplianceEvent]) -> List[str]:
        """Generate compliance insights from analysis"""
        insights = []
        
        # Temporal insights
        timestamps = [event.timestamp for event in events]
        if timestamps:
            time_span = max(timestamps) - min(timestamps)
            insights.append(f"Analysis covers {time_span.days} days of compliance events")
        
        # Risk insights
        risk_scores = [event.risk_score for event in events]
        if risk_scores:
            avg_risk = statistics.mean(risk_scores)
            high_risk_count = sum(1 for score in risk_scores if score > 0.7)
            insights.append(f"Average risk score: {avg_risk:.2f}, {high_risk_count} high-risk events detected")
        
        # Pattern insights
        if patterns:
            insights.append(f"Identified {len(patterns)} compliance patterns requiring attention")
        
        # Anomaly insights
        if anomalies:
            insights.append(f"Detected {len(anomalies)} anomalous compliance events for investigation")
        
        # Regulation insights
        regulations = [event.regulation for event in events if event.regulation]
        if regulations:
            reg_counts = defaultdict(int)
            for reg in regulations:
                reg_counts[reg] += 1
            most_common_reg = max(reg_counts.items(), key=lambda x: x[1])
            insights.append(f"Most frequently referenced regulation: {most_common_reg[0]} ({most_common_reg[1]} events)")
        
        return insights
    
    def _analyze_risk_trends(self, events: List[ComplianceEvent]) -> Dict[str, Any]:
        """Analyze risk score trends over time"""
        if not events:
            return {}
        
        # Group events by day
        daily_risks = defaultdict(list)
        for event in events:
            day = event.timestamp.date()
            daily_risks[day].append(event.risk_score)
        
        # Calculate daily averages
        risk_trend = []
        for day in sorted(daily_risks.keys()):
            avg_risk = statistics.mean(daily_risks[day])
            risk_trend.append({"date": day.isoformat(), "average_risk": avg_risk})
        
        # Calculate trend direction
        if len(risk_trend) >= 2:
            recent_avg = statistics.mean([point["average_risk"] for point in risk_trend[-7:]])
            earlier_avg = statistics.mean([point["average_risk"] for point in risk_trend[:-7]] or [0])
            trend_direction = "increasing" if recent_avg > earlier_avg else "decreasing"
        else:
            trend_direction = "stable"
        
        return {
            "trend_data": risk_trend,
            "trend_direction": trend_direction,
            "current_average": risk_trend[-1]["average_risk"] if risk_trend else 0
        }
    
    def _calculate_compliance_scores(self, events: List[ComplianceEvent]) -> Dict[str, float]:
        """Calculate compliance scores by category"""
        scores = {}
        
        # Group events by regulation
        regulation_events = defaultdict(list)
        for event in events:
            if event.regulation:
                regulation_events[event.regulation].append(event)
        
        # Calculate score for each regulation
        for regulation, reg_events in regulation_events.items():
            violations = sum(1 for event in reg_events if event.compliance_status == "non_compliant")
            total = len(reg_events)
            compliance_rate = (total - violations) / total if total > 0 else 1.0
            scores[regulation] = compliance_rate
        
        # Overall compliance score
        if scores:
            scores["overall"] = statistics.mean(scores.values())
        
        return scores

class AuditTrailVerifier:
    """Immutable audit trail verification system"""
    
    def __init__(self, encryption_key: bytes):
        self.encryption_key = encryption_key
        self.fernet = Fernet(encryption_key)
    
    def create_hash_chain(self, event: ComplianceEvent, previous_hash: Optional[str] = None) -> str:
        """Create hash chain for immutable audit trail"""
        try:
            # Create event content for hashing
            event_content = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "action": event.action,
                "entity_type": event.entity_type,
                "entity_id": event.entity_id,
                "details": event.details
            }
            
            # Convert to canonical JSON string
            content_json = json.dumps(event_content, sort_keys=True, separators=(',', ':'))
            
            # Include previous hash in chain
            chain_content = f"{previous_hash or ''}{content_json}"
            
            # Create SHA-256 hash
            hash_object = hashlib.sha256(chain_content.encode('utf-8'))
            return hash_object.hexdigest()
            
        except Exception as e:
            logger.error(f"Error creating hash chain: {e}")
            return ""
    
    def verify_hash_chain(self, events: List[ComplianceEvent]) -> Dict[str, Any]:
        """Verify integrity of hash chain"""
        try:
            verification_results = []
            previous_hash = None
            
            for i, event in enumerate(events):
                expected_hash = self.create_hash_chain(event, previous_hash)
                is_valid = expected_hash == event.hash_chain_current
                
                verification_results.append({
                    "event_id": event.event_id,
                    "position": i,
                    "is_valid": is_valid,
                    "expected_hash": expected_hash,
                    "actual_hash": event.hash_chain_current
                })
                
                if not is_valid:
                    logger.warning(f"Hash chain verification failed for event {event.event_id}")
                
                previous_hash = event.hash_chain_current
            
            # Calculate overall integrity
            valid_count = sum(1 for result in verification_results if result["is_valid"])
            integrity_score = valid_count / len(verification_results) if verification_results else 0
            
            integrity_status = AuditTrailIntegrity.VERIFIED if integrity_score == 1.0 else AuditTrailIntegrity.COMPROMISED
            
            return {
                "integrity_status": integrity_status.value,
                "integrity_score": integrity_score,
                "total_events": len(events),
                "valid_events": valid_count,
                "invalid_events": len(events) - valid_count,
                "verification_details": verification_results,
                "verification_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error verifying hash chain: {e}")
            return {
                "integrity_status": AuditTrailIntegrity.UNKNOWN.value,
                "error": str(e)
            }
    
    def create_digital_signature(self, event: ComplianceEvent) -> str:
        """Create digital signature for event"""
        try:
            # Create signature content
            signature_content = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "hash_chain_current": event.hash_chain_current
            }
            
            content_json = json.dumps(signature_content, sort_keys=True)
            
            # Create HMAC signature
            signature = hmac.new(
                self.encryption_key,
                content_json.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            logger.error(f"Error creating digital signature: {e}")
            return ""

class ComplianceReportGenerator:
    """Advanced compliance reporting system"""
    
    def __init__(self):
        self.report_templates = self._load_report_templates()
    
    def _load_report_templates(self) -> Dict[str, Any]:
        """Load compliance report templates"""
        return {
            "regulatory_compliance": {
                "title": "Regulatory Compliance Report",
                "sections": ["executive_summary", "compliance_status", "violations", "remediation", "recommendations"],
                "frequency": "monthly"
            },
            "audit_summary": {
                "title": "Audit Trail Summary",
                "sections": ["audit_statistics", "integrity_verification", "anomalies", "investigations"],
                "frequency": "weekly"
            },
            "incident_report": {
                "title": "Compliance Incident Report",
                "sections": ["incident_details", "impact_assessment", "root_cause", "corrective_actions"],
                "frequency": "as_needed"
            },
            "trend_analysis": {
                "title": "Compliance Trend Analysis",
                "sections": ["trend_overview", "pattern_analysis", "risk_assessment", "forecasting"],
                "frequency": "quarterly"
            }
        }
    
    async def generate_regulatory_report(self, regulation: str, start_date: datetime, 
                                       end_date: datetime, events: List[ComplianceEvent],
                                       violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Generate comprehensive regulatory compliance report"""
        try:
            # Filter events for regulation and date range
            filtered_events = [
                event for event in events
                if event.regulation == regulation and start_date <= event.timestamp <= end_date
            ]
            
            filtered_violations = [
                violation for violation in violations
                if violation.regulation == regulation and start_date <= violation.detection_timestamp <= end_date
            ]
            
            # Executive Summary
            executive_summary = self._generate_executive_summary(filtered_events, filtered_violations)
            
            # Compliance Status
            compliance_status = self._calculate_compliance_status(filtered_events, filtered_violations)
            
            # Violation Analysis
            violation_analysis = self._analyze_violations(filtered_violations)
            
            # Remediation Status
            remediation_status = self._analyze_remediation_status(filtered_violations)
            
            # Risk Assessment
            risk_assessment = self._assess_compliance_risk(filtered_events, filtered_violations)
            
            # Recommendations
            recommendations = self._generate_compliance_recommendations(filtered_events, filtered_violations)
            
            # Generate visualizations
            visualizations = await self._generate_compliance_visualizations(filtered_events, filtered_violations)
            
            report = {
                "report_id": str(uuid.uuid4()),
                "report_type": "regulatory_compliance",
                "regulation": regulation,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "generated_at": datetime.now().isoformat(),
                "executive_summary": executive_summary,
                "compliance_status": compliance_status,
                "violation_analysis": violation_analysis,
                "remediation_status": remediation_status,
                "risk_assessment": risk_assessment,
                "recommendations": recommendations,
                "visualizations": visualizations,
                "metadata": {
                    "total_events": len(filtered_events),
                    "total_violations": len(filtered_violations),
                    "data_sources": ["audit_trail", "violation_tracking", "investigation_records"]
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating regulatory report: {e}")
            return {"error": str(e)}
    
    def _generate_executive_summary(self, events: List[ComplianceEvent], 
                                   violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Generate executive summary for compliance report"""
        total_events = len(events)
        total_violations = len(violations)
        
        # Compliance rate
        compliance_rate = ((total_events - total_violations) / total_events * 100) if total_events > 0 else 100
        
        # Severity breakdown
        severity_counts = defaultdict(int)
        for violation in violations:
            severity_counts[violation.severity.value] += 1
        
        # Trend analysis
        if events:
            recent_events = [e for e in events if e.timestamp >= datetime.now() - timedelta(days=7)]
            trend = "improving" if len(recent_events) < total_events / 4 else "stable"
        else:
            trend = "stable"
        
        return {
            "compliance_rate": round(compliance_rate, 2),
            "total_events_analyzed": total_events,
            "total_violations_found": total_violations,
            "severity_breakdown": dict(severity_counts),
            "compliance_trend": trend,
            "key_findings": [
                f"Overall compliance rate: {compliance_rate:.1f}%",
                f"Total violations detected: {total_violations}",
                f"Critical violations: {severity_counts.get('critical', 0)}",
                f"Compliance trend: {trend}"
            ]
        }
    
    def _calculate_compliance_status(self, events: List[ComplianceEvent], 
                                   violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Calculate detailed compliance status"""
        # Group violations by type
        violation_types = defaultdict(int)
        for violation in violations:
            violation_types[violation.violation_type] += 1
        
        # Group by severity
        severity_distribution = defaultdict(int)
        for violation in violations:
            severity_distribution[violation.severity.value] += 1
        
        # Resolution status
        resolved_violations = sum(1 for v in violations if v.resolution_timestamp is not None)
        pending_violations = len(violations) - resolved_violations
        
        # Time to resolution analysis
        resolution_times = []
        for violation in violations:
            if violation.resolution_timestamp:
                resolution_time = (violation.resolution_timestamp - violation.detection_timestamp).days
                resolution_times.append(resolution_time)
        
        avg_resolution_time = statistics.mean(resolution_times) if resolution_times else 0
        
        return {
            "violation_types": dict(violation_types),
            "severity_distribution": dict(severity_distribution),
            "resolution_statistics": {
                "resolved_violations": resolved_violations,
                "pending_violations": pending_violations,
                "average_resolution_time_days": round(avg_resolution_time, 1),
                "resolution_rate": round(resolved_violations / len(violations) * 100, 1) if violations else 100
            },
            "compliance_score": self._calculate_compliance_score(events, violations)
        }
    
    def _calculate_compliance_score(self, events: List[ComplianceEvent], 
                                  violations: List[ComplianceViolation]) -> float:
        """Calculate overall compliance score"""
        if not events:
            return 100.0
        
        # Base compliance rate
        base_score = ((len(events) - len(violations)) / len(events)) * 100
        
        # Severity penalties
        severity_penalties = {
            ComplianceSeverity.CRITICAL: 10,
            ComplianceSeverity.HIGH: 5,
            ComplianceSeverity.MEDIUM: 2,
            ComplianceSeverity.LOW: 0.5
        }
        
        penalty = 0
        for violation in violations:
            penalty += severity_penalties.get(violation.severity, 0)
        
        # Apply penalties but ensure score doesn't go below 0
        final_score = max(0, base_score - penalty)
        
        return round(final_score, 2)
    
    def _analyze_violations(self, violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Analyze compliance violations in detail"""
        if not violations:
            return {"total_violations": 0}
        
        # Temporal analysis
        violation_timeline = defaultdict(int)
        for violation in violations:
            date_key = violation.detection_timestamp.date().isoformat()
            violation_timeline[date_key] += 1
        
        # Root cause analysis
        violation_causes = defaultdict(int)
        for violation in violations:
            # Extract potential causes from description
            if "unauthorized" in violation.description.lower():
                violation_causes["unauthorized_access"] += 1
            elif "missing" in violation.description.lower():
                violation_causes["missing_documentation"] += 1
            elif "expired" in violation.description.lower():
                violation_causes["expired_credentials"] += 1
            else:
                violation_causes["other"] += 1
        
        # Impact analysis
        high_impact_violations = [v for v in violations if v.severity in [ComplianceSeverity.HIGH, ComplianceSeverity.CRITICAL]]
        
        return {
            "total_violations": len(violations),
            "violation_timeline": dict(violation_timeline),
            "root_causes": dict(violation_causes),
            "high_impact_violations": len(high_impact_violations),
            "most_common_violation_type": max(
                [v.violation_type for v in violations], 
                key=[v.violation_type for v in violations].count
            ) if violations else None,
            "average_detection_to_resolution_hours": self._calculate_avg_resolution_time(violations)
        }
    
    def _calculate_avg_resolution_time(self, violations: List[ComplianceViolation]) -> float:
        """Calculate average time from detection to resolution"""
        resolution_times = []
        for violation in violations:
            if violation.resolution_timestamp:
                hours = (violation.resolution_timestamp - violation.detection_timestamp).total_seconds() / 3600
                resolution_times.append(hours)
        
        return round(statistics.mean(resolution_times), 2) if resolution_times else 0
    
    def _analyze_remediation_status(self, violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Analyze remediation efforts and status"""
        if not violations:
            return {"total_remediations": 0}
        
        # Remediation status breakdown
        status_counts = defaultdict(int)
        for violation in violations:
            status_counts[violation.remediation_status] += 1
        
        # Overdue remediations
        overdue_count = 0
        for violation in violations:
            if (violation.remediation_deadline and 
                violation.remediation_deadline < datetime.now() and 
                violation.remediation_status != "completed"):
                overdue_count += 1
        
        # Remediation effectiveness
        completed_remediations = sum(1 for v in violations if v.remediation_status == "completed")
        effectiveness_rate = (completed_remediations / len(violations) * 100) if violations else 100
        
        return {
            "status_breakdown": dict(status_counts),
            "overdue_remediations": overdue_count,
            "effectiveness_rate": round(effectiveness_rate, 1),
            "total_remediation_actions": sum(len(v.remediation_actions) for v in violations),
            "average_actions_per_violation": round(
                sum(len(v.remediation_actions) for v in violations) / len(violations), 1
            ) if violations else 0
        }
    
    def _assess_compliance_risk(self, events: List[ComplianceEvent], 
                              violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Assess overall compliance risk"""
        # Risk factors
        risk_factors = []
        risk_score = 0
        
        # High violation rate
        if events:
            violation_rate = len(violations) / len(events)
            if violation_rate > 0.1:  # More than 10% violation rate
                risk_factors.append("High violation rate detected")
                risk_score += 30
        
        # Critical violations
        critical_violations = sum(1 for v in violations if v.severity == ComplianceSeverity.CRITICAL)
        if critical_violations > 0:
            risk_factors.append(f"{critical_violations} critical violations found")
            risk_score += critical_violations * 20
        
        # Overdue remediations
        overdue_remediations = sum(1 for v in violations 
                                 if v.remediation_deadline and 
                                 v.remediation_deadline < datetime.now() and 
                                 v.remediation_status != "completed")
        if overdue_remediations > 0:
            risk_factors.append(f"{overdue_remediations} overdue remediations")
            risk_score += overdue_remediations * 10
        
        # Risk level classification
        if risk_score >= 80:
            risk_level = "CRITICAL"
        elif risk_score >= 60:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        elif risk_score >= 20:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            "overall_risk_score": min(risk_score, 100),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "risk_mitigation_priority": "immediate" if risk_score >= 80 else "high" if risk_score >= 60 else "medium"
        }
    
    def _generate_compliance_recommendations(self, events: List[ComplianceEvent], 
                                           violations: List[ComplianceViolation]) -> List[str]:
        """Generate actionable compliance recommendations"""
        recommendations = []
        
        # Based on violation patterns
        if violations:
            violation_types = [v.violation_type for v in violations]
            most_common_type = max(violation_types, key=violation_types.count)
            recommendations.append(f"Focus on preventing {most_common_type} violations through targeted controls")
        
        # Based on resolution times
        resolution_times = [
            (v.resolution_timestamp - v.detection_timestamp).days
            for v in violations if v.resolution_timestamp
        ]
        if resolution_times and statistics.mean(resolution_times) > 7:
            recommendations.append("Improve violation resolution processes to reduce average resolution time")
        
        # Based on severity
        critical_violations = sum(1 for v in violations if v.severity == ComplianceSeverity.CRITICAL)
        if critical_violations > 0:
            recommendations.append("Implement enhanced monitoring for critical compliance areas")
        
        # Based on trends
        if events:
            recent_violations = [v for v in violations if v.detection_timestamp >= datetime.now() - timedelta(days=30)]
            if len(recent_violations) > len(violations) / 2:  # More than half are recent
                recommendations.append("Investigate recent increase in violations and implement corrective measures")
        
        # General recommendations
        recommendations.extend([
            "Conduct regular compliance training for all staff",
            "Implement automated compliance monitoring where possible",
            "Establish clear escalation procedures for critical violations",
            "Review and update compliance policies quarterly"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    async def _generate_compliance_visualizations(self, events: List[ComplianceEvent], 
                                                violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Generate compliance visualizations"""
        try:
            visualizations = {}
            
            # Violation trend chart
            if violations:
                violation_dates = [v.detection_timestamp.date() for v in violations]
                date_counts = defaultdict(int)
                for date in violation_dates:
                    date_counts[date] += 1
                
                visualizations["violation_trend"] = {
                    "type": "line_chart",
                    "data": [{"date": date.isoformat(), "count": count} 
                           for date, count in sorted(date_counts.items())],
                    "title": "Violation Trend Over Time"
                }
            
            # Severity distribution pie chart
            if violations:
                severity_counts = defaultdict(int)
                for violation in violations:
                    severity_counts[violation.severity.value] += 1
                
                visualizations["severity_distribution"] = {
                    "type": "pie_chart",
                    "data": [{"label": severity, "value": count} 
                           for severity, count in severity_counts.items()],
                    "title": "Violation Severity Distribution"
                }
            
            # Compliance score gauge
            compliance_score = self._calculate_compliance_score(events, violations)
            visualizations["compliance_gauge"] = {
                "type": "gauge",
                "value": compliance_score,
                "min": 0,
                "max": 100,
                "title": "Overall Compliance Score"
            }
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return {}

class ComplianceAuditTrailManager:
    """Main manager for comprehensive compliance audit trail system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_pool = None
        self.redis_pool = None
        
        # Initialize components
        encryption_key = self._generate_encryption_key()
        self.verifier = AuditTrailVerifier(encryption_key)
        self.analyzer = ComplianceAnalyzer()
        self.report_generator = ComplianceReportGenerator()
        
        # Performance metrics
        self.metrics = {
            "events_logged": 0,
            "violations_detected": 0,
            "investigations_opened": 0,
            "reports_generated": 0,
            "integrity_checks": 0
        }
        
        # Event processing queue
        self.event_queue = asyncio.Queue()
        self.processing_active = False
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for audit trail"""
        password = self.config.get('encryption_password', 'default_password').encode()
        salt = self.config.get('encryption_salt', 'default_salt').encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    async def initialize(self, database_url: str, redis_url: str):
        """Initialize the compliance audit trail system"""
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
            
            # Start background processing
            asyncio.create_task(self._process_event_queue())
            
            logger.info("Compliance Audit Trail Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Compliance Audit Trail Manager: {e}")
            raise
    
    async def log_compliance_event(self, event_type: ComplianceEventType, user_id: Optional[str],
                                 entity_type: str, entity_id: str, action: str,
                                 details: Dict[str, Any], regulation: Optional[str] = None,
                                 severity: ComplianceSeverity = ComplianceSeverity.INFO,
                                 risk_score: float = 0.0, **kwargs) -> str:
        """Log a comprehensive compliance event"""
        try:
            # Create event ID
            event_id = str(uuid.uuid4())
            
            # Get previous hash for chain
            previous_hash = await self._get_last_hash()
            
            # Create compliance event
            event = ComplianceEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=kwargs.get('session_id'),
                ip_address=kwargs.get('ip_address'),
                user_agent=kwargs.get('user_agent'),
                geo_location=kwargs.get('geo_location'),
                entity_type=entity_type,
                entity_id=entity_id,
                action=action,
                details=details,
                regulation=regulation,
                compliance_status=kwargs.get('compliance_status', 'compliant'),
                severity=severity,
                risk_score=risk_score,
                data_classification=kwargs.get('data_classification', 'internal'),
                retention_period=kwargs.get('retention_period', 2555),  # 7 years default
                encryption_key_id=kwargs.get('encryption_key_id'),
                hash_chain_previous=previous_hash,
                hash_chain_current="",  # Will be calculated
                digital_signature=None,  # Will be calculated
                investigation_id=kwargs.get('investigation_id'),
                tags=kwargs.get('tags', []),
                metadata=kwargs.get('metadata', {})
            )
            
            # Create hash chain
            event.hash_chain_current = self.verifier.create_hash_chain(event, previous_hash)
            
            # Create digital signature
            event.digital_signature = self.verifier.create_digital_signature(event)
            
            # Add to processing queue
            await self.event_queue.put(event)
            
            # Update metrics
            self.metrics["events_logged"] += 1
            
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging compliance event: {e}")
            raise
    
    async def detect_compliance_violation(self, event: ComplianceEvent, 
                                        violation_type: str, description: str,
                                        affected_entities: List[str]) -> str:
        """Detect and log compliance violation"""
        try:
            violation_id = str(uuid.uuid4())
            
            violation = ComplianceViolation(
                violation_id=violation_id,
                event_id=event.event_id,
                regulation=event.regulation or "unknown",
                article="",  # To be determined based on regulation
                violation_type=violation_type,
                description=description,
                severity=event.severity,
                risk_impact=self._assess_risk_impact(event.severity, affected_entities),
                affected_entities=affected_entities,
                detection_method="automated_analysis",
                detection_timestamp=datetime.now(),
                remediation_actions=[],
                remediation_deadline=None,
                remediation_status="pending",
                investigation_required=event.severity in [ComplianceSeverity.HIGH, ComplianceSeverity.CRITICAL],
                notification_sent=False,
                escalation_level=self._determine_escalation_level(event.severity),
                compliance_officer_assigned=None,
                resolution_timestamp=None,
                resolution_details=None
            )
            
            # Store violation
            await self._store_violation(violation)
            
            # Update metrics
            self.metrics["violations_detected"] += 1
            
            # Trigger investigation if required
            if violation.investigation_required:
                await self._trigger_investigation(violation)
            
            return violation_id
            
        except Exception as e:
            logger.error(f"Error detecting compliance violation: {e}")
            raise
    
    async def create_investigation(self, title: str, description: str,
                                 investigation_type: str, priority: str,
                                 assigned_investigator: str,
                                 related_events: List[str] = None,
                                 related_violations: List[str] = None) -> str:
        """Create compliance investigation"""
        try:
            investigation_id = str(uuid.uuid4())
            
            investigation = Investigation(
                investigation_id=investigation_id,
                title=title,
                description=description,
                investigation_type=investigation_type,
                priority=priority,
                status=InvestigationStatus.OPEN,
                assigned_investigator=assigned_investigator,
                created_timestamp=datetime.now(),
                updated_timestamp=datetime.now(),
                deadline=self._calculate_investigation_deadline(priority),
                related_events=related_events or [],
                related_violations=related_violations or [],
                evidence_collected=[],
                findings=[],
                conclusions=None,
                recommendations=[],
                actions_taken=[],
                timeline=[],
                stakeholders=[assigned_investigator],
                confidentiality_level="internal",
                tags=[]
            )
            
            # Store investigation
            await self._store_investigation(investigation)
            
            # Update metrics
            self.metrics["investigations_opened"] += 1
            
            return investigation_id
            
        except Exception as e:
            logger.error(f"Error creating investigation: {e}")
            raise
    
    async def generate_compliance_report(self, report_type: str, regulation: str,
                                       start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        try:
            # Retrieve relevant data
            events = await self._get_events_by_regulation(regulation, start_date, end_date)
            violations = await self._get_violations_by_regulation(regulation, start_date, end_date)
            
            # Generate report based on type
            if report_type == "regulatory_compliance":
                report = await self.report_generator.generate_regulatory_report(
                    regulation, start_date, end_date, events, violations
                )
            else:
                report = {"error": f"Unknown report type: {report_type}"}
            
            # Store report
            await self._store_report(report)
            
            # Update metrics
            self.metrics["reports_generated"] += 1
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {"error": str(e)}
    
    async def verify_audit_trail_integrity(self, start_date: Optional[datetime] = None,
                                         end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Verify integrity of audit trail"""
        try:
            # Retrieve events for verification
            events = await self._get_events_for_verification(start_date, end_date)
            
            # Perform integrity verification
            verification_result = self.verifier.verify_hash_chain(events)
            
            # Update metrics
            self.metrics["integrity_checks"] += 1
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Error verifying audit trail integrity: {e}")
            return {"error": str(e)}
    
    async def analyze_compliance_patterns(self, regulation: Optional[str] = None,
                                        days: int = 30) -> Dict[str, Any]:
        """Analyze compliance patterns and anomalies"""
        try:
            # Get recent events
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            if regulation:
                events = await self._get_events_by_regulation(regulation, start_date, end_date)
            else:
                events = await self._get_events_by_date_range(start_date, end_date)
            
            # Perform analysis
            analysis_result = self.analyzer.analyze_compliance_patterns(events)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing compliance patterns: {e}")
            return {"error": str(e)}
    
    async def _process_event_queue(self):
        """Background task to process compliance events"""
        self.processing_active = True
        
        while self.processing_active:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Store event in database
                await self._store_event(event)
                
                # Perform real-time compliance analysis
                await self._analyze_event_compliance(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event queue: {e}")
                await asyncio.sleep(1)
    
    async def _store_event(self, event: ComplianceEvent):
        """Store compliance event in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO compliance_audit_events (
                        event_id, event_type, timestamp, user_id, session_id,
                        ip_address, user_agent, geo_location, entity_type, entity_id,
                        action, details, regulation, compliance_status, severity,
                        risk_score, data_classification, retention_period,
                        hash_chain_previous, hash_chain_current, digital_signature,
                        investigation_id, tags, metadata, created_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                        $16, $17, $18, $19, $20, $21, $22, $23, $24, $25
                    )
                """, 
                    event.event_id, event.event_type.value, event.timestamp, event.user_id,
                    event.session_id, event.ip_address, event.user_agent,
                    json.dumps(event.geo_location), event.entity_type, event.entity_id,
                    event.action, json.dumps(event.details), event.regulation,
                    event.compliance_status, event.severity.value, event.risk_score,
                    event.data_classification, event.retention_period,
                    event.hash_chain_previous, event.hash_chain_current, event.digital_signature,
                    event.investigation_id, json.dumps(event.tags), json.dumps(event.metadata),
                    datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error storing compliance event: {e}")
            raise
    
    async def _analyze_event_compliance(self, event: ComplianceEvent):
        """Perform real-time compliance analysis on event"""
        try:
            # Check for compliance violations
            violations = await self._check_compliance_violations(event)
            
            for violation in violations:
                await self.detect_compliance_violation(
                    event, violation["type"], violation["description"], violation["entities"]
                )
                
        except Exception as e:
            logger.error(f"Error analyzing event compliance: {e}")
    
    async def _check_compliance_violations(self, event: ComplianceEvent) -> List[Dict[str, Any]]:
        """Check for compliance violations in event"""
        violations = []
        
        # Example compliance checks
        if event.event_type == ComplianceEventType.DATA_ACCESS:
            if event.risk_score > 0.8:
                violations.append({
                    "type": "high_risk_data_access",
                    "description": "High-risk data access detected",
                    "entities": [event.entity_id]
                })
        
        if event.event_type == ComplianceEventType.AUTHENTICATION:
            if "failed" in event.action.lower():
                violations.append({
                    "type": "authentication_failure",
                    "description": "Authentication failure detected",
                    "entities": [event.user_id] if event.user_id else []
                })
        
        return violations
    
    def _assess_risk_impact(self, severity: ComplianceSeverity, affected_entities: List[str]) -> str:
        """Assess risk impact of violation"""
        entity_count = len(affected_entities)
        
        if severity == ComplianceSeverity.CRITICAL:
            return "high" if entity_count > 10 else "medium"
        elif severity == ComplianceSeverity.HIGH:
            return "medium" if entity_count > 5 else "low"
        else:
            return "low"
    
    def _determine_escalation_level(self, severity: ComplianceSeverity) -> int:
        """Determine escalation level based on severity"""
        escalation_map = {
            ComplianceSeverity.CRITICAL: 3,
            ComplianceSeverity.HIGH: 2,
            ComplianceSeverity.MEDIUM: 1,
            ComplianceSeverity.LOW: 0,
            ComplianceSeverity.INFO: 0
        }
        return escalation_map.get(severity, 0)
    
    def _calculate_investigation_deadline(self, priority: str) -> datetime:
        """Calculate investigation deadline based on priority"""
        priority_days = {
            "critical": 1,
            "high": 3,
            "medium": 7,
            "low": 14
        }
        days = priority_days.get(priority.lower(), 7)
        return datetime.now() + timedelta(days=days)
    
    async def _trigger_investigation(self, violation: ComplianceViolation):
        """Trigger investigation for compliance violation"""
        try:
            investigation_id = await self.create_investigation(
                title=f"Investigation: {violation.violation_type}",
                description=f"Automated investigation triggered for violation: {violation.description}",
                investigation_type="compliance_violation",
                priority="high" if violation.severity == ComplianceSeverity.CRITICAL else "medium",
                assigned_investigator="system",
                related_violations=[violation.violation_id]
            )
            
            # Update violation with investigation ID
            violation.investigation_id = investigation_id
            await self._update_violation(violation)
            
        except Exception as e:
            logger.error(f"Error triggering investigation: {e}")
    
    async def _get_last_hash(self) -> Optional[str]:
        """Get the last hash in the chain"""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT hash_chain_current FROM compliance_audit_events
                    ORDER BY timestamp DESC LIMIT 1
                """)
                return result['hash_chain_current'] if result else None
                
        except Exception as e:
            logger.error(f"Error getting last hash: {e}")
            return None
    
    async def _store_violation(self, violation: ComplianceViolation):
        """Store compliance violation in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO compliance_violations (
                        violation_id, event_id, regulation, article, violation_type,
                        description, severity, risk_impact, affected_entities,
                        detection_method, detection_timestamp, remediation_actions,
                        remediation_deadline, remediation_status, investigation_required,
                        notification_sent, escalation_level, compliance_officer_assigned,
                        created_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
                    )
                """,
                    violation.violation_id, violation.event_id, violation.regulation,
                    violation.article, violation.violation_type, violation.description,
                    violation.severity.value, violation.risk_impact,
                    json.dumps(violation.affected_entities), violation.detection_method,
                    violation.detection_timestamp, json.dumps(violation.remediation_actions),
                    violation.remediation_deadline, violation.remediation_status,
                    violation.investigation_required, violation.notification_sent,
                    violation.escalation_level, violation.compliance_officer_assigned,
                    datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error storing violation: {e}")
            raise
    
    async def _store_investigation(self, investigation: Investigation):
        """Store investigation in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO compliance_investigations (
                        investigation_id, title, description, investigation_type,
                        priority, status, assigned_investigator, created_timestamp,
                        updated_timestamp, deadline, related_events, related_violations,
                        evidence_collected, findings, conclusions, recommendations,
                        actions_taken, timeline, stakeholders, confidentiality_level, tags
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21
                    )
                """,
                    investigation.investigation_id, investigation.title, investigation.description,
                    investigation.investigation_type, investigation.priority, investigation.status.value,
                    investigation.assigned_investigator, investigation.created_timestamp,
                    investigation.updated_timestamp, investigation.deadline,
                    json.dumps(investigation.related_events), json.dumps(investigation.related_violations),
                    json.dumps(investigation.evidence_collected), json.dumps(investigation.findings),
                    investigation.conclusions, json.dumps(investigation.recommendations),
                    json.dumps(investigation.actions_taken), json.dumps(investigation.timeline),
                    json.dumps(investigation.stakeholders), investigation.confidentiality_level,
                    json.dumps(investigation.tags)
                )
                
        except Exception as e:
            logger.error(f"Error storing investigation: {e}")
            raise
    
    async def _store_report(self, report: Dict[str, Any]):
        """Store compliance report in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO compliance_reports (
                        report_id, report_type, regulation, period_start, period_end,
                        generated_at, report_data, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    report.get("report_id"), report.get("report_type"),
                    report.get("regulation"), 
                    datetime.fromisoformat(report["period"]["start_date"]) if "period" in report else None,
                    datetime.fromisoformat(report["period"]["end_date"]) if "period" in report else None,
                    datetime.fromisoformat(report.get("generated_at", datetime.now().isoformat())),
                    json.dumps(report), datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error storing report: {e}")
    
    async def _get_events_by_regulation(self, regulation: str, start_date: datetime, 
                                      end_date: datetime) -> List[ComplianceEvent]:
        """Get compliance events by regulation and date range"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM compliance_audit_events
                    WHERE regulation = $1 AND timestamp BETWEEN $2 AND $3
                    ORDER BY timestamp
                """, regulation, start_date, end_date)
                
                return [self._row_to_event(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting events by regulation: {e}")
            return []
    
    async def _get_violations_by_regulation(self, regulation: str, start_date: datetime,
                                          end_date: datetime) -> List[ComplianceViolation]:
        """Get compliance violations by regulation and date range"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM compliance_violations
                    WHERE regulation = $1 AND detection_timestamp BETWEEN $2 AND $3
                    ORDER BY detection_timestamp
                """, regulation, start_date, end_date)
                
                return [self._row_to_violation(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting violations by regulation: {e}")
            return []
    
    def _row_to_event(self, row) -> ComplianceEvent:
        """Convert database row to ComplianceEvent"""
        return ComplianceEvent(
            event_id=row['event_id'],
            event_type=ComplianceEventType(row['event_type']),
            timestamp=row['timestamp'],
            user_id=row['user_id'],
            session_id=row['session_id'],
            ip_address=row['ip_address'],
            user_agent=row['user_agent'],
            geo_location=json.loads(row['geo_location']) if row['geo_location'] else None,
            entity_type=row['entity_type'],
            entity_id=row['entity_id'],
            action=row['action'],
            details=json.loads(row['details']),
            regulation=row['regulation'],
            compliance_status=row['compliance_status'],
            severity=ComplianceSeverity(row['severity']),
            risk_score=row['risk_score'],
            data_classification=row['data_classification'],
            retention_period=row['retention_period'],
            encryption_key_id=row.get('encryption_key_id'),
            hash_chain_previous=row['hash_chain_previous'],
            hash_chain_current=row['hash_chain_current'],
            digital_signature=row['digital_signature'],
            investigation_id=row['investigation_id'],
            tags=json.loads(row['tags']),
            metadata=json.loads(row['metadata'])
        )
    
    def _row_to_violation(self, row) -> ComplianceViolation:
        """Convert database row to ComplianceViolation"""
        return ComplianceViolation(
            violation_id=row['violation_id'],
            event_id=row['event_id'],
            regulation=row['regulation'],
            article=row['article'],
            violation_type=row['violation_type'],
            description=row['description'],
            severity=ComplianceSeverity(row['severity']),
            risk_impact=row['risk_impact'],
            affected_entities=json.loads(row['affected_entities']),
            detection_method=row['detection_method'],
            detection_timestamp=row['detection_timestamp'],
            remediation_actions=json.loads(row['remediation_actions']),
            remediation_deadline=row['remediation_deadline'],
            remediation_status=row['remediation_status'],
            investigation_required=row['investigation_required'],
            notification_sent=row['notification_sent'],
            escalation_level=row['escalation_level'],
            compliance_officer_assigned=row['compliance_officer_assigned'],
            resolution_timestamp=row['resolution_timestamp'],
            resolution_details=row['resolution_details']
        )
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return {
            **self.metrics,
            "queue_size": self.event_queue.qsize(),
            "processing_active": self.processing_active
        }
    
    async def close(self):
        """Clean up resources"""
        self.processing_active = False
        
        if self.db_pool:
            await self.db_pool.close()
        
        if self.redis_pool:
            await self.redis_pool.close()

# Example usage
async def main():
    """Example usage of Compliance Audit Trail System"""
    config = {
        'encryption_password': 'secure_password_2025',
        'encryption_salt': 'compliance_salt_2025'
    }
    
    manager = ComplianceAuditTrailManager(config)
    
    # In real usage, provide actual database and Redis URLs
    # await manager.initialize("postgresql://user:pass@localhost/db", "redis://localhost:6379")
    
    # Example compliance event logging
    # event_id = await manager.log_compliance_event(
    #     ComplianceEventType.DATA_ACCESS,
    #     user_id="user123",
    #     entity_type="customer_data",
    #     entity_id="customer456",
    #     action="view_sensitive_data",
    #     details={"fields_accessed": ["bsn", "income"], "purpose": "mortgage_assessment"},
    #     regulation="GDPR",
    #     severity=ComplianceSeverity.MEDIUM,
    #     risk_score=0.6,
    #     ip_address="192.168.1.100",
    #     tags=["sensitive_data", "mortgage_process"]
    # )
    
    print("Compliance Audit Trail System demo completed!")

if __name__ == "__main__":
    asyncio.run(main())