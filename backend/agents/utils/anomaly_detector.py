"""
Anomaly Detection Engine for Mortgage Applications

This module detects statistical anomalies and outliers in mortgage application data
using rule-based and statistical methods to identify potential fraud or errors.
"""

import logging
import statistics
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import asyncio

from ..config import settings


class AnomalyDetector:
    """
    Advanced anomaly detection for mortgage applications.

    Detects:
    - Statistical outliers in financial ratios
    - Unusual patterns in application data
    - Potential fraud indicators
    - Data inconsistencies
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Statistical thresholds
        self.outlier_threshold = settings.ANOMALY_DETECTION_THRESHOLD
        self.z_score_threshold = 3.0  # Standard deviations

        # Domain-specific thresholds
        self.domain_thresholds = {
            'ltv_ratio': {'min': 0.1, 'max': 0.95, 'typical_range': (0.5, 0.85)},
            'dti_ratio': {'min': 0.1, 'max': 0.45, 'typical_range': (0.25, 0.40)},
            'credit_score': {'min': 300, 'max': 900, 'typical_range': (600, 850)},
            'loan_term': {'min': 5, 'max': 40, 'typical_range': (15, 30)},
            'income': {'min': 15000, 'max': 500000, 'typical_range': (30000, 150000)},
            'mortgage_amount': {'min': 50000, 'max': 2000000, 'typical_range': (150000, 500000)},
        }

        # Known fraud patterns
        self.fraud_indicators = {
            'round_numbers': [10000, 25000, 50000, 100000, 250000, 500000],
            'unusual_ratios': {
                'ltv_extreme_high': 0.95,
                'dti_extreme_high': 0.60,
                'income_to_loan_unusual': 0.5,  # Income much lower than loan
            }
        }

    async def detect_anomalies(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in mortgage application data.

        Args:
            application_data: Complete application data dictionary

        Returns:
            Dictionary containing detected anomalies and severity scores
        """
        try:
            anomalies = []

            # Individual field anomaly detection
            field_anomalies = await self._detect_field_anomalies(application_data)
            anomalies.extend(field_anomalies)

            # Cross-field relationship anomalies
            relationship_anomalies = await self._detect_relationship_anomalies(application_data)
            anomalies.extend(relationship_anomalies)

            # Fraud pattern detection
            fraud_anomalies = await self._detect_fraud_patterns(application_data)
            anomalies.extend(fraud_anomalies)

            # Temporal anomalies
            temporal_anomalies = await self._detect_temporal_anomalies(application_data)
            anomalies.extend(temporal_anomalies)

            # Calculate overall severity
            severity_score = self._calculate_severity_score(anomalies)

            result = {
                'anomalies': anomalies,
                'severity_score': severity_score,
                'risk_level': self._classify_risk_level(severity_score),
                'total_anomalies': len(anomalies),
                'critical_count': len([a for a in anomalies if a.get('severity') == 'critical']),
                'high_count': len([a for a in anomalies if a.get('severity') == 'high']),
                'medium_count': len([a for a in anomalies if a.get('severity') == 'medium']),
                'detected_at': datetime.utcnow().isoformat()
            }

            self.logger.info(f"Detected {len(anomalies)} anomalies with severity score {severity_score:.2f}")
            return result

        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            return {
                'anomalies': [],
                'severity_score': 0,
                'error': str(e)
            }

    async def _detect_field_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in individual fields."""
        anomalies = []

        for field, value in data.items():
            if field in self.domain_thresholds and value is not None:
                threshold = self.domain_thresholds[field]

                # Check absolute bounds
                if value < threshold['min'] or value > threshold['max']:
                    anomalies.append({
                        'type': 'out_of_bounds',
                        'field': field,
                        'value': value,
                        'threshold': threshold,
                        'description': f"{field} value {value} is outside acceptable range [{threshold['min']}, {threshold['max']}]",
                        'severity': 'high' if abs(value) > threshold['max'] * 1.5 else 'medium'
                    })

                # Check if value is in typical range
                typical_min, typical_max = threshold['typical_range']
                if value < typical_min or value > typical_max:
                    anomalies.append({
                        'type': 'unusual_value',
                        'field': field,
                        'value': value,
                        'typical_range': threshold['typical_range'],
                        'description': f"{field} value {value} is outside typical range [{typical_min}, {typical_max}]",
                        'severity': 'low'
                    })

                # Check for round number fraud indicators
                if field in ['income', 'mortgage_amount'] and value in self.fraud_indicators['round_numbers']:
                    anomalies.append({
                        'type': 'round_number_suspicion',
                        'field': field,
                        'value': value,
                        'description': f"{field} uses a round number {value} which may indicate fabricated data",
                        'severity': 'medium'
                    })

        return anomalies

    async def _detect_relationship_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in relationships between fields."""
        anomalies = []

        # Loan-to-value ratio anomalies
        mortgage_amount = data.get('mortgage_amount')
        property_value = data.get('property_value')

        if mortgage_amount and property_value and property_value > 0:
            ltv = mortgage_amount / property_value

            if ltv > self.fraud_indicators['unusual_ratios']['ltv_extreme_high']:
                anomalies.append({
                    'type': 'extreme_ltv',
                    'field': 'ltv_ratio',
                    'value': ltv,
                    'description': f"LTV ratio {ltv:.1%} is extremely high, approaching maximum lending limits",
                    'severity': 'high'
                })

        # Debt-to-income ratio anomalies
        income = data.get('income')
        if mortgage_amount and income and income > 0:
            # Approximate DTI calculation
            term = data.get('loan_term', 25)
            monthly_payment = mortgage_amount / (term * 12)
            dti = (monthly_payment * 12) / income

            if dti > self.fraud_indicators['unusual_ratios']['dti_extreme_high']:
                anomalies.append({
                    'type': 'extreme_dti',
                    'field': 'dti_ratio',
                    'value': dti,
                    'description': f"DTI ratio {dti:.1%} is extremely high, may indicate affordability issues",
                    'severity': 'critical'
                })

        # Income to loan amount relationship
        if income and mortgage_amount:
            income_to_loan_ratio = income / mortgage_amount

            if income_to_loan_ratio < self.fraud_indicators['unusual_ratios']['income_to_loan_unusual']:
                anomalies.append({
                    'type': 'income_loan_mismatch',
                    'field': 'income_loan_ratio',
                    'value': income_to_loan_ratio,
                    'description': f"Income to loan ratio {income_to_loan_ratio:.2f} is unusually low",
                    'severity': 'high'
                })

        # Age and loan term relationship
        dob = data.get('date_of_birth')
        term = data.get('loan_term')

        if dob and term:
            try:
                if isinstance(dob, str):
                    dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
                elif isinstance(dob, date):
                    dob_date = dob
                else:
                    dob_date = None

                if dob_date:
                    current_age = date.today().year - dob_date.year
                    age_at_end = current_age + term

                    if age_at_end > 80:  # Very old age
                        anomalies.append({
                            'type': 'age_term_concern',
                            'field': 'age_term_relationship',
                            'value': age_at_end,
                            'description': f"Applicant will be {age_at_end} years old at loan end, which may be concerning",
                            'severity': 'medium'
                        })
            except (ValueError, TypeError):
                pass

        return anomalies

    async def _detect_fraud_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect potential fraud patterns in the application."""
        anomalies = []

        # Check for identical values across different fields
        numeric_fields = ['mortgage_amount', 'property_value', 'income']
        values = [data.get(field) for field in numeric_fields if data.get(field) is not None]

        if len(values) > 1 and len(set(values)) == 1:
            anomalies.append({
                'type': 'identical_values',
                'field': 'multiple_fields',
                'value': values[0],
                'description': f"Multiple financial fields have identical values {values[0]}, which may indicate data fabrication",
                'severity': 'critical'
            })

        # Check for suspicious address patterns
        address = data.get('address', '')
        if address:
            # Check for PO Box or virtual addresses
            suspicious_patterns = ['po box', 'virtual', 'mailbox', 'p.o.', 'postbus']
            if any(pattern in address.lower() for pattern in suspicious_patterns):
                anomalies.append({
                    'type': 'suspicious_address',
                    'field': 'address',
                    'value': address,
                    'description': "Address appears to be a PO Box or virtual address",
                    'severity': 'high'
                })

        # Check for unrealistic income levels
        income = data.get('income')
        mortgage_amount = data.get('mortgage_amount')

        if income and mortgage_amount:
            # Income should be reasonable relative to loan amount
            if income > mortgage_amount * 5:  # Income much higher than loan
                anomalies.append({
                    'type': 'income_disproportionate',
                    'field': 'income_mortgage_ratio',
                    'value': income / mortgage_amount,
                    'description': f"Income is {income / mortgage_amount:.1f}x the mortgage amount, which seems disproportionate",
                    'severity': 'medium'
                })

        return anomalies

    async def _detect_temporal_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect temporal anomalies (time-based patterns)."""
        anomalies = []

        # Check document expiry dates
        expiry_date = data.get('expiry_date')
        if expiry_date:
            try:
                if isinstance(expiry_date, str):
                    expiry = datetime.strptime(expiry_date, '%Y-%m-%d').date()
                elif isinstance(expiry_date, date):
                    expiry = expiry_date
                else:
                    expiry = None

                if expiry:
                    days_to_expiry = (expiry - date.today()).days

                    if days_to_expiry < 30:  # Expires soon
                        anomalies.append({
                            'type': 'document_expiring_soon',
                            'field': 'expiry_date',
                            'value': expiry_date,
                            'description': f"Document expires in {days_to_expiry} days, which is very soon",
                            'severity': 'high'
                        })
                    elif days_to_expiry < 0:  # Already expired
                        anomalies.append({
                            'type': 'document_expired',
                            'field': 'expiry_date',
                            'value': expiry_date,
                            'description': "Document has already expired",
                            'severity': 'critical'
                        })
            except (ValueError, TypeError):
                pass

        # Check application timing patterns
        created_at = data.get('created_at')
        submitted_at = data.get('submitted_at')

        if created_at and submitted_at:
            try:
                if isinstance(created_at, str):
                    created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    created = created_at

                if isinstance(submitted_at, str):
                    submitted = datetime.fromisoformat(submitted_at.replace('Z', '+00:00'))
                else:
                    submitted = submitted_at

                processing_time = (submitted - created).total_seconds() / 3600  # hours

                if processing_time < 0.1:  # Submitted too quickly
                    anomalies.append({
                        'type': 'too_fast_submission',
                        'field': 'processing_time',
                        'value': processing_time,
                        'description': f"Application was submitted after only {processing_time:.2f} hours, which seems rushed",
                        'severity': 'low'
                    })
            except (ValueError, TypeError):
                pass

        return anomalies

    def _calculate_severity_score(self, anomalies: List[Dict[str, Any]]) -> float:
        """Calculate overall severity score from anomalies."""
        if not anomalies:
            return 0.0

        severity_weights = {
            'critical': 1.0,
            'high': 0.7,
            'medium': 0.4,
            'low': 0.1
        }

        total_weight = 0.0
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'medium')
            weight = severity_weights.get(severity, 0.4)
            total_weight += weight

        # Normalize to 0-100 scale
        max_possible_score = len(anomalies) * 1.0  # Maximum weight per anomaly
        normalized_score = (total_weight / max_possible_score) * 100 if max_possible_score > 0 else 0

        return round(normalized_score, 2)

    def _classify_risk_level(self, severity_score: float) -> str:
        """Classify overall risk level based on severity score."""
        if severity_score >= 70:
            return 'critical'
        elif severity_score >= 50:
            return 'high'
        elif severity_score >= 30:
            return 'medium'
        elif severity_score >= 10:
            return 'low'
        else:
            return 'minimal'
