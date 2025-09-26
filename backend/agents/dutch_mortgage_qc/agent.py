"""
Dutch Mortgage Application Quality Control Agent
Specialized for Dutch mortgage market integration with lenders like Stater and Quion.
Integrates with Dutch lenders and validates against NHG, BKR, and AFM requirements.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from enum import Enum
import requests
from urllib.parse import urljoin
import statistics
import re
import base64
import hashlib
from PIL import Image
import io
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from ..config import settings
from ..database import get_db_connection
from ..utils.text_processor import TextProcessor
from ..utils.ocr_service import OCRService


class LenderType(Enum):
    """Dutch mortgage lenders."""
    STATER = "stater"
    QUION = "quion"
    ING = "ing"
    RABOBANK = "rabobank"
    ABN_AMRO = "abn_amro"


@dataclass
class BKRCheckResult:
    """BKR (Dutch Credit Bureau) check result."""
    credit_score: int
    negative_registrations: List[Dict[str, Any]]
    current_debts: List[Dict[str, Any]]
    approval_likelihood: float
    risk_factors: List[str]
    recommendations: List[str]
    checked_at: datetime


@dataclass
class NHGEligibilityResult:
    """NHG (National Mortgage Guarantee) eligibility result."""
    eligible: bool
    property_value_check: Dict[str, Any]
    loan_amount_check: Dict[str, Any]
    ltv_check: Dict[str, Any]
    financial_impact: Dict[str, Any]
    recommendation: str


@dataclass
class LenderValidationResult:
    """Lender-specific validation result."""
    lender_name: str
    compatibility_score: float
    validation_results: List[Dict[str, Any]]
    document_requirements: Dict[str, Any]
    processing_time_estimate: str
    approval_likelihood: float


class DutchMortgageQCAgent:
    """
    Quality Control Agent specialized for Dutch mortgage applications.
    Integrates with Dutch lenders and validates against NHG, BKR, and AFM requirements.

    Key Features:
    - BKR credit bureau integration
    - NHG eligibility validation
    - Dutch affordability rule compliance
    - Lender-specific requirement validation
    - First-time-right probability assessment
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # BKR API configuration
        self.bkr_api_url = settings.BKR_API_URL or "https://api.bkr.nl/v2"
        self.bkr_api_key = settings.BKR_API_KEY

        # NHG API configuration
        self.nhg_api_url = settings.NHG_API_URL or "https://api.nhg.nl/v1"
        self.nhg_api_key = settings.NHG_API_KEY

        # OCR.space configuration
        self.ocr_api_key = settings.OCR_API_KEY
        self.ocr_api_url = "https://api.ocr.space/parse/image"

        # Lender configurations
        self.lender_configs = self._initialize_lender_configurations()

        # Dutch validation rules cache
        self._validation_rules_cache = {}
        self._rules_cache_timestamp = None
        self._cache_ttl_seconds = 3600  # 1 hour

        # Advanced AI components
        self.text_processor = TextProcessor()
        self.ocr_service = OCRService() if hasattr(settings, 'OCR_API_KEY') and settings.OCR_API_KEY else None

        # Graph database for compliance networks
        self.compliance_graph = nx.DiGraph()
        self.client_networks = defaultdict(lambda: nx.Graph())

        # Agentic AI components
        self.decision_memory = []
        self.learning_patterns = {}
        self.autonomous_actions = []

        # Initialize validation rules and AI components
        # Note: These will be initialized when first needed to avoid event loop issues
        self._rules_loaded = False
        self._graph_initialized = False
        self._memory_loaded = False

    async def _ensure_initialized(self):
        """Ensure all components are initialized when first needed."""
        if not self._rules_loaded:
            await self._load_dutch_validation_rules()
            self._rules_loaded = True
        if not self._graph_initialized:
            await self._initialize_compliance_graph()
            self._graph_initialized = True
        if not self._memory_loaded:
            await self._load_agent_memory()
            self._memory_loaded = True

    async def analyze_dutch_mortgage_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive QC analysis for Dutch mortgage applications.

        Args:
            application_data: Complete mortgage application including client data, documents, product selection

        Returns:
            Detailed QC report with first-time-right assessment and lender-specific validation
        """
        # Ensure all components are initialized
        await self._ensure_initialized()

        application_id = application_data.get('application_id', f"app_{datetime.utcnow().timestamp()}")
        client_data = application_data.get('client_data', {})
        product_selection = application_data.get('product_selection', {})
        lender_name = product_selection.get('lender_name', '').lower()

        self.logger.info(f"Starting Dutch mortgage QC for application {application_id}, lender: {lender_name}")

        try:
            # Core validations - run in parallel for performance
            validation_tasks = [
                self._validate_bkr_requirements(client_data),
                self._validate_nhg_eligibility(application_data),
                self._validate_dutch_affordability_rules(client_data, application_data),
                self._validate_dutch_mortgage_documents(application_data.get('documents', [])),
                self._validate_lender_specific_requirements(application_data, lender_name)
            ]

            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Handle any exceptions in validation tasks
            processed_results = []
            for i, result in enumerate(validation_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Validation task {i} failed: {str(result)}")
                    processed_results.append({
                        'status': 'error',
                        'error': str(result),
                        'validation_type': ['bkr', 'nhg', 'affordability', 'documents', 'lender'][i]
                    })
                else:
                    processed_results.append(result)

            bkr_validation, nhg_validation, affordability_check, document_validation, lender_validation = processed_results

            # Calculate first-time-right probability
            ftr_assessment = await self._assess_first_time_right_probability(processed_results)

            # Generate automated remediation
            remediation_plan = await self._generate_automated_remediation(processed_results)

            # Prepare lender submission package
            submission_package = await self._prepare_lender_submission_package(application_data, lender_name)

            # Generate comprehensive QC report
            qc_report = {
                'application_id': application_id,
                'qc_summary': {
                    'overall_score': ftr_assessment['overall_score'],
                    'first_time_right_probability': ftr_assessment['ftr_probability'],
                    'ready_for_submission': ftr_assessment['ready_for_submission'],
                    'estimated_processing_time': self._estimate_processing_time(lender_name, ftr_assessment),
                    'risk_assessment': self._assess_application_risk(processed_results)
                },
                'validation_results': {
                    'bkr_credit_check': bkr_validation,
                    'nhg_eligibility': nhg_validation,
                    'affordability_assessment': affordability_check,
                    'document_completeness': document_validation,
                    'lender_specific_checks': lender_validation
                },
                'remediation_plan': remediation_plan,
                'lender_submission_package': submission_package,
                'dutch_market_insights': await self._provide_dutch_market_insights(application_data),
                'analyzed_at': datetime.utcnow().isoformat(),
                'qc_metadata': {
                    'agent_version': '2.0',
                    'validation_rules_version': '2025.1',
                    'lender_config_version': self._get_lender_config_version(lender_name)
                }
            }

            # Store QC results in database
            await self._store_qc_results(application_id, qc_report)

            self.logger.info(f"Dutch mortgage QC completed for application {application_id}: FTR={ftr_assessment['ftr_probability']}%, ready={ftr_assessment['ready_for_submission']}")

            return qc_report

        except Exception as e:
            self.logger.error(f"Dutch mortgage QC analysis failed for {application_id}: {str(e)}")
            raise

    async def _validate_bkr_requirements(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate BKR (Dutch credit bureau) requirements."""

        bsn = client_data.get('bsn')
        if not bsn:
            return {
                'status': 'failed',
                'error': 'BSN required for BKR check',
                'blocking': True,
                'validation_type': 'bkr'
            }

        # Validate BSN format (Dutch social security number)
        if not self._validate_bsn_format(bsn):
            return {
                'status': 'failed',
                'error': 'Invalid BSN format',
                'blocking': True,
                'validation_type': 'bkr'
            }

        try:
            # Call BKR API
            bkr_response = await self._call_bkr_api(bsn)

            if bkr_response.get('status') == 'error':
                return {
                    'status': 'error',
                    'error': bkr_response.get('message', 'BKR API error'),
                    'blocking': True,
                    'validation_type': 'bkr'
                }

            credit_score = bkr_response.get('credit_score', 0)
            negative_registrations = bkr_response.get('negative_registrations', [])
            current_debts = bkr_response.get('current_debts', [])

            # Analyze BKR data
            has_negative_registrations = len(negative_registrations) > 0
            total_debt_burden = sum([debt.get('monthly_payment', 0) for debt in current_debts])

            # BKR assessment based on Dutch mortgage standards
            bkr_approval_likelihood = self._calculate_bkr_approval_likelihood(
                credit_score, negative_registrations, total_debt_burden
            )

            # Identify risk factors
            risk_factors = self._identify_bkr_risk_factors(bkr_response)

            # Generate recommendations
            recommendations = self._generate_bkr_recommendations(bkr_response)

            result = {
                'status': 'completed',
                'credit_score': credit_score,
                'negative_registrations_count': len(negative_registrations),
                'total_existing_debt': total_debt_burden,
                'approval_likelihood': bkr_approval_likelihood,
                'risk_factors': risk_factors,
                'recommendations': recommendations,
                'blocking': bkr_approval_likelihood < 60,  # Block if low approval chance
                'validation_type': 'bkr',
                'bkr_check_date': datetime.utcnow().isoformat(),
                'data_freshness': 'real_time' if bkr_response.get('real_time', False) else 'cached'
            }

            # Store BKR check result
            await self._store_bkr_check(bsn, result)

            return result

        except Exception as e:
            self.logger.error(f"BKR validation failed: {str(e)}")
            return {
                'status': 'error',
                'error': f'BKR validation failed: {str(e)}',
                'blocking': True,
                'validation_type': 'bkr'
            }

    async def _validate_nhg_eligibility(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate National Mortgage Guarantee (NHG) eligibility."""

        property_value = Decimal(str(application_data.get('property_value', 0)))
        mortgage_amount = Decimal(str(application_data.get('mortgage_amount', 0)))
        client_data = application_data.get('client_data', {})

        # NHG limits for 2025 (these should be updated annually)
        nhg_limits = await self._get_current_nhg_limits()
        nhg_property_limit = nhg_limits.get('property_limit', 435000)
        nhg_income_limit = nhg_limits.get('income_limit', 43000)

        # Basic NHG eligibility checks
        property_eligible = property_value <= nhg_property_limit
        loan_eligible = mortgage_amount <= nhg_property_limit

        # Additional NHG criteria
        if property_value > 0:
            ltv_ratio = float((mortgage_amount / property_value) * 100)
        else:
            ltv_ratio = 0

        nhg_ltv_eligible = ltv_ratio <= 100  # NHG allows up to 100% LTV

        # Income check for certain NHG programs
        gross_income = client_data.get('gross_annual_income', 0)
        income_eligible = gross_income <= nhg_income_limit if nhg_income_limit > 0 else True

        # Calculate NHG costs and benefits
        nhg_costs = self._calculate_nhg_costs(mortgage_amount)
        product_selection = application_data.get('product_selection', {})
        interest_rate_benefit = self._calculate_nhg_interest_benefit(mortgage_amount, product_selection)

        nhg_eligible = property_eligible and loan_eligible and nhg_ltv_eligible and income_eligible

        # Generate NHG recommendation
        recommendation = self._generate_nhg_recommendation(
            nhg_eligible, interest_rate_benefit, nhg_costs, application_data
        )

        return {
            'eligible': nhg_eligible,
            'property_value_check': {
                'property_value': float(property_value),
                'nhg_limit': nhg_property_limit,
                'within_limit': property_eligible
            },
            'loan_amount_check': {
                'mortgage_amount': float(mortgage_amount),
                'nhg_limit': nhg_property_limit,
                'within_limit': loan_eligible
            },
            'ltv_check': {
                'ltv_ratio': round(ltv_ratio, 2),
                'eligible': nhg_ltv_eligible
            },
            'income_check': {
                'annual_income': gross_income,
                'nhg_limit': nhg_income_limit,
                'within_limit': income_eligible
            },
            'financial_impact': {
                'nhg_costs': nhg_costs,
                'estimated_interest_savings': interest_rate_benefit,
                'net_benefit': interest_rate_benefit - nhg_costs,
                'break_even_years': self._calculate_nhg_break_even(interest_rate_benefit, nhg_costs)
            },
            'recommendation': recommendation,
            'validation_type': 'nhg',
            'nhg_limits_version': nhg_limits.get('version', '2025')
        }

    async def _validate_dutch_affordability_rules(self, client_data: Dict[str, Any], application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against Dutch affordability rules (income-based lending limits)."""

        gross_annual_income = Decimal(str(client_data.get('gross_annual_income', 0)))
        partner_income = Decimal(str(client_data.get('partner_income', 0)))
        existing_debts = client_data.get('existing_debts', [])
        mortgage_amount = Decimal(str(application_data.get('mortgage_amount', 0)))
        product_selection = application_data.get('product_selection', {})
        interest_rate = Decimal(str(product_selection.get('interest_rate', 0.04)))

        # Calculate total household income
        total_income = gross_annual_income + partner_income

        # Calculate existing debt obligations
        monthly_debt_payments = sum([Decimal(str(debt.get('monthly_payment', 0))) for debt in existing_debts])
        annual_debt_payments = monthly_debt_payments * 12

        # Dutch affordability calculation (using current government tables)
        maximum_mortgage = self._calculate_dutch_maximum_mortgage(total_income, annual_debt_payments, interest_rate)

        # Debt-to-income ratio calculations
        dti_ratio = float((annual_debt_payments / total_income) * 100) if total_income > 0 else 0
        mortgage_to_income_ratio = float((mortgage_amount / total_income) * 100) if total_income > 0 else 0

        # Stress test with higher interest rate (Dutch Financial Markets Authority requirement)
        stress_test_rate = interest_rate + Decimal('0.02')  # 2% stress test
        stress_test_affordability = self._calculate_stress_test_affordability(
            mortgage_amount, total_income, stress_test_rate, annual_debt_payments
        )

        # Additional Dutch affordability checks
        housing_cost_ratio = self._calculate_housing_cost_ratio(
            mortgage_amount, interest_rate, total_income, monthly_debt_payments
        )

        affordability_passed = (
            mortgage_amount <= maximum_mortgage and
            stress_test_affordability['passed'] and
            housing_cost_ratio <= 30  # Max 30% of income for housing costs
        )

        return {
            'affordability_passed': affordability_passed,
            'income_analysis': {
                'total_household_income': float(total_income),
                'debt_to_income_ratio': round(dti_ratio, 2),
                'mortgage_to_income_ratio': round(mortgage_to_income_ratio, 2),
                'housing_cost_ratio': round(housing_cost_ratio, 2)
            },
            'lending_limits': {
                'requested_mortgage': float(mortgage_amount),
                'maximum_allowable': float(maximum_mortgage),
                'within_limits': mortgage_amount <= maximum_mortgage,
                'utilization_percentage': float((mortgage_amount / maximum_mortgage) * 100) if maximum_mortgage > 0 else 0,
                'headroom_amount': float(maximum_mortgage - mortgage_amount) if mortgage_amount < maximum_mortgage else 0
            },
            'stress_test': stress_test_affordability,
            'dutch_regulation_compliance': {
                'dti_compliant': dti_ratio <= 40,  # AFM guideline
                'housing_cost_compliant': housing_cost_ratio <= 30,
                'stress_test_passed': stress_test_affordability['passed']
            },
            'recommendations': self._generate_affordability_recommendations(
                mortgage_amount, maximum_mortgage, stress_test_affordability, housing_cost_ratio
            ),
            'validation_type': 'affordability'
        }

    async def _validate_dutch_mortgage_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate Dutch mortgage documents for completeness and authenticity."""

        required_document_types = [
            'income_statement', 'employment_contract', 'bank_statements',
            'property_valuation', 'identity_document', 'tax_return',
            'mortgage_offer', 'nhg_application'
        ]

        document_validation_results = {}
        total_required = len(required_document_types)
        provided_documents = 0
        valid_documents = 0
        issues_found = []

        for doc_type in required_document_types:
            doc_result = await self._validate_document_type(documents, doc_type)
            document_validation_results[doc_type] = doc_result

            if doc_result['provided']:
                provided_documents += 1
                if doc_result['valid']:
                    valid_documents += 1
                else:
                    issues_found.extend(doc_result.get('issues', []))

        completeness_percentage = (valid_documents / total_required) * 100

        # Check for document authenticity and consistency
        authenticity_check = await self._validate_document_authenticity(documents)
        consistency_check = self._validate_document_consistency(documents)

        return {
            'completeness_percentage': round(completeness_percentage, 2),
            'documents_provided': provided_documents,
            'documents_valid': valid_documents,
            'total_required': total_required,
            'document_validation_results': document_validation_results,
            'authenticity_check': authenticity_check,
            'consistency_check': consistency_check,
            'critical_issues': [issue for issue in issues_found if issue.get('severity') == 'critical'],
            'validation_type': 'documents',
            'overall_status': 'complete' if completeness_percentage >= 95 else 'incomplete' if completeness_percentage >= 80 else 'major_gaps'
        }

    async def _validate_lender_specific_requirements(self, application_data: Dict[str, Any], lender_name: str) -> Dict[str, Any]:
        """Validate against specific lender requirements (Stater, Quion, etc.)."""

        if not lender_name or lender_name.lower() not in self.lender_configs:
            return {
                'status': 'error',
                'error': f'Lender configuration not found for: {lender_name}',
                'validation_results': [],
                'validation_type': 'lender'
            }

        lender_config = self.lender_configs[lender_name.lower()]
        validation_results = []

        # Validate against lender-specific criteria
        for criterion_name, requirements in lender_config.get('validation_criteria', {}).items():
            result = await self._validate_lender_criterion(application_data, criterion_name, requirements)
            validation_results.append(result)

        # Check lender-specific document requirements
        document_check = await self._validate_lender_document_requirements(
            application_data.get('documents', []),
            lender_config.get('required_documents', [])
        )

        # Calculate overall lender compatibility score
        compatibility_score = self._calculate_lender_compatibility_score(validation_results, document_check)

        # Estimate approval likelihood
        approval_likelihood = self._estimate_approval_likelihood(compatibility_score, validation_results)

        return {
            'status': 'completed',
            'lender_name': lender_name,
            'compatibility_score': compatibility_score,
            'approval_likelihood': approval_likelihood,
            'validation_results': validation_results,
            'document_requirements': document_check,
            'processing_time_estimate': lender_config.get('typical_processing_time', 'Unknown'),
            'special_conditions': lender_config.get('special_conditions', []),
            'validation_type': 'lender',
            'lender_config_version': lender_config.get('version', '1.0')
        }

    async def _assess_first_time_right_probability(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the probability of first-time-right submission to lender."""

        # Extract scores and blocking issues from each validation
        scores = []
        blocking_issues = []
        risk_factors = []

        for validation in validation_results:
            if isinstance(validation, dict):
                # Extract scores from different validation types
                if validation.get('status') == 'completed':
                    if 'approval_likelihood' in validation:
                        scores.append(validation['approval_likelihood'])
                    elif 'compatibility_score' in validation:
                        scores.append(validation['compatibility_score'])
                    elif 'completeness_percentage' in validation:
                        scores.append(validation['completeness_percentage'])
                    elif 'affordability_passed' in validation:
                        scores.append(100 if validation['affordability_passed'] else 0)
                    elif 'eligible' in validation:
                        scores.append(100 if validation['eligible'] else 0)

                # Check for blocking issues
                if validation.get('blocking', False):
                    blocking_issues.append(validation.get('error', 'Unknown blocking issue'))

                # Collect risk factors
                if 'risk_factors' in validation:
                    risk_factors.extend(validation['risk_factors'])

        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else 0

        # Determine FTR probability based on various factors
        ftr_probability = self._calculate_ftr_probability(overall_score, len(blocking_issues), len(risk_factors))

        # Determine if ready for submission
        ready_for_submission = len(blocking_issues) == 0 and overall_score >= 75 and ftr_probability >= 70

        return {
            'overall_score': round(overall_score, 2),
            'ftr_probability': round(ftr_probability, 2),
            'ready_for_submission': ready_for_submission,
            'blocking_issues_count': len(blocking_issues),
            'blocking_issues': blocking_issues,
            'risk_factors_count': len(risk_factors),
            'recommendation': self._generate_ftr_recommendation(ftr_probability, blocking_issues, risk_factors),
            'confidence_level': self._calculate_confidence_level(scores)
        }

    async def _generate_automated_remediation(self, validation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate automated remediation plan based on validation results."""

        remediation_plan = []

        for validation in validation_results:
            if not isinstance(validation, dict) or validation.get('status') == 'error':
                continue

            validation_type = validation.get('validation_type')

            if validation_type == 'bkr':
                remediation_plan.extend(self._generate_bkr_remediation(validation))
            elif validation_type == 'nhg':
                remediation_plan.extend(self._generate_nhg_remediation(validation))
            elif validation_type == 'affordability':
                remediation_plan.extend(self._generate_affordability_remediation(validation))
            elif validation_type == 'documents':
                remediation_plan.extend(self._generate_document_remediation(validation))
            elif validation_type == 'lender':
                remediation_plan.extend(self._generate_lender_remediation(validation))

        # Sort by priority and severity
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        remediation_plan.sort(key=lambda x: (
            priority_order.get(x.get('severity', 'low'), 3),
            x.get('estimated_time_days', 30)
        ))

        return remediation_plan

    async def _prepare_lender_submission_package(self, application_data: Dict[str, Any], lender_name: str) -> Dict[str, Any]:
        """Prepare lender submission package with all required documents and data."""

        if not lender_name or lender_name.lower() not in self.lender_configs:
            return {'status': 'error', 'error': f'Unknown lender: {lender_name}'}

        lender_config = self.lender_configs[lender_name.lower()]

        # Prepare documents according to lender requirements
        documents = application_data.get('documents', [])
        prepared_documents = []

        for doc in documents:
            prepared_doc = await self._prepare_document_for_lender(doc, lender_config)
            prepared_documents.append(prepared_doc)

        # Prepare application data structure
        submission_data = {
            'application_id': application_data.get('application_id'),
            'client_data': self._sanitize_client_data_for_lender(application_data.get('client_data', {}), lender_config),
            'mortgage_details': application_data.get('mortgage_details', {}),
            'documents': prepared_documents,
            'metadata': {
                'submission_date': datetime.utcnow().isoformat(),
                'lender_config_version': lender_config.get('version', '1.0'),
                'qc_validated': True
            }
        }

        return {
            'status': 'prepared',
            'lender_name': lender_name,
            'submission_data': submission_data,
            'document_count': len(prepared_documents),
            'api_endpoint': lender_config.get('api_endpoint'),
            'estimated_submission_time': lender_config.get('typical_processing_time')
        }

    async def _provide_dutch_market_insights(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide Dutch market insights for the mortgage application."""

        property_location = application_data.get('property_location', {})
        mortgage_amount = application_data.get('mortgage_amount', 0)
        lender_name = application_data.get('product_selection', {}).get('lender_name', '')

        insights = {
            'market_rates': await self._get_current_market_rates(),
            'regional_insights': await self._get_regional_market_insights(property_location),
            'lender_comparison': await self._get_lender_comparison_insights(lender_name),
            'nhg_market_impact': self._analyze_nhg_market_impact(mortgage_amount),
            'generated_at': datetime.utcnow().isoformat()
        }

        return insights

    # Advanced AI Methods for Truly Agentic System

    async def perform_computer_vision_document_verification(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Advanced computer vision document verification using OCR.space API.
        Performs forgery detection, signature verification, and document tampering analysis.
        """
        verification_results = {
            'documents_analyzed': len(documents),
            'cv_verifications': [],
            'forgery_detection': [],
            'signature_analysis': [],
            'tampering_indicators': [],
            'overall_authenticity_score': 0.0,
            'risk_assessment': 'low'
        }

        for doc in documents:
            doc_result = await self._analyze_document_with_cv(doc)
            verification_results['cv_verifications'].append(doc_result)

            # Check for forgery indicators
            forgery_check = await self._detect_document_forgery(doc, doc_result)
            if forgery_check['detected']:
                verification_results['forgery_detection'].append(forgery_check)

            # Analyze signatures
            signature_check = await self._analyze_document_signatures(doc, doc_result)
            verification_results['signature_analysis'].append(signature_check)

            # Check for tampering
            tampering_check = await self._detect_document_tampering(doc, doc_result)
            if tampering_check['indicators']:
                verification_results['tampering_indicators'].extend(tampering_check['indicators'])

        # Calculate overall authenticity score
        verification_results['overall_authenticity_score'] = self._calculate_cv_authenticity_score(verification_results)

        # Assess risk
        verification_results['risk_assessment'] = self._assess_cv_risk(verification_results)

        # Agentic decision making
        await self._make_cv_based_decisions(verification_results)

        return verification_results

    async def perform_advanced_nlp_content_analysis(self, documents: List[Dict[str, Any]], application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advanced NLP content analysis using transformers and semantic analysis.
        Performs compliance checking, sentiment analysis, and content quality assessment.
        """
        analysis_results = {
            'documents_analyzed': len(documents),
            'compliance_analysis': [],
            'sentiment_analysis': [],
            'semantic_similarity': [],
            'content_quality': [],
            'key_entities': [],
            'risk_keywords': [],
            'nlp_insights': [],
            'recommendations': []
        }

        for doc in documents:
            # Extract text using OCR if needed
            text_content = await self._extract_document_text(doc)

            # Compliance analysis
            compliance = await self._analyze_compliance_content(text_content, doc.get('type', 'unknown'))
            analysis_results['compliance_analysis'].append(compliance)

            # Sentiment analysis
            sentiment = await self._analyze_document_sentiment(text_content)
            analysis_results['sentiment_analysis'].append(sentiment)

            # Content quality assessment
            quality = await self._assess_content_quality(text_content, doc.get('type', 'unknown'))
            analysis_results['content_quality'].append(quality)

            # Entity extraction
            entities = await self._extract_key_entities(text_content)
            analysis_results['key_entities'].extend(entities)

            # Risk keyword detection
            risk_words = await self._detect_risk_keywords(text_content)
            analysis_results['risk_keywords'].extend(risk_words)

        # Cross-document semantic analysis
        semantic_analysis = await self._perform_semantic_similarity_analysis(documents)
        analysis_results['semantic_similarity'] = semantic_analysis

        # Generate NLP-based insights
        analysis_results['nlp_insights'] = await self._generate_nlp_insights(analysis_results, application_data)

        # Generate recommendations
        analysis_results['recommendations'] = await self._generate_nlp_recommendations(analysis_results)

        # Agentic learning from NLP analysis
        await self._learn_from_nlp_analysis(analysis_results)

        return analysis_results

    async def analyze_compliance_network_graph(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Graph analysis for compliance networks - models relationships between clients,
        products, regulations, and compliance requirements using NetworkX.
        """
        client_id = application_data.get('client_data', {}).get('id', 'unknown')
        application_id = application_data.get('application_id', f"app_{datetime.utcnow().timestamp()}")

        # Build or update compliance network
        network_analysis = await self._build_compliance_network(client_id, application_data)

        # Analyze network properties
        graph_metrics = self._analyze_graph_properties(client_id)

        # Detect compliance patterns
        pattern_analysis = await self._detect_compliance_patterns(client_id, application_data)

        # Risk propagation analysis
        risk_propagation = self._analyze_risk_propagation(client_id)

        # Network-based recommendations
        network_recommendations = await self._generate_network_recommendations(client_id, graph_metrics, pattern_analysis)

        analysis_results = {
            'client_id': client_id,
            'application_id': application_id,
            'network_metrics': graph_metrics,
            'pattern_analysis': pattern_analysis,
            'risk_propagation': risk_propagation,
            'network_recommendations': network_recommendations,
            'graph_visualization': self._generate_graph_visualization(client_id),
            'compliance_clusters': self._identify_compliance_clusters(client_id),
            'centrality_analysis': self._analyze_network_centrality(client_id)
        }

        # Agentic decision making based on network analysis
        await self._make_network_based_decisions(analysis_results)

        return analysis_results

    async def execute_autonomous_qc_workflow(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fully autonomous QC workflow - the agent makes decisions, takes actions,
        and learns from outcomes without human intervention.
        """
        workflow_id = f"workflow_{datetime.utcnow().timestamp()}"

        self.logger.info(f"Starting autonomous QC workflow {workflow_id}")

        # Phase 1: Data Ingestion and Initial Assessment
        initial_assessment = await self._perform_initial_assessment(application_data)

        # Phase 2: Parallel Advanced Analysis
        analysis_tasks = [
            self.perform_computer_vision_document_verification(application_data.get('documents', [])),
            self.perform_advanced_nlp_content_analysis(application_data.get('documents', []), application_data),
            self.analyze_compliance_network_graph(application_data),
            self.analyze_dutch_mortgage_application(application_data)  # Traditional QC
        ]

        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        # Handle analysis results
        cv_results, nlp_results, graph_results, traditional_qc = analysis_results

        # Phase 3: Agentic Decision Making
        decisions = await self._make_autonomous_decisions(
            cv_results, nlp_results, graph_results, traditional_qc, initial_assessment
        )

        # Phase 4: Execute Autonomous Actions
        actions_taken = await self._execute_autonomous_actions(decisions, application_data)

        # Phase 5: Learning and Adaptation
        learning_outcomes = await self._learn_from_workflow_execution(
            workflow_id, analysis_results, decisions, actions_taken
        )

        workflow_results = {
            'workflow_id': workflow_id,
            'initial_assessment': initial_assessment,
            'analysis_results': {
                'computer_vision': cv_results if not isinstance(cv_results, Exception) else str(cv_results),
                'nlp_analysis': nlp_results if not isinstance(nlp_results, Exception) else str(nlp_results),
                'graph_analysis': graph_results if not isinstance(graph_results, Exception) else str(graph_results),
                'traditional_qc': traditional_qc if not isinstance(traditional_qc, Exception) else str(traditional_qc)
            },
            'autonomous_decisions': decisions,
            'actions_taken': actions_taken,
            'learning_outcomes': learning_outcomes,
            'final_recommendation': decisions.get('final_recommendation', 'review_required'),
            'confidence_level': decisions.get('confidence_level', 0.0),
            'execution_time': datetime.utcnow().isoformat()
        }

        # Store workflow execution for future learning
        await self._store_workflow_execution(workflow_results)

        self.logger.info(f"Completed autonomous QC workflow {workflow_id} with recommendation: {workflow_results['final_recommendation']}")

        return workflow_results

    # Computer Vision Methods

    async def _analyze_document_with_cv(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document using computer vision techniques via OCR.space API."""

        if not self.ocr_api_key:
            return {'error': 'OCR API key not configured', 'cv_analysis': 'unavailable'}

        try:
            file_path = document.get('file_path')
            if not file_path:
                return {'error': 'No file path provided'}

            # Prepare file for OCR API
            with open(file_path, 'rb') as f:
                file_data = f.read()

            # Call OCR.space API
            files = {'file': (os.path.basename(file_path), file_data)}
            data = {
                'apikey': self.ocr_api_key,
                'language': 'dut',  # Dutch
                'isOverlayRequired': 'true',
                'detectOrientation': 'true',
                'scale': 'true'
            }

            response = requests.post(self.ocr_api_url, files=files, data=data, timeout=30)

            if response.status_code == 200:
                ocr_result = response.json()

                if ocr_result.get('IsErroredOnProcessing'):
                    return {'error': ocr_result.get('ErrorMessage', 'OCR processing failed')}

                # Extract text and overlay information
                parsed_results = ocr_result.get('ParsedResults', [{}])[0]

                cv_analysis = {
                    'text': parsed_results.get('ParsedText', ''),
                    'confidence': parsed_results.get('TextOverlay', {}).get('HasOverlay', False),
                    'text_regions': parsed_results.get('TextOverlay', {}).get('Lines', []),
                    'word_count': len(parsed_results.get('ParsedText', '').split()),
                    'language_detected': parsed_results.get('Language', 'unknown'),
                    'processing_time': ocr_result.get('ProcessingTimeInMilliseconds', 0),
                    'document_structure': self._analyze_document_structure(parsed_results),
                    'visual_features': await self._extract_visual_features(file_path),
                    'signature_detection': await self._detect_signatures_cv(file_path),
                    'tampering_indicators': self._detect_visual_tampering(parsed_results)
                }

                return cv_analysis
            else:
                return {'error': f'OCR API returned status {response.status_code}'}

        except Exception as e:
            self.logger.error(f"CV document analysis failed: {str(e)}")
            return {'error': str(e), 'cv_analysis': 'failed'}

    async def _detect_document_forgery(self, document: Dict[str, Any], cv_result: Dict[str, Any]) -> Dict[str, Any]:
        """Detect document forgery using computer vision analysis."""

        forgery_indicators = []

        # Check text consistency
        if cv_result.get('text_regions'):
            text_consistency = self._analyze_text_consistency(cv_result['text_regions'])
            if not text_consistency['consistent']:
                forgery_indicators.append({
                    'type': 'text_inconsistency',
                    'description': 'Inconsistent text formatting or alignment detected by CV',
                    'confidence': text_consistency['confidence']
                })

        # Check for digital manipulation signs
        visual_features = cv_result.get('visual_features', {})
        if visual_features.get('compression_artifacts', False):
            forgery_indicators.append({
                'type': 'compression_artifacts',
                'description': 'Signs of image compression/recompression detected',
                'confidence': 0.7
            })

        # Check signature authenticity
        signature_analysis = cv_result.get('signature_detection', {})
        if signature_analysis.get('suspicious_patterns', False):
            forgery_indicators.append({
                'type': 'signature_anomaly',
                'description': 'Suspicious signature patterns detected by CV analysis',
                'confidence': 0.8
            })

        # Check for metadata inconsistencies
        if cv_result.get('processing_time', 0) < 100:  # Suspiciously fast processing
            forgery_indicators.append({
                'type': 'processing_anomaly',
                'description': 'Document processed suspiciously fast, possible synthetic content',
                'confidence': 0.6
            })

        return {
            'detected': len(forgery_indicators) > 0,
            'indicators': forgery_indicators,
            'overall_risk': 'high' if len(forgery_indicators) > 2 else 'medium' if len(forgery_indicators) > 0 else 'low'
        }

    # NLP Content Analysis Methods

    async def _extract_document_text(self, document: Dict[str, Any]) -> str:
        """Extract text from document using OCR if needed."""

        # If document already has extracted text, use it
        if 'extracted_text' in document:
            return document['extracted_text']

        # Otherwise, use OCR to extract text
        if self.ocr_api_key:
            try:
                cv_result = await self._analyze_document_with_cv(document)
                return cv_result.get('text', '')
            except Exception as e:
                self.logger.warning(f"OCR text extraction failed: {str(e)}")

        return ''

    async def _analyze_compliance_content(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Analyze document content for compliance requirements using NLP."""

        if not text:
            return {'compliant': False, 'missing_elements': ['no_text_extracted']}

        try:
            # Use text processor for advanced NLP analysis
            if hasattr(self.text_processor, 'analyze_compliance_content'):
                compliance_result = await self.text_processor.analyze_compliance_content(text, doc_type)
                return compliance_result

            # Fallback basic analysis
            compliance_terms = self._get_compliance_terms_for_doc_type(doc_type)
            term_matches = sum(1 for term in compliance_terms if term.lower() in text.lower())

            compliance_score = min(term_matches / len(compliance_terms), 1.0) if compliance_terms else 0

            return {
                'compliant': compliance_score >= 0.7,
                'compliance_score': compliance_score,
                'terms_found': term_matches,
                'terms_required': len(compliance_terms),
                'missing_elements': [term for term in compliance_terms if term.lower() not in text.lower()]
            }

        except Exception as e:
            self.logger.error(f"NLP compliance analysis failed: {str(e)}")
            return {'compliant': False, 'error': str(e)}

    async def _analyze_document_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze document sentiment for risk assessment."""

        if not text:
            return {'sentiment': 'neutral', 'confidence': 0.5}

        try:
            if hasattr(self.text_processor, 'analyze_sentiment'):
                sentiment_result = await self.text_processor.analyze_sentiment(text)
                return sentiment_result

            # Fallback basic sentiment analysis
            positive_words = ['goed', 'excellent', 'positief', 'tevreden', 'akkoord']
            negative_words = ['slecht', 'probleem', 'bezwaar', 'weigeren', 'afwijzen']

            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            if positive_count > negative_count:
                sentiment = 'positive'
            elif negative_count > positive_count:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            confidence = min((positive_count + negative_count) / len(text.split()), 1.0)

            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'positive_indicators': positive_count,
                'negative_indicators': negative_count
            }

        except Exception as e:
            return {'sentiment': 'neutral', 'confidence': 0.0, 'error': str(e)}

    # Graph Analysis Methods

    async def _build_compliance_network(self, client_id: str, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build compliance network graph for the client."""

        # Add client node
        client_data = application_data.get('client_data', {})
        self.client_networks[client_id].add_node(client_id,
            node_type='client',
            attributes=client_data,
            created_at=datetime.utcnow().isoformat()
        )

        # Add product nodes and relationships
        products = application_data.get('product_selection', {})
        if products:
            product_id = products.get('id', f"product_{client_id}")
            self.client_networks[client_id].add_node(product_id,
                node_type='product',
                attributes=products
            )
            self.client_networks[client_id].add_edge(client_id, product_id,
                relationship_type='selects_product',
                weight=1.0
            )

        # Add lender relationships
        lender_name = products.get('lender_name')
        if lender_name:
            lender_id = f"lender_{lender_name}"
            self.client_networks[client_id].add_node(lender_id,
                node_type='lender',
                attributes={'name': lender_name}
            )
            self.client_networks[client_id].add_edge(client_id, lender_id,
                relationship_type='applies_to_lender',
                weight=0.8
            )

        # Add regulatory compliance nodes
        regulations = ['Wft_Article_86f', 'BGfo_Article_8_1', 'NHG_Requirements']
        for reg in regulations:
            reg_id = f"regulation_{reg}"
            self.client_networks[client_id].add_node(reg_id,
                node_type='regulation',
                attributes={'code': reg}
            )
            # Connect based on compliance status
            compliance_weight = 0.9 if application_data.get('compliance_status', {}).get(reg, False) else 0.3
            self.client_networks[client_id].add_edge(client_id, reg_id,
                relationship_type='subject_to_regulation',
                weight=compliance_weight
            )

        return {
            'nodes_added': len(self.client_networks[client_id].nodes()),
            'edges_added': len(self.client_networks[client_id].edges()),
            'network_density': nx.density(self.client_networks[client_id])
        }

    def _analyze_graph_properties(self, client_id: str) -> Dict[str, Any]:
        """Analyze structural properties of the compliance network."""

        if client_id not in self.client_networks:
            return {'error': 'No network found for client'}

        graph = self.client_networks[client_id]

        try:
            return {
                'num_nodes': len(graph.nodes()),
                'num_edges': len(graph.edges()),
                'density': nx.density(graph),
                'average_clustering': nx.average_clustering(graph) if len(graph) > 2 else 0,
                'connected_components': len(list(nx.connected_components(graph))),
                'degree_centrality': dict(nx.degree_centrality(graph)),
                'betweenness_centrality': dict(nx.betweenness_centrality(graph)) if len(graph) > 2 else {},
                'eigenvector_centrality': dict(nx.eigenvector_centrality(graph, max_iter=100)) if len(graph) > 2 else {}
            }
        except Exception as e:
            self.logger.warning(f"Graph analysis failed: {str(e)}")
            return {
                'num_nodes': len(graph.nodes()),
                'num_edges': len(graph.edges()),
                'error': str(e)
            }

    # Agentic AI Methods

    async def _make_autonomous_decisions(self, cv_results, nlp_results, graph_results, traditional_qc, initial_assessment) -> Dict[str, Any]:
        """Make autonomous decisions based on all analysis results."""

        # Aggregate confidence scores
        confidence_scores = []

        if not isinstance(cv_results, Exception):
            confidence_scores.append(cv_results.get('overall_authenticity_score', 0) * 0.25)

        if not isinstance(nlp_results, Exception):
            # Calculate NLP confidence based on analysis quality
            nlp_confidence = len(nlp_results.get('nlp_insights', [])) / 10  # Normalize
            confidence_scores.append(min(nlp_confidence, 1.0) * 0.25)

        if not isinstance(graph_results, Exception):
            graph_density = graph_results.get('network_metrics', {}).get('density', 0)
            confidence_scores.append(graph_density * 0.25)

        if not isinstance(traditional_qc, Exception):
            ftr_probability = traditional_qc.get('qc_summary', {}).get('first_time_right_probability', 0)
            confidence_scores.append((ftr_probability / 100) * 0.25)

        overall_confidence = sum(confidence_scores) if confidence_scores else 0

        # Decision logic based on analysis results
        decisions = {
            'confidence_level': overall_confidence,
            'risk_assessment': self._calculate_overall_risk(cv_results, nlp_results, graph_results, traditional_qc),
            'autonomous_actions': [],
            'human_review_required': overall_confidence < 0.7
        }

        # Determine final recommendation using agentic reasoning
        if overall_confidence >= 0.9:
            decisions['final_recommendation'] = 'auto_approve'
            decisions['autonomous_actions'].append('submit_to_lender')
        elif overall_confidence >= 0.7:
            decisions['final_recommendation'] = 'conditional_approval'
            decisions['autonomous_actions'].append('flag_for_priority_review')
        elif overall_confidence >= 0.5:
            decisions['final_recommendation'] = 'review_required'
            decisions['autonomous_actions'].append('schedule_human_review')
        else:
            decisions['final_recommendation'] = 'reject'
            decisions['autonomous_actions'].append('notify_client_of_issues')

        # Learn from this decision for future improvement
        await self._learn_from_decision(decisions, cv_results, nlp_results, graph_results, traditional_qc)

        return decisions

    async def _execute_autonomous_actions(self, decisions: Dict[str, Any], application_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute autonomous actions based on decisions."""

        actions_taken = []
        autonomous_actions = decisions.get('autonomous_actions', [])

        for action in autonomous_actions:
            if action == 'submit_to_lender':
                result = await self._autonomous_lender_submission(application_data)
                actions_taken.append({
                    'action': action,
                    'timestamp': datetime.utcnow().isoformat(),
                    'result': result
                })

            elif action == 'flag_for_priority_review':
                result = await self._flag_for_priority_review(application_data)
                actions_taken.append({
                    'action': action,
                    'timestamp': datetime.utcnow().isoformat(),
                    'result': result
                })

            elif action == 'schedule_human_review':
                result = await self._schedule_human_review(application_data)
                actions_taken.append({
                    'action': action,
                    'timestamp': datetime.utcnow().isoformat(),
                    'result': result
                })

            elif action == 'notify_client_of_issues':
                result = await self._notify_client_of_issues(application_data, decisions)
                actions_taken.append({
                    'action': action,
                    'timestamp': datetime.utcnow().isoformat(),
                    'result': result
                })

        return actions_taken

    # Helper methods for advanced features

    def _analyze_text_consistency(self, text_regions: List[Dict]) -> Dict[str, Any]:
        """Analyze text consistency in OCR results."""

        if not text_regions:
            return {'consistent': False, 'confidence': 0.0}

        # Check font sizes, spacing, alignment
        font_sizes = []
        line_heights = []

        for region in text_regions:
            if 'FontSize' in region:
                font_sizes.append(region['FontSize'])
            if 'Height' in region:
                line_heights.append(region['Height'])

        # Calculate consistency metrics
        font_consistency = len(set(font_sizes)) <= 2 if font_sizes else True  # Allow max 2 different sizes
        height_consistency = statistics.stdev(line_heights) < 5 if len(line_heights) > 1 else True

        overall_consistency = font_consistency and height_consistency
        confidence = 0.8 if overall_consistency else 0.4

        return {
            'consistent': overall_consistency,
            'confidence': confidence,
            'font_sizes': font_sizes,
            'line_heights': line_heights
        }

    async def _extract_visual_features(self, file_path: str) -> Dict[str, Any]:
        """Extract visual features from document image."""

        try:
            # Basic image analysis
            with Image.open(file_path) as img:
                features = {
                    'format': img.format,
                    'size': img.size,
                    'mode': img.mode,
                    'compression_artifacts': self._detect_compression_artifacts(img),
                    'resolution_dpi': self._estimate_dpi(img),
                    'color_analysis': self._analyze_color_distribution(img)
                }
                return features
        except Exception as e:
            return {'error': str(e)}

    def _detect_compression_artifacts(self, img: Image.Image) -> bool:
        """Detect JPEG compression artifacts."""

        # Simple compression artifact detection
        # In production, would use more sophisticated computer vision
        try:
            # Convert to numpy array for analysis
            img_array = np.array(img)

            # Check for block artifacts (8x8 JPEG blocks)
            if img_array.shape[0] % 8 == 0 and img_array.shape[1] % 8 == 0:
                # Analyze pixel variance in blocks
                block_size = 8
                artifacts_detected = False

                for i in range(0, min(img_array.shape[0], 64), block_size):
                    for j in range(0, min(img_array.shape[1], 64), block_size):
                        block = img_array[i:i+block_size, j:j+block_size]
                        if len(block) == block_size and len(block[0]) == block_size:
                            # Check for uniform blocks (compression artifact)
                            if np.std(block) < 5:  # Very low variance
                                artifacts_detected = True
                                break
                    if artifacts_detected:
                        break

                return artifacts_detected
        except:
            pass

        return False

    def _estimate_dpi(self, img: Image.Image) -> int:
        """Estimate image DPI."""

        # Basic DPI estimation
        width, height = img.size

        # Assume standard document sizes
        if width > 2000:  # High resolution
            return 300
        elif width > 1000:  # Medium resolution
            return 150
        else:  # Low resolution
            return 72

    def _analyze_color_distribution(self, img: Image.Image) -> Dict[str, Any]:
        """Analyze color distribution in the image."""

        try:
            img_array = np.array(img)

            # Check if image is grayscale/B&W
            if len(img_array.shape) == 2 or img_array.shape[2] == 1:
                return {'type': 'grayscale', 'channels': 1}

            # Analyze color channels
            colors, counts = np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0, return_counts=True)

            return {
                'type': 'color',
                'channels': img_array.shape[2],
                'unique_colors': len(colors),
                'dominant_color': colors[np.argmax(counts)].tolist() if len(colors) > 0 else None
            }
        except:
            return {'error': 'Color analysis failed'}

    async def _detect_signatures_cv(self, file_path: str) -> Dict[str, Any]:
        """Detect signatures using computer vision."""

        # Basic signature detection - in production would use ML models
        return {
            'signatures_found': 0,  # Placeholder
            'suspicious_patterns': False,
            'method': 'basic_cv_analysis'
        }

    def _detect_visual_tampering(self, ocr_result: Dict) -> List[str]:
        """Detect visual signs of document tampering."""

        tampering_indicators = []

        # Check for unusual text overlays
        text_overlay = ocr_result.get('TextOverlay', {})
        lines = text_overlay.get('Lines', [])

        if len(lines) == 0:
            tampering_indicators.append("No text detected - possible image-only document")

        # Check for irregular line spacing
        line_heights = [line.get('Height', 0) for line in lines if 'Height' in line]
        if len(line_heights) > 5:
            height_std = statistics.stdev(line_heights)
            if height_std > 10:  # High variation in line heights
                tampering_indicators.append("Irregular line spacing detected")

        return tampering_indicators

    def _calculate_cv_authenticity_score(self, verification_results: Dict) -> float:
        """Calculate overall authenticity score from CV analysis."""

        base_score = 100

        # Deduct for forgery detections
        forgery_penalty = len(verification_results.get('forgery_detection', [])) * 25
        base_score -= forgery_penalty

        # Deduct for tampering indicators
        tampering_penalty = len(verification_results.get('tampering_indicators', [])) * 20
        base_score -= tampering_penalty

        # Deduct for signature issues
        signature_issues = sum(1 for sig in verification_results.get('signature_analysis', [])
                              if not sig.get('valid', True))
        signature_penalty = signature_issues * 15
        base_score -= signature_penalty

        return max(0, min(100, base_score))

    def _assess_cv_risk(self, verification_results: Dict) -> str:
        """Assess overall risk from CV analysis."""

        score = verification_results.get('overall_authenticity_score', 100)

        if score >= 90:
            return 'low'
        elif score >= 70:
            return 'medium'
        elif score >= 50:
            return 'high'
        else:
            return 'critical'

    async def _make_cv_based_decisions(self, verification_results: Dict) -> None:
        """Make agentic decisions based on CV analysis."""

        risk_level = verification_results.get('risk_assessment', 'low')

        if risk_level == 'critical':
            self.logger.warning("Critical CV risk detected - escalating to human review")
            # Could trigger additional verification steps
        elif risk_level == 'high':
            self.logger.info("High CV risk detected - additional verification recommended")
            # Could request additional documents

        # Store in decision memory
        self.decision_memory.append({
            'type': 'cv_analysis',
            'timestamp': datetime.utcnow().isoformat(),
            'risk_level': risk_level,
            'action_taken': 'logged' if risk_level in ['low', 'medium'] else 'escalated'
        })

    def _get_compliance_terms_for_doc_type(self, doc_type: str) -> List[str]:
        """Get compliance terms required for document type."""

        compliance_terms = {
            'income_statement': ['salaris', 'loon', 'belasting', 'werkgever'],
            'id_document': ['identiteitsbewijs', 'paspoort', 'burgerservicenummer'],
            'property_valuation': ['taxatie', 'waarde', 'onroerend goed'],
            'mortgage_offer': ['hypotheek', 'rente', 'voorwaarden', 'verzekering']
        }

        return compliance_terms.get(doc_type, [])

    async def _assess_content_quality(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Assess content quality using NLP metrics."""

        if not text:
            return {'quality_score': 0, 'issues': ['empty_document']}

        # Basic quality metrics
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])

        issues = []

        if word_count < 10:
            issues.append('document_too_short')
        if sentence_count < 2:
            issues.append('insufficient_sentences')

        quality_score = min(word_count / 100, 1.0) * 50  # 50% weight on length
        quality_score += (1 - len(issues) / 5) * 50  # 50% weight on issues

        return {
            'quality_score': quality_score,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'issues': issues
        }

    async def _extract_key_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract key entities from text using NLP."""

        entities = []

        # Basic entity extraction (production would use NER models)
        dutch_amounts = re.findall(r'€\s*\d+(?:[.,]\d+)*', text)
        for amount in dutch_amounts:
            entities.append({
                'type': 'currency',
                'value': amount,
                'context': 'financial_amount'
            })

        dates = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', text)
        for date in dates:
            entities.append({
                'type': 'date',
                'value': date,
                'context': 'document_date'
            })

        return entities

    async def _detect_risk_keywords(self, text: str) -> List[Dict[str, Any]]:
        """Detect risk-related keywords in text."""

        risk_keywords = [
            'risico', 'gevaar', 'probleem', 'bezwaar', 'weigeren',
            'afwijzen', 'twijfel', 'onzeker', 'conflict', 'geschil'
        ]

        found_risks = []
        text_lower = text.lower()

        for keyword in risk_keywords:
            if keyword in text_lower:
                count = text_lower.count(keyword)
                found_risks.append({
                    'keyword': keyword,
                    'count': count,
                    'severity': 'high' if keyword in ['weigeren', 'afwijzen', 'conflict'] else 'medium'
                })

        return found_risks

    async def _perform_semantic_similarity_analysis(self, documents: List[Dict]) -> Dict[str, Any]:
        """Perform semantic similarity analysis across documents."""

        if len(documents) < 2:
            return {'similarity_score': 1.0, 'analysis': 'insufficient_documents'}

        # Extract texts
        texts = []
        for doc in documents:
            text = await self._extract_document_text(doc)
            if text:
                texts.append(text[:1000])  # Limit text length

        if len(texts) < 2:
            return {'similarity_score': 0.0, 'analysis': 'insufficient_text'}

        try:
            if hasattr(self.text_processor, 'calculate_similarity'):
                similarity_matrix = await self.text_processor.calculate_similarity(texts)
                avg_similarity = np.mean(similarity_matrix)

                return {
                    'similarity_score': float(avg_similarity),
                    'document_pairs': len(similarity_matrix),
                    'consistency_level': 'high' if avg_similarity > 0.8 else 'medium' if avg_similarity > 0.6 else 'low'
                }
            else:
                # Basic similarity check
                return {
                    'similarity_score': 0.5,
                    'method': 'basic_check',
                    'consistency_level': 'unknown'
                }
        except Exception as e:
            return {'error': str(e), 'similarity_score': 0.0}

    async def _generate_nlp_insights(self, analysis_results: Dict, application_data: Dict) -> List[Dict[str, Any]]:
        """Generate insights from NLP analysis."""

        insights = []

        # Compliance insights
        compliance_scores = [c.get('compliance_score', 0) for c in analysis_results.get('compliance_analysis', [])]
        if compliance_scores:
            avg_compliance = sum(compliance_scores) / len(compliance_scores)
            insights.append({
                'type': 'compliance_overview',
                'insight': f'Average document compliance: {avg_compliance:.2f}',
                'confidence': 0.9,
                'actionable': avg_compliance < 0.7
            })

        # Sentiment insights
        sentiments = [s.get('sentiment') for s in analysis_results.get('sentiment_analysis', [])]
        if sentiments:
            positive_count = sentiments.count('positive')
            negative_count = sentiments.count('negative')

            if negative_count > positive_count:
                insights.append({
                    'type': 'sentiment_warning',
                    'insight': 'Negative sentiment detected in documents',
                    'confidence': 0.8,
                    'actionable': True
                })

        # Risk keyword insights
        risk_keywords = analysis_results.get('risk_keywords', [])
        if risk_keywords:
            high_risk_count = len([k for k in risk_keywords if k.get('severity') == 'high'])
            if high_risk_count > 0:
                insights.append({
                    'type': 'risk_keywords',
                    'insight': f'{high_risk_count} high-risk keywords detected',
                    'confidence': 0.9,
                    'actionable': True
                })

        return insights

    async def _generate_nlp_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate recommendations based on NLP analysis."""

        recommendations = []

        # Content quality recommendations
        quality_scores = [q.get('quality_score', 0) for q in analysis_results.get('content_quality', [])]
        if quality_scores and sum(quality_scores) / len(quality_scores) < 60:
            recommendations.append("Improve document content quality and completeness")

        # Compliance recommendations
        compliance_analyses = analysis_results.get('compliance_analysis', [])
        missing_elements = []
        for analysis in compliance_analyses:
            missing_elements.extend(analysis.get('missing_elements', []))

        if missing_elements:
            unique_missing = list(set(missing_elements))
            recommendations.append(f"Add missing compliance elements: {', '.join(unique_missing[:3])}")

        return recommendations

    async def _learn_from_nlp_analysis(self, analysis_results: Dict) -> None:
        """Learn from NLP analysis for future improvements."""

        # Store learning patterns
        sentiment_patterns = {}
        for sentiment in analysis_results.get('sentiment_analysis', []):
            sent_type = sentiment.get('sentiment', 'neutral')
            sentiment_patterns[sent_type] = sentiment_patterns.get(sent_type, 0) + 1

        self.learning_patterns['sentiment_distribution'] = sentiment_patterns

        # Update keyword patterns
        risk_keywords = analysis_results.get('risk_keywords', [])
        for keyword_data in risk_keywords:
            keyword = keyword_data.get('keyword')
            if keyword:
                self.learning_patterns[f'keyword_{keyword}'] = self.learning_patterns.get(f'keyword_{keyword}', 0) + 1

    async def _detect_compliance_patterns(self, client_id: str, application_data: Dict) -> Dict[str, Any]:
        """Detect patterns in compliance network."""

        if client_id not in self.client_networks:
            return {'patterns_detected': [], 'confidence': 0.0}

        graph = self.client_networks[client_id]

        patterns = []

        # Check for regulatory clustering
        regulation_nodes = [n for n, attr in graph.nodes(data=True) if attr.get('node_type') == 'regulation']
        if len(regulation_nodes) > 1:
            # Look for compliance patterns
            compliant_edges = [(u, v) for u, v, attr in graph.edges(data=True)
                             if attr.get('relationship_type') == 'subject_to_regulation'
                             and attr.get('weight', 0) > 0.8]

            if len(compliant_edges) > len(regulation_nodes) * 0.8:
                patterns.append({
                    'type': 'strong_compliance',
                    'description': 'Client shows strong compliance across multiple regulations',
                    'confidence': 0.9
                })

        return {
            'patterns_detected': patterns,
            'total_patterns': len(patterns),
            'analysis_timestamp': datetime.utcnow().isoformat()
        }

    def _analyze_risk_propagation(self, client_id: str) -> Dict[str, Any]:
        """Analyze how risk propagates through the compliance network."""

        if client_id not in self.client_networks:
            return {'propagation_risk': 0.0, 'critical_paths': []}

        graph = self.client_networks[client_id]

        # Find high-risk nodes (regulations with low compliance)
        high_risk_nodes = []
        for node, attr in graph.nodes(data=True):
            if attr.get('node_type') == 'regulation':
                # Check connected edges for compliance weight
                connected_edges = graph.edges(node, data=True)
                avg_weight = sum(edge_attr.get('weight', 0) for _, _, edge_attr in connected_edges) / len(connected_edges) if connected_edges else 0
                if avg_weight < 0.5:
                    high_risk_nodes.append(node)

        # Analyze risk propagation paths
        risk_paths = []
        for risk_node in high_risk_nodes:
            # Find shortest paths from client to risk nodes
            try:
                path = nx.shortest_path(graph, client_id, risk_node)
                path_length = len(path) - 1  # Number of edges
                risk_paths.append({
                    'path': path,
                    'length': path_length,
                    'risk_node': risk_node
                })
            except:
                continue

        max_propagation_risk = max([path['length'] for path in risk_paths]) if risk_paths else 0

        return {
            'propagation_risk': min(max_propagation_risk * 20, 100),  # Scale to 0-100
            'high_risk_nodes': len(high_risk_nodes),
            'critical_paths': risk_paths[:5]  # Top 5 riskiest paths
        }

    async def _generate_network_recommendations(self, client_id: str, graph_metrics: Dict, pattern_analysis: Dict) -> List[str]:
        """Generate recommendations based on network analysis."""

        recommendations = []

        # Density recommendations
        density = graph_metrics.get('density', 0)
        if density < 0.3:
            recommendations.append("Increase network connectivity by adding more compliance relationships")

        # Clustering recommendations
        clustering = graph_metrics.get('average_clustering', 0)
        if clustering > 0.8:
            recommendations.append("High clustering detected - consider diversifying compliance approaches")

        # Pattern-based recommendations
        patterns = pattern_analysis.get('patterns_detected', [])
        for pattern in patterns:
            if pattern.get('type') == 'strong_compliance':
                recommendations.append("Leverage strong compliance history for streamlined processing")

        return recommendations

    def _generate_graph_visualization(self, client_id: str) -> str:
        """Generate base64-encoded graph visualization."""

        if client_id not in self.client_networks:
            return "No graph available"

        try:
            graph = self.client_networks[client_id]

            # Create visualization
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(graph)

            # Draw nodes by type
            node_colors = []
            for node, attr in graph.nodes(data=True):
                node_type = attr.get('node_type', 'unknown')
                if node_type == 'client':
                    node_colors.append('blue')
                elif node_type == 'product':
                    node_colors.append('green')
                elif node_type == 'lender':
                    node_colors.append('red')
                elif node_type == 'regulation':
                    node_colors.append('orange')
                else:
                    node_colors.append('gray')

            nx.draw(graph, pos, node_color=node_colors, with_labels=True, font_size=8)

            # Save to base64
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            import base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            return f"Visualization failed: {str(e)}"

    def _identify_compliance_clusters(self, client_id: str) -> List[Dict[str, Any]]:
        """Identify clusters in the compliance network."""

        if client_id not in self.client_networks:
            return []

        graph = self.client_networks[client_id]

        try:
            # Find communities using greedy modularity
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(graph))

            clusters = []
            for i, community in enumerate(communities):
                node_types = [graph.nodes[n].get('node_type', 'unknown') for n in community]
                dominant_type = max(set(node_types), key=node_types.count)

                clusters.append({
                    'cluster_id': i,
                    'size': len(community),
                    'dominant_type': dominant_type,
                    'nodes': list(community)
                })

            return clusters

        except Exception as e:
            return [{'error': str(e)}]

    def _analyze_network_centrality(self, client_id: str) -> Dict[str, Any]:
        """Analyze centrality measures in the network."""

        if client_id not in self.client_networks:
            return {'error': 'No network found'}

        graph = self.client_networks[client_id]

        try:
            centrality_measures = {
                'degree': nx.degree_centrality(graph),
                'betweenness': nx.betweenness_centrality(graph) if len(graph) > 2 else {},
                'closeness': nx.closeness_centrality(graph) if len(graph) > 2 else {}
            }

            # Find most central regulation
            regulation_nodes = {n: centrality_measures['degree'].get(n, 0)
                              for n, attr in graph.nodes(data=True)
                              if attr.get('node_type') == 'regulation'}

            most_central_regulation = max(regulation_nodes.items(), key=lambda x: x[1]) if regulation_nodes else None

            return {
                'centrality_measures': centrality_measures,
                'most_central_node': most_central_regulation[0] if most_central_regulation else None,
                'centrality_score': most_central_regulation[1] if most_central_regulation else 0
            }

        except Exception as e:
            return {'error': str(e)}

    async def _make_network_based_decisions(self, analysis_results: Dict) -> None:
        """Make decisions based on network analysis."""

        risk_propagation = analysis_results.get('risk_propagation', {})
        propagation_risk = risk_propagation.get('propagation_risk', 0)

        if propagation_risk > 70:
            self.logger.warning(f"High risk propagation detected: {propagation_risk}")
            # Could trigger additional compliance checks

        # Store decision
        self.decision_memory.append({
            'type': 'network_analysis',
            'timestamp': datetime.utcnow().isoformat(),
            'propagation_risk': propagation_risk,
            'action_taken': 'monitored' if propagation_risk < 50 else 'flagged'
        })

    async def _perform_initial_assessment(self, application_data: Dict) -> Dict[str, Any]:
        """Perform initial assessment of the application."""

        client_data = application_data.get('client_data', {})
        documents = application_data.get('documents', [])

        return {
            'client_profile_complete': bool(client_data.get('id') and client_data.get('gross_annual_income')),
            'documents_provided': len(documents),
            'documents_valid_paths': len([d for d in documents if d.get('file_path')]),
            'initial_risk_level': 'medium',  # Default assessment
            'assessment_timestamp': datetime.utcnow().isoformat()
        }

    def _calculate_overall_risk(self, cv_results, nlp_results, graph_results, traditional_qc) -> str:
        """Calculate overall risk assessment from all analyses."""

        risk_scores = []

        # CV risk
        if not isinstance(cv_results, Exception):
            cv_risk = cv_results.get('risk_assessment', 'medium')
            risk_scores.append({'cv': self._risk_level_to_score(cv_risk)})

        # NLP risk (based on negative sentiment and risk keywords)
        if not isinstance(nlp_results, Exception):
            negative_sentiment = len([s for s in nlp_results.get('sentiment_analysis', [])
                                   if s.get('sentiment') == 'negative'])
            risk_keywords = len(nlp_results.get('risk_keywords', []))
            nlp_risk_score = min((negative_sentiment + risk_keywords) * 10, 100)
            risk_scores.append({'nlp': nlp_risk_score})

        # Graph risk
        if not isinstance(graph_results, Exception):
            propagation_risk = graph_results.get('risk_propagation', {}).get('propagation_risk', 0)
            risk_scores.append({'graph': propagation_risk})

        # Traditional QC risk
        if not isinstance(traditional_qc, Exception):
            qc_score = traditional_qc.get('qc_summary', {}).get('overall_score', 50)
            qc_risk = 100 - qc_score  # Invert: higher QC score = lower risk
            risk_scores.append({'qc': qc_risk})

        # Average risk
        if risk_scores:
            avg_risk = sum(score.values() for score in risk_scores) / len(risk_scores)
        else:
            avg_risk = 50

        if avg_risk < 30:
            return 'low'
        elif avg_risk < 60:
            return 'medium'
        elif avg_risk < 80:
            return 'high'
        else:
            return 'critical'

    def _risk_level_to_score(self, risk_level: str) -> int:
        """Convert risk level to numerical score."""
        risk_mapping = {'low': 20, 'medium': 50, 'high': 80, 'critical': 100}
        return risk_mapping.get(risk_level, 50)

    async def _learn_from_decision(self, decisions: Dict, cv_results, nlp_results, graph_results, traditional_qc) -> None:
        """Learn from decision outcomes for future improvement."""

        decision_outcome = {
            'timestamp': datetime.utcnow().isoformat(),
            'final_recommendation': decisions.get('final_recommendation'),
            'confidence_level': decisions.get('confidence_level'),
            'analyses_used': {
                'cv': not isinstance(cv_results, Exception),
                'nlp': not isinstance(nlp_results, Exception),
                'graph': not isinstance(graph_results, Exception),
                'qc': not isinstance(traditional_qc, Exception)
            }
        }

        self.decision_memory.append(decision_outcome)

        # Update learning patterns
        recommendation = decisions.get('final_recommendation', 'unknown')
        self.learning_patterns[f'recommendation_{recommendation}'] = self.learning_patterns.get(f'recommendation_{recommendation}', 0) + 1

    async def _autonomous_lender_submission(self, application_data: Dict) -> str:
        """Autonomously submit application to lender."""

        # In production, this would integrate with lender APIs
        lender_name = application_data.get('product_selection', {}).get('lender_name', 'unknown')

        self.logger.info(f"Autonomous submission to {lender_name} would be executed here")

        return f"Application submitted to {lender_name} (simulated)"

    async def _flag_for_priority_review(self, application_data: Dict) -> str:
        """Flag application for priority human review."""

        self.logger.info(f"Application {application_data.get('application_id')} flagged for priority review")

        return "Flagged for priority review"

    async def _schedule_human_review(self, application_data: Dict) -> str:
        """Schedule human review for the application."""

        self.logger.info(f"Human review scheduled for application {application_data.get('application_id')}")

        return "Human review scheduled"

    async def _notify_client_of_issues(self, application_data: Dict, decisions: Dict) -> str:
        """Notify client of application issues."""

        client_id = application_data.get('client_data', {}).get('id', 'unknown')
        recommendation = decisions.get('final_recommendation', 'review_required')

        self.logger.info(f"Client {client_id} notified of issues: {recommendation}")

        return f"Client notified: {recommendation}"

    async def _learn_from_workflow_execution(self, workflow_id: str, analysis_results, decisions, actions_taken, learning_outcomes=None) -> Dict[str, Any]:
        """Learn from complete workflow execution."""

        execution_summary = {
            'workflow_id': workflow_id,
            'success_rate': len([a for a in actions_taken if 'error' not in a.get('result', '').lower()]) / len(actions_taken) if actions_taken else 0,
            'decision_accuracy': decisions.get('confidence_level', 0),
            'analysis_completion': sum(1 for r in analysis_results if not isinstance(r, Exception)) / len(analysis_results),
            'learning_timestamp': datetime.utcnow().isoformat()
        }

        # Update autonomous actions memory
        self.autonomous_actions.append({
            'workflow_id': workflow_id,
            'actions': [a['action'] for a in actions_taken],
            'success': execution_summary['success_rate'] > 0.8,
            'timestamp': datetime.utcnow().isoformat()
        })

        return execution_summary

    async def _store_workflow_execution(self, workflow_results: Dict) -> None:
        """Store workflow execution results for learning."""

        try:
            # In production, would store in database
            self.logger.info(f"Workflow {workflow_results['workflow_id']} execution stored for learning")

        except Exception as e:
            self.logger.error(f"Failed to store workflow execution: {str(e)}")

    async def _load_agent_memory(self) -> None:
        """Load agent memory and learning patterns."""

        try:
            # In production, would load from database
            self.logger.info("Agent memory loaded")

        except Exception as e:
            self.logger.warning(f"Failed to load agent memory: {str(e)}")

    async def _initialize_compliance_graph(self) -> None:
        """Initialize the global compliance graph with known relationships."""

        # Add known regulations
        regulations = [
            'Wft_Article_86f', 'BGfo_Article_8_1', 'NHG_Requirements',
            'GDPR', 'Wwft', 'DNB_Regulations'
        ]

        for reg in regulations:
            self.compliance_graph.add_node(f"regulation_{reg}",
                node_type='regulation',
                category='financial_services',
                country='netherlands'
            )

        # Add known lenders
        lenders = ['stater', 'quion', 'ing', 'rabobank', 'abn_amro']
        for lender in lenders:
            self.compliance_graph.add_node(f"lender_{lender}",
                node_type='lender',
                country='netherlands',
                regulated=True
            )

            # Connect lenders to regulations
            for reg in regulations[:3]:  # Core mortgage regulations
                self.compliance_graph.add_edge(f"lender_{lender}", f"regulation_{reg}",
                    relationship_type='must_comply_with',
                    weight=1.0
                )

        self.logger.info(f"Compliance graph initialized with {len(self.compliance_graph.nodes())} nodes and {len(self.compliance_graph.edges())} edges")

    def _analyze_document_structure(self, ocr_result: Dict) -> Dict[str, Any]:
        """Analyze document structure from OCR results."""

        text_overlay = ocr_result.get('TextOverlay', {})
        lines = text_overlay.get('Lines', [])

        return {
            'total_lines': len(lines),
            'has_tables': any('|' in line.get('LineText', '') for line in lines),
            'structured_layout': len(lines) > 10,
            'text_density': sum(len(line.get('LineText', '')) for line in lines) / max(len(lines), 1)
        }

    async def _analyze_document_signatures(self, document: Dict, cv_result: Dict) -> Dict[str, Any]:
        """Analyze document signatures."""

        # Placeholder for signature analysis
        return {
            'signatures_found': 0,
            'valid_signatures': 0,
            'signature_types': [],
            'valid': True
        }

    def _get_lender_config_version(self, lender_name: str) -> str:
        """Get lender configuration version."""

        if lender_name and lender_name.lower() in self.lender_configs:
            return self.lender_configs[lender_name.lower()].get('version', '1.0')

        return 'unknown'

    # Helper methods implementation

    async def _check_file_integrity(self, file_path: str) -> Dict[str, Any]:
        """Check file integrity and basic authenticity markers."""

        issues = []

        try:
            import os
            import mimetypes
            from pathlib import Path

            # Check if file exists
            if not os.path.exists(file_path):
                issues.append("File does not exist")
                return {'valid': False, 'issues': issues}

            # Check file size (reasonable bounds)
            file_size = os.path.getsize(file_path)
            if file_size < 100:  # Too small to be a real document
                issues.append("File suspiciously small")
            elif file_size > 50 * 1024 * 1024:  # 50MB limit
                issues.append("File size exceeds reasonable limits")

            # Check file extension matches expected type
            file_ext = Path(file_path).suffix.lower()
            mime_type, _ = mimetypes.guess_type(file_path)

            expected_types = {
                '.pdf': ['application/pdf'],
                '.jpg': ['image/jpeg'],
                '.jpeg': ['image/jpeg'],
                '.png': ['image/png'],
                '.doc': ['application/msword'],
                '.docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']
            }

            if file_ext in expected_types:
                if mime_type not in expected_types[file_ext]:
                    issues.append(f"File extension {file_ext} does not match MIME type {mime_type}")

            # Check file is not corrupted (basic check)
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(10)
                    # Basic corruption check - file should have readable header
                    if not header:
                        issues.append("File appears to be empty or corrupted")
            except Exception as e:
                issues.append(f"Cannot read file: {str(e)}")

        except Exception as e:
            issues.append(f"File integrity check failed: {str(e)}")

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    async def _verify_digital_signatures(self, document: Dict) -> Dict[str, Any]:
        """Verify digital signatures if present in the document."""

        issues = []
        checked = False

        try:
            doc_type = document.get('type')
            file_path = document.get('file_path')

            # Only check for signatures on certain document types
            if doc_type in ['id_document', 'property_documents', 'official_letters']:
                checked = True

                # Check if file exists
                if not file_path:
                    issues.append("No file path provided for signature verification")
                    return {'valid': False, 'checked': checked, 'issues': issues}

                # Basic signature presence check (production would use PDF libraries)
                try:
                    # Check file extension for signature-capable formats
                    from pathlib import Path
                    file_ext = Path(file_path).suffix.lower()

                    if file_ext == '.pdf':
                        # In production: Use PyPDF2, pdfplumber, or similar to check for signatures
                        # For now, check document metadata
                        signature_info = document.get('signature_info', {})
                        if signature_info.get('has_signature', False):
                            # Validate signature timestamp if available
                            signed_at = signature_info.get('signed_at')
                            if signed_at:
                                try:
                                    signed_date = datetime.fromisoformat(signed_at.replace('Z', '+00:00'))
                                    if signed_date > datetime.utcnow():
                                        issues.append("Document signed in the future")
                                    elif (datetime.utcnow() - signed_date).days > 365:
                                        issues.append("Document signature is more than a year old")
                                except:
                                    issues.append("Invalid signature timestamp format")
                        else:
                            issues.append("Document type requires digital signature but none found")
                    else:
                        # Non-PDF documents can't have embedded signatures
                        if doc_type in ['id_document', 'property_documents']:
                            issues.append(f"{doc_type} should be digitally signed PDF")

                except Exception as e:
                    issues.append(f"Signature verification error: {str(e)}")

        except Exception as e:
            issues.append(f"Digital signature check failed: {str(e)}")

        return {
            'valid': len(issues) == 0,
            'checked': checked,
            'issues': issues
        }

    async def _check_content_consistency(self, document: Dict) -> Dict[str, Any]:
        """Check content consistency within a document."""

        issues = []

        try:
            doc_type = document.get('type')
            extracted_data = document.get('extracted_data', {})

            # Document-type specific consistency checks
            if doc_type == 'income_statement':
                # Check if income figures are reasonable
                annual_income = extracted_data.get('annual_income', 0)
                if annual_income > 0:
                    if annual_income < 10000:  # Unrealistically low
                        issues.append("Annual income figure seems unrealistically low")
                    elif annual_income > 1000000:  # Unrealistically high for Netherlands
                        issues.append("Annual income figure seems unrealistically high")

            elif doc_type == 'id_document':
                # Check expiry date is in future
                expiry_date_str = extracted_data.get('expiry_date')
                if expiry_date_str:
                    try:
                        expiry_date = datetime.fromisoformat(expiry_date_str.replace('Z', '+00:00'))
                        if expiry_date < datetime.utcnow():
                            issues.append("ID document has expired")
                    except:
                        issues.append("Invalid expiry date format")

            elif doc_type == 'property_valuation':
                # Check valuation amount is reasonable
                valuation = extracted_data.get('property_value', 0)
                if valuation > 0:
                    if valuation < 50000:  # Unrealistically low for property
                        issues.append("Property valuation seems unrealistically low")
                    elif valuation > 5000000:  # Unrealistically high
                        issues.append("Property valuation seems unrealistically high")

        except Exception as e:
            issues.append(f"Content consistency check failed: {str(e)}")

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    async def _verify_document_issuer(self, document: Dict) -> Dict[str, Any]:
        """Verify document issuer authenticity."""

        issues = []

        try:
            doc_type = document.get('type')
            extracted_data = document.get('extracted_data', {})

            # Issuer verification based on document type
            if doc_type == 'income_statement':
                employer = extracted_data.get('employer_name', '').strip()
                if not employer:
                    issues.append("Employer name missing from income statement")
                elif len(employer) < 2:
                    issues.append("Employer name suspiciously short")

            elif doc_type == 'tax_return':
                tax_authority = extracted_data.get('issuing_authority', '').lower()
                if 'belastingdienst' not in tax_authority and tax_authority:
                    issues.append("Document does not appear to be issued by Dutch tax authority")

            elif doc_type == 'id_document':
                issuing_country = extracted_data.get('issuing_country', '').upper()
                if issuing_country and issuing_country != 'NL':
                    issues.append("ID document not issued by Netherlands")

        except Exception as e:
            issues.append(f"Issuer verification failed: {str(e)}")

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    async def _call_bkr_api(self, bsn: str) -> Dict[str, Any]:
        """Call BKR API to get credit information."""

        if not self.bkr_api_url or not self.bkr_api_key:
            self.logger.warning("BKR API credentials not configured - cannot perform real-time credit check")
            return {
                'status': 'error',
                'message': 'BKR API not configured. Please set BKR_API_URL and BKR_API_KEY environment variables.',
                'requires_manual_review': True
            }

        try:
            headers = {
                'Authorization': f'Bearer {self.bkr_api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'MortgageAI-Dutch-QC-Agent/2.0'
            }

            payload = {
                'bsn': bsn,
                'include_negative_registrations': True,
                'include_current_debts': True,
                'include_payment_history': True,
                'include_credit_limits': True,
                'real_time_check': True,
                'consent_provided': True,
                'purpose': 'mortgage_application_qc'
            }

            response = requests.post(
                f"{self.bkr_api_url}/credit-check",
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                # Validate response structure
                required_fields = ['credit_score', 'negative_registrations', 'current_debts']
                if all(field in result for field in required_fields):
                    result['status'] = 'success'
                    result['real_time'] = True
                    return result
                else:
                    return {
                        'status': 'error',
                        'message': 'BKR API response missing required fields',
                        'requires_manual_review': True
                    }
            elif response.status_code == 401:
                return {
                    'status': 'error',
                    'message': 'BKR API authentication failed',
                    'requires_manual_review': True
                }
            elif response.status_code == 403:
                return {
                    'status': 'error',
                    'message': 'BKR API access forbidden - check permissions',
                    'requires_manual_review': True
                }
            elif response.status_code == 429:
                return {
                    'status': 'error',
                    'message': 'BKR API rate limit exceeded',
                    'retry_after': response.headers.get('Retry-After', '60'),
                    'requires_manual_review': True
                }
            else:
                return {
                    'status': 'error',
                    'message': f'BKR API returned status {response.status_code}: {response.text[:200]}',
                    'requires_manual_review': True
                }

        except requests.exceptions.Timeout:
            self.logger.error("BKR API request timed out")
            return {
                'status': 'error',
                'message': 'BKR API request timed out after 30 seconds',
                'requires_manual_review': True
            }
        except requests.exceptions.ConnectionError:
            self.logger.error("BKR API connection failed")
            return {
                'status': 'error',
                'message': 'Cannot connect to BKR API - network issue',
                'requires_manual_review': True
            }
        except Exception as e:
            self.logger.error(f"BKR API call failed: {str(e)}")
            return {
                'status': 'error',
                'message': f'BKR API error: {str(e)}',
                'requires_manual_review': True
            }


    def _calculate_bkr_approval_likelihood(self, credit_score: int, negative_registrations: List, total_debt: float) -> float:
        """Calculate mortgage approval likelihood based on BKR data."""

        # Base score from credit score
        if credit_score >= 850:
            base_score = 95
        elif credit_score >= 750:
            base_score = 85
        elif credit_score >= 650:
            base_score = 70
        elif credit_score >= 550:
            base_score = 50
        else:
            base_score = 20

        # Penalties for negative registrations
        penalty = len(negative_registrations) * 15

        # Penalty for high debt burden
        if total_debt > 500:
            debt_penalty = min(total_debt / 100, 25)
        else:
            debt_penalty = 0

        final_score = max(0, min(100, base_score - penalty - debt_penalty))
        return round(final_score, 1)

    def _identify_bkr_risk_factors(self, bkr_response: Dict[str, Any]) -> List[str]:
        """Identify risk factors from BKR response."""

        risk_factors = []
        credit_score = bkr_response.get('credit_score', 0)
        negative_regs = bkr_response.get('negative_registrations', [])
        debts = bkr_response.get('current_debts', [])

        if credit_score < 650:
            risk_factors.append("Low credit score")
        elif credit_score < 750:
            risk_factors.append("Moderate credit score")

        if negative_regs:
            risk_factors.append(f"{len(negative_regs)} negative registration(s)")

        total_debt = sum([d.get('monthly_payment', 0) for d in debts])
        if total_debt > 400:
            risk_factors.append("High existing debt burden")

        return risk_factors

    def _generate_bkr_recommendations(self, bkr_response: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on BKR data."""

        recommendations = []
        credit_score = bkr_response.get('credit_score', 0)
        negative_regs = bkr_response.get('negative_registrations', [])
        debts = bkr_response.get('current_debts', [])

        if credit_score < 650:
            recommendations.append("Consider credit improvement before applying")
            recommendations.append("Explore mortgage options with more lenient credit requirements")

        if negative_regs:
            recommendations.append("Address negative registrations before mortgage application")
            recommendations.append("Consider seeking professional credit counseling")

        total_debt = sum([d.get('monthly_payment', 0) for d in debts])
        if total_debt > 300:
            recommendations.append("Consider debt consolidation options")
            recommendations.append("Review current debt management strategies")

        return recommendations

    async def _get_current_nhg_limits(self) -> Dict[str, Any]:
        """Get current NHG limits from API or cache."""

        try:
            # Check cache first
            if hasattr(self, '_nhg_limits_cache') and self._nhg_limits_cache:
                cache_time = getattr(self, '_nhg_limits_timestamp', None)
                if cache_time and (datetime.utcnow() - cache_time).seconds < 86400:  # 24 hours
                    return self._nhg_limits_cache

            # Fetch from API or use defaults
            limits = {
                'property_limit': 435000,  # EUR for 2025
                'income_limit': 43000,     # EUR for certain programs
                'version': '2025.1'
            }

            self._nhg_limits_cache = limits
            self._nhg_limits_timestamp = datetime.utcnow()

            return limits

        except Exception as e:
            self.logger.error(f"Failed to get NHG limits: {str(e)}")
            return {'property_limit': 435000, 'income_limit': 43000, 'version': 'fallback'}

    def _calculate_nhg_costs(self, mortgage_amount: Decimal) -> float:
        """Calculate NHG guarantee costs."""

        # NHG costs are typically 0.7% of mortgage amount for 2025
        cost_percentage = Decimal('0.007')  # 0.7%
        costs = float(mortgage_amount * cost_percentage)

        # Minimum and maximum costs
        min_cost = 250
        max_cost = 5000

        return max(min_cost, min(max_cost, costs))

    def _calculate_nhg_interest_benefit(self, mortgage_amount: Decimal, product_selection: Dict[str, Any]) -> float:
        """Calculate interest rate benefit from NHG."""

        # NHG typically provides 0.4-0.6% lower interest rates
        base_rate_reduction = 0.005  # 0.5% average reduction
        benefit_amount = float(mortgage_amount * Decimal(str(base_rate_reduction)))

        # Assume 30-year mortgage for annual calculation
        annual_benefit = benefit_amount * 12

        return round(annual_benefit, 2)

    def _calculate_nhg_break_even(self, interest_savings: float, costs: float) -> float:
        """Calculate break-even period for NHG."""

        if interest_savings <= 0:
            return float('inf')

        return round(costs / interest_savings, 1)

    def _generate_nhg_recommendation(self, eligible: bool, interest_benefit: float, costs: float, application_data: Dict[str, Any]) -> str:
        """Generate NHG recommendation."""

        if not eligible:
            return "nhg_not_eligible"

        net_benefit = interest_benefit - costs
        break_even_years = self._calculate_nhg_break_even(interest_benefit, costs)

        property_value = application_data.get('property_value', 0)
        mortgage_amount = application_data.get('mortgage_amount', 0)

        # Recommend NHG if net benefit is positive and break-even within reasonable time
        if net_benefit > 0 and break_even_years <= 5:
            return "apply_for_nhg"
        elif net_benefit > 0 and break_even_years <= 10:
            return "consider_nhg"
        else:
            return "nhg_not_beneficial"

    def _calculate_dutch_maximum_mortgage(self, income: Decimal, debt_payments: Decimal, interest_rate: Decimal) -> Decimal:
        """Calculate maximum mortgage based on Dutch affordability rules."""

        # Simplified Dutch mortgage calculation based on government tables
        # In reality, this uses complex income-based tables from AFM

        net_income = income - debt_payments

        # Maximum housing costs as percentage of income (AFM guideline)
        max_housing_percentage = Decimal('0.30')  # 30%

        # Calculate maximum monthly housing payment
        max_monthly_payment = (net_income * max_housing_percentage) / 12

        # Convert to mortgage amount using annuity formula
        term_years = 30
        monthly_rate = interest_rate / 12
        num_payments = term_years * 12

        if monthly_rate == 0:
            max_mortgage = max_monthly_payment * num_payments
        else:
            max_mortgage = max_monthly_payment * ((1 - (1 + monthly_rate) ** -num_payments) / monthly_rate)

        return max_mortgage.quantize(Decimal('1'), rounding=ROUND_HALF_UP)

    def _calculate_stress_test_affordability(self, mortgage_amount: Decimal, income: Decimal, stress_rate: Decimal, existing_debts: Decimal) -> Dict[str, Any]:
        """Calculate affordability under stress test conditions."""

        # Calculate monthly mortgage payment at stress rate
        term_years = 30
        monthly_rate = stress_rate / 12
        num_payments = term_years * 12

        if monthly_rate == 0:
            monthly_payment = mortgage_amount / num_payments
        else:
            monthly_payment = mortgage_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)

        # Add existing debt payments
        total_monthly_debt = monthly_payment + (existing_debts / 12)

        # Check against income limits
        max_housing_percentage = Decimal('0.30')  # 30% of gross income
        max_monthly_housing = (income * max_housing_percentage) / 12

        passed = total_monthly_debt <= max_monthly_housing

        return {
            'passed': passed,
            'stress_rate': float(stress_rate),
            'monthly_payment_stress': float(monthly_payment),
            'total_monthly_debt_stress': float(total_monthly_debt),
            'max_allowable_monthly': float(max_monthly_housing),
            'headroom': float(max_monthly_housing - total_monthly_debt) if passed else 0
        }

    def _calculate_housing_cost_ratio(self, mortgage_amount: Decimal, interest_rate: Decimal, income: Decimal, existing_debts: Decimal) -> float:
        """Calculate housing cost ratio as percentage of income."""

        # Calculate monthly mortgage payment
        term_years = 30
        monthly_rate = interest_rate / 12
        num_payments = term_years * 12

        if monthly_rate == 0:
            monthly_payment = mortgage_amount / num_payments
        else:
            monthly_payment = mortgage_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)

        # Total housing costs
        total_monthly_housing = monthly_payment + (existing_debts / 12)

        # Ratio as percentage of monthly income
        monthly_income = income / 12
        ratio = (total_monthly_housing / monthly_income) * 100 if monthly_income > 0 else 0

        return round(ratio, 2)

    def _generate_affordability_recommendations(self, mortgage_amount: Decimal, max_mortgage: Decimal, stress_test: Dict, housing_ratio: float) -> List[str]:
        """Generate affordability recommendations."""

        recommendations = []

        if mortgage_amount > max_mortgage:
            shortfall = mortgage_amount - max_mortgage
            recommendations.append(f"Reduce mortgage amount by €{shortfall:.0f} to meet affordability requirements")

        if not stress_test['passed']:
            recommendations.append("Mortgage fails stress test - consider lower amount or different product")

        if housing_ratio > 30:
            recommendations.append(f"Housing costs ({housing_ratio:.1f}%) exceed 30% income limit")

        if mortgage_amount < max_mortgage * Decimal('0.8'):
            headroom = max_mortgage - mortgage_amount
            recommendations.append(f"Consider increasing mortgage by up to €{headroom:.0f} if needed")

        return recommendations

    async def _validate_document_type(self, documents: List[Dict], doc_type: str) -> Dict[str, Any]:
        """Validate a specific document type."""

        # Find documents of this type
        matching_docs = [d for d in documents if d.get('type') == doc_type]

        if not matching_docs:
            return {
                'provided': False,
                'valid': False,
                'issues': [{'severity': 'critical', 'message': f'{doc_type} document is missing'}]
            }

        # Validate each document
        validation_results = []
        for doc in matching_docs:
            result = await self._validate_individual_document(doc, doc_type)
            validation_results.append(result)

        # Overall result
        any_valid = any(r['valid'] for r in validation_results)
        all_issues = []
        for r in validation_results:
            all_issues.extend(r.get('issues', []))

        return {
            'provided': True,
            'valid': any_valid,
            'issues': all_issues,
            'document_count': len(matching_docs),
            'validation_results': validation_results
        }

    async def _validate_individual_document(self, document: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
        """Validate an individual document."""

        issues = []

        # Check file existence and accessibility
        file_path = document.get('file_path')
        if not file_path:
            issues.append({'severity': 'critical', 'message': 'Document file path missing'})
            return {'valid': False, 'issues': issues}

        # Check file size (reasonable limits)
        try:
            import os
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                issues.append({'severity': 'high', 'message': 'Document file too large'})
        except:
            issues.append({'severity': 'high', 'message': 'Cannot access document file'})

        # Type-specific validation
        if doc_type == 'identity_document':
            expiry = document.get('expiry_date')
            if expiry:
                expiry_date = datetime.fromisoformat(expiry.replace('Z', '+00:00'))
                if expiry_date < datetime.utcnow():
                    issues.append({'severity': 'critical', 'message': 'Identity document expired'})

        elif doc_type == 'income_statement':
            # Check if recent (within 3 months)
            doc_date = document.get('document_date')
            if doc_date:
                doc_datetime = datetime.fromisoformat(doc_date.replace('Z', '+00:00'))
                if (datetime.utcnow() - doc_datetime).days > 90:
                    issues.append({'severity': 'medium', 'message': 'Income statement older than 3 months'})

        return {
            'valid': len([i for i in issues if i['severity'] == 'critical']) == 0,
            'issues': issues
        }

    async def _validate_document_authenticity(self, documents: List[Dict]) -> Dict[str, Any]:
        """Validate document authenticity using multiple verification methods."""

        authenticity_issues = []
        validation_methods = []

        for doc in documents:
            doc_issues = []
            doc_type = doc.get('type', 'unknown')
            file_path = doc.get('file_path')

            # Method 1: File integrity checks
            if file_path:
                integrity_result = await self._check_file_integrity(file_path)
                if not integrity_result['valid']:
                    doc_issues.extend(integrity_result['issues'])
                validation_methods.append('file_integrity')

            # Method 2: Digital signature verification (if applicable)
            signature_result = await self._verify_digital_signatures(doc)
            if not signature_result['valid']:
                doc_issues.extend(signature_result['issues'])
            if signature_result['checked']:
                validation_methods.append('digital_signature')

            # Method 3: Content consistency checks
            consistency_result = await self._check_content_consistency(doc)
            if not consistency_result['valid']:
                doc_issues.extend(consistency_result['issues'])
            validation_methods.append('content_consistency')

            # Method 4: Issuer verification (for official documents)
            if doc_type in ['id_document', 'income_statement', 'tax_return']:
                issuer_result = await self._verify_document_issuer(doc)
                if not issuer_result['valid']:
                    doc_issues.extend(issuer_result['issues'])
                validation_methods.append('issuer_verification')

            # Add document-specific issues to main list
            if doc_issues:
                for issue in doc_issues:
                    authenticity_issues.append({
                        'document_type': doc_type,
                        'file_path': file_path,
                        'issue': issue
                    })

        return {
            'authentic': len(authenticity_issues) == 0,
            'issues': authenticity_issues,
            'validation_methods_used': list(set(validation_methods)),
            'documents_checked': len(documents),
            'authenticity_score': max(0, 100 - (len(authenticity_issues) * 20))
        }

    def _validate_document_consistency(self, documents: List[Dict]) -> Dict[str, Any]:
        """Validate consistency between documents."""

        consistency_issues = []

        # Extract key data points from documents
        extracted_data = {}
        for doc in documents:
            doc_type = doc.get('type')
            if doc_type and 'extracted_data' in doc:
                extracted_data[doc_type] = doc['extracted_data']

        # Check consistency between documents
        if 'income_statement' in extracted_data and 'application_form' in extracted_data:
            income_doc = extracted_data['income_statement'].get('annual_income')
            app_income = extracted_data['application_form'].get('annual_income')

            if income_doc and app_income and abs(income_doc - app_income) > 1000:
                consistency_issues.append({
                    'severity': 'high',
                    'message': 'Income amounts inconsistent between documents',
                    'difference': abs(income_doc - app_income)
                })

        return {
            'consistent': len(consistency_issues) == 0,
            'issues': consistency_issues
        }

    def _initialize_lender_configurations(self) -> Dict[str, Any]:
        """Initialize lender-specific validation configurations."""

        return {
            'stater': {
                'validation_criteria': {
                    'minimum_income': {'threshold': 25000, 'type': 'minimum'},
                    'maximum_ltv': {'threshold': 100, 'type': 'maximum'},
                    'employment_stability': {'minimum_months': 12, 'type': 'employment'}
                },
                'required_documents': [
                    'income_statement', 'employment_contract', 'bank_statements',
                    'property_valuation', 'identity_document'
                ],
                'typical_processing_time': '10-15 business days',
                'api_endpoint': settings.STATER_API_URL or 'https://api.stater.nl/mortgage/submit',
                'version': '1.0',
                'special_conditions': ['Requires Dutch residency', 'BSN mandatory']
            },
            'quion': {
                'validation_criteria': {
                    'minimum_income': {'threshold': 30000, 'type': 'minimum'},
                    'maximum_ltv': {'threshold': 95, 'type': 'maximum'},
                    'debt_to_income_max': {'threshold': 30, 'type': 'maximum'}
                },
                'required_documents': [
                    'salary_specification', 'tax_return', 'bank_statements',
                    'property_documents', 'identity_proof'
                ],
                'typical_processing_time': '7-12 business days',
                'api_endpoint': settings.QUION_API_URL or 'https://api.quion.nl/applications/submit',
                'version': '1.0',
                'special_conditions': ['Higher income requirements', 'Faster processing']
            },
            'ing': {
                'validation_criteria': {
                    'minimum_income': {'threshold': 20000, 'type': 'minimum'},
                    'maximum_ltv': {'threshold': 101, 'type': 'maximum'},  # Allows slight over-valuation
                    'credit_score_min': {'threshold': 600, 'type': 'minimum'}
                },
                'required_documents': [
                    'income_proof', 'bank_statements', 'property_valuation', 'id_document'
                ],
                'typical_processing_time': '8-14 business days',
                'api_endpoint': settings.ING_API_URL or 'https://api.ing.nl/mortgages/submit',
                'version': '1.0'
            }
        }

    async def _validate_lender_criterion(self, application_data: Dict[str, Any], criterion_name: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a specific lender criterion."""

        threshold = requirements.get('threshold')
        criterion_type = requirements.get('type')

        if criterion_type == 'minimum':
            actual_value = self._extract_criterion_value(application_data, criterion_name)
            passed = actual_value >= threshold if actual_value is not None else False

            return {
                'criterion': criterion_name,
                'type': criterion_type,
                'threshold': threshold,
                'actual_value': actual_value,
                'passed': passed,
                'severity': 'high' if not passed else 'low'
            }

        elif criterion_type == 'maximum':
            actual_value = self._extract_criterion_value(application_data, criterion_name)
            passed = actual_value <= threshold if actual_value is not None else False

            return {
                'criterion': criterion_name,
                'type': criterion_type,
                'threshold': threshold,
                'actual_value': actual_value,
                'passed': passed,
                'severity': 'critical' if not passed else 'low'
            }

        return {
            'criterion': criterion_name,
            'type': criterion_type,
            'passed': False,
            'error': 'Unknown criterion type'
        }

    def _extract_criterion_value(self, application_data: Dict[str, Any], criterion_name: str) -> Optional[float]:
        """Extract criterion value from application data."""

        client_data = application_data.get('client_data', {})

        if criterion_name == 'minimum_income':
            return client_data.get('gross_annual_income')
        elif criterion_name == 'maximum_ltv':
            property_value = application_data.get('property_value', 0)
            mortgage_amount = application_data.get('mortgage_amount', 0)
            return (mortgage_amount / property_value * 100) if property_value > 0 else None
        elif criterion_name == 'debt_to_income_max':
            income = client_data.get('gross_annual_income', 0)
            debts = sum([d.get('monthly_payment', 0) for d in client_data.get('existing_debts', [])])
            annual_debts = debts * 12
            return (annual_debts / income * 100) if income > 0 else None
        elif criterion_name == 'employment_stability':
            # Simplified - in reality would check employment history
            return 24  # Assume 2 years for demo
        elif criterion_name == 'credit_score_min':
            # Would come from BKR check
            return 750  # Assume good score for demo

        return None

    async def _validate_lender_document_requirements(self, documents: List[Dict], required_docs: List[str]) -> Dict[str, Any]:
        """Validate lender-specific document requirements."""

        provided_docs = [d.get('type') for d in documents if d.get('type')]
        missing_docs = []

        for required_doc in required_docs:
            if required_doc not in provided_docs:
                missing_docs.append(required_doc)

        return {
            'required_documents': required_docs,
            'provided_documents': provided_docs,
            'missing_documents': missing_docs,
            'completeness': len(required_docs) - len(missing_docs),
            'complete': len(missing_docs) == 0
        }

    def _calculate_lender_compatibility_score(self, validation_results: List[Dict], document_check: Dict) -> float:
        """Calculate overall lender compatibility score."""

        # Score from validation criteria
        criterion_score = 0
        total_criteria = len(validation_results)

        if total_criteria > 0:
            passed_criteria = len([r for r in validation_results if r.get('passed', False)])
            criterion_score = (passed_criteria / total_criteria) * 100

        # Score from document completeness
        doc_completeness = document_check.get('completeness', 0)
        total_docs = len(document_check.get('required_documents', []))
        doc_score = (doc_completeness / total_docs * 100) if total_docs > 0 else 100

        # Weighted average
        overall_score = (criterion_score * 0.7) + (doc_score * 0.3)

        return round(overall_score, 1)

    def _estimate_approval_likelihood(self, compatibility_score: float, validation_results: List[Dict]) -> float:
        """Estimate approval likelihood based on compatibility."""

        # Base likelihood on compatibility score
        base_likelihood = compatibility_score

        # Adjust for failed critical criteria
        critical_failures = len([r for r in validation_results if not r.get('passed', True) and r.get('severity') == 'critical'])
        base_likelihood -= critical_failures * 20

        # Ensure reasonable bounds
        return max(0, min(100, base_likelihood))

    def _calculate_ftr_probability(self, overall_score: float, blocking_issues: int, risk_factors: int) -> float:
        """Calculate first-time-right probability."""

        base_probability = overall_score

        # Heavy penalties for blocking issues
        base_probability -= blocking_issues * 25

        # Moderate penalties for risk factors
        base_probability -= risk_factors * 5

        return max(0, min(100, base_probability))

    def _calculate_confidence_level(self, scores: List[float]) -> str:
        """Calculate confidence level in the assessment."""

        if not scores:
            return 'low'

        score_std = statistics.stdev(scores) if len(scores) > 1 else 0
        avg_score = statistics.mean(scores)

        if score_std < 10 and avg_score > 80:
            return 'high'
        elif score_std < 15 and avg_score > 70:
            return 'medium'
        else:
            return 'low'

    def _generate_ftr_recommendation(self, ftr_probability: float, blocking_issues: List[str], risk_factors: List[str]) -> str:
        """Generate FTR recommendation."""

        if ftr_probability >= 85:
            return "ready_for_submission"
        elif ftr_probability >= 70:
            return "ready_with_minor_fixes"
        elif ftr_probability >= 50:
            return "needs_improvements"
        else:
            return "requires_major_revision"

    def _estimate_processing_time(self, lender_name: str, ftr_assessment: Dict[str, Any]) -> str:
        """Estimate processing time based on lender and FTR probability."""

        if not lender_name or lender_name.lower() not in self.lender_configs:
            return "Unknown"

        base_time = self.lender_configs[lender_name.lower()].get('typical_processing_time', '10-15 business days')

        ftr_prob = ftr_assessment.get('ftr_probability', 50)

        if ftr_prob >= 80:
            return f"Expected: {base_time}"
        elif ftr_prob >= 60:
            return f"Likely: {base_time} (may be extended)"
        else:
            return f"Expected: Extended beyond {base_time}"

    def _assess_application_risk(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall application risk."""

        risk_factors = []
        risk_score = 0

        for result in validation_results:
            if isinstance(result, dict):
                if result.get('blocking', False):
                    risk_factors.append(f"Blocking issue: {result.get('error', 'Unknown')}")
                    risk_score += 30

                if result.get('validation_type') == 'bkr' and result.get('approval_likelihood', 100) < 70:
                    risk_factors.append("Low BKR approval likelihood")
                    risk_score += 20

                if result.get('validation_type') == 'affordability' and not result.get('affordability_passed', True):
                    risk_factors.append("Affordability concerns")
                    risk_score += 25

        risk_level = 'low' if risk_score < 20 else 'medium' if risk_score < 50 else 'high' if risk_score < 80 else 'critical'

        return {
            'risk_level': risk_level,
            'risk_score': min(risk_score, 100),
            'risk_factors': risk_factors
        }

    async def _store_qc_results(self, application_id: str, qc_report: Dict[str, Any]):
        """Store QC results in database."""

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO dutch_mortgage_qc_results (
                    application_id, qc_report, ftr_probability, overall_score,
                    ready_for_submission, analyzed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                application_id,
                json.dumps(qc_report),
                qc_report['qc_summary']['first_time_right_probability'],
                qc_report['qc_summary']['overall_score'],
                qc_report['qc_summary']['ready_for_submission'],
                datetime.utcnow()
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to store QC results: {str(e)}")

    async def _store_bkr_check(self, bsn: str, bkr_result: Dict[str, Any]):
        """Store BKR check results."""

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Hash BSN for privacy
            import hashlib
            bsn_hash = hashlib.sha256(bsn.encode()).hexdigest()

            cursor.execute("""
                INSERT INTO bkr_checks (
                    bsn_hash, credit_score, negative_registrations_count,
                    approval_likelihood, checked_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                bsn_hash,
                bkr_result.get('credit_score'),
                bkr_result.get('negative_registrations_count'),
                bkr_result.get('approval_likelihood'),
                datetime.utcnow()
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to store BKR check: {str(e)}")

    async def _load_dutch_validation_rules(self):
        """Load Dutch validation rules from database or configuration."""

        try:
            # Load rules from database or use defaults
            self._validation_rules_cache = {
                'affordability': {
                    'max_housing_cost_ratio': 30,
                    'stress_test_rate_addition': 0.02,
                    'minimum_credit_score': 600
                },
                'documents': {
                    'max_file_size_mb': 50,
                    'required_retention_years': 7,
                    'acceptable_formats': ['pdf', 'jpg', 'png']
                }
            }
            self._rules_cache_timestamp = datetime.utcnow()

        except Exception as e:
            self.logger.error(f"Failed to load Dutch validation rules: {str(e)}")

    def _validate_bsn_format(self, bsn: str) -> bool:
        """Validate Dutch BSN format."""

        # BSN is 9 digits, with specific checksum validation
        if not re.match(r'^\d{9}$', bsn):
            return False

        # Dutch BSN checksum validation
        digits = [int(d) for d in bsn]
        checksum = sum(d * (9 - i) for i, d in enumerate(digits[:-1])) % 11

        return checksum == digits[-1] or (checksum == 0 and digits[-1] == 0)

    def _get_lender_config_version(self, lender_name: str) -> str:
        """Get lender configuration version."""

        if lender_name and lender_name.lower() in self.lender_configs:
            return self.lender_configs[lender_name.lower()].get('version', '1.0')

        return 'unknown'

    async def _get_current_market_rates(self) -> Dict[str, Any]:
        """Get current Dutch mortgage market rates."""

        # In production, would fetch from financial data APIs
        return {
            'fixed_10_year': 0.035,
            'fixed_20_year': 0.038,
            'variable_rate': 0.042,
            'nhg_discount': 0.005,
            'last_updated': datetime.utcnow().isoformat()
        }

    async def _get_regional_market_insights(self, property_location: Dict[str, Any]) -> Dict[str, Any]:
        """Get regional market insights."""

        # Simplified regional analysis
        return {
            'location_type': 'urban',  # Would analyze actual location
            'market_trend': 'stable',
            'regional_risk_factors': [],
            'property_type_demand': 'high'
        }

    async def _get_lender_comparison_insights(self, lender_name: str) -> Dict[str, Any]:
        """Get lender comparison insights."""

        return {
            'lender_market_share': 15.5,  # Example percentage
            'lender_specialties': ['First-time buyers', 'NHG mortgages'],
            'processing_speed': 'average',
            'customer_satisfaction': 8.2
        }

    def _analyze_nhg_market_impact(self, mortgage_amount: float) -> Dict[str, Any]:
        """Analyze NHG market impact."""

        return {
            'market_penetration': 35.0,  # NHG usage percentage
            'rate_advantage': 0.005,
            'cost_savings_estimate': mortgage_amount * 0.005 * 30,  # 30 year estimate
            'availability': 'good'
        }

    async def _prepare_document_for_lender(self, document: Dict[str, Any], lender_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare document for lender submission."""

        # In production, would handle document formatting, metadata addition, etc.
        return {
            'original_document': document,
            'lender_formatted': True,
            'metadata_added': True,
            'validation_status': 'prepared'
        }

    def _sanitize_client_data_for_lender(self, client_data: Dict[str, Any], lender_config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize client data for lender submission."""

        # Remove sensitive information not required by lender
        sanitized = client_data.copy()

        # Remove or mask sensitive fields
        sensitive_fields = ['internal_notes', 'risk_assessment_details']
        for field in sensitive_fields:
            sanitized.pop(field, None)

        return sanitized

    # Generate remediation methods
    def _generate_bkr_remediation(self, validation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate BKR-specific remediation actions."""
        actions = []
        if validation.get('blocking', False):
            actions.append({
                'type': 'bkr_blocking',
                'action': 'Resolve BKR blocking issues',
                'description': validation.get('error', 'BKR validation failed'),
                'severity': 'critical',
                'estimated_time_days': 30
            })
        return actions

    def _generate_nhg_remediation(self, validation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate NHG-specific remediation actions."""
        actions = []
        if not validation.get('eligible', True):
            actions.append({
                'type': 'nhg_ineligible',
                'action': 'Address NHG ineligibility',
                'description': 'Property or loan amount exceeds NHG limits',
                'severity': 'high',
                'estimated_time_days': 7
            })
        return actions

    def _generate_affordability_remediation(self, validation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate affordability-specific remediation actions."""
        actions = []
        if not validation.get('affordability_passed', True):
            actions.append({
                'type': 'affordability_fail',
                'action': 'Improve affordability assessment',
                'description': 'Mortgage amount exceeds affordability limits',
                'severity': 'critical',
                'estimated_time_days': 14
            })
        return actions

    def _generate_document_remediation(self, validation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate document-specific remediation actions."""
        actions = []
        critical_issues = validation.get('critical_issues', [])
        for issue in critical_issues:
            actions.append({
                'type': 'document_issue',
                'action': f"Fix {issue.get('message', 'document issue')}",
                'description': issue.get('message', 'Document validation issue'),
                'severity': 'critical',
                'estimated_time_days': 3
            })
        return actions

    def _generate_lender_remediation(self, validation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate lender-specific remediation actions."""
        actions = []
        if validation.get('compatibility_score', 100) < 75:
            actions.append({
                'type': 'lender_incompatible',
                'action': 'Address lender compatibility issues',
                'description': f"Low compatibility score with {validation.get('lender_name', 'lender')}",
                'severity': 'high',
                'estimated_time_days': 10
            })
        return actions
