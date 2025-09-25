"""
Mortgage Application Quality Control Agent Implementation

This module implements automated quality control for mortgage applications,
ensuring completeness, accuracy, and compliance before submission to lenders.
"""

import json
import logging
import asyncio
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import statistics

import cv2
import numpy as np
import pytesseract
from PIL import Image
import pdfplumber
import tabula

from ..config import settings
from ..database import get_db_connection
from ..utils.document_processor import DocumentProcessor
from ..utils.validation_engine import ValidationEngine
from ..utils.anomaly_detector import AnomalyDetector


class QualityControlAgent:
    """
    Automated QC Agent for mortgage application validation.

    Performs comprehensive checks:
    1. Document ingestion and OCR processing
    2. Field-level validation against schemas
    3. Anomaly detection in financial data
    4. Completeness scoring (>95% target)
    5. Automated remediation suggestions
    """

    def __init__(self):
        """Initialize the QC Agent with required components."""
        self.document_processor = DocumentProcessor()
        self.validation_engine = ValidationEngine()
        self.anomaly_detector = AnomalyDetector()
        self.logger = logging.getLogger(__name__)

        # QC thresholds
        self.completeness_threshold = settings.QC_COMPLETENESS_THRESHOLD
        self.anomaly_threshold = settings.ANOMALY_DETECTION_THRESHOLD

        # Required fields by document type
        self.required_fields = {
            'application_form': [
                'applicant_name', 'date_of_birth', 'address', 'income',
                'mortgage_amount', 'property_value', 'loan_term'
            ],
            'income_proof': [
                'employer_name', 'annual_income', 'tax_year', 'signature'
            ],
            'property_documents': [
                'property_address', 'valuation_amount', 'valuation_date'
            ],
            'id_document': [
                'document_number', 'expiry_date', 'issue_date'
            ]
        }

    async def analyze_application(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze mortgage application for completeness and accuracy.

        Args:
            payload: Dictionary containing applicant data and document paths

        Returns:
            Comprehensive QC analysis results
        """
        try:
            application_id = payload.get('application_id')
            applicant_data = payload.get('applicant_data', {})
            documents = payload.get('documents', [])

            self.logger.info(f"Starting QC analysis for application {application_id}")

            # Process all documents
            processed_docs = []
            for doc in documents:
                processed_doc = await self._process_document(doc)
                processed_docs.append(processed_doc)

            # Extract and validate fields
            field_validation = await self._validate_fields(applicant_data, processed_docs)

            # Detect anomalies
            anomaly_check = await self._detect_anomalies(applicant_data, processed_docs)

            # Calculate completeness score
            completeness_score = self._calculate_completeness_score(field_validation, processed_docs)

            # Generate remediation instructions
            remediation = await self._generate_remediation_instructions(
                field_validation, anomaly_check, completeness_score
            )

            result = {
                'application_id': application_id,
                'completeness_score': completeness_score,
                'passed': completeness_score >= self.completeness_threshold,
                'field_validation': field_validation,
                'anomaly_check': anomaly_check,
                'document_analysis': processed_docs,
                'remediation_instructions': remediation,
                'analyzed_at': datetime.utcnow().isoformat(),
                'processing_summary': {
                    'documents_processed': len(processed_docs),
                    'fields_validated': len(field_validation.get('results', [])),
                    'anomalies_found': len(anomaly_check.get('anomalies', [])),
                    'critical_issues': len([r for r in remediation if r.get('severity') == 'critical'])
                }
            }

            self.logger.info(f"QC analysis completed for application {application_id}: score {completeness_score:.1f}%")
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing application: {str(e)}")
            raise

    async def process_document(self, document_path: str, document_type: str) -> Dict[str, Any]:
        """
        Process a single document for OCR and data extraction.

        Args:
            document_path: Path to the document file
            document_type: Type of document (application_form, income_proof, etc.)

        Returns:
            Processed document data
        """
        try:
            return await self.document_processor.process_document(document_path, document_type)
        except Exception as e:
            self.logger.error(f"Error processing document {document_path}: {str(e)}")
            raise

    async def validate_application_fields(self, applicant_data: Dict[str, Any], extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate application fields against requirements and cross-reference data.

        Args:
            applicant_data: Applicant-provided data
            extracted_data: Data extracted from documents

        Returns:
            Field validation results
        """
        try:
            return await self.validation_engine.validate_fields(applicant_data, extracted_data)
        except Exception as e:
            self.logger.error(f"Error validating fields: {str(e)}")
            raise

    async def detect_anomalies(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies in application data using statistical analysis.

        Args:
            application_data: Complete application data

        Returns:
            Anomaly detection results
        """
        try:
            return await self.anomaly_detector.detect_anomalies(application_data)
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            raise

    def _calculate_completeness_score(self, field_validation: Dict[str, Any], documents: List[Dict[str, Any]]) -> float:
        """
        Calculate overall completeness score based on validation results.

        Args:
            field_validation: Results from field validation
            documents: Processed document data

        Returns:
            Completeness score (0-100)
        """
        try:
            total_checks = 0
            passed_checks = 0

            # Count field validation results
            validation_results = field_validation.get('results', [])
            total_checks += len(validation_results)
            passed_checks += len([r for r in validation_results if r.get('valid', False)])

            # Count document processing results
            for doc in documents:
                if doc.get('processing_status') == 'success':
                    extracted_fields = doc.get('extracted_data', {})
                    total_checks += len(extracted_fields)
                    # Assume extracted fields are valid if present
                    passed_checks += len(extracted_fields)

            # Calculate score
            if total_checks == 0:
                return 0.0

            score = (passed_checks / total_checks) * 100
            return round(score, 1)

        except Exception as e:
            self.logger.error(f"Error calculating completeness score: {str(e)}")
            return 0.0

    async def _process_document(self, document_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document from the application."""
        try:
            doc_path = document_info.get('path')
            doc_type = document_info.get('type', 'unknown')

            if not doc_path or not Path(doc_path).exists():
                return {
                    'document_type': doc_type,
                    'processing_status': 'error',
                    'error': 'Document file not found',
                    'extracted_data': {}
                }

            # Process document
            processed_data = await self.process_document(doc_path, doc_type)

            return {
                'document_type': doc_type,
                'processing_status': 'success',
                'file_path': doc_path,
                'extracted_data': processed_data,
                'required_fields': self.required_fields.get(doc_type, []),
                'completeness': self._check_document_completeness(processed_data, doc_type)
            }

        except Exception as e:
            return {
                'document_type': document_info.get('type', 'unknown'),
                'processing_status': 'error',
                'error': str(e),
                'extracted_data': {}
            }

    async def _validate_fields(self, applicant_data: Dict[str, Any], documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate fields across all documents and applicant data."""
        try:
            all_extracted_data = {}
            for doc in documents:
                if doc.get('processing_status') == 'success':
                    all_extracted_data.update(doc.get('extracted_data', {}))

            # Cross-validate applicant data with extracted data
            validation_results = await self.validate_application_fields(applicant_data, all_extracted_data)

            return {
                'results': validation_results,
                'cross_validation_passed': self._check_cross_validation(applicant_data, all_extracted_data),
                'summary': {
                    'total_fields': len(validation_results),
                    'valid_fields': len([r for r in validation_results if r.get('valid')]),
                    'invalid_fields': len([r for r in validation_results if not r.get('valid')])
                }
            }

        except Exception as e:
            self.logger.error(f"Error in field validation: {str(e)}")
            return {'results': [], 'cross_validation_passed': False, 'error': str(e)}

    async def _detect_anomalies(self, applicant_data: Dict[str, Any], documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect anomalies in the application data."""
        try:
            # Combine all data for anomaly detection
            combined_data = {**applicant_data}
            for doc in documents:
                if doc.get('processing_status') == 'success':
                    combined_data.update(doc.get('extracted_data', {}))

            anomaly_results = await self.detect_anomalies(combined_data)

            return {
                'anomalies': anomaly_results.get('anomalies', []),
                'severity_score': anomaly_results.get('severity_score', 0),
                'requires_review': len(anomaly_results.get('anomalies', [])) > 0
            }

        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            return {'anomalies': [], 'severity_score': 0, 'error': str(e)}

    async def _generate_remediation_instructions(self, field_validation: Dict[str, Any], anomaly_check: Dict[str, Any], completeness_score: float) -> List[Dict[str, Any]]:
        """Generate specific remediation instructions for identified issues."""
        instructions = []

        # Field validation issues
        for result in field_validation.get('results', []):
            if not result.get('valid', False):
                instructions.append({
                    'type': 'field_correction',
                    'field': result.get('field'),
                    'issue': result.get('error'),
                    'severity': result.get('severity', 'medium'),
                    'instruction': self._generate_field_fix_instruction(result)
                })

        # Anomaly issues
        for anomaly in anomaly_check.get('anomalies', []):
            instructions.append({
                'type': 'anomaly_resolution',
                'field': anomaly.get('field'),
                'issue': anomaly.get('description'),
                'severity': anomaly.get('severity', 'high'),
                'instruction': self._generate_anomaly_fix_instruction(anomaly)
            })

        # Completeness issues
        if completeness_score < self.completeness_threshold:
            instructions.append({
                'type': 'completeness_improvement',
                'issue': f'Application completeness is {completeness_score:.1f}%, below {self.completeness_threshold}% threshold',
                'severity': 'critical' if completeness_score < 80 else 'high',
                'instruction': 'Please review all required fields and provide missing information or documents.'
            })

        return instructions

    def _check_document_completeness(self, extracted_data: Dict[str, Any], doc_type: str) -> float:
        """Check completeness of a single document."""
        required = self.required_fields.get(doc_type, [])
        if not required:
            return 100.0

        found_fields = 0
        for field in required:
            if field in extracted_data and extracted_data[field]:
                found_fields += 1

        return round((found_fields / len(required)) * 100, 1)

    def _check_cross_validation(self, applicant_data: Dict[str, Any], extracted_data: Dict[str, Any]) -> bool:
        """Check if applicant data matches extracted document data."""
        # Simple cross-validation checks
        checks = []

        # Name consistency
        applicant_name = applicant_data.get('applicant_name', '').lower().strip()
        extracted_name = extracted_data.get('applicant_name', '').lower().strip()
        if applicant_name and extracted_name:
            checks.append(applicant_name == extracted_name)

        # Amount consistency
        applicant_amount = applicant_data.get('mortgage_amount')
        extracted_amount = extracted_data.get('mortgage_amount')
        if applicant_amount and extracted_amount:
            checks.append(abs(applicant_amount - extracted_amount) < 1)  # Allow small differences

        return all(checks) if checks else True

    def _generate_field_fix_instruction(self, validation_result: Dict[str, Any]) -> str:
        """Generate specific fix instruction for a field validation error."""
        field = validation_result.get('field', 'unknown')
        error = validation_result.get('error', 'invalid value')

        instructions = {
            'missing': f"Please provide the {field.replace('_', ' ')}.",
            'invalid_format': f"The {field.replace('_', ' ')} format is invalid. Please check and correct.",
            'out_of_range': f"The {field.replace('_', ' ')} is outside acceptable range. Please verify.",
            'inconsistent': f"The {field.replace('_', ' ')} is inconsistent with other information. Please review."
        }

        return instructions.get(error, f"Please correct the {field.replace('_', ' ')}.")

    def _generate_anomaly_fix_instruction(self, anomaly: Dict[str, Any]) -> str:
        """Generate fix instruction for an anomaly."""
        field = anomaly.get('field', 'unknown')
        anomaly_type = anomaly.get('type', 'unknown')

        instructions = {
            'unusual_ratio': f"The {field.replace('_', ' ')} ratio seems unusual. Please verify the amounts are correct.",
            'outlier_value': f"The {field.replace('_', ' ')} appears to be an outlier. Please double-check this value.",
            'missing_correlation': f"Expected relationship between {field.replace('_', ' ')} and related fields not found. Please review.",
            'data_inconsistency': f"Inconsistency detected in {field.replace('_', ' ')}. Please ensure all data is accurate."
        }

        return instructions.get(anomaly_type, f"Please review the {field.replace('_', ' ')} for accuracy.")
