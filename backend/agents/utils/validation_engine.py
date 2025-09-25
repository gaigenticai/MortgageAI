"""
Validation Engine for Mortgage Application Fields

This module provides comprehensive field validation for mortgage applications,
including cross-validation, format checking, and business rule enforcement.
"""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import asyncio

from ..config import settings


class ValidationEngine:
    """
    Advanced validation engine for mortgage application data.

    Validates:
    - Field format and data type
    - Business rules and constraints
    - Cross-field consistency
    - Regulatory compliance
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Validation rules for different field types
        self.field_validators = {
            'applicant_name': self._validate_name,
            'date_of_birth': self._validate_date_of_birth,
            'address': self._validate_address,
            'mortgage_amount': self._validate_mortgage_amount,
            'property_value': self._validate_property_value,
            'income': self._validate_income,
            'loan_term': self._validate_loan_term,
            'loan_to_value_ratio': self._validate_ltv_ratio,
            'debt_to_income_ratio': self._validate_dti_ratio,
            'credit_score': self._validate_credit_score,
            'employer_name': self._validate_employer_name,
            'document_number': self._validate_document_number,
            'expiry_date': self._validate_expiry_date,
        }

        # Business rules
        self.business_rules = {
            'minimum_age': 18,
            'maximum_age': 75,
            'minimum_income': 15000,  # Annual minimum
            'maximum_ltv': 0.95,  # 95% LTV
            'maximum_dti': 0.45,  # 45% DTI
            'minimum_loan': 50000,
            'maximum_loan': 2000000,
            'minimum_term': 5,  # years
            'maximum_term': 40,  # years
        }

    async def validate_fields(self, applicant_data: Dict[str, Any], extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate all fields in mortgage application data.

        Args:
            applicant_data: Data provided by applicant
            extracted_data: Data extracted from documents

        Returns:
            List of validation results for each field
        """
        try:
            validation_results = []

            # Combine all data for validation
            all_data = {**applicant_data, **extracted_data}

            # Validate individual fields
            for field_name, validator in self.field_validators.items():
                if field_name in all_data:
                    result = await validator(all_data[field_name], all_data)
                    validation_results.append({
                        'field': field_name,
                        'value': all_data[field_name],
                        'valid': result['valid'],
                        'error': result.get('error'),
                        'severity': result.get('severity', 'medium'),
                        'source': 'applicant' if field_name in applicant_data else 'extracted'
                    })

            # Cross-field validation
            cross_validation_results = await self._validate_cross_fields(all_data)
            validation_results.extend(cross_validation_results)

            # Business rule validation
            business_validation_results = await self._validate_business_rules(all_data)
            validation_results.extend(business_validation_results)

            self.logger.info(f"Validated {len(validation_results)} fields: {len([r for r in validation_results if r['valid']])} passed")
            return validation_results

        except Exception as e:
            self.logger.error(f"Error in field validation: {str(e)}")
            return []

    async def _validate_name(self, value: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate applicant name field."""
        try:
            if not isinstance(value, str) or not value.strip():
                return {'valid': False, 'error': 'missing', 'severity': 'high'}

            name = value.strip()

            # Check length
            if len(name) < 2 or len(name) > 100:
                return {'valid': False, 'error': 'invalid_length', 'severity': 'medium'}

            # Check for valid characters (allow letters, spaces, hyphens, apostrophes)
            if not re.match(r"^[A-Za-z\s\-']+$", name):
                return {'valid': False, 'error': 'invalid_characters', 'severity': 'medium'}

            # Check for at least one letter
            if not re.search(r'[A-Za-z]', name):
                return {'valid': False, 'error': 'no_letters', 'severity': 'high'}

            return {'valid': True}

        except Exception:
            return {'valid': False, 'error': 'validation_error', 'severity': 'medium'}

    async def _validate_date_of_birth(self, value: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate date of birth."""
        try:
            if not value:
                return {'valid': False, 'error': 'missing', 'severity': 'high'}

            # Parse date
            if isinstance(value, str):
                try:
                    dob = datetime.strptime(value, '%Y-%m-%d').date()
                except ValueError:
                    return {'valid': False, 'error': 'invalid_format', 'severity': 'medium'}
            elif isinstance(value, date):
                dob = value
            else:
                return {'valid': False, 'error': 'invalid_type', 'severity': 'medium'}

            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

            # Check age constraints
            if age < self.business_rules['minimum_age']:
                return {'valid': False, 'error': 'too_young', 'severity': 'high'}

            if age > self.business_rules['maximum_age']:
                return {'valid': False, 'error': 'too_old', 'severity': 'high'}

            return {'valid': True}

        except Exception:
            return {'valid': False, 'error': 'validation_error', 'severity': 'medium'}

    async def _validate_address(self, value: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate address field."""
        try:
            if not isinstance(value, str) or not value.strip():
                return {'valid': False, 'error': 'missing', 'severity': 'high'}

            address = value.strip()

            # Check minimum length
            if len(address) < 10:
                return {'valid': False, 'error': 'too_short', 'severity': 'medium'}

            # Check for basic address components
            has_number = bool(re.search(r'\d+', address))
            has_street = bool(re.search(r'\b[A-Za-z]{3,}\b', address))

            if not (has_number and has_street):
                return {'valid': False, 'error': 'incomplete_address', 'severity': 'medium'}

            return {'valid': True}

        except Exception:
            return {'valid': False, 'error': 'validation_error', 'severity': 'medium'}

    async def _validate_mortgage_amount(self, value: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate mortgage amount."""
        try:
            amount = self._parse_numeric(value)
            if amount is None:
                return {'valid': False, 'error': 'missing', 'severity': 'high'}

            # Check range
            if amount < self.business_rules['minimum_loan']:
                return {'valid': False, 'error': 'too_small', 'severity': 'high'}

            if amount > self.business_rules['maximum_loan']:
                return {'valid': False, 'error': 'too_large', 'severity': 'high'}

            return {'valid': True}

        except Exception:
            return {'valid': False, 'error': 'validation_error', 'severity': 'medium'}

    async def _validate_property_value(self, value: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate property value."""
        try:
            amount = self._parse_numeric(value)
            if amount is None:
                return {'valid': False, 'error': 'missing', 'severity': 'high'}

            # Property value should be reasonable
            if amount < 50000 or amount > 10000000:
                return {'valid': False, 'error': 'out_of_range', 'severity': 'medium'}

            return {'valid': True}

        except Exception:
            return {'valid': False, 'error': 'validation_error', 'severity': 'medium'}

    async def _validate_income(self, value: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate income field."""
        try:
            income = self._parse_numeric(value)
            if income is None:
                return {'valid': False, 'error': 'missing', 'severity': 'high'}

            if income < self.business_rules['minimum_income']:
                return {'valid': False, 'error': 'too_low', 'severity': 'high'}

            return {'valid': True}

        except Exception:
            return {'valid': False, 'error': 'validation_error', 'severity': 'medium'}

    async def _validate_loan_term(self, value: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate loan term."""
        try:
            term = self._parse_numeric(value)
            if term is None:
                return {'valid': False, 'error': 'missing', 'severity': 'medium'}

            if term < self.business_rules['minimum_term'] or term > self.business_rules['maximum_term']:
                return {'valid': False, 'error': 'out_of_range', 'severity': 'medium'}

            return {'valid': True}

        except Exception:
            return {'valid': False, 'error': 'validation_error', 'severity': 'medium'}

    async def _validate_ltv_ratio(self, value: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate loan-to-value ratio."""
        try:
            ltv = self._parse_numeric(value)
            if ltv is None:
                # Calculate if possible
                mortgage_amount = all_data.get('mortgage_amount')
                property_value = all_data.get('property_value')

                if mortgage_amount and property_value and property_value > 0:
                    ltv = mortgage_amount / property_value
                else:
                    return {'valid': False, 'error': 'cannot_calculate', 'severity': 'medium'}

            if ltv > self.business_rules['maximum_ltv']:
                return {'valid': False, 'error': 'too_high', 'severity': 'high'}

            if ltv < 0.1:  # Minimum 10% equity
                return {'valid': False, 'error': 'too_low', 'severity': 'medium'}

            return {'valid': True}

        except Exception:
            return {'valid': False, 'error': 'validation_error', 'severity': 'medium'}

    async def _validate_dti_ratio(self, value: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate debt-to-income ratio."""
        try:
            dti = self._parse_numeric(value)
            if dti is None:
                # Calculate if possible
                mortgage_amount = all_data.get('mortgage_amount')
                income = all_data.get('income')
                term = all_data.get('loan_term', 25)

                if mortgage_amount and income and income > 0:
                    # Simple monthly payment calculation (approximate)
                    monthly_payment = mortgage_amount / (term * 12)
                    dti = (monthly_payment * 12) / income
                else:
                    return {'valid': False, 'error': 'cannot_calculate', 'severity': 'medium'}

            if dti > self.business_rules['maximum_dti']:
                return {'valid': False, 'error': 'too_high', 'severity': 'high'}

            return {'valid': True}

        except Exception:
            return {'valid': False, 'error': 'validation_error', 'severity': 'medium'}

    async def _validate_credit_score(self, value: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate credit score."""
        try:
            score = self._parse_numeric(value)
            if score is None:
                return {'valid': False, 'error': 'missing', 'severity': 'medium'}

            # Dutch credit scores typically range from 0-1000 or similar
            if score < 0 or score > 1200:
                return {'valid': False, 'error': 'out_of_range', 'severity': 'medium'}

            return {'valid': True}

        except Exception:
            return {'valid': False, 'error': 'validation_error', 'severity': 'medium'}

    async def _validate_employer_name(self, value: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate employer name."""
        try:
            if not isinstance(value, str) or not value.strip():
                return {'valid': False, 'error': 'missing', 'severity': 'medium'}

            employer = value.strip()

            if len(employer) < 2:
                return {'valid': False, 'error': 'too_short', 'severity': 'low'}

            return {'valid': True}

        except Exception:
            return {'valid': False, 'error': 'validation_error', 'severity': 'low'}

    async def _validate_document_number(self, value: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document number (ID, passport, etc.)."""
        try:
            if not isinstance(value, str) or not value.strip():
                return {'valid': False, 'error': 'missing', 'severity': 'high'}

            doc_num = value.strip()

            # Basic format check (alphanumeric, reasonable length)
            if not re.match(r'^[A-Z0-9\-]+$', doc_num):
                return {'valid': False, 'error': 'invalid_format', 'severity': 'medium'}

            if len(doc_num) < 5 or len(doc_num) > 20:
                return {'valid': False, 'error': 'invalid_length', 'severity': 'medium'}

            return {'valid': True}

        except Exception:
            return {'valid': False, 'error': 'validation_error', 'severity': 'medium'}

    async def _validate_expiry_date(self, value: Any, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document expiry date."""
        try:
            if not value:
                return {'valid': False, 'error': 'missing', 'severity': 'high'}

            # Parse date
            if isinstance(value, str):
                try:
                    expiry = datetime.strptime(value, '%Y-%m-%d').date()
                except ValueError:
                    return {'valid': False, 'error': 'invalid_format', 'severity': 'medium'}
            elif isinstance(value, date):
                expiry = value
            else:
                return {'valid': False, 'error': 'invalid_type', 'severity': 'medium'}

            today = date.today()

            # Should not be expired
            if expiry <= today:
                return {'valid': False, 'error': 'expired', 'severity': 'critical'}

            # Should not be too far in the future (max 20 years for IDs)
            max_future = today.replace(year=today.year + 20)
            if expiry > max_future:
                return {'valid': False, 'error': 'too_far_future', 'severity': 'medium'}

            return {'valid': True}

        except Exception:
            return {'valid': False, 'error': 'validation_error', 'severity': 'medium'}

    async def _validate_cross_fields(self, all_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate relationships between fields."""
        validations = []

        # Mortgage amount vs property value
        mortgage_amount = all_data.get('mortgage_amount')
        property_value = all_data.get('property_value')

        if mortgage_amount and property_value:
            ltv = mortgage_amount / property_value
            if ltv > self.business_rules['maximum_ltv']:
                validations.append({
                    'field': 'mortgage_property_consistency',
                    'value': f"LTV: {ltv:.2%}",
                    'valid': False,
                    'error': 'ltv_too_high',
                    'severity': 'high'
                })

        # Income vs mortgage affordability
        income = all_data.get('income')
        if mortgage_amount and income:
            term = all_data.get('loan_term', 25)
            monthly_payment = mortgage_amount / (term * 12)
            dti = (monthly_payment * 12) / income

            if dti > self.business_rules['maximum_dti']:
                validations.append({
                    'field': 'income_mortgage_affordability',
                    'value': f"DTI: {dti:.2%}",
                    'valid': False,
                    'error': 'dti_too_high',
                    'severity': 'high'
                })

        # Age vs loan term
        dob = all_data.get('date_of_birth')
        term = all_data.get('loan_term')

        if dob and term:
            try:
                if isinstance(dob, str):
                    dob_date = datetime.strptime(dob, '%Y-%m-%d').date()
                else:
                    dob_date = dob

                age = date.today().year - dob_date.year
                retirement_age = 67  # Typical retirement age

                if age + term > retirement_age:
                    validations.append({
                        'field': 'age_term_consistency',
                        'value': f"Age: {age}, Term: {term} years",
                        'valid': False,
                        'error': 'term_extends_beyond_retirement',
                        'severity': 'medium'
                    })
            except (ValueError, TypeError):
                pass

        return validations

    async def _validate_business_rules(self, all_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate business rules and regulatory requirements."""
        validations = []

        # Check for required AFM disclosures in advice text
        advice_text = all_data.get('advice_text', '')
        if advice_text:
            required_disclosures = [
                'fees', 'commissions', 'risks', 'independent advice'
            ]

            for disclosure in required_disclosures:
                if disclosure.lower() not in advice_text.lower():
                    validations.append({
                        'field': f'afm_disclosure_{disclosure}',
                        'value': advice_text[:100] + '...',
                        'valid': False,
                        'error': f'missing_{disclosure}_disclosure',
                        'severity': 'high'
                    })

        return validations

    def _parse_numeric(self, value: Any) -> Optional[float]:
        """Parse numeric value from various formats."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                # Remove common separators and currency symbols
                cleaned = re.sub(r'[€£$\s,]', '', value)
                return float(cleaned)
            return None
        except (ValueError, TypeError):
            return None
