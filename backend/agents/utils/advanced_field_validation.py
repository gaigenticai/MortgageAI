#!/usr/bin/env python3
"""
Advanced Field Validation Engine
Created: 2024-01-15
Author: MortgageAI Development Team
Description: Advanced field validation system with real-time validation, error correction suggestions, and compliance checking for Dutch mortgage applications.
"""

import re
import json
import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import unicodedata
import phonenumbers
import math
import statistics
from decimal import Decimal, ROUND_HALF_UP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation message severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class FieldType(Enum):
    """Supported field types for validation."""
    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    DATE = "date"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    ENUM = "enum"
    BSN = "bsn"  # Dutch social security number
    IBAN = "iban"
    POSTCODE = "postcode"  # Dutch postal code
    ADDRESS = "address"
    NAME = "name"
    CUSTOM = "custom"

@dataclass
class ValidationRule:
    """Individual validation rule definition."""
    rule_id: str
    rule_name: str
    field_path: str
    field_type: FieldType
    rule_type: str  # required, format, range, length, pattern, custom, afm_compliance
    parameters: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    suggestion_template: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR
    is_active: bool = True
    afm_article: Optional[str] = None  # AFM regulation reference
    priority: int = 1  # 1=highest, 5=lowest
    dependencies: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationMessage:
    """Validation result message."""
    message_id: str
    rule_id: str
    field_path: str
    message: str
    severity: ValidationSeverity
    suggestion: Optional[str] = None
    corrected_value: Optional[str] = None
    afm_reference: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ValidationResult:
    """Complete validation result for a data set."""
    is_valid: bool
    total_fields: int
    validated_fields: int
    errors: int
    warnings: int
    infos: int
    messages: List[ValidationMessage] = field(default_factory=list)
    field_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    compliance_score: float = 0.0
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DutchValidators:
    """Dutch-specific validation utilities."""
    
    @staticmethod
    def validate_bsn(bsn: str) -> Tuple[bool, str]:
        """Validate Dutch BSN (Burgerservicenummer) using the 11-test."""
        try:
            # Remove spaces and dashes
            clean_bsn = re.sub(r'[\s-]', '', str(bsn))
            
            # Must be exactly 9 digits
            if not re.match(r'^\d{9}$', clean_bsn):
                return False, "BSN must be exactly 9 digits"
            
            # Convert to list of integers
            digits = [int(d) for d in clean_bsn]
            
            # Apply 11-test algorithm
            check_sum = sum(digits[i] * (9 - i) for i in range(8))
            check_sum += digits[8] * -1
            
            if check_sum % 11 == 0:
                return True, ""
            else:
                return False, "Invalid BSN checksum"
                
        except Exception as e:
            return False, f"BSN validation error: {str(e)}"
    
    @staticmethod
    def validate_dutch_postcode(postcode: str) -> Tuple[bool, str]:
        """Validate Dutch postal code format (1234AB)."""
        try:
            clean_postcode = re.sub(r'\s+', '', postcode.upper())
            
            if not re.match(r'^\d{4}[A-Z]{2}$', clean_postcode):
                return False, "Dutch postcode must be in format 1234AB"
            
            # Additional validation: first digit cannot be 0
            if clean_postcode[0] == '0':
                return False, "Dutch postcode cannot start with 0"
                
            return True, ""
            
        except Exception as e:
            return False, f"Postcode validation error: {str(e)}"
    
    @staticmethod
    def validate_dutch_phone(phone: str) -> Tuple[bool, str]:
        """Validate Dutch phone number."""
        try:
            # Parse using phonenumbers library
            parsed = phonenumbers.parse(phone, "NL")
            
            if not phonenumbers.is_valid_number(parsed):
                return False, "Invalid Dutch phone number format"
            
            # Check if it's a Dutch number
            if parsed.country_code != 31:
                return False, "Phone number must be a Dutch number"
                
            return True, ""
            
        except Exception as e:
            return False, f"Phone validation error: {str(e)}"
    
    @staticmethod
    def validate_iban(iban: str) -> Tuple[bool, str]:
        """Validate IBAN format and checksum."""
        try:
            # Remove spaces and convert to uppercase
            clean_iban = re.sub(r'\s+', '', iban.upper())
            
            # Basic format check
            if not re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]{4,30}$', clean_iban):
                return False, "Invalid IBAN format"
            
            # Move first 4 characters to the end
            rearranged = clean_iban[4:] + clean_iban[:4]
            
            # Convert letters to numbers (A=10, B=11, etc.)
            numeric_string = ""
            for char in rearranged:
                if char.isdigit():
                    numeric_string += char
                else:
                    numeric_string += str(ord(char) - ord('A') + 10)
            
            # Calculate checksum
            checksum = int(numeric_string) % 97
            
            if checksum == 1:
                return True, ""
            else:
                return False, "Invalid IBAN checksum"
                
        except Exception as e:
            return False, f"IBAN validation error: {str(e)}"

class CorrectionEngine:
    """Intelligent error correction and suggestion engine."""
    
    def __init__(self):
        self.common_corrections = {
            'email': {
                'patterns': [
                    (r'@gmai\.com$', '@gmail.com'),
                    (r'@hotmai\.com$', '@hotmail.com'),
                    (r'@yaho\.com$', '@yahoo.com'),
                    (r'@outloo\.com$', '@outlook.com'),
                    (r'\.co$', '.com'),
                    (r'\.ne$', '.net')
                ]
            },
            'phone': {
                'patterns': [
                    (r'^00', '+'),  # Replace 00 with +
                    (r'^0031', '+31'),  # Dutch international format
                    (r'^06([0-9]{8})$', '+31 6 \\1')  # Dutch mobile format
                ]
            },
            'postcode': {
                'patterns': [
                    (r'^(\d{4})\s*([a-zA-Z]{2})$', '\\1\\2'),  # Remove spaces
                    (r'^(\d{4})([a-zA-Z])([a-zA-Z])$', '\\1\\2\\3')  # Ensure uppercase
                ]
            }
        }
    
    def suggest_correction(self, field_type: FieldType, value: str, 
                          validation_error: str) -> Optional[str]:
        """Generate correction suggestion for invalid field value."""
        try:
            if field_type == FieldType.EMAIL:
                return self._correct_email(value)
            elif field_type == FieldType.PHONE:
                return self._correct_phone(value)
            elif field_type == FieldType.POSTCODE:
                return self._correct_postcode(value)
            elif field_type == FieldType.BSN:
                return self._correct_bsn(value)
            elif field_type == FieldType.IBAN:
                return self._correct_iban(value)
            elif field_type == FieldType.CURRENCY:
                return self._correct_currency(value)
            elif field_type == FieldType.DATE:
                return self._correct_date(value)
            else:
                return self._generic_correction(value, validation_error)
                
        except Exception as e:
            logger.error(f"Error generating correction for {field_type}: {e}")
            return None
    
    def _correct_email(self, value: str) -> Optional[str]:
        """Attempt to correct common email errors."""
        corrected = value.lower().strip()
        
        # Apply common email corrections
        for pattern, replacement in self.common_corrections['email']['patterns']:
            corrected = re.sub(pattern, replacement, corrected)
        
        # Basic email structure correction
        if '@' not in corrected:
            return None
        
        # Split and validate parts
        parts = corrected.split('@')
        if len(parts) != 2:
            return None
            
        local, domain = parts
        
        # Clean local part
        local = re.sub(r'[^\w\.\-\+]', '', local)
        
        # Clean domain part
        domain = re.sub(r'[^\w\.\-]', '', domain)
        
        # Ensure domain has at least one dot
        if '.' not in domain:
            domain += '.com'
        
        return f"{local}@{domain}"
    
    def _correct_phone(self, value: str) -> Optional[str]:
        """Attempt to correct Dutch phone number format."""
        # Remove all non-digit characters except +
        corrected = re.sub(r'[^\d\+]', '', value)
        
        # Apply phone corrections
        for pattern, replacement in self.common_corrections['phone']['patterns']:
            corrected = re.sub(pattern, replacement, corrected)
        
        # Ensure Dutch mobile format
        if corrected.startswith('06') and len(corrected) == 10:
            return f"+31 6 {corrected[2:4]} {corrected[4:6]} {corrected[6:8]} {corrected[8:10]}"
        elif corrected.startswith('+316') and len(corrected) == 13:
            return f"{corrected[:4]} {corrected[4:6]} {corrected[6:8]} {corrected[8:10]} {corrected[10:12]}"
        
        return corrected if corrected != value else None
    
    def _correct_postcode(self, value: str) -> Optional[str]:
        """Attempt to correct Dutch postcode format."""
        corrected = value.upper().strip()
        
        # Remove extra spaces
        corrected = re.sub(r'\s+', '', corrected)
        
        # Ensure proper format
        if re.match(r'^\d{4}[A-Z]{2}$', corrected):
            return f"{corrected[:4]} {corrected[4:]}"
        
        # Try to extract digits and letters
        digits = re.findall(r'\d', corrected)
        letters = re.findall(r'[A-Z]', corrected)
        
        if len(digits) == 4 and len(letters) == 2:
            return f"{''.join(digits)} {''.join(letters)}"
        
        return None
    
    def _correct_bsn(self, value: str) -> Optional[str]:
        """Attempt to correct BSN format."""
        # Extract only digits
        digits = re.findall(r'\d', str(value))
        
        if len(digits) == 9:
            return ''.join(digits)
        elif len(digits) == 8:
            # Try adding leading zero
            return '0' + ''.join(digits)
        
        return None
    
    def _correct_iban(self, value: str) -> Optional[str]:
        """Attempt to correct IBAN format."""
        corrected = value.upper().replace(' ', '')
        
        # Ensure NL prefix for Dutch IBANs
        if len(corrected) == 16 and corrected.isalnum() and not corrected.startswith('NL'):
            corrected = 'NL' + corrected
        
        # Add proper spacing
        if len(corrected) == 18 and corrected.startswith('NL'):
            return f"NL{corrected[2:4]} {corrected[4:8]} {corrected[8:12]} {corrected[12:16]} {corrected[16:18]}"
        
        return corrected if corrected != value else None
    
    def _correct_currency(self, value: str) -> Optional[str]:
        """Attempt to correct currency format."""
        # Extract number from string
        number_str = re.sub(r'[^\d\.\,\-]', '', str(value))
        
        # Handle European decimal notation (comma as decimal separator)
        if ',' in number_str and '.' in number_str:
            # Both comma and dot present - assume dot is thousands separator
            number_str = number_str.replace('.', '').replace(',', '.')
        elif ',' in number_str:
            # Only comma - assume it's decimal separator
            number_str = number_str.replace(',', '.')
        
        try:
            amount = float(number_str)
            return f"€ {amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        except ValueError:
            return None
    
    def _correct_date(self, value: str) -> Optional[str]:
        """Attempt to correct date format."""
        # Common date patterns to try
        patterns = [
            r'(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{4})',  # DD-MM-YYYY
            r'(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})',  # YYYY-MM-DD
            r'(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{2})',  # DD-MM-YY
        ]
        
        for pattern in patterns:
            match = re.match(pattern, value)
            if match:
                day, month, year = match.groups()
                
                # Handle 2-digit years
                if len(year) == 2:
                    year = '20' + year if int(year) < 50 else '19' + year
                
                # Ensure proper format
                try:
                    date_obj = datetime(int(year), int(month), int(day))
                    return date_obj.strftime('%d-%m-%Y')
                except ValueError:
                    # Try swapping day and month
                    try:
                        date_obj = datetime(int(year), int(day), int(month))
                        return date_obj.strftime('%d-%m-%Y')
                    except ValueError:
                        continue
        
        return None
    
    def _generic_correction(self, value: str, error: str) -> Optional[str]:
        """Generic correction attempts based on error type."""
        corrected = str(value).strip()
        
        # Remove common unwanted characters
        if "special characters" in error.lower():
            corrected = re.sub(r'[^\w\s\-\.]', '', corrected)
        
        # Fix common spacing issues
        if "format" in error.lower():
            corrected = re.sub(r'\s+', ' ', corrected)
        
        # Capitalize names
        if "name" in error.lower():
            corrected = corrected.title()
        
        return corrected if corrected != value else None

class AFMComplianceValidator:
    """AFM compliance validation for mortgage application fields."""
    
    def __init__(self):
        self.afm_rules = {
            # Wft Article 86f - Client suitability assessment
            'client_income_verification': {
                'required_documents': ['salary_slip', 'tax_return', 'employment_contract'],
                'minimum_verification_period': 12,  # months
                'article': 'Wft 86f'
            },
            # BGfo Article 8.1 - Product information requirements
            'mortgage_product_disclosure': {
                'required_fields': ['interest_rate', 'apr', 'total_costs', 'early_repayment_penalty'],
                'calculation_method': 'standardized',
                'article': 'BGfo 8.1'
            },
            # Wft Article 86c - Advice documentation
            'advice_documentation': {
                'required_elements': ['client_profile', 'needs_analysis', 'product_selection_rationale', 'risk_explanation'],
                'retention_period': 84,  # months (7 years)
                'article': 'Wft 86c'
            }
        }
    
    def validate_compliance(self, field_path: str, value: Any, 
                          context: Dict[str, Any] = None) -> List[ValidationMessage]:
        """Validate field against AFM compliance requirements."""
        messages = []
        
        try:
            # Income verification compliance
            if 'income' in field_path.lower():
                messages.extend(self._validate_income_compliance(field_path, value, context or {}))
            
            # Documentation requirements
            if 'document' in field_path.lower():
                messages.extend(self._validate_documentation_compliance(field_path, value, context or {}))
            
            # Financial data compliance
            if any(term in field_path.lower() for term in ['amount', 'rate', 'cost', 'price']):
                messages.extend(self._validate_financial_compliance(field_path, value, context or {}))
            
            # Personal data compliance (GDPR/AVG)
            if any(term in field_path.lower() for term in ['name', 'address', 'bsn', 'phone', 'email']):
                messages.extend(self._validate_personal_data_compliance(field_path, value, context or {}))
            
        except Exception as e:
            logger.error(f"AFM compliance validation error for {field_path}: {e}")
            messages.append(ValidationMessage(
                message_id=f"afm_error_{datetime.now().timestamp()}",
                rule_id="afm_validation_error",
                field_path=field_path,
                message=f"AFM compliance validation error: {str(e)}",
                severity=ValidationSeverity.WARNING
            ))
        
        return messages
    
    def _validate_income_compliance(self, field_path: str, value: Any, 
                                  context: Dict[str, Any]) -> List[ValidationMessage]:
        """Validate income-related fields for AFM compliance."""
        messages = []
        
        # Check if income verification documents are referenced
        if isinstance(value, (int, float)) and value > 0:
            required_docs = self.afm_rules['client_income_verification']['required_documents']
            
            # Check if context contains verification documents
            has_verification = any(doc in str(context).lower() for doc in required_docs)
            
            if not has_verification:
                messages.append(ValidationMessage(
                    message_id=f"afm_income_{datetime.now().timestamp()}",
                    rule_id="afm_income_verification",
                    field_path=field_path,
                    message="Income must be verified with required documentation per AFM regulations",
                    severity=ValidationSeverity.ERROR,
                    suggestion=f"Provide one of: {', '.join(required_docs)}",
                    afm_reference="Wft Article 86f"
                ))
        
        return messages
    
    def _validate_documentation_compliance(self, field_path: str, value: Any, 
                                         context: Dict[str, Any]) -> List[ValidationMessage]:
        """Validate documentation requirements for AFM compliance."""
        messages = []
        
        # Check document retention and completeness
        if isinstance(value, str) and value.strip():
            required_elements = self.afm_rules['advice_documentation']['required_elements']
            
            # Basic completeness check
            if len(value.strip()) < 50:  # Minimal documentation threshold
                messages.append(ValidationMessage(
                    message_id=f"afm_doc_{datetime.now().timestamp()}",
                    rule_id="afm_documentation_completeness",
                    field_path=field_path,
                    message="Documentation appears insufficient for AFM compliance requirements",
                    severity=ValidationSeverity.WARNING,
                    suggestion="Ensure documentation includes all required elements",
                    afm_reference="Wft Article 86c"
                ))
        
        return messages
    
    def _validate_financial_compliance(self, field_path: str, value: Any, 
                                     context: Dict[str, Any]) -> List[ValidationMessage]:
        """Validate financial data for AFM compliance."""
        messages = []
        
        # Check for proper financial disclosure
        if isinstance(value, (int, float)) and value > 0:
            # Interest rate disclosure requirements
            if 'rate' in field_path.lower():
                if value > 0.15:  # 15% seems unusually high
                    messages.append(ValidationMessage(
                        message_id=f"afm_rate_{datetime.now().timestamp()}",
                        rule_id="afm_rate_disclosure",
                        field_path=field_path,
                        message="High interest rate requires additional disclosure and client confirmation",
                        severity=ValidationSeverity.WARNING,
                        suggestion="Ensure proper risk disclosure and client acknowledgment",
                        afm_reference="BGfo Article 8.1"
                    ))
        
        return messages
    
    def _validate_personal_data_compliance(self, field_path: str, value: Any, 
                                         context: Dict[str, Any]) -> List[ValidationMessage]:
        """Validate personal data handling for GDPR/AVG compliance."""
        messages = []
        
        # Check for sensitive data handling
        if isinstance(value, str) and value.strip():
            # BSN handling requirements
            if 'bsn' in field_path.lower():
                messages.append(ValidationMessage(
                    message_id=f"gdpr_bsn_{datetime.now().timestamp()}",
                    rule_id="gdpr_bsn_handling",
                    field_path=field_path,
                    message="BSN data requires special handling and encryption",
                    severity=ValidationSeverity.INFO,
                    suggestion="Ensure BSN is encrypted in storage and transmission",
                    afm_reference="AVG Article 32"
                ))
        
        return messages

class AdvancedFieldValidationEngine:
    """Main validation engine with comprehensive field validation capabilities."""
    
    def __init__(self):
        self.rules: Dict[str, ValidationRule] = {}
        self.correction_engine = CorrectionEngine()
        self.afm_validator = AFMComplianceValidator()
        self.dutch_validators = DutchValidators()
        
        # Performance tracking
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'average_processing_time': 0.0,
            'rule_usage_count': {},
            'field_type_stats': {}
        }
        
        # Initialize default validation rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default validation rules for common fields."""
        
        # Email validation rules
        self.add_rule(ValidationRule(
            rule_id="email_required",
            rule_name="Email Required",
            field_path="email",
            field_type=FieldType.EMAIL,
            rule_type="required",
            error_message="Email address is required",
            severity=ValidationSeverity.ERROR
        ))
        
        self.add_rule(ValidationRule(
            rule_id="email_format",
            rule_name="Email Format",
            field_path="email",
            field_type=FieldType.EMAIL,
            rule_type="format",
            parameters={'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
            error_message="Invalid email format",
            suggestion_template="Use format: name@domain.com",
            severity=ValidationSeverity.ERROR
        ))
        
        # Phone validation rules
        self.add_rule(ValidationRule(
            rule_id="phone_format_nl",
            rule_name="Dutch Phone Format",
            field_path="phone",
            field_type=FieldType.PHONE,
            rule_type="format",
            error_message="Invalid Dutch phone number format",
            suggestion_template="Use format: +31 6 12 34 56 78 or 06 12345678",
            severity=ValidationSeverity.ERROR
        ))
        
        # BSN validation rules
        self.add_rule(ValidationRule(
            rule_id="bsn_required",
            rule_name="BSN Required",
            field_path="bsn",
            field_type=FieldType.BSN,
            rule_type="required",
            error_message="BSN (Burgerservicenummer) is required",
            severity=ValidationSeverity.ERROR,
            afm_article="Wft 86f"
        ))
        
        self.add_rule(ValidationRule(
            rule_id="bsn_format",
            rule_name="BSN Format",
            field_path="bsn",
            field_type=FieldType.BSN,
            rule_type="format",
            error_message="Invalid BSN format",
            suggestion_template="BSN must be 9 digits",
            severity=ValidationSeverity.ERROR,
            afm_article="Wft 86f"
        ))
        
        # Income validation rules
        self.add_rule(ValidationRule(
            rule_id="income_required",
            rule_name="Income Required",
            field_path="monthly_income",
            field_type=FieldType.CURRENCY,
            rule_type="required",
            error_message="Monthly income is required",
            severity=ValidationSeverity.ERROR,
            afm_article="Wft 86f"
        ))
        
        self.add_rule(ValidationRule(
            rule_id="income_range",
            rule_name="Income Range",
            field_path="monthly_income",
            field_type=FieldType.CURRENCY,
            rule_type="range",
            parameters={'min_value': 0, 'max_value': 50000},
            error_message="Monthly income must be between €0 and €50,000",
            severity=ValidationSeverity.ERROR
        ))
        
        # Mortgage amount validation
        self.add_rule(ValidationRule(
            rule_id="mortgage_amount_required",
            rule_name="Mortgage Amount Required",
            field_path="mortgage_amount",
            field_type=FieldType.CURRENCY,
            rule_type="required",
            error_message="Mortgage amount is required",
            severity=ValidationSeverity.ERROR
        ))
        
        # Date validation rules
        self.add_rule(ValidationRule(
            rule_id="birth_date_format",
            rule_name="Birth Date Format",
            field_path="birth_date",
            field_type=FieldType.DATE,
            rule_type="format",
            error_message="Invalid birth date format",
            suggestion_template="Use format: DD-MM-YYYY",
            severity=ValidationSeverity.ERROR
        ))
        
        self.add_rule(ValidationRule(
            rule_id="birth_date_range",
            rule_name="Birth Date Range",
            field_path="birth_date",
            field_type=FieldType.DATE,
            rule_type="range",
            parameters={'min_age': 18, 'max_age': 100},
            error_message="Age must be between 18 and 100 years",
            severity=ValidationSeverity.ERROR
        ))
        
        # Postcode validation
        self.add_rule(ValidationRule(
            rule_id="postcode_format_nl",
            rule_name="Dutch Postcode Format",
            field_path="postcode",
            field_type=FieldType.POSTCODE,
            rule_type="format",
            error_message="Invalid Dutch postcode format",
            suggestion_template="Use format: 1234 AB",
            severity=ValidationSeverity.ERROR
        ))
        
        # IBAN validation
        self.add_rule(ValidationRule(
            rule_id="iban_format",
            rule_name="IBAN Format",
            field_path="iban",
            field_type=FieldType.IBAN,
            rule_type="format",
            error_message="Invalid IBAN format",
            suggestion_template="Use format: NL91 ABNA 0417 1643 00",
            severity=ValidationSeverity.ERROR
        ))
    
    def add_rule(self, rule: ValidationRule) -> bool:
        """Add a new validation rule."""
        try:
            self.rules[rule.rule_id] = rule
            logger.info(f"Added validation rule: {rule.rule_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add validation rule {rule.rule_id}: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a validation rule."""
        try:
            if rule_id in self.rules:
                del self.rules[rule_id]
                logger.info(f"Removed validation rule: {rule_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove validation rule {rule_id}: {e}")
            return False
    
    def get_rule(self, rule_id: str) -> Optional[ValidationRule]:
        """Get a validation rule by ID."""
        return self.rules.get(rule_id)
    
    def list_rules(self, field_path: str = None, 
                   field_type: FieldType = None) -> List[ValidationRule]:
        """List validation rules, optionally filtered."""
        rules = list(self.rules.values())
        
        if field_path:
            rules = [r for r in rules if r.field_path == field_path]
        
        if field_type:
            rules = [r for r in rules if r.field_type == field_type]
        
        return sorted(rules, key=lambda r: r.priority)
    
    async def validate_field(self, field_path: str, value: Any, 
                           field_type: FieldType = None,
                           context: Dict[str, Any] = None) -> List[ValidationMessage]:
        """Validate a single field against all applicable rules."""
        start_time = datetime.now()
        messages = []
        
        try:
            # Get applicable rules for this field
            applicable_rules = [
                rule for rule in self.rules.values()
                if rule.is_active and (
                    rule.field_path == field_path or 
                    rule.field_path in field_path or
                    field_path.endswith(rule.field_path.split('.')[-1])
                )
            ]
            
            # Sort by priority
            applicable_rules.sort(key=lambda r: r.priority)
            
            for rule in applicable_rules:
                # Check rule conditions if any
                if not self._check_rule_conditions(rule, context or {}):
                    continue
                
                # Validate based on rule type
                validation_messages = await self._validate_with_rule(
                    rule, field_path, value, context or {}
                )
                messages.extend(validation_messages)
                
                # Update usage statistics
                self.validation_stats['rule_usage_count'][rule.rule_id] = \
                    self.validation_stats['rule_usage_count'].get(rule.rule_id, 0) + 1
            
            # AFM compliance validation
            if context and context.get('check_afm_compliance', True):
                afm_messages = self.afm_validator.validate_compliance(
                    field_path, value, context
                )
                messages.extend(afm_messages)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_validation_stats(field_path, field_type, processing_time, len(messages) == 0)
            
        except Exception as e:
            logger.error(f"Field validation error for {field_path}: {e}")
            messages.append(ValidationMessage(
                message_id=f"validation_error_{datetime.now().timestamp()}",
                rule_id="system_error",
                field_path=field_path,
                message=f"Validation system error: {str(e)}",
                severity=ValidationSeverity.CRITICAL
            ))
        
        return messages
    
    async def validate_data(self, data: Dict[str, Any], 
                          validation_config: Dict[str, Any] = None) -> ValidationResult:
        """Validate a complete data set."""
        start_time = datetime.now()
        all_messages = []
        field_scores = {}
        
        try:
            config = validation_config or {}
            
            # Flatten nested data for validation
            flattened_data = self._flatten_data(data)
            
            # Validate each field
            for field_path, value in flattened_data.items():
                field_messages = await self.validate_field(
                    field_path, value, context=config.get('context', {})
                )
                all_messages.extend(field_messages)
                
                # Calculate field score
                error_count = sum(1 for m in field_messages if m.severity == ValidationSeverity.ERROR)
                warning_count = sum(1 for m in field_messages if m.severity == ValidationSeverity.WARNING)
                
                field_score = max(0, 100 - (error_count * 30) - (warning_count * 10))
                field_scores[field_path] = field_score
            
            # Calculate summary statistics
            total_fields = len(flattened_data)
            validated_fields = len(field_scores)
            errors = sum(1 for m in all_messages if m.severity == ValidationSeverity.ERROR)
            warnings = sum(1 for m in all_messages if m.severity == ValidationSeverity.WARNING)
            infos = sum(1 for m in all_messages if m.severity == ValidationSeverity.INFO)
            
            # Calculate overall scores
            overall_score = statistics.mean(field_scores.values()) if field_scores else 0
            compliance_score = self._calculate_compliance_score(all_messages)
            
            # Create result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ValidationResult(
                is_valid=(errors == 0),
                total_fields=total_fields,
                validated_fields=validated_fields,
                errors=errors,
                warnings=warnings,
                infos=infos,
                messages=all_messages,
                field_scores=field_scores,
                overall_score=overall_score,
                compliance_score=compliance_score,
                processing_time=processing_time,
                metadata={
                    'validation_config': config,
                    'total_rules_applied': len(self.rules),
                    'afm_compliance_checked': config.get('context', {}).get('check_afm_compliance', True)
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ValidationResult(
                is_valid=False,
                total_fields=0,
                validated_fields=0,
                errors=1,
                warnings=0,
                infos=0,
                messages=[ValidationMessage(
                    message_id=f"system_error_{datetime.now().timestamp()}",
                    rule_id="system_error",
                    field_path="system",
                    message=f"Validation system error: {str(e)}",
                    severity=ValidationSeverity.CRITICAL
                )],
                processing_time=processing_time,
                metadata={'error': str(e)}
            )
    
    async def _validate_with_rule(self, rule: ValidationRule, field_path: str, 
                                value: Any, context: Dict[str, Any]) -> List[ValidationMessage]:
        """Apply a specific validation rule to a field value."""
        messages = []
        
        try:
            # Handle different rule types
            if rule.rule_type == "required":
                if not self._is_value_present(value):
                    messages.append(self._create_validation_message(
                        rule, field_path, rule.error_message
                    ))
            
            elif rule.rule_type == "format":
                format_valid, format_error = await self._validate_format(
                    rule, field_path, value
                )
                if not format_valid:
                    suggestion = self.correction_engine.suggest_correction(
                        rule.field_type, str(value) if value else "", format_error
                    )
                    
                    message = self._create_validation_message(
                        rule, field_path, format_error or rule.error_message,
                        suggestion=suggestion
                    )
                    messages.append(message)
            
            elif rule.rule_type == "range":
                range_valid, range_error = self._validate_range(rule, value)
                if not range_valid:
                    messages.append(self._create_validation_message(
                        rule, field_path, range_error or rule.error_message
                    ))
            
            elif rule.rule_type == "length":
                length_valid, length_error = self._validate_length(rule, value)
                if not length_valid:
                    messages.append(self._create_validation_message(
                        rule, field_path, length_error or rule.error_message
                    ))
            
            elif rule.rule_type == "pattern":
                pattern_valid, pattern_error = self._validate_pattern(rule, value)
                if not pattern_valid:
                    suggestion = self.correction_engine.suggest_correction(
                        rule.field_type, str(value) if value else "", pattern_error
                    )
                    
                    message = self._create_validation_message(
                        rule, field_path, pattern_error or rule.error_message,
                        suggestion=suggestion
                    )
                    messages.append(message)
            
            elif rule.rule_type == "custom":
                custom_messages = await self._validate_custom(rule, field_path, value, context)
                messages.extend(custom_messages)
            
            elif rule.rule_type == "afm_compliance":
                afm_messages = self.afm_validator.validate_compliance(field_path, value, context)
                messages.extend(afm_messages)
        
        except Exception as e:
            logger.error(f"Rule validation error for {rule.rule_id}: {e}")
            messages.append(ValidationMessage(
                message_id=f"rule_error_{datetime.now().timestamp()}",
                rule_id=rule.rule_id,
                field_path=field_path,
                message=f"Rule validation error: {str(e)}",
                severity=ValidationSeverity.WARNING
            ))
        
        return messages
    
    async def _validate_format(self, rule: ValidationRule, field_path: str, 
                             value: Any) -> Tuple[bool, str]:
        """Validate field format based on field type."""
        if not value:
            return True, ""  # Empty values are handled by required rule
        
        str_value = str(value).strip()
        
        try:
            if rule.field_type == FieldType.EMAIL:
                pattern = rule.parameters.get('pattern', r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
                if not re.match(pattern, str_value):
                    return False, "Invalid email format"
                return True, ""
            
            elif rule.field_type == FieldType.PHONE:
                return self.dutch_validators.validate_dutch_phone(str_value)
            
            elif rule.field_type == FieldType.BSN:
                return self.dutch_validators.validate_bsn(str_value)
            
            elif rule.field_type == FieldType.POSTCODE:
                return self.dutch_validators.validate_dutch_postcode(str_value)
            
            elif rule.field_type == FieldType.IBAN:
                return self.dutch_validators.validate_iban(str_value)
            
            elif rule.field_type == FieldType.DATE:
                return self._validate_date_format(str_value)
            
            elif rule.field_type == FieldType.CURRENCY:
                return self._validate_currency_format(str_value)
            
            elif rule.field_type == FieldType.PERCENTAGE:
                return self._validate_percentage_format(str_value)
            
            else:
                # Generic pattern validation
                pattern = rule.parameters.get('pattern')
                if pattern and not re.match(pattern, str_value):
                    return False, f"Value does not match required pattern"
                
                return True, ""
        
        except Exception as e:
            return False, f"Format validation error: {str(e)}"
    
    def _validate_date_format(self, value: str) -> Tuple[bool, str]:
        """Validate date format."""
        date_formats = [
            '%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y',
            '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
            '%d-%m-%y', '%d/%m/%y', '%d.%m.%y'
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(value, fmt)
                # Check for reasonable date range
                if not (datetime(1900, 1, 1) <= parsed_date <= datetime.now()):
                    return False, "Date is outside reasonable range"
                return True, ""
            except ValueError:
                continue
        
        return False, "Invalid date format. Use DD-MM-YYYY"
    
    def _validate_currency_format(self, value: str) -> Tuple[bool, str]:
        """Validate currency format."""
        # Remove common currency symbols and separators
        clean_value = re.sub(r'[€$£\s,]', '', value)
        
        try:
            amount = float(clean_value)
            if amount < 0:
                return False, "Currency amount cannot be negative"
            return True, ""
        except ValueError:
            return False, "Invalid currency format. Use numbers only"
    
    def _validate_percentage_format(self, value: str) -> Tuple[bool, str]:
        """Validate percentage format."""
        clean_value = value.replace('%', '').strip()
        
        try:
            percentage = float(clean_value)
            if not (0 <= percentage <= 100):
                return False, "Percentage must be between 0 and 100"
            return True, ""
        except ValueError:
            return False, "Invalid percentage format"
    
    def _validate_range(self, rule: ValidationRule, value: Any) -> Tuple[bool, str]:
        """Validate value range."""
        try:
            min_val = rule.parameters.get('min_value')
            max_val = rule.parameters.get('max_value')
            
            # Convert value to number if possible
            if isinstance(value, str):
                # Try to extract number from string
                clean_value = re.sub(r'[€$£,\s%]', '', value)
                numeric_value = float(clean_value)
            else:
                numeric_value = float(value)
            
            if min_val is not None and numeric_value < min_val:
                return False, f"Value must be at least {min_val}"
            
            if max_val is not None and numeric_value > max_val:
                return False, f"Value must be at most {max_val}"
            
            # Age-specific validation for dates
            if 'age' in rule.parameters:
                min_age = rule.parameters.get('min_age', 0)
                max_age = rule.parameters.get('max_age', 150)
                
                # Assume value is a birth date, calculate age
                try:
                    birth_date = datetime.strptime(str(value), '%d-%m-%Y')
                    age = (datetime.now() - birth_date).days // 365
                    
                    if not (min_age <= age <= max_age):
                        return False, f"Age must be between {min_age} and {max_age} years"
                except:
                    return False, "Invalid date for age calculation"
            
            return True, ""
        
        except Exception as e:
            return False, f"Range validation error: {str(e)}"
    
    def _validate_length(self, rule: ValidationRule, value: Any) -> Tuple[bool, str]:
        """Validate value length."""
        try:
            str_value = str(value) if value else ""
            min_length = rule.parameters.get('min_length', 0)
            max_length = rule.parameters.get('max_length', 1000000)
            
            if len(str_value) < min_length:
                return False, f"Value must be at least {min_length} characters"
            
            if len(str_value) > max_length:
                return False, f"Value must be at most {max_length} characters"
            
            return True, ""
        
        except Exception as e:
            return False, f"Length validation error: {str(e)}"
    
    def _validate_pattern(self, rule: ValidationRule, value: Any) -> Tuple[bool, str]:
        """Validate value against regex pattern."""
        try:
            if not value:
                return True, ""
            
            pattern = rule.parameters.get('pattern')
            if not pattern:
                return True, ""
            
            if not re.match(pattern, str(value)):
                return False, f"Value does not match required pattern"
            
            return True, ""
        
        except Exception as e:
            return False, f"Pattern validation error: {str(e)}"
    
    async def _validate_custom(self, rule: ValidationRule, field_path: str, 
                             value: Any, context: Dict[str, Any]) -> List[ValidationMessage]:
        """Handle custom validation logic."""
        messages = []
        
        try:
            # Custom validation based on rule ID
            if rule.rule_id.startswith('debt_to_income'):
                messages.extend(self._validate_debt_to_income_ratio(field_path, value, context))
            elif rule.rule_id.startswith('loan_to_value'):
                messages.extend(self._validate_loan_to_value_ratio(field_path, value, context))
            elif rule.rule_id.startswith('affordability'):
                messages.extend(self._validate_affordability(field_path, value, context))
            
        except Exception as e:
            logger.error(f"Custom validation error for {rule.rule_id}: {e}")
            messages.append(ValidationMessage(
                message_id=f"custom_error_{datetime.now().timestamp()}",
                rule_id=rule.rule_id,
                field_path=field_path,
                message=f"Custom validation error: {str(e)}",
                severity=ValidationSeverity.WARNING
            ))
        
        return messages
    
    def _validate_debt_to_income_ratio(self, field_path: str, value: Any, 
                                     context: Dict[str, Any]) -> List[ValidationMessage]:
        """Validate debt-to-income ratio."""
        messages = []
        
        try:
            monthly_income = context.get('monthly_income', 0)
            monthly_debt = float(value) if value else 0
            
            if monthly_income > 0:
                ratio = monthly_debt / monthly_income
                
                if ratio > 0.36:  # 36% DTI threshold
                    messages.append(ValidationMessage(
                        message_id=f"dti_{datetime.now().timestamp()}",
                        rule_id="debt_to_income_ratio",
                        field_path=field_path,
                        message=f"Debt-to-income ratio ({ratio:.1%}) exceeds recommended maximum (36%)",
                        severity=ValidationSeverity.ERROR,
                        suggestion=f"Reduce monthly debt by €{(monthly_debt - monthly_income * 0.36):.2f}",
                        context={'calculated_ratio': ratio, 'threshold': 0.36}
                    ))
                elif ratio > 0.28:  # Warning threshold
                    messages.append(ValidationMessage(
                        message_id=f"dti_warning_{datetime.now().timestamp()}",
                        rule_id="debt_to_income_ratio_warning",
                        field_path=field_path,
                        message=f"Debt-to-income ratio ({ratio:.1%}) is approaching recommended maximum",
                        severity=ValidationSeverity.WARNING,
                        suggestion="Consider reducing debt before applying for additional credit",
                        context={'calculated_ratio': ratio, 'threshold': 0.28}
                    ))
        
        except Exception as e:
            logger.error(f"DTI validation error: {e}")
        
        return messages
    
    def _validate_loan_to_value_ratio(self, field_path: str, value: Any, 
                                    context: Dict[str, Any]) -> List[ValidationMessage]:
        """Validate loan-to-value ratio."""
        messages = []
        
        try:
            property_value = context.get('property_value', 0)
            loan_amount = float(value) if value else 0
            
            if property_value > 0:
                ratio = loan_amount / property_value
                
                if ratio > 1.0:  # 100% LTV
                    messages.append(ValidationMessage(
                        message_id=f"ltv_{datetime.now().timestamp()}",
                        rule_id="loan_to_value_ratio",
                        field_path=field_path,
                        message=f"Loan-to-value ratio ({ratio:.1%}) exceeds maximum allowed (100%)",
                        severity=ValidationSeverity.ERROR,
                        suggestion=f"Reduce loan amount by €{loan_amount - property_value:.2f} or increase down payment",
                        context={'calculated_ratio': ratio, 'threshold': 1.0}
                    ))
                elif ratio > 0.9:  # Warning threshold
                    messages.append(ValidationMessage(
                        message_id=f"ltv_warning_{datetime.now().timestamp()}",
                        rule_id="loan_to_value_ratio_warning",
                        field_path=field_path,
                        message=f"High loan-to-value ratio ({ratio:.1%}) may affect loan terms",
                        severity=ValidationSeverity.WARNING,
                        suggestion="Consider increasing down payment for better loan terms",
                        context={'calculated_ratio': ratio, 'threshold': 0.9}
                    ))
        
        except Exception as e:
            logger.error(f"LTV validation error: {e}")
        
        return messages
    
    def _validate_affordability(self, field_path: str, value: Any, 
                              context: Dict[str, Any]) -> List[ValidationMessage]:
        """Validate affordability based on Dutch lending standards."""
        messages = []
        
        try:
            monthly_income = context.get('monthly_income', 0)
            mortgage_payment = float(value) if value else 0
            
            if monthly_income > 0:
                # Dutch affordability calculation (simplified)
                max_payment = monthly_income * 0.28  # 28% of gross income
                
                if mortgage_payment > max_payment:
                    messages.append(ValidationMessage(
                        message_id=f"affordability_{datetime.now().timestamp()}",
                        rule_id="dutch_affordability",
                        field_path=field_path,
                        message=f"Mortgage payment (€{mortgage_payment:.2f}) exceeds affordability limit (€{max_payment:.2f})",
                        severity=ValidationSeverity.ERROR,
                        suggestion=f"Reduce mortgage payment by €{mortgage_payment - max_payment:.2f} or increase income",
                        context={'max_affordable': max_payment, 'ratio_used': 0.28}
                    ))
        
        except Exception as e:
            logger.error(f"Affordability validation error: {e}")
        
        return messages
    
    def _is_value_present(self, value: Any) -> bool:
        """Check if a value is considered present/valid."""
        if value is None:
            return False
        
        if isinstance(value, str):
            return len(value.strip()) > 0
        
        if isinstance(value, (list, dict)):
            return len(value) > 0
        
        if isinstance(value, (int, float)):
            return not math.isnan(float(value))
        
        return bool(value)
    
    def _check_rule_conditions(self, rule: ValidationRule, context: Dict[str, Any]) -> bool:
        """Check if rule conditions are met."""
        if not rule.conditions:
            return True
        
        try:
            # Simple condition checking
            for key, expected_value in rule.conditions.items():
                if key not in context:
                    return False
                
                if context[key] != expected_value:
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Rule condition check error for {rule.rule_id}: {e}")
            return False
    
    def _create_validation_message(self, rule: ValidationRule, field_path: str, 
                                 message: str, suggestion: str = None) -> ValidationMessage:
        """Create a validation message from a rule."""
        return ValidationMessage(
            message_id=f"{rule.rule_id}_{datetime.now().timestamp()}",
            rule_id=rule.rule_id,
            field_path=field_path,
            message=message,
            severity=rule.severity,
            suggestion=suggestion or rule.suggestion_template,
            afm_reference=rule.afm_article,
            context={'rule_name': rule.rule_name, 'rule_type': rule.rule_type}
        )
    
    def _flatten_data(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary for validation."""
        flattened = {}
        
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(self._flatten_data(value, new_key))
            else:
                flattened[new_key] = value
        
        return flattened
    
    def _calculate_compliance_score(self, messages: List[ValidationMessage]) -> float:
        """Calculate AFM compliance score based on validation messages."""
        afm_messages = [m for m in messages if m.afm_reference]
        
        if not afm_messages:
            return 100.0
        
        # Weight penalties by severity
        penalty = 0
        for msg in afm_messages:
            if msg.severity == ValidationSeverity.CRITICAL:
                penalty += 50
            elif msg.severity == ValidationSeverity.ERROR:
                penalty += 30
            elif msg.severity == ValidationSeverity.WARNING:
                penalty += 10
        
        return max(0, 100 - penalty)
    
    def _update_validation_stats(self, field_path: str, field_type: FieldType, 
                               processing_time: float, success: bool):
        """Update internal validation statistics."""
        self.validation_stats['total_validations'] += 1
        
        if success:
            self.validation_stats['successful_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
        
        # Update average processing time
        current_avg = self.validation_stats['average_processing_time']
        total_validations = self.validation_stats['total_validations']
        
        new_avg = ((current_avg * (total_validations - 1)) + processing_time) / total_validations
        self.validation_stats['average_processing_time'] = new_avg
        
        # Update field type statistics
        if field_type:
            field_type_key = field_type.value
            if field_type_key not in self.validation_stats['field_type_stats']:
                self.validation_stats['field_type_stats'][field_type_key] = {
                    'count': 0, 'success_count': 0, 'average_time': 0.0
                }
            
            stats = self.validation_stats['field_type_stats'][field_type_key]
            stats['count'] += 1
            if success:
                stats['success_count'] += 1
            
            # Update average time for this field type
            stats['average_time'] = ((stats['average_time'] * (stats['count'] - 1)) + processing_time) / stats['count']
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation engine statistics."""
        return {
            'validation_stats': self.validation_stats.copy(),
            'total_rules': len(self.rules),
            'active_rules': len([r for r in self.rules.values() if r.is_active]),
            'rule_breakdown': {
                rule_type: len([r for r in self.rules.values() if r.rule_type == rule_type])
                for rule_type in set(r.rule_type for r in self.rules.values())
            },
            'field_type_coverage': {
                field_type.value: len([r for r in self.rules.values() if r.field_type == field_type])
                for field_type in FieldType
            }
        }
    
    async def export_validation_rules(self, format: str = "json") -> str:
        """Export validation rules in specified format."""
        try:
            rules_data = []
            
            for rule in self.rules.values():
                rule_dict = {
                    'rule_id': rule.rule_id,
                    'rule_name': rule.rule_name,
                    'field_path': rule.field_path,
                    'field_type': rule.field_type.value,
                    'rule_type': rule.rule_type,
                    'parameters': rule.parameters,
                    'error_message': rule.error_message,
                    'suggestion_template': rule.suggestion_template,
                    'severity': rule.severity.value,
                    'is_active': rule.is_active,
                    'afm_article': rule.afm_article,
                    'priority': rule.priority,
                    'dependencies': rule.dependencies,
                    'conditions': rule.conditions,
                    'created_at': rule.created_at.isoformat(),
                    'updated_at': rule.updated_at.isoformat()
                }
                rules_data.append(rule_dict)
            
            if format.lower() == "json":
                return json.dumps(rules_data, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
        
        except Exception as e:
            logger.error(f"Rule export error: {e}")
            raise
    
    async def import_validation_rules(self, rules_data: str, format: str = "json", 
                                    merge: bool = False) -> int:
        """Import validation rules from specified format."""
        try:
            if format.lower() == "json":
                imported_rules = json.loads(rules_data)
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            if not merge:
                self.rules.clear()
            
            imported_count = 0
            
            for rule_data in imported_rules:
                try:
                    rule = ValidationRule(
                        rule_id=rule_data['rule_id'],
                        rule_name=rule_data['rule_name'],
                        field_path=rule_data['field_path'],
                        field_type=FieldType(rule_data['field_type']),
                        rule_type=rule_data['rule_type'],
                        parameters=rule_data.get('parameters', {}),
                        error_message=rule_data.get('error_message', ''),
                        suggestion_template=rule_data.get('suggestion_template', ''),
                        severity=ValidationSeverity(rule_data.get('severity', 'error')),
                        is_active=rule_data.get('is_active', True),
                        afm_article=rule_data.get('afm_article'),
                        priority=rule_data.get('priority', 1),
                        dependencies=rule_data.get('dependencies', []),
                        conditions=rule_data.get('conditions', {}),
                        created_at=datetime.fromisoformat(rule_data.get('created_at', datetime.now().isoformat())),
                        updated_at=datetime.fromisoformat(rule_data.get('updated_at', datetime.now().isoformat()))
                    )
                    
                    self.add_rule(rule)
                    imported_count += 1
                    
                except Exception as rule_error:
                    logger.error(f"Failed to import rule {rule_data.get('rule_id', 'unknown')}: {rule_error}")
                    continue
            
            logger.info(f"Successfully imported {imported_count} validation rules")
            return imported_count
        
        except Exception as e:
            logger.error(f"Rule import error: {e}")
            raise

# Global validation engine instance
_validation_engine = None

def get_validation_engine() -> AdvancedFieldValidationEngine:
    """Get the global validation engine instance."""
    global _validation_engine
    if _validation_engine is None:
        _validation_engine = AdvancedFieldValidationEngine()
    return _validation_engine

# Main execution function for testing
async def main():
    """Main function for testing the validation engine."""
    try:
        engine = AdvancedFieldValidationEngine()
        
        # Test data
        test_data = {
            'personal_info': {
                'first_name': 'Jan',
                'last_name': 'de Vries',
                'email': 'jan@example.com',
                'phone': '06 12345678',
                'bsn': '123456782',
                'birth_date': '15-03-1985',
                'postcode': '1012 AB',
                'iban': 'NL91 ABNA 0417 1643 00'
            },
            'financial_info': {
                'monthly_income': 4500.00,
                'monthly_debt': 800.00,
                'mortgage_amount': 350000.00,
                'property_value': 400000.00
            }
        }
        
        # Validate the data
        result = await engine.validate_data(test_data, {
            'context': {
                'check_afm_compliance': True,
                'monthly_income': 4500.00,
                'property_value': 400000.00
            }
        })
        
        # Print results
        print(f"\n=== Validation Results ===")
        print(f"Overall Valid: {result.is_valid}")
        print(f"Overall Score: {result.overall_score:.1f}%")
        print(f"Compliance Score: {result.compliance_score:.1f}%")
        print(f"Processing Time: {result.processing_time:.3f}s")
        print(f"Total Fields: {result.total_fields}")
        print(f"Errors: {result.errors}, Warnings: {result.warnings}, Info: {result.infos}")
        
        if result.messages:
            print(f"\n=== Validation Messages ===")
            for msg in result.messages:
                print(f"[{msg.severity.value.upper()}] {msg.field_path}: {msg.message}")
                if msg.suggestion:
                    print(f"  Suggestion: {msg.suggestion}")
                if msg.afm_reference:
                    print(f"  AFM Reference: {msg.afm_reference}")
        
        # Print statistics
        stats = engine.get_validation_statistics()
        print(f"\n=== Engine Statistics ===")
        print(f"Total Validations: {stats['validation_stats']['total_validations']}")
        print(f"Success Rate: {stats['validation_stats']['successful_validations'] / max(1, stats['validation_stats']['total_validations']):.1%}")
        print(f"Average Processing Time: {stats['validation_stats']['average_processing_time']:.3f}s")
        print(f"Active Rules: {stats['active_rules']}/{stats['total_rules']}")
        
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
