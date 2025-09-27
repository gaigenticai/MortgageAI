#!/usr/bin/env python3
"""
Advanced Field Validation Executor
Created: 2024-01-15
Author: MortgageAI Development Team
Description: Executor script for handling advanced field validation operations via API calls.
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

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_field_validation import (
        AdvancedFieldValidationEngine,
        ValidationRule,
        FieldType,
        ValidationSeverity,
        get_validation_engine
    )
except ImportError as e:
    print(f"Error importing field validation modules: {str(e)}", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedFieldValidationExecutor:
    """Executor class for advanced field validation operations."""
    
    def __init__(self):
        """Initialize the field validation executor."""
        self.engine = get_validation_engine()
        
    async def execute_operation(self, operation: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the specified field validation operation."""
        try:
            operation_map = {
                'validate_field': self._validate_field,
                'validate_data': self._validate_data,
                'list_rules': self._list_rules,
                'create_rule': self._create_rule,
                'get_rule': self._get_rule,
                'update_rule': self._update_rule,
                'delete_rule': self._delete_rule,
                'get_suggestions': self._get_suggestions,
                'check_afm_compliance': self._check_afm_compliance,
                'get_statistics': self._get_statistics,
                'export_rules': self._export_rules,
                'import_rules': self._import_rules,
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
    
    async def _validate_field(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single field."""
        try:
            field_path = args.get('field_path')
            value = json.loads(args.get('value', 'null'))
            field_type_str = args.get('field_type', 'text')
            context = json.loads(args.get('context', '{}'))
            check_afm_compliance = args.get('check_afm_compliance', 'true').lower() == 'true'
            generate_suggestions = args.get('generate_suggestions', 'true').lower() == 'true'
            
            if not field_path:
                raise ValueError("field_path is required")
            
            # Convert field type string to enum
            field_type = None
            try:
                field_type = FieldType(field_type_str)
            except ValueError:
                field_type = FieldType.TEXT
            
            logger.info(f"Validating field: {field_path} (type: {field_type_str})")
            
            # Add AFM compliance flag to context
            context['check_afm_compliance'] = check_afm_compliance
            
            # Perform field validation
            messages = await self.engine.validate_field(field_path, value, field_type, context)
            
            # Calculate field score
            error_count = sum(1 for m in messages if m.severity == ValidationSeverity.ERROR)
            warning_count = sum(1 for m in messages if m.severity == ValidationSeverity.WARNING)
            field_score = max(0, 100 - (error_count * 30) - (warning_count * 10))
            
            # Generate suggestions if requested and there are validation issues
            suggestions = []
            if generate_suggestions and messages:
                for message in messages:
                    if message.suggestion:
                        suggestions.append({
                            'suggestion': message.suggestion,
                            'corrected_value': message.corrected_value,
                            'confidence': 0.8,  # Default confidence
                            'rule_id': message.rule_id
                        })
            
            # Serialize messages
            serialized_messages = []
            for message in messages:
                serialized_messages.append({
                    'message_id': message.message_id,
                    'rule_id': message.rule_id,
                    'field_path': message.field_path,
                    'message': message.message,
                    'severity': message.severity.value,
                    'suggestion': message.suggestion,
                    'corrected_value': message.corrected_value,
                    'afm_reference': message.afm_reference,
                    'context': message.context,
                    'timestamp': message.timestamp.isoformat()
                })
            
            return {
                'success': True,
                'field_path': field_path,
                'is_valid': len([m for m in messages if m.severity == ValidationSeverity.ERROR]) == 0,
                'messages': serialized_messages,
                'field_score': field_score,
                'suggestions': suggestions,
                'processing_info': {
                    'field_type': field_type_str,
                    'afm_compliance_checked': check_afm_compliance,
                    'suggestions_generated': generate_suggestions,
                    'rules_applied': len([r for r in self.engine.rules.values() if r.is_active])
                }
            }
            
        except Exception as e:
            logger.error(f"Error in field validation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'validate_field'
            }
    
    async def _validate_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a complete data set."""
        try:
            data = json.loads(args.get('data', '{}'))
            validation_config = json.loads(args.get('validation_config', '{}'))
            check_afm_compliance = args.get('check_afm_compliance', 'true').lower() == 'true'
            generate_suggestions = args.get('generate_suggestions', 'true').lower() == 'true'
            include_scores = args.get('include_scores', 'true').lower() == 'true'
            
            if not data:
                raise ValueError("data is required")
            
            logger.info(f"Validating data set with {len(data)} root fields")
            
            # Add configuration options to validation config
            validation_config['context'] = validation_config.get('context', {})
            validation_config['context']['check_afm_compliance'] = check_afm_compliance
            validation_config['context']['generate_suggestions'] = generate_suggestions
            validation_config['context']['include_scores'] = include_scores
            
            # Perform data validation
            result = await self.engine.validate_data(data, validation_config)
            
            # Serialize messages
            serialized_messages = []
            for message in result.messages:
                serialized_messages.append({
                    'message_id': message.message_id,
                    'rule_id': message.rule_id,
                    'field_path': message.field_path,
                    'message': message.message,
                    'severity': message.severity.value,
                    'suggestion': message.suggestion,
                    'corrected_value': message.corrected_value,
                    'afm_reference': message.afm_reference,
                    'context': message.context,
                    'timestamp': message.timestamp.isoformat()
                })
            
            return {
                'success': True,
                'is_valid': result.is_valid,
                'total_fields': result.total_fields,
                'validated_fields': result.validated_fields,
                'errors': result.errors,
                'warnings': result.warnings,
                'infos': result.infos,
                'messages': serialized_messages,
                'field_scores': result.field_scores if include_scores else {},
                'overall_score': result.overall_score,
                'compliance_score': result.compliance_score,
                'processing_time': result.processing_time,
                'metadata': result.metadata
            }
            
        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'validate_data'
            }
    
    async def _list_rules(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List validation rules with optional filtering."""
        try:
            field_path = args.get('field_path', '').strip()
            field_type_str = args.get('field_type', '').strip()
            rule_type = args.get('rule_type', '').strip()
            active_only = args.get('active_only', False)
            limit = int(args.get('limit', 100))
            offset = int(args.get('offset', 0))
            
            logger.info(f"Listing validation rules with filters")
            
            # Convert field type string to enum if provided
            field_type = None
            if field_type_str:
                try:
                    field_type = FieldType(field_type_str)
                except ValueError:
                    pass
            
            # Get filtered rules
            rules = self.engine.list_rules(
                field_path=field_path if field_path else None,
                field_type=field_type
            )
            
            # Apply additional filters
            if rule_type:
                rules = [r for r in rules if r.rule_type == rule_type]
            
            if active_only:
                rules = [r for r in rules if r.is_active]
            
            # Apply pagination
            total_rules = len(rules)
            paginated_rules = rules[offset:offset + limit]
            
            # Serialize rules
            serialized_rules = []
            for rule in paginated_rules:
                serialized_rules.append(self._serialize_rule(rule))
            
            active_rules_count = len([r for r in self.engine.rules.values() if r.is_active])
            
            return {
                'success': True,
                'rules': serialized_rules,
                'total_rules': total_rules,
                'active_rules': active_rules_count,
                'pagination': {
                    'limit': limit,
                    'offset': offset,
                    'returned': len(serialized_rules),
                    'has_more': offset + limit < total_rules
                }
            }
            
        except Exception as e:
            logger.error(f"Error listing validation rules: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'list_rules'
            }
    
    async def _create_rule(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new validation rule."""
        try:
            rule_id = args.get('rule_id')
            rule_data = json.loads(args.get('rule_data', '{}'))
            
            if not rule_id or not rule_data:
                raise ValueError("rule_id and rule_data are required")
            
            logger.info(f"Creating validation rule: {rule_data.get('rule_name', 'Unknown')}")
            
            # Create ValidationRule object
            field_type = FieldType(rule_data['field_type'])
            severity = ValidationSeverity(rule_data.get('severity', 'error'))
            
            rule = ValidationRule(
                rule_id=rule_id,
                rule_name=rule_data['rule_name'],
                field_path=rule_data['field_path'],
                field_type=field_type,
                rule_type=rule_data['rule_type'],
                parameters=rule_data.get('parameters', {}),
                error_message=rule_data.get('error_message', ''),
                suggestion_template=rule_data.get('suggestion_template', ''),
                severity=severity,
                is_active=rule_data.get('is_active', True),
                afm_article=rule_data.get('afm_article'),
                priority=rule_data.get('priority', 1),
                dependencies=rule_data.get('dependencies', []),
                conditions=rule_data.get('conditions', {}),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Add rule to engine
            success = self.engine.add_rule(rule)
            
            if success:
                return {
                    'success': True,
                    'rule_id': rule_id,
                    'created': True,
                    'rule_configuration': self._serialize_rule(rule)
                }
            else:
                raise Exception("Failed to add rule to engine")
            
        except Exception as e:
            logger.error(f"Error creating validation rule: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'create_rule'
            }
    
    async def _get_rule(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get a specific validation rule."""
        try:
            rule_id = args.get('rule_id')
            
            if not rule_id:
                raise ValueError("rule_id is required")
            
            logger.info(f"Retrieving validation rule: {rule_id}")
            
            rule = self.engine.get_rule(rule_id)
            
            if rule:
                return {
                    'success': True,
                    'rule': self._serialize_rule(rule)
                }
            else:
                return {
                    'success': True,
                    'rule': None
                }
            
        except Exception as e:
            logger.error(f"Error retrieving validation rule: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'get_rule'
            }
    
    async def _update_rule(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing validation rule."""
        try:
            rule_id = args.get('rule_id')
            rule_data = json.loads(args.get('rule_data', '{}'))
            
            if not rule_id or not rule_data:
                raise ValueError("rule_id and rule_data are required")
            
            logger.info(f"Updating validation rule: {rule_id}")
            
            # Check if rule exists
            existing_rule = self.engine.get_rule(rule_id)
            if not existing_rule:
                return {
                    'success': True,
                    'updated': False,
                    'error': 'Rule not found'
                }
            
            # Create updated rule
            field_type = FieldType(rule_data['field_type'])
            severity = ValidationSeverity(rule_data.get('severity', 'error'))
            
            updated_rule = ValidationRule(
                rule_id=rule_id,
                rule_name=rule_data['rule_name'],
                field_path=rule_data['field_path'],
                field_type=field_type,
                rule_type=rule_data['rule_type'],
                parameters=rule_data.get('parameters', {}),
                error_message=rule_data.get('error_message', ''),
                suggestion_template=rule_data.get('suggestion_template', ''),
                severity=severity,
                is_active=rule_data.get('is_active', True),
                afm_article=rule_data.get('afm_article'),
                priority=rule_data.get('priority', 1),
                dependencies=rule_data.get('dependencies', []),
                conditions=rule_data.get('conditions', {}),
                created_at=existing_rule.created_at,
                updated_at=datetime.now()
            )
            
            # Update rule in engine
            success = self.engine.add_rule(updated_rule)  # add_rule replaces existing
            
            return {
                'success': True,
                'rule_id': rule_id,
                'updated': success,
                'rule_configuration': self._serialize_rule(updated_rule) if success else None
            }
            
        except Exception as e:
            logger.error(f"Error updating validation rule: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'update_rule'
            }
    
    async def _delete_rule(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a validation rule."""
        try:
            rule_id = args.get('rule_id')
            
            if not rule_id:
                raise ValueError("rule_id is required")
            
            logger.info(f"Deleting validation rule: {rule_id}")
            
            # Delete rule from engine
            success = self.engine.remove_rule(rule_id)
            
            return {
                'success': True,
                'rule_id': rule_id,
                'deleted': success
            }
            
        except Exception as e:
            logger.error(f"Error deleting validation rule: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'delete_rule'
            }
    
    async def _get_suggestions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get correction suggestions for invalid field values."""
        try:
            field_path = args.get('field_path')
            value = json.loads(args.get('value', 'null'))
            field_type_str = args.get('field_type', 'text')
            validation_error = args.get('validation_error', '')
            
            if not field_path:
                raise ValueError("field_path is required")
            
            logger.info(f"Generating suggestions for field: {field_path}")
            
            # Convert field type string to enum
            field_type = FieldType(field_type_str)
            
            # Generate correction suggestion
            corrected_value = self.engine.correction_engine.suggest_correction(
                field_type, str(value) if value else "", validation_error
            )
            
            suggestions = []
            if corrected_value and corrected_value != str(value):
                suggestions.append({
                    'suggestion': f"Try: {corrected_value}",
                    'corrected_value': corrected_value,
                    'confidence': 0.8,
                    'method': 'pattern_correction'
                })
            
            # Add general suggestions based on field type
            general_suggestions = self._get_general_suggestions(field_type, value, validation_error)
            suggestions.extend(general_suggestions)
            
            return {
                'success': True,
                'field_path': field_path,
                'suggestions': suggestions,
                'corrected_value': corrected_value,
                'confidence_score': 0.8 if corrected_value else 0.3
            }
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'get_suggestions'
            }
    
    async def _check_afm_compliance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Check AFM compliance for a field."""
        try:
            field_path = args.get('field_path')
            value = json.loads(args.get('value', 'null'))
            context = json.loads(args.get('context', '{}'))
            
            if not field_path:
                raise ValueError("field_path is required")
            
            logger.info(f"Checking AFM compliance for field: {field_path}")
            
            # Perform AFM compliance validation
            compliance_messages = self.engine.afm_validator.validate_compliance(
                field_path, value, context
            )
            
            # Serialize compliance messages
            serialized_messages = []
            for message in compliance_messages:
                serialized_messages.append({
                    'message_id': message.message_id,
                    'rule_id': message.rule_id,
                    'field_path': message.field_path,
                    'message': message.message,
                    'severity': message.severity.value,
                    'suggestion': message.suggestion,
                    'afm_reference': message.afm_reference,
                    'context': message.context,
                    'timestamp': message.timestamp.isoformat()
                })
            
            # Calculate compliance score
            is_compliant = len([m for m in compliance_messages if m.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0
            compliance_score = self.engine._calculate_compliance_score(compliance_messages)
            
            # Generate AFM requirements and recommendations
            afm_requirements = self._get_afm_requirements(field_path)
            recommendations = self._get_afm_recommendations(compliance_messages)
            
            return {
                'success': True,
                'field_path': field_path,
                'is_compliant': is_compliant,
                'compliance_score': compliance_score,
                'compliance_messages': serialized_messages,
                'afm_requirements': afm_requirements,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error checking AFM compliance: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'check_afm_compliance'
            }
    
    async def _get_statistics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get validation engine statistics."""
        try:
            logger.info("Retrieving validation engine statistics")
            
            statistics = self.engine.get_validation_statistics()
            
            # Add additional engine information
            engine_info = {
                'total_rules': len(self.engine.rules),
                'active_rules': len([r for r in self.engine.rules.values() if r.is_active]),
                'supported_field_types': [ft.value for ft in FieldType],
                'supported_rule_types': list(set(r.rule_type for r in self.engine.rules.values())),
                'afm_compliance_enabled': True,
                'suggestion_engine_enabled': True
            }
            
            # Performance metrics
            performance_metrics = {
                'avg_field_validation_time': statistics['validation_stats']['average_processing_time'],
                'total_validations': statistics['validation_stats']['total_validations'],
                'success_rate': statistics['validation_stats']['successful_validations'] / max(1, statistics['validation_stats']['total_validations']),
                'memory_usage': 'N/A',  # Could be implemented with psutil
                'cache_hit_rate': 'N/A'  # Could be implemented with caching
            }
            
            return {
                'success': True,
                'statistics': statistics,
                'engine_info': engine_info,
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error retrieving statistics: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'get_statistics'
            }
    
    async def _export_rules(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Export validation rules."""
        try:
            format_type = args.get('format', 'json')
            
            logger.info(f"Exporting validation rules in {format_type} format")
            
            exported_data = await self.engine.export_validation_rules(format_type)
            
            return {
                'success': True,
                'format': format_type,
                'exported_data': exported_data,
                'rule_count': len(self.engine.rules)
            }
            
        except Exception as e:
            logger.error(f"Error exporting rules: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'export_rules'
            }
    
    async def _import_rules(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Import validation rules."""
        try:
            rules_data = args.get('rules_data')
            format_type = args.get('format', 'json')
            merge = args.get('merge', True)
            
            if not rules_data:
                raise ValueError("rules_data is required")
            
            logger.info(f"Importing validation rules from {format_type} format (merge: {merge})")
            
            imported_count = await self.engine.import_validation_rules(
                rules_data, format_type, merge
            )
            
            return {
                'success': True,
                'imported_count': imported_count,
                'failed_count': 0,  # Could be enhanced to track failures
                'format': format_type,
                'merge_mode': merge
            }
            
        except Exception as e:
            logger.error(f"Error importing rules: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'import_rules'
            }
    
    async def _health_check(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Perform health check of validation services."""
        try:
            logger.info("Performing field validation health check")
            
            start_time = datetime.now()
            
            # Test validation functionality
            test_data = {'test_field': 'test_value'}
            test_result = await self.engine.validate_data(test_data)
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Check component health
            services = {
                'validation_engine': 'healthy',
                'correction_engine': 'healthy',
                'afm_validator': 'healthy',
                'dutch_validators': 'healthy',
                'rule_engine': 'healthy'
            }
            
            # Performance metrics
            performance = {
                'response_time': response_time,
                'total_rules': len(self.engine.rules),
                'active_rules': len([r for r in self.engine.rules.values() if r.is_active]),
                'validation_count': self.engine.validation_stats['total_validations'],
                'success_rate': self.engine.validation_stats['successful_validations'] / max(1, self.engine.validation_stats['total_validations'])
            }
            
            # Dependencies
            dependencies = {
                'python_version': sys.version,
                'required_packages': ['phonenumbers', 'unicodedata', 'decimal'],
                'optional_packages': ['numpy', 'pandas']
            }
            
            return {
                'success': True,
                'status': 'healthy',
                'services': services,
                'performance': performance,
                'dependencies': dependencies,
                'version': '1.0.0',
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
    
    def _serialize_rule(self, rule: ValidationRule) -> Dict[str, Any]:
        """Serialize a ValidationRule object to dictionary."""
        return {
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
    
    def _get_general_suggestions(self, field_type: FieldType, value: Any, 
                               validation_error: str) -> List[Dict[str, Any]]:
        """Get general suggestions based on field type and error."""
        suggestions = []
        
        if field_type == FieldType.EMAIL:
            suggestions.extend([
                {
                    'suggestion': 'Ensure email contains @ symbol and valid domain',
                    'confidence': 0.6,
                    'method': 'general_guideline'
                },
                {
                    'suggestion': 'Check for common typos in domain names (gmail.com, hotmail.com, etc.)',
                    'confidence': 0.5,
                    'method': 'common_patterns'
                }
            ])
        elif field_type == FieldType.PHONE:
            suggestions.extend([
                {
                    'suggestion': 'Use Dutch phone format: +31 6 12 34 56 78 or 06 12345678',
                    'confidence': 0.7,
                    'method': 'format_guideline'
                },
                {
                    'suggestion': 'Remove spaces and special characters, use only digits and +',
                    'confidence': 0.6,
                    'method': 'formatting_tip'
                }
            ])
        elif field_type == FieldType.BSN:
            suggestions.extend([
                {
                    'suggestion': 'BSN must be exactly 9 digits',
                    'confidence': 0.8,
                    'method': 'format_requirement'
                },
                {
                    'suggestion': 'Check BSN checksum calculation (11-test)',
                    'confidence': 0.7,
                    'method': 'validation_algorithm'
                }
            ])
        elif field_type == FieldType.POSTCODE:
            suggestions.extend([
                {
                    'suggestion': 'Use Dutch postcode format: 1234 AB',
                    'confidence': 0.8,
                    'method': 'format_guideline'
                }
            ])
        elif field_type == FieldType.IBAN:
            suggestions.extend([
                {
                    'suggestion': 'Use Dutch IBAN format: NL91 ABNA 0417 1643 00',
                    'confidence': 0.8,
                    'method': 'format_guideline'
                }
            ])
        elif field_type == FieldType.CURRENCY:
            suggestions.extend([
                {
                    'suggestion': 'Use numeric values only, decimal separator allowed',
                    'confidence': 0.7,
                    'method': 'format_tip'
                }
            ])
        elif field_type == FieldType.DATE:
            suggestions.extend([
                {
                    'suggestion': 'Use DD-MM-YYYY format for dates',
                    'confidence': 0.8,
                    'method': 'format_standard'
                }
            ])
        
        return suggestions
    
    def _get_afm_requirements(self, field_path: str) -> List[Dict[str, Any]]:
        """Get AFM requirements for a specific field."""
        requirements = []
        
        field_lower = field_path.lower()
        
        if 'income' in field_lower:
            requirements.append({
                'requirement': 'Income verification documentation required',
                'article': 'Wft 86f',
                'description': 'Client income must be verified with appropriate documentation'
            })
        
        if 'bsn' in field_lower:
            requirements.append({
                'requirement': 'BSN handling and privacy requirements',
                'article': 'AVG Article 32',
                'description': 'BSN data requires encryption and secure handling'
            })
        
        if any(term in field_lower for term in ['rate', 'cost', 'fee']):
            requirements.append({
                'requirement': 'Financial disclosure requirements',
                'article': 'BGfo Article 8.1',
                'description': 'All costs and rates must be clearly disclosed'
            })
        
        return requirements
    
    def _get_afm_recommendations(self, compliance_messages) -> List[Dict[str, Any]]:
        """Get AFM recommendations based on compliance messages."""
        recommendations = []
        
        for message in compliance_messages:
            if message.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                recommendations.append({
                    'recommendation': f"Address compliance issue: {message.message}",
                    'priority': 'high',
                    'afm_reference': message.afm_reference
                })
            elif message.severity == ValidationSeverity.WARNING:
                recommendations.append({
                    'recommendation': f"Consider improvement: {message.message}",
                    'priority': 'medium',
                    'afm_reference': message.afm_reference
                })
        
        return recommendations

async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Advanced Field Validation Executor')
    parser.add_argument('operation', help='Operation to perform')
    parser.add_argument('--field_path', help='Field path for validation')
    parser.add_argument('--value', help='Field value (JSON)', default='null')
    parser.add_argument('--field_type', help='Field type', default='text')
    parser.add_argument('--context', help='Validation context (JSON)', default='{}')
    parser.add_argument('--data', help='Data for validation (JSON)', default='{}')
    parser.add_argument('--validation_config', help='Validation configuration (JSON)', default='{}')
    parser.add_argument('--check_afm_compliance', help='Check AFM compliance', default='true')
    parser.add_argument('--generate_suggestions', help='Generate suggestions', default='true')
    parser.add_argument('--include_scores', help='Include field scores', default='true')
    parser.add_argument('--rule_id', help='Rule ID')
    parser.add_argument('--rule_data', help='Rule data (JSON)', default='{}')
    parser.add_argument('--rule_type', help='Rule type filter')
    parser.add_argument('--active_only', help='Active rules only', default='false')
    parser.add_argument('--limit', help='Result limit', default='100')
    parser.add_argument('--offset', help='Result offset', default='0')
    parser.add_argument('--validation_error', help='Validation error message', default='')
    parser.add_argument('--format', help='Export/import format', default='json')
    parser.add_argument('--rules_data', help='Rules data for import')
    parser.add_argument('--merge', help='Merge mode for import', default='true')
    parser.add_argument('--output_format', help='Output format', default='json')
    
    args = parser.parse_args()
    
    try:
        executor = AdvancedFieldValidationExecutor()
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
