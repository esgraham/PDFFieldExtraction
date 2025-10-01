"""
Enhanced Validation and Rules Engine

Comprehensive validation system with regex patterns, Luhn algorithm, 
date validation, cross-field consistency checks, and business rules.
"""

import re
import logging
import math
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal, InvalidOperation

# Optional imports with fallbacks
try:
    import phonenumbers
    from phonenumbers import NumberParseException
    HAS_PHONENUMBERS = True
except ImportError:
    HAS_PHONENUMBERS = False
    NumberParseException = Exception

try:
    import validators
    HAS_VALIDATORS = True
except ImportError:
    HAS_VALIDATORS = False

try:
    from dateutil import parser as date_parser
    from dateutil.parser import ParserError
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    ParserError = ValueError

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation severity levels."""
    ERROR = "error"          # Blocks processing, requires HITL
    WARNING = "warning"      # Logs issue but continues
    INFO = "info"           # Informational only

class ValidationRule(Enum):
    """Built-in validation rules."""
    REQUIRED = "required"
    REGEX = "regex"
    LUHN = "luhn"
    DATE_FORMAT = "date_format"
    DATE_RANGE = "date_range"
    NUMERIC_RANGE = "numeric_range"
    CURRENCY_FORMAT = "currency_format"
    EMAIL_FORMAT = "email_format"
    PHONE_FORMAT = "phone_format"
    LENGTH = "length"
    CROSS_FIELD = "cross_field"
    CHECKSUM = "checksum"
    BUSINESS_LOGIC = "business_logic"

@dataclass
class ValidationResult:
    """Field validation result."""
    field_name: str
    is_valid: bool
    severity: ValidationSeverity
    message: str
    rule_name: str
    suggested_value: Any = None

@dataclass
class ValidationConfig:
    """Configuration for field validation."""
    required: bool = False
    patterns: List[str] = None
    luhn_check: bool = False
    date_formats: List[str] = None
    date_range: Dict[str, Any] = None
    numeric_range: Dict[str, float] = None
    length_range: Dict[str, int] = None
    business_rules: List[str] = None
    cross_field_rules: List[Dict] = None

class LuhnValidator:
    """Luhn algorithm implementation for credit card, account number validation."""
    
    @staticmethod
    def validate(number: str) -> bool:
        """Validate using Luhn algorithm."""
        if not number or not str(number).isdigit():
            return False
            
        # Convert to list of integers
        digits = [int(d) for d in str(number)]
        
        # Apply Luhn algorithm
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        
        return sum(digits) % 10 == 0
    
    @staticmethod
    def generate_check_digit(partial_number: str) -> str:
        """Generate Luhn check digit for partial number."""
        if not str(partial_number).isdigit():
            raise ValueError("Partial number must contain only digits")
            
        # Calculate check digit
        digits = [int(d) for d in str(partial_number) + '0']
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        
        total = sum(digits)
        check_digit = (10 - (total % 10)) % 10
        return str(check_digit)

class AdvancedValidator:
    """Advanced validation utilities with regex, date, and cross-field checks."""
    
    # Common regex patterns
    PATTERNS = {
        'ssn': r'^\\d{3}-?\\d{2}-?\\d{4}$',
        'ein': r'^\\d{2}-?\\d{7}$',
        'credit_card': r'^\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}$',
        'invoice_number': r'^[A-Za-z0-9\\-_]{3,20}$',
        'po_number': r'^[A-Za-z0-9\\-_]{3,25}$',
        'amount': r'^\\$?[0-9]{1,3}(?:,?[0-9]{3})*(?:\\.[0-9]{2})?$',
        'percentage': r'^\\d{1,3}(?:\\.\\d{1,2})?%?$',
        'zipcode': r'^\\d{5}(?:-\\d{4})?$',
        'account_number': r'^[0-9]{8,17}$',
        'routing_number': r'^[0-9]{9}$',
        'aba_number': r'^[0-9]{9}$'
    }
    
    @classmethod
    def validate_regex(cls, value: str, pattern: str, field_name: str = "") -> ValidationResult:
        """Validate value against regex pattern."""
        if not value:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Empty value for regex validation",
                rule_name="regex"
            )
        
        try:
            is_valid = bool(re.match(pattern, str(value).strip()))
            return ValidationResult(
                field_name=field_name,
                is_valid=is_valid,
                severity=ValidationSeverity.ERROR if not is_valid else ValidationSeverity.INFO,
                message=f"Regex validation {'passed' if is_valid else 'failed'}: {pattern}",
                rule_name="regex"
            )
        except re.error as e:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid regex pattern: {str(e)}",
                rule_name="regex"
            )
    
    @classmethod
    def validate_luhn(cls, value: str, field_name: str = "") -> ValidationResult:
        """Validate using Luhn algorithm."""
        # Clean the value (remove spaces, dashes)
        clean_value = re.sub(r'[\\s-]', '', str(value))
        
        is_valid = LuhnValidator.validate(clean_value)
        
        return ValidationResult(
            field_name=field_name,
            is_valid=is_valid,
            severity=ValidationSeverity.ERROR if not is_valid else ValidationSeverity.INFO,
            message=f"Luhn validation {'passed' if is_valid else 'failed'} for: {value}",
            rule_name="luhn"
        )
    
    @classmethod
    def validate_date_format(cls, value: str, expected_formats: List[str], field_name: str = "") -> ValidationResult:
        """Validate date format and parse."""
        if not value:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Empty date value",
                rule_name="date_format"
            )
        
        # Try dateutil parser first (most flexible) if available
        if HAS_DATEUTIL:
            try:
                parsed_date = date_parser.parse(str(value))
                return ValidationResult(
                    field_name=field_name,
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Date parsed successfully: {parsed_date.strftime('%Y-%m-%d')}",
                    rule_name="date_format",
                    suggested_value=parsed_date
                )
            except (ParserError, ValueError):
                pass
        
        # Try specific formats
        for fmt in expected_formats:
            try:
                parsed_date = datetime.strptime(str(value), fmt)
                return ValidationResult(
                    field_name=field_name,
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Date parsed with format {fmt}: {parsed_date.strftime('%Y-%m-%d')}",
                    rule_name="date_format",
                    suggested_value=parsed_date
                )
            except ValueError:
                continue
        
        return ValidationResult(
            field_name=field_name,
            is_valid=False,
            severity=ValidationSeverity.ERROR,
            message=f"Date format validation failed. Tried formats: {expected_formats}",
            rule_name="date_format"
        )
    
    @classmethod
    def validate_date_range(cls, value: Any, min_date: Optional[datetime], max_date: Optional[datetime], field_name: str = "") -> ValidationResult:
        """Validate date is within specified range."""
        if isinstance(value, str):
            if HAS_DATEUTIL:
                try:
                    date_value = date_parser.parse(value)
                except (ParserError, ValueError):
                    return ValidationResult(
                        field_name=field_name,
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Cannot parse date for range validation: {value}",
                        rule_name="date_range"
                    )
            else:
                try:
                    date_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    return ValidationResult(
                        field_name=field_name,
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Cannot parse date for range validation: {value}",
                        rule_name="date_range"
                    )
        elif isinstance(value, datetime):
            date_value = value
        else:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid date type for range validation: {type(value)}",
                rule_name="date_range"
            )
        
        errors = []
        if min_date and date_value < min_date:
            errors.append(f"Date {date_value.strftime('%Y-%m-%d')} is before minimum {min_date.strftime('%Y-%m-%d')}")
        if max_date and date_value > max_date:
            errors.append(f"Date {date_value.strftime('%Y-%m-%d')} is after maximum {max_date.strftime('%Y-%m-%d')}")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            field_name=field_name,
            is_valid=is_valid,
            severity=ValidationSeverity.ERROR if not is_valid else ValidationSeverity.INFO,
            message="Date range validation passed" if is_valid else "; ".join(errors),
            rule_name="date_range"
        )
    
    @classmethod
    def validate_currency(cls, value: str, field_name: str = "") -> ValidationResult:
        """Validate and normalize currency value."""
        if not value:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Empty currency value",
                rule_name="currency_format"
            )
        
        # Remove currency symbols and whitespace
        clean_value = re.sub(r'[\\$£€¥\\s,]', '', str(value))
        
        try:
            decimal_value = Decimal(clean_value)
            normalized = f"${decimal_value:.2f}"
            
            return ValidationResult(
                field_name=field_name,
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message=f"Currency validated and normalized: {normalized}",
                rule_name="currency_format",
                suggested_value=normalized
            )
        except (InvalidOperation, ValueError):
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid currency format: {value}",
                rule_name="currency_format"
            )
    
    @classmethod
    def validate_email(cls, value: str, field_name: str = "") -> ValidationResult:
        """Validate email format."""
        if not value:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Empty email value",
                rule_name="email_format"
            )
        
        # Basic email regex if validators not available
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
        
        try:
            if HAS_VALIDATORS:
                is_valid = validators.email(str(value))
            else:
                is_valid = bool(re.match(email_pattern, str(value)))
                
            return ValidationResult(
                field_name=field_name,
                is_valid=is_valid,
                severity=ValidationSeverity.ERROR if not is_valid else ValidationSeverity.INFO,
                message=f"Email validation {'passed' if is_valid else 'failed'}: {value}",
                rule_name="email_format"
            )
        except Exception as e:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Email validation error: {str(e)}",
                rule_name="email_format"
            )
    
    @classmethod
    def validate_phone(cls, value: str, country_code: str = "US", field_name: str = "") -> ValidationResult:
        """Validate and format phone number."""
        if not value:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Empty phone number",
                rule_name="phone_format"
            )
        
        if not HAS_PHONENUMBERS:
            # Basic phone validation without phonenumbers library
            phone_pattern = r'^[\\+]?[1-9]?[0-9]{7,15}$'
            clean_phone = re.sub(r'[\\s\\-\\(\\)\\.]', '', str(value))
            is_valid = bool(re.match(phone_pattern, clean_phone))
            
            return ValidationResult(
                field_name=field_name,
                is_valid=is_valid,
                severity=ValidationSeverity.ERROR if not is_valid else ValidationSeverity.INFO,
                message=f"Phone validation {'passed' if is_valid else 'failed'} (basic): {value}",
                rule_name="phone_format"
            )
        
        try:
            phone_number = phonenumbers.parse(str(value), country_code)
            is_valid = phonenumbers.is_valid_number(phone_number)
            
            if is_valid:
                formatted = phonenumbers.format_number(phone_number, phonenumbers.PhoneNumberFormat.NATIONAL)
                return ValidationResult(
                    field_name=field_name,
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Phone validated and formatted: {formatted}",
                    rule_name="phone_format",
                    suggested_value=formatted
                )
            else:
                return ValidationResult(
                    field_name=field_name,
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid phone number: {value}",
                    rule_name="phone_format"
                )
        except NumberParseException as e:
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Phone parsing error: {str(e)}",
                rule_name="phone_format"
            )

class CrossFieldValidator:
    """Cross-field validation for business logic consistency."""
    
    @staticmethod
    def validate_date_sequence(date1: Any, date2: Any, field1_name: str, field2_name: str, 
                              rule: str = "date1_before_date2") -> ValidationResult:
        """Validate date sequence (e.g., invoice_date before due_date)."""
        try:
            # Parse dates if they're strings
            if isinstance(date1, str):
                if HAS_DATEUTIL:
                    date1 = date_parser.parse(date1)
                else:
                    date1 = datetime.fromisoformat(date1.replace('Z', '+00:00'))
            if isinstance(date2, str):
                if HAS_DATEUTIL:
                    date2 = date_parser.parse(date2)
                else:
                    date2 = datetime.fromisoformat(date2.replace('Z', '+00:00'))
            
            if rule == "date1_before_date2":
                is_valid = date1 <= date2
                message = f"{field1_name} ({date1.strftime('%Y-%m-%d')}) should be before or equal to {field2_name} ({date2.strftime('%Y-%m-%d')})"
            elif rule == "date1_after_date2":
                is_valid = date1 >= date2
                message = f"{field1_name} ({date1.strftime('%Y-%m-%d')}) should be after or equal to {field2_name} ({date2.strftime('%Y-%m-%d')})"
            else:
                is_valid = False
                message = f"Unknown date sequence rule: {rule}"
            
            return ValidationResult(
                field_name=f"{field1_name}+{field2_name}",
                is_valid=is_valid,
                severity=ValidationSeverity.ERROR if not is_valid else ValidationSeverity.INFO,
                message=message,
                rule_name="cross_field_date"
            )
            
        except (ValueError, AttributeError) as e:
            return ValidationResult(
                field_name=f"{field1_name}+{field2_name}",
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Date sequence validation error: {str(e)}",
                rule_name="cross_field_date"
            )
    
    @staticmethod
    def validate_amount_consistency(amounts: Dict[str, Any], formula: str) -> ValidationResult:
        """Validate amount consistency (e.g., total = subtotal + tax)."""
        try:
            # Convert amounts to Decimal for precise calculation
            decimal_amounts = {}
            for key, value in amounts.items():
                if isinstance(value, str):
                    clean_value = re.sub(r'[\\$,\\s]', '', value)
                    decimal_amounts[key] = Decimal(clean_value)
                else:
                    decimal_amounts[key] = Decimal(str(value))
            
            # Common formula patterns
            if formula == "total = subtotal + tax":
                expected_total = decimal_amounts.get('subtotal', Decimal('0')) + decimal_amounts.get('tax', Decimal('0'))
                actual_total = decimal_amounts.get('total', Decimal('0'))
                is_valid = abs(expected_total - actual_total) < Decimal('0.01')  # Allow 1 cent tolerance
                message = f"Total validation: {actual_total} {'==' if is_valid else '!='} {expected_total} (subtotal + tax)"
            
            elif formula == "total = subtotal + tax - discount":
                expected_total = (decimal_amounts.get('subtotal', Decimal('0')) + 
                                decimal_amounts.get('tax', Decimal('0')) - 
                                decimal_amounts.get('discount', Decimal('0')))
                actual_total = decimal_amounts.get('total', Decimal('0'))
                is_valid = abs(expected_total - actual_total) < Decimal('0.01')
                message = f"Total validation: {actual_total} {'==' if is_valid else '!='} {expected_total} (subtotal + tax - discount)"
            
            else:
                # Custom formula evaluation (basic)
                is_valid = False
                message = f"Unknown formula: {formula}"
            
            return ValidationResult(
                field_name="amount_consistency",
                is_valid=is_valid,
                severity=ValidationSeverity.ERROR if not is_valid else ValidationSeverity.INFO,
                message=message,
                rule_name="cross_field_amount"
            )
            
        except (InvalidOperation, ValueError, KeyError) as e:
            return ValidationResult(
                field_name="amount_consistency",
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Amount consistency validation error: {str(e)}",
                rule_name="cross_field_amount"
            )
    
    @staticmethod
    def validate_business_logic(extracted_fields: Dict[str, Any], rules: List[Dict]) -> List[ValidationResult]:
        """Validate custom business logic rules."""
        results = []
        
        for rule in rules:
            rule_name = rule.get('name', 'unnamed_rule')
            rule_type = rule.get('type')
            
            try:
                if rule_type == 'date_sequence':
                    field1 = rule['field1']
                    field2 = rule['field2']
                    date_rule = rule.get('rule', 'date1_before_date2')
                    
                    if field1 in extracted_fields and field2 in extracted_fields:
                        result = CrossFieldValidator.validate_date_sequence(
                            extracted_fields[field1], extracted_fields[field2],
                            field1, field2, date_rule
                        )
                        result.rule_name = rule_name
                        results.append(result)
                
                elif rule_type == 'amount_consistency':
                    formula = rule['formula']
                    required_fields = rule.get('fields', [])
                    
                    # Check if all required fields are present
                    amounts = {}
                    missing_fields = []
                    for field in required_fields:
                        if field in extracted_fields:
                            amounts[field] = extracted_fields[field]
                        else:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        results.append(ValidationResult(
                            field_name=rule_name,
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            message=f"Missing fields for amount validation: {missing_fields}",
                            rule_name=rule_name
                        ))
                    else:
                        result = CrossFieldValidator.validate_amount_consistency(amounts, formula)
                        result.rule_name = rule_name
                        results.append(result)
                
                else:
                    results.append(ValidationResult(
                        field_name=rule_name,
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Unknown business rule type: {rule_type}",
                        rule_name=rule_name
                    ))
                    
            except Exception as e:
                results.append(ValidationResult(
                    field_name=rule_name,
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Business rule validation error: {str(e)}",
                    rule_name=rule_name
                ))
        
        return results

class ComprehensiveValidator:
    """Main validation engine that combines all validation types."""
    
    def __init__(self):
        self.advanced_validator = AdvancedValidator()
        self.cross_field_validator = CrossFieldValidator()
        
        # Built-in business rules
        self.default_business_rules = [
            {
                'name': 'invoice_date_before_due_date',
                'type': 'date_sequence',
                'field1': 'invoice_date',
                'field2': 'due_date',
                'rule': 'date1_before_date2'
            },
            {
                'name': 'invoice_total_consistency',
                'type': 'amount_consistency',
                'formula': 'total = subtotal + tax',
                'fields': ['subtotal', 'tax', 'total']
            },
            {
                'name': 'po_date_before_delivery',
                'type': 'date_sequence',
                'field1': 'po_date',
                'field2': 'delivery_date',
                'rule': 'date1_before_date2'
            }
        ]
    
    def validate_field(self, field_name: str, value: Any, config: ValidationConfig) -> List[ValidationResult]:
        """Validate a single field with comprehensive rules."""
        results = []
        
        # Extract actual value if it's in dictionary format
        actual_value = value
        if isinstance(value, dict) and 'value' in value:
            actual_value = value['value']
        
        # Required check
        if config.required and (not actual_value or str(actual_value).strip() == ""):
            results.append(ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Required field '{field_name}' is empty",
                rule_name="required"
            ))
            return results  # Don't continue if required field is empty
        
        if not actual_value:
            return results  # Skip other validations if value is empty for optional fields
        
        # Regex patterns
        if config.patterns:
            for pattern in config.patterns:
                result = self.advanced_validator.validate_regex(actual_value, pattern, field_name)
                results.append(result)
        
        # Luhn check
        if config.luhn_check:
            result = self.advanced_validator.validate_luhn(actual_value, field_name)
            results.append(result)
        
        # Date format validation
        if config.date_formats:
            result = self.advanced_validator.validate_date_format(actual_value, config.date_formats, field_name)
            results.append(result)
        
        # Date range validation
        if config.date_range:
            min_date = config.date_range.get('min_date')
            max_date = config.date_range.get('max_date')
            result = self.advanced_validator.validate_date_range(actual_value, min_date, max_date, field_name)
            results.append(result)
        
        # Length validation
        if config.length_range:
            min_len = config.length_range.get('min', 0)
            max_len = config.length_range.get('max', float('inf'))
            value_len = len(str(actual_value))
            
            if value_len < min_len or value_len > max_len:
                results.append(ValidationResult(
                    field_name=field_name,
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Length {value_len} not in range [{min_len}, {max_len}]",
                    rule_name="length"
                ))
        
        return results
    
    def validate_document(self, extracted_fields: Dict[str, Any], 
                         field_configs: Union[Dict[str, ValidationConfig], str],
                         custom_business_rules: List[Dict] = None) -> List[ValidationResult]:
        """Validate entire document with field and cross-field rules."""
        all_results = []
        
        # Handle case where doc_type string is passed instead of field_configs
        if isinstance(field_configs, str):
            doc_type = field_configs
            field_configs = self._generate_field_configs_for_document_type(doc_type, extracted_fields)
        
        # Validate individual fields
        for field_name, config in field_configs.items():
            field_value = extracted_fields.get(field_name)
            field_results = self.validate_field(field_name, field_value, config)
            all_results.extend(field_results)
        
        # Cross-field validation with default rules
        business_rules = self.default_business_rules.copy()
        if custom_business_rules:
            business_rules.extend(custom_business_rules)
        
        cross_field_results = self.cross_field_validator.validate_business_logic(
            extracted_fields, business_rules
        )
        all_results.extend(cross_field_results)
        
        return all_results
    
    def _generate_field_configs_for_document_type(self, doc_type: str, extracted_fields: Dict[str, Any]) -> Dict[str, ValidationConfig]:
        """Generate field validation configs based on document type and extracted fields."""
        configs = {}
        
        # Create basic configs for all extracted fields
        for field_name, field_value in extracted_fields.items():
            if field_name in ['document_type', 'template_type', 'overall_confidence']:
                continue  # Skip metadata fields
                
            # Determine field type and create appropriate config
            if isinstance(field_value, dict) and 'value' in field_value:
                actual_value = field_value['value']
            else:
                actual_value = field_value
            
            # Create basic validation config based on field content
            field_type = self._infer_field_type(field_name, actual_value)
            
            config = ValidationConfig(
                required=field_name in self._get_required_fields_for_doc_type(doc_type),
                patterns=self._get_patterns_for_field_type(field_type),
                length_range={'min': 1 if actual_value else 0, 'max': 1000} if field_type == 'text' else None,
                date_formats=['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y'] if field_type == 'date' else None,
                numeric_range={'min': 0, 'max': 999999999} if field_type == 'currency' else None
            )
            configs[field_name] = config
        
        return configs
    
    def _get_required_fields_for_doc_type(self, doc_type: str) -> List[str]:
        """Get required fields for a document type."""
        required_fields_map = {
            'invoice': ['invoice_number', 'amount', 'date'],
            'receipt': ['total', 'date'],
            'purchase_order': ['po_number', 'amount'],
            'custom': []  # No required fields for custom documents
        }
        return required_fields_map.get(doc_type.lower(), [])
    
    def _infer_field_type(self, field_name: str, value: Any) -> str:
        """Infer field type from field name and value."""
        field_name_lower = field_name.lower()
        
        if 'date' in field_name_lower:
            return 'date'
        elif any(word in field_name_lower for word in ['amount', 'total', 'price', 'cost', 'sum']):
            return 'currency'
        elif any(word in field_name_lower for word in ['number', 'id', 'code']):
            return 'alphanumeric'
        elif 'email' in field_name_lower:
            return 'email'
        elif 'phone' in field_name_lower:
            return 'phone'
        else:
            return 'text'
    
    def _get_patterns_for_field_type(self, field_type: str) -> List[str]:
        """Get validation patterns for a field type."""
        pattern_map = {
            'email': [r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'],
            'phone': [r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'],
            'currency': [r'^\$?[\d,]+\.?\d{0,2}$'],
            'alphanumeric': [r'^[A-Za-z0-9-]+$'],
            'text': [],
            'date': []  # Date validation handled by date_formats
        }
        return pattern_map.get(field_type, [])
    
    def should_route_to_hitl(self, validation_results: List[ValidationResult], 
                           confidence_scores: Dict[str, float],
                           confidence_threshold: float = 0.7) -> tuple[bool, List[str]]:
        """Determine if document should be routed to HITL based on validation and confidence."""
        reasons = []
        
        # Check validation failures
        critical_failures = [r for r in validation_results if not r.is_valid and r.severity == ValidationSeverity.ERROR]
        if critical_failures:
            reasons.extend([f"Validation failed: {r.message}" for r in critical_failures])
        
        # Check confidence scores
        low_confidence_fields = [field for field, score in confidence_scores.items() if score < confidence_threshold]
        if low_confidence_fields:
            reasons.append(f"Low confidence fields: {', '.join(low_confidence_fields)}")
        
        should_route = len(reasons) > 0
        return should_route, reasons