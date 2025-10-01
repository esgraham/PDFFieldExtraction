"""
Field Extraction Engine

This module provides comprehensive field extraction capabilities with:
- Template-based extraction for different document types
- Business rules validation and normalization
- Confidence scoring and HITL routing
- Queue management with poison queue pattern
- Schema mapping and synonym resolution
"""

import logging
import re
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
import uuid
from pathlib import Path
import hashlib

# Business logic imports
from dateutil import parser as date_parser
from dateutil.parser import ParserError
import phonenumbers
from phonenumbers import NumberParseException
from email_validator import validate_email, EmailNotValidError

# Optional imports with fallbacks
try:
    import validators
    HAS_VALIDATORS = True
except ImportError:
    HAS_VALIDATORS = False
    validators = None

# Validation imports
from .validation_engine import ValidationResult, LuhnValidator

logger = logging.getLogger(__name__)

class FieldType(Enum):
    """Supported field types for extraction."""
    TEXT = "text"
    NUMBER = "number"
    CURRENCY = "currency"
    DATE = "date"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    PERCENTAGE = "percentage"
    BOOLEAN = "boolean"
    TABLE = "table"
    SIGNATURE = "signature"

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

# Import enhanced validation system
try:
    from .validation_engine import (
        ComprehensiveValidator, ValidationConfig, ValidationResult as EnhancedValidationResult,
        ValidationSeverity, ValidationRule
    )
    HAS_VALIDATION_ENGINE = True
except ImportError:
    HAS_VALIDATION_ENGINE = False
    ComprehensiveValidator = None

def validate_luhn(number: str) -> bool:
    """Validate using Luhn algorithm."""
    if not number or not number.isdigit():
        return False
        
    # Convert to list of integers
    digits = [int(d) for d in number]
    
    # Apply Luhn algorithm
    for i in range(len(digits) - 2, -1, -2):
        digits[i] *= 2
        if digits[i] > 9:
            digits[i] -= 9
    
    return sum(digits) % 10 == 0

    @staticmethod
    def generate_check_digit(partial_number: str) -> str:
        """Generate Luhn check digit for partial number."""
        if not partial_number.isdigit():
            raise ValueError("Partial number must contain only digits")
            
        # Calculate check digit
        digits = [int(d) for d in partial_number + '0']
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
        'ssn': r'^\d{3}-?\d{2}-?\d{4}$',
        'ein': r'^\d{2}-?\d{7}$',
        'credit_card': r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$',
        'invoice_number': r'^[A-Za-z0-9\-_]{3,20}$',
        'po_number': r'^[A-Za-z0-9\-_]{3,25}$',
        'amount': r'^\$?[0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]{2})?$',
        'percentage': r'^\d{1,3}(?:\.\d{1,2})?%?$',
        'zipcode': r'^\d{5}(?:-\d{4})?$',
        'account_number': r'^[0-9]{8,17}$'
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
        clean_value = re.sub(r'[\s-]', '', str(value))
        
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
        
        # Try dateutil parser first (most flexible)
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
        clean_value = re.sub(r'[\$£€¥\s,]', '', str(value))
        
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
        
        try:
            if HAS_VALIDATORS and validators:
                is_valid = validators.email(str(value))
            else:
                # Fallback to basic regex validation
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
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
                date1 = date_parser.parse(date1)
            if isinstance(date2, str):
                date2 = date_parser.parse(date2)
            
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
            
        except (ParserError, ValueError, AttributeError) as e:
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
                    clean_value = re.sub(r'[\$,\s]', '', value)
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

class DocumentTemplate(Enum):
    """Document template types."""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    PURCHASE_ORDER = "purchase_order"
    TAX_FORM = "tax_form"
    CONTRACT = "contract"
    FORM_APPLICATION = "form_application"
    BANK_STATEMENT = "bank_statement"
    INSURANCE_CLAIM = "insurance_claim"
    CUSTOM = "custom"

@dataclass
class FieldSchema:
    """Schema definition for a field."""
    name: str
    field_type: FieldType
    required: bool = False
    synonyms: List[str] = None
    validation_rules: List[str] = None
    default_value: Any = None
    confidence_threshold: float = 0.7
    description: str = ""
    
    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []
        if self.validation_rules is None:
            self.validation_rules = []

@dataclass
class ExtractedFieldValue:
    """Extracted field value with metadata."""
    field_name: str
    value: Any
    confidence: float
    source_text: str
    bounding_box: Optional[Dict] = None
    extraction_method: str = "ocr"
    validation_results: List[Dict] = None
    normalized_value: Any = None
    
    def __post_init__(self):
        if self.validation_results is None:
            self.validation_results = []

@dataclass
class TableColumn:
    """Table column definition."""
    name: str
    field_type: FieldType
    required: bool = False
    validation_rules: List[str] = None
    
    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = []

@dataclass
class TableSchema:
    """Schema for table extraction."""
    name: str
    columns: List[TableColumn]
    min_rows: int = 0
    max_rows: int = 1000
    header_required: bool = True
    sum_columns: List[str] = None  # Columns that should sum to totals
    
    def __post_init__(self):
        if self.sum_columns is None:
            self.sum_columns = []

@dataclass
class DocumentSchema:
    """Complete document schema definition."""
    template_type: DocumentTemplate
    version: str
    fields: List[FieldSchema]
    tables: List[TableSchema] = None
    business_rules: List[str] = None
    required_confidence: float = 0.8
    
    def __post_init__(self):
        if self.tables is None:
            self.tables = []
        if self.business_rules is None:
            self.business_rules = []

@dataclass
@dataclass
class ExtractionResult:
    """Complete field extraction result."""
    document_id: str
    template_type: DocumentTemplate
    extracted_fields: List[ExtractedFieldValue]
    extracted_tables: List[Dict]
    validation_results: List[ValidationResult]
    overall_confidence: float
    requires_hitl: bool
    processing_time: float
    timestamp: datetime
    
    def get_field_value(self, field_name: str) -> Optional[Any]:
        """Get normalized value for a field."""
        for field in self.extracted_fields:
            if field.field_name == field_name:
                return field.normalized_value or field.value
        return None
    
    def get_field_confidence(self, field_name: str) -> float:
        """Get confidence score for a field."""
        for field in self.extracted_fields:
            if field.field_name == field_name:
                return field.confidence
        return 0.0

class FieldExtractor:
    """
    Main field extraction engine with template-based processing.
    """
    
    def __init__(
        self,
        schema_directory: str = "./config/schemas",
        enable_business_rules: bool = True,
        confidence_threshold: float = 0.7,
        hitl_threshold: float = 0.6
    ):
        """
        Initialize field extractor.
        
        Args:
            schema_directory: Directory containing document schemas
            enable_business_rules: Enable business rule validation
            confidence_threshold: Default confidence threshold
            hitl_threshold: Threshold below which documents go to HITL
        """
        self.schema_directory = Path(schema_directory)
        self.enable_business_rules = enable_business_rules
        self.confidence_threshold = confidence_threshold
        self.hitl_threshold = hitl_threshold
        
        # Load document schemas
        self.schemas: Dict[DocumentTemplate, DocumentSchema] = {}
        self._load_schemas()
        
        # Business rule validators
        self.validators = self._initialize_validators()
        
        # Normalization functions
        self.normalizers = self._initialize_normalizers()
        
        # Enhanced validation engine
        if HAS_VALIDATION_ENGINE:
            self.comprehensive_validator = ComprehensiveValidator()
            logger.info("Enhanced validation engine initialized")
        else:
            self.comprehensive_validator = None
            logger.warning("Enhanced validation engine not available - using basic validation")
        
        # Statistics
        self.stats = {
            "documents_processed": 0,
            "fields_extracted": 0,
            "validation_errors": 0,
            "hitl_documents": 0,
            "processing_time": 0.0
        }
        
        logger.info(f"Field extractor initialized with {len(self.schemas)} schemas")
    
    def extract_fields(
        self,
        ocr_result: Any,
        document_type: DocumentTemplate,
        document_id: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract fields from OCR result using document template.
        
        Args:
            ocr_result: Azure Document Intelligence OCR result
            document_type: Type of document to process
            document_id: Unique document identifier
            
        Returns:
            Complete extraction result with validation
        """
        start_time = datetime.now()
        
        if document_id is None:
            document_id = str(uuid.uuid4())
        
        logger.info(f"Extracting fields for document {document_id} ({document_type.value})")
        
        # Get document schema
        schema = self.schemas.get(document_type)
        if not schema:
            raise ValueError(f"No schema found for document type: {document_type}")
        
        # Extract fields using template
        extracted_fields = self._extract_template_fields(ocr_result, schema)
        
        # Extract tables if defined
        extracted_tables = self._extract_tables(ocr_result, schema)
        
        # Normalize field values
        self._normalize_fields(extracted_fields)
        
        # Validate fields and apply business rules
        validation_results = []
        if self.enable_business_rules:
            validation_results = self._validate_extraction(
                extracted_fields, extracted_tables, schema
            )
        
        # Calculate confidence scores
        overall_confidence = self._calculate_overall_confidence(
            extracted_fields, validation_results
        )
        
        # Determine if HITL is required
        requires_hitl = self._requires_hitl(
            extracted_fields, validation_results, overall_confidence, schema
        )
        
        # Create result
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = ExtractionResult(
            document_id=document_id,
            template_type=document_type,
            extracted_fields=extracted_fields,
            extracted_tables=extracted_tables,
            validation_results=validation_results,
            overall_confidence=overall_confidence,
            requires_hitl=requires_hitl,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
        # Update statistics
        self._update_stats(result)
        
        logger.info(
            f"Field extraction completed for {document_id}: "
            f"{len(extracted_fields)} fields, confidence={overall_confidence:.2f}, "
            f"HITL={'required' if requires_hitl else 'not required'}"
        )
        
        return result
    
    def _load_schemas(self):
        """Load document schemas from configuration files."""
        # Create default schemas if directory doesn't exist
        if not self.schema_directory.exists():
            self.schema_directory.mkdir(parents=True, exist_ok=True)
            self._create_default_schemas()
        
        # Load schemas from files
        for schema_file in self.schema_directory.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema_data = json.load(f)
                
                # Convert to DocumentSchema
                template_type = DocumentTemplate(schema_data["template_type"])
                
                # Convert fields
                fields = []
                for field_data in schema_data["fields"]:
                    field = FieldSchema(
                        name=field_data["name"],
                        field_type=FieldType(field_data["field_type"]),
                        required=field_data.get("required", False),
                        synonyms=field_data.get("synonyms", []),
                        validation_rules=field_data.get("validation_rules", []),
                        default_value=field_data.get("default_value"),
                        confidence_threshold=field_data.get("confidence_threshold", 0.7),
                        description=field_data.get("description", "")
                    )
                    fields.append(field)
                
                # Convert tables
                tables = []
                for table_data in schema_data.get("tables", []):
                    columns = []
                    for col_data in table_data["columns"]:
                        column = TableColumn(
                            name=col_data["name"],
                            field_type=FieldType(col_data["field_type"]),
                            required=col_data.get("required", False),
                            validation_rules=col_data.get("validation_rules", [])
                        )
                        columns.append(column)
                    
                    table = TableSchema(
                        name=table_data["name"],
                        columns=columns,
                        min_rows=table_data.get("min_rows", 0),
                        max_rows=table_data.get("max_rows", 1000),
                        header_required=table_data.get("header_required", True),
                        sum_columns=table_data.get("sum_columns", [])
                    )
                    tables.append(table)
                
                schema = DocumentSchema(
                    template_type=template_type,
                    version=schema_data["version"],
                    fields=fields,
                    tables=tables,
                    business_rules=schema_data.get("business_rules", []),
                    required_confidence=schema_data.get("required_confidence", 0.8)
                )
                
                self.schemas[template_type] = schema
                logger.info(f"Loaded schema for {template_type.value}")
                
            except Exception as e:
                logger.error(f"Failed to load schema from {schema_file}: {e}")
    
    def _create_default_schemas(self):
        """Create default document schemas."""
        
        # Invoice schema
        invoice_schema = {
            "template_type": "invoice",
            "version": "1.0",
            "required_confidence": 0.8,
            "fields": [
                {
                    "name": "invoice_number",
                    "field_type": "text",
                    "required": True,
                    "synonyms": ["invoice #", "inv #", "invoice id", "bill number"],
                    "validation_rules": ["not_empty", "alphanumeric"],
                    "confidence_threshold": 0.9,
                    "description": "Unique invoice identifier"
                },
                {
                    "name": "vendor_name",
                    "field_type": "text",
                    "required": True,
                    "synonyms": ["from", "bill from", "seller", "company"],
                    "validation_rules": ["not_empty", "min_length:2"],
                    "confidence_threshold": 0.8
                },
                {
                    "name": "invoice_date",
                    "field_type": "date",
                    "required": True,
                    "synonyms": ["date", "bill date", "invoice dt"],
                    "validation_rules": ["valid_date", "past_date"],
                    "confidence_threshold": 0.8
                },
                {
                    "name": "due_date",
                    "field_type": "date",
                    "required": False,
                    "synonyms": ["payment due", "due", "pay by"],
                    "validation_rules": ["valid_date", "future_date_from_invoice"]
                },
                {
                    "name": "total_amount",
                    "field_type": "currency",
                    "required": True,
                    "synonyms": ["total", "amount due", "balance", "grand total"],
                    "validation_rules": ["positive_amount", "currency_format"],
                    "confidence_threshold": 0.9
                },
                {
                    "name": "subtotal",
                    "field_type": "currency",
                    "required": False,
                    "synonyms": ["sub total", "net amount", "before tax"],
                    "validation_rules": ["positive_amount"]
                },
                {
                    "name": "tax_amount",
                    "field_type": "currency",
                    "required": False,
                    "synonyms": ["tax", "vat", "sales tax", "gst"],
                    "validation_rules": ["non_negative_amount"]
                },
                {
                    "name": "customer_name",
                    "field_type": "text",
                    "required": False,
                    "synonyms": ["bill to", "customer", "client", "buyer"],
                    "validation_rules": ["not_empty"]
                }
            ],
            "tables": [
                {
                    "name": "line_items",
                    "header_required": True,
                    "min_rows": 1,
                    "sum_columns": ["amount"],
                    "columns": [
                        {
                            "name": "description",
                            "field_type": "text",
                            "required": True,
                            "validation_rules": ["not_empty"]
                        },
                        {
                            "name": "quantity",
                            "field_type": "number",
                            "required": False,
                            "validation_rules": ["positive_number"]
                        },
                        {
                            "name": "unit_price",
                            "field_type": "currency",
                            "required": False,
                            "validation_rules": ["positive_amount"]
                        },
                        {
                            "name": "amount",
                            "field_type": "currency",
                            "required": True,
                            "validation_rules": ["positive_amount"]
                        }
                    ]
                }
            ],
            "business_rules": [
                "total_equals_subtotal_plus_tax",
                "due_date_after_invoice_date",
                "line_items_sum_to_subtotal"
            ]
        }
        
        # Receipt schema
        receipt_schema = {
            "template_type": "receipt",
            "version": "1.0",
            "required_confidence": 0.7,
            "fields": [
                {
                    "name": "merchant_name",
                    "field_type": "text",
                    "required": True,
                    "synonyms": ["store", "retailer", "shop"],
                    "validation_rules": ["not_empty"],
                    "confidence_threshold": 0.8
                },
                {
                    "name": "transaction_date",
                    "field_type": "date",
                    "required": True,
                    "synonyms": ["date", "purchase date", "trans date"],
                    "validation_rules": ["valid_date", "past_date"],
                    "confidence_threshold": 0.8
                },
                {
                    "name": "total_amount",
                    "field_type": "currency",
                    "required": True,
                    "synonyms": ["total", "amount", "grand total"],
                    "validation_rules": ["positive_amount"],
                    "confidence_threshold": 0.9
                },
                {
                    "name": "payment_method",
                    "field_type": "text",
                    "required": False,
                    "synonyms": ["payment", "paid by", "method"],
                    "validation_rules": ["valid_payment_method"]
                }
            ],
            "tables": [
                {
                    "name": "items",
                    "header_required": False,
                    "min_rows": 1,
                    "columns": [
                        {
                            "name": "item_name",
                            "field_type": "text",
                            "required": True
                        },
                        {
                            "name": "price",
                            "field_type": "currency",
                            "required": True,
                            "validation_rules": ["positive_amount"]
                        }
                    ]
                }
            ],
            "business_rules": [
                "items_sum_to_total"
            ]
        }
        
        # Save schemas
        schemas = [invoice_schema, receipt_schema]
        
        for schema in schemas:
            schema_file = self.schema_directory / f"{schema['template_type']}.json"
            with open(schema_file, 'w') as f:
                json.dump(schema, f, indent=2)
            
            logger.info(f"Created default schema: {schema_file}")
    
    def _extract_template_fields(
        self,
        ocr_result: Any,
        schema: DocumentSchema
    ) -> List[ExtractedFieldValue]:
        """Extract fields using template matching and synonyms."""
        extracted_fields = []
        
        # Get all text from OCR result
        full_text = ocr_result.full_text.lower() if ocr_result.full_text else ""
        
        # Process structured fields from prebuilt models
        ocr_fields = getattr(ocr_result, 'fields', [])
        
        for field_schema in schema.fields:
            field_value = self._extract_single_field(
                field_schema, full_text, ocr_fields, ocr_result
            )
            if field_value:
                extracted_fields.append(field_value)
        
        return extracted_fields
    
    def _extract_single_field(
        self,
        field_schema: FieldSchema,
        full_text: str,
        ocr_fields: List,
        ocr_result: Any
    ) -> Optional[ExtractedFieldValue]:
        """Extract a single field using multiple strategies."""
        
        # Strategy 1: Direct match from OCR structured fields
        for ocr_field in ocr_fields:
            if hasattr(ocr_field, 'field_name') and ocr_field.field_name.lower() == field_schema.name.lower():
                return ExtractedFieldValue(
                    field_name=field_schema.name,
                    value=ocr_field.value,
                    confidence=getattr(ocr_field, 'confidence', 1.0),
                    source_text=ocr_field.value,
                    bounding_box=self._get_field_bbox(ocr_field),
                    extraction_method="ocr_structured"
                )
        
        # Strategy 2: Synonym matching in text
        for synonym in [field_schema.name] + field_schema.synonyms:
            value = self._extract_field_by_pattern(
                synonym, field_schema.field_type, full_text, ocr_result
            )
            if value:
                return value
        
        # Strategy 3: Field type-specific patterns
        pattern_value = self._extract_by_field_type_pattern(
            field_schema, full_text, ocr_result
        )
        if pattern_value:
            return pattern_value
        
        # Return default value if configured
        if field_schema.default_value is not None:
            return ExtractedFieldValue(
                field_name=field_schema.name,
                value=field_schema.default_value,
                confidence=0.5,
                source_text="default",
                extraction_method="default"
            )
        
        return None
    
    def _extract_field_by_pattern(
        self,
        field_label: str,
        field_type: FieldType,
        full_text: str,
        ocr_result: Any
    ) -> Optional[ExtractedFieldValue]:
        """Extract field value using label pattern matching."""
        
        # Common patterns for different field types
        patterns = {
            FieldType.TEXT: r"(?:^|\s)({label})[:\s]+([^\n\r]+)",
            FieldType.NUMBER: r"(?:^|\s)({label})[:\s]+([0-9,]+(?:\.[0-9]+)?)",
            FieldType.CURRENCY: r"(?:^|\s)({label})[:\s]+(\$?[0-9,]+(?:\.[0-9]{{2}})?)",
            FieldType.DATE: r"(?:^|\s)({label})[:\s]+([0-9]{{1,2}}[\/\-][0-9]{{1,2}}[\/\-][0-9]{{2,4}})",
            FieldType.EMAIL: r"(?:^|\s)({label})[:\s]+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{{2,}})",
            FieldType.PHONE: r"(?:^|\s)({label})[:\s]+(\+?[0-9\-\(\)\s]+)"
        }
        
        pattern = patterns.get(field_type, patterns[FieldType.TEXT])
        regex = re.compile(pattern.format(label=re.escape(field_label)), re.IGNORECASE | re.MULTILINE)
        
        match = regex.search(full_text)
        if match:
            return ExtractedFieldValue(
                field_name=field_label,
                value=match.group(2).strip(),
                confidence=0.8,
                source_text=match.group(0),
                extraction_method="pattern_match"
            )
        
        return None
    
    def _extract_by_field_type_pattern(
        self,
        field_schema: FieldSchema,
        full_text: str,
        ocr_result: Any
    ) -> Optional[ExtractedFieldValue]:
        """Extract field using field type-specific patterns."""
        
        if field_schema.field_type == FieldType.CURRENCY:
            # Look for currency amounts
            currency_pattern = r"\$[0-9,]+(?:\.[0-9]{2})?"
            matches = re.findall(currency_pattern, full_text)
            if matches:
                # Return the largest amount (likely total)
                amounts = [float(m.replace('$', '').replace(',', '')) for m in matches]
                max_amount = max(amounts)
                return ExtractedFieldValue(
                    field_name=field_schema.name,
                    value=f"${max_amount:,.2f}",
                    confidence=0.6,
                    source_text=f"${max_amount:,.2f}",
                    extraction_method="pattern_currency"
                )
        
        elif field_schema.field_type == FieldType.DATE:
            # Look for date patterns
            date_patterns = [
                r"[0-9]{1,2}\/[0-9]{1,2}\/[0-9]{2,4}",
                r"[0-9]{1,2}\-[0-9]{1,2}\-[0-9]{2,4}",
                r"[A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{2,4}"
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, full_text)
                if matches:
                    return ExtractedFieldValue(
                        field_name=field_schema.name,
                        value=matches[0],
                        confidence=0.7,
                        source_text=matches[0],
                        extraction_method="pattern_date"
                    )
        
        return None
    
    def _extract_tables(
        self,
        ocr_result: Any,
        schema: DocumentSchema
    ) -> List[Dict]:
        """Extract table data according to schema."""
        extracted_tables = []
        
        ocr_tables = getattr(ocr_result, 'tables', [])
        
        for table_schema in schema.tables:
            # Find matching table from OCR results
            matching_table = self._find_matching_table(table_schema, ocr_tables)
            
            if matching_table:
                processed_table = self._process_table(matching_table, table_schema)
                extracted_tables.append(processed_table)
        
        return extracted_tables
    
    def _find_matching_table(self, table_schema: TableSchema, ocr_tables: List) -> Optional[Any]:
        """Find OCR table that matches the schema."""
        # For now, return the first table that meets minimum requirements
        for table in ocr_tables:
            if hasattr(table, 'rows') and len(table.rows) >= table_schema.min_rows:
                return table
        return None
    
    def _process_table(self, ocr_table: Any, table_schema: TableSchema) -> Dict:
        """Process OCR table according to schema."""
        processed_rows = []
        
        # Extract table data
        table_rows = getattr(ocr_table, 'rows', [])
        
        # Skip header row if required
        start_row = 1 if table_schema.header_required and len(table_rows) > 1 else 0
        
        for i, row in enumerate(table_rows[start_row:], start_row):
            processed_row = {}
            
            for j, column_schema in enumerate(table_schema.columns):
                if j < len(row):
                    cell_value = row[j]
                    
                    # Normalize cell value based on column type
                    normalized_value = self._normalize_cell_value(
                        cell_value, column_schema.field_type
                    )
                    
                    processed_row[column_schema.name] = {
                        'value': normalized_value,
                        'raw_value': cell_value,
                        'confidence': 0.8  # Default confidence for table cells
                    }
            
            processed_rows.append(processed_row)
        
        return {
            'name': table_schema.name,
            'rows': processed_rows,
            'row_count': len(processed_rows),
            'confidence': getattr(ocr_table, 'confidence', 0.8)
        }
    
    def _normalize_cell_value(self, value: str, field_type: FieldType) -> Any:
        """Normalize cell value based on field type."""
        if not value or not value.strip():
            return None
        
        value = value.strip()
        
        try:
            if field_type == FieldType.CURRENCY:
                # Remove currency symbols and normalize
                clean_value = re.sub(r'[^\d.,\-]', '', value)
                return float(clean_value.replace(',', ''))
            
            elif field_type == FieldType.NUMBER:
                return float(value.replace(',', ''))
            
            elif field_type == FieldType.DATE:
                return date_parser.parse(value).date()
            
            else:
                return value
                
        except (ValueError, InvalidOperation):
            return value  # Return original if normalization fails
    
    def _normalize_fields(self, extracted_fields: List[ExtractedFieldValue]):
        """Normalize extracted field values."""
        for field in extracted_fields:
            normalizer = self.normalizers.get(field.field_name)
            if normalizer:
                try:
                    field.normalized_value = normalizer(field.value)
                except Exception as e:
                    logger.warning(f"Normalization failed for {field.field_name}: {e}")
                    field.normalized_value = field.value
    
    def _initialize_normalizers(self) -> Dict[str, callable]:
        """Initialize field normalization functions."""
        return {
            'invoice_date': self._normalize_date,
            'due_date': self._normalize_date,
            'transaction_date': self._normalize_date,
            'total_amount': self._normalize_currency,
            'subtotal': self._normalize_currency,
            'tax_amount': self._normalize_currency,
            'phone': self._normalize_phone,
            'email': self._normalize_email
        }
    
    def _normalize_date(self, value: str) -> Optional[date]:
        """Normalize date values."""
        if not value:
            return None
        
        try:
            return date_parser.parse(str(value)).date()
        except (ValueError, TypeError):
            return None
    
    def _normalize_currency(self, value: str) -> Optional[Decimal]:
        """Normalize currency values."""
        if not value:
            return None
        
        try:
            # Remove currency symbols and normalize
            clean_value = re.sub(r'[^\d.,\-]', '', str(value))
            return Decimal(clean_value.replace(',', ''))
        except (ValueError, InvalidOperation):
            return None
    
    def _normalize_phone(self, value: str) -> Optional[str]:
        """Normalize phone numbers."""
        if not value:
            return None
        
        try:
            # Parse phone number (assuming US if no country code)
            parsed = phonenumbers.parse(value, "US")
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        except phonenumbers.NumberParseException:
            return value  # Return original if parsing fails
    
    def _normalize_email(self, value: str) -> Optional[str]:
        """Normalize email addresses."""
        if not value:
            return None
        
        try:
            validated = validate_email(value)
            return validated.email
        except EmailNotValidError:
            return value  # Return original if validation fails
    
    def _initialize_validators(self) -> Dict[str, callable]:
        """Initialize validation functions."""
        return {
            'not_empty': lambda x: bool(x and str(x).strip()),
            'min_length': lambda x, min_len: len(str(x)) >= min_len if x else False,
            'max_length': lambda x, max_len: len(str(x)) <= max_len if x else True,
            'alphanumeric': lambda x: bool(re.match(r'^[a-zA-Z0-9\s\-_]+$', str(x))) if x else False,
            'valid_date': self._validate_date,
            'past_date': self._validate_past_date,
            'future_date': self._validate_future_date,
            'positive_amount': self._validate_positive_amount,
            'non_negative_amount': self._validate_non_negative_amount,
            'currency_format': self._validate_currency_format,
            'valid_email': self._validate_email,
            'valid_phone': self._validate_phone,
            'valid_payment_method': self._validate_payment_method
        }
    
    def _validate_date(self, value: Any) -> bool:
        """Validate date format."""
        if not value:
            return False
        
        try:
            if isinstance(value, date):
                return True
            date_parser.parse(str(value))
            return True
        except (ValueError, TypeError):
            return False
    
    def _validate_past_date(self, value: Any) -> bool:
        """Validate that date is in the past."""
        if not self._validate_date(value):
            return False
        
        try:
            if isinstance(value, date):
                test_date = value
            else:
                test_date = date_parser.parse(str(value)).date()
            
            return test_date <= date.today()
        except (ValueError, TypeError):
            return False
    
    def _validate_future_date(self, value: Any) -> bool:
        """Validate that date is in the future."""
        if not self._validate_date(value):
            return False
        
        try:
            if isinstance(value, date):
                test_date = value
            else:
                test_date = date_parser.parse(str(value)).date()
            
            return test_date >= date.today()
        except (ValueError, TypeError):
            return False
    
    def _validate_positive_amount(self, value: Any) -> bool:
        """Validate positive monetary amount."""
        try:
            if isinstance(value, (int, float, Decimal)):
                return value > 0
            
            # Parse string currency
            clean_value = re.sub(r'[^\d.,\-]', '', str(value))
            amount = float(clean_value.replace(',', ''))
            return amount > 0
        except (ValueError, TypeError):
            return False
    
    def _validate_non_negative_amount(self, value: Any) -> bool:
        """Validate non-negative monetary amount."""
        try:
            if isinstance(value, (int, float, Decimal)):
                return value >= 0
            
            # Parse string currency
            clean_value = re.sub(r'[^\d.,\-]', '', str(value))
            amount = float(clean_value.replace(',', ''))
            return amount >= 0
        except (ValueError, TypeError):
            return False
    
    def _validate_currency_format(self, value: Any) -> bool:
        """Validate currency format."""
        if not value:
            return False
        
        currency_pattern = r'^\$?[0-9,]+(?:\.[0-9]{2})?$'
        return bool(re.match(currency_pattern, str(value).strip()))
    
    def _validate_email(self, value: Any) -> bool:
        """Validate email format."""
        if not value:
            return False
        
        try:
            validate_email(str(value))
            return True
        except EmailNotValidError:
            return False
    
    def _validate_phone(self, value: Any) -> bool:
        """Validate phone number format."""
        if not value:
            return False
        
        try:
            parsed = phonenumbers.parse(str(value), "US")
            return phonenumbers.is_valid_number(parsed)
        except phonenumbers.NumberParseException:
            return False
    
    def _validate_payment_method(self, value: Any) -> bool:
        """Validate payment method."""
        if not value:
            return False
        
        valid_methods = [
            'cash', 'credit card', 'debit card', 'check', 'cheque',
            'visa', 'mastercard', 'amex', 'discover', 'paypal',
            'bank transfer', 'wire transfer', 'ach'
        ]
        
        return any(method in str(value).lower() for method in valid_methods)
    
    def _validate_extraction(
        self,
        extracted_fields: List[ExtractedFieldValue],
        extracted_tables: List[Dict],
        schema: DocumentSchema
    ) -> List[ValidationResult]:
        """Validate extracted fields and apply business rules."""
        validation_results = []
        
        # Field-level validation
        for field in extracted_fields:
            field_schema = self._get_field_schema(field.field_name, schema)
            if field_schema:
                field_results = self._validate_field(field, field_schema)
                validation_results.extend(field_results)
        
        # Business rule validation
        for rule_name in schema.business_rules:
            rule_result = self._apply_business_rule(
                rule_name, extracted_fields, extracted_tables, schema
            )
            if rule_result:
                validation_results.append(rule_result)
        
        return validation_results
    
    def _validate_field(
        self,
        field: ExtractedFieldValue,
        field_schema: FieldSchema
    ) -> List[ValidationResult]:
        """Validate a single field against its schema."""
        results = []
        
        for rule in field_schema.validation_rules:
            # Parse rule (format: "rule_name" or "rule_name:parameter")
            rule_parts = rule.split(':', 1)
            rule_name = rule_parts[0]
            rule_param = rule_parts[1] if len(rule_parts) > 1 else None
            
            validator = self.validators.get(rule_name)
            if not validator:
                continue
            
            try:
                # Apply validation
                if rule_param:
                    if rule_name == 'min_length':
                        is_valid = validator(field.normalized_value or field.value, int(rule_param))
                    elif rule_name == 'max_length':
                        is_valid = validator(field.normalized_value or field.value, int(rule_param))
                    else:
                        is_valid = validator(field.normalized_value or field.value, rule_param)
                else:
                    is_valid = validator(field.normalized_value or field.value)
                
                if not is_valid:
                    results.append(ValidationResult(
                        field_name=field.field_name,
                        is_valid=False,
                        severity=ValidationSeverity.ERROR if field_schema.required else ValidationSeverity.WARNING,
                        message=f"Field '{field.field_name}' failed validation rule '{rule}'",
                        rule_name=rule
                    ))
                
            except Exception as e:
                logger.error(f"Validation error for {field.field_name} rule {rule}: {e}")
                results.append(ValidationResult(
                    field_name=field.field_name,
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation rule '{rule}' failed with error: {e}",
                    rule_name=rule
                ))
        
        return results
    
    def _apply_business_rule(
        self,
        rule_name: str,
        extracted_fields: List[ExtractedFieldValue],
        extracted_tables: List[Dict],
        schema: DocumentSchema
    ) -> Optional[ValidationResult]:
        """Apply business rule validation."""
        
        # Get field values for business rule checks
        field_values = {f.field_name: f.normalized_value or f.value for f in extracted_fields}
        
        if rule_name == "total_equals_subtotal_plus_tax":
            return self._validate_total_calculation(field_values)
        
        elif rule_name == "due_date_after_invoice_date":
            return self._validate_due_date(field_values)
        
        elif rule_name == "line_items_sum_to_subtotal":
            return self._validate_line_items_sum(field_values, extracted_tables)
        
        elif rule_name == "items_sum_to_total":
            return self._validate_items_sum(field_values, extracted_tables)
        
        return None
    
    def _validate_total_calculation(self, field_values: Dict) -> Optional[ValidationResult]:
        """Validate that total = subtotal + tax."""
        total = field_values.get('total_amount')
        subtotal = field_values.get('subtotal')
        tax = field_values.get('tax_amount', 0)
        
        if not all(isinstance(x, (int, float, Decimal)) for x in [total, subtotal, tax] if x is not None):
            return None
        
        if total and subtotal:
            expected_total = Decimal(str(subtotal)) + Decimal(str(tax or 0))
            actual_total = Decimal(str(total))
            
            # Allow small rounding differences
            if abs(expected_total - actual_total) > Decimal('0.01'):
                return ValidationResult(
                    field_name="total_amount",
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Total amount ({actual_total}) does not equal subtotal + tax ({expected_total})",
                    rule_name="total_equals_subtotal_plus_tax",
                    suggested_value=float(expected_total)
                )
        
        return None
    
    def _validate_due_date(self, field_values: Dict) -> Optional[ValidationResult]:
        """Validate that due date is after invoice date."""
        invoice_date = field_values.get('invoice_date')
        due_date = field_values.get('due_date')
        
        if invoice_date and due_date:
            try:
                if isinstance(invoice_date, str):
                    invoice_date = date_parser.parse(invoice_date).date()
                if isinstance(due_date, str):
                    due_date = date_parser.parse(due_date).date()
                
                if due_date <= invoice_date:
                    return ValidationResult(
                        field_name="due_date",
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Due date ({due_date}) should be after invoice date ({invoice_date})",
                        rule_name="due_date_after_invoice_date"
                    )
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _validate_line_items_sum(
        self,
        field_values: Dict,
        extracted_tables: List[Dict]
    ) -> Optional[ValidationResult]:
        """Validate that line items sum to subtotal."""
        subtotal = field_values.get('subtotal')
        
        if not subtotal:
            return None
        
        # Find line items table
        line_items_table = None
        for table in extracted_tables:
            if table['name'] == 'line_items':
                line_items_table = table
                break
        
        if not line_items_table:
            return None
        
        # Sum amounts from line items
        total_from_items = Decimal('0')
        for row in line_items_table['rows']:
            amount = row.get('amount', {}).get('value')
            if amount:
                try:
                    total_from_items += Decimal(str(amount))
                except (ValueError, InvalidOperation):
                    continue
        
        subtotal_decimal = Decimal(str(subtotal))
        
        # Allow small rounding differences
        if abs(total_from_items - subtotal_decimal) > Decimal('0.01'):
            return ValidationResult(
                field_name="subtotal",
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Line items sum ({total_from_items}) does not equal subtotal ({subtotal_decimal})",
                rule_name="line_items_sum_to_subtotal",
                suggested_value=float(total_from_items)
            )
        
        return None
    
    def _validate_items_sum(
        self,
        field_values: Dict,
        extracted_tables: List[Dict]
    ) -> Optional[ValidationResult]:
        """Validate that items sum to total (for receipts)."""
        total = field_values.get('total_amount')
        
        if not total:
            return None
        
        # Find items table
        items_table = None
        for table in extracted_tables:
            if table['name'] == 'items':
                items_table = table
                break
        
        if not items_table:
            return None
        
        # Sum prices from items
        total_from_items = Decimal('0')
        for row in items_table['rows']:
            price = row.get('price', {}).get('value')
            if price:
                try:
                    total_from_items += Decimal(str(price))
                except (ValueError, InvalidOperation):
                    continue
        
        total_decimal = Decimal(str(total))
        
        # Allow small rounding differences
        if abs(total_from_items - total_decimal) > Decimal('0.01'):
            return ValidationResult(
                field_name="total_amount",
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Items sum ({total_from_items}) does not equal total ({total_decimal})",
                rule_name="items_sum_to_total",
                suggested_value=float(total_from_items)
            )
        
        return None
    
    def _get_field_schema(self, field_name: str, schema: DocumentSchema) -> Optional[FieldSchema]:
        """Get field schema by name."""
        for field_schema in schema.fields:
            if field_schema.name == field_name:
                return field_schema
        return None
    
    def _calculate_overall_confidence(
        self,
        extracted_fields: List[ExtractedFieldValue],
        validation_results: List[ValidationResult]
    ) -> float:
        """Calculate overall document confidence score."""
        if not extracted_fields:
            return 0.0
        
        # Base confidence from field extractions
        field_confidences = [f.confidence for f in extracted_fields]
        base_confidence = sum(field_confidences) / len(field_confidences)
        
        # Reduce confidence for validation errors
        error_count = sum(1 for r in validation_results if not r.is_valid and r.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for r in validation_results if not r.is_valid and r.severity == ValidationSeverity.WARNING)
        
        # Penalty for errors and warnings
        error_penalty = error_count * 0.1
        warning_penalty = warning_count * 0.05
        
        final_confidence = max(0.0, base_confidence - error_penalty - warning_penalty)
        
        return min(1.0, final_confidence)
    
    def _requires_hitl(
        self,
        extracted_fields: List[ExtractedFieldValue],
        validation_results: List[ValidationResult],
        overall_confidence: float,
        schema: DocumentSchema
    ) -> bool:
        """Determine if document requires human-in-the-loop review."""
        
        # Check overall confidence threshold
        if overall_confidence < self.hitl_threshold:
            return True
        
        # Check if any required fields are missing or below threshold
        for field_schema in schema.fields:
            if field_schema.required:
                field = next((f for f in extracted_fields if f.field_name == field_schema.name), None)
                if not field or field.confidence < field_schema.confidence_threshold:
                    return True
        
        # Check for validation errors
        error_count = sum(1 for r in validation_results if not r.is_valid and r.severity == ValidationSeverity.ERROR)
        if error_count > 0:
            return True
        
        # Check document-level confidence requirement
        if overall_confidence < schema.required_confidence:
            return True
        
        return False
    
    def _get_field_bbox(self, ocr_field: Any) -> Optional[Dict]:
        """Extract bounding box from OCR field."""
        if hasattr(ocr_field, 'bounding_box') and ocr_field.bounding_box:
            bbox = ocr_field.bounding_box
            return {
                'x': bbox.x,
                'y': bbox.y,
                'width': bbox.width,
                'height': bbox.height
            }
        return None
    
    def _update_stats(self, result: ExtractionResult):
        """Update processing statistics."""
        self.stats["documents_processed"] += 1
        self.stats["fields_extracted"] += len(result.extracted_fields)
        self.stats["validation_errors"] += sum(
            1 for r in result.validation_results 
            if not r.is_valid and r.severity == ValidationSeverity.ERROR
        )
        if result.requires_hitl:
            self.stats["hitl_documents"] += 1
        self.stats["processing_time"] += result.processing_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        if stats["documents_processed"] > 0:
            stats["average_fields_per_document"] = stats["fields_extracted"] / stats["documents_processed"]
            stats["average_processing_time"] = stats["processing_time"] / stats["documents_processed"]
            stats["hitl_rate"] = stats["hitl_documents"] / stats["documents_processed"]
        return stats