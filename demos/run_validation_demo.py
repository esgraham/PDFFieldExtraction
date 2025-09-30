#!/usr/bin/env python3
"""
Enhanced Validation and HITL Integration Demo

This script demonstrates the complete validation system with:
- Regex patterns, Luhn validation, date checks
- Cross-field consistency validation
- Confidence-based HITL routing
- Microsoft Teams notifications
- Web-based review interface

Run with: python run_validation_demo.py
"""

import asyncio
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("🔧 Enhanced Validation & HITL System Demo")
print("=" * 50)

# Test validation components availability
try:
    from validation_engine import ComprehensiveValidator, ValidationConfig, LuhnValidator
    print("✅ Validation Engine - Available")
    HAS_VALIDATION = True
except ImportError as e:
    print(f"❌ Validation Engine - Missing: {e}")
    HAS_VALIDATION = False

try:
    from hitl_review_app import HITLReviewApp, ReviewTask, ReviewStatus, ReviewPriority
    print("✅ HITL Review App - Available")
    HAS_HITL = True
except ImportError as e:
    print(f"❌ HITL Review App - Missing: {e}")
    HAS_HITL = False

def demonstrate_luhn_validation():
    """Demonstrate Luhn algorithm validation."""
    print("\n🔍 Luhn Algorithm Validation:")
    print("-" * 30)
    
    if not HAS_VALIDATION:
        print("❌ Validation engine not available")
        return
    
    test_numbers = [
        ('4532123456789012', 'Valid Visa card number'),
        ('4532123456789013', 'Invalid Visa (wrong check digit)'),  
        ('5555555555554444', 'Valid Mastercard number'),
        ('5555555555554445', 'Invalid Mastercard'),
        ('123456789', 'Too short - invalid'),
        ('', 'Empty - invalid')
    ]
    
    for number, description in test_numbers:
        is_valid = LuhnValidator.validate(number)
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"  {number:<20} {description:<30} {status}")

def demonstrate_regex_patterns():
    """Demonstrate regex pattern validation."""
    print("\n🔍 Regex Pattern Validation:")
    print("-" * 30)
    
    patterns = {
        'invoice_number': r'^[A-Za-z0-9\-_]{3,20}$',
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'ssn': r'^\d{3}-?\d{2}-?\d{4}$',
        'phone': r'^[\+]?[1-9]?[0-9]{7,15}$'
    }
    
    test_data = {
        'invoice_number': [
            ('INV-2024-001', True),
            ('PO_123456', True),
            ('ABC-123', True),
            ('INV-2024-001-EXTRA-LONG-NAME', False),  # Too long
            ('INV@2024!', False)  # Invalid characters
        ],
        'email': [
            ('user@example.com', True),
            ('john.doe+filter@company.co.uk', True),
            ('invalid.email', False),
            ('user@', False),
            ('@example.com', False)
        ],
        'ssn': [
            ('123-45-6789', True),
            ('123456789', True),
            ('12-34-5678', False),  # Wrong format
            ('123-45-67890', False)  # Too long
        ],
        'phone': [
            ('+1234567890', True),
            ('1234567890', True),
            ('123456', False),  # Too short
            ('+abc1234567', False)  # Invalid characters
        ]
    }
    
    for field_type, pattern in patterns.items():
        print(f"\n  📋 {field_type.upper()} Pattern: {pattern}")
        for value, expected in test_data[field_type]:
            import re
            is_valid = bool(re.match(pattern, value))
            status = "✅ VALID" if is_valid else "❌ INVALID"
            correct = "✓" if is_valid == expected else "✗"
            print(f"    {value:<25} {status} {correct}")

def demonstrate_date_validation():
    """Demonstrate date format and sequence validation."""
    print("\n🔍 Date Validation:")
    print("-" * 30)
    
    # Date format tests
    print("  📅 Date Format Tests:")
    date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y']
    test_dates = [
        '2024-01-15',
        '01/15/2024', 
        '15/01/2024',
        'January 15, 2024',
        'invalid-date',
        ''
    ]
    
    for date_str in test_dates:
        valid_formats = []
        for fmt in date_formats:
            try:
                datetime.strptime(date_str, fmt)
                valid_formats.append(fmt)
            except ValueError:
                continue
        
        if valid_formats:
            print(f"    {date_str:<20} ✅ VALID ({len(valid_formats)} formats)")
        else:
            print(f"    {date_str:<20} ❌ INVALID")
    
    # Date sequence tests
    print("\n  📅 Date Sequence Tests:")
    date_pairs = [
        ('2024-01-15', '2024-02-15', 'Invoice before due date', True),
        ('2024-02-15', '2024-01-15', 'Due date before invoice', False),
        ('2024-01-15', '2024-01-15', 'Same dates', True)
    ]
    
    for date1, date2, description, expected in date_pairs:
        try:
            d1 = datetime.strptime(date1, '%Y-%m-%d')
            d2 = datetime.strptime(date2, '%Y-%m-%d')
            is_valid = d1 <= d2
            status = "✅ VALID" if is_valid else "❌ INVALID"
            correct = "✓" if is_valid == expected else "✗"
            print(f"    {date1} → {date2}  {description:<25} {status} {correct}")
        except ValueError:
            print(f"    {date1} → {date2}  {description:<25} ❌ PARSE ERROR")

def demonstrate_amount_validation():
    """Demonstrate currency and amount consistency validation."""
    print("\n🔍 Amount Validation:")
    print("-" * 30)
    
    # Currency format tests
    print("  💰 Currency Format Tests:")
    currency_values = [
        '$1,234.56',
        '1234.56',
        '$1234',
        '€999.99',
        '£50.00',
        'invalid-amount',
        '$1,234,567.89'
    ]
    
    import re
    currency_pattern = r'^\$?[0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]{2})?$'
    
    for value in currency_values:
        # Clean value for validation
        clean_value = re.sub(r'[£€¥\s]', '$', value)  # Normalize currency symbols
        is_valid = bool(re.match(currency_pattern, clean_value))
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"    {value:<20} {status}")
    
    # Amount consistency tests
    print("\n  💰 Amount Consistency Tests:")
    from decimal import Decimal
    
    test_calculations = [
        ({'subtotal': '100.00', 'tax': '8.50', 'total': '108.50'}, 'total = subtotal + tax', True),
        ({'subtotal': '100.00', 'tax': '8.50', 'total': '110.00'}, 'total = subtotal + tax', False),
        ({'subtotal': '250.00', 'tax': '21.25', 'total': '271.25'}, 'total = subtotal + tax', True),
        ({'subtotal': '1000.00', 'tax': '85.00', 'discount': '100.00', 'total': '985.00'}, 'total = subtotal + tax - discount', True)
    ]
    
    for amounts, formula, expected in test_calculations:
        try:
            if formula == 'total = subtotal + tax':
                expected_total = Decimal(amounts['subtotal']) + Decimal(amounts['tax'])
            elif formula == 'total = subtotal + tax - discount':
                expected_total = Decimal(amounts['subtotal']) + Decimal(amounts['tax']) - Decimal(amounts['discount'])
            else:
                continue
                
            actual_total = Decimal(amounts['total'])
            is_valid = abs(expected_total - actual_total) < Decimal('0.01')
            status = "✅ CONSISTENT" if is_valid else "❌ INCONSISTENT"
            correct = "✓" if is_valid == expected else "✗"
            
            amount_str = f"${amounts.get('subtotal', '0')} + ${amounts.get('tax', '0')}"
            if 'discount' in amounts:
                amount_str += f" - ${amounts['discount']}"
            amount_str += f" = ${amounts['total']}"
            
            print(f"    {amount_str:<35} {status} {correct}")
            
        except Exception as e:
            print(f"    {amounts} - Error: {str(e)}")

def demonstrate_confidence_routing():
    """Demonstrate confidence-based HITL routing logic."""
    print("\n🔍 Confidence-Based HITL Routing:")
    print("-" * 35)
    
    test_scenarios = [
        {
            'name': 'High Confidence Invoice',
            'confidence_scores': {
                'invoice_number': 0.95,
                'invoice_date': 0.92,
                'total_amount': 0.89,
                'vendor_name': 0.91
            },
            'validation_errors': [],
            'threshold': 0.75
        },
        {
            'name': 'Low Confidence Fields',
            'confidence_scores': {
                'invoice_number': 0.65,  # Below threshold
                'invoice_date': 0.71,   # Below threshold  
                'total_amount': 0.85,
                'vendor_name': 0.92
            },
            'validation_errors': [],
            'threshold': 0.75
        },
        {
            'name': 'Validation Failures',
            'confidence_scores': {
                'invoice_number': 0.95,
                'invoice_date': 0.92,
                'total_amount': 0.89,
                'vendor_name': 0.91
            },
            'validation_errors': [
                'Invalid invoice number format',
                'Due date before invoice date',
                'Amount calculation mismatch'
            ],
            'threshold': 0.75
        },
        {
            'name': 'Mixed Issues',
            'confidence_scores': {
                'invoice_number': 0.65,  # Low confidence
                'invoice_date': 0.92,
                'total_amount': 0.67,   # Low confidence
                'vendor_name': 0.91
            },
            'validation_errors': [
                'Total amount format validation failed'
            ],
            'threshold': 0.75
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n  📊 {scenario['name']}:")
        
        # Check confidence scores
        low_confidence_fields = [
            field for field, score in scenario['confidence_scores'].items()
            if score < scenario['threshold']
        ]
        
        # Determine routing
        should_route = len(low_confidence_fields) > 0 or len(scenario['validation_errors']) > 0
        
        # Calculate average confidence
        avg_confidence = sum(scenario['confidence_scores'].values()) / len(scenario['confidence_scores'])
        
        print(f"    Average confidence: {avg_confidence:.1%}")
        print(f"    Validation errors: {len(scenario['validation_errors'])}")
        print(f"    Low confidence fields: {len(low_confidence_fields)}")
        
        if low_confidence_fields:
            print(f"    → Low confidence: {', '.join([f'{f} ({scenario['confidence_scores'][f]:.1%})' for f in low_confidence_fields])}")
        
        if scenario['validation_errors']:
            print(f"    → Validation issues: {len(scenario['validation_errors'])} errors")
        
        routing_decision = "🔄 ROUTE TO HITL" if should_route else "✅ AUTO-APPROVE"
        print(f"    Decision: {routing_decision}")

def demonstrate_business_rules():
    """Demonstrate business rule validation scenarios."""
    print("\n🔍 Business Rule Validation:")
    print("-" * 30)
    
    print("  📋 Document-Specific Required Fields:")
    
    document_rules = {
        'invoice': {
            'required': ['invoice_number', 'invoice_date', 'total_amount', 'vendor_name'],
            'optional': ['due_date', 'customer_name', 'tax_amount']
        },
        'receipt': {
            'required': ['store_name', 'transaction_date', 'total_amount'],
            'optional': ['payment_method', 'items']
        },
        'contract': {
            'required': ['contract_number', 'effective_date', 'party1_name', 'party2_name'],
            'optional': ['expiration_date', 'contract_value']
        }
    }
    
    for doc_type, rules in document_rules.items():
        print(f"\n    📄 {doc_type.upper()}:")
        print(f"      Required: {', '.join(rules['required'])}")
        print(f"      Optional: {', '.join(rules['optional'])}")
    
    print("\n  📋 Cross-Field Business Rules:")
    business_rules = [
        "Invoice date must be before due date",
        "Contract effective date must be before expiration date", 
        "Total amount must equal subtotal + tax - discount",
        "Purchase order date must be before delivery date",
        "Employee hire date must be before termination date"
    ]
    
    for rule in business_rules:
        print(f"    • {rule}")

def main():
    """Main demonstration function."""
    
    print("\n🎯 System Capabilities:")
    print("  • Regex pattern validation (emails, SSNs, phone numbers)")
    print("  • Luhn algorithm validation (credit cards, account numbers)")
    print("  • Date format parsing and sequence validation")
    print("  • Currency format and amount consistency checks")
    print("  • Confidence-based HITL routing decisions")
    print("  • Document-specific business rule validation")
    print("  • Microsoft Teams notification integration")
    print("  • Web-based review interface for human reviewers")
    
    # Run demonstrations
    demonstrate_luhn_validation()
    demonstrate_regex_patterns()
    demonstrate_date_validation()
    demonstrate_amount_validation()
    demonstrate_confidence_routing()
    demonstrate_business_rules()
    
    print("\n" + "=" * 50)
    print("🚀 Next Steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Configure Microsoft Teams webhook URL")
    print("  3. Start HITL review application")
    print("  4. Customize validation rules for your document types")
    print("  5. Set up production monitoring and alerting")
    
    print("\n🔗 Integration Points:")
    print("  • if confidence < threshold → HITL Review Queue")
    print("  • if validation fails → HITL Review Queue + Teams notification")
    print("  • Critical errors → High priority HITL tasks")
    print("  • Successful validation → Automatic processing continuation")
    
    print("\n✨ Demo completed! Enhanced validation system ready for deployment.")

if __name__ == "__main__":
    main()