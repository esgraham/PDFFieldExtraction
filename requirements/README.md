# Enhanced PDF Field Extraction System - Requirements

This directory contains all dependency files for the project.

## Files:

### Core Requirements
- `requirements.txt` - Original core dependencies
- `requirements_complete.txt` - Complete dependency list with all optional features
- `requirements_dashboard.txt` - Dashboard-specific dependencies

### Version Constraints
- `=1.3.3` - Version constraint file (if applicable)

## Installation:

### Minimal Installation
```bash
pip install -r requirements/requirements.txt
```

### Complete Installation (Recommended)
```bash
pip install -r requirements/requirements_complete.txt
```

### Dashboard Only
```bash
pip install -r requirements/requirements_dashboard.txt
```

## Development Installation
```bash
# Install in development mode with all features
pip install -r requirements/requirements_complete.txt
pip install -e .
```