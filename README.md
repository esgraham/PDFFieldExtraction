# Enhanced PDF Field Extraction System

An Azure-based PDF processing pipeline with automated extraction, human-in-the-loop review, and comprehensive analytics dashboard.

## ✨ Key Features

**🔄 Automated Processing Pipeline**
- Azure Storage integration with real-time monitoring
- Advanced OCR with Azure Document Intelligence v4
- AI-powered preprocessing (deskew, denoise, enhance)
- Smart document classification and field extraction
- Comprehensive validation with business rules

**👥 Human-in-the-Loop System**
- Interactive PDF viewer with bounding box overlays
- Field correction interface with confidence scoring
- SLA tracking and automatic reviewer assignment
- Training data collection for ML improvement

**📊 Multi-View Analytics Dashboard**
- **Executive**: KPIs, ROI metrics, business impact
- **Operational**: Real-time metrics, alerts, queue management
- **Technical**: System health, resource usage, API performance
- Interactive charts, real-time updates, alert system

## 🏗️ Architecture

```
PDFFieldExtraction/
├── src/
│   ├── core/          # Processing pipeline
│   ├── hitl/          # Human-in-the-loop system
│   └── dashboard/     # Analytics dashboard
├── data/
│   └── databases/     # SQLite databases (demo & production)
├── templates/         # Dashboard HTML templates
├── static/           # Dashboard assets (CSS, JS)
├── config/           # Configuration files
└── requirements/     # Dependency files
```

**Entry Points:**
- `main.py` - Core PDF processing pipeline
- `main_enhanced_hitl.py` - HITL web interface
- `src/dashboard/run_dashboard_demo.py` - Analytics dashboard

## 🌐 System Components

**Core Processing**: `main.py`
```bash
python main.py monitor     # Continuous monitoring
python main.py process     # Process specific file
python main.py batch       # Batch process all files
python main.py info        # Show environment info
```

**HITL Interface**: `main_enhanced_hitl.py` (Port 8000)
- Interactive PDF review and field correction
- SLA tracking and reviewer assignment

**Analytics Dashboard**: `src/dashboard/run_dashboard_demo.py` (Port 8000)
- Executive, Operational, and Technical views
- Real-time KPIs and system monitoring

## 💻 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements/requirements.txt
# For full features: pip install -r requirements/requirements_complete.txt
# For dashboard: pip install -r requirements/requirements_dashboard.txt
```

### 2. Configure Azure
```bash
# Create sample configuration
python main.py config

# Edit .env with your Azure credentials
# AZURE_STORAGE_ACCOUNT_NAME=your_account
# AZURE_STORAGE_CONTAINER_NAME=pdfs
# AZURE_STORAGE_CONNECTION_STRING=your_connection_string
```

### 3. Run System
```bash
# Core processing pipeline
python main.py monitor

# HITL review interface
python main_enhanced_hitl.py

# Analytics dashboard  
python src/dashboard/run_dashboard_demo.py
```

## 🚀 Setup

### Prerequisites
- Python 3.8+
- Azure Storage Account with blob container
- Optional: Azure Document Intelligence for advanced OCR

### Azure Setup
1. Create Azure Storage Account
2. Create a container (e.g., `pdfs`)
3. Get connection string or use Azure Identity
4. Optional: Create Document Intelligence resource

### Installation
```bash
# Clone and install
git clone <repository-url>
cd PDFFieldExtraction
pip install -r requirements/requirements.txt

# Configure environment
python main.py config  # Creates sample .env file
# Edit .env with your Azure credentials
```

### Authentication
Supports flexible authentication:
- **Connection String**: Traditional method with shared keys
- **Azure Identity**: Modern approach using managed identity
- **Automatic Fallback**: Tries connection string, falls back to identity

## 🔄 Processing Pipeline

When you run `main.py monitor`, each PDF goes through:

1. **📥 Download** - Retrieves PDF from Azure Storage
2. **🖼️ Preprocess** - Deskew, denoise, enhance for OCR
3. **🏷️ Classify** - Determines document type (invoice, receipt, etc.)
4. **📄 OCR/HWR** - Extracts text and handwritten content
5. **🔍 Extract** - Identifies structured fields (amounts, dates, names)
6. **✅ Validate** - Applies business rules and data validation
7. **📊 Summarize** - Outputs JSON results with confidence scores

## 🧪 Testing & Deployment

```bash
# Test setup
python main.py info  # Show environment info
python -m pytest tests/  # Run tests

# Docker deployment
docker build -t pdf-extraction-system .
docker run -p 8000:8000 --env-file .env pdf-extraction-system
```

## 📚 Documentation

- **[Dashboard Guide](docs/guides/DASHBOARD_GUIDE.md)** - Analytics dashboard system
- **[HITL System Guide](docs/guides/ENHANCED_HITL_SYSTEM.md)** - Human-in-the-loop documentation
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 🔧 Common Issues

```bash
# Test Azure connection
python tests/test_azure_connection.py

# Check environment
python main.py info

# Validate configuration
python main.py config
```

For detailed troubleshooting, see the [troubleshooting guide](docs/TROUBLESHOOTING.md).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Add tests for new functionality
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and development purposes.

---

**🎯 Ready to process PDFs at scale with Azure and AI!**