# Enhanced PDF Field Extraction System

A comprehensive Azure-based PDF processing pipeline with advanced human-in-the-loop review, machine learning validation, and real-time analytics.

## ✨ Key Features

### 📥 Azure Storage Integration
- Automatic PDF file detection and monitoring
- Event-driven processing with real-time notifications
- Secure file handling with Azure Identity

### 🖼️ Advanced Preprocessing
- AI-powered Hough/Radon-based deskew correction
- Adaptive Gaussian blur denoising
- OpenCV optimization pipelines
- Image quality enhancement for optimal OCR

### 🤖 Document Classification
- Multi-modal layout + text feature extraction
- Advanced scikit-learn classification models
- Transformer-ready architecture for scaling
- Confidence-based routing and fallback

### 📄 OCR Processing
- Azure Document Intelligence v4 integration
- Handwritten text recognition capabilities
- Advanced table and form extraction
- Multi-language support and confidence scoring

### 🏷️ Enhanced Field Extraction
- Template-light forms processing engine
- Custom extraction models with bounding boxes
- Multi-confidence scoring and validation
- Synonym mapping and canonical schema

### ✅ Advanced Validation & Rules
- Comprehensive regex pattern validation
- Luhn algorithm and checksum verification
- Cross-field consistency and business rules
- Real-time validation with error reporting

### 👥 Enhanced HITL System
- **Interactive PDF Viewer**: Side-by-side display with bounding box overlays
- **Field Correction Interface**: Confidence scores and rule violation highlights
- **SLA Tracking**: Priority-based assignment with age-to-resolution metrics
- **Training Data Collection**: Automated ML improvement from reviewer feedback
- **Real-time Assignment**: Specialized reviewer routing with workload balancing

### 📊 Comprehensive Dashboard
- **Real-time Analytics**: Queue status, throughput, and performance KPIs
- **Training Insights**: Field correction patterns and model improvement data
- **Operational Metrics**: SLA compliance, reviewer performance, and bottleneck analysis
- **Deterministic Summaries**: JSON-to-Markdown conversion with PII masking
- **Interactive Interface**: Auto-refresh dashboard with drill-down capabilities

## 🏗️ Architecture

### 📁 Project Structure
```
PDFFieldExtraction/
├── 📁 src/                          # Core source code
│   ├── 📁 core/                     # Processing pipeline (Azure, OCR, validation)
│   ├── 📁 hitl/                     # Human-in-the-loop system
│   └── 📁 dashboard/                # Analytics and KPI dashboard
├── 📁 web/                          # Web interface and templates
├── 📁 docs/                         # All documentation and guides
├── 📁 tests/                        # Unit and integration tests
├── 📁 examples/                     # Usage examples
├── 📁 demos/                        # Working demonstrations
├── 📁 config/                       # Configuration files
├── 📁 requirements/                 # All dependency files
├── main_enhanced_hitl.py            # Enhanced HITL application (primary)
└── main.py                          # Original application
```

## 🌐 API Endpoints

### Enhanced HITL System
- `GET /` - Main dashboard interface
- `POST /api/create-sample-task` - Create demo task
- `GET /review/{task_id}` - Task review interface
- `POST /api/tasks/{task_id}/complete` - Complete review

### Core Processing
- `GET /health` - System health check
- `POST /process` - Process uploaded PDF
- `GET /status/{job_id}` - Processing status

## 💻 Usage Examples

### Basic PDF Monitoring
```python
from src.core.azure_pdf_listener import AzurePDFListener

listener = AzurePDFListener(
    storage_account_name="your_account",
    container_name="pdfs",
    connection_string="your_connection_string"
)

def handle_new_pdf(blob_name, blob_client):
    print(f"New PDF detected: {blob_name}")
    # Add your processing logic here

listener.set_pdf_callback(handle_new_pdf)
listener.start_polling()
```

### Enhanced HITL System
```python
from src.hitl.enhanced_hitl_clean import EnhancedHITLReviewApp

# Start enhanced HITL system
app = EnhancedHITLReviewApp()
task_id = app.create_enhanced_review_task(
    document_id="DOC-001",
    document_type="invoice", 
    extracted_fields=fields,
    validation_errors=errors
)
```

### Run Examples
```bash
# Dashboard demo
python demos/run_dashboard_demo.py

# Basic monitoring
python examples/simple_monitor.py
```

## ⚙️ Configuration


## 🚀 Getting Started

### Prerequisites
- Python 3.8+ installed
- Azure account (free tier works fine)
- Git (to clone the repository)

### Step 1: Azure Setup (Required)

#### Create Azure Storage Account
1. **Login to Azure Portal**: https://portal.azure.com
2. **Create Storage Account**:
   - Click "Create a resource" → "Storage account"
   - Choose subscription and resource group
   - Enter storage account name (must be globally unique)
   - Select region and performance tier
   - Click "Review + create"

3. **Create PDF Container**:
   - Go to your storage account → "Containers"
   - Click "+ Container"
   - Name: `pdfs` (or your preferred name)
   - Public access level: "Private"

4. **Get Connection String**:
   - Go to "Access keys" in your storage account
   - Copy "Connection string" from key1 or key2

#### Optional: Azure Document Intelligence (for advanced OCR)
1. **Create Document Intelligence Resource**:
   - Search "Document Intelligence" in Azure Portal
   - Click "Create" → Fill required fields
   - Copy endpoint URL and API key after creation

### Step 2: Local Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd PDFFieldExtraction

# 2. Install dependencies
pip install -r requirements.txt
# Or for full features: pip install -r requirements/requirements_complete.txt

# 3. Setup configuration
cp config/.env.example .env
```

### Step 3: Configure Environment

Edit the `.env` file with your Azure credentials:
```env
# Required - Azure Storage
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account_name
AZURE_STORAGE_CONTAINER_NAME=pdfs
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...

# Optional - Document Intelligence OCR
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-region.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your_api_key
```

### Step 4: Test Your Setup

```bash
# Test basic connection
python -c "from src.core.azure_pdf_listener import AzurePDFListener; print('✅ Setup successful!')"

# Or run validation
python main.py validate
```

### Step 5: Start the System

#### Option A: Enhanced HITL Web Interface (Recommended)
```bash
python main_enhanced_hitl.py
# Access at: http://localhost:8000
# Click "Create Sample Task" to see the system in action
```

#### Option B: Basic PDF Monitoring
```bash
python main.py simple
# Monitors your Azure container for new PDF files
```

#### Option C: Run Dashboard Demo (No Azure Required)
```bash
python demos/run_dashboard_demo.py
# See analytics dashboard with sample data
```

### Step 6: Upload Test PDF

1. **Upload a PDF to your Azure container**:
   - Go to Azure Portal → Your storage account → Containers → `pdfs`
   - Click "Upload" and select a PDF file

2. **Watch the system process it**:
   - If running HITL system: Check http://localhost:8000
   - If running basic monitor: Watch console output

### 🎯 Quick Test Without Azure

Want to see the system first? Run the demo:
```bash
python demos/run_dashboard_demo.py
# Opens browser with working dashboard and sample data
```

### 🎉 What's Next?

Once you have the system running:

1. **Upload PDFs**: Add PDF files to your Azure container
2. **Review Tasks**: Use the HITL interface at http://localhost:8000
3. **View Analytics**: Check the dashboard for processing metrics
4. **Customize**: Modify validation rules in `src/core/validation_engine.py`
5. **Scale**: Deploy to Azure Container Instances for production


## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Unit tests only
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# Test with coverage
python -m pytest tests/ --cov=src
```

## 🚀 Production Deployment

### Docker
```bash
# Build image
docker build -t pdf-extraction-system .

# Run container
docker run -p 8000:8000 --env-file .env pdf-extraction-system
```

### Azure Container Instances
```bash
az container create --resource-group myResourceGroup \
    --name pdf-extraction --image myregistry.azurecr.io/pdf-extraction-system:latest \
    --environment-variables AZURE_STORAGE_ACCOUNT_NAME=myaccount
```

## 📚 Documentation

- **[Complete HITL System Guide](docs/guides/ENHANCED_HITL_SYSTEM.md)** - Full system documentation
- **[Dashboard Guide](docs/guides/DASHBOARD_GUIDE.md)** - Analytics and KPI dashboard
- **[API Reference](docs/api/api_reference.md)** - Complete API documentation
- **[Configuration Guide](docs/configuration.md)** - Detailed setup instructions

## 🔧 Troubleshooting

### Common Setup Issues

#### "No module named 'azure'" Error
```bash
pip install -r requirements.txt
```

#### "Storage account not found" Error
- Verify storage account name is correct in `.env`
- Check that storage account exists in Azure Portal
- Ensure connection string is complete and valid

#### "Container not found" Error  
- Create container named `pdfs` in your storage account
- Or update `AZURE_STORAGE_CONTAINER_NAME` in `.env`

#### Can't access web interface
```bash
# Check if port 8000 is available
python main_enhanced_hitl.py
# Try different port if needed
```

### Getting Help

1. **Check logs**: Most issues show helpful error messages
2. **Validate setup**: `python main.py validate`  
3. **Test connection**: Upload a PDF to your Azure container
4. **View documentation**: [docs/guides/](docs/guides/) for detailed guides

### Need Azure Help?

- **Free Azure Account**: https://azure.microsoft.com/free/
- **Azure Storage Tutorial**: https://docs.microsoft.com/en-us/azure/storage/
- **Document Intelligence**: https://docs.microsoft.com/en-us/azure/applied-ai-services/form-recognizer/

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