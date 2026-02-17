# CapsuleRAG README

**Enterprise-Grade Retrieval-Augmented Generation System**

CapsuleRAG is an advanced document intelligence platform that transforms enterprise document management through intelligent retrieval, security governance, and modular design. The system processes complex documents into searchable "capsules" with comprehensive metadata tracking and access controls.

##  Key Features

- **Hybrid Search Architecture**: Combines BM25 lexical search, semantic embeddings, and cross-encoder reranking
- **Modular Design**: 7 focused modules for search, security, ingestion, health monitoring, relationships, and enrichment
- **Enterprise Security**: Adversarial detection, ACL-based permissions, API key authentication, and rate limiting
- **Model Context Protocol (MCP) Integration**: Exposes capabilities as standardized MCP tools for AI agent interoperability
- **Advanced Document Processing**: Structure-aware chunking with byte-range citations and entity extraction
- **Real-time Analytics**: Health monitoring, caching performance, and relationship analysis

##  Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB+ RAM (for embedding models)
- Internet connection (for initial model downloads)

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sungodemk/CAPSULRAG.git
cd CAPSULRAG
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Packages:**
- `fastapi==0.111.0` - Modern web framework for building APIs
- `uvicorn[standard]==0.30.1` - ASGI server for running FastAPI
- `pdfplumber==0.11.4` - PDF text extraction and processing
- `numpy==1.26.4` - Numerical computing for embeddings
- `sentence-transformers==3.0.1` - Pre-trained embedding models
- `rank-bm25==0.2.2` - BM25 lexical search implementation
- `hnswlib==0.8.0` - High-performance vector similarity search
- `mcp` - Model Context Protocol for AI agent integration
- `pdf2image` - PDF to image conversion for previews
- `Pillow` - Image processing library

### 3. Environment Configuration (Optional)
Create a `.env` file for custom settings:
```bash
# Performance Settings
FAST_MODE=false                    # Use smaller models for faster startup
PRELOAD_MODELS=true               # Pre-load models at startup

# Security Settings
CAPSULE_DEMO_MODE=true            # Enable demo mode with relaxed security
ADMIN_API_KEY=your-admin-key      # Admin API key
USER_API_KEY=your-user-key        # User API key
MAX_UPLOAD_BYTES=52428800         # Max upload size (50MB)

# Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Embedding model name
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## 🚀 Running the Application

### Standard Web Application
```bash
python main.py
```
The application will start on `http://localhost:8000`

### MCP Server Mode (for AI Agents)
```bash
python mcp_server.py
```

### Simple MCP Server
```bash
python mcp_server_simple.py
```

### Live Demo Script
```bash
python live_demo.py
```

## 📁 Project Structure

```
CAPSULRAG/
├── main.py                 # Main FastAPI application
├── server.py              # Legacy server (now points to main.py)
├── mcp_server.py          # Full MCP server implementation
├── mcp_server_simple.py   # Simplified MCP server
├── live_demo.py           # Interactive demo script
├── requirements.txt       # Python dependencies
├── modules/               # Core system modules
│   ├── search.py         # Hybrid search engine
│   ├── security.py       # Security and authentication
│   ├── ingest.py         # Document ingestion
│   ├── health.py         # System health monitoring
│   ├── relationships.py  # Document relationship analysis
│   ├── enrichment.py     # Content enrichment
│   └── utils.py          # Utility functions
├── frontend/             # Web interface
│   └── index.html        # Modern web UI
├── test_docs/           # Sample documents
└── test_folder/         # Additional test data
```

## 🔧 API Endpoints

### Core Operations
- `POST /ingest` - Upload and process documents
- `POST /search` - Hybrid search across documents
- `GET /status` - System status and statistics
- `GET /health` - Capsule health monitoring

### Document Management
- `GET /summarize/{doc_id}` - Generate bullet-point summaries
- `GET /preview/{doc_id}` - Document preview images
- `GET /download/{doc_id}` - Download original files
- `GET /relationships/{doc_id}` - Related documents

### Analytics & Monitoring
- `GET /analytics` - Performance analytics
- `GET /security` - Security monitoring
- `GET /network` - Entity relationship network
- `GET /enrichment` - Content enrichment overview

##  Testing

### Quick Test with Sample Documents
1. Start the server: `python main.py`
2. Open `http://localhost:8000` in your browser
3. Upload documents from the `test_docs/` folder
4. Try searching for "pump maintenance" or "emergency procedures"

### MCP Integration Test
```bash
python test_mcp_client.py
```

### Full System Test
```bash
python test_mcp_full.py
```

##  Security Features

- **Adversarial Query Detection**: Prevents malicious input patterns
- **Rate Limiting**: Protects against abuse
- **Access Control Lists (ACL)**: Document-level permissions
- **API Key Authentication**: Secure endpoint access
- **Content Sanitization**: Safe document processing
- **Security Headers**: OWASP-compliant HTTP headers

##  Performance Optimization

- **Intelligent Caching**: Query result caching with TTL
- **Model Pre-loading**: Background model initialization
- **Capsule Routing**: Smart document selection
- **Health Monitoring**: Automatic performance tracking
- **Deduplication**: Prevents redundant document storage

##  Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit with descriptive messages
5. Push and create a pull request


##  Troubleshooting

### Common Issues

**Models not loading:**
```bash
# Set environment variable for CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

**Memory issues:**
```bash
# Enable fast mode for smaller models
export FAST_MODE=true
```

**Port already in use:**
```bash
# Change port in main.py or use environment variable
export PORT=8001
```

**Permission errors on uploads:**
```bash
# Check file permissions and disk space
chmod 755 uploads/
```

##  Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the API documentation at `http://localhost:8000/docs`
3. Check the live demo script for usage examples
4. Open an issue on GitHub




