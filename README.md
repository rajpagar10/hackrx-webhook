# LLM Document Processing System - Complete Setup Guide

## 🚀 Project Overview

This is a complete, production-ready LLM Document Processing System that processes natural language queries against unstructured documents (insurance policies, contracts, emails) and returns structured JSON responses with decisions and justifications.

## 📁 Project Structure

```
llm_document_processing/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.yaml                    # Main configuration
├── .env                          # Environment variables (create this)
├── .gitignore                    # Git ignore file
│
├── data/                         # All data files
│   ├── raw_documents/           # Place your PDF/DOCX/EML files here
│   ├── processed/               # Generated processed files
│   └── vector_db/               # ChromaDB storage
│
├── src/                         # Main source code
│   ├── phase1_document_processing.py
│   ├── phase2_semantic_search.py
│   ├── phase3_query_processing.py
│   ├── phase4_llm_analysis.py
│   └── main_pipeline.py         # Main orchestrator
│
├── notebooks/                   # Your existing notebooks
│   ├── Phase1.ipynb
│   ├── Phase2.ipynb
│   └── Phase3.ipynb
│
├── api/                         # FastAPI REST API
│   ├── main.py
│   ├── routes.py
│   └── models.py
│
└── tests/                       # Test files
    ├── test_integration.py
    └── sample_queries.json
```

## 🛠 Installation & Setup

### Step 1: Environment Setup

```bash
# Clone or create the project directory
mkdir llm_document_processing
cd llm_document_processing

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configuration

1. **Copy the provided files into your project directory:**
   - `requirements.txt`
   - `config.yaml`
   - All Python files in `src/` directory

2. **Create `.env` file with your API keys:**
```bash
# Copy the .env template and add your keys
OPENAI_API_KEY=your_openai_api_key_here
ENVIRONMENT=development
DEBUG=True
```

3. **Create directory structure:**
```bash
mkdir -p data/raw_documents data/processed data/vector_db
mkdir -p src api tests notebooks
```

### Step 3: Document Preparation

1. **Place your documents** in `data/raw_documents/`:
   - Insurance policies (PDF)
   - Contracts (PDF, DOCX)
   - Email files (EML)
   - Text files (TXT)

2. **Example document structure:**
```
data/raw_documents/
├── health_policy_2024.pdf
├── travel_insurance.pdf
├── claims_procedure.docx
└── policy_terms.pdf
```

## 🏃 Quick Start

### Option 1: Run Complete Pipeline

```bash
# From project root directory
python src/main_pipeline.py
```

This will:
1. ✅ Process all documents in `data/raw_documents/`
2. ✅ Generate embeddings and build search index  
3. ✅ Run system tests
4. ✅ Start interactive query mode

### Option 2: Step-by-Step Execution

```bash
# Step 1: Process documents
python src/phase1_document_processing.py

# Step 2: Build search index  
python src/phase2_semantic_search.py

# Step 3: Test query processing
python src/phase3_query_processing.py

# Step 4: Test LLM analysis (requires OpenAI API key)
python src/phase4_llm_analysis.py
```

### Option 3: Use Your Existing Notebooks

Your current notebooks (`Phase1.ipynb`, `Phase2.ipynb`, `Phase3.ipynb`) will work with this structure. Just update the file paths in them:

```python
# Update paths in your notebooks
config_path = "../config.yaml"
raw_documents_path = "../data/raw_documents/"
processed_path = "../data/processed/"
```

## 📝 Usage Examples

### Basic Query Processing

```python
from src.main_pipeline import LLMDocumentProcessor

# Initialize system
processor = LLMDocumentProcessor()

# Setup (only needed once)
processor.setup_system()

# Process a query
response = processor.process_query(
    "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
)

print(f"Decision: {response.decision}")
print(f"Amount: {response.amount}")  
print(f"Justification: {response.justification}")
```

### Batch Processing

```python
queries = [
    "What is the grace period for premium payment?",
    "Is cataract surgery covered under the policy?",
    "How to file a cashless claim?",
    "What are waiting periods for pre-existing diseases?"
]

responses = processor.process_batch_queries(queries)

for response in responses:
    print(f"Q: {response.query}")
    print(f"A: {response.decision} - {response.justification[:100]}...")
```

## 🔧 Configuration

### Key Configuration Options (config.yaml)

```yaml
# Document processing
document_processing:
  chunk_size: 200          # Words per chunk
  chunk_overlap: 50        # Overlapping words

# Embeddings  
embeddings:
  model_name: "BAAI/bge-base-en-v1.5"
  batch_size: 32

# LLM settings
llm:
  provider: "openai"
  model: "gpt-4-turbo-preview" 
  temperature: 0.1
  max_tokens: 2000

# Search settings
semantic_search:
  top_k: 5                 # Number of results to retrieve
  similarity_threshold: 0.7
```

## 🌐 API Server (Optional)

Start the REST API server:

```bash
cd api/
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints:

```bash
# Process single query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the grace period for premium payment?"}'

# System health check
curl "http://localhost:8000/health"

# System statistics
curl "http://localhost:8000/stats"
```

## 🧪 Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Test Individual Components
```bash
# Test document processing
python src/phase1_document_processing.py

# Test semantic search
python src/phase2_semantic_search.py

# Test query processing  
python src/phase3_query_processing.py

# Test LLM analysis (requires API key)
python src/phase4_llm_analysis.py
```

## 📊 Expected Output Format

### Sample Query Response

```json
{
  "query": "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
  "decision": "rejected", 
  "amount": null,
  "justification": "Knee surgery requires 24-month waiting period. Policy is only 3 months old, therefore not eligible for coverage under current terms.",
  "relevant_clauses": [
    {
      "clause_text": "Knee surgery and joint replacement procedures are covered after completion of 24 months waiting period...",
      "source_document": "health_policy.pdf",
      "relevance": "Defines waiting period for orthopedic procedures"
    }
  ],
  "query_analysis": {
    "category": "Coverage",
    "entities": {
      "age": 46,
      "gender": "male", 
      "procedure": "knee surgery",
      "location": "Pune",
      "policy_duration": "3 months"
    }
  },
  "confidence": 0.92,
  "processing_time": 2.34,
  "timestamp": "2024-08-06T22:30:45"
}
```

## 🔍 Troubleshooting

### Common Issues

1. **"Config file not found"**
   - Ensure `config.yaml` is in the project root
   - Check file permissions

2. **"OpenAI API key not set"**
   - Add your API key to `.env` file
   - Set environment variable: `export OPENAI_API_KEY=your_key`

3. **"No documents found"**  
   - Place PDF/DOCX files in `data/raw_documents/`
   - Check file permissions and formats

4. **"ChromaDB collection not found"**
   - Run Phase 2 to build the search index
   - Check `data/vector_db/` directory exists

5. **Memory issues with large documents**
   - Reduce `chunk_size` in config.yaml
   - Process documents in smaller batches

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

Or set in `.env`:
```bash
LOG_LEVEL=DEBUG
```

## 📈 Performance Tips

1. **For large document collections:**
   - Use smaller chunk sizes (150-200 words)
   - Enable embedding caching
   - Consider using GPU for embeddings

2. **For faster queries:**
   - Reduce `top_k` in semantic search
   - Use lighter LLM models for development
   - Implement response caching

3. **For production deployment:**
   - Use async processing for batch queries
   - Implement proper error handling and retries
   - Monitor API rate limits

## 🔐 Security Considerations

1. **Never commit API keys to version control**
   - Use `.env` files (add to `.gitignore`)
   - Use environment variables in production

2. **Secure document storage**
   - Encrypt sensitive documents at rest
   - Implement access controls

3. **API security**
   - Add authentication/authorization
   - Implement rate limiting
   - Validate all inputs

## 🚀 Deployment

### Docker Deployment (Optional)

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "src/main_pipeline.py"]
```

```bash
# Build and run
docker build -t llm-doc-processor .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key llm-doc-processor
```

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the logs in `logs/app.log`
3. Test individual components separately
4. Verify all dependencies are installed correctly

## 🎯 Next Steps

1. **Add more document types** (Excel, PowerPoint, etc.)
2. **Implement user authentication** for API
3. **Add support for multiple languages**
4. **Create web interface** for easier interaction
5. **Add document versioning** and change tracking
6. **Implement advanced analytics** and reporting

---

**Congratulations! 🎉 Your LLM Document Processing System is ready to use!**

This system provides you with a complete, production-ready solution for processing insurance queries against policy documents. The modular design allows you to extend and customize each component as needed.