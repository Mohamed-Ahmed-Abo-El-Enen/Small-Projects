# 📱 Telecom Egypt Intelligent Assistant

A production-ready RAG-powered chatbot with multi-lingual support (Arabic/English), vision capabilities, and conversation history management.

## 🌟 Features

- **Multi-lingual Support**: Automatic language detection (Arabic/English)
- **Vision AI**: Image analysis using local or cloud vision models
- **RAG Pipeline**: Semantic search with FAISS vector store
- **Conversation History**: SQLite-based persistent chat history
- **LangGraph Workflow**: Sophisticated orchestration with quality checks
- **Dual Deployment**: FastAPI backend + Streamlit frontend
- **Local/Cloud Models**: Support for Ollama (local) or OpenAI (cloud)

## 📁 Project Structure

```
telecom-egypt-assistant/
├── app/
│   ├── core/              # Configuration & utilities
│   ├── services/          # Business logic
│   ├── api/               # API routes
│   ├── main.py            # FastAPI app
│   └── streamlit_app.py   # Streamlit UI
├── data/                  # Generated data (git-ignored)
├── docker/                # Docker configurations
├── scripts/               # Utility scripts
├── tests/                 # Unit tests
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

1. **Start Services**
```bash
docker-compose up -d
```

2. **Initialize Models (Ollama)**
```bash
docker exec -it telecom-ollama ollama pull qwen3:latest
docker exec -it telecom-ollama ollama pull minicpm-v:latest
```

3. **Access Applications**
- FastAPI: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

### Option 2: Local Development

1. **Install Dependencies**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Install Ollama** (for local models)
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen3:latest
ollama pull minicpm-v:latest
```

3. **Initialize System**
```bash
python scripts/initialize.py
```

4. **Run Services**

Terminal 1 (API):
```bash
uvicorn app.main:app --reload --port 8000
```

Terminal 2 (Streamlit):
```bash
streamlit run app/streamlit_app.py
```

## 📚 API Documentation

### Initialize System
```bash
POST /initialize
{
  "max_pages": 50
}
```

### Load Existing Index
```bash
POST /load-index
```

### Chat (Text Only)
```bash
POST /chat
{
  "query": "What are your internet packages?",
  "conversation_id": "user_123",
  "user_id": "default"
}
```

### Chat with Image
```bash
POST /chat-with-image
Content-Type: multipart/form-data

query: "What's in this image?"
conversation_id: "user_123"
image: [file upload]
```

### Manage Conversations
```bash
# Create conversation
POST /conversations
{
  "user_id": "user_123"
}

# Get all conversations
GET /conversations?user_id=user_123

# Get conversation history
GET /conversations/{conversation_id}/history

# Delete conversation
DELETE /conversations/{conversation_id}
```

### Upload Documents
```bash
POST /documents
Content-Type: multipart/form-data

file: [pdf, docx, txt, html, image]
```

## 🐳 Docker Deployment

### Production Deployment

1. **Build Images**
```bash
docker-compose build
```

2. **Deploy**
```bash
docker-compose up -d
```

3. **Check Status**
```bash
docker-compose ps
docker-compose logs -f
```

### Individual Services

Run only API:
```bash
docker-compose up -d ollama api
```

Run only Streamlit:
```bash
docker-compose up -d ollama streamlit
```

### Scaling

```bash
docker-compose up -d --scale api=3
```

## ⚙️ Configuration

### Environment Variables

Key configurations in `.env`:

- `USE_LOCAL_MODEL`: true for Ollama, false for OpenAI
- `OLLAMA_BASE_URL`: Ollama server URL
- `OPENAI_API_KEY`: Required if USE_LOCAL_MODEL=false
- `LOCAL_MODEL_NAME`: Model for text (qwen3:latest)
- `LOCAL_VISION_MODEL_NAME`: Model for images (minicpm-v:latest)

### Model Selection

**Local Models (Ollama):**
- Text: qwen3, llama2, mistral
- Vision: minicpm-v, llava, bakllava

**Cloud Models (OpenAI):**
- Text: gpt-4, gpt-3.5-turbo
- Vision: gpt-4o, gpt-4-vision-preview

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_scraper.py

# With coverage
pytest --cov=app tests/
```

## 📊 Monitoring

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# System status
curl http://localhost:8000/status
```

### Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f streamlit
```

## 🔧 Troubleshooting

### Ollama Model Issues
```bash
# Check if models are loaded
docker exec -it telecom-ollama ollama list

# Pull models manually
docker exec -it telecom-ollama ollama pull qwen3:latest
```

### Database Issues
```bash
# Reset database
rm data/chat_history.db
python scripts/initialize.py
```

### Vector Store Issues
```bash
# Rebuild index
rm -rf data/faiss_index
curl -X POST http://localhost:8000/initialize -H "Content-Type: application/json" -d '{"max_pages": 50}'
```

## 🛠️ Development

### Adding New Features

1. Create service in `app/services/`
2. Add routes in `app/api/routes.py`
3. Update tests in `tests/`
4. Update documentation
