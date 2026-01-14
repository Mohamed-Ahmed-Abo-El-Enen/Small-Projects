# 🐳 Docker Files & Run Commands

---

## Commands to Run

### 1. Build Images
```bash
docker-compose build
```

### 2. Start All Services in Background
```bash
docker-compose up -d
```

### 3. Wait for Ollama to Start (30 seconds)
```bash
sleep 30
```

### 4. Pull Ollama Models
```bash
# Pull text model
docker exec -it telecom-ollama ollama pull qwen3:latest

# Pull vision model
docker exec -it telecom-ollama ollama pull minicpm-v:latest
```

### 5. Verify Models
```bash
docker exec -it telecom-ollama ollama list
```

### 6. Initialize System
```bash
curl -X POST http://localhost:8000/initialize \
  -H "Content-Type: application/json" \
  -d '{"max_pages": 50}'
```

### 7. Access Services
```
API: http://localhost:8000
API Docs: http://localhost:8000/docs
Streamlit: http://localhost:8501
```

---

## Management Commands

### View Running Containers
```bash
docker-compose ps
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f streamlit
docker-compose logs -f ollama
```

### Stop Services
```bash
docker-compose down
```

### Restart Services
```bash
docker-compose restart
```

### Stop and Remove Everything
```bash
docker-compose down -v
```

---

### Complete Local Ollama Deployment Script

Create `deploy_local_ollama.sh`:

```bash
#!/bin/bash

echo "🔧 Telecom Egypt Assistant - Local Ollama Setup"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Installing..."
    sudo apt install -y pciutils lshw
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✅ Ollama already installed"
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434 > /dev/null; then
    echo "🚀 Starting Ollama server in background..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 5
else
    echo "✅ Ollama server already running"
fi

# Pull models if not present
echo "📥 Checking/pulling Ollama models..."
if ! ollama list | grep -q "qwen3:latest"; then
    echo "Pulling qwen3:latest..."
    ollama pull qwen3:latest
fi

if ! ollama list | grep -q "minicpm-v:latest"; then
    echo "Pulling minicpm-v:latest..."
    ollama pull minicpm-v:latest
fi

echo "✅ Models ready:"
ollama list

# Build and start Docker services
echo "🐳 Building Docker images..."
docker-compose build

echo "🚀 Starting services in background..."
docker-compose up -d

# Wait for API
echo "⏳ Waiting for API to be ready..."
sleep 15

# Initialize system
echo "🔧 Initializing system..."
curl -X POST http://localhost:8000/initialize \
  -H "Content-Type: application/json" \
  -d '{"max_pages": 50}'

echo ""
echo "✅ Deployment Complete!"
echo ""
echo "Ollama: http://localhost:11434 (running on host)"
echo "API: http://localhost:8000"
echo "Streamlit: http://localhost:8501"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"
```

**Run it:**
```bash
chmod +x deploy_local_ollama.sh
./deploy_local_ollama.sh
```

---

## Quick Reference

**Start everything:**
```bash
docker-compose up -d
```

**Stop everything:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs -f
```

**Restart a service:**
```bash
docker-compose restart api
```

**Check status:**
```bash
docker-compose ps
```