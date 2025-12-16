# LLM Evaluation Pipeline

Complete LLM evaluation system with Docker containerization and REST API.

## Model
- **Qwen/Qwen2-0.5B** (0.5B parameters) ***(Tiny For Fast evaluation & can be changed from configuration.json)***
- Small, efficient, and capable for MCQ evaluation

## Quick Start

### 1. Build Docker Image
```bash
docker build -t llm-evaluation .
```

### 2. Run Container
```bash
docker run -p 8000:8000 llm-evaluation
```

Or use docker-compose:
```bash
docker-compose up --build
```

### 3. Test API
```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AI?", "max_new_tokens": 50}'

# Evaluate MCQ
curl -X POST http://localhost:8000/evaluate_mcq \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is 2+2?",
    "choices": {"A": "3", "B": "4", "C": "5", "D": "6"},
    "correct_answer": "B"
  }'
```

## Alternative Deployments

### Ollama (Better Performance)

For improved performance and resource utilization, use Ollama instead of FastAPI:

```bash
# Install and run
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama run llama3.1:8b
```

See `ollama_llama.ipynb` notebook for complete setup including custom model conversion.

**Benefits**: Lower latency (~1-3s vs 2-5s), better memory usage, simpler deployment

### vLLM (Production Use)

For production deployments with GPUs, vLLM offers the best performance.

**Resources**: [Complete vLLM deployment guides](https://github.com/Mohamed-Ahmed-Abo-El-Enen/LargeLanguageModels/tree/main/LLMs/Deployment)

## Running Evaluation

### 1. Prepare Dataset

```python
python - c
"
from utils.evaluation.data_loader import load_arc_dataset, save_dataset
import os

os.makedirs('n/data', exist_ok=True)
data = load_arc_dataset(num_samples=100)
save_dataset(data)
"
```

### 2. Run Evaluation

```python
python - c
"
from utils.evaluation.evaluate import MCQEvaluator
from utils.evaluation.data_loader import load_dataset_from_file
import os

os.makedirs('results', exist_ok=True)
evaluator = MCQEvaluator()
data = load_dataset_from_file()
evaluator.evaluate_dataset(data)
evaluator.save_results()
"
```

### 3. Generate Analysis

```python
python - c
"
from utils.analysis.error_analysis import ErrorAnalyzer

analyzer = ErrorAnalyzer()
report = analyzer.generate_report()
print(report)

with open('results/error_analysis_report.txt', 'w') as f:
    f.write(report)

analyzer.create_visualizations()
"
```

## API Documentation

### Endpoints

#### GET /health
Health check endpoint

#### POST /generate
Generate text completions
```json
{
  "prompt": "string",
  "max_new_tokens": 100,
  "temperature": 0.7
}
```

#### POST /evaluate_mcq
Evaluate MCQ
```json
{
  "question": "string",
  "choices": {"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"},
  "correct_answer": "A"
}
```

## Project Structure
```
pidima-llm-evaluation/
├── app/
│   ├── main.py
│   └── model_loader.py
├── notebook
    ├── ollama_llama.ipynb
    └── poc_notebook.ipynb
├── utils/
│   ├── analysis/
│       └── error_analysis.py
│   ├── data/
│       └── mcq_dataset.json
│   ├── evaluation/
│       ├── evaluate.py
│       └── data_loader.py
│   ├── resulst/
│       └── evaluation_results.json
│   ├── generate_analysis.py
│   ├── prepare_dataset.py
│   └── run_evaluation.py
├── .dockerignore
├── configuration.json
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```

## Performance Notes
- First request may be slow due to model loading
- CPU inference: ~2-5 seconds per question
- GPU inference: ~0.5-1 second per question

## Assumptions
- Using ARC-Easy dataset (100 science questions)
- Temperature set low (0.1) for deterministic answers
- Simple answer extraction (first A-D letter found)