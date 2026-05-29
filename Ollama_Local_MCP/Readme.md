# 🦙 Ollama MCP — Local AI Assistant

A fully local AI stack: Ollama + MCP server + Streamlit UI.
No cloud APIs. No API keys. Everything runs on your machine.

---

## Project structure

```
ollama_mcp/
├── main.py              ← entry point (CLI)
├── streamlit_app.py     ← Streamlit web UI
├── requirements.txt
├── README.md
└── src/
    ├── config.py        ← all settings in one place
    ├── tools.py         ← every tool as a plain Python function
    ├── server.py        ← MCP server (wires tools.py into MCP protocol)
    ├── db.py            ← SQLite session engine
    ├── registry.py      ← custom skill / agent registry
    ├── vector.py        ← FAISS vector store (RAG)
    └── pipeline.py      ← sequential LangGraph of custom agents
```

Auto-created at runtime under `data/` (override with `LOCAL_MCP_DATA_DIR` env var):
```
data/
├── sessions.db          ← conversation history, memory, file refs
├── skills_registry.json ← custom skills and agents
├── uploads/             ← PDF and image uploads from Streamlit
└── faiss_index/         ← FAISS vector store for RAG
```

---

## Setup

### 1. Install Ollama
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com
```

### 2. Pull gemma3 (supports vision)
```bash
ollama pull gemma3
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Start the Streamlit UI (recommended)
```bash
python main.py ui
# Opens http://localhost:8501
```

### Start the MCP server (for Claude Desktop / Claude Code)
```bash
python main.py server
# Or directly: python server.py
```

### Other commands
```bash
python main.py check      # verify Ollama is running + list models
python main.py skills     # list registered custom skills
python main.py sessions   # list saved sessions
python main.py seed       # seed default skills (auto on first run)
```

---

## Tools

| Category    | Tool              | Description                                      |
|-------------|-------------------|--------------------------------------------------|
| Core        | `chat`            | Multi-turn chat with Ollama                      |
| Core        | `generate`        | Raw text completion                              |
| Core        | `list_models`     | List locally available models                    |
| Web         | `web_search`      | DuckDuckGo search — no API key                   |
| Web         | `fetch_page`      | Fetch and clean text from any URL                |
| Web         | `research`        | Search + Ollama synthesis with citations         |
| PDF         | `read_pdf`        | Extract text from a local PDF                   |
| PDF         | `pdf_qa`          | Ask a question about a PDF                      |
| Image       | `describe_image`  | Describe an image (gemma3 vision)               |
| Image       | `image_qa`        | Ask a question about an image                   |
| Image       | `ocr_image`       | Extract text from an image                      |
| Skill       | `summarize`       | Summarize text (bullets / tldr / paragraph)     |
| Skill       | `translate`       | Translate to any language                       |
| Skill       | `code_review`     | Review code for bugs and improvements           |
| Skill       | `run_skill`       | Run any registered custom skill                 |
| Session     | `session_create`  | Create a new conversation session               |
| Session     | `session_list`    | List all sessions                               |
| Session     | `session_resume`  | Resume a session with full history              |
| Session     | `session_history` | Get full message history                        |
| Session     | `session_summary` | Stats: messages, memory, files, tool calls      |
| Session     | `session_delete`  | Delete a session and all its data               |
| Session     | `session_export`  | Export a session to JSON                        |
| Memory      | `memory_save`     | Save a key-value fact to session memory         |
| Memory      | `memory_recall`   | Recall a fact from memory                       |
| Memory      | `memory_list`     | List all memory entries for a session           |

---

## Custom skills

Register a new skill from Python:
```python
from registry import register_skill

register_skill(
    name="haiku",
    description="Turn any text into a haiku.",
    system_prompt="You write haiku poems. 5-7-5 syllables. Return only the haiku.",
    user_prompt_template="Write a haiku about: {input}",
)
```

Or from the Streamlit sidebar — no code needed.

---

## Connect to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "ollama": {
      "command": "python",
      "args": ["/absolute/path/to/ollama_mcp/main.py", "server"]
    }
  }
}
```
Restart Claude Desktop. All tools appear automatically.

---

## Vision models

Image tools require a vision-capable model. Supported:
- `gemma3` (default — recommended)
- `llava`
- `llava-phi3`
- `llava-llama3`

Pull with: `ollama pull llava`
