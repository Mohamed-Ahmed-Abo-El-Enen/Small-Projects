import asyncio
import json

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

import src.tools as T
from src.config import DEFAULT_MODEL, MCP_SERVER_NAME
from src.logger import get_logger
from src.registry import delete_agent, list_all, register_agent, seed_defaults


log = get_logger(__name__)
from src.db import (
    add_message,
    create_session,
    delete_session,
    export_session,
    get_messages,
    get_ollama_history,
    init_db,
    list_sessions,
    memory_list,
    memory_recall,
    memory_save,
    rename_session,
    session_summary,
)

init_db()
seed_defaults()

app = Server(MCP_SERVER_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    custom_tools = [
        types.Tool(
            name=f"skill__{skill['name']}",
            description=f"[Custom Skill] {skill['description']}",
            inputSchema={
                "type": "object",
                "required": ["input"],
                "properties": {
                    "input": {"type": "string"},
                    "model": {"type": "string"},
                },
            },
        )
        for skill in list_all()["skills"]
    ]

    return [
        # ── Core ──────────────────────────────────────────────────────────────
        types.Tool(
            name="chat",
            description="Send a chat message to a local Ollama model.",
            inputSchema={
                "type": "object", "required": ["prompt"],
                "properties": {
                    "prompt":     {"type": "string"},
                    "model":      {"type": "string", "description": f"Default: {DEFAULT_MODEL}"},
                    "system":     {"type": "string"},
                    "session_id": {"type": "string", "description": "Pass a session_id to include history"},
                },
            },
        ),
        types.Tool(
            name="generate",
            description="Raw text completion from an Ollama model.",
            inputSchema={
                "type": "object", "required": ["prompt"],
                "properties": {
                    "prompt":      {"type": "string"},
                    "model":       {"type": "string"},
                    "temperature": {"type": "number", "default": 0.7},
                },
            },
        ),
        types.Tool(
            name="list_models",
            description="List all locally available Ollama models.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Web search ────────────────────────────────────────────────────────
        types.Tool(
            name="web_search",
            description="Search the web via DuckDuckGo. No API key required.",
            inputSchema={
                "type": "object", "required": ["query"],
                "properties": {
                    "query":       {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
            },
        ),
        types.Tool(
            name="fetch_page",
            description="Fetch a URL and return clean readable text.",
            inputSchema={
                "type": "object", "required": ["url"],
                "properties": {
                    "url":       {"type": "string"},
                    "max_chars": {"type": "integer", "default": 3000},
                },
            },
        ),
        types.Tool(
            name="research",
            description="Search the web then synthesize a cited answer with Ollama.",
            inputSchema={
                "type": "object", "required": ["query"],
                "properties": {
                    "query":       {"type": "string"},
                    "max_results": {"type": "integer", "default": 3},
                    "model":       {"type": "string"},
                },
            },
        ),

        # ── PDF ───────────────────────────────────────────────────────────────
        types.Tool(
            name="read_pdf",
            description="Extract text from a local PDF file. Set include_images=true to inline OCR of embedded images.",
            inputSchema={
                "type": "object", "required": ["path"],
                "properties": {
                    "path":           {"type": "string"},
                    "pages":          {"type": "string", "default": "all"},
                    "max_chars":      {"type": "integer", "default": 8000},
                    "include_images": {"type": "boolean", "default": False},
                },
            },
        ),
        types.Tool(
            name="pdf_ocr",
            description="Extract every image embedded in the PDF and OCR it. Tesseract first; falls back to vision model if Tesseract returns nothing.",
            inputSchema={
                "type": "object", "required": ["path"],
                "properties": {
                    "path":         {"type": "string"},
                    "pages":        {"type": "string", "default": "all"},
                    "vision_model": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="pdf_qa",
            description="Read a PDF and answer a question about its content.",
            inputSchema={
                "type": "object", "required": ["path", "question"],
                "properties": {
                    "path":           {"type": "string"},
                    "question":       {"type": "string"},
                    "model":          {"type": "string"},
                    "include_images": {"type": "boolean", "default": False},
                },
            },
        ),
        types.Tool(
            name="pdf_summarize",
            description="Summarize a PDF file.",
            inputSchema={
                "type": "object", "required": ["path"],
                "properties": {
                    "path":           {"type": "string"},
                    "model":          {"type": "string"},
                    "include_images": {"type": "boolean", "default": False},
                },
            },
        ),

        # ── Image ─────────────────────────────────────────────────────────────
        types.Tool(
            name="describe_image",
            description="Describe an image using a vision model (configured via VISION_MODEL / OLLAMA_VISION_MODEL).",
            inputSchema={
                "type": "object", "required": ["path"],
                "properties": {
                    "path":   {"type": "string"},
                    "prompt": {"type": "string", "default": "Describe this image in detail."},
                    "model":  {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="image_qa",
            description="Ask a specific question about an image.",
            inputSchema={
                "type": "object", "required": ["path", "question"],
                "properties": {
                    "path":     {"type": "string"},
                    "question": {"type": "string"},
                    "model":    {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="ocr_image",
            description="Extract all visible text from an image (OCR via vision model).",
            inputSchema={
                "type": "object", "required": ["path"],
                "properties": {
                    "path":  {"type": "string"},
                    "model": {"type": "string"},
                },
            },
        ),

        # ── Skills ────────────────────────────────────────────────────────────
        types.Tool(
            name="summarize",
            description="Summarize text. style: bullets | tldr | paragraph",
            inputSchema={
                "type": "object", "required": ["text"],
                "properties": {
                    "text":  {"type": "string"},
                    "style": {"type": "string", "enum": ["bullets", "tldr", "paragraph"]},
                    "model": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="translate",
            description="Translate text into any language.",
            inputSchema={
                "type": "object", "required": ["text", "target_language"],
                "properties": {
                    "text":            {"type": "string"},
                    "target_language": {"type": "string"},
                    "model":           {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="code_review",
            description="Review code for bugs and improvements.",
            inputSchema={
                "type": "object", "required": ["code"],
                "properties": {
                    "code":     {"type": "string"},
                    "language": {"type": "string", "default": "Python"},
                    "model":    {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="run_skill",
            description="Run any custom skill registered in the registry by name.",
            inputSchema={
                "type": "object", "required": ["skill_name", "input"],
                "properties": {
                    "skill_name": {"type": "string"},
                    "input":      {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="list_custom_skills",
            description="List all registered custom skills and agents.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Sessions ──────────────────────────────────────────────────────────
        types.Tool(
            name="session_create",
            description="Create a new conversation session.",
            inputSchema={
                "type": "object", "required": ["name"],
                "properties": {
                    "name":  {"type": "string"},
                    "model": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="session_list",
            description="List all saved sessions.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="session_resume",
            description="Resume a session: load its history and send a new message.",
            inputSchema={
                "type": "object", "required": ["session_id", "message"],
                "properties": {
                    "session_id": {"type": "string"},
                    "message":    {"type": "string"},
                    "model":      {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="session_history",
            description="Return the full message history of a session.",
            inputSchema={
                "type": "object", "required": ["session_id"],
                "properties": {"session_id": {"type": "string"}},
            },
        ),
        types.Tool(
            name="session_summary",
            description="Return a stats summary of a session.",
            inputSchema={
                "type": "object", "required": ["session_id"],
                "properties": {"session_id": {"type": "string"}},
            },
        ),
        types.Tool(
            name="session_rename",
            description="Rename a session.",
            inputSchema={
                "type": "object", "required": ["session_id", "new_name"],
                "properties": {
                    "session_id": {"type": "string"},
                    "new_name":   {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="session_delete",
            description="Permanently delete a session and all its data.",
            inputSchema={
                "type": "object", "required": ["session_id"],
                "properties": {"session_id": {"type": "string"}},
            },
        ),
        types.Tool(
            name="session_export",
            description="Export a full session to JSON.",
            inputSchema={
                "type": "object", "required": ["session_id"],
                "properties": {"session_id": {"type": "string"}},
            },
        ),

        # ── Session memory ─────────────────────────────────────────────────────
        types.Tool(
            name="memory_save",
            description="Save a key-value fact to the active session's memory.",
            inputSchema={
                "type": "object", "required": ["session_id", "key", "value"],
                "properties": {
                    "session_id": {"type": "string"},
                    "key":        {"type": "string"},
                    "value":      {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="memory_recall",
            description="Recall a value from session memory by key.",
            inputSchema={
                "type": "object", "required": ["session_id", "key"],
                "properties": {
                    "session_id": {"type": "string"},
                    "key":        {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="memory_list",
            description="List all memory entries for a session.",
            inputSchema={
                "type": "object", "required": ["session_id"],
                "properties": {"session_id": {"type": "string"}},
            },
        ),

        # ── RAG / FAISS ───────────────────────────────────────────────────────
        types.Tool(
            name="index_pdf",
            description="Chunk a PDF and add its chunks to the FAISS vector store.",
            inputSchema={
                "type": "object", "required": ["path"],
                "properties": {
                    "path":         {"type": "string"},
                    "source_label": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="index_text",
            description="Chunk arbitrary text and add it to the FAISS store.",
            inputSchema={
                "type": "object", "required": ["text", "source_label"],
                "properties": {
                    "text":         {"type": "string"},
                    "source_label": {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="index_url",
            description="Scrape a URL and add its chunked content to the FAISS vector store. Preserves paragraph/heading structure.",
            inputSchema={
                "type": "object", "required": ["url"],
                "properties": {"url": {"type": "string"}},
            },
        ),
        types.Tool(
            name="scrape_url",
            description="Scrape a URL and return {url, title, text}. Does NOT index — use index_url for that.",
            inputSchema={
                "type": "object", "required": ["url"],
                "properties": {"url": {"type": "string"}},
            },
        ),
        types.Tool(
            name="retrieve",
            description="Similarity-search the FAISS store. Returns top-k chunks with source metadata.",
            inputSchema={
                "type": "object", "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "k":     {"type": "integer", "default": 4},
                },
            },
        ),
        types.Tool(
            name="clear_index",
            description="Delete the on-disk FAISS index.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="index_stats",
            description="Return stats about the current FAISS index.",
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── Pipeline / agents ─────────────────────────────────────────────────
        types.Tool(
            name="agent_register",
            description="Append a custom agent to the sequential LangGraph pipeline.",
            inputSchema={
                "type": "object",
                "required": ["name", "description", "system_prompt"],
                "properties": {
                    "name":          {"type": "string"},
                    "description":   {"type": "string"},
                    "system_prompt": {"type": "string"},
                    "tools":         {"type": "array", "items": {"type": "string"}},
                    "model":         {"type": "string"},
                },
            },
        ),
        types.Tool(
            name="agent_delete",
            description="Remove a custom agent from the pipeline by name.",
            inputSchema={
                "type": "object", "required": ["name"],
                "properties": {"name": {"type": "string"}},
            },
        ),
        types.Tool(
            name="list_pipeline",
            description="List custom agents in pipeline execution order.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="run_pipeline",
            description="Run the sequential LangGraph of all registered agents against an input. Optionally prepends FAISS retrieval context.",
            inputSchema={
                "type": "object", "required": ["input"],
                "properties": {
                    "input":         {"type": "string"},
                    "use_retrieval": {"type": "boolean", "default": True},
                },
            },
        ),

        # ── Dynamic custom skill tools ────────────────────────────────────────
        *custom_tools,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TOOL DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
@app.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    args = arguments or {}
    log.info("mcp tool dispatch: %s args_keys=%s", name, sorted(args.keys()))
    try:
        result = _dispatch(name, args)
    except Exception:
        log.exception("mcp tool failed: %s", name)
        raise

    if not isinstance(result, str):
        result = json.dumps(result, default=str, indent=2)

    return [types.TextContent(type="text", text=result)]


def _dispatch(name: str, args: dict):
    if name.startswith("skill__"):
        return T.run_skill(name[len("skill__"):], args["input"])

    if name == "chat":
        history = get_ollama_history(args["session_id"]) if args.get("session_id") else None
        return T.chat(
            prompt=args["prompt"],
            system=args.get("system", ""),
            model=args.get("model", DEFAULT_MODEL),
            history=history,
        )

    if name == "session_resume":
        sid = args["session_id"]
        msg = args["message"]
        history = get_ollama_history(sid)
        reply = T.chat(prompt=msg, model=args.get("model", DEFAULT_MODEL), history=history)
        add_message(sid, "user", msg)
        add_message(sid, "assistant", reply)
        return reply

    if name == "memory_save":
        memory_save(args["session_id"], args["key"], args["value"])
        return "ok"

    if name == "memory_recall":
        return memory_recall(args["session_id"], args["key"]) or "(not found)"

    simple = {
        "generate":           lambda: T.generate(args["prompt"], args.get("model", DEFAULT_MODEL), args.get("temperature", 0.7)),
        "list_models":        lambda: T.list_models(),
        "web_search":         lambda: T.web_search(args["query"], args.get("max_results", 5)),
        "fetch_page":         lambda: T.fetch_page(args["url"], args.get("max_chars", 3000)),
        "research":           lambda: T.research(args["query"], args.get("model", DEFAULT_MODEL), args.get("max_results", 3)),
        "read_pdf":           lambda: T.read_pdf(args["path"], args.get("pages", "all"), args.get("max_chars", 8000), args.get("include_images", False)),
        "pdf_ocr":            lambda: T.pdf_ocr(args["path"], args.get("pages", "all"), args.get("vision_model", DEFAULT_MODEL)),
        "pdf_qa":             lambda: T.pdf_qa(args["path"], args["question"], args.get("model", DEFAULT_MODEL), args.get("include_images", False)),
        "pdf_summarize":      lambda: T.pdf_summarize(args["path"], args.get("model", DEFAULT_MODEL), args.get("include_images", False)),
        "describe_image":     lambda: T.describe_image(args["path"], args.get("prompt", "Describe this image in detail."), args.get("model", DEFAULT_MODEL)),
        "image_qa":           lambda: T.image_qa(args["path"], args["question"], args.get("model", DEFAULT_MODEL)),
        "ocr_image":          lambda: T.ocr_image(args["path"], args.get("model", DEFAULT_MODEL)),
        "summarize":          lambda: T.summarize(args["text"], args.get("style", "bullets"), args.get("model", DEFAULT_MODEL)),
        "translate":          lambda: T.translate(args["text"], args["target_language"], args.get("model", DEFAULT_MODEL)),
        "code_review":        lambda: T.code_review(args["code"], args.get("language", "Python"), args.get("model", DEFAULT_MODEL)),
        "run_skill":          lambda: T.run_skill(args["skill_name"], args["input"]),
        "list_custom_skills": lambda: T.list_custom_skills(),
        "session_create":     lambda: create_session(args["name"], args.get("model", DEFAULT_MODEL)),
        "session_list":       lambda: list_sessions(),
        "session_history":    lambda: get_messages(args["session_id"]),
        "session_summary":    lambda: session_summary(args["session_id"]),
        "session_rename":     lambda: rename_session(args["session_id"], args["new_name"]),
        "session_delete":     lambda: delete_session(args["session_id"]),
        "session_export":     lambda: export_session(args["session_id"]),
        "memory_list":        lambda: memory_list(args["session_id"]),

        # RAG
        "index_pdf":          lambda: T.index_pdf(args["path"], args.get("source_label")),
        "index_text":         lambda: T.index_text(args["text"], args["source_label"]),
        "index_url":          lambda: T.index_url(args["url"]),
        "scrape_url":         lambda: T.scrape_url(args["url"]),
        "retrieve":           lambda: T.retrieve(args["query"], args.get("k", 4)),
        "clear_index":        lambda: T.clear_index(),
        "index_stats":        lambda: T.index_stats(),

        # Pipeline / agents
        "agent_register":     lambda: register_agent(
            name=args["name"],
            description=args["description"],
            system_prompt=args["system_prompt"],
            tools=args.get("tools", []),
            model=args.get("model", DEFAULT_MODEL),
        ),
        "agent_delete":       lambda: {"deleted": delete_agent(args["name"])},
        "list_pipeline":      lambda: T.list_pipeline(),
        "run_pipeline":       lambda: T.run_pipeline(args["input"], args.get("use_retrieval", True)),
    }

    if name in simple:
        return simple[name]()

    raise ValueError(f"Unknown tool: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run():
    """Start the MCP server over stdio."""
    async def _main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(_main())
