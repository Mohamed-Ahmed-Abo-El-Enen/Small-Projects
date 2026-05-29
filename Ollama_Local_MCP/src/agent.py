import json
import time
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent

import src.tools as T
from src.config import DEFAULT_MODEL, OLLAMA_BASE
from src.db import get_ollama_history
from src.logger import get_logger, log_call
from src.registry import get_agent, list_agents, list_skills
from src.logger import _preview as _prev
from src.tools import envelope
from src.prompts import build_router_system_prompt

log = get_logger(__name__)


@tool
def web_search(query: str) -> list:
    """Search the web via DuckDuckGo. Returns title/url/snippet results. Use when the user wants to find information online."""
    return T.web_search(query)


@tool
def fetch_page(url: str) -> str:
    """Fetch a single URL and return its cleaned plain-text content. Use when the user gives a specific URL they want read."""
    return T.fetch_page(url)


@tool
def research(query: str) -> str:
    """Search the web AND synthesize a cited answer. Use for research questions where the user wants a summarized answer with sources."""
    return T.research(query)


@tool
def read_pdf(path: str) -> str:
    """Extract the plain text of a local PDF file. Use when the user wants to see the contents of a PDF."""
    return T.read_pdf(path)


@tool
def pdf_qa(path: str, question: str) -> str:
    """Answer a specific question about a PDF file at the given local path."""
    return T.pdf_qa(path, question)


@tool
def pdf_summarize(path: str) -> str:
    """Summarize the contents of a PDF file at the given local path."""
    return T.pdf_summarize(path)


@tool
def pdf_ocr(path: str) -> list:
    """Extract every image embedded in a PDF and OCR it (Tesseract first, vision-model fallback). Use when the PDF contains images that hold text, or when the user asks for OCR."""
    return T.pdf_ocr(path)


@tool
def describe_image(path: str, prompt: str = "Describe this image in detail.") -> str:
    """Describe what's visible in an image using a vision model."""
    return T.describe_image(path, prompt=prompt)


@tool
def image_qa(path: str, question: str) -> str:
    """Answer a specific question about an image."""
    return T.image_qa(path, question)


@tool
def ocr_image(path: str) -> str:
    """Extract all visible text from an image using a vision model."""
    return T.ocr_image(path)


@tool
def run_skill(skill_name: str, input: str, context: Optional[dict] = None) -> dict:
    """Run a registered custom skill by name. Returns a standardized envelope:
    {ok, skill, output, metadata, data, error}.

    For the next step in a chain, read the `output` field of the returned dict —
    never the whole dict. Only use skill_name values that appear in the system
    prompt's 'Available custom skills' list.
    """
    return T.run_skill(skill_name, input, context=context)


@tool
def index_url(url: str) -> dict:
    """Scrape a URL and add its content to the FAISS vector store for later retrieval."""
    return T.index_url(url)


@tool
def summarize(text: str, style: str = "bullets") -> dict:
    """Summarize a block of text. in concise bullet list."""
    return T.summarize(text, style=style)


@tool
def translate(text: str, target_language: str) -> dict:
    """Translate text into the target language."""
    return T.translate(text, target_language=target_language)


@tool
def code_review(code: str, language: str = "Python") -> dict:
    """Review code for bugs / improvements."""
    return T.code_review(code, language=language)


@tool
def run_agent(agent_name: str, task: str, context: Optional[dict] = None) -> dict:
    """Invoke a registered custom agent as a sub-agent. The sub-agent has access to all basic tools (web_search, fetch_page, pdf_*, *_image, run_skill, summarize, etc.)."""
    agent_def = get_agent(agent_name)
    if not agent_def:
        names = [a["name"] for a in list_agents()]
        return envelope(
            skill=agent_name,
            error=f"Agent '{agent_name}' not found. Available: {names or ['(none registered)']}",
        )

    model = agent_def.get("model") or DEFAULT_MODEL
    log.info("run_agent start: %s task=%r", agent_name, task[:80])
    sub_llm = ChatOllama(model=model, base_url=OLLAMA_BASE)
    sub_tools = [t for t in TOOLS if t.name != "run_agent"]
    sub = create_agent(sub_llm, tools=sub_tools, system_prompt=agent_def["system_prompt"])

    t0 = time.perf_counter()
    try:
        result = sub.invoke({"messages": [HumanMessage(content=task)]})
    except Exception as exc:
        return envelope(
            skill=agent_name,
            error=f"{type(exc).__name__}: {exc}",
            metadata={
                "model": model,
                "input_chars": len(task or ""),
                "duration_s": round(time.perf_counter() - t0, 3),
            },
        )

    out_msgs = result.get("messages", []) or []
    final = out_msgs[-1].content if out_msgs else ""
    tool_hops = sum(
        1 for m in out_msgs
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
    )
    sub_trace = _extract_tool_trace(out_msgs)
    log.info(
        "run_agent done: %s chars=%d tool_hops=%d trace=%s",
        agent_name, len(final), tool_hops, sub_trace,
    )

    return envelope(
        skill=agent_name,
        output=final,
        metadata={
            "model": model,
            "input_chars": len(task or ""),
            "output_chars": len(final),
            "tool_hops": tool_hops,
            "tool_trace": sub_trace,
            "total_msgs": len(out_msgs),
            "duration_s": round(time.perf_counter() - t0, 3),
            "context_keys": list((context or {}).keys()),
        },
    )


TOOLS = [
    web_search,
    fetch_page,
    research,
    read_pdf,
    pdf_qa,
    pdf_summarize,
    pdf_ocr,
    describe_image,
    image_qa,
    ocr_image,
    summarize,
    translate,
    code_review,
    run_skill,
    index_url,
    run_agent,
]


def _system_prompt(
    file_path: Optional[Path],
    skills: list[dict],
    agents: Optional[list[dict]] = None,
) -> str:
    return build_router_system_prompt(file_path, skills, agents)


def _history_messages(session_id: Optional[str]) -> list:
    if not session_id:
        return []
    msgs = []
    for m in get_ollama_history(session_id):
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
    return msgs


def _is_unsupported_tools_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "does not support tools" in msg or "tools not supported" in msg


def _fallback_chat(
    user_input: str,
    file_path: Optional[Path],
    session_id: Optional[str],
    model: str,
) -> str:
    # Some Ollama models reject the tools param. Strip it and just chat.
    llm = ChatOllama(model=model, base_url=OLLAMA_BASE)

    parts = ["You are a helpful assistant."]
    if file_path:
        kind = "PDF" if file_path.suffix.lower() == ".pdf" else "image"
        parts.append(
            f"The user has attached a {kind} at `{file_path}`, but this model "
            f"cannot call tools so you cannot open it. Answer from the text only."
        )

    messages = [SystemMessage(content="\n\n".join(parts))]
    messages.extend(_history_messages(session_id))
    messages.append(HumanMessage(content=user_input))

    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


@log_call
def run_router(
    user_input: str,
    file_path: Optional[Path] = None,
    session_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Stage 1: ReAct router. Stage 2: pipe through the agent pipeline if any agents are registered."""
    llm = ChatOllama(model=model, base_url=OLLAMA_BASE)
    system = _system_prompt(file_path, list_skills(), list_agents())
    agent = create_agent(llm, tools=TOOLS, system_prompt=system)

    messages = _history_messages(session_id)
    messages.append(HumanMessage(content=user_input))

    log.info("agent start: input=%r file=%s history=%d", user_input[:80], file_path, len(messages) - 1)
    try:
        result = agent.invoke({"messages": messages})
    except Exception as exc:
        if _is_unsupported_tools_error(exc):
            log.warning("model %s has no tool support — falling back to plain chat", model)
            body = _fallback_chat(user_input, file_path, session_id, model)
            hint = (
                f"\n\n---\n_Note: `{model}` does not support tool calling in Ollama, "
                "so I answered without invoking any tools. "
                "For autonomous tool use, pull a tool-capable model "
                "(e.g. `ollama pull llama3.2` or `ollama pull qwen2.5`) "
                "and select it in the sidebar._"
            )
            return body + hint
        raise

    out_msgs = result["messages"]
    final = out_msgs[-1].content if out_msgs else ""
    tool_trace = _extract_tool_trace(out_msgs)

    _log_message_trace(out_msgs)

    log.info(
        "agent done: total_msgs=%d tool_calls=%d tools=%s final_chars=%d",
        len(out_msgs), len(tool_trace), tool_trace, len(final),
    )

    if not tool_trace:
        preview = (final[:300] + "...") if len(final) > 300 else final
        log.warning(
            "agent ran ZERO real tool calls. Reply is text only (%d chars). "
            "Raw preview: %r. If this contains {\"tool_call\": ...} JSON, the "
            "model emitted a fake tool call as text — switch to a tool-capable "
            "model (llama3.2, qwen2.5).",
            len(final), preview,
        )

    _unwrapped = _unwrap_envelope_if_echoed(final)
    if _unwrapped is not None:
        log.warning(
            "final message was an echoed envelope; unwrapping to .output (%d -> %d chars).",
            len(final), len(_unwrapped),
        )
        final = _unwrapped

    if _looks_like_fake_tool_call_json(final):
        final = _recover_from_fake_final(final, out_msgs)

    elif _looks_like_self_apology(final):
        recovered = None
        for m in reversed(out_msgs):
            if type(m).__name__ == "ToolMessage":
                body = str(getattr(m, "content", "") or "")
                if body:
                    recovered = body
                    break
        if recovered:
            log.warning(
                "final message was a self-apology (%d chars). "
                "Falling back to last tool result (%d chars).",
                len(final), len(recovered),
            )
            final = recovered
        else:
            log.warning(
                "final message was a self-apology and no tool result to fall back on. Suppressing it.",
            )
            final = "_No usable answer was produced. Please rephrase the request or retry._"

    pipeline_trace = _run_pipeline_stage(final)
    if pipeline_trace is not None:
        final = pipeline_trace["final"]

    footer_parts: list[str] = []
    if tool_trace:
        footer_parts.append(f"Tools used: {' -> '.join(tool_trace)}")
    if pipeline_trace and pipeline_trace["trace"]:
        footer_parts.append(f"Pipeline: {' -> '.join(pipeline_trace['trace'])}")
    if footer_parts:
        final = f"{final}\n\n---\n_" + " | ".join(footer_parts) + "_"
    return final


def _run_pipeline_stage(stage1_final: str) -> Optional[dict]:
    """Run the agent pipeline on Stage 1's output. Returns None when no agents are registered (skip Stage 2)."""
    agents = list_agents()
    if not agents:
        return None

    from src.pipeline import run_pipeline
    log.info(
        "pipeline stage start: %d agents, stage1_chars=%d",
        len(agents), len(stage1_final),
    )
    try:
        stage2 = run_pipeline(stage1_final, use_retrieval=False)
    except Exception:
        log.exception("pipeline stage failed; keeping Stage 1 output")
        return {"final": stage1_final, "trace": []}

    trace: list[str] = []
    for o in stage2.get("outputs", []) or []:
        sub = o.get("tool_trace") or []
        trace.append(
            f"{o['agent']}[{' -> '.join(sub)}]" if sub else o["agent"]
        )

    final = stage2.get("final") or stage1_final
    log.info(
        "pipeline stage done: agents_ran=%d trace=%s final_chars=%d",
        stage2.get("agent_count", 0), trace, len(final),
    )
    return {"final": final, "trace": trace}


def _log_message_trace(messages: list) -> None:
    for i, m in enumerate(messages):
        kind = type(m).__name__
        content = getattr(m, "content", "")
        if isinstance(m, AIMessage):
            tcs = getattr(m, "tool_calls", None) or []
            if tcs:
                for tc in tcs:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                    log.info("  msg[%d] AI tool_call -> %s(%s)", i, name, _prev(args, 100))
            else:
                log.info("  msg[%d] AI text (no tool_call): %s", i, _prev(content, 160))
        elif kind == "ToolMessage":
            tool_name = getattr(m, "name", "?")
            log.info("  msg[%d] tool result <- %s: %s", i, tool_name, _prev(content, 120))
        elif isinstance(m, HumanMessage):
            log.info("  msg[%d] human: %s", i, _prev(content, 120))
        elif isinstance(m, SystemMessage):
            log.info("  msg[%d] system: <system prompt, %d chars>", i, len(content))
        else:
            log.info("  msg[%d] %s: %s", i, kind, _prev(content, 120))


def _unwrap_envelope_if_echoed(text: str) -> Optional[str]:
    """If the model parroted a SkillResult envelope as its final reply, return its .output."""
    stripped = (text or "").strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return None
    try:
        obj = json.loads(stripped)
    except Exception:
        return None
    if not isinstance(obj, dict) or "output" not in obj:
        return None
    if not any(k in obj for k in ("skill", "ok", "metadata")):
        return None
    inner = obj.get("output")
    return str(inner) if inner else None


_APOLOGY_MARKERS = (
    "i made an error",
    "i apologize",
    "i should have",
    "my previous response was incorrect",
    "sorry, i made a mistake",
)


def _looks_like_self_apology(text: str) -> bool:
    if not text:
        return False
    lower = text.strip().lower()[:400]
    return any(m in lower for m in _APOLOGY_MARKERS)


def _looks_like_fake_tool_call_json(text: str) -> bool:
    """True when the final reply is actually a tool-call JSON blob the model emitted as text instead of dispatching."""
    stripped = (text or "").strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return False
    try:
        obj = json.loads(stripped)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    fake_shapes = (
        {"tool_call"},
        {"name", "parameters"},
        {"name", "arguments"},
        {"tool", "tool_input"},
        {"action", "action_input"},
    )
    return any(shape <= keys for shape in fake_shapes)


def _recover_from_fake_final(final_text: str, messages: list) -> str:
    for m in reversed(messages):
        if type(m).__name__ == "ToolMessage":
            content = str(getattr(m, "content", "") or "")
            if content:
                log.warning(
                    "final message was fake tool-call JSON; falling back to last tool result (%d chars).",
                    len(content),
                )
                return content
    log.warning("final message was fake tool-call JSON and no tool result to fall back on.")
    return (
        "_The model emitted a fake tool call as its final reply instead of producing a real answer. "
        "Raw emission:_\n\n```\n" + final_text.strip() + "\n```"
    )


def _sub_trace_from_envelope(content) -> list[str]:
    if isinstance(content, dict):
        meta = content.get("metadata") or {}
        return [str(x) for x in (meta.get("tool_trace") or [])]
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            try:
                obj = json.loads(stripped)
            except Exception:
                return []
            if isinstance(obj, dict):
                meta = obj.get("metadata") or {}
                return [str(x) for x in (meta.get("tool_trace") or [])]
    return []


def _extract_tool_trace(messages: list) -> list[str]:
    """Ordered list of tool names. run_skill expands to run_skill:<skill>, run_agent to run_agent:<name>[nested...]."""
    tm_by_id: dict = {}
    for m in messages:
        if type(m).__name__ == "ToolMessage":
            tcid = getattr(m, "tool_call_id", None)
            if tcid:
                tm_by_id[tcid] = getattr(m, "content", "")

    names: list[str] = []
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        calls = getattr(m, "tool_calls", None) or []
        for tc in calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if not name:
                continue
            if name == "run_skill" and isinstance(args, dict) and args.get("skill_name"):
                names.append(f"run_skill:{args['skill_name']}")
            elif name == "run_agent" and isinstance(args, dict) and args.get("agent_name"):
                agent_name = args["agent_name"]
                sub = _sub_trace_from_envelope(tm_by_id.get(tcid, ""))
                if sub:
                    names.append(f"run_agent:{agent_name}[{' → '.join(sub)}]")
                else:
                    names.append(f"run_agent:{agent_name}")
            else:
                names.append(name)
    return names
