import json
from typing import Optional

from src.config import REGISTRY_PATH


def _load() -> dict:
    if REGISTRY_PATH.exists():
        data = json.loads(REGISTRY_PATH.read_text())
        data.setdefault("skills", [])
        data.setdefault("agents", [])
        return data
    return {"skills": [], "agents": []}


def _save(reg: dict):
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


# ── Skills ─────────────────────────────────────────────────────────────────────
def register_skill(
    name: str,
    description: str,
    system_prompt: str,
    user_prompt_template: str,
    model: Optional[str] = None,
) -> dict:
    reg = _load()
    reg["skills"] = [s for s in reg["skills"] if s["name"] != name]

    skill = {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
    }
    if model:
        skill["model"] = model

    reg["skills"].append(skill)
    _save(reg)

    return skill


def get_skill(name: str) -> Optional[dict]:
    reg = _load()
    return next((s for s in reg["skills"] if s["name"] == name), None)


def list_skills() -> list[dict]:
    return _load()["skills"]


def delete_skill(name: str) -> bool:
    reg = _load()
    before = len(reg["skills"])
    reg["skills"] = [s for s in reg["skills"] if s["name"] != name]
    _save(reg)
    return len(reg["skills"]) < before


# ── Agents ─────────────────────────────────────────────────────────────────────
def register_agent(
    name: str,
    description: str,
    system_prompt: str,
    tools: list[str],
    model: Optional[str] = None,
    prev_nodes: Optional[list[str]] = None,
    next_nodes: Optional[list[str]] = None,
) -> dict:
    # prev_nodes / next_nodes are edge hints for the sidebar network viz; they don't affect execution.
    reg = _load()
    reg["agents"] = [a for a in reg["agents"] if a["name"] != name]

    agent = {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "tools": tools,
    }
    if model:
        agent["model"] = model
    if prev_nodes:
        agent["prev_nodes"] = list(prev_nodes)
    if next_nodes:
        agent["next_nodes"] = list(next_nodes)

    reg["agents"].append(agent)
    _save(reg)

    return agent


def get_agent(name: str) -> Optional[dict]:
    reg = _load()
    return next((a for a in reg["agents"] if a["name"] == name), None)


def list_agents() -> list[dict]:
    return _load()["agents"]


def delete_agent(name: str) -> bool:
    reg = _load()
    before = len(reg["agents"])
    reg["agents"] = [a for a in reg["agents"] if a["name"] != name]
    _save(reg)
    return len(reg["agents"]) < before


# ── Full listing ───────────────────────────────────────────────────────────────
def list_all() -> dict:
    reg = _load()
    return {
        "skills": [{"name": s["name"], "description": s["description"]} for s in reg["skills"]],
        "agents": [{"name": a["name"], "description": a["description"]} for a in reg["agents"]],
    }


# ── Seed defaults  ───────────────────────────────────
def seed_defaults():
    if REGISTRY_PATH.exists():
        return

    defaults = [
        dict(
            name="grammar_fix",
            description="Fix grammar and spelling. Returns corrected text only.",
            system_prompt="Fix all grammar, spelling, and punctuation. Return ONLY the corrected text.",
            user_prompt_template="Fix this: {input}",
        ),
        dict(
            name="sentiment",
            description="Detect sentiment: Positive, Negative, or Neutral.",
            system_prompt="Return ONLY one word: Positive, Negative, or Neutral.",
            user_prompt_template="Sentiment of: {input}",
        ),
        dict(
            name="bullet_points",
            description="Rewrite text as a clean bullet-point list. No intro or outro.",
            system_prompt="Rewrite the input as a concise bullet-point list. Return ONLY the bullets, no preamble or summary sentence.",
            user_prompt_template="Convert to bullet points:\n\n{input}",
        ),
    ]

    for skill in defaults:
        register_skill(**skill)

    register_agent(
        name="news_analyst",
        description="Search for news on a topic and produce an executive summary with sources.",
        system_prompt="You are a senior analyst. Summarize clearly. Cite sources by [number].",
        tools=["web_search", "summarize"],
    )

    register_agent(
        name="research_brief",
        description=(
            "Deep-research a topic and write a concise executive brief. Use this when "
            "the user wants a structured overview of a subject with citations — not a "
            "single-page summary."
        ),
        system_prompt=(
            "You are a senior research analyst. Your job is to produce a tight, "
            "well-structured brief on the topic the user gives you.\n\n"
            "Workflow:\n"
            "1. Call `research` with the topic to gather sources and an initial synthesis.\n"
            "2. If one source is clearly authoritative and needs depth, call `fetch_page` "
            "   on its URL for the full text.\n"
            "3. Call `summarize` with style='bullets' on the assembled material if it's "
            "   longer than a few paragraphs.\n"
            "4. Optionally call `run_skill` with skill_name='bullet_points' to tighten "
            "   the final bullet list.\n\n"
            "Output format (markdown):\n"
            "  ## Overview\n"
            "  <2-3 sentence plain-prose summary>\n\n"
            "  ## Key findings\n"
            "  - <bullet 1>\n  - <bullet 2>\n  - <bullet 3-7>\n\n"
            "  ## Sources\n"
            "  [1] <title> — <url>\n  [2] ...\n\n"
            "Cite findings inline as [1], [2]. Do not invent sources or URLs — if a "
            "claim has no source, drop it. Keep the whole brief under ~400 words."
        ),
        tools=["research", "fetch_page", "summarize", "run_skill"],
        next_nodes=["news_analyst"],
    )
