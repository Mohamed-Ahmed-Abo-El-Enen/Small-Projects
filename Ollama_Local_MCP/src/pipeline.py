from typing import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph
from src.vector import retrieve

from src.config import DEFAULT_MODEL, OLLAMA_BASE
from src.logger import get_logger, log_call
from src.registry import list_agents

log = get_logger(__name__)


class PipelineState(TypedDict):
    input:   str
    context: str
    outputs: list[dict]
    final:   str


def _resolve_agent_tools(agent: dict) -> list:
    """Map the agent's configured tool names to real @tool objects. Excludes run_agent to prevent recursion."""
    names = set(agent.get("tools") or [])
    if not names:
        return []
    from src.agent import TOOLS as _TOP_TOOLS
    return [t for t in _TOP_TOOLS if t.name in names and t.name != "run_agent"]


def _format_prior(outputs: list[dict]) -> str:
    return "\n\n".join(f"[{o['agent']}]\n{o['output']}" for o in outputs)


def _build_user_prompt(state: PipelineState) -> str:
    parts = [f"User input:\n{state['input']}"]
    if state.get("context"):
        parts.append(f"Retrieved context:\n{state['context']}")
    prior = _format_prior(state["outputs"])
    if prior:
        parts.append(f"Prior agent outputs:\n{prior}")
    return "\n\n".join(parts)


def _make_agent_node(agent: dict):
    """Build a LangGraph node. ReAct if the agent has tools, plain llm.invoke if not."""
    tools = _resolve_agent_tools(agent)

    def node(state: PipelineState) -> dict:
        log.info(
            "pipeline agent start: %s (prior=%d, tools=%d)",
            agent["name"], len(state["outputs"]), len(tools),
        )
        llm = ChatOllama(
            model=agent.get("model", DEFAULT_MODEL),
            base_url=OLLAMA_BASE,
        )
        user_prompt = _build_user_prompt(state)

        output = ""
        sub_trace: list[str] = []

        if tools:
            from src.agent import _extract_tool_trace
            react = create_agent(llm, tools=tools, system_prompt=agent["system_prompt"])
            try:
                result = react.invoke({"messages": [HumanMessage(content=user_prompt)]})
                msgs = result.get("messages", []) or []
                if msgs:
                    last = msgs[-1]
                    output = last.content if hasattr(last, "content") else str(last)
                sub_trace = _extract_tool_trace(msgs)
            except Exception as exc:
                log.exception("pipeline agent %s failed in ReAct mode", agent["name"])
                output = f"(agent error: {type(exc).__name__}: {exc})"
        else:
            response = llm.invoke([
                SystemMessage(content=agent["system_prompt"]),
                HumanMessage(content=user_prompt),
            ])
            output = response.content if hasattr(response, "content") else str(response)

        log.info(
            "pipeline agent done: %s (chars=%d, trace=%s)",
            agent["name"], len(output), sub_trace,
        )

        entry = {"agent": agent["name"], "output": output}
        if sub_trace:
            entry["tool_trace"] = sub_trace

        return {
            "outputs": state["outputs"] + [entry],
            "final":   output,
        }

    node.__name__ = f"agent_{agent['name']}"
    return node


def _retrieve_node(state: PipelineState) -> dict:
    try:
        docs = retrieve(state["input"])
        if not docs:
            return {"context": ""}
        return {
            "context": "\n---\n".join(
                f"[{d.get('source')} p.{d.get('page')}]\n{d['content']}"
                for d in docs
            )
        }
    except Exception as e:
        return {"context": f"(retrieval unavailable: {e})"}


def build_graph(use_retrieval: bool = True):
    agents = list_agents()
    graph = StateGraph(PipelineState)

    prev = START
    if use_retrieval:
        graph.add_node("retrieve", _retrieve_node)
        graph.add_edge(START, "retrieve")
        prev = "retrieve"

    if not agents:
        graph.add_node("passthrough", lambda s: {"final": s["input"]})
        graph.add_edge(prev, "passthrough")
        graph.add_edge("passthrough", END)
        return graph.compile()

    for agent in agents:
        name = f"agent_{agent['name']}"
        graph.add_node(name, _make_agent_node(agent))
        graph.add_edge(prev, name)
        prev = name

    graph.add_edge(prev, END)
    return graph.compile()


@log_call
def run_pipeline(user_input: str, use_retrieval: bool = True) -> dict:
    graph = build_graph(use_retrieval=use_retrieval)
    final_state = graph.invoke({
        "input":   user_input,
        "context": "",
        "outputs": [],
        "final":   "",
    })
    return {
        "input":   user_input,
        "context": final_state.get("context", ""),
        "outputs": final_state["outputs"],
        "final":   final_state["final"],
        "agent_count": len(final_state["outputs"]),
    }


def list_pipeline() -> list[dict]:
    return [{"name": a["name"], "description": a["description"]} for a in list_agents()]
