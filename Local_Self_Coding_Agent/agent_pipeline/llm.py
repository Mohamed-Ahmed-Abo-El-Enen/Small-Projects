from __future__ import annotations

import json
import logging
import re
import time
import typing
from typing import Type, TypeVar, get_args, get_origin

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from . import config

_log = logging.getLogger("self_coding_agent")


class LLMUnavailable(RuntimeError):
    pass


_TRANSIENT_HINTS = (
    "connection", "timeout", "timed out", "max retries",
    "temporary failure", "name resolution",
    "502", "503", "504", "520", "521", "522", "523", "524", "525", "526",
    "527", "528", "529", "530", "599",
    "cloudflare tunnel error", "bad gateway", "service unavailable",
)


def _is_transient(exc: Exception) -> bool:
    """Return True for network blips worth retrying."""
    msg = (str(exc) or "").lower()
    return any(hint in msg for hint in _TRANSIENT_HINTS)


def ping_ollama(timeout: float = 10.0) -> tuple[bool, str]:
    """Check that Ollama is reachable and has the configured models."""
    url = config.OLLAMA_HOST.rstrip("/") + "/api/tags"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        models = [m.get("name", "") for m in r.json().get("models", [])]
        needed = {config.REASONING_MODEL, config.CODING_MODEL, config.EMBEDDING_MODEL}
        missing = [m for m in needed if m and m not in models]
        if missing:
            return False, f"Ollama reachable but missing models: {missing}. Available: {models[:5]}"
        return True, f"Ollama OK at {config.OLLAMA_HOST} (models: {len(models)})"
    except Exception as exc:
        return False, f"Ollama unreachable at {config.OLLAMA_HOST}: {exc}"


T = TypeVar("T", bound=BaseModel)


def reasoning_llm(temperature: float | None = None,
                  json_mode: bool = False) -> ChatOllama:
    """Return the reasoning model for planning, requirements, review."""
    kwargs: dict = dict(
        model=config.REASONING_MODEL,
        base_url=config.OLLAMA_HOST,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT,
    )
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


def coding_llm(temperature: float | None = None,
               json_mode: bool = False) -> ChatOllama:
    """Return the coding model for source and test files."""
    kwargs: dict = dict(
        model=config.CODING_MODEL,
        base_url=config.OLLAMA_HOST,
        temperature=temperature if temperature is not None else 0.1,
        timeout=config.LLM_TIMEOUT,
    )
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


def chat(llm: ChatOllama, system: str, user: str,
         max_attempts: int = 4, base_delay: float = 2.0) -> str:
    """Send a system+user pair, retrying transient backend errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
            return (response.content if isinstance(response.content, str)
                    else str(response.content))
        except Exception as exc:
            last_exc = exc
            if not _is_transient(exc) or attempt == max_attempts:
                break
            wait = base_delay * (2 ** (attempt - 1))
            _log.warning("LLM call failed (attempt %d/%d): %s — retrying in %.0fs",
                         attempt, max_attempts,
                         str(exc).splitlines()[0][:200], wait)
            time.sleep(wait)

    msg = (str(last_exc).splitlines()[0] if last_exc else "unknown").strip()[:300]
    raise LLMUnavailable(
        f"LLM backend unreachable after {max_attempts} attempts at "
        f"{config.OLLAMA_HOST}: {msg}"
    )


def _extract_json(text: str) -> str:
    """Pull the first JSON object or array out of an LLM reply."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    obj = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if obj:
        return obj.group(1)
    return text


def _looks_like_schema(text: str) -> bool:
    """Return True when the LLM parroted the JSON Schema instead of an instance."""
    head = text[:400]
    markers = ('"$schema"', '"$defs"', '"properties"', '"definitions"',
               '"additionalProperties"', '"required"')
    return sum(1 for m in markers if m in head) >= 1


def _type_label(annotation) -> str:
    """Render a Python type annotation as a short human hint."""
    origin = get_origin(annotation)
    if origin is list or origin is typing.List:  # noqa: UP006
        inner = get_args(annotation)
        if inner:
            return f"array of {_type_label(inner[0])}"
        return "array"
    if origin is dict or origin is typing.Dict:  # noqa: UP006
        return "object"
    if origin is typing.Union:
        parts = [_type_label(a) for a in get_args(annotation) if a is not type(None)]
        return parts[0] if parts else "any"
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return f"object {{{describe_fields(annotation)}}}"
        return {
            str: "string", int: "integer", float: "number", bool: "boolean",
        }.get(annotation, annotation.__name__)
    return str(annotation)


def _stringify(item) -> str:
    """Flatten a dict or list to a single string the schema can accept."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if "name" in item and "description" in item:
            return f"{item['name']}: {item['description']}"
        if "title" in item and "description" in item:
            return f"{item['title']}: {item['description']}"
        for k in ("name", "title", "value", "label", "text"):
            if k in item and isinstance(item[k], str):
                return item[k]
        return ", ".join(f"{k}={v}" for k, v in item.items())
    if isinstance(item, list):
        return ", ".join(_stringify(x) for x in item)
    return str(item)


def _coerce_to_schema(data, schema: Type[BaseModel]):
    """Coerce list-of-dict values into list-of-str when the schema expects strings."""
    if not isinstance(data, dict):
        return data
    fields = getattr(schema, "model_fields", {})
    for name, field in fields.items():
        if name not in data:
            continue
        ann = field.annotation
        origin = get_origin(ann)
        if origin in (list, typing.List):  # noqa: UP006
            inner = get_args(ann)
            if inner and inner[0] is str and isinstance(data[name], list):
                data[name] = [
                    item if isinstance(item, str) else _stringify(item)
                    for item in data[name]
                ]
    return data


def describe_fields(schema: Type[BaseModel]) -> str:
    """Return a plain-language list of the schema's fields."""
    lines = []
    for name, field in schema.model_fields.items():
        kind = _type_label(field.annotation)
        required = "required" if field.is_required() else "optional"
        desc = (field.description or "").strip()
        line = f"  - {name} ({kind}, {required})"
        if desc:
            line += f": {desc}"
        lines.append(line)
    return "\n".join(lines)


def chat_json(llm: ChatOllama, system: str, user: str, schema: Type[T]) -> T:
    """Ask the LLM for a JSON instance of a Pydantic schema, with retries."""
    fields_doc = describe_fields(schema)
    base_instruction = (
        "Return ONE JSON object that is an INSTANCE matching the field "
        "specification below. Do NOT return the schema itself: no `$schema`, "
        "no `properties`, no `type: object`, no `$defs`. Output only the "
        "JSON object — no prose, no code fences.\n\n"
        f"Required fields:\n{fields_doc}"
    )
    prompt = f"{user}\n\n{base_instruction}"

    last_raw = ""
    for attempt in range(3):
        raw = chat(llm, system, prompt)
        last_raw = raw
        candidate = _extract_json(raw)
        if _looks_like_schema(candidate):
            _log.warning("LLM returned a JSON Schema instead of an instance "
                         "(attempt %d); retrying with stricter prompt.", attempt + 1)
            prompt = (
                f"{user}\n\nThe previous response was the schema definition, "
                "not an instance. Return a concrete JSON object whose KEYS are "
                f"exactly: {list(schema.model_fields.keys())}. "
                "Do not include the words 'properties', '$schema', or 'type'.\n\n"
                f"Required fields:\n{fields_doc}"
            )
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            _log.warning("JSON parse failed (attempt %d): %s",
                         attempt + 1, str(exc)[:200])
            prompt = (
                f"{user}\n\nThe previous reply was not valid JSON:\n{raw[:600]}\n\n"
                "Fix it and return a corrected JSON object. "
                f"Required fields:\n{fields_doc}"
            )
            continue

        coerced = _coerce_to_schema(parsed, schema) if isinstance(parsed, dict) else parsed
        try:
            return schema.model_validate(coerced)
        except ValidationError as exc:
            _log.warning("Schema validation failed (attempt %d): %s",
                         attempt + 1, str(exc)[:200])
            prompt = (
                f"{user}\n\nThe previous reply did not match the schema:\n"
                f"{raw[:600]}\n\nValidation error: {str(exc)[:300]}\n\n"
                "Fix it and return a corrected JSON object. "
                f"Required fields:\n{fields_doc}"
            )

    raise ValueError(
        f"chat_json could not get a valid {schema.__name__} after 3 attempts. "
        f"Last raw response:\n{last_raw[:1000]}"
    )
