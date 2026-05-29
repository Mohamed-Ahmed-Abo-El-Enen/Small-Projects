import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional
from src.config import DB_PATH
from src.logger import log_call


# ── Schema ─────────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    model       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    meta        TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    tool_name   TEXT NOT NULL,
    arguments   TEXT NOT NULL,
    result      TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_memory (
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (session_id, key)
);

CREATE TABLE IF NOT EXISTS file_refs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    file_path   TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    label       TEXT,
    created_at  TEXT NOT NULL
);
"""

# ── Internal helpers ───────────────────────────────────────────────────────────
def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript("PRAGMA foreign_keys = ON;")
    return con

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _touch(session_id: str,
           con: sqlite3.Connection):
    con.execute("""
                UPDATE 
                    sessions 
                SET 
                    updated_at=? 
                WHERE 
                    id=?
                """,
                (_now(), session_id),
            )
    
# ── Init ───────────────────────────────────────────────────────────────────────
def init_db():
    """Create all tables if they don't exist. Safe to call on every startup."""
    with _conn() as con:
        con.executescript(_SCHEMA)

# ── Sessions ───────────────────────────────────────────────────────────────────
@log_call
def create_session(name: str,
                   model: str,
                   meta: dict | None = None) -> str:
    """Create a new session and return its UUID."""
    sid = str(uuid.uuid4())
    now = _now()
    with _conn() as con:
        con.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?)",
            (sid, name, model, now, now, json.dumps(meta or {})),
        )
    return sid

def list_sessions() -> list[dict]:
    """Return all sessions ordered by most recently updated."""
    with _conn() as con:
        rows = con.execute("""
                            SELECT 
                                id, name, model, created_at, updated_at, meta
                            FROM 
                                sessions
                            ORDER BY 
                                updated_at DESC
                            """).fetchall()
    return [dict(row) for row in rows]

def get_session(session_id: str) -> Optional[dict]:
    """Return session by ID or None if not found."""
    with _conn() as con:
        row = con.execute("""
                            SELECT 
                                id, name, model, created_at, updated_at, meta
                            FROM 
                                sessions
                            WHERE 
                                id=?
                            """, (session_id,)).fetchone()
    return dict(row) if row else None

def rename_session(session_id: str, new_name: str) -> bool:
    """Rename session. Returns True if successful."""
    with _conn() as con:
        cur = con.execute("""
                          UPDATE 
                            sessions 
                          SET 
                            name=?, updated_at=? 
                          WHERE 
                            id=?
                          """, (new_name, _now(), session_id))
    return cur.rowcount > 0

@log_call
def delete_session(session_id: str) -> bool:
    """Delete session and all related data. Returns True if successful."""
    with _conn() as con:
        cur = con.execute("""
                          DELETE FROM 
                            sessions 
                          WHERE 
                            id=?""", (session_id,))
    return cur.rowcount > 0

# ── Messages ───────────────────────────────────────────────────────────────────
def add_message(session_id: str,
                role: str,
                content: str) -> int:
    """Add message to session and return message ID."""
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO messages (session_id, role, content, created_at) "
            "VALUES (?,?,?,?)",
            (session_id, role, content, _now()),
        )
        _touch(session_id, con)
        return cur.lastrowid

def get_messages(session_id: str) -> list[dict]:
    """Return full message history ordered by insertion."""
    with _conn() as con:
        rows = con.execute("""
                            SELECT 
                                id, role, content, created_at
                            FROM 
                                messages
                            WHERE 
                                session_id=?
                            ORDER BY 
                                created_at ASC
                            """, (session_id,)).fetchall()
    return [dict(row) for row in rows]

def get_ollama_history(session_id: str) -> list[dict]:
    """Return messages formatted for the Ollama `messages` parameter."""
    messages = get_messages(session_id)
    return [
        {"role": message["role"], 
         "content": message["content"]}

        for message in messages

        if message["role"] in ("user", "assistant", "system")
    ]

def clear_messages(session_id: str):
    """Delete all messages for a session."""
    with _conn() as con:
        con.execute("""
                    DELETE FROM 
                        messages 
                    WHERE 
                        session_id=?
                    """, (session_id,))
        _touch(session_id, con)


# ── Tool call log ──────────────────────────────────────────────────────────────
def log_tool_call(
    session_id: str,
    tool_name: str,
    arguments: dict,
    result: str | None = None,
):
    with _conn() as con:
        con.execute(
            "INSERT INTO tool_calls (session_id, tool_name, arguments, result, created_at) "
            "VALUES (?,?,?,?,?)",
            (session_id, tool_name, json.dumps(arguments), result, _now()),
        )
        _touch(session_id, con)


def get_tool_calls(session_id: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute("""
                            SELECT 
                                tool_name, arguments, result, created_at
                            FROM 
                                tool_calls
                            WHERE 
                                session_id=?
                            ORDER BY 
                                id
                            """, (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Session memory ─────────────────────────────────────────────────────────────
def memory_save(session_id: str, key: str, value: str):
    with _conn() as con:
        con.execute(
            "INSERT INTO session_memory VALUES (?,?,?,?) "
            "ON CONFLICT(session_id,key) DO UPDATE SET "
            "value=excluded.value, updated_at=excluded.updated_at",
            (session_id, key, value, _now()),
        )
        _touch(session_id, con)


def memory_recall(session_id: str, key: str) -> Optional[str]:
    with _conn() as con:
        row = con.execute("""
                        SELECT 
                            value 
                        FROM 
                            session_memory 
                        WHERE 
                            session_id=? AND key=?
                        """, (session_id, key),
        ).fetchone()
    return row["value"] if row else None


def memory_list(session_id: str) -> dict:
    with _conn() as con:
        rows = con.execute("""
                            SELECT 
                                key, value 
                            FROM 
                                session_memory 
                            WHERE 
                                session_id=?
                            """, (session_id,),
        ).fetchall()
    return {r["key"]: r["value"] for r in rows}


def memory_delete(session_id: str, key: str):
    with _conn() as con:
        con.execute("""
                    DELETE FROM 
                        session_memory 
                    WHERE 
                        session_id=? AND key=?""",
            (session_id, key),
        )


# ── File references ────────────────────────────────────────────────────────────

def add_file_ref(
    session_id: str,
    file_path: str,
    file_type: str,
    label: str | None = None,
):
    with _conn() as con:
        con.execute(
            "INSERT INTO file_refs (session_id, file_path, file_type, label, created_at) "
            "VALUES (?,?,?,?,?)",
            (session_id, file_path, file_type, label, _now()),
        )
        _touch(session_id, con)


def get_file_refs(session_id: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute("""
                           SELECT 
                                file_path, file_type, label, created_at FROM file_refs 
                           WHERE 
                                session_id=? 
                           ORDER BY 
                                id""", (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Export / Import ────────────────────────────────────────────────────────────
def export_session(session_id: str) -> dict:
    """Serialize a full session to a JSON-safe dict."""
    session = get_session(session_id)
    
    if not session:
        return {}
    
    return {
        "session":    session,
        "messages":   get_messages(session_id),
        "tool_calls": get_tool_calls(session_id),
        "memory":     memory_list(session_id),
        "files":      get_file_refs(session_id),
    }


@log_call
def import_session(data: dict) -> str:
    """Restore an exported session. Returns the new session ID."""
    s   = data["session"]
    sid = create_session(s["name"] + " (imported)", s["model"])
    
    for m in data.get("messages", []):
        add_message(sid, m["role"], m["content"])
    
    for k, v in data.get("memory", {}).items():
        memory_save(sid, k, v)
    
    for f in data.get("files", []):
        add_file_ref(sid, f["file_path"], f["file_type"], f.get("label"))
    
    return sid


# ── Summary ────────────────────────────────────────────────────────────────────

def session_summary(session_id: str) -> str:
    s    = get_session(session_id)
    
    if not s:
        return "Session not found."
        
    msgs = get_messages(session_id)
    mem  = memory_list(session_id)
    fls  = get_file_refs(session_id)
    tcs  = get_tool_calls(session_id)
    
    lines = [
        f"Name      : {s['name']}",
        f"ID        : {s['id']}",
        f"Model     : {s['model']}",
        f"Created   : {s['created_at']}",
        f"Updated   : {s['updated_at']}",
        f"Messages  : {len(msgs)}",
        f"Tool calls: {len(tcs)}",
        f"Memory    : {len(mem)} keys",
        f"Files     : {len(fls)}",
    ]

    return "\n".join(lines)
