from __future__ import annotations

import ast
import builtins
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

from .models import CodeFile, TestResult

_PY_BUILTINS = set(dir(builtins))
_TEST_FRAMEWORK_NAMES = {"pytest", "unittest", "mock"}


SKIP_DIRS = {
    ".venv", "venv", "env",
    "build", "target", "obj", "bin", "dist",
    "__pycache__", ".pytest_cache", ".mypy_cache",
    ".git", "node_modules",
}

_BAD_PKG_PATTERNS = (
    re.compile(r"No matching distribution found for\s+(\S+)", re.I),
    re.compile(r"Could not find a version that satisfies the requirement\s+(\S+)", re.I),
    re.compile(r"ERROR: Could not find a version of\s+(\S+)", re.I),
    re.compile(r"ERROR: No matching distribution for\s+(\S+)", re.I),
)

_PHANTOM_REQUIREMENTS = {
    "python", "python3", "python2",
    "stdlib", "standard-library", "standardlibrary",
    "builtins", "builtin",
    "venv", "virtual-env",
    "pip",
}

_ISOLATED_ENV_VARS = (
    "PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME",
    "PYTHONSTARTUP", "PYTHONUSERBASE",
)


def is_test_path(path: str) -> bool:
    """Return True when a project-relative path lives under a test directory."""
    p = path.replace("\\", "/").lower()
    return (
        p.startswith("tests/")
        or "/tests/" in p
        or p.startswith("test/")
        or "src/test/" in p
        or p.endswith("_test.py") or p.endswith("test.py")
    )


def _strip_phantom_requirements(req_text: str) -> tuple[str, list[str]]:
    """Drop fake requirements like 'python' before pip ever sees them."""
    kept: list[str] = []
    dropped: list[str] = []
    for line in req_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            kept.append(line)
            continue
        name = re.split(r"[=<>!~\[]", stripped, 1)[0].strip().lower()
        if name in _PHANTOM_REQUIREMENTS:
            dropped.append(stripped)
        else:
            kept.append(line)
    return "\n".join(kept), dropped


def dep_name(spec: str) -> str:
    """Return the lowercase package name from a requirement spec."""
    return re.split(r"[=<>!~\[]", (spec or "").strip(), 1)[0].strip().lower()


def dedup_requirements(specs: list[str]) -> tuple[list[str], list[str]]:
    """Dedup specs by package name, keeping the first occurrence."""
    seen: set[str] = set()
    kept: list[str] = []
    dropped: list[str] = []
    for spec in specs:
        stripped = spec.strip()
        if not stripped or stripped.startswith("#"):
            kept.append(spec)
            continue
        name = dep_name(stripped)
        if not name:
            kept.append(spec)
            continue
        if name in seen:
            dropped.append(stripped)
            continue
        seen.add(name)
        kept.append(stripped)
    return kept, dropped


def _parse_bad_packages(output: str) -> list[str]:
    """Extract package names pip could not resolve from its error output."""
    bad: list[str] = []
    seen: set[str] = set()
    for pat in _BAD_PKG_PATTERNS:
        for m in pat.finditer(output):
            raw = m.group(1).strip().strip(".,")
            name = re.split(r"[=<>!~\[]", raw, 1)[0].strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                bad.append(name)
    return bad


def _clean_subprocess_env() -> dict[str, str]:
    """Return an env dict with the agent's Python config stripped out."""
    env = os.environ.copy()
    for var in _ISOLATED_ENV_VARS:
        env.pop(var, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def iter_source_files(project_dir: str | Path,
                      extensions: tuple[str, ...] | list[str] = (".py",),
                      max_files: int = 2000) -> list[Path]:
    """Walk the project for files with the given extensions, pruning build dirs."""
    base = Path(project_dir)
    exts = tuple(extensions)
    out: list[Path] = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if name.endswith(exts):
                out.append(Path(root) / name)
                if len(out) >= max_files:
                    out.sort()
                    return out
    out.sort()
    return out


def resolve_venv_dir(project_dir: str | Path) -> Path:
    """Return the project's venv path under the workspace."""
    from . import config
    return config.WORKSPACE_DIR / Path(project_dir).resolve().name / ".venv"


def venv_python(project_dir: str | Path) -> Path:
    """Return the path to the project's venv Python interpreter."""
    base = resolve_venv_dir(project_dir)
    win = base / "Scripts" / "python.exe"
    nix = base / "bin" / "python"
    return win if win.exists() else nix


def setup_project_env(project_dir: str | Path,
                      timeout: int = 1800,
                      on_progress=None) -> tuple[bool, str, list[str]]:
    """Create the project venv and install requirements.txt into it."""
    def _say(msg: str) -> None:
        if on_progress is not None:
            try:
                on_progress(msg)
            except Exception:
                pass

    base = Path(project_dir)
    venv_dir = resolve_venv_dir(base)
    req_file = base / "requirements.txt"

    clean_env = _clean_subprocess_env()
    clean_env["PIP_REQUIRE_VIRTUALENV"] = "1"

    if not venv_dir.exists():
        _say("creating venv")
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_dir)],
                check=True, capture_output=True, text=True, timeout=180,
                env=clean_env,
            )
        except subprocess.CalledProcessError as exc:
            return False, f"venv creation failed: {exc.stderr or exc}", []
        except subprocess.SubprocessError as exc:
            return False, f"venv creation failed: {exc}", []

    if not req_file.exists() or not req_file.read_text(encoding="utf-8").strip():
        return True, "venv ready (no requirements.txt to install)", []

    py = venv_python(base)
    if not py.exists():
        return False, f"venv python not found at {py}", []

    req_text = req_file.read_text(encoding="utf-8")
    cleaned_text, phantoms = _strip_phantom_requirements(req_text)
    if phantoms:
        _say(f"stripped phantom requirements: {phantoms}")

    deduped, dropped_dupes = dedup_requirements(cleaned_text.splitlines())
    if dropped_dupes:
        _say(f"dropped duplicate requirements: {dropped_dupes}")
        cleaned_text = "\n".join(deduped) + ("\n" if deduped else "")

    if phantoms or dropped_dupes:
        req_file.write_text(cleaned_text, encoding="utf-8")

    req_bytes = req_file.read_bytes()
    req_hash = hashlib.sha256(req_bytes).hexdigest()
    hash_file = venv_dir / ".last-installed-hash"
    if hash_file.exists():
        try:
            if hash_file.read_text(encoding="utf-8").strip() == req_hash:
                _say("requirements.txt unchanged since last install — skipping pip")
                return True, "venv ready (cached: requirements unchanged)", []
        except OSError:
            pass

    deps = [d.strip() for d in req_bytes.decode("utf-8", errors="replace").splitlines() if d.strip()]
    _say(f"pip installing {len(deps)} packages from requirements.txt")
    try:
        proc = subprocess.run(
            [str(py), "-m", "pip", "install",
             "--disable-pip-version-check", "-r", str(req_file)],
            capture_output=True, text=True, timeout=timeout,
            env=clean_env,
        )
    except subprocess.TimeoutExpired:
        return (False,
                f"pip install timed out after {timeout}s. "
                f"Try installing manually:\n  {py} -m pip install -r {req_file}",
                [])
    except subprocess.SubprocessError as exc:
        return False, f"pip install error: {exc}", []

    output = (proc.stderr or "") + "\n" + (proc.stdout or "")
    if proc.returncode != 0:
        bad = _parse_bad_packages(output)
        snippet = output.strip().splitlines()[-6:]
        msg = "pip install failed"
        if bad:
            msg += f"; unresolvable packages: {bad}"
        return False, f"{msg}\n  " + "\n  ".join(snippet), bad

    try:
        hash_file.write_text(req_hash, encoding="utf-8")
    except OSError:
        pass

    _say("ensuring pytest + pytest-timeout are installed")
    subprocess.run(
        [str(py), "-m", "pip", "install", "-q",
         "--disable-pip-version-check", "pytest", "pytest-timeout"],
        capture_output=True, text=True, timeout=180,
        env=clean_env,
    )

    try:
        ver = subprocess.run(
            [str(py), "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=30, env=clean_env,
        )
        pip_path = (ver.stdout or "").strip()
        venv_str = str(venv_dir.resolve())
        if venv_str not in pip_path:
            return (False,
                    f"isolation check failed: pip not inside venv "
                    f"({pip_path!r} does not contain {venv_str!r})",
                    [])
    except subprocess.SubprocessError:
        pass

    return True, "venv ready", []


def write_files(project_dir: str | Path, files: list[CodeFile]) -> None:
    """Persist code files under project_dir, creating parents as needed."""
    base = Path(project_dir)
    for f in files:
        target = base / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f.content, encoding="utf-8")


_GREEN_DIR = ".last_green"
_SNAPSHOT_EXTS = (".py", ".txt", ".md", ".cfg", ".toml", ".ini", ".json")


def _iter_snapshot_files(root: Path) -> list[Path]:
    """Walk a project tree for committable files, pruning build/venv dirs."""
    out: list[Path] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if name.endswith(_SNAPSHOT_EXTS):
                out.append(Path(dirpath) / name)
    return out


def snapshot_green(project_dir: str | Path, workspace_dir: str | Path) -> None:
    """Save the current project source as the last-known-green state."""
    import shutil
    src = Path(project_dir)
    dest = Path(workspace_dir) / _GREEN_DIR
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    for p in _iter_snapshot_files(src):
        target = dest / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)


def restore_green(project_dir: str | Path, workspace_dir: str | Path) -> bool:
    """Restore the project from the last green snapshot. Returns True if restored."""
    import shutil
    snap = Path(workspace_dir) / _GREEN_DIR
    if not snap.exists() or not any(snap.iterdir()):
        return False
    dest = Path(project_dir)
    for p in _iter_snapshot_files(dest):
        try:
            p.unlink()
        except OSError:
            pass
    for p in _iter_snapshot_files(snap):
        target = dest / p.relative_to(snap)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    return True


def has_green_snapshot(workspace_dir: str | Path) -> bool:
    """Return True if a last-green snapshot exists for this workspace."""
    snap = Path(workspace_dir) / _GREEN_DIR
    return snap.exists() and any(snap.iterdir())


_GREEN_STATE_FILE = ".green_state.json"


def save_green_state(workspace_dir: str | Path, blob: str) -> None:
    """Persist the requirements/plan that were green, alongside the file snapshot."""
    try:
        (Path(workspace_dir) / _GREEN_STATE_FILE).write_text(blob, encoding="utf-8")
    except OSError:
        pass


def load_green_state(workspace_dir: str | Path) -> dict | None:
    """Load the last-green requirements/plan, or None if absent/invalid."""
    import json
    path = Path(workspace_dir) / _GREEN_STATE_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_all_sources(project_dir: str | Path) -> list[CodeFile]:
    """Read every project source file."""
    base = Path(project_dir)
    files: list[CodeFile] = []
    for p in iter_source_files(base):
        rel = p.relative_to(base).as_posix()
        try:
            content = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        files.append(CodeFile(path=rel, content=content))
    return files


def run_tests(project_dir: str | Path,
              timeout: int = 150,
              per_test_timeout: int = 10) -> TestResult:
    """Run the project's pytest suite and return a coarse summary."""
    base = Path(project_dir)
    if not (base / "tests").exists():
        return TestResult(passed=True, total=0, output="No tests directory; skipped.")
    py = venv_python(base)
    interpreter = str(py) if py.exists() else sys.executable

    env = _clean_subprocess_env()
    env["PYTHONPATH"] = str(base)

    timeout_method = "signal" if os.name == "posix" else "thread"
    session_timeout = max(per_test_timeout * 3, timeout // 2)
    log_file = base / ".pytest_last_output.log"
    cmd = [
        interpreter, "-m", "pytest", "-v", "--tb=long",
        f"--timeout={per_test_timeout}",
        f"--timeout-method={timeout_method}",
        f"--session-timeout={session_timeout}",
    ]

    returncode: int | None = None
    timed_out = False
    try:
        with log_file.open("w", encoding="utf-8") as fh:
            try:
                proc = subprocess.run(
                    cmd, cwd=str(base),
                    stdout=fh, stderr=subprocess.STDOUT,
                    text=True, timeout=timeout, env=env,
                )
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
    except FileNotFoundError:
        return TestResult(passed=False,
                          output="python/pytest not available on PATH")

    try:
        out = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        out = ""

    if timed_out:
        summary = summarize_test_failures(out)
        details = extract_failure_details(out)
        collecting = _hanging_collection_targets(out, base)
        parts = [
            f"pytest TIMED OUT after {timeout}s — a test or a test module's import "
            "is HANGING (infinite loop, blocking I/O, a network call, or a "
            "multiprocessing/Pool call that never returns). The hanging test or "
            "its dependency must be fixed, given a small bounded input, or removed."
        ]
        if collecting:
            parts.append("# Was still importing/collecting when killed (suspect these)\n"
                         + "\n".join(collecting))
        if summary:
            parts.append("# Tests already failing/erroring before the kill\n"
                         + "\n".join(summary))
        if details:
            parts.append("# Error tracebacks\n" + details)
        parts.append("# Partial output tail\n" + out[-2000:])
        return TestResult(
            passed=False, total=0, failed=len(summary), errors=0,
            output="\n\n".join(parts),
        )

    if returncode == 5:
        return TestResult(
            passed=False, total=0, failed=0,
            output="NO TESTS COLLECTED — pytest found no test functions.\n\n"
                   + out[-3000:],
        )

    passed = returncode == 0
    total = failed = errors = 0
    for line in out.splitlines():
        line_l = line.lower()
        if "passed" in line_l or "failed" in line_l or "error" in line_l:
            for token in line_l.replace(",", " ").split():
                if token.isdigit():
                    n = int(token)
                    if "failed" in line_l and failed == 0:
                        failed = n
                    elif "error" in line_l and errors == 0:
                        errors = n
                    elif "passed" in line_l and total == 0:
                        total = n

    if passed:
        return TestResult(passed=True, total=total, output=out[-2000:])

    summary = summarize_test_failures(out)
    details = extract_failure_details(out)
    parts: list[str] = []
    if summary:
        parts.append("# Failing tests\n" + "\n".join(summary))
    if details:
        parts.append("# Error tracebacks (the actual cause)\n" + details)
    composed = "\n\n".join(parts) if parts else out[-4000:]
    return TestResult(passed=passed, total=total, failed=failed, errors=errors,
                      output=composed)


def summarize_test_failures(output: str, limit: int = 20) -> list[str]:
    """Pull deduped `FAILED ...` / `ERROR ...` summary lines from pytest output."""
    fails: list[str] = []
    seen: set[str] = set()
    for line in output.splitlines():
        s = line.strip()
        if s.startswith("FAILED ") or s.startswith("ERROR "):
            if s in seen:
                continue
            seen.add(s)
            fails.append(s)
            if len(fails) >= limit:
                break
    return fails


_FAIL_SECTION_RE = re.compile(r"^=+\s*(FAILURES|ERRORS)\s*=+\s*$")
_SUMMARY_SECTION_RE = re.compile(r"^=+\s*short test summary info\s*=+\s*$")


def extract_failure_details(output: str, max_chars: int = 4500) -> str:
    """Return pytest's FAILURES/ERRORS traceback region (the actual error cause)."""
    lines = output.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if _FAIL_SECTION_RE.match(ln.strip())), None)
    if start is None:
        return ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _SUMMARY_SECTION_RE.match(lines[j].strip()):
            end = j
            break
    detail = "\n".join(lines[start:end]).strip()
    if len(detail) > max_chars:
        detail = detail[:max_chars] + "\n... (traceback truncated)"
    return detail


def _hanging_collection_targets(output: str, project_dir: Path) -> list[str]:
    """Test files that reported no pass/fail/error — prime suspects for hanging."""
    reported: set[str] = set()
    for line in output.splitlines():
        if any(tok in line for tok in
               ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")):
            for m in re.finditer(r"(tests?/[^\s:]+\.py)", line):
                reported.add(m.group(1))
    suspects: list[str] = []
    for sub in ("tests", "test"):
        d = project_dir / sub
        if not d.exists():
            continue
        for p in iter_source_files(d):
            if not (p.name.startswith("test_") or p.name.endswith("_test.py")):
                continue
            rel = p.relative_to(project_dir).as_posix()
            if rel not in reported and rel not in suspects:
                suspects.append(rel)
    return suspects


def format_failure_summary(output: str, inline_limit: int = 3) -> str:
    """One-line digest of the failure summary, suitable for a log message."""
    fails = summarize_test_failures(output)
    if not fails:
        return ""
    head = " | ".join(fails[:inline_limit])
    extra = f" (+{len(fails) - inline_limit} more)" if len(fails) > inline_limit else ""
    return f"{len(fails)} fail(s): {head}{extra}"


def failing_test_files(output: str, project_dir: str | Path) -> list[CodeFile]:
    """Read the test files referenced by FAILED/ERROR lines in pytest output."""
    base = Path(project_dir)
    seen: set[str] = set()
    out: list[CodeFile] = []
    for line in summarize_test_failures(output, limit=50):
        m = re.search(r"(?:FAILED|ERROR)\s+([^\s:]+\.py)", line)
        if not m:
            continue
        rel = m.group(1)
        if rel in seen:
            continue
        seen.add(rel)
        path = base / rel
        if path.exists():
            try:
                out.append(CodeFile(path=rel, content=path.read_text(encoding="utf-8")))
            except OSError:
                continue
    return out


def _module_name_for_path(rel_path: str) -> str:
    """Convert a project-relative .py path into its dotted module name."""
    if rel_path.endswith(".py"):
        rel_path = rel_path[:-3]
    parts = rel_path.replace("\\", "/").split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def project_exports(project_dir: str | Path) -> dict[str, str]:
    """Map each top-level function/class name in the source to its module path."""
    base = Path(project_dir)
    exports: dict[str, str] = {}
    for p in iter_source_files(base, [".py"]):
        rel = p.relative_to(base).as_posix()
        if is_test_path(rel) or rel == "conftest.py" or rel.endswith("/__init__.py"):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        module = _module_name_for_path(rel)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    exports.setdefault(node.name, module)
    return exports


def format_import_map(exports: dict[str, str]) -> str:
    """Render the export map as a `from X import ...` cheat sheet for the LLM."""
    if not exports:
        return "(no project sources exporting names yet)"
    by_module: dict[str, list[str]] = {}
    for name, module in exports.items():
        by_module.setdefault(module, []).append(name)
    lines: list[str] = []
    for module in sorted(by_module):
        names = sorted(set(by_module[module]))
        lines.append(f"from {module} import {', '.join(names)}")
    return "\n".join(lines)


def auto_inject_imports(file_content: str, exports: dict[str, str]) -> str:
    """Add `from <module> import <name>` lines for any project name used but not imported."""
    if not exports:
        return file_content
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return file_content

    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            referenced.add(node.value.id)

    known: set[str] = set(_PY_BUILTINS) | _TEST_FRAMEWORK_NAMES
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                known.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                known.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            known.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    known.add(target.id)

    needed: dict[str, str] = {
        name: exports[name]
        for name in referenced
        if name in exports and name not in known
    }
    if not needed:
        return file_content

    by_module: dict[str, list[str]] = {}
    for name, module in needed.items():
        by_module.setdefault(module, []).append(name)

    import_lines = [
        f"from {module} import {', '.join(sorted(set(names)))}"
        for module, names in sorted(by_module.items())
    ]
    return "\n".join(import_lines) + "\n\n" + file_content
