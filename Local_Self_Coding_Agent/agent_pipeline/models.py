from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ResearchNote(BaseModel):
    source: str
    title: str
    summary: str
    snippets: list[str] = Field(default_factory=list)


class Requirement(BaseModel):
    rid: str
    kind: Literal["functional", "non_functional", "constraint"]
    statement: str
    rationale: str
    acceptance: list[str] = Field(default_factory=list)


class FileSpec(BaseModel):
    path: str
    purpose: str
    public_api: list[str] = Field(default_factory=list)


class ProjectPlan(BaseModel):
    name: str
    description: str
    entry_point: Optional[str] = None
    files: list[FileSpec] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)


class CodeFile(BaseModel):
    path: str
    content: str


class TestResult(BaseModel):
    passed: bool
    total: int = 0
    failed: int = 0
    errors: int = 0
    output: str = ""


class BugReport(BaseModel):
    bid: str
    severity: Literal["low", "medium", "high", "critical"]
    location: str
    description: str
    suggested_fix: str = ""
    resolved: bool = False


class FeatureIdea(BaseModel):
    fid: str
    title: str
    motivation: str
    proposed_changes: list[str] = Field(default_factory=list)


class IterationReport(BaseModel):
    iteration: int
    started_at: datetime
    finished_at: Optional[datetime] = None
    requirements: list[Requirement] = Field(default_factory=list)
    plan: Optional[ProjectPlan] = None
    test_result: Optional[TestResult] = None
    bugs_found: list[BugReport] = Field(default_factory=list)
    bugs_fixed: list[str] = Field(default_factory=list)
    new_features: list[FeatureIdea] = Field(default_factory=list)
    notes: str = ""


class ProjectState(BaseModel):
    project_name: str
    project_dir: str
    workspace_dir: str
    idea: str

    iteration: int = 0
    max_iterations: int = 5

    consecutive_failures: int = 0
    max_consecutive_failures: int = 3

    research: list[ResearchNote] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    plan: Optional[ProjectPlan] = None
    files: list[CodeFile] = Field(default_factory=list)
    tests: list[CodeFile] = Field(default_factory=list)

    last_test_result: Optional[TestResult] = None
    open_bugs: list[BugReport] = Field(default_factory=list)
    closed_bugs: list[BugReport] = Field(default_factory=list)

    pending_features: list[FeatureIdea] = Field(default_factory=list)
    abandoned_features: list[str] = Field(default_factory=list)
    history: list[IterationReport] = Field(default_factory=list)

    fix_attempts: int = 0
    max_fix_attempts: int = 5
    no_progress_streak: int = 0
    last_failure_keys: list[str] = Field(default_factory=list)
    test_doctor_visits: int = 0

    final_doc_path: Optional[str] = None
    done: bool = False
    stop_reason: str = ""

    force_innovation: bool = False

    def model_dump_json_pretty(self) -> str:
        """Return state as pretty-printed JSON."""
        return self.model_dump_json(indent=2)


class Event(BaseModel):
    ts: datetime = Field(default_factory=datetime.utcnow)
    iteration: int
    agent: str
    level: Literal["info", "warn", "error", "result"] = "info"
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
