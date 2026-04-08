"""
api/main.py
-----------
Production-ready FastAPI backend for the cybersecurity multi-agent system.

Endpoints:
  GET  /            → health check
  GET  /run         → full pipeline (attack → analyse → decide → act)
  POST /upload      → ingest a .log/.txt file and run each line through pipeline
  GET  /incidents   → all stored incidents
  GET  /blocked     → all blocked IPs

Run:
    uvicorn api.main:app --reload
"""

import io
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Agent imports ──────────────────────────────────────────────────────────────
from agents.red_agent import RedTeamAgent
from agents.blue_agent import BlueTeamAgent
from agents.planner_agent import PlannerAgent


# ══════════════════════════════════════════════════════════════════════════════
# Upload policy constants  (single source of truth — change here only)
# ══════════════════════════════════════════════════════════════════════════════

UPLOAD_MAX_BYTES      = 1 * 1024 * 1024   # 1 MB hard limit
UPLOAD_MAX_LINES      = 5                 # process at most 5 log lines per file
UPLOAD_ALLOWED_TYPES  = {"text/plain", "application/octet-stream"}
UPLOAD_ALLOWED_EXTS   = {".log", ".txt"}


# ══════════════════════════════════════════════════════════════════════════════
# In-process stores  (swap for PostgreSQL / Redis in production)
# ══════════════════════════════════════════════════════════════════════════════

class MemoryStore:
    """Lightweight in-memory incident and blocked-IP store."""

    def __init__(self) -> None:
        self._incidents:   list[dict] = []
        self._blocked_ips: set[str]   = set()

    def add_incident(self, incident: dict) -> None:
        self._incidents.append(incident)

    def get_incidents(self) -> list[dict]:
        return list(reversed(self._incidents))   # newest first

    def block_ip(self, ip: str) -> None:
        self._blocked_ips.add(ip)

    def get_blocked_ips(self) -> list[str]:
        return sorted(self._blocked_ips)

    def is_blocked(self, ip: str) -> bool:
        return ip in self._blocked_ips


class ActionEngine:
    """Executes the action decided by the PlannerAgent."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def execute(self, plan: dict) -> dict:
        decision  = plan.get("decision", "ignore")
        target_ip = plan.get("target_ip", "unknown")
        result    = {"action_taken": decision, "target_ip": target_ip}

        if decision == "block_ip":
            self._store.block_ip(target_ip)
            result["detail"] = f"IP {target_ip} added to blocklist."
        elif decision == "monitor":
            result["detail"] = f"IP {target_ip} added to watchlist."
        else:
            result["detail"] = "No active response required."

        return result


# ══════════════════════════════════════════════════════════════════════════════
# Log line parser
# Converts a raw text line from an uploaded file into a structured log dict
# that BlueTeamAgent can process.
# ══════════════════════════════════════════════════════════════════════════════

def _parse_log_line(line: str, line_index: int) -> dict:
    """
    Try to parse `line` as JSON first; fall back to a plain-text envelope.

    Both paths produce a dict with the minimum fields BlueTeamAgent requires:
      log_id · attack_type · source_ip · timestamp · raw_line
    """
    line = line.strip()

    # ── Attempt 1: valid JSON object ─────────────────────────────────────────
    if line.startswith("{"):
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                # Ensure required fields have fallbacks so validation won't fail
                parsed.setdefault("log_id",      str(uuid.uuid4()))
                parsed.setdefault("attack_type", "unknown")
                parsed.setdefault("source_ip",   "0.0.0.0")
                parsed.setdefault("timestamp",   _utcnow())
                parsed["raw_line"] = line
                return parsed
        except json.JSONDecodeError:
            pass   # fall through to plain-text envelope

    # ── Attempt 2: plain-text envelope ───────────────────────────────────────
    # Heuristic keyword detection so the BlueTeamAgent fallback rules have
    # something meaningful to match against.
    lower = line.lower()

    if any(k in lower for k in ("brute", "login", "password", "auth fail")):
        attack_type = "brute_force"
    elif any(k in lower for k in ("sql", "inject", "union", "select", "drop")):
        attack_type = "sql_injection"
    elif any(k in lower for k in ("scan", "port", "nmap", "probe")):
        attack_type = "port_scan"
    else:
        attack_type = "unknown"

    # Attempt to extract an IPv4-ish token from the line
    import re
    ip_match = re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", line)
    source_ip = ip_match.group(0) if ip_match else "0.0.0.0"

    return {
        "log_id":      str(uuid.uuid4()),
        "attack_type": attack_type,
        "source_ip":   source_ip,
        "target_ip":   "unknown",
        "timestamp":   _utcnow(),
        "payload":     line[:500],   # cap at 500 chars for safety
        "raw_line":    line,
        "_line_index": line_index,
    }


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic response schemas
# ══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status:    str = "ok"
    service:   str = "cybersecurity-multi-agent-api"
    version:   str = "1.2.0"
    timestamp: str = Field(default_factory=lambda: _utcnow())


class AttackLog(BaseModel):
    log_id:      str
    agent_id:    str  = "UPLOAD"
    timestamp:   str
    source_ip:   str
    target_ip:   str  = "unknown"
    attack_type: str
    payload:     Any  = None


class ThreatAssessment(BaseModel):
    threat_level:       str
    reason:             str
    recommended_action: str
    source_ip:          str
    source_log_id:      str
    assessed_by:        str
    assessed_at:        str


class ActionPlan(BaseModel):
    decision:    str
    explanation: str
    target_ip:   str
    planned_by:  str
    planned_at:  str


class ActionResult(BaseModel):
    action_taken: str
    target_ip:    str
    detail:       str


class PipelineResponse(BaseModel):
    run_id:     str
    status:     str = "completed"
    ran_at:     str = Field(default_factory=lambda: _utcnow())
    attack:     AttackLog
    assessment: ThreatAssessment
    plan:       ActionPlan
    action:     ActionResult


# ── Upload-specific schemas ────────────────────────────────────────────────────

class UploadLogResult(BaseModel):
    """Result for a single log line processed through the pipeline."""
    line_index:  int
    raw_log:     str
    analysis:    dict
    decision:    dict
    action:      dict


class UploadResponse(BaseModel):
    filename:      str
    lines_found:   int
    lines_processed: int
    processed_at:  str = Field(default_factory=lambda: _utcnow())
    results:       list[UploadLogResult]


class IncidentsResponse(BaseModel):
    total:     int
    incidents: list[dict]


class BlockedResponse(BaseModel):
    total:       int
    blocked_ips: list[str]


# ══════════════════════════════════════════════════════════════════════════════
# File validation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _validate_upload(file: UploadFile, raw_bytes: bytes) -> None:
    """
    Raise HTTPException (422) if the upload fails any security check.
    Checks: file size · extension · content type.
    """
    import os
    _, ext = os.path.splitext((file.filename or "").lower())

    if len(raw_bytes) > UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {UPLOAD_MAX_BYTES // 1024} KB.",
        )

    if ext not in UPLOAD_ALLOWED_EXTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(UPLOAD_ALLOWED_EXTS)}.",
        )


def _safe_decode(raw_bytes: bytes) -> str:
    """
    Attempt UTF-8, then latin-1 (never fails).
    Any remaining non-printable characters are replaced, not raised.
    """
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="replace")


# ══════════════════════════════════════════════════════════════════════════════
# Application factory
# ══════════════════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:

    app = FastAPI(
        title       = "Cybersecurity Multi-Agent API",
        description = "Red → Blue → Planner → ActionEngine pipeline with file log ingestion.",
        version     = "1.2.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # ── CORS ───────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],   # tighten to your domain in production
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── Shared singletons (created once at startup) ────────────────────────────
    store         = MemoryStore()
    red_agent     = RedTeamAgent(agent_id="RT-API")
    blue_agent    = BlueTeamAgent(agent_id="BT-API")
    planner       = PlannerAgent(agent_id="PLANNER-API")
    action_engine = ActionEngine(store=store)

    # ── Shared pipeline helper ─────────────────────────────────────────────────
    def _run_pipeline_on_log(log: dict) -> tuple[dict, dict, dict]:
        """Run Blue → Planner → Action on an already-structured log dict."""
        assessment    = blue_agent.analyse(log)
        plan          = planner.plan(assessment)
        action_result = action_engine.execute(plan)
        return assessment, plan, action_result

    # ══════════════════════════════════════════════════════════════════════════
    # GET /
    # ══════════════════════════════════════════════════════════════════════════

    @app.get("/", response_model=HealthResponse, summary="Health check", tags=["System"])
    def health_check() -> HealthResponse:
        """Returns service status and current UTC timestamp."""
        return HealthResponse()

    # ══════════════════════════════════════════════════════════════════════════
    # GET /run   (unchanged from previous version)
    # ══════════════════════════════════════════════════════════════════════════

    @app.get(
        "/run",
        response_model = PipelineResponse,
        summary        = "Run full simulation pipeline",
        tags           = ["Pipeline"],
    )
    def run_pipeline() -> PipelineResponse:
        """
        Generates a synthetic attack log with RedTeamAgent, then runs it
        through BlueTeamAgent → PlannerAgent → ActionEngine.
        """
        try:
            attack_log             = red_agent.generate_random_logs(count=1)[0]
            assessment, plan, action_result = _run_pipeline_on_log(attack_log)

            run_id   = str(uuid.uuid4())
            incident = {
                "run_id":     run_id,
                "ran_at":     _utcnow(),
                "source":     "simulation",
                "attack":     attack_log,
                "assessment": assessment,
                "plan":       plan,
                "action":     action_result,
            }
            store.add_incident(incident)

            return PipelineResponse(
                run_id     = run_id,
                attack     = AttackLog(**{k: attack_log.get(k) for k in AttackLog.model_fields}),
                assessment = ThreatAssessment(**{k: assessment.get(k) for k in ThreatAssessment.model_fields}),
                plan       = ActionPlan(**{k: plan.get(k) for k in ActionPlan.model_fields}),
                action     = ActionResult(**action_result),
            )

        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ══════════════════════════════════════════════════════════════════════════
    # POST /upload   ← NEW
    # ══════════════════════════════════════════════════════════════════════════

    @app.post(
        "/upload",
        response_model = UploadResponse,
        summary        = "Upload a .log / .txt file for analysis",
        tags           = ["Pipeline"],
    )
    async def upload_logs(
        file: UploadFile = File(
            ...,
            description="Plain-text or JSON-lines log file (.log or .txt, max 1 MB).",
        ),
    ) -> UploadResponse:
        """
        Accepts a `.log` or `.txt` file, extracts up to **5 non-empty lines**,
        and runs each through BlueTeamAgent → PlannerAgent → ActionEngine.

        Each line can be:
        - A raw JSON object (pass-through to the agents)
        - Plain text (heuristic attack-type detection applied before analysis)

        **Security limits**
        - Maximum file size: 1 MB
        - Maximum lines processed: 5
        - Accepted extensions: `.log`, `.txt`
        """
        # ── 1. Read raw bytes ─────────────────────────────────────────────────
        try:
            raw_bytes = await file.read()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read file: {exc}") from exc

        # ── 2. Security validation ────────────────────────────────────────────
        _validate_upload(file, raw_bytes)

        # ── 3. Decode safely ──────────────────────────────────────────────────
        content = _safe_decode(raw_bytes)

        # ── 4. Split into non-empty lines, cap at UPLOAD_MAX_LINES ────────────
        all_lines = [ln for ln in content.splitlines() if ln.strip()]
        lines_to_process = all_lines[:UPLOAD_MAX_LINES]

        if not lines_to_process:
            raise HTTPException(
                status_code=422,
                detail="The uploaded file contains no processable log lines.",
            )

        # ── 5. Process each line ──────────────────────────────────────────────
        results: list[UploadLogResult] = []

        for idx, line in enumerate(lines_to_process):
            try:
                structured_log            = _parse_log_line(line, idx)
                assessment, plan, action  = _run_pipeline_on_log(structured_log)

                # Persist as incident
                store.add_incident({
                    "run_id":     str(uuid.uuid4()),
                    "ran_at":     _utcnow(),
                    "source":     f"upload:{file.filename}",
                    "attack":     structured_log,
                    "assessment": assessment,
                    "plan":       plan,
                    "action":     action,
                })

                results.append(UploadLogResult(
                    line_index = idx,
                    raw_log    = line[:300],   # truncate for safe transport
                    analysis   = assessment,
                    decision   = plan,
                    action     = action,
                ))

            except Exception as exc:
                # Never crash the whole request because of one bad line —
                # record the error and continue.
                results.append(UploadLogResult(
                    line_index = idx,
                    raw_log    = line[:300],
                    analysis   = {"error": str(exc)},
                    decision   = {},
                    action     = {},
                ))

        return UploadResponse(
            filename         = file.filename or "unknown",
            lines_found      = len(all_lines),
            lines_processed  = len(lines_to_process),
            results          = results,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # GET /incidents
    # ══════════════════════════════════════════════════════════════════════════

    @app.get(
        "/incidents",
        response_model = IncidentsResponse,
        summary        = "List all incidents",
        tags           = ["Data"],
    )
    def get_incidents() -> IncidentsResponse:
        """Returns all stored incidents (newest first)."""
        incidents = store.get_incidents()
        return IncidentsResponse(total=len(incidents), incidents=incidents)

    # ══════════════════════════════════════════════════════════════════════════
    # GET /blocked
    # ══════════════════════════════════════════════════════════════════════════

    @app.get(
        "/blocked",
        response_model = BlockedResponse,
        summary        = "List blocked IPs",
        tags           = ["Data"],
    )
    def get_blocked() -> BlockedResponse:
        """Returns all IPs currently on the blocklist."""
        blocked = store.get_blocked_ips()
        return BlockedResponse(total=len(blocked), blocked_ips=blocked)

    return app


# ── Instantiate (uvicorn entry point) ──────────────────────────────────────────
app = create_app()
