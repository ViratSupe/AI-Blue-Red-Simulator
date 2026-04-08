"""
blue_team_agent.py
------------------
Blue Team Agent with a two-tier detection strategy:

  Tier 1 (Primary)  — Gemini AI analysis via google-generativeai
  Tier 2 (Fallback) — Rule-based detection (keyword + attack-type matching)

The fallback fires automatically whenever Gemini is:
  - unavailable (no API key, library not installed, quota exceeded)
  - too slow     (exceeds GEMINI_TIMEOUT_SECS)
  - returning an unparseable response

Every response — regardless of which tier produced it — is guaranteed to
match the canonical schema:
  threat_level · reason · recommended_action ·
  source_ip    · source_log_id · assessed_by · assessed_at

Install Tier 1 dependency (optional):
    pip install google-generativeai
"""

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv(override=True)

# ── Optional Gemini import ────────────────────────────────────────────────────
try:
    import google.generativeai as genai   # pip install google-generativeai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Configuration (override via environment variables) ────────────────────────
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL        = os.getenv("GEMINI_MODEL",   "gemini-2.5-flash")
GEMINI_TIMEOUT_SECS = int(os.getenv("GEMINI_TIMEOUT_SECS", "8"))


# ══════════════════════════════════════════════════════════════════════════════
# Tier 2 rule tables
# ══════════════════════════════════════════════════════════════════════════════

# Attack-type → assessment  (most precise: matched before keyword scan)
ATTACK_TYPE_RULES: dict[str, dict] = {
    "brute_force": {
        "threat_level":       "HIGH",
        "reason":             "Repeated login attempts detected. Credential stuffing or "
                              "dictionary attack in progress — account takeover risk.",
        "recommended_action": "block_ip",
    },
    "sql_injection": {
        "threat_level":       "HIGH",
        "reason":             "SQL injection payload identified. Attacker may be attempting "
                              "unauthorised data access or database manipulation.",
        "recommended_action": "block_ip",
    },
    "port_scan": {
        "threat_level":       "MEDIUM",
        "reason":             "Systematic port scan detected. Likely reconnaissance activity "
                              "to map open services before a follow-up attack.",
        "recommended_action": "monitor",
    },
}

# Keyword rules — evaluated in order; first match wins.
# Keywords are checked against a lowercased blob of all log field values.
KEYWORD_RULES: list[dict] = [
    {
        "keywords": [
            "failed", "unauthorized", "forbidden",
            "invalid credentials", "access denied", "brute",
        ],
        "threat_level":       "HIGH",
        "reason":             "Log contains indicators of failed authentication or "
                              "unauthorised access attempts.",
        "recommended_action": "block_ip",
    },
    {
        "keywords": ["inject", "select ", "union ", "drop table", "exec(", "script>", "xss", "payload"],
        "threat_level":       "HIGH",
        "reason":             "Log contains injection or exploit keywords. "
                              "Possible SQL injection or XSS attack.",
        "recommended_action": "block_ip",
    },
    {
        "keywords": ["scan", "port", "probe", "enumerate", "nmap", "masscan"],
        "threat_level":       "MEDIUM",
        "reason":             "Log contains reconnaissance keywords (scan / port). "
                              "Possible network-mapping activity.",
        "recommended_action": "monitor",
    },
]

# Default when no rule matches
FALLBACK_DEFAULT: dict = {
    "threat_level":       "LOW",
    "reason":             "No matching threat signature found. Log appears benign or unrecognised.",
    "recommended_action": "ignore",
}


# ══════════════════════════════════════════════════════════════════════════════
# Gemini prompt
# ══════════════════════════════════════════════════════════════════════════════

_GEMINI_SYSTEM_PROMPT = """
You are a cybersecurity threat analyst (Blue Team).
Analyse the provided JSON attack log and return ONLY a valid JSON object —
no markdown fences, no extra text — with exactly these keys:

{
  "threat_level":       "HIGH" | "MEDIUM" | "LOW",
  "reason":             "<one or two sentence explanation>",
  "recommended_action": "block_ip" | "monitor" | "ignore"
}

Rules:
- brute_force or sql_injection → HIGH, block_ip
- port_scan, reconnaissance    → MEDIUM, monitor
- benign / unrecognised        → LOW, ignore
- Base your reasoning on the full log content, not just attack_type.
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# Blue Team Agent
# ══════════════════════════════════════════════════════════════════════════════

class BlueTeamAgent:
    """
    Two-tier threat detection agent.

    Primary  : Gemini AI  (requires GEMINI_API_KEY env var + google-generativeai)
    Fallback : Rule-based (always available, zero extra dependencies)

    Usage:
        agent = BlueTeamAgent()
        result = agent.analyse(log_dict)
    """

    def __init__(self, agent_id: str = "BT-01"):
        self.agent_id = agent_id
        self._history: list[dict] = []
        self._gemini_model = self._init_gemini()

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(self, log: dict | str) -> dict:
        """
        Analyse a single log entry and return a threat assessment.

        Tries Gemini first; falls back to rule-based detection on any failure.
        The returned dict always matches the canonical schema.
        """
        if isinstance(log, str):
            log = self._parse_json(log)

        self._validate(log)

        # ── Tier 1: Gemini ────────────────────────────────────────────────────
        detection_source = "gemini"
        core = self._detect_with_gemini(log) if self._gemini_model else None

        # ── Tier 2: rule-based fallback ───────────────────────────────────────
        if core is None:
            detection_source = "rule-based-fallback"
            core = self._detect_with_rules(log)

        # ── Context modifiers (enrich reason with concrete log data) ──────────
        core = self._apply_context_modifiers(core, log)

        # ── Build canonical response ──────────────────────────────────────────
        assessment = {
            "threat_level":       core["threat_level"],
            "reason":             core["reason"],
            "recommended_action": core["recommended_action"],
            "source_log_id":      log.get("log_id",    "unknown"),
            "source_ip":          log.get("source_ip", "unknown"),
            "assessed_by":        self.agent_id,
            "assessed_at":        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "detection_source":   detection_source,
        }

        self._history.append(assessment)
        return assessment

    def analyse_batch(self, logs: list[dict]) -> list[dict]:
        """Analyse a list of logs and return a list of assessments."""
        return [self.analyse(log) for log in logs]

    def summary(self) -> dict:
        """Session-level summary including detection-source breakdown."""
        if not self._history:
            return {"message": "No assessments recorded yet."}

        threat_counts  = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        source_counts  = {"gemini": 0, "rule-based-fallback": 0}

        for entry in self._history:
            lvl = entry.get("threat_level", "LOW")
            threat_counts[lvl] = threat_counts.get(lvl, 0) + 1
            src = entry.get("detection_source", "rule-based-fallback")
            source_counts[src] = source_counts.get(src, 0) + 1

        return {
            "agent_id":          self.agent_id,
            "total_assessed":    len(self._history),
            "threat_breakdown":  threat_counts,
            "detection_sources": source_counts,
            "high_threat_ips":   self._high_threat_ips(),
        }

    # ── Gemini initialisation ─────────────────────────────────────────────────

    def _init_gemini(self):
        """Return a configured Gemini model, or None if unavailable."""
        if not _GEMINI_AVAILABLE:
            logger.info("BlueTeamAgent: google-generativeai not installed — using fallback.")
            return None
        if not GEMINI_API_KEY:
            logger.info("BlueTeamAgent: GEMINI_API_KEY not set — using fallback.")
            return None
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=_GEMINI_SYSTEM_PROMPT,
            )
            logger.info("BlueTeamAgent: Gemini ready (model=%s).", GEMINI_MODEL)
            return model
        except Exception as exc:
            logger.warning("BlueTeamAgent: Gemini init failed (%s) — using fallback.", exc)
            return None

    # ── Tier 1: Gemini detection ──────────────────────────────────────────────

    def _detect_with_gemini(self, log: dict) -> dict | None:
        """
        Call Gemini with a timeout enforced via a background thread.
        Returns None on timeout, network error, or bad JSON so the caller
        can seamlessly fall back to rule-based detection.
        """
        result_holder: list[dict | None] = [None]
        error_holder:  list[str]         = []

        def _call():
            try:
                prompt   = f"Analyse this attack log:\n\n{json.dumps(log, indent=2)}"
                response = self._gemini_model.generate_content(prompt)
                raw      = response.text.strip()

                # Strip markdown fences Gemini sometimes wraps JSON in
                raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
                raw = re.sub(r"\n?```$",           "", raw).strip()

                parsed = json.loads(raw)
                self._validate_gemini_response(parsed)
                result_holder[0] = parsed

            except json.JSONDecodeError as exc:
                error_holder.append(f"JSON parse error: {exc}")
            except KeyError as exc:
                error_holder.append(f"Missing key in response: {exc}")
            except Exception as exc:
                error_holder.append(f"API error: {exc}")

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        thread.join(timeout=GEMINI_TIMEOUT_SECS)

        if thread.is_alive():
            logger.warning(
                "BlueTeamAgent: Gemini timed out after %ds — switching to fallback.",
                GEMINI_TIMEOUT_SECS,
            )
            return None

        if error_holder:
            logger.warning("BlueTeamAgent: Gemini failed (%s) — switching to fallback.",
                           error_holder[0])
            return None

        return result_holder[0]

    # ── Tier 2: Rule-based detection ──────────────────────────────────────────

    def _detect_with_rules(self, log: dict) -> dict:
        """
        Rule-based detection — always succeeds, never raises.

        Priority:
          1. Exact attack_type match  (most specific)
          2. Keyword scan over all log values
          3. Default LOW
        """
        # 1. Attack-type lookup
        attack_type = log.get("attack_type", "").lower().strip()
        if attack_type in ATTACK_TYPE_RULES:
            logger.debug("BlueTeamAgent: fallback matched attack_type='%s'.", attack_type)
            return ATTACK_TYPE_RULES[attack_type].copy()

        # 2. Keyword scan
        log_text = self._log_to_searchable_text(log)
        for rule in KEYWORD_RULES:
            if any(kw in log_text for kw in rule["keywords"]):
                matched_kw = next(kw for kw in rule["keywords"] if kw in log_text)
                logger.debug("BlueTeamAgent: fallback keyword match='%s'.", matched_kw)
                return {
                    "threat_level":       rule["threat_level"],
                    "reason":             rule["reason"],
                    "recommended_action": rule["recommended_action"],
                }

        # 3. Default
        logger.debug("BlueTeamAgent: fallback — no rule matched, returning LOW.")
        return FALLBACK_DEFAULT.copy()

    # ── Context modifiers ─────────────────────────────────────────────────────

    @staticmethod
    def _apply_context_modifiers(core: dict, log: dict) -> dict:
        """
        Append concrete log details to the reason string.
        Works on a copy — never mutates rule tables.
        """
        result      = core.copy()
        attack_type = log.get("attack_type", "").lower()

        if attack_type == "brute_force":
            attempts = log.get("attempts", 0)
            if attempts > 200:
                result["reason"] += (
                    f" ({attempts} attempts recorded — high-volume attack, "
                    "immediate block strongly advised.)"
                )

        if attack_type == "port_scan":
            open_ports = log.get("open_ports") or []
            if open_ports:
                result["reason"] += (
                    f" Open ports discovered: {open_ports}. "
                    "Attacker now has a confirmed target surface map."
                )

        return result

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self, log: dict) -> None:
        required = {"log_id", "attack_type", "source_ip", "timestamp"}
        missing  = required - log.keys()
        if missing:
            raise ValueError(f"Log is missing required fields: {missing}")

    @staticmethod
    def _validate_gemini_response(parsed: dict) -> None:
        required = {"threat_level", "reason", "recommended_action"}
        missing  = required - parsed.keys()
        if missing:
            raise KeyError(f"Gemini response missing keys: {missing}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON input: {exc}") from exc

    @staticmethod
    def _log_to_searchable_text(log: dict) -> str:
        """Flatten all scalar + nested log values into one lowercase string."""
        parts = []
        for v in log.values():
            if isinstance(v, (str, int, float)):
                parts.append(str(v))
            elif isinstance(v, (list, dict)):
                parts.append(json.dumps(v))
        return " ".join(parts).lower()

    def _high_threat_ips(self) -> list[str]:
        return list({
            e["source_ip"] for e in self._history
            if e.get("threat_level") == "HIGH"
        })


# ══════════════════════════════════════════════════════════════════════════════
# Pretty printer
# ══════════════════════════════════════════════════════════════════════════════

def print_assessment(a: dict) -> None:
    level  = a["threat_level"]
    source = a.get("detection_source", "?")
    colour = {"HIGH": "\033[91m", "MEDIUM": "\033[93m", "LOW": "\033[92m"}.get(level, "")
    cyan   = "\033[96m"
    reset  = "\033[0m"
    print(f"\n{colour}{'─' * 60}")
    print(f"  THREAT LEVEL : {level}   {cyan}[via {source}]{reset}{colour}")
    print(f"{'─' * 60}{reset}")
    print(json.dumps(a, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# Example usage / self-test
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    sample_logs = [
        # ① Known attack_type → attack-type rule
        {
            "log_id": "0001", "agent_id": "RT-ALPHA",
            "timestamp": "2025-04-05T14:22:01Z",
            "source_ip": "185.44.23.101", "target_ip": "10.0.0.5",
            "attack_type": "brute_force", "payload": "admin:password123",
            "status": "failed", "attempts": 342,
        },
        # ② Known attack_type → attack-type rule
        {
            "log_id": "0002", "agent_id": "RT-ALPHA",
            "timestamp": "2025-04-05T14:25:18Z",
            "source_ip": "91.120.55.18", "target_ip": "10.0.0.5",
            "attack_type": "sql_injection",
            "payload": "' UNION SELECT username, password FROM users --",
            "http_method": "POST", "response_code": 500,
        },
        # ③ Known attack_type → rule + context modifier (open ports)
        {
            "log_id": "0003", "agent_id": "RT-ALPHA",
            "timestamp": "2025-04-05T14:30:44Z",
            "source_ip": "198.0.22.7", "target_ip": "10.0.0.5",
            "attack_type": "port_scan", "payload": None,
            "ports_scanned": [22, 80, 443, 3306], "open_ports": [80, 443],
        },
        # ④ Unknown attack_type → keyword rule fires on "unauthorized"
        {
            "log_id": "0004", "agent_id": "RT-ALPHA",
            "timestamp": "2025-04-05T14:35:00Z",
            "source_ip": "203.55.12.9", "target_ip": "10.0.0.5",
            "attack_type": "unknown", "status": "unauthorized", "payload": None,
        },
        # ⑤ Unknown attack_type → keyword rule fires on "scan"
        {
            "log_id": "0005", "agent_id": "RT-ALPHA",
            "timestamp": "2025-04-05T14:40:00Z",
            "source_ip": "77.33.10.2", "target_ip": "10.0.0.5",
            "attack_type": "recon", "message": "host scan detected", "payload": None,
        },
        # ⑥ Benign → default LOW
        {
            "log_id": "0006", "agent_id": "RT-ALPHA",
            "timestamp": "2025-04-05T14:45:00Z",
            "source_ip": "10.0.0.1", "target_ip": "10.0.0.5",
            "attack_type": "healthcheck", "payload": None,
        },
    ]

    print("\n" + "=" * 60)
    print("  Blue Team Agent — Dual-Tier Detection Demo")
    print("=" * 60)
    print("  (No GEMINI_API_KEY → all detections via rule-based fallback)\n")

    agent = BlueTeamAgent(agent_id="BT-SENTINEL")

    for log in sample_logs:
        result = agent.analyse(log)
        print_assessment(result)

    print("\n\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(json.dumps(agent.summary(), indent=2))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()