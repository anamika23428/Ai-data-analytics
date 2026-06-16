import re
import json
import logging
import sys
import requests
from config import OLLAMA_BASE_URL, ROUTER_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

# ── ANSI colour helpers (terminal output) ────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RED    = "\033[91m"
_DIM    = "\033[2m"

_ROUTE_COLOUR = {
    "visualization": "\033[94m",   # blue
    "sql_answer":    "\033[92m",   # green
    "metadata":      "\033[96m",   # cyan
    "statistical":   "\033[95m",   # magenta
    "reasoning":     "\033[93m",   # yellow
}

def _print_route(result: dict) -> None:
    """Pretty-print routing decision to stdout / terminal."""
    parsed = result.get("parsed") or {}
    route  = parsed.get("route", "unknown")
    conf   = parsed.get("confidence", "?")
    stage  = result.get("stage", "?")
    chart  = parsed.get("chart_type") or "—"
    title  = parsed.get("title") or ""
    expl   = parsed.get("explanation") or ""
    model  = result.get("model") or "keyword"
    ok     = result.get("success", False)

    col = _ROUTE_COLOUR.get(route, _RESET)

    if ok:
        print(
            f"\n{_BOLD}╔══ QUERY ROUTER ══════════════════════════════════╗{_RESET}\n"
            f"  {_BOLD}Route    :{_RESET} {col}{_BOLD}{route.upper():16}{_RESET}  "
            f"stage={_CYAN}{stage}{_RESET}  model={_DIM}{model}{_RESET}\n"
            f"  {_BOLD}Confidence:{_RESET} {_YELLOW}{conf}{_RESET}\n"
            f"  {_BOLD}Chart type:{_RESET} {chart}\n"
            f"  {_BOLD}Title     :{_RESET} {title}\n"
            f"  {_BOLD}Reasoning :{_RESET} {_DIM}{expl}{_RESET}\n"
            f"{_BOLD}╚══════════════════════════════════════════════════╝{_RESET}\n",
            flush=True,
        )
    else:
        err = result.get("error", "unknown error")
        print(
            f"\n{_RED}{_BOLD}╔══ QUERY ROUTER — FAILED ══════════════════════╗{_RESET}\n"
            f"  Error: {err}\n"
            f"{_RED}{_BOLD}╚════════════════════════════════════════════════╝{_RESET}\n",
            flush=True,
        )


# ══════════════════════════════════════════════
#  Stage 1 – Keyword pre-filter
# ══════════════════════════════════════════════

_KEYWORD_PATTERNS: dict[str, list[re.Pattern]] = {
    "metadata": [
        re.compile(r"\bhow many (columns?|fields?|rows?|records?)\b"),
        re.compile(r"\bgive (me )?(the )?(schema|columns?|fields?|structure)\b"),
        re.compile(r"\bwhat (columns?|fields?|types?|schema)\b"),
        re.compile(r"\blist (the )?(columns?|fields?|schema)\b"),
        re.compile(r"\bdescribe (the )?(table|dataset|data|schema)\b"),
        re.compile(r"\bwhat (are the |is the )?(data ?types?|column names?)\b"),
        re.compile(r"\bshow (me )?(the )?(schema|columns?|fields?|structure)\b"),
        re.compile(r"\bis there a .+ column\b"),
        re.compile(r"\bdo you have a .+ (column|field)\b"),
    ],
    "visualization": [
        re.compile(r"\b(bar|pie|line|scatter|heatmap|histogram)\s?(chart|plot|graph)\b"),
        re.compile(r"\b(plot|draw|graph|chart|visuali[sz]e|render)\b"),
        re.compile(r"\bshow (me )?(a |the )?(bar|pie|line|scatter|trend|breakdown)\b"),
        re.compile(r"\btrend (over|across|by|per)\b"),
        re.compile(r"\bdistribution of\b"),
        re.compile(r"\b(compare|breakdown|split)\b.{0,40}\b(by|across|between|per)\b"),
        # Aggregation word + a "grouping" word (by/per/across/for each) almost
        # always means "give me a breakdown" → best shown as a chart, e.g.
        # "show average salary by department", "total revenue per region".
        re.compile(
            r"\b(average|avg|total|sum|count|mean|median|max|maximum|min|minimum)\b"
            r".{0,40}\b(by|per|across|for each|grouped by)\b"
        ),
    ],
    "sql_answer": [
        re.compile(r"\b(total|sum|average|mean|maximum|minimum|max|min)\b(?!.*(columns?|fields?|rows?|records?|schema|types?))"),
        re.compile(r"\bhow (much|many) (?!.*(columns?|fields?|rows?|records?|schema))"),
        re.compile(r"\b(what is the|what was the|give me the) (?!.*(columns?|fields?|schema|types?))"),
        re.compile(r"\b(top|bottom|highest|lowest|best|worst) \d+\b"),
        re.compile(r"\b(who had|which (product|region|item|category|department|person))\b"),
        re.compile(r"\b(calculate|compute|find the|list all) (?!.*(columns?|fields?|schema))"),
        re.compile(r"\bcount of\b(?!.*(columns?|fields?|schema))"),
    ],
    "statistical": [
        re.compile(r"\b(outlier|anomal|unusual|abnormal)\b"),
        re.compile(r"\b(percentile|quartile|iqr|interquartile)\b"),
        re.compile(r"\b(standard deviation|std ?dev|variance|spread)\b"),
        re.compile(r"\b(correlation|z.?score|skewness|kurtosis)\b"),
        re.compile(r"\b(frequency distribution|histogram buckets?)\b"),
    ],
}


def _keyword_route(prompt: str) -> str | None:
    text = prompt.lower().strip()
    priority_order = ["metadata", "statistical", "visualization", "sql_answer"]
    for route in priority_order:
        for pattern in _KEYWORD_PATTERNS[route]:
            if pattern.search(text):
                logger.debug(
                    "Stage 1 keyword match — route='%s' pattern='%s'",
                    route, pattern.pattern,
                )
                return route
    return None


# ══════════════════════════════════════════════
#  Stage 2 – Local Ollama LLM system prompt
# ══════════════════════════════════════════════

SYSTEM_PROMPT = """
You are the Query Router for a Chat-to-Data analytics platform.

YOUR ONLY JOB is to read a user's natural language question about a dataset
and return a structured JSON routing decision. You do NOT answer the question.
You do NOT explain anything. You ONLY output a single JSON object.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — Return EXACTLY this JSON structure, nothing else:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "route":       "<route_label>",
  "confidence":  "<HIGH|MEDIUM|LOW>",
  "chart_type":  "<bar|pie|line|scatter|table|null>",
  "x_axis":      "<column name for x-axis, or null>",
  "y_axis":      "<column name for y-axis, or null>",
  "aggregation": "<sum|avg|count|min|max|none>",
  "title":       "<short descriptive title for the result>",
  "explanation": "<one sentence explaining your routing decision>"
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTE LABELS — Choose exactly one:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"visualization"  → User wants a chart, graph, or plot.
"sql_answer"     → User wants a specific number, name, list, or ranking.
"metadata"       → User asks about structure — columns, types, schema.
"statistical"    → User wants outlier detection, percentile, z-score, etc.
"reasoning"      → User wants explanation, summary, or business insight.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE LEVELS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIGH   → Question is very clear and unambiguous.
MEDIUM → Question is mostly clear but has minor ambiguity.
LOW    → Question is unclear or could belong to multiple routes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Output ONLY the JSON object. No text before. No text after. No markdown.
- Do NOT wrap the JSON in backticks or code blocks.
- Use ONLY column names from the schema — never invent column names.
- All string values must use double quotes.
- confidence must be exactly one of: HIGH, MEDIUM, LOW
"""


# ══════════════════════════════════════════════
#  Ollama caller
# ══════════════════════════════════════════════

def _call_ollama_sync(messages: list[dict]) -> dict:
    payload = {
        "model":    ROUTER_MODEL,
        "messages": messages,
        "stream":   False,
        "format":   "json",
        "options": {
            "temperature": 0.0,
            "num_predict": 256,
        },
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _parse_json_safe(text: str) -> dict:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*",     "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract valid JSON from LLM response. "
        f"First 300 chars: {text[:300]}"
    )


# ══════════════════════════════════════════════
#  Public entry point
# ══════════════════════════════════════════════

def route_query(
    user_prompt:  str,
    ddl_schema:   str,
    table_name:   str,
    sample_rows:  list[dict] | None = None,
    print_to_terminal: bool = True,
) -> dict:
    """
    Route a user query to the correct analytics handler.

    Two-stage pipeline:
      Stage 1 – Regex keyword match  (no LLM, instant, ~70% of queries)
      Stage 2 – Local Ollama LLM     (all data stays on your machine)

    Args:
        user_prompt:       The user's natural language question.
        ddl_schema:        Privacy-safe DDL string (schema only, no row data).
                           May contain MULTIPLE table definitions if more than
                           one file is loaded in this session.
        table_name:        Comma-separated names of all tables currently
                           loaded in this session.
        sample_rows:       Optional first-3 rows for LLM context (stage 2 only).
                           NOTE: only include if your data policy permits it.
        print_to_terminal: If True, pretty-prints the decision to stdout.

    Returns a dict with keys:
        success, stage, parsed (route/confidence/chart_type/…), model, error
    """

    # ── Stage 1: Keyword pre-filter ───────────
    keyword_route = _keyword_route(user_prompt)
    if keyword_route is not None:
        logger.info(
            "Stage 1 keyword match — route='%s' prompt='%s'",
            keyword_route, user_prompt[:80],
        )
        result = {
            "success": True,
            "stage":   "keyword",
            "parsed": {
                "route":       keyword_route,
                "confidence":  "HIGH",
                "chart_type":  None,
                "x_axis":      None,
                "y_axis":      None,
                "aggregation": "none",
                "title":       "",
                "explanation": "Resolved by keyword pre-filter — no LLM call.",
            },
            "model":        None,
            "error":        None,
            "user_message": "",
        }
        if print_to_terminal:
            _print_route(result)
        return result

    # ── Stage 2: Local Ollama LLM ─────────────
    # Only schema DDL (+ optional sample rows) is sent.
    # Raw data is never included. Ollama runs locally — nothing leaves your machine.
    sample_section = ""
    if sample_rows:
        sample_section = (
            "\nSample Rows (first 3 rows — column names only, for context):\n"
            + json.dumps(sample_rows, indent=2, default=str)
        )

    user_message = (
        f"Database Schema:\n{ddl_schema}\n"
        f"{sample_section}\n"
        f"Available Tables: {table_name}\n\n"
        f"User Question: {user_prompt}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    logger.info(
        "Stage 2 LLM routing — model=%s table=%s prompt='%s'",
        ROUTER_MODEL, table_name, user_prompt[:80],
    )

    try:
        raw_response = _call_ollama_sync(messages)
        raw_text     = raw_response["message"]["content"]
        parsed       = _parse_json_safe(raw_text)

        parsed["confidence"] = str(parsed.get("confidence", "MEDIUM")).upper()
        if parsed["confidence"] not in ("HIGH", "MEDIUM", "LOW"):
            parsed["confidence"] = "MEDIUM"

        logger.info(
            "Stage 2 routing decision — route=%s confidence=%s chart=%s",
            parsed.get("route"), parsed.get("confidence"), parsed.get("chart_type"),
        )

        result = {
            "success":      True,
            "stage":        "llm",
            "parsed":       parsed,
            "model":        ROUTER_MODEL,
            "error":        None,
            "user_message": user_message,
        }
        if print_to_terminal:
            _print_route(result)
        return result

    except requests.exceptions.ConnectionError:
        msg = (
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Make sure Ollama is running: `ollama serve`"
        )
        logger.error(msg)
        result = {
            "success": False, "stage": "llm", "parsed": None,
            "model": ROUTER_MODEL, "error": msg, "user_message": user_message,
        }
        if print_to_terminal:
            _print_route(result)
        return result

    except Exception as exc:
        logger.error("LLM routing failed: %s", exc, exc_info=True)
        result = {
            "success": False, "stage": "llm", "parsed": None,
            "model": ROUTER_MODEL, "error": str(exc), "user_message": user_message,
        }
        if print_to_terminal:
            _print_route(result)
        return result