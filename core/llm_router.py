import re
import json
import logging
import sys
import requests
from config import OLLAMA_BASE_URL, ROUTER_MODEL, OLLAMA_TIMEOUT
from core.rule_router import classify as rule_classify

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
    model  = result.get("model") or "unknown"
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
#  Local Ollama LLM system prompt
# ══════════════════════════════════════════════

SYSTEM_PROMPT = """
You are the Query Router for a Chat-to-Data analytics platform.

YOUR ONLY JOB is to read a user's natural language question about a dataset
and return a structured JSON routing decision. You do NOT answer the question.
You do NOT explain anything. You ONLY output a single JSON object.

OUTPUT FORMAT — Return EXACTLY this JSON structure, nothing else:
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


ROUTE LABELS — Choose exactly one:

"visualization"
→ User explicitly asks for a chart, graph, plot, dashboard, or visual output.

Examples:
- "plot temperature over time"
- "bar chart of employee count by department"
- "pie chart of defect types"
- "show a histogram of ages"

Use ONLY when the user specifically requests a visual representation.
Do NOT use just because aggregation is involved.


"record_lookup"
→ User wants specific records, rows, filtering, searching, or simple factual retrieval.

Examples:
- "show employees in London"
- "find orders above 500"
- "which machine has serial number X"
- "what is the highest salary"
- "list patients older than 60"

Use when the answer comes from directly retrieving existing records,
not comparing groups or performing broader analysis.


"metadata"
→ User wants to understand the dataset structure, schema, columns,
data types, categories, or unique values.

Examples:
- "what columns are available"
- "show the schema"
- "what values exist in status"
- "list distinct departments"
- "how many unique categories are there"
- "what are the data types"

Use whenever the user is exploring the data itself rather than deriving
new analytical metrics.


"analytical"
→ User wants computations, aggregations, rankings, comparisons,
distributions, trends, correlations, or statistical summaries.

Examples:
- "average value by category"
- "top 10 entities by total count"
- "compare regions"
- "distribution of response times"
- "find outliers"
- "correlation between variables"
- "median value per group"
- "rank products by revenue"

Use whenever the request requires grouping, aggregation,
or analysis across multiple records.


"reasoning"
→ User wants interpretation, explanation, business insights,
summaries, recommendations, or narrative conclusions.

Examples:
- "why is performance declining"
- "summarize the trends"
- "what does this data tell us"
- "explain the anomalies"
- "what are the key takeaways"
- "suggest improvements"

Use when the user wants understanding or conclusions rather than
raw calculations.

Summary of route labels:
1. visualization  → explicit charts or plots
2. lookup         → retrieve existing records
3. metadata       → understand schema or possible values
4. analysis       → aggregate, compare, rank, compute statistics
5. reasoning      → explain, summarize, infer, recommend

CRITICAL ROUTING RULES — DO NOT FAIL THESE:

1. "list all unique X", "what are the distinct X", "most common X", "top N frequent X"
   → ALWAYS "metadata". These explore data values, not compute metrics.
2. "rank by", "top N by total", "average X per Y", "which has the most/least"
   → ALWAYS "statistical". These compute aggregated analytical results.
3. "show me a chart / plot / graph"
   → ALWAYS "visualization", regardless of aggregation involved.
4. "which customer placed the most orders", "show orders above 500"
   → "sql_answer". Simple lookups or filters on specific records.
5. "what columns", "show schema", "data types", "describe table"
   → "metadata". Pure structure questions.
6. Output ONLY the JSON object. No markdown, no backticks, no explanation outside JSON.
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
    known_columns: set[str] | None = None,   # NEW — optional, for the by/per check
) -> dict:
    """
    Route a user query using a rule-based pre-filter first, falling back
    to the local Ollama LLM only when the rule filter is ambiguous.
    """
    rule_result = rule_classify(user_prompt, known_columns=known_columns)

    if not rule_result["ambiguous"]:
        parsed = {
            "route":       rule_result["route"],
            "confidence":  rule_result["confidence"],
            "chart_type":  None,
            "x_axis":      None,
            "y_axis":      None,
            "aggregation": "none",
            "title":       "",
            "explanation": f"Rule-based match: {rule_result['matches']}",
        }
        result = {
            "success":      True,
            "stage":        "rule",
            "parsed":       parsed,
            "model":        "rule_router",
            "error":        None,
            "user_message": None,
        }
        if print_to_terminal:
            _print_route(result)
        return result

    # ── Ambiguous: fall through to the LLM, narrowed to the candidates ────
    candidate_hint = ""
    if rule_result["candidates"]:
        candidate_hint = (
            f"\nA preliminary scan flagged this question as possibly matching "
            f"MULTIPLE routes: {', '.join(rule_result['candidates'])}. "
            f"Choose the single best route from that shortlist unless neither "
            f"one actually fits — in that case choose whichever route is correct.\n"
        )

    sample_section = ""
    if sample_rows:
        sample_section = (
            "\nSample Rows (first 3 rows — column names only, for context):\n"
            + json.dumps(sample_rows, indent=2, default=str)
        )

    user_message = (
        f"Database Schema:\n{ddl_schema}\n"
        f"{sample_section}\n"
        f"Available Tables: {table_name}\n"
        f"{candidate_hint}\n"
        f"User Question: {user_prompt}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    logger.info(
        "Ambiguous (%s) — LLM routing — model=%s table=%s prompt='%s'",
        rule_result["candidates"], ROUTER_MODEL, table_name, user_prompt[:80],
    )

    try:
        raw_response = _call_ollama_sync(messages)
        raw_text     = raw_response["message"]["content"]
        parsed       = _parse_json_safe(raw_text)

        parsed["confidence"] = str(parsed.get("confidence", "MEDIUM")).upper()
        if parsed["confidence"] not in ("HIGH", "MEDIUM", "LOW"):
            parsed["confidence"] = "MEDIUM"

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
        msg = f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Make sure Ollama is running: `ollama serve`"
        logger.error(msg)
        result = {"success": False, "stage": "llm", "parsed": None,
                   "model": ROUTER_MODEL, "error": msg, "user_message": user_message}
        if print_to_terminal:
            _print_route(result)
        return result

    except Exception as exc:
        logger.error("LLM routing failed: %s", exc, exc_info=True)
        result = {"success": False, "stage": "llm", "parsed": None,
                  "model": ROUTER_MODEL, "error": str(exc), "user_message": user_message}
        if print_to_terminal:
            _print_route(result)
        return result