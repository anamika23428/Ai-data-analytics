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

VALID route values: "visualization", "sql_answer", "metadata", "statistical", "reasoning"
Do NOT use any other route label (e.g. do NOT use "analytical", "lookup", "record_lookup").

ROUTE LABELS — Choose exactly one:

"visualization"
→ User explicitly asks for a chart, graph, plot, or visual output.
→ Use ONLY when the user specifically requests a visual. Do NOT use
  just because aggregation is involved.
Examples:
  "bar chart of sales by category"
  "pie chart of orders by city"
  "plot revenue over time"
  "show a histogram of ages"

"sql_answer"
→ User wants specific records, rows, or simple direct lookups.
→ Use when the answer is a direct retrieval — NOT a ranking or group comparison.
Examples:
  "show all orders above 500"
  "list customers from New York"
  "what is the email of customer 101"
  "show all completed sessions"

"metadata"
→ User wants to explore the dataset structure, schema, columns, data types,
  or what values/categories exist in the data.
→ Use whenever the user is asking WHAT EXISTS in the data, not computing metrics.
Examples:
  "what columns are in the customers table"
  "list all unique cities"
  "what are the distinct product categories"
  "how many unique products are there"
  "what values exist in the status column"
  "most common city"
  "top 5 frequent categories"
  "show the schema"
  "what data types are used"

"statistical"
→ User wants aggregations, rankings, comparisons across groups, or
  statistical analysis computed from the data.
→ Use when the request requires GROUP BY, ORDER BY aggregation, or stats.
Examples:
  "average order amount per category"
  "rank products by total quantity sold descending"
  "top 5 customers by total spend"
  "which product has the highest total revenue"
  "standard deviation of prices"
  "correlation between age and order amount"
  "distribution of order amounts"
  "compare onshore vs offshore headcount"
  "which city has the most orders"
  "total revenue by product"

"reasoning"
→ User wants a business explanation, summary, or narrative insight.
Examples:
  "why is revenue dropping"
  "summarize the sales trend"
  "what does this data tell us"
  "explain the pattern"
  "what are the key takeaways"

CRITICAL ROUTING RULES — DO NOT FAIL THESE:

1. "list all unique X", "what are the distinct X", "most common X", "top N frequent X"
   → ALWAYS "metadata". These explore existing data values, not compute new metrics.
2. "rank by", "top N by total", "average X per Y", "which has the most/least total"
   → ALWAYS "statistical". These compute aggregated analytical results.
3. "show me a chart / plot / bar chart / pie chart"
   → ALWAYS "visualization", regardless of aggregation involved.
4. "show orders above 500", "list customers from London", "find order by id"
   → ALWAYS "sql_answer". Direct record lookups or row filters.
5. "what columns", "show schema", "data types", "describe table"
   → ALWAYS "metadata". Pure structure questions.
6. NEVER output route labels like "analytical", "lookup", or "record_lookup" —
   they are NOT valid. Use only the five labels listed above.
7. Output ONLY the JSON object. No markdown, no backticks, no text outside JSON.
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