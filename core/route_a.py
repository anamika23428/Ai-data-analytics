# ─────────────────────────────────────────────────────────────────────────────
# core/route_a.py  –  Route A: Visualization Pipeline
#
# Full pipeline:
#   INTENT EXTRACTION  →  SQL GENERATION  →  3-LAYER VALIDATION
#   →  DUCKDB EXECUTION  →  VISUALIZATION BUILDER (Plotly)
#
# 100% local Ollama — zero data leaves your machine.
# Only DDL schema / column names are sent to the model, never raw row data.
#
# Models (configurable in config.py):
#   INTENT_MODEL  = llama3.2:3b        (intent extraction)
#   SQL_MODEL     = qwen2.5-coder:7b   (SQL generation)
#
# Key improvements over original:
#   • Both _extract_intent() and _generate_sql() now inject a live COLUMN
#     DICTIONARY built from DuckDB DESCRIBE — model can no longer hallucinate
#     column names that don't exist.
#   • num_predict raised 300→400 (intent) and 512→1024 (SQL) so responses
#     are never truncated on complex queries.
#   • _generate_sql() includes a self-repair pass: if DuckDB EXPLAIN fails,
#     the error is sent back to qwen2.5-coder for a one-shot local fix.
#   • conn is now threaded through to both functions so they can call DESCRIBE.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

from config import (
    DDL_MAX_COLUMNS,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
)
from core.ddl_utils import generate_privacy_safe_ddl, generate_multi_table_ddl

# ── Reuse the SAME SQL engine internals as core/sql_engine.py ─────────────────
# (delimiter-aware column dictionary builder + the TRY_CAST / decimal-strip
#  hardening safety nets) so Route A and Route B never drift out of sync.
from core.sql_engine import (
    _build_column_dict as _se_build_column_dict,
    _harden_sql,
)

logger = logging.getLogger(__name__)

# ── ANSI colours ──────────────────────────────────────────────────────────────
_R    = "\033[0m"
_BOLD = "\033[1m"
_BLUE = "\033[94m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED  = "\033[91m"
_DIM  = "\033[2m"
_CYAN = "\033[96m"


# ══════════════════════════════════════════════════════════════════════════════
#  Config — model names (override in config.py if needed)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from config import INTENT_MODEL
except ImportError:
    INTENT_MODEL = "llama3.2:3b"

try:
    from config import SQL_MODEL
except ImportError:
    SQL_MODEL = "qwen2.5-coder:1.5b"  


def _qident(name: str) -> str:
    """Return a safely double-quoted DuckDB identifier (handles spaces & special chars)."""
    return '"' + name.replace('"', '""')+'"'  


# ══════════════════════════════════════════════════════════════════════════════
#  Result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RouteAResult:
    success: bool
    intent: dict                  = field(default_factory=dict)
    sql: str | None               = None
    df: pd.DataFrame | None       = None
    fig: go.Figure | None         = None
    validation_log: list[str]     = field(default_factory=list)
    error: str | None             = None
    stage_reached: str            = "none"  # intent|sql|validation|execution|visualization


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helper — build a column dictionary from live DuckDB metadata
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  Shared helper — build a column dictionary from live DuckDB metadata
#
#  NOTE: this now delegates straight to core.sql_engine._build_column_dict so
#  Route A sees the exact same column dictionary as Route B — including the
#  [DELIMITED with 'X'] flags for pipe/semicolon-joined VARCHAR columns.
# ══════════════════════════════════════════════════════════════════════════════

def _build_column_dict(conn, tables: list[str]) -> str:
    return _se_build_column_dict(conn, tables)


# ══════════════════════════════════════════════════════════════════════════════
#  Layer 1 – Intent Extraction
# ══════════════════════════════════════════════════════════════════════════════

_INTENT_SYSTEM = """\
You are an intent extractor for a data analytics platform.
Given a user question and a COLUMN DICTIONARY, extract visualization intent.

Output ONLY a single JSON object — no text, no markdown, no backticks.

{
  "table":       "<name of the table that best answers the question>",
  "chart_type":  "<bar|pie|donut|line|area|scatter|histogram|heatmap|box|funnel|treemap>",
  "x_axis":      "<column name exactly as in COLUMN DICTIONARY, or null>",
  "y_axis":      "<column name exactly as in COLUMN DICTIONARY, or null>",
  "aggregation": "<sum|avg|count|min|max|none>",
  "group_by":    "<column name exactly as in COLUMN DICTIONARY, or null>",
  "filters":     "<plain English filter description or null>",
  "order_by":    "<column name exactly as in COLUMN DICTIONARY, or null>",
  "limit":       <integer or null>,
  "title":       "<short chart title>",
  "x_label":     "<x axis label>",
  "y_label":     "<y axis label>"
}

Rules:
- "table" MUST be one of the table names given in the schema. Pick the table
  whose columns best match the question. If the question needs columns from
  more than one table, pick the table with the most relevant columns and the
  SQL writer will add JOINs as needed.
- Use ONLY column names that exist in the schema. Never invent columns.
- chart_type must be one of: bar, pie, donut, line, area, scatter, histogram, heatmap, box, funnel, treemap
- If the user explicitly names a chart/graph type in their question (e.g. "pie chart",
  "line graph", "scatter plot", "box plot", "donut chart", "area chart", "treemap",
  "funnel"), you MUST use that exact chart_type — never substitute "bar" instead.
- If the user just says "graph" or "chart" or "visualize" with no specific type named,
  pick whichever chart_type best fits the data and question (e.g. a trend over time → line,
  a part-to-whole comparison → pie, a distribution → histogram).
- aggregation must be one of: sum, avg, count, min, max, none
- All string values must use double quotes.
- null values must be JSON null (not the string "null").
- Column names in the JSON MUST match the schema exactly, including spaces
  (e.g. use "total joining count" not "total_joining_count").
- If a column name contains spaces, still write it exactly as in the schema.
"""

_INTENT_USER = """\
COLUMN DICTIONARY (use ONLY these exact table and column names):
{column_dict}

Available tables: {tables}

User question: {question}
"""


# Order matters: more specific terms (e.g. "donut") are checked before the
# generic ones they could otherwise be mistaken for (e.g. "pie").
_CHART_KEYWORDS: list[tuple[str, str]] = [
    (r"\bdonut\b", "donut"),
    (r"\bpie\s*(chart|graph)?\b", "pie"),
    (r"\btreemap\b", "treemap"),
    (r"\bfunnel\b", "funnel"),
    (r"\bbox\s*(plot|chart)?\b", "box"),
    (r"\bheatmap\b", "heatmap"),
    (r"\bhistogram\b", "histogram"),
    (r"\bscatter\s*(plot|chart)?\b", "scatter"),
    (r"\barea\s*(chart|graph)?\b", "area"),
    (r"\bline\s*(chart|graph)?\b", "line"),
    (r"\bbar\s*(chart|graph)?\b", "bar"),
]


def _detect_explicit_chart_type(question: str) -> str | None:
    """
    Safety net for when a user explicitly names a chart/graph type
    ("show me a pie chart of ...", "line graph of ...", "scatter plot
    between ..."). The small local intent model sometimes ignores this
    and defaults to 'bar' — this regex check catches it and always wins.
    """
    q = question.lower()
    for pattern, chart_type in _CHART_KEYWORDS:
        if re.search(pattern, q):
            return chart_type
    return None


# Some small local models put a computed expression like "average(quantity)"
# or "AVG(quantity)" directly into x_axis/y_axis/group_by/order_by instead of
# the plain column name "quantity" + aggregation:"avg". That breaks Layer 1
# validation (the expression isn't a real column) even though the SQL itself
# may be fine. This strips the wrapper and recovers the real column + agg.
_AGG_WRAP_RE = re.compile(
    r'^\s*(avg|average|sum|total|count|min|minimum|max|maximum)\s*\(\s*"?'
    r'([^()"]+?)"?\s*\)\s*$',
    re.IGNORECASE,
)
_AGG_WORD_TO_CODE = {
    "avg": "avg", "average": "avg",
    "sum": "sum", "total": "sum",
    "count": "count",
    "min": "min", "minimum": "min",
    "max": "max", "maximum": "max",
}


def _unwrap_agg_column(col: str | None) -> tuple[str | None, str | None]:
    """Return (base_column_name, inferred_aggregation) — aggregation is
    None if `col` wasn't wrapped in a function call."""
    if not col:
        return col, None
    m = _AGG_WRAP_RE.match(col)
    if not m:
        return col, None
    return m.group(2).strip(), _AGG_WORD_TO_CODE.get(m.group(1).lower())


def _extract_intent(ddl: str, tables: list[str], question: str, conn=None) -> dict:
    """
    Stage 1: Call Ollama to extract chart intent from the user question.

    Now accepts an optional `conn` (DuckDB connection) to build a live
    column dictionary — prevents llama3.2:3b from hallucinating column names.
    Falls back to DDL text if conn is not available.
    """
    # Build column dict from live DuckDB metadata when possible
    if conn and tables:
        column_dict = _build_column_dict(conn, tables)
    else:
        column_dict = f"(DDL schema — use column names from this only):\n{ddl}"

    user_msg = _INTENT_USER.format(
        column_dict=column_dict,
        tables=", ".join(tables),
        question=question,
    )
    messages = [
        {"role": "system", "content": _INTENT_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]

    payload = {
        "model":    INTENT_MODEL,
        "messages": messages,
        "stream":   False,
        "format":   "json",
        "options":  {"temperature": 0.0, "num_predict": 400},  # was 300
    }

    print(
        f"\n{_BOLD}{_BLUE}┌── Route A · Stage 1: Intent Extraction{_R}\n"
        f"   model={_DIM}{INTENT_MODEL}{_R}  tables={_CYAN}{', '.join(tables)}{_R}",
        flush=True,
    )

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]

    # Strip markdown fences if model adds them
    raw = re.sub(r"```json\s*|```", "", raw).strip()
    intent = json.loads(raw)

    # Normalise chart_type and aggregation
    intent["chart_type"]  = str(intent.get("chart_type",  "bar")).lower()
    intent["aggregation"] = str(intent.get("aggregation", "none")).lower()
    _VALID_CHART_TYPES = (
        "bar", "pie", "donut", "line", "area", "scatter",
        "histogram", "heatmap", "box", "funnel", "treemap",
    )
    if intent["chart_type"] not in _VALID_CHART_TYPES:
        intent["chart_type"] = "bar"
    if intent["aggregation"] not in ("sum", "avg", "count", "min", "max", "none"):
        intent["aggregation"] = "none"

    # Unwrap any axis the model expressed as a function call, e.g.
    # y_axis: "average(quantity)" -> y_axis: "quantity", aggregation: "avg"
    for axis_key in ("x_axis", "y_axis", "group_by", "order_by"):
        base, inferred_agg = _unwrap_agg_column(intent.get(axis_key))
        if base != intent.get(axis_key):
            print(
                f"   {_YELLOW}Unwrapped {axis_key} '{intent.get(axis_key)}' -> '{base}'{_R}",
                flush=True,
            )
            intent[axis_key] = base
            if inferred_agg and intent["aggregation"] == "none":
                intent["aggregation"] = inferred_agg

    # If the user explicitly named a chart/graph type in plain English,
    # that always wins over the model's guess (handles "any kind of graph").
    explicit_chart = _detect_explicit_chart_type(question)
    if explicit_chart:
        intent["chart_type"] = explicit_chart

    # Resolve / validate the chosen table against the actual list of tables
    chosen_table = str(intent.get("table") or "").strip()
    table_lookup = {t.lower(): t for t in tables}
    if chosen_table.lower() in table_lookup:
        intent["table"] = table_lookup[chosen_table.lower()]
    else:
        # Model didn't pick a valid table — default to the first one
        intent["table"] = tables[0] if tables else chosen_table

    print(
        f"   table={_CYAN}{intent['table']}{_R}  "
        f"chart={_GREEN}{intent['chart_type']}{_R}  "
        f"x={intent.get('x_axis')}  y={intent.get('y_axis')}  "
        f"agg={intent.get('aggregation')}\n"
        f"{_BOLD}{_BLUE}└──{_R}",
        flush=True,
    )
    return intent


# ══════════════════════════════════════════════════════════════════════════════
#  Layer 2 – SQL Generation
# ══════════════════════════════════════════════════════════════════════════════

_SQL_SYSTEM = """\
You are an expert DuckDB SQL writer for data visualization queries.
Given a schema (which may contain MULTIPLE tables), a primary table name,
an intent JSON, and a user question, write a single DuckDB SELECT query
that retrieves the data needed for the chart.

Rules:
- Output ONLY the SQL query. No explanation, no markdown fences, no preamble.
- Use only SELECT or WITH. Never INSERT, UPDATE, DELETE, DROP, CREATE, ALTER.
- Reference only tables and columns from the provided schema.
- Prefer the \"primary table\" given below. If the question requires columns
  from another table in the schema, you may JOIN to that table using a
  sensible shared column (e.g. matching id/key column names).
- ALWAYS double-quote every column name and table name — especially those
  containing spaces (e.g. \"total joining count\", \"session date\").
- End with a semicolon.
- Apply aggregations and GROUP BY as indicated by the intent.
- ALWAYS alias an aggregated column with the PLAIN column name from the intent
  (e.g. AVG("quantity") AS "quantity"), never a function-call-shaped alias
  like AVG("quantity") AS "average(quantity)". The chart builder matches
  output columns against the intent's x_axis/y_axis by plain name.
- Apply ORDER BY and LIMIT as indicated by the intent.
- Keep the query minimal — only fetch columns needed for the chart.
- PIPE-DELIMITED COLUMNS: if a column stores multiple values separated by '|'
  (e.g. category), use UNNEST(STRING_SPLIT(col, '|')) to expand them before
  grouping or counting.

DUCKDB STRING CONVERSION RULES (CRITICAL):
- To convert currency/price strings to numbers, remove symbols first: TRY_CAST(REPLACE(REPLACE(col, '₹', ''), ',', '') AS DOUBLE).
- To convert percentage strings to numbers: TRY_CAST(REPLACE(col, '%', '') AS DOUBLE).
- To convert comma-separated number strings (like rating_count) to numbers: TRY_CAST(REPLACE(col, ',', '') AS DOUBLE).
- NEVER remove or replace decimal points ('.') when cleaning numeric columns like ratings or prices, as this alters the underlying numerical value.
  WRONG: REPLACE(rating, '.', '') turns "4.5" into "45" — this corrupts the value. Do not do this.
- ALWAYS use TRY_CAST, never plain CAST, when converting any text/VARCHAR column.
- Never include the '%' symbol in mathematical comparisons in the WHERE/HAVING clause (e.g., use > 50 instead of > 50%).
- ALWAYS clean and TRY_CAST string columns to DOUBLE *before* doing math, filtering in WHERE/HAVING clauses, or sorting in ORDER BY.
- If the COLUMN DICTIONARY marks a column as [DELIMITED with 'X'], ALWAYS expand it using
  UNNEST(STRING_SPLIT(col, 'X')) before SELECT DISTINCT, COUNT, GROUP BY, or filtering.
- LIMIT always goes after ORDER BY.
"""

_SQL_USER = """\
COLUMN DICTIONARY (use ONLY these exact column names):
{column_dict}

Primary table: {table}
Available tables: {tables}

Visualization intent:
{intent_json}

User question: {question}

SQL query:
"""

_SQL_REPAIR = """\
The SQL query you wrote caused this DuckDB error:

  {error}

Here is the broken query:
{sql}

Write a corrected DuckDB SQL query that fixes this error exactly.
CRITICAL: If the error says "Referenced column not found" and provides "Candidate bindings", you MUST replace your hallucinated column name with one of the exact Candidate bindings provided in the error message!
Output ONLY the raw SQL, nothing else.
"""


def _extract_sql_from_response(raw: str) -> str | None:
    """Pull the first SELECT/WITH block out of a model response."""
    raw = re.sub(r"```sql\s*|```", "", raw, flags=re.I).strip()
    m = re.search(r"(SELECT|WITH)\b[\s\S]*", raw, flags=re.I)
    if not m:
        return None
    sql = m.group(0).strip()
    if ";" in sql:
        sql = sql[:sql.index(";") + 1]
    elif not sql.endswith(";"):
        sql += ";"
    return sql


def _generate_sql(
    ddl: str,
    table: str,
    tables: list[str],
    intent: dict,
    question: str,
    conn=None,
) -> str | None:
    """
    Stage 2: Call Ollama (SQL model) to generate a DuckDB SELECT query.

    Now accepts an optional `conn` to build a live column dictionary,
    uses num_predict=1024 (was 512), and includes a self-repair pass
    on DuckDB EXPLAIN failure.
    """
    # Build column dict from live DuckDB metadata when possible
    if conn and tables:
        column_dict = _build_column_dict(conn, tables)
    else:
        column_dict = f"(DDL schema — use column names from this only):\n{ddl}"

    user_msg = _SQL_USER.format(
        column_dict=column_dict,
        table=table,
        tables=", ".join(tables),
        intent_json=json.dumps(intent, indent=2),
        question=question,
    )
    messages = [
        {"role": "system", "content": _SQL_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]

    payload = {
        "model":    SQL_MODEL,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": 0.0, "num_predict": 1024, "num_ctx": 8192},
    }

    print(
        f"\n{_BOLD}{_BLUE}┌── Route A · Stage 2: SQL Generation{_R}\n"
        f"   model={_DIM}{SQL_MODEL}{_R}",
        flush=True,
    )

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]
    sql = _extract_sql_from_response(raw)

    if not sql:
        print(f"   {_RED}No valid SQL extracted{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
        return None

    # Same regex safety nets as sql_engine.py: TRY_CAST enforcement +
    # decimal-point-stripping fix, so both routes harden SQL identically.
    sql = _harden_sql(sql)
    logger.info("SQL first attempt: %s…", sql[:100])

    # ── Self-repair pass (3 attempts — same as sql_engine.py) ─────────────────
    if conn:
        max_repairs = 3
        repair_messages = list(messages)

        for attempt in range(1, max_repairs + 1):
            try:
                conn.execute(f"EXPLAIN {sql}")
                if attempt > 1:
                    print(f"   {_GREEN}Self-repair succeeded (Attempt {attempt}/{max_repairs}){_R}", flush=True)
                    logger.info("Self-repair succeeded: %s…", sql[:100])
                break
            except Exception as explain_err:
                if attempt == max_repairs:
                    print(
                        f"   {_RED}Self-repair failed after {max_repairs} attempts ({explain_err}) — "
                        f"passing original to validator{_R}",
                        flush=True,
                    )
                    logger.warning("Self-repair exhausted (%s)", explain_err)
                    break

                print(
                    f"   {_YELLOW}SQL failed EXPLAIN ({explain_err}) — "
                    f"self-repair attempt {attempt}/{max_repairs}…{_R}",
                    flush=True,
                )
                repair_msg = _SQL_REPAIR.format(error=str(explain_err), sql=sql)
                repair_messages.append({"role": "assistant", "content": sql})
                repair_messages.append({"role": "user", "content": repair_msg})

                try:
                    repair_payload = {**payload, "messages": repair_messages}
                    repair_resp = requests.post(
                        f"{OLLAMA_BASE_URL}/api/chat",
                        json=repair_payload,
                        timeout=OLLAMA_TIMEOUT,
                    )
                    repair_resp.raise_for_status()
                    repaired = _extract_sql_from_response(repair_resp.json()["message"]["content"])
                    if repaired:
                        sql = _harden_sql(repaired)
                    else:
                        print(f"   {_RED}Self-repair produced no SQL{_R}", flush=True)
                        break
                except Exception as repair_exc:
                    print(f"   {_RED}Self-repair call failed: {repair_exc}{_R}", flush=True)
                    break

    print(f"   sql={_DIM}{sql[:120]}…{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
    return sql


# ══════════════════════════════════════════════════════════════════════════════
#  Layer 3 – Three-layer SQL Validation
# ══════════════════════════════════════════════════════════════════════════════

def _validate_sql(conn, sql: str, table: str, tables: list[str], intent: dict) -> tuple[bool, list[str]]:
    """
    Three-layer validation:
      Layer 1 — Column existence check (x_axis, y_axis present in any loaded table)
      Layer 2 — DuckDB EXPLAIN pre-flight (catches syntax errors without reading data)
      Layer 3 — Result shape check (non-empty, correct column count)

    Returns (passed: bool, log: list[str])
    """
    log: list[str] = []

    print(
        f"\n{_BOLD}{_BLUE}┌── Route A · Stage 3: Validation (3 layers){_R}",
        flush=True,
    )

    # ── Layer 1: Column existence ──────────────────────────────────────────────
    try:
        target_cols: set[str] = set(
            row[0].lower()
            for row in conn.execute(f"DESCRIBE {_qident(table)}").fetchall()
        )

        other_cols: set[str] = set()
        for t in (tables or []):
            if t == table:
                continue
            try:
                other_cols.update(
                    row[0].lower()
                    for row in conn.execute(f"DESCRIBE {_qident(t)}").fetchall()
                )
            except Exception:
                pass

        # Aggregation aliases the LLM may return as y_axis/x_axis —
        # these are computed by the SQL itself, not real table columns,
        # so Layer 1 must not reject them. Layer 2 (EXPLAIN) catches
        # any real SQL errors involving these names.
        _COMPUTED_ALIASES = {
            "count", "total", "average", "avg", "sum", "min", "max",
            "frequency", "percent", "pct", "ratio", "rate", "rank",
            "median", "stddev", "variance", "revenue", "spend",
        }

        for axis in ("x_axis", "y_axis", "group_by", "order_by"):
            col = intent.get(axis)
            if not col:
                continue
            # Unwrap wrapped expressions like "average(quantity)" → "quantity"
            unwrapped, _ = _unwrap_agg_column(col)
            col_l = unwrapped.lower().strip()

            # Skip validation if it is a known computed alias — it does not
            # exist as a table column; it is produced by the SQL aggregation.
            if col_l in _COMPUTED_ALIASES:
                log.append(f"Layer 1 SKIP: '{col}' is a computed alias, not a table column")
                continue

            # Skip validation if the intent also specifies an aggregation —
            # the LLM returned the alias name rather than the source column.
            if intent.get("aggregation") and intent.get("aggregation") != "none":
                if axis == "y_axis":
                    log.append(f"Layer 1 SKIP: '{col}' is y_axis with aggregation '{intent['aggregation']}' — alias, not a table column")
                    continue

            if col_l in target_cols:
                continue
            if col_l in other_cols:
                msg = (
                    f"Layer 1 WARN: column '{unwrapped}' not in target table '{table}' "
                    f"but found in another loaded table — SQL must JOIN to use it"
                )
                log.append(msg)
                print(f"   {_YELLOW}{msg}{_R}", flush=True)
                continue
            msg = f"Layer 1 FAIL: column '{unwrapped}' not found in '{table}' or any other loaded table"
            log.append(msg)
            print(f"   {_RED}{msg}{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
            return False, log

        log.append(f"Layer 1 PASS: all intent columns verified against table '{table}'")
        print(f"   {_GREEN}Layer 1 PASS{_R}: columns verified", flush=True)
    except Exception as e:
        msg = f"Layer 1 ERROR: {e}"
        log.append(msg)
        print(f"   {_RED}{msg}{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
        return False, log

    # ── Layer 2: DuckDB EXPLAIN pre-flight ────────────────────────────────────
    try:
        conn.execute(f"EXPLAIN {sql}")
        log.append("Layer 2 PASS: DuckDB EXPLAIN succeeded")
        print(f"   {_GREEN}Layer 2 PASS{_R}: EXPLAIN ok", flush=True)
    except Exception as e:
        msg = f"Layer 2 FAIL: SQL syntax error — {e}"
        log.append(msg)
        print(f"   {_RED}{msg}{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
        return False, log

    # ── Layer 3: Result shape check ───────────────────────────────────────────
    try:
        probe = conn.execute(f"SELECT * FROM ({sql.rstrip(';')}) __probe LIMIT 1").df()
        if probe.empty:
            msg = "Layer 3 WARN: query returned 0 rows — chart will be empty"
            log.append(msg)
            print(f"   {_YELLOW}{msg}{_R}", flush=True)
        else:
            log.append(f"Layer 3 PASS: result has {len(probe.columns)} columns")
            print(
                f"   {_GREEN}Layer 3 PASS{_R}: cols={list(probe.columns)}",
                flush=True,
            )
    except Exception as e:
        msg = f"Layer 3 FAIL: result shape check error — {e}"
        log.append(msg)
        print(f"   {_RED}{msg}{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
        return False, log

    print(f"{_BOLD}{_BLUE}└──{_R}", flush=True)
    return True, log


# ══════════════════════════════════════════════════════════════════════════════
#  Layer 4 – DuckDB Execution
# ══════════════════════════════════════════════════════════════════════════════

def _execute_sql(conn, sql: str) -> pd.DataFrame:
    print(
        f"\n{_BOLD}{_BLUE}┌── Route A · Stage 4: DuckDB Execution{_R}",
        flush=True,
    )
    df = conn.execute(sql).df()
    print(
        f"   {_GREEN}OK{_R}: {len(df)} rows × {len(df.columns)} cols\n"
        f"{_BOLD}{_BLUE}└──{_R}",
        flush=True,
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Layer 5 – Visualization Builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_chart(df: pd.DataFrame, intent: dict) -> go.Figure:
    """
    Build a Plotly figure from the result DataFrame + intent spec.
    Falls back gracefully if columns don't match.
    """
    chart_type = intent.get("chart_type", "bar")
    x_col      = intent.get("x_axis")
    y_col      = intent.get("y_axis")
    title      = intent.get("title", "")
    x_label    = intent.get("x_label") or x_col or ""
    y_label    = intent.get("y_label") or y_col or ""
    group_col  = intent.get("group_by")

    # Resolve columns case-insensitively against actual DataFrame columns.
    # Also tolerant of an aggregate-shaped alias (e.g. "average(quantity)")
    # in case the SQL model didn't follow the plain-alias instruction.
    cols_lower = {c.lower(): c for c in df.columns}
    agg = intent.get("aggregation") or "none"

    def _resolve(col):
        if col is None:
            return None
        if col in df.columns:
            return col
        hit = cols_lower.get(col.lower())
        if hit:
            return hit
        if agg != "none":
            for candidate in (f"{agg}({col})", f"{agg}_{col}"):
                hit = cols_lower.get(candidate.lower())
                if hit:
                    return hit
        return None

    x_col     = _resolve(x_col)
    y_col     = _resolve(y_col)
    group_col = _resolve(group_col)

    print(
        f"\n{_BOLD}{_BLUE}┌── Route A · Stage 5: Visualization Builder{_R}\n"
        f"   type={_GREEN}{chart_type}{_R}  x={x_col}  y={y_col}  "
        f"group={group_col}",
        flush=True,
    )

    fig = None

    try:
        if chart_type == "bar":
            if x_col and y_col:
                fig = px.bar(
                    df, x=x_col, y=y_col,
                    color=group_col,
                    title=title,
                    labels={x_col: x_label, y_col: y_label},
                    text_auto=True,
                )
                fig.update_layout(xaxis_tickangle=-35)
            else:
                fig = _fallback_bar(df, title)

        elif chart_type in ("pie", "donut"):
            names_col  = x_col or (df.columns[0] if len(df.columns) >= 1 else None)
            values_col = y_col or (df.columns[1] if len(df.columns) >= 2 else None)
            if names_col and values_col:
                fig = px.pie(
                    df, names=names_col, values=values_col, title=title,
                    hole=0.45 if chart_type == "donut" else 0.0,
                )
            else:
                fig = _fallback_bar(df, title)

        elif chart_type in ("line", "area"):
            if x_col and y_col:
                try:
                    df[x_col] = pd.to_datetime(df[x_col])
                    df = df.sort_values(x_col)
                except Exception:
                    pass
                chart_fn = px.area if chart_type == "area" else px.line
                fig = chart_fn(
                    df, x=x_col, y=y_col,
                    color=group_col,
                    title=title,
                    labels={x_col: x_label, y_col: y_label},
                    markers=True if chart_type == "line" else False,
                )
            else:
                fig = _fallback_bar(df, title)

        elif chart_type == "scatter":
            if x_col and y_col:
                fig = px.scatter(
                    df, x=x_col, y=y_col,
                    color=group_col,
                    title=title,
                    labels={x_col: x_label, y_col: y_label},
                    trendline="ols" if group_col is None else None,
                )
            else:
                fig = _fallback_bar(df, title)

        elif chart_type == "histogram":
            col = x_col or y_col or df.select_dtypes(include="number").columns[0]
            fig = px.histogram(
                df, x=col,
                color=group_col,
                title=title,
                labels={col: x_label or col},
                nbins=30,
            )

        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include="number")
            if not numeric_df.empty:
                fig = px.imshow(
                    numeric_df.corr(),
                    title=title or "Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                )
            else:
                fig = _fallback_bar(df, title)

        elif chart_type == "box":
            if y_col:
                fig = px.box(
                    df, x=x_col, y=y_col,
                    color=group_col,
                    title=title,
                    labels={y_col: y_label, **({x_col: x_label} if x_col else {})},
                )
            else:
                fig = _fallback_bar(df, title)

        elif chart_type == "funnel":
            names_col  = x_col or (df.columns[0] if len(df.columns) >= 1 else None)
            values_col = y_col or (df.columns[1] if len(df.columns) >= 2 else None)
            if names_col and values_col:
                fig = px.funnel(df, x=values_col, y=names_col, title=title)
            else:
                fig = _fallback_bar(df, title)

        elif chart_type == "treemap":
            path_col   = x_col or (df.columns[0] if len(df.columns) >= 1 else None)
            values_col = y_col or (df.columns[1] if len(df.columns) >= 2 else None)
            if path_col and values_col:
                fig = px.treemap(
                    df, path=[path_col], values=values_col, title=title,
                )
            else:
                fig = _fallback_bar(df, title)

        else:
            fig = _fallback_bar(df, title)

    except Exception as e:
        logger.warning("Chart build failed (%s), using fallback bar chart", e)
        fig = _fallback_bar(df, title)

    # Consistent layout polish
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        margin=dict(t=60, b=40, l=40, r=20),
    )

    print(f"{_BOLD}{_BLUE}└──{_R}", flush=True)
    return fig


def _fallback_bar(df: pd.DataFrame, title: str) -> go.Figure:
    """Last-resort bar chart using first two columns."""
    if df.empty:
        return go.Figure().update_layout(title=title or "No data")
    cols = list(df.columns)
    x = cols[0]
    y = cols[1] if len(cols) > 1 else cols[0]
    return px.bar(df, x=x, y=y, title=title or f"{y} by {x}")


# ══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def run(
    conn,
    tables: list[str],
    prompt: str,
    router_intent: dict | None = None,
) -> RouteAResult:
    """
    Execute the full Route A visualization pipeline.

    Args:
        conn          : DuckDB connection
        tables        : list of available table names
        prompt        : user's natural language question
        router_intent : partial intent already extracted by the query router
                        (chart_type, x_axis, y_axis, aggregation). If provided,
                        Stage 1 is still called to fill in missing fields.

    Returns RouteAResult with .fig (Plotly Figure), .df, .sql, .intent, etc.
    """
    primary_table = tables[0] if tables else ""
    if not primary_table:
        return RouteAResult(success=False, error="No tables loaded.")

    # Build DDL for any callers that still need it (kept for compatibility)
    ddl = generate_multi_table_ddl(
        conn, tables, redact=False, max_columns=DDL_MAX_COLUMNS
    )

    print(
        f"\n{_BOLD}{'═'*55}{_R}\n"
        f"  {_BLUE}{_BOLD}ROUTE A — VISUALIZATION PIPELINE{_R}\n"
        f"  tables={_CYAN}{', '.join(tables)}{_R}  prompt='{prompt[:60]}'\n"
        f"{_BOLD}{'═'*55}{_R}",
        flush=True,
    )

    result = RouteAResult(success=False)

    # ── Stage 1: Intent Extraction ────────────────────────────────────────────
    try:
        if router_intent and router_intent.get("chart_type"):
            intent = dict(router_intent)
            try:
                # Enrich with full intent (filters, labels, limit, etc.)
                full_intent = _extract_intent(ddl, tables, prompt, conn=conn)
                # Router values take priority for chart_type/axes
                full_intent.update({
                    k: v for k, v in intent.items()
                    if v is not None and k in ("chart_type", "x_axis", "y_axis", "aggregation")
                })
                intent = full_intent
            except Exception:
                pass  # use router intent as-is if Ollama fails
        else:
            intent = _extract_intent(ddl, tables, prompt, conn=conn)

        result.intent        = intent
        result.stage_reached = "intent"
    except Exception as e:
        result.error = f"Stage 1 (Intent Extraction) failed: {e}"
        logger.error(result.error, exc_info=True)
        _print_fail(result.error)
        return result

    # Resolve the chosen table (falls back to first table if LLM returned invalid name)
    table_lookup = {t.lower(): t for t in tables}
    target_table = table_lookup.get(str(intent.get("table") or "").lower(), primary_table)
    intent["table"] = target_table

    # ── Stage 2: SQL Generation ───────────────────────────────────────────────
    try:
        sql = _generate_sql(ddl, target_table, tables, intent, prompt, conn=conn)
        if not sql:
            result.error = "Stage 2 (SQL Generation) produced no valid SQL."
            _print_fail(result.error)
            return result
        result.sql           = sql
        result.stage_reached = "sql"
    except Exception as e:
        result.error = f"Stage 2 (SQL Generation) failed: {e}"
        logger.error(result.error, exc_info=True)
        _print_fail(result.error)
        return result

    # ── Stage 3: Three-layer Validation ──────────────────────────────────────
    passed, log = _validate_sql(conn, sql, target_table, tables, intent)
    result.validation_log = log
    result.stage_reached  = "validation"
    if not passed:
        result.error = "Stage 3 (Validation) failed. " + (log[-1] if log else "")
        _print_fail(result.error)
        return result

    # ── Stage 4: DuckDB Execution ─────────────────────────────────────────────
    try:
        df = _execute_sql(conn, sql)
        result.df            = df
        result.stage_reached = "execution"
    except Exception as e:
        result.error = f"Stage 4 (DuckDB Execution) failed: {e}"
        logger.error(result.error, exc_info=True)
        _print_fail(result.error)
        return result

    # ── Stage 5: Visualization Builder ───────────────────────────────────────
    try:
        fig = _build_chart(df, intent)
        result.fig           = fig
        result.stage_reached = "visualization"
        result.success       = True

        print(
            f"\n{_BOLD}{_GREEN}{'═'*55}{_R}\n"
            f"  {_GREEN}{_BOLD}ROUTE A COMPLETE ✓{_R}  "
            f"chart={intent.get('chart_type')}  rows={len(df)}\n"
            f"{_BOLD}{_GREEN}{'═'*55}{_R}\n",
            flush=True,
        )
    except Exception as e:
        result.error = f"Stage 5 (Visualization) failed: {e}"
        logger.error(result.error, exc_info=True)
        _print_fail(result.error)
        return result

    return result


def _print_fail(msg: str) -> None:
    print(
        f"\n{_RED}{_BOLD}  ROUTE A FAILED: {msg}{_R}\n",
        flush=True,
    )



# User question
#     │
#     ▼
# Stage 1: Intent Extraction      (llama3.2:3b)
#     │   "what chart type? which columns? which table?"
#     ▼
# Stage 2: SQL Generation         (qwen2.5-coder:7b)
#     │   writes a DuckDB SELECT query
#     ▼
# Stage 3: 3-Layer Validation
#     │   Layer 1 → column existence check
#     │   Layer 2 → DuckDB EXPLAIN (syntax check, no data read)
#     │   Layer 3 → result shape check (non-empty, correct cols)
#     ▼
# Stage 4: DuckDB Execution
#     │   runs the validated SQL, gets a DataFrame
#     ▼
# Stage 5: Plotly Chart Builder
#         outputs a go.Figure