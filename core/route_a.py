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
    _extract_sql,
    _call_ollama,
    _REPAIR_TEMPLATE,
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


def _sql_output_columns(conn, sql: str) -> set[str] | None:
    """
    Ask DuckDB what columns the given SQL would *actually* produce, without
    reading any real rows (LIMIT 0). This is the ground truth for whether an
    intent axis like "order_quantity" is a real, valid result column — far
    more reliable than guessing from intent['aggregation'], which is set by
    the INTENT model and can drift out of sync with whatever the SQL model
    actually wrote (e.g. intent says aggregation:"none" but the SQL still
    computes COUNT(...) AS "order_quantity" because the question asked for
    "order quantity").

    Returns None if the SQL can't even be planned (real syntax error) — in
    that case Layer 2's EXPLAIN check below will surface the actual error,
    so Layer 1 should not assume a column is missing just because the probe
    here failed.
    """
    try:
        probe = conn.execute(
            f"SELECT * FROM ({sql.rstrip(';')}) __schema_probe LIMIT 0"
        ).df()
        return set(c.lower() for c in probe.columns)
    except Exception:
        return None


def _table_column_map(conn, tables: list[str]) -> dict[str, set[str]]:
    """Map each table name -> set of its lowercase column names, via DESCRIBE."""
    out: dict[str, set[str]] = {}
    for t in tables:
        try:
            out[t] = set(
                row[0].lower()
                for row in conn.execute(f"DESCRIBE {_qident(t)}").fetchall()
            )
        except Exception:
            out[t] = set()
    return out


def _build_join_hints(intent: dict, target_table: str, table_cols: dict[str, set[str]]) -> str:
    """
    For every intent axis (x_axis/y_axis/group_by/order_by) that is NOT a
    column of the primary/target table but IS a column of exactly one (or
    more) OTHER loaded table, emit an explicit "you must JOIN to table X"
    instruction for the SQL model.

    Why this matters: the SQL model sometimes picks a primary table and then
    writes a single-table query even though the question also needs a
    column that actually lives elsewhere (e.g. "department" lives in
    "employees", not in "performance") — it just forgets the JOIN. DuckDB's
    own binder error in that case ("Referenced column not found", with
    "Candidate bindings") only lists columns already in scope from tables
    already in the FROM clause, so the self-repair pass often can't recover
    on its own. Telling the model up front, by name, which table actually
    has the column is far more reliable than hoping it notices.

    Columns that don't exist in ANY loaded table (true computed aliases like
    "total_projects_completed") are silently skipped here — that's correct;
    they're meant to be produced by aggregation, not looked up by JOIN.
    """
    target_cols = table_cols.get(target_table, set())
    hints: list[str] = []
    seen: set[str] = set()

    for axis in ("x_axis", "y_axis", "group_by", "order_by"):
        col = intent.get(axis)
        if not col or col in seen:
            continue
        seen.add(col)

        unwrapped, _ = _unwrap_agg_column(col)
        col_l = unwrapped.lower().strip()

        if col_l in target_cols:
            continue  # already reachable from the primary table — nothing to do

        owners = [
            t for t, cols in table_cols.items()
            if t != target_table and col_l in cols
        ]
        if owners:
            owners_str = " or ".join(f'"{o}"' for o in owners)
            hints.append(
                f'- "{unwrapped}" is NOT a column of "{target_table}" — it belongs to '
                f'{owners_str}. You MUST JOIN "{target_table}" to that table on a shared '
                f'id/key column to use it. Do not skip this column or assume it is local '
                f'to "{target_table}".'
            )

    if not hints:
        return ""
    return "\n\nIMPORTANT — CROSS-TABLE COLUMNS DETECTED:\n" + "\n".join(hints)


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
    intent["chart_type"]  = str(intent.get("chart_type",  "")).lower()
    intent["aggregation"] = str(intent.get("aggregation", "none")).lower()
    _VALID_CHART_TYPES = (
        "bar", "pie", "donut", "line", "area", "scatter",
        "histogram", "heatmap", "box", "funnel", "treemap",
    )
    if intent["chart_type"] not in _VALID_CHART_TYPES:
        raise ValueError(
            f"The AI could not determine a valid chart type for your question "
            f"(got '{intent['chart_type']}'). Please specify the chart type explicitly, "
            f"e.g. 'show a bar chart of ...' or 'pie chart of ...'."
        )
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
        raise ValueError(
            f"The AI selected a table '{chosen_table}' that does not exist in your loaded data. "
            f"Available tables: {', '.join(tables)}. "
            "Please rephrase your question or check that the right file is loaded."
        )

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

    # Proactively detect intent columns that live in a DIFFERENT table than
    # the chosen primary table, and tell the model exactly where to find
    # them and that a JOIN is required — see _build_join_hints docstring.
    join_hints = ""
    if conn and tables:
        table_cols = _table_column_map(conn, tables)
        join_hints = _build_join_hints(intent, table, table_cols)
        if join_hints:
            print(f"   {_YELLOW}Join hints injected:{_R}{join_hints}", flush=True)

    user_msg = _SQL_USER.format(
        column_dict=column_dict,
        table=table,
        tables=", ".join(tables),
        intent_json=json.dumps(intent, indent=2),
        question=question,
    ) + join_hints
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

        # Ground truth: what columns will this exact SQL actually produce?
        # This catches every case where the SQL model invented a result
        # alias (e.g. COUNT("order_id") AS "order_quantity") that matches
        # the intent's axis name but isn't a raw table column — regardless
        # of which axis it's on or whether intent['aggregation'] happens to
        # say "none" (the two models can disagree; see _sql_output_columns).
        output_cols = _sql_output_columns(conn, sql) or set()

        for axis in ("x_axis", "y_axis", "group_by", "order_by"):
            col = intent.get(axis)
            if not col:
                continue
            # Unwrap wrapped expressions like "average(quantity)" → "quantity"
            unwrapped, _ = _unwrap_agg_column(col)
            col_l = unwrapped.lower().strip()

            # Ground-truth check FIRST: if the SQL we're about to run already
            # outputs a column with this exact name, it's valid — full stop.
            if col_l in output_cols:
                log.append(f"Layer 1 SKIP: '{col}' present in the SQL's actual output schema")
                continue

            # Skip validation if it is a known computed alias — it does not
            # exist as a table column; it is produced by the SQL aggregation.
            if col_l in _COMPUTED_ALIASES:
                log.append(f"Layer 1 SKIP: '{col}' is a computed alias, not a table column")
                continue

            # Skip validation if the intent also specifies an aggregation —
            # the LLM returned the alias name rather than the source column.
            # Applies to ANY axis (x_axis, y_axis, group_by, order_by), not
            # just y_axis — an aggregated value can legitimately sit on any
            # of these (e.g. a horizontal bar chart aggregates on x_axis).
            if intent.get("aggregation") and intent.get("aggregation") != "none":
                log.append(
                    f"Layer 1 SKIP: '{col}' is {axis} with aggregation "
                    f"'{intent['aggregation']}' — likely an alias, not a table column"
                )
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

    # ── When intent axes are null, infer from DataFrame columns ──────────────
    # The router sometimes sends x_axis/y_axis as NULL (rule-based match with
    # no LLM enrichment). Auto-assign: numeric col → x, categorical col → y
    # so downstream chart builders always have something to work with.
    if (not x_col or not y_col) and len(df.columns) >= 2:
        num_cols = list(df.select_dtypes(include="number").columns)
        cat_cols = [c for c in df.columns if c not in num_cols]
        if not x_col and not y_col:
            x_col = num_cols[0] if num_cols else df.columns[0]
            y_col = cat_cols[0] if cat_cols else df.columns[1]
        elif not x_col:
            x_col = num_cols[0] if num_cols else df.columns[0]
        elif not y_col:
            y_col = cat_cols[0] if cat_cols else df.columns[1]
        # Refresh labels to match inferred columns
        x_label = x_label or x_col or ""
        y_label = y_label or y_col or ""

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
                raise ValueError(
                    f"A bar chart needs both an X axis and a Y axis, but the query result "
                    f"did not provide {'an X column' if not x_col else 'a Y column'}. "
                    "Please rephrase your question to specify what you want on each axis."
                )

        elif chart_type in ("pie", "donut"):
            names_col  = x_col or (df.columns[0] if len(df.columns) >= 1 else None)
            values_col = y_col or (df.columns[1] if len(df.columns) >= 2 else None)
            if names_col and values_col:
                fig = px.pie(
                    df, names=names_col, values=values_col, title=title,
                    hole=0.45 if chart_type == "donut" else 0.0,
                )
            else:
                raise ValueError(
                    "A pie/donut chart needs at least two columns (category and value), "
                    "but the query result did not return enough columns. "
                    "Please rephrase your question to include a category and a numeric value."
                )

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
                raise ValueError(
                    f"A {chart_type} chart needs both an X axis and a Y axis, but the query result "
                    f"did not provide {'an X column' if not x_col else 'a Y column'}. "
                    "Please rephrase your question to specify what you want on each axis."
                )

        elif chart_type == "scatter":
            # ── Auto-resolve: pick the best available columns if intent left them null ──
            if not x_col or not y_col:
                num_cols = list(df.select_dtypes(include="number").columns)
                cat_cols = [c for c in df.columns if c not in num_cols]
                if not x_col and not y_col:
                    # Assign: numeric → x, categorical → y (strip-plot style)
                    x_col = num_cols[0] if num_cols else (df.columns[0] if len(df.columns) >= 1 else None)
                    y_col = cat_cols[0] if cat_cols else (df.columns[1] if len(df.columns) >= 2 else None)
                elif not x_col:
                    x_col = num_cols[0] if num_cols else df.columns[0]
                elif not y_col:
                    y_col = cat_cols[0] if cat_cols else (df.columns[1] if len(df.columns) >= 2 else None)

            if x_col and y_col:
                # ── Detect axis types ─────────────────────────────────────────
                x_is_numeric = pd.api.types.is_numeric_dtype(df[x_col])
                y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col])

                # If both are categorical, fall back to count-based bar
                if not x_is_numeric and not y_is_numeric:
                    fig = px.bar(
                        df.groupby([x_col, y_col]).size().reset_index(name="count"),
                        x=x_col, y="count", color=y_col,
                        title=title or f"Distribution of {x_col} by {y_col}",
                        labels={x_col: x_label or x_col, "count": "Count"},
                    )

                # One numeric, one categorical → strip plot
                # (px.strip handles categorical y-axis natively; no numeric conversion needed)
                elif x_is_numeric and not y_is_numeric:
                    # numeric on x, categorical on y — standard strip/dot layout
                    fig = px.strip(
                        df, x=x_col, y=y_col,
                        color=group_col or y_col,
                        title=title or f"{x_label or x_col} by {y_label or y_col}",
                        labels={x_col: x_label or x_col, y_col: y_label or y_col},
                    )
                    fig.update_traces(jitter=0.4, marker=dict(size=6, opacity=0.7))

                elif not x_is_numeric and y_is_numeric:
                    # categorical on x, numeric on y — swap so numeric is on x for readability
                    fig = px.strip(
                        df, x=y_col, y=x_col,
                        color=group_col or x_col,
                        title=title or f"{y_label or y_col} by {x_label or x_col}",
                        labels={y_col: y_label or y_col, x_col: x_label or x_col},
                    )
                    fig.update_traces(jitter=0.4, marker=dict(size=6, opacity=0.7))

                else:
                    # Both numeric — standard scatter with optional OLS trendline
                    fig = px.scatter(
                        df, x=x_col, y=y_col,
                        color=group_col,
                        title=title,
                        labels={x_col: x_label, y_col: y_label},
                        trendline="ols" if group_col is None else None,
                    )
            else:
                raise ValueError(
                    "A scatter plot needs at least one numeric column. "
                    "Please rephrase your question, e.g. 'scatter plot of age by city'."
                )

        elif chart_type == "histogram":
            col = x_col or y_col or (df.select_dtypes(include="number").columns[0] if not df.select_dtypes(include="number").empty else None)
            if col is None:
                raise ValueError(
                    "A histogram needs a numeric column, but no numeric columns were found in the result. "
                    "Please rephrase your question to target a numeric field."
                )
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
                raise ValueError(
                    "A heatmap requires numeric columns, but no numeric data was found in the query result. "
                    "Please rephrase your question to include numeric fields."
                )

        elif chart_type == "box":
            if y_col:
                fig = px.box(
                    df, x=x_col, y=y_col,
                    color=group_col,
                    title=title,
                    labels={y_col: y_label, **({x_col: x_label} if x_col else {})},
                )
            else:
                raise ValueError(
                    "A box plot needs a Y axis (numeric) column. "
                    "Please rephrase your question, e.g. 'box plot of price by category'."
                )

        elif chart_type == "funnel":
            names_col  = x_col or (df.columns[0] if len(df.columns) >= 1 else None)
            values_col = y_col or (df.columns[1] if len(df.columns) >= 2 else None)
            if names_col and values_col:
                fig = px.funnel(df, x=values_col, y=names_col, title=title)
            else:
                raise ValueError(
                    "A funnel chart needs at least two columns (stage name and value). "
                    "Please rephrase your question to include a category and a numeric value."
                )

        elif chart_type == "treemap":
            path_col   = x_col or (df.columns[0] if len(df.columns) >= 1 else None)
            values_col = y_col or (df.columns[1] if len(df.columns) >= 2 else None)
            if path_col and values_col:
                fig = px.treemap(
                    df, path=[path_col], values=values_col, title=title,
                )
            else:
                raise ValueError(
                    "A treemap needs at least two columns (category path and value). "
                    "Please rephrase your question to include a category and a numeric value."
                )

        else:
            raise ValueError(
                f"'{chart_type}' is not a supported chart type. "
                "Supported types: bar, pie, donut, line, area, scatter, histogram, heatmap, box, funnel, treemap."
            )

    except Exception as e:
        logger.warning("Chart build failed (%s)", e)
        raise

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
    # Primary path: Route-A's original Ollama-based SQL generation.
    # Fallback path: Route-D-style generation (same engine as Routes B/D/C
    # which are known to work reliably) — gives us data output first,
    # then the visualization builder works on that data exactly as before.
    sql: str | None = None
    try:
        sql = _generate_sql(ddl, target_table, tables, intent, prompt, conn=conn)
    except Exception as e:
        logger.warning("Stage 2 primary SQL generation raised: %s — trying D-style fallback", e)

    if not sql:
        print(
            f"   {_YELLOW}Primary SQL generation failed — switching to Route-D-style "
            f"SQL engine (same approach as routes B/D/C){_R}",
            flush=True,
        )
        try:
            sql = _generate_sql_via_route_d(conn, tables, intent, prompt)
        except Exception as e:
            logger.warning("D-style SQL fallback raised: %s", e)

    if not sql:
        result.error = (
            "Stage 2 (SQL Generation) failed: both the primary Ollama path and "
            "the Route-D-style fallback could not produce valid SQL. "
            "Please rephrase your question or check your data."
        )
        _print_fail(result.error)
        return result

    result.sql           = sql
    result.stage_reached = "sql"

    # ── Stage 3: Three-layer Validation ──────────────────────────────────────
    passed, log = _validate_sql(conn, sql, target_table, tables, intent)
    result.validation_log = log
    result.stage_reached  = "validation"
    if not passed:
        # Primary SQL failed validation — attempt Route-D-style fallback before giving up
        print(
            f"   {_YELLOW}Stage 3 validation failed — trying Route-D-style SQL fallback{_R}",
            flush=True,
        )
        fallback_sql: str | None = None
        try:
            fallback_sql = _generate_sql_via_route_d(conn, tables, intent, prompt)
        except Exception as fb_exc:
            logger.warning("D-style validation-fallback raised: %s", fb_exc)

        if fallback_sql:
            passed_fb, log_fb = _validate_sql(conn, fallback_sql, target_table, tables, intent)
            if passed_fb:
                print(f"   {_GREEN}D-style fallback SQL passed validation ✓{_R}", flush=True)
                sql = fallback_sql
                log = log_fb
                result.sql = sql
                result.validation_log = log
                passed = True
            else:
                log = log + log_fb  # append both logs for debugging

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


# ══════════════════════════════════════════════════════════════════════════════
#  Route-D-style SQL fallback — same engine Routes B/D/C use successfully
# ══════════════════════════════════════════════════════════════════════════════

_VIZ_SQL_SYSTEM = """\
You are an expert DuckDB SQL writer for data visualization queries.
Given a COLUMN DICTIONARY and a visualization intent JSON, write a single
DuckDB SELECT query that retrieves ONLY the columns needed to plot the chart.

STRICT OUTPUT RULES:
- Output ONLY the raw SQL query. No markdown, no backticks, no explanation.
- Start with SELECT or WITH. End with a semicolon.
- Never use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER.

COLUMN RULES:
- Use ONLY column names that exist in the COLUMN DICTIONARY.
- Double-quote every column and table name: "my column", "my_table".
- ALWAYS alias aggregated columns with the plain column name from the intent
  (e.g. AVG("price") AS "price").
- Apply GROUP BY, ORDER BY, and LIMIT as indicated in the intent JSON.
- If a column is marked [DELIMITED with 'X'] in the dictionary, expand it with
  UNNEST(STRING_SPLIT(col, 'X')) before grouping or counting.
- Use TRY_CAST (never plain CAST) when converting VARCHAR to numeric.
- LIMIT always goes after ORDER BY.
"""

_VIZ_SQL_USER = """\
### COLUMN DICTIONARY (use ONLY these exact column names):
{column_dict}

### VISUALIZATION INTENT:
{intent_json}

### USER QUESTION:
{question}

SQL query:
"""


def _generate_sql_via_route_d(
    conn,
    tables: list[str],
    intent: dict,
    question: str,
) -> str | None:
    """
    Route-D-style SQL generation for visualization queries.
    Uses the same _call_ollama / _extract_sql / _harden_sql / self-repair
    loop that Routes B and D use — which are known to work reliably.

    Returns the SQL string on success, or None on failure.
    """
    column_dict = _build_column_dict(conn, tables)

    target_table = str(intent.get("table") or (tables[0] if tables else ""))
    table_cols = _table_column_map(conn, tables)
    join_hints = _build_join_hints(intent, target_table, table_cols)
    if join_hints:
        print(f"   {_YELLOW}Join hints injected:{_R}{join_hints}", flush=True)

    user_payload = _VIZ_SQL_USER.format(
        column_dict=column_dict,
        intent_json=json.dumps(intent, indent=2),
        question=question,
    ) + join_hints
    messages = [
        {"role": "system", "content": _VIZ_SQL_SYSTEM},
        {"role": "user",   "content": user_payload},
    ]

    print(
        f"\n{_BOLD}{_BLUE}┌── Route A · Stage 2 (D-style fallback): SQL Generation{_R}",
        flush=True,
    )

    try:
        raw = _call_ollama(messages)
    except Exception as exc:
        print(f"   {_RED}D-style SQL call failed: {exc}{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
        return None

    sql = _extract_sql(raw)
    if not sql:
        print(f"   {_RED}No valid SQL extracted{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
        return None

    sql = _harden_sql(sql)

    # Self-repair loop — identical to Route D (up to 3 attempts)
    repair_messages = list(messages)
    max_repairs = 3
    for attempt in range(1, max_repairs + 1):
        try:
            conn.execute(f"EXPLAIN {sql}")
            if attempt > 1:
                print(f"   {_GREEN}Self-repair succeeded (attempt {attempt}){_R}", flush=True)
            break
        except Exception as explain_err:
            if attempt == max_repairs:
                print(
                    f"   {_YELLOW}Self-repair exhausted after {max_repairs} attempts "
                    f"({explain_err}){_R}",
                    flush=True,
                )
                break
            repair_messages.append({"role": "assistant", "content": sql})
            repair_messages.append({
                "role": "user",
                "content": _REPAIR_TEMPLATE.format(error=str(explain_err), sql=sql),
            })
            try:
                raw_repair = _call_ollama(repair_messages)
                repaired = _extract_sql(raw_repair)
                if repaired:
                    sql = _harden_sql(repaired)
            except Exception:
                break

    print(f"   sql={_DIM}{sql[:120]}…{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
    return sql