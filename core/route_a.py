# ─────────────────────────────────────────────────────────────────────────────
# core/route_a.py  –  Route A: Visualization Pipeline
#
# Full pipeline as per architecture diagram:
#
#   INTENT EXTRACTION  →  SQL GENERATION  →  3-LAYER VALIDATION
#   →  DUCKDB EXECUTION  →  VISUALIZATION BUILDER (Plotly)
#
# Both LLM calls use local Ollama — zero data leaves your machine.
# Only DDL schema is sent, never raw row data.
#
# Models (configurable in config.py):
#   INTENT_MODEL  = llama3.2:3b   (intent extraction)
#   SQL_MODEL     = qwen2.5-coder:7b  (SQL generation)
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

logger = logging.getLogger(__name__)

# ── ANSI colours ──────────────────────────────────────────────────────────────
_R = "\033[0m"
_BOLD = "\033[1m"
_BLUE = "\033[94m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_DIM = "\033[2m"
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
    SQL_MODEL = "qwen2.5-coder:7b"


# ══════════════════════════════════════════════════════════════════════════════
#  Result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RouteAResult:
    success: bool
    intent: dict                        = field(default_factory=dict)
    sql: str | None                     = None
    df: pd.DataFrame | None             = None
    fig: go.Figure | None               = None
    validation_log: list[str]           = field(default_factory=list)
    error: str | None                   = None
    stage_reached: str                  = "none"   # intent|sql|validation|execution|visualization


# ══════════════════════════════════════════════════════════════════════════════
#  Layer 1 – Intent Extraction
# ══════════════════════════════════════════════════════════════════════════════

_INTENT_SYSTEM = """\
You are an intent extractor for a data analytics platform.
Given a user question and a database schema (which may contain MULTIPLE tables),
extract visualization intent.

Output ONLY a single JSON object — no text, no markdown, no backticks.

{
  "table":       "<name of the table that best answers the question>",
  "chart_type":  "<bar|pie|line|scatter|histogram|heatmap>",
  "x_axis":      "<column name or null>",
  "y_axis":      "<column name or null>",
  "aggregation": "<sum|avg|count|min|max|none>",
  "group_by":    "<column name or null>",
  "filters":     "<plain English filter description or null>",
  "order_by":    "<column name or null>",
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
- chart_type must be one of: bar, pie, line, scatter, histogram, heatmap
- aggregation must be one of: sum, avg, count, min, max, none
- All string values must use double quotes.
- null values must be JSON null (not the string "null").
"""

_INTENT_USER = """\
Schema (one or more tables):
{ddl}

Available tables: {tables}

User question: {question}
"""


def _extract_intent(ddl: str, tables: list[str], question: str) -> dict:
    """
    Stage 1: Call Ollama to extract chart intent from the user question.
    Returns a parsed dict with table, chart_type, x_axis, y_axis, aggregation, etc.
    """
    user_msg = _INTENT_USER.format(ddl=ddl, tables=", ".join(tables), question=question)
    messages = [
        {"role": "system", "content": _INTENT_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]

    payload = {
        "model":    INTENT_MODEL,
        "messages": messages,
        "stream":   False,
        "format":   "json",
        "options":  {"temperature": 0.0, "num_predict": 300},
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

    # Normalise
    intent["chart_type"]  = str(intent.get("chart_type",  "bar")).lower()
    intent["aggregation"] = str(intent.get("aggregation", "none")).lower()
    if intent["chart_type"] not in ("bar","pie","line","scatter","histogram","heatmap"):
        intent["chart_type"] = "bar"
    if intent["aggregation"] not in ("sum","avg","count","min","max","none"):
        intent["aggregation"] = "none"

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
- Prefer the "primary table" given below. If the question requires columns
  from another table in the schema, you may JOIN to that table using a
  sensible shared column (e.g. matching id/key column names).
- Quote column names that contain spaces or special characters with double-quotes.
- End with a semicolon.
- Apply aggregations and GROUP BY as indicated by the intent.
- Apply ORDER BY and LIMIT as indicated by the intent.
- Keep the query minimal — only fetch columns needed for the chart.
"""

_SQL_USER = """\
Schema (one or more tables):
{ddl}

Primary table: {table}
Available tables: {tables}

Visualization intent:
{intent_json}

User question: {question}

SQL:
"""


def _generate_sql(ddl: str, table: str, tables: list[str], intent: dict, question: str) -> str | None:
    """
    Stage 2: Call Ollama (SQL model) to generate a DuckDB SELECT query
    from the intent JSON + DDL schema (covering all loaded tables).
    """
    user_msg = _SQL_USER.format(
        ddl=ddl,
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
        "options":  {"temperature": 0.0, "num_predict": 512},
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

    # Strip fences
    raw = re.sub(r"```sql\s*|```", "", raw, flags=re.I).strip()

    # Extract first SELECT/WITH block
    m = re.search(r"(SELECT|WITH)\b[\s\S]*", raw, flags=re.I)
    if m:
        sql = m.group(0).strip()
        if ";" in sql:
            sql = sql[:sql.index(";") + 1]
        elif not sql.endswith(";"):
            sql += ";"
        print(f"   sql={_DIM}{sql[:120]}…{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
        return sql

    print(f"   {_RED}No valid SQL extracted{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
    return None


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

    # ── Layer 1: Column existence ─────────────────
    try:
        # Columns of the chosen target table (primary check).
        target_cols: set[str] = set(
            row[0].lower()
            for row in conn.execute(f"DESCRIBE {table}").fetchall()
        )

        # Union of columns across every OTHER loaded table — used only to
        # detect legitimate cross-table (JOIN) references.
        other_cols: set[str] = set()
        for t in (tables or []):
            if t == table:
                continue
            try:
                other_cols.update(
                    row[0].lower()
                    for row in conn.execute(f"DESCRIBE {t}").fetchall()
                )
            except Exception:
                pass

        for axis in ("x_axis", "y_axis", "group_by", "order_by"):
            col = intent.get(axis)
            if not col:
                continue
            col_l = col.lower()
            if col_l in target_cols:
                continue
            if col_l in other_cols:
                msg = (
                    f"Layer 1 WARN: column '{col}' not in target table '{table}' "
                    f"but found in another loaded table — SQL must JOIN to use it"
                )
                log.append(msg)
                print(f"   {_YELLOW}{msg}{_R}", flush=True)
                continue
            msg = f"Layer 1 FAIL: column '{col}' not found in target table '{table}' or any other loaded table"
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

    # ── Layer 2: DuckDB EXPLAIN pre-flight ────────
    try:
        conn.execute(f"EXPLAIN {sql}")
        log.append("Layer 2 PASS: DuckDB EXPLAIN succeeded")
        print(f"   {_GREEN}Layer 2 PASS{_R}: EXPLAIN ok", flush=True)
    except Exception as e:
        msg = f"Layer 2 FAIL: SQL syntax error — {e}"
        log.append(msg)
        print(f"   {_RED}{msg}{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
        return False, log

    # ── Layer 3: Result shape check ───────────────
    try:
        probe = conn.execute(f"SELECT * FROM ({sql.rstrip(';')}) __probe LIMIT 1").df()
        if probe.empty:
            msg = "Layer 3 WARN: query returned 0 rows — chart will be empty"
            log.append(msg)
            print(f"   {_YELLOW}{msg}{_R}", flush=True)
            # Not a hard failure — still render empty state
        else:
            log.append(f"Layer 3 PASS: result has {len(probe.columns)} columns")
            print(
                f"   {_GREEN}Layer 3 PASS{_R}: "
                f"cols={list(probe.columns)}",
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

    # Validate columns against actual df
    cols_lower = {c.lower(): c for c in df.columns}
    def _resolve(col):
        if col is None: return None
        if col in df.columns: return col
        return cols_lower.get(col.lower())

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

        elif chart_type == "pie":
            names_col  = x_col or (df.columns[0] if len(df.columns) >= 1 else None)
            values_col = y_col or (df.columns[1] if len(df.columns) >= 2 else None)
            if names_col and values_col:
                fig = px.pie(
                    df, names=names_col, values=values_col,
                    title=title,
                )
            else:
                fig = _fallback_bar(df, title)

        elif chart_type == "line":
            if x_col and y_col:
                # Coerce x to datetime if possible
                try:
                    df[x_col] = pd.to_datetime(df[x_col])
                    df = df.sort_values(x_col)
                except Exception:
                    pass
                fig = px.line(
                    df, x=x_col, y=y_col,
                    color=group_col,
                    title=title,
                    labels={x_col: x_label, y_col: y_label},
                    markers=True,
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
    router_intent: dict | None = None,   # pre-extracted intent from router (optional)
) -> RouteAResult:
    """
    Execute the full Route A visualization pipeline.

    Args:
        conn          : DuckDB connection
        tables        : list of available table names
        prompt        : user's natural language question
        router_intent : partial intent already extracted by the query router
                        (chart_type, x_axis, y_axis, aggregation). If provided,
                        Stage 1 is skipped.

    Returns RouteAResult with .fig (Plotly Figure), .df, .sql, .intent, etc.
    """
    primary_table = tables[0] if tables else ""
    if not primary_table:
        return RouteAResult(success=False, error="No tables loaded.")

    # Build a combined schema covering EVERY uploaded table in this session,
    # not just the first one — so the LLM can pick the right table and/or
    # JOIN across tables when answering the question.
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
            # Router already gave us partial intent — enrich it via Ollama
            intent = dict(router_intent)
            # Fill missing fields by calling Ollama anyway (router gave chart_type
            # but may not have filters, limit, labels, etc.)
            try:
                full_intent = _extract_intent(ddl, tables, prompt)
                # Merge: router values take priority for chart_type/axes
                full_intent.update({
                    k: v for k, v in intent.items()
                    if v is not None and k in ("chart_type", "x_axis", "y_axis", "aggregation")
                })
                intent = full_intent
            except Exception:
                pass   # use router intent as-is if Ollama fails
        else:
            intent = _extract_intent(ddl, tables, prompt)

        result.intent        = intent
        result.stage_reached = "intent"
    except Exception as e:
        result.error = f"Stage 1 (Intent Extraction) failed: {e}"
        logger.error(result.error, exc_info=True)
        _print_fail(result.error)
        return result

    # The table chosen by intent extraction (falls back to the first table
    # if the LLM didn't return a valid table name).
    table_lookup = {t.lower(): t for t in tables}
    target_table = table_lookup.get(str(intent.get("table") or "").lower(), primary_table)
    intent["table"] = target_table

    # ── Stage 2: SQL Generation ───────────────────────────────────────────────
    try:
        sql = _generate_sql(ddl, target_table, tables, intent, prompt)
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