# ─────────────────────────────────────────────────────────────────────────────
# core/route_d.py  –  Route D: Statistical / Analytical SQL answers
#
# Handles all miscellaneous analytical queries that are NOT:
#   - visualizations (Route A)
#   - plain-English lookups (Route B / sql_answer fallback)
#   - metadata / keyword questions (Route C)
#
# Pipeline:
#   1. Use sql_engine helpers to build column dict + generate DuckDB SQL
#   2. Self-repair via EXPLAIN (up to 3 attempts, same as Route B)
#   3. Execute against DuckDB
#   4. Use Ollama (INSIGHT_MODEL) to produce a concise AI observation
#
# Shared with sql_engine (no duplication):
#   _build_column_dict  – live column dict with delimiter detection
#   _extract_sql        – strips fences, pulls SELECT/WITH block
#   _harden_sql         – TRY_CAST upgrade + decimal-strip removal
#   _call_ollama        – unified Ollama HTTP call with error handling
#   _REPAIR_TEMPLATE    – self-repair prompt template
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    SQL_MODEL,
    INSIGHT_MODEL,
)

# ── Reuse shared helpers from sql_engine (no duplication) ────────────────────
from core.sql_engine import (
    _build_column_dict,
    _extract_sql,
    _harden_sql,
    _call_ollama,
    _REPAIR_TEMPLATE,
)

logger = logging.getLogger(__name__)


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class RouteDResult:
    success: bool
    route: str
    answer: str = ""
    dataframe: pd.DataFrame | None = None
    sql: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# ─── Prompts ──────────────────────────────────────────────────────────────────

# Route D has its own system prompt — statistical focus differs from Route B's
# plain lookup focus, so they intentionally have separate prompts.
_STAT_SYSTEM_PROMPT = """\
You are an expert DuckDB SQL developer specialising in statistical and analytical queries.
Your only job is to write a single valid DuckDB SELECT query that answers the user's request.

STRICT OUTPUT RULES:
- Output ONLY the raw SQL query. No markdown, no backticks, no explanation.
- Start with SELECT or WITH. End with a semicolon.
- Never use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, ATTACH, PRAGMA, COPY.

COLUMN RULES — THIS IS CRITICAL:
- You will be given a COLUMN DICTIONARY listing every table and its exact column names.
- You MUST use column names EXACTLY as they appear in that dictionary.
- Never invent, guess, abbreviate, or rename columns.
- If a column name contains spaces or special characters, wrap it in double-quotes: "My Column".

STATISTICAL QUERY RULES:
- For averages / totals use AVG(), SUM(), COUNT() with GROUP BY.
- For spread use STDDEV_POP(); for median use PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col).
- For ranking use ORDER BY … LIMIT N.
- For frequency distribution use GROUP BY … ORDER BY COUNT(*) DESC.
- For correlation (DuckDB has no built-in CORR()):
    (SUM(a*b) - COUNT(*)*AVG(a)*AVG(b)) / NULLIF(COUNT(*)*STDDEV_POP(a)*STDDEV_POP(b), 0) AS correlation
- Wrap numeric aggregations in TRY_CAST if the column is VARCHAR.
- Use UNNEST(STRING_SPLIT(col, '|')) to expand pipe-delimited multi-value columns — the
  COLUMN DICTIONARY marks these as [DELIMITED with '|'].

DUCKDB-SPECIFIC RULES:
- ALWAYS use TRY_CAST, never plain CAST, when converting text/VARCHAR to numeric or date.
- Column aliases defined in SELECT can be used in ORDER BY, but NOT in WHERE or HAVING.
- LIMIT always goes after ORDER BY.
"""

_STAT_USER_TEMPLATE = """\
### COLUMN DICTIONARY (use ONLY these exact column names):
{column_dict}

### USER REQUEST:
{prompt}

SQL query:
"""

# ─── Observation prompt ───────────────────────────────────────────────────────

_OBS_SYSTEM_PROMPT = """\
You are a precise data analyst.
Your job is to provide a clear, concise statistical observation based on the data provided.

RULES:
1. Be specific — mention actual numbers, ranges, or trends you see in the data.
2. Keep the response under 4 sentences.
3. Do NOT restate the SQL or schema.
4. Do NOT make up data that is not in the result set.
5. If the result is a single number, interpret it in plain language.
"""


# ─── helpers ─────────────────────────────────────────────────────────────────

def _generate_sql(prompt: str, column_dict: str, tables: list[str], conn) -> dict:
    """
    Generate DuckDB SQL via Ollama with up to 3 self-repair attempts on
    EXPLAIN failure — identical repair strategy to sql_engine (Route B).
    Returns {"success": True, "sql": "..."} or {"success": False, "error": "..."}.
    """
    user_payload = _STAT_USER_TEMPLATE.format(
        column_dict=column_dict,
        prompt=prompt,
    )
    messages = [
        {"role": "system", "content": _STAT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_payload},
    ]

    # ── First attempt ─────────────────────────────────────────────────────────
    try:
        raw = _call_ollama(messages)
    except requests.exceptions.ConnectionError:
        msg = f"Cannot connect to Ollama at {OLLAMA_BASE_URL}."
        logger.error(msg)
        return {"success": False, "error": msg}
    except requests.exceptions.Timeout:
        msg = f"Ollama timed out after {OLLAMA_TIMEOUT}s."
        logger.error(msg)
        return {"success": False, "error": msg}
    except Exception as exc:
        return {"success": False, "error": f"Ollama call failed: {exc}"}

    sql = _extract_sql(raw)
    if not sql:
        return {"success": False, "error": "Model returned no valid SQL."}

    sql = _harden_sql(sql)
    logger.info("Route D SQL first attempt: %s…", sql[:120])

    # ── Self-repair loop (up to 3 attempts via EXPLAIN) ───────────────────────
    repair_messages = list(messages)
    max_repairs = 3

    for attempt in range(1, max_repairs + 1):
        try:
            conn.execute(f"EXPLAIN {sql}")
            logger.info("Route D SQL passed EXPLAIN (attempt %d).", attempt)
            break  # valid — exit loop
        except Exception as explain_err:
            if attempt == max_repairs:
                logger.warning("Route D self-repair exhausted after %d attempts: %s", max_repairs, explain_err)
                break

            logger.warning("Route D EXPLAIN failed (attempt %d): %s — retrying…", attempt, explain_err)
            repair_payload = _REPAIR_TEMPLATE.format(error=str(explain_err), sql=sql)
            repair_messages.append({"role": "assistant", "content": sql})
            repair_messages.append({"role": "user",      "content": repair_payload})

            try:
                raw_repair = _call_ollama(repair_messages)
                repaired   = _extract_sql(raw_repair)
                if repaired:
                    sql = _harden_sql(repaired)
                else:
                    logger.warning("Route D self-repair produced no valid SQL.")
                    break
            except Exception as repair_exc:
                logger.warning("Route D self-repair Ollama call failed: %s", repair_exc)
                break

    return {"success": True, "sql": sql}


def _generate_observation(prompt: str, sql: str, df: pd.DataFrame) -> str:
    """
    Call Ollama (INSIGHT_MODEL) for a concise analytical observation.
    Uses a direct requests.post with INSIGHT_MODEL (different model from SQL_MODEL),
    so we cannot reuse _call_ollama which is hardcoded to SQL_MODEL.
    """
    limit    = 200
    data_csv = df.head(limit).to_csv(index=False)
    total    = len(df)

    user_payload = (
        f"### USER QUESTION:\n{prompt}\n\n"
        f"### SQL QUERY USED:\n{sql}\n\n"
        f"### QUERY RESULT (first {min(total, limit)} of {total} rows):\n{data_csv}\n\n"
        "Provide your analytical observation now:"
    )

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model":    INSIGHT_MODEL,
            "messages": [
                {"role": "system", "content": _OBS_SYSTEM_PROMPT},
                {"role": "user",   "content": user_payload},
            ],
            "stream":  False,
            "options": {"temperature": 0.1},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _validate_sql(sql: str, tables: list[str]) -> tuple[bool, str]:
    """Safety check — no write keywords, must reference a known table."""
    sql_upper = sql.upper()
    dangerous = ("DROP", "DELETE", "TRUNCATE", "INSERT", "UPDATE", "ALTER", "CREATE", "REPLACE")
    for kw in dangerous:
        if re.search(rf"\b{kw}\b", sql_upper):
            return False, f"SQL contains a disallowed keyword: {kw}"

    if not any(
        t.lower() in sql.lower() or f'"{t}"'.lower() in sql.lower()
        for t in tables
    ):
        return False, "Generated SQL does not reference any loaded table."

    return True, ""


# ─── public entry point ───────────────────────────────────────────────────────

def run(
    conn,
    tables: list[str],
    prompt: str,
    ddl_schema: str | None = None,   # kept for API compatibility, no longer used
    route_label: str = "statistical",
) -> RouteDResult:
    """
    Route D pipeline:
      1. Build live column dict (same as Route B — exact names + delimiter flags)
      2. Generate SQL + self-repair via EXPLAIN (up to 3 attempts)
      3. Safety-validate SQL
      4. Execute against DuckDB
      5. Generate AI observation (only on non-empty results)
    """
    if not tables:
        return RouteDResult(success=False, route=route_label, error="No tables loaded.")

    # ── Step 1: Build column dict (live DESCRIBE, same quality as Route B) ────
    column_dict = _build_column_dict(conn, tables)

    # ── Step 2: Generate + self-repair SQL ────────────────────────────────────
    sql_result = _generate_sql(prompt, column_dict, tables, conn)
    if not sql_result["success"]:
        return RouteDResult(
            success=False, route=route_label,
            error=f"SQL generation failed: {sql_result['error']}",
        )

    sql = sql_result["sql"]

    # ── Step 3: Safety validation ─────────────────────────────────────────────
    ok, reason = _validate_sql(sql, tables)
    if not ok:
        return RouteDResult(
            success=False, route=route_label,
            error=f"SQL validation failed: {reason}",
            sql=sql,
        )

    # ── Step 4: Execute ───────────────────────────────────────────────────────
    try:
        df = conn.execute(sql).df()
    except Exception as exc:
        logger.error("Route D execution error: %s", exc)
        return RouteDResult(
            success=False, route=route_label,
            error=f"Query execution failed: {exc}",
            sql=sql,
        )

    if df.empty:
        return RouteDResult(
            success=True, route=route_label,
            answer="The query ran successfully but returned no rows.",
            dataframe=df,
            sql=sql,
        )

    # ── Step 5: AI observation ────────────────────────────────────────────────
    try:
        observation = _generate_observation(prompt, sql, df)
    except Exception as exc:
        logger.warning("Route D observation failed (non-fatal): %s", exc)
        observation = f"Query returned {len(df):,} row(s)."

    return RouteDResult(
        success=True,
        route=route_label,
        answer=observation,
        dataframe=df,
        sql=sql,
        details={"tables_used": tables, "row_count": len(df)},
    )