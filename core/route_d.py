# ─────────────────────────────────────────────────────────────────────────────
# core/route_d.py  –  Route D: Statistical / Analytical SQL answers
#
# Handles all miscellaneous analytical queries that are NOT:
#   - visualizations (Route A)
#   - plain-English lookups (Route B / sql_answer fallback)
#   - metadata / keyword questions (Route C)
#
# Pipeline:
#   1. Use Ollama (qwen2.5-coder) to generate DuckDB SQL from the prompt
#   2. Validate + execute the SQL against DuckDB
#   3. Use Ollama (llama3.1) to produce a concise AI observation / summary
#
# Returns RouteDResult with:
#   - sql        : the generated query (shown in UI expander)
#   - dataframe  : the raw result set
#   - answer     : AI-generated natural-language observation
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

logger = logging.getLogger(__name__)


@dataclass
class RouteDResult:
    success: bool
    route: str
    answer: str = ""
    dataframe: pd.DataFrame | None = None
    sql: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# ─── helpers ─────────────────────────────────────────────────────────────────

def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    raw = re.sub(r"```[a-zA-Z]*", "", raw)
    return raw.replace("```", "").strip()


def _build_ddl_hint(conn, tables: list[str]) -> str:
    """
    Generate a compact DDL string for the SQL-generation prompt.
    Includes a pipe-category hint if any column contains '|'-delimited values.
    """
    parts = []
    for t in tables:
        try:
            rows = conn.execute(f"DESCRIBE {_qident(t)}").fetchall()
        except Exception:
            continue
        col_defs = ", ".join(f"{_qident(r[0])} {r[1]}" for r in rows)
        parts.append(f"CREATE TABLE {_qident(t)} ({col_defs});")

    ddl = "\n".join(parts)

    # Data dictionary hint for pipe-delimited category columns
    ddl += (
        "\n\n-- DATA DICTIONARY HINTS --\n"
        "-- Some columns (e.g. 'category') store multiple sub-values separated by '|'.\n"
        "-- To list or count individual sub-values use:\n"
        "--   UNNEST(STRING_SPLIT(category, '|'))\n"
    )
    return ddl


def _generate_sql(prompt: str, ddl: str, tables: list[str]) -> dict:
    """
    Ask Ollama (SQL_MODEL) to produce a single DuckDB SQL query.
    Returns {"success": True, "sql": "..."} or {"success": False, "error": "..."}.
    """
    system_prompt = (
        "You are an expert DuckDB SQL developer.\n"
        "Generate a single, valid, executable DuckDB SQL query that answers the user's analytical request.\n\n"
        "STRICT RULES:\n"
        "- Output ONLY raw SQL — no markdown fences, no explanations, no comments.\n"
        "- Match table and column names exactly as shown in the DDL schema.\n"
        "- For statistical queries include appropriate aggregations (AVG, STDDEV_POP, PERCENTILE_CONT, etc.).\n"
        "- For ranking queries use ORDER BY … LIMIT.\n"
        "- For distribution / frequency use GROUP BY … ORDER BY … DESC.\n"
        "- For correlation between two numeric columns use a sub-select or CTE; DuckDB has no built-in CORR() — use:\n"
        "    (SUM(a*b) - COUNT(*)*AVG(a)*AVG(b)) / (COUNT(*)*STDDEV_POP(a)*STDDEV_POP(b)) AS correlation\n"
        "- If the user asks about categories split by '|', apply UNNEST(STRING_SPLIT(col, '|')).\n"
        "- Never use semicolons inside a single query.\n"
        "- Keep the query concise and efficient.\n"
    )

    user_payload = (
        f"### TABLE SCHEMA:\n{ddl}\n\n"
        f"### USER REQUEST:\n{prompt}\n\n"
        "Write the DuckDB SQL query now:"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": SQL_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_payload},
                ],
                "stream":  False,
                "options": {"temperature": 0.0},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        sql = _clean_sql(raw)
        if not sql:
            return {"success": False, "error": "Model returned an empty response."}
        return {"success": True, "sql": sql}

    except Exception as exc:
        logger.error("Route D SQL generation failed: %s", exc)
        return {"success": False, "error": str(exc)}


def _generate_observation(prompt: str, sql: str, df: pd.DataFrame, ddl: str) -> str:
    """
    Ask Ollama (INSIGHT_MODEL) to produce a concise analytical observation
    based on the query result.  Returns plain text or raises on failure.
    """
    total_rows = len(df)
    limit      = 200
    data_csv   = df.head(limit).to_csv(index=False)

    system_prompt = (
        "You are a precise data analyst.\n"
        "Your job is to provide a clear, concise statistical observation based on the data provided.\n\n"
        "RULES:\n"
        "1. Be specific — mention actual numbers, ranges, or trends you see in the data.\n"
        "2. Keep the response under 4 sentences.\n"
        "3. Do NOT restate the SQL or schema.\n"
        "4. Do NOT make up data that is not in the result set.\n"
        "5. If the result is a single number, interpret it in plain language.\n"
    )

    user_payload = (
        f"### USER QUESTION:\n{prompt}\n\n"
        f"### SQL QUERY USED:\n{sql}\n\n"
        f"### QUERY RESULT (first {min(total_rows, limit)} of {total_rows} rows):\n{data_csv}\n\n"
        "Provide your analytical observation now:"
    )

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model":   INSIGHT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_payload},
            ],
            "stream":  False,
            "options": {"temperature": 0.1},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _validate_sql(conn, sql: str, tables: list[str]) -> tuple[bool, str]:
    """
    Lightweight validation:
      - Must reference at least one known table
      - Must not contain dangerous write keywords
    """
    sql_upper = sql.upper()
    dangerous = ("DROP", "DELETE", "TRUNCATE", "INSERT", "UPDATE", "ALTER", "CREATE", "REPLACE")
    for kw in dangerous:
        if re.search(rf"\b{kw}\b", sql_upper):
            return False, f"SQL contains a disallowed keyword: {kw}"

    table_refs = [t for t in tables if t.lower() in sql.lower() or f'"{t}"'.lower() in sql.lower()]
    if not table_refs:
        return False, "Generated SQL does not reference any loaded table."

    return True, ""


# ─── public entry point ───────────────────────────────────────────────────────

def run(
    conn,
    tables: list[str],
    prompt: str,
    ddl_schema: str | None = None,
    route_label: str = "statistical",
) -> RouteDResult:
    """
    Route D pipeline:
      1. Generate SQL via Ollama
      2. Validate safety
      3. Execute against DuckDB
      4. Generate AI observation via Ollama
    """
    if not tables:
        return RouteDResult(success=False, route=route_label, error="No tables loaded.")

    # Build DDL if not supplied by caller
    ddl = ddl_schema if ddl_schema else _build_ddl_hint(conn, tables)

    # ── Step 1: Generate SQL ──────────────────────────────────────────────────
    sql_result = _generate_sql(prompt, ddl, tables)
    if not sql_result["success"]:
        return RouteDResult(
            success=False, route=route_label,
            error=f"SQL generation failed: {sql_result['error']}",
        )

    sql = sql_result["sql"]
    logger.info("Route D generated SQL: %s", sql)

    # ── Step 2: Validate SQL ──────────────────────────────────────────────────
    ok, reason = _validate_sql(conn, sql, tables)
    if not ok:
        return RouteDResult(
            success=False, route=route_label,
            error=f"SQL validation failed: {reason}",
            sql=sql,
        )

    # ── Step 3: Execute SQL ───────────────────────────────────────────────────
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

    # ── Step 4: Generate AI observation ──────────────────────────────────────
    try:
        observation = _generate_observation(prompt, sql, df, ddl)
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
