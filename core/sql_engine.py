# core/sql_engine.py  –  100% local Ollama, zero data leaves your machine
#
# Key improvements over original:
#   1. Injects a live COLUMN DICTIONARY (from DuckDB DESCRIBE) into the prompt
#      so the model sees exact column names — eliminates hallucinated names.
#   2. num_predict raised 512 → 1024 so long queries are never truncated.
#   3. Self-repair pass: on DuckDB EXPLAIN failure, sends the error back to
#      qwen2.5-coder and asks it to fix the query (one retry, fully local).
#   4. Removed the hardcoded Amazon 'category|pipe' hint — that was dataset-
#      specific and confused the model on every other dataset.

import logging
import re
import requests

from config import OLLAMA_BASE_URL, SQL_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an expert DuckDB SQL developer. Your only job is to write a single \
valid DuckDB SELECT query that answers the user's question.

STRICT OUTPUT RULES:
- Output ONLY the raw SQL query. No markdown, no backticks, no explanation.
- Start with SELECT or WITH. End with a semicolon.
- Never use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, ATTACH, PRAGMA, COPY.

COLUMN RULES — THIS IS CRITICAL:
- You will be given a COLUMN DICTIONARY listing every table and its exact column names.
- You MUST use column names EXACTLY as they appear in that dictionary.
- Never invent, guess, abbreviate, or rename columns.
- If a column name contains spaces or special characters, wrap it in double-quotes: "My Column".

MULTI-TABLE & "ENTIRE FILE" QUERIES:
1. If the user asks about the whole file or all data, use UNION ALL.
2. CRITICAL FILTER: You MUST read the COLUMN DICTIONARY before adding a table to the UNION. 
3. If a table DOES NOT have the relevant column (e.g., you need employee names, but the table only has "Session Name", "Track Name", or metrics), YOU MUST SKIP THAT TABLE ENTIRELY.
4. NEVER write `SELECT name FROM table` unless the exact word "name" is in the dictionary for that specific table. If it uses "EMPNAME" or "Employee Name", use that exactly and alias it (e.g., SELECT "EMPNAME" AS name).

DUCKDB-SPECIFIC RULES:
- When using SUM/AVG/COUNT/MIN/MAX, always include the matching GROUP BY clause.
- Use UNNEST(STRING_SPLIT(col, '|')) to expand pipe-delimited multi-value columns.
- Use TRY_CAST(col AS DOUBLE) to safely convert text columns to numbers.
- Use STRFTIME('%Y-%m', col) for month-level date grouping.
- Column aliases defined in SELECT can be used in ORDER BY, but NOT in WHERE or HAVING.
- Do NOT wrap a simple aggregation query in an unnecessary subquery.
- LIMIT always goes after ORDER BY.
"""

_USER_TEMPLATE = """\
### COLUMN DICTIONARY (use ONLY these exact column names):
{column_dict}

### USER QUESTION:
{prompt}

SQL query:
"""

_REPAIR_TEMPLATE = """\
The SQL query you wrote caused this DuckDB error:

  {error}

Here is the broken query:
{sql}

Write a corrected DuckDB SQL query that fixes this error exactly.
Output ONLY the raw SQL, nothing else.
"""


def _build_column_dict(conn, table_names: list[str]) -> str:
    """
    Query DuckDB DESCRIBE for each table and build an explicit column dictionary.
    This is the key fix — the model sees exact column names and cannot hallucinate.
    """
    lines = []
    for table in table_names:
        try:
            rows = conn.execute(f"DESCRIBE {table}").fetchall()
            col_list = ", ".join(f'"{r[0]}" ({r[1]})' for r in rows)
            lines.append(f'Table "{table}": [ {col_list} ]')
        except Exception as e:
            lines.append(f'Table "{table}": (could not describe — {e})')
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    text = re.sub(r"```sql\s*", "", text, flags=re.I)
    text = re.sub(r"```", "", text)
    return text.strip()


def _extract_sql(raw: str) -> str | None:
    """Pull the first SELECT/WITH block out of the model response."""
    raw = _strip_fences(raw)
    m = re.search(r"(SELECT|WITH)\b[\s\S]*", raw, flags=re.I)
    if not m:
        return None
    sql = m.group(0).strip()
    if ";" in sql:
        sql = sql[:sql.index(";") + 1]
    elif not sql.endswith(";"):
        sql += ";"
    return sql


def _call_ollama(messages: list[dict], max_tokens: int = 1024) -> str:
    payload = {
        "model":    SQL_MODEL,
        "messages": messages,
        "stream":   False,
        "options":  {
            "temperature": 0.0,
            "num_predict": max_tokens,
        },
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def generate_safe_sql(
    prompt: str,
    ddl_schema: str,      # kept for backward-compatibility
    table_name: str,      # comma-separated table names (primary first)
    db_session=None,      # DuckDB connection — used to build live column dict
) -> dict:
    """
    Generate a valid DuckDB SELECT query for *prompt*.

    Returns:
        {"success": True,  "sql": "<query>"}
        {"success": False, "error": "<message>"}
    """
    conn = db_session
    all_tables = [t.strip() for t in table_name.split(",") if t.strip()] if table_name else []

    # ── Build column dictionary from live DuckDB metadata ─────────────────────
    if conn and all_tables:
        column_dict = _build_column_dict(conn, all_tables)
    else:
        # Fallback: use the raw DDL text if no live connection
        column_dict = f"(DDL schema — use column names from this only):\n{ddl_schema}"

    user_payload = _USER_TEMPLATE.format(
        column_dict=column_dict,
        prompt=prompt,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_payload},
    ]

    # ── First attempt ──────────────────────────────────────────────────────────
    try:
        raw = _call_ollama(messages)
    except requests.exceptions.ConnectionError:
        msg = (
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: `ollama serve`"
        )
        logger.error(msg)
        return {"success": False, "error": msg}
    except Exception as e:
        return {"success": False, "error": f"Ollama call failed: {e}"}

    sql = _extract_sql(raw)
    if not sql:
        return {"success": False, "error": "Model response contained no valid SQL."}

    logger.info("SQL first attempt (%d chars): %s…", len(sql), sql[:100])

    # ── Self-repair pass ───────────────────────────────────────────────────────
    # If DuckDB EXPLAIN rejects the query, send the error back to the model
    # and ask it to fix the query. One retry, fully local, no data sent anywhere.
    if conn:
        try:
            conn.execute(f"EXPLAIN {sql}")
            logger.info("SQL passed EXPLAIN on first attempt.")
        except Exception as explain_err:
            logger.warning("SQL failed EXPLAIN (%s) — attempting self-repair…", explain_err)

            repair_payload = _REPAIR_TEMPLATE.format(error=str(explain_err), sql=sql)
            repair_messages = messages + [
                {"role": "assistant", "content": sql},
                {"role": "user",      "content": repair_payload},
            ]
            try:
                raw_repair = _call_ollama(repair_messages, max_tokens=1024)
                repaired = _extract_sql(raw_repair)
                if repaired:
                    try:
                        conn.execute(f"EXPLAIN {repaired}")
                        logger.info("Self-repair succeeded: %s…", repaired[:100])
                        sql = repaired
                    except Exception as repair_err:
                        logger.warning(
                            "Self-repair also failed EXPLAIN (%s) — "
                            "returning original for validator to handle.", repair_err
                        )
                else:
                    logger.warning("Self-repair produced no valid SQL.")
            except Exception as repair_exc:
                logger.warning("Self-repair Ollama call failed: %s", repair_exc)

    logger.info("generate_safe_sql returning: %s…", sql[:120])
    return {"success": True, "sql": sql}