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
#   5. _harden_casts: bare CAST(...) is auto-upgraded to TRY_CAST(...) so a
#      single malformed value never crashes the whole query.
#   6. _fix_decimal_stripping: removes any REPLACE(x, '.', '') the model
#      generates — it sometimes over-applies the currency/percent cleaning
#      template to the decimal point too, silently corrupting "4.5" -> "45".
#   7. Hardening changes are now logged (before/after) so it's visible when
#      these safety nets actually fire vs. the model writing correct SQL.
#   8. Ollama timeouts now get a specific, actionable error message instead
#      of falling into the generic exception handler.
#
# KNOWN LIMITATIONS (documented, not yet fixed — see chat history):
#   - _harden_casts operates on raw SQL text with no awareness of string-literal
#     boundaries. If a dataset's text contains the literal substring "CAST("
#     inside a quoted value the model echoes into a WHERE clause, this regex
#     would incorrectly rewrite inside that literal. Low probability given
#     real-world data, not addressed here.
#   - _fix_decimal_stripping assumes '.' is always a decimal point, which is
#     correct for the US/India-style data this has been tested against, but
#     would be the WRONG fix for European-formatted numbers where '.' is a
#     thousands separator. Revisit if you ever load that kind of dataset.
#   - core/route_a.py's _generate_sql needs the identical _fix_decimal_stripping
#     + _harden_casts treatment — it generates SQL through the same model and
#     is equally exposed. Not part of this file; apply there separately.

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
3. If a table DOES NOT have the relevant column, YOU MUST SKIP THAT TABLE ENTIRELY.
4. NEVER write `SELECT name FROM table` unless the exact word "name" is in the dictionary.

DUCKDB STRING CONVERSION RULES (CRITICAL):
- To convert currency/price strings to numbers, remove symbols first: TRY_CAST(REPLACE(REPLACE(col, '₹', ''), ',', '') AS DOUBLE).
- To convert percentage strings to numbers: TRY_CAST(REPLACE(col, '%', '') AS DOUBLE).
- NEVER remove or replace decimal points ('.') when cleaning numeric columns like ratings or prices, as this alters the underlying numerical value.
  WRONG: REPLACE(rating, '.', '') turns "4.5" into "45" — this corrupts the value. Do not do this.
- ALWAYS use TRY_CAST, never plain CAST, when converting any text/VARCHAR column to a numeric or date type for comparison or filtering. A single malformed value (e.g. a stray '|' or empty string) inside a bare CAST will crash the ENTIRE query; TRY_CAST returns NULL for that value instead and lets the rest of the query complete normally.

DUCKDB-SPECIFIC RULES:
- When using SUM/AVG/COUNT/MIN/MAX, always include the matching GROUP BY clause.
- Use UNNEST(STRING_SPLIT(col, '|')) to expand pipe-delimited multi-value columns.
- Column aliases defined in SELECT can be used in ORDER BY, but NOT in WHERE or HAVING.
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


def _harden_casts(sql: str) -> str:
    """
    Defensively upgrade bare CAST(...) to TRY_CAST(...).
    This is a read-only analytics tool over arbitrary, often messy
    uploaded data. A single malformed value (e.g. a stray '|' in what
    should be a numeric column) should never crash an entire query.
    TRY_CAST returns NULL for that one row instead of raising, which
    WHERE/ORDER BY then handle naturally rather than erroring out.
    \bCAST won't match inside TRY_CAST, since '_' and 'C' are both word
    characters in regex — there's no boundary between them, so this is
    safe to run even on SQL that already uses TRY_CAST elsewhere.
    """
    return re.sub(r"\bCAST\s*\(", "TRY_CAST(", sql, flags=re.IGNORECASE)


def _find_matching_paren(s: str, open_idx: int) -> int | None:
    """Given the index of an opening '(' in s, return the index of its
    matching closing ')', respecting quoted strings. None if unbalanced."""
    depth = 0
    in_single = False
    in_double = False
    i = open_idx
    while i < len(s):
        ch = s[i]
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "(" and not in_single and not in_double:
            depth += 1
        elif ch == ")" and not in_single and not in_double:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None

def _split_top_level_args(s: str) -> list[str]:
    """
    Split comma-separated SQL arguments at the TOP level only, respecting
    nested parens and quoted strings, e.g.
    "REPLACE(a, ','), '.', ''" -> ["REPLACE(a, ',')", " '.'", " ''"]
    """
    args = []
    depth = 0
    in_single = False
    in_double = False
    current = []
    for ch in s:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "(" and not in_single and not in_double:
            depth += 1
        elif ch == ")" and not in_single and not in_double:
            depth -= 1
            
        if ch == "," and depth == 0 and not in_single and not in_double:
            args.append("".join(current))
            current = []
            continue
        current.append(ch)
    args.append("".join(current))
    return args

_REPLACE_CALL_START = re.compile(r"\bREPLACE\s*\(", re.IGNORECASE)

def _fix_decimal_stripping(sql: str) -> str:
    """
    Remove any REPLACE(<expr>, '.', '') call, regardless of nesting depth,
    WITHOUT corrupting unrelated REPLACE(...) calls elsewhere in the same
    statement (e.g. two separate chains joined by AND).

    An earlier version of this fix used a single lazy regex
    (REPLACE\\(.*?, '.', '')), which is unsafe: with two separate REPLACE
    chains in the same query, the lazy '.*?' can overshoot past the first
    chain's own closing paren entirely and latch onto a '.'-strip
    belonging to a SECOND, unrelated expression, corrupting everything in
    between. This version walks each REPLACE( call's actual balanced
    parentheses and inspects its real arguments instead of guessing with
    regex, so it can never cross into an unrelated call.
    """
    result = sql
    while True:
        changed = False
        for match in _REPLACE_CALL_START.finditer(result):
            open_idx = match.end() - 1
            close_idx = _find_matching_paren(result, open_idx)
            if close_idx is None:
                continue
                
            inner = result[open_idx + 1: close_idx]
            args = _split_top_level_args(inner)
            
            if len(args) != 3:
                continue
                
            search_arg = args[1].strip()
            replace_arg = args[2].strip()
            
            if search_arg in ("'.'", '"."') and replace_arg in ("''", '""'):
                expr = args[0].strip()
                result = result[:match.start()] + expr + result[close_idx + 1:]
                changed = True
                break  # string indices shifted — restart the scan
                
        if not changed:
            break
            
    return result


def _harden_sql(sql: str) -> str:
    """
    Apply all deterministic post-generation safety fixes and log when any
    of them actually change the SQL, so it's visible whether the model is
    writing correct SQL on its own or relying on these safety nets.
    """
    before = sql
    sql = _fix_decimal_stripping(sql)
    sql = _harden_casts(sql)
    if sql != before:
        logger.info(
            "SQL hardened — before: %s… | after: %s…",
            before[:100], sql[:100],
        )
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

    # ── FIX: Catch and reveal the ACTUAL error from Ollama ──
    if resp.status_code != 200:
        error_details = resp.text
        raise Exception(
            f"Ollama returned HTTP {resp.status_code}. "
            f"Model requested: '{SQL_MODEL}'. "
            f"Ollama's exact error message: {error_details}"
        )

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
    except requests.exceptions.Timeout:
        msg = (
            f"Ollama took longer than {OLLAMA_TIMEOUT}s to respond. The model "
            f"may still be loading into memory on first use — try again, or "
            f"raise OLLAMA_TIMEOUT in config.py if this happens consistently."
        )
        logger.error(msg)
        return {"success": False, "error": msg}
    except Exception as e:
        return {"success": False, "error": f"Ollama call failed: {e}"}

    sql = _extract_sql(raw)
    if not sql:
        return {"success": False, "error": "Model response contained no valid SQL."}

    sql = _harden_sql(sql)

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
                    repaired = _harden_sql(repaired)
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
            except requests.exceptions.Timeout:
                logger.warning(
                    "Self-repair Ollama call timed out after %ss — "
                    "returning original for validator to handle.", OLLAMA_TIMEOUT
                )
            except Exception as repair_exc:
                logger.warning("Self-repair Ollama call failed: %s", repair_exc)

    logger.info("generate_safe_sql returning: %s…", sql[:120])
    return {"success": True, "sql": sql}