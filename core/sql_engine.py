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
#   9. _detect_delimiter: _build_column_dict now flags internally-delimited
#      VARCHAR columns (e.g. a pipe-joined "category" hierarchy) directly in
#      the column dictionary, so the model knows to UNNEST(STRING_SPLIT(...))
#      instead of treating the whole joined string as one literal value.

import logging
import re
import requests

from config import OLLAMA_BASE_URL, SQL_MODEL, OLLAMA_TIMEOUT

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
- To convert comma-separated number strings (like rating_count) to numbers: TRY_CAST(REPLACE(col, ',', '') AS DOUBLE).
- NEVER remove or replace decimal points ('.') when cleaning numeric columns like ratings or prices, as this alters the underlying numerical value.
  WRONG: REPLACE(rating, '.', '') turns "4.5" into "45" — this corrupts the value. Do not do this.
- ALWAYS use TRY_CAST, never plain CAST, when converting any text/VARCHAR column.
- Never include the '%' symbol in mathematical comparisons in the WHERE/HAVING clause (e.g., use > 50 instead of > 50%).
- ALWAYS clean and TRY_CAST string columns to DOUBLE *before* doing math, filtering in WHERE/HAVING clauses, or sorting in ORDER BY. Never compare a text column directly to a string number (e.g., `rating_count > '1,000'` is WRONG. It must be `TRY_CAST(...) > 1000`).

DUCKDB-SPECIFIC RULES:
- When using SUM/AVG/COUNT/MIN/MAX, always include the matching GROUP BY clause.
- If the COLUMN DICTIONARY marks a column as [DELIMITED with 'X'], you MUST ALWAYS expand it using UNNEST(STRING_SPLIT(col, 'X')) before doing SELECT DISTINCT, COUNT, GROUP BY, or filtering. Never select or group the raw joined string.
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

# ── Delimiter detection ─────────────────────────────────────────────────────
_DELIMITER_CANDIDATES = ["|", ";", "/", ">"]
_DELIMITER_MIN_RATIO = 0.5
_DELIMITER_SAMPLE_LIMIT = 200


def _detect_delimiter(conn, table_name: str, col: str) -> str | None:
    """
    Check whether a VARCHAR column's values are internally delimited
    (e.g. "Computers&Accessories|Cables|USBCables"). Without this, the
    model has no way to know a column needs UNNEST(STRING_SPLIT(...))
    instead of a direct equality/IN/DISTINCT match, and will silently
    generate SQL that treats the full joined string as one value.
    """
    escaped = f'"{col}"'
    try:
        sample = conn.execute(
            f"SELECT {escaped} FROM {table_name} "
            f"WHERE {escaped} IS NOT NULL LIMIT {_DELIMITER_SAMPLE_LIMIT}"
        ).fetchall()
    except Exception:
        return None

    values = [str(r[0]) for r in sample if r[0] is not None]
    if not values:
        return None

    for delim in _DELIMITER_CANDIDATES:
        hits = sum(1 for v in values if delim in v)
        if hits / len(values) >= _DELIMITER_MIN_RATIO:
            return delim
    return None


def _build_column_dict(conn, table_names: list[str]) -> str:
    """
    Query DuckDB DESCRIBE for each table and build an explicit column
    dictionary. VARCHAR/TEXT columns are additionally checked for internal
    delimiters (e.g. a pipe-joined category hierarchy) and flagged inline
    so the model knows to UNNEST(STRING_SPLIT(...)) rather than treat the
    full joined string as one literal value.
    """
    lines = []
    for table in table_names:
        try:
            rows = conn.execute(f"DESCRIBE {table}").fetchall()
            col_parts = []
            for r in rows:
                col_name, col_type = r[0], r[1]
                entry = f'"{col_name}" ({col_type})'
                if any(t in col_type.upper() for t in ("VARCHAR", "TEXT", "CHAR")):
                    delim = _detect_delimiter(conn, table, col_name)
                    if delim:
                        entry += (
                            f" [DELIMITED with '{delim}' — ALWAYS expand using "
                            f"UNNEST(STRING_SPLIT(\"{col_name}\", '{delim}')) before "
                            f"doing SELECT DISTINCT, GROUP BY, or filtering]"
                        )
                col_parts.append(entry)
            lines.append(f'Table "{table}": [ {", ".join(col_parts)} ]')
        except Exception as e:
            lines.append(f'Table "{table}": (could not describe — {e})')
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    text = re.sub(r"```sql\s*", "", text, flags=re.I)
    text = re.sub(r"```", "", text)
    return text.strip()


def _extract_sql(raw: str) -> str | None:
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
    return re.sub(r"\bCAST\s*\(", "TRY_CAST(", sql, flags=re.IGNORECASE)


def _find_matching_paren(s: str, open_idx: int) -> int | None:
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
                break  
                
        if not changed:
            break
            
    return result


def _harden_sql(sql: str) -> str:
    before = sql
    sql = _fix_decimal_stripping(sql)
    sql = _harden_casts(sql)
    if sql != before:
        print(f"   {_YELLOW}⚙️ SQL Hardened (Regex safety nets applied){_R}", flush=True)
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
            "num_ctx": 8192,
        },
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )

    if resp.status_code != 200:
        error_details = resp.text
        raise Exception(
            f"Ollama returned HTTP {resp.status_code}. "
            f"Model requested: '{SQL_MODEL}'. "
            f"Ollama's exact error message: {error_details}"
        )

    return resp.json()["message"]["content"].strip()


def _print_fail(msg: str) -> None:
    print(f"\n{_RED}{_BOLD}  ROUTE B FAILED: {msg}{_R}\n", flush=True)


def generate_safe_sql(
    prompt: str,
    ddl_schema: str,      
    table_name: str,      
    db_session=None,      
) -> dict:
    conn = db_session
    all_tables = [t.strip() for t in table_name.split(",") if t.strip()] if table_name else []

    print(
        f"\n{_BOLD}{'═'*55}{_R}\n"
        f"  {_BLUE}{_BOLD}ROUTE B — DATA ENGINE (SQL){_R}\n"
        f"  tables={_CYAN}{table_name}{_R}  prompt='{prompt[:60]}'\n"
        f"{_BOLD}{'═'*55}{_R}",
        flush=True,
    )

    if conn and all_tables:
        column_dict = _build_column_dict(conn, all_tables)
    else:
        column_dict = f"(DDL schema — use column names from this only):\n{ddl_schema}"

    user_payload = _USER_TEMPLATE.format(
        column_dict=column_dict,
        prompt=prompt,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_payload},
    ]

    print(
        f"\n{_BOLD}{_BLUE}┌── Route B · Stage 1: SQL Generation{_R}\n"
        f"   model={_DIM}{SQL_MODEL}{_R}",
        flush=True,
    )

    try:
        raw = _call_ollama(messages)
    except requests.exceptions.ConnectionError:
        msg = f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Make sure Ollama is running: `ollama serve`"
        logger.error(msg)
        _print_fail(msg)
        return {"success": False, "error": msg}
    except requests.exceptions.Timeout:
        msg = f"Ollama took longer than {OLLAMA_TIMEOUT}s to respond."
        logger.error(msg)
        _print_fail(msg)
        return {"success": False, "error": msg}
    except Exception as e:
        _print_fail(f"Ollama call failed: {e}")
        return {"success": False, "error": f"Ollama call failed: {e}"}

    sql = _extract_sql(raw)
    if not sql:
        _print_fail("Model response contained no valid SQL.")
        return {"success": False, "error": "Model response contained no valid SQL."}

    sql = _harden_sql(sql)
    logger.info("SQL first attempt (%d chars): %s…", len(sql), sql[:100])
    
    print(f"   sql={_DIM}{sql[:120]}…{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)

    # ── Self-repair pass ───────────────────────────────────────────────────────
    if conn:
        print(
            f"\n{_BOLD}{_BLUE}┌── Route B · Stage 2: Validation & Repair{_R}",
            flush=True,
        )
        try:
            conn.execute(f"EXPLAIN {sql}")
            logger.info("SQL passed EXPLAIN on first attempt.")
            print(f"   {_GREEN}EXPLAIN PASS{_R}: SQL is valid\n{_BOLD}{_BLUE}└──{_R}", flush=True)
        except Exception as explain_err:
            logger.warning("SQL failed EXPLAIN (%s) — attempting self-repair…", explain_err)
            print(f"   {_YELLOW}EXPLAIN FAIL ({explain_err}) — attempting self-repair…{_R}", flush=True)

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
                        print(f"   {_GREEN}Self-repair succeeded{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
                    except Exception as repair_err:
                        logger.warning("Self-repair also failed EXPLAIN (%s)", repair_err)
                        print(f"   {_RED}Self-repair failed — passing original to validator{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
                else:
                    logger.warning("Self-repair produced no valid SQL.")
                    print(f"   {_RED}Self-repair produced no valid SQL{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)
            except Exception as repair_exc:
                logger.warning("Self-repair Ollama call failed: %s", repair_exc)
                print(f"   {_RED}Self-repair Ollama call failed{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)

    print(
        f"\n{_BOLD}{_GREEN}{'═'*55}{_R}\n"
        f"  {_GREEN}{_BOLD}ROUTE B COMPLETE ✓{_R}\n"
        f"{_BOLD}{_GREEN}{'═'*55}{_R}\n",
        flush=True,
    )

    logger.info("generate_safe_sql returning: %s…", sql[:120])
    return {"success": True, "sql": sql}