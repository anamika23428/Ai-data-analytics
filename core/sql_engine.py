# core/sql_engine.py  –  100% local Ollama, zero data leaves your machine
#
# Key improvements over original:
#   1. Injects a live COLUMN DICTIONARY (from DuckDB DESCRIBE) into the prompt
#      so the model sees exact column names — eliminates hallucinated names.
#   2. num_predict raised 512 → 1024 so long queries are never truncated.
#   3. Self-repair pass: on DuckDB EXPLAIN failure, sends the error back to
#      qwen2.5-coder and asks it to fix the query (up to 3 attempts).
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
#  10. _fix_unnest_in_where: deterministically rewrites the model's common
#      mistake of list_contains(UNNEST(STRING_SPLIT(...))) in WHERE clauses,
#      which DuckDB rejects. Enforced in code since self-repair consistently
#      fails to break this habit across all 3 repair attempts.
#  11. _humanize_error: converts raw technical error strings into plain-English
#      user-facing messages using llama3.2:3b (already warm from the routing
#      step). Falls back to a pattern-based plain-English classifier (NOT the
#      broken regex) when the LLM call fails — raw technical text is NEVER
#      shown to the user regardless of whether the LLM is available.
#
#      FIX vs previous version:
#      The old fallback `re.sub(r"^[\w.]+Error:\s*", "", technical_msg)` did
#      nothing when the error string did not start with a bare "SomeError:"
#      prefix. validator.py prepends "Generated SQL could not be parsed or
#      validated by DuckDB: Binder Error: ..." which starts with "Generated",
#      so the regex matched nothing and the full raw DuckDB error leaked
#      through verbatim — exactly what the screenshot showed.
#      Replaced with a pattern-based classifier that recognises common DuckDB
#      error shapes (column not found, type mismatch, syntax error, etc.) and
#      returns a plain-English sentence for each, even when the LLM is down.
#
#  12. _fix_unnest_with_aggregation: rewrites the model's common mistake of
#      mixing UNNEST(STRING_SPLIT(...)) and aggregate functions (AVG/SUM/COUNT/
#      MIN/MAX) at the same SELECT level, which DuckDB rejects. Restructures
#      into the correct two-level subquery pattern automatically.

import logging
import re
import requests

from config import OLLAMA_BASE_URL, SQL_MODEL, OLLAMA_TIMEOUT, ROUTER_MODEL

logger = logging.getLogger(__name__)

# ── ANSI colours ──────────────────────────────────────────────────────────────
_R      = "\033[0m"
_BOLD   = "\033[1m"
_BLUE   = "\033[94m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_DIM    = "\033[2m"
_CYAN   = "\033[96m"

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

DELIMITED COLUMN RULES (CRITICAL — READ CAREFULLY):
If the COLUMN DICTIONARY marks a column as [DELIMITED with 'X'], it stores
multiple segments joined by 'X' in a single string.

There are EXACTLY TWO ways to use a delimited column. Choose based on context:

1. To LIST or GROUP segments (SELECT DISTINCT, GROUP BY):
   Use UNNEST(STRING_SPLIT(col, 'X')) in the SELECT or FROM clause.
   CORRECT:
     SELECT DISTINCT UNNEST(STRING_SPLIT(category, '|')) AS cat FROM t;
     SELECT UNNEST(STRING_SPLIT(category, '|')) AS cat, COUNT(*) FROM t GROUP BY cat;

2. To FILTER rows that contain a segment (WHERE clause):
   Use list_contains(STRING_SPLIT(col, 'X'), 'value') — NO UNNEST.
   CORRECT:   WHERE list_contains(STRING_SPLIT(category, '|'), 'Electronics')
   WRONG:     WHERE list_contains(UNNEST(STRING_SPLIT(category, '|')), 'Electronics')
   WRONG:     WHERE category = 'Electronics'
   WRONG:     WHERE category LIKE '%Electronics%'

NEVER use UNNEST inside a WHERE clause. DuckDB does not support it there.

AGGREGATION OVER DELIMITED COLUMNS — THIS IS CRITICAL:
When you need BOTH UNNEST (to expand segments) AND an aggregate like AVG/SUM/COUNT,
you MUST use a two-level subquery — NEVER put UNNEST and an aggregate in the same
SELECT level, as DuckDB does not support this.

WRONG (flat — DuckDB rejects this):
  SELECT UNNEST(STRING_SPLIT(category, '|')) AS cat, AVG(price) AS avg_price
  FROM t GROUP BY cat;

CORRECT (subquery — always use this pattern):
  SELECT cat, AVG(price) AS avg_price
  FROM (
    SELECT UNNEST(STRING_SPLIT(category, '|')) AS cat, price
    FROM t
  ) sub
  WHERE cat IS NOT NULL AND TRIM(cat) <> ''
  GROUP BY cat
  ORDER BY avg_price DESC;

DUCKDB-SPECIFIC RULES:
- When using SUM/AVG/COUNT/MIN/MAX, always include the matching GROUP BY clause.
- If you use math operators (<, >, =) on a text/VARCHAR column, you MUST ALWAYS wrap the column in TRY_CAST(... AS DOUBLE) first. Never compare a string directly to an integer.
- If you use aggregation functions like SUM() or AVG(), you MUST wrap the column in TRY_CAST(... AS DOUBLE). Never attempt to SUM() or AVG() a raw VARCHAR.
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

REPAIR INSTRUCTIONS:
- If the error says "UNNEST not supported here" in a WHERE clause:
  UNNEST cannot be used inside WHERE. Replace it with list_contains:
  WRONG: WHERE list_contains(UNNEST(STRING_SPLIT(col, '|')), 'value')
  RIGHT: WHERE list_contains(STRING_SPLIT(col, '|'), 'value')

- If the error mentions UNNEST with an aggregate function (AVG, SUM, COUNT, MIN, MAX):
  You cannot use UNNEST and aggregate functions at the same SELECT level.
  Use a subquery to expand first, then aggregate:
  WRONG: SELECT UNNEST(STRING_SPLIT(col, '|')) AS cat, AVG(price) AS avg FROM t GROUP BY cat;
  RIGHT: SELECT cat, AVG(price) AS avg FROM (
           SELECT UNNEST(STRING_SPLIT(col, '|')) AS cat, price FROM t
         ) sub WHERE cat IS NOT NULL GROUP BY cat ORDER BY avg DESC;

- If the error says "Referenced column not found" and provides "Candidate bindings":
  Replace your hallucinated column name with one of the exact Candidate bindings
  provided in the error message.

- For all other errors: fix the specific syntax or logic error described.

Write a corrected DuckDB SQL query that fixes this error exactly.
Output ONLY the raw SQL, nothing else.
"""

# ── LLM-based error humanization ─────────────────────────────────────────────
_ERROR_HUMANIZER_SYSTEM = """\
You are a helpful data assistant that explains technical errors to non-technical users.

You will be given:
1. A technical error message from a data analytics system
2. The user's original question
3. Context about what the system was trying to do when the error occurred

Your job is to write a single clear, friendly explanation (maximum 2 sentences) that:
- Explains what went wrong in plain English that anyone can understand
- Tells the user one specific thing they can try to fix it
- NEVER mentions SQL, DuckDB, Ollama, Python, regex, HTTP, or any technical system names
- NEVER says "the AI" — say "the system" instead
- NEVER repeats or paraphrases the raw error message
- NEVER uses bullet points, lists, or markdown formatting
- NEVER adds a preamble like "Sure!" or "Of course!" — go straight to the explanation

Output ONLY the plain-English explanation. Nothing else.
"""

_ERROR_HUMANIZER_USER = """\
User's original question: {prompt}

What the system was doing when the error occurred: {context}

Technical error message: {error}

Plain-English explanation for the user:
"""


def _humanize_error(
    technical_msg: str,
    prompt: str = "",
    context: str = "generating a query for your data",
) -> str:
    """
    Convert a raw technical error string into a plain-English user-facing
    message using llama3.2:3b (ROUTER_MODEL).

    Three-level cascade:
      1. Ollama connectivity errors → hardcoded message (no HTTP call).
      2. llama3.2:3b LLM call → returns plain-English sentence on success.
      3. Pattern-based classifier → recognises common DuckDB error shapes
         and returns a genuinely plain-English sentence. Raw technical text
         is NEVER shown regardless of whether the LLM is available.

    NOTE on the old approach:
      The previous fallback used `re.sub(r"^[\w.]+Error:\\s*", "", msg)` which
      only stripped a leading "SomethingError:" class-name prefix. When
      validator.py prepends "Generated SQL could not be parsed or validated
      by DuckDB: Binder Error: ..." the string starts with "Generated", the
      regex matches nothing, and the full raw DuckDB text leaked through.
      That is why the screenshot showed raw DuckDB error text in the UI.
      This version never falls back to raw technical text under any condition.
    """
    logger.debug("Raw error (before humanization): %s", technical_msg)

    # ── Guard: Ollama connectivity — never call Ollama to explain this ────────
    if re.search(
        r"Cannot connect|Connection refused|ConnectionError|Failed to establish",
        technical_msg, re.IGNORECASE,
    ):
        return (
            "The system couldn't reach the local AI model server. "
            "Please make sure Ollama is running (`ollama serve`) and try again."
        )

    # ── LLM-based humanization ────────────────────────────────────────────────
    try:
        user_payload = _ERROR_HUMANIZER_USER.format(
            prompt  = prompt  or "not specified",
            context = context,
            error   = technical_msg,
        )
        payload = {
            "model":    ROUTER_MODEL,
            "messages": [
                {"role": "system", "content": _ERROR_HUMANIZER_SYSTEM},
                {"role": "user",   "content": user_payload},
            ],
            "stream":  False,
            "options": {
                "temperature": 0.1,
                "num_predict": 80,
                "num_ctx":     512,
            },
        }
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=15,
        )
        if resp.status_code == 200:
            explanation = resp.json()["message"]["content"].strip()
            if explanation:
                logger.debug("LLM humanized error: %s", explanation)
                return explanation

        logger.debug(
            "Humanization LLM returned HTTP %d — falling back to classifier.",
            resp.status_code,
        )

    except Exception as humanize_exc:
        # Never let error humanization itself crash the app.
        logger.debug("Humanization LLM call failed: %s", humanize_exc)

    # ── Fallback: pattern-based plain-English classifier ──────────────────────
    # The LLM call failed or returned empty. NEVER show raw technical text.
    # Recognise common DuckDB error shapes and return a plain-English message
    # for each. The old regex approach (re.sub r"^[\w.]+Error:\s*") was
    # silently broken — it matched nothing when the string started with
    # "Generated SQL could not be parsed..." — so this replaces it entirely.

    # Pattern 1: Column does not exist in dataset
    col_match = re.search(
        r'[Rr]eferenced column\s+["\']?([\w\s]+?)["\']?\s+not found',
        technical_msg, re.IGNORECASE,
    )
    if col_match:
        bad_col = col_match.group(1).strip()
        return (
            f'Your dataset doesn\'t have a column called "{bad_col}". '
            f"Open the schema panel on the left to see the exact column "
            f"names available, then rephrase your question using one of those."
        )

    # Pattern 2: Ambiguous column — exists in more than one table
    if re.search(r"ambiguous (column|reference|binding)", technical_msg, re.IGNORECASE):
        return (
            "Your question refers to a column name that appears in more than "
            "one table in your dataset. Try being more specific by mentioning "
            "which table you mean alongside the column name."
        )

    # Pattern 3: Type mismatch — trying to do math on a text column
    if (
        re.search(
            r"(cannot|could not|unable to)\s+(cast|convert)",
            technical_msg, re.IGNORECASE,
        )
        or "no function matches" in technical_msg.lower()
        or "conversion error" in technical_msg.lower()
    ):
        return (
            "The system tried to do a calculation on a column that holds "
            "text instead of numbers. Check the schema panel to see which "
            "columns are numeric, then rephrase your question to use one of those."
        )

    # Pattern 4: General binder / parser error (covers most other DuckDB errors)
    if re.search(
        r"binder error|parser error|syntax error|catalog error",
        technical_msg, re.IGNORECASE,
    ):
        return (
            "The system had difficulty understanding how to search your data "
            "for that question. Try rephrasing it more simply, or check the "
            "schema panel to confirm the column names you're asking about."
        )

    # Pattern 5: Table or relation not found
    if re.search(
        r"(table|relation|catalog entry)\s+.+?\s+(not found|does not exist)",
        technical_msg, re.IGNORECASE,
    ):
        return (
            "The system couldn't find the data table it was looking for. "
            "This usually means the session has expired — try re-uploading "
            "your file and asking again."
        )

    # Pattern 6: Division by zero
    if re.search(r"division by zero|divide by zero", technical_msg, re.IGNORECASE):
        return (
            "The calculation in your question resulted in a division by zero, "
            "which usually means some rows in your data have a zero or empty "
            "value in the column being used as a divisor. Try filtering those "
            "rows out or asking a different question."
        )

    # Catch-all: completely unknown error shape — generic but never shows raw text
    return (
        "The system wasn't able to produce a result for that question. "
        "This usually happens when the question refers to a column or "
        "calculation that doesn't match your dataset. Check the schema "
        "panel on the left for the exact column names and try rephrasing."
    )


# ── Delimiter detection ────────────────────────────────────────────────────────
_DELIMITER_CANDIDATES   = ["|", ";", "/", ">"]
_DELIMITER_MIN_RATIO    = 0.5
_DELIMITER_SAMPLE_LIMIT = 200


def _detect_delimiter(conn, table_name: str, col: str) -> str | None:
    """
    Check whether a VARCHAR column's values are internally delimited
    (e.g. "Computers&Accessories|Cables|USBCables"). Without this hint,
    the model treats the whole joined string as one literal value and
    generates SQL that silently returns 0 rows.
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
    delimiters and flagged inline so the model knows to use
    UNNEST(STRING_SPLIT(...)) or list_contains(STRING_SPLIT(...)) correctly.
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
                            f" [DELIMITED with '{delim}' — use "
                            f"UNNEST(STRING_SPLIT(\"{col_name}\", '{delim}')) in SELECT/GROUP BY "
                            f"or list_contains(STRING_SPLIT(\"{col_name}\", '{delim}'), 'value') "
                            f"in WHERE — NEVER use = or IN on this column directly]"
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
    """Upgrade bare CAST(...) → TRY_CAST(...) so one bad value never crashes a query."""
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
    """
    Remove any REPLACE(<expr>, '.', '') the model generates.
    Stripping '.' from a decimal number silently corrupts values:
    "4.5" → "45", producing wildly wrong aggregates.
    Uses paren-matching instead of regex to avoid cross-expression corruption.
    """
    result = sql
    while True:
        changed = False
        for match in _REPLACE_CALL_START.finditer(result):
            open_idx  = match.end() - 1
            close_idx = _find_matching_paren(result, open_idx)
            if close_idx is None:
                continue

            inner = result[open_idx + 1: close_idx]
            args  = _split_top_level_args(inner)

            if len(args) != 3:
                continue

            search_arg  = args[1].strip()
            replace_arg = args[2].strip()

            if search_arg in ("'.'", '"."') and replace_arg in ("''", '""'):
                expr   = args[0].strip()
                result = result[:match.start()] + expr + result[close_idx + 1:]
                changed = True
                break

        if not changed:
            break

    return result


_UNNEST_IN_LIST_CONTAINS = re.compile(
    r"list_contains\(\s*UNNEST\(\s*STRING_SPLIT\(\s*([^,]+?)\s*,\s*('[^']*'|\"[^\"]*\")\s*\)\s*\)\s*,",
    re.IGNORECASE,
)


def _fix_unnest_in_where(sql: str) -> str:
    """
    Rewrite list_contains(UNNEST(STRING_SPLIT(col, 'X')), ...)
          → list_contains(STRING_SPLIT(col, 'X'), ...)

    DuckDB does not support UNNEST inside a WHERE clause or inside
    list_contains(). Enforced deterministically since self-repair
    consistently fails to break this habit across all 3 attempts.
    """
    before = sql
    sql = _UNNEST_IN_LIST_CONTAINS.sub(
        r"list_contains(STRING_SPLIT(\1, \2),",
        sql,
    )
    if sql != before:
        logger.info("Fixed UNNEST-in-list_contains pattern in WHERE clause.")
    return sql


_UNNEST_WITH_AGG_PATTERN = re.compile(
    r"SELECT\s+(?:DISTINCT\s+)?"
    r"UNNEST\s*\(\s*STRING_SPLIT\s*\(\s*"
    r"([^\s,]+)"
    r"\s*,\s*"
    r"('[^']*'|\"[^\"]*\")"
    r"\s*\)\s*\)"
    r"\s+AS\s+(\w+)"
    r"\s*,\s*"
    r"(AVG|SUM|COUNT|MIN|MAX)"
    r"\s*\(([^)]+)\)"
    r"\s+AS\s+(\w+)"
    r"\s+FROM\s+(\w+)"
    r"(?:\s+(?:WHERE|GROUP\s+BY|ORDER\s+BY|LIMIT)[^;]*)?",
    re.IGNORECASE,
)


def _fix_unnest_with_aggregation(sql: str) -> str:
    """
    Rewrite the model's common mistake of mixing UNNEST(STRING_SPLIT(...))
    and an aggregate function (AVG/SUM/COUNT/MIN/MAX) at the same SELECT level.

    DuckDB requires the UNNEST to be in an inner subquery that expands rows
    first, with the aggregation happening in the outer query over those
    expanded rows.
    """
    match = _UNNEST_WITH_AGG_PATTERN.search(sql)
    if not match:
        return sql

    col_expr  = match.group(1).strip()
    delim     = match.group(2).strip()
    cat_alias = match.group(3).strip()
    agg_func  = match.group(4).upper()
    agg_col   = match.group(5).strip()
    agg_alias = match.group(6).strip()
    table     = match.group(7).strip()

    rewritten = (
        f"SELECT {cat_alias}, {agg_func}({agg_col}) AS {agg_alias} "
        f"FROM ("
        f"SELECT UNNEST(STRING_SPLIT({col_expr}, {delim})) AS {cat_alias}, "
        f"{agg_col} "
        f"FROM {table}"
        f") sub "
        f"WHERE {cat_alias} IS NOT NULL AND TRIM({cat_alias}) <> '' "
        f"GROUP BY {cat_alias} "
        f"ORDER BY {agg_alias} DESC;"
    )

    logger.info(
        "Fixed UNNEST-with-aggregation: rewrote flat SELECT into two-level subquery."
    )
    return rewritten


def _harden_sql(sql: str) -> str:
    """
    Apply all deterministic post-generation safety rewrites in order.
    Order matters:
      1. _fix_unnest_with_aggregation — restructures the whole query first
      2. _fix_decimal_stripping       — fix decimal-stripping inside expressions
      3. _fix_unnest_in_where         — fix any WHERE-clause UNNEST misuse
      4. _harden_casts                — upgrade bare CAST → TRY_CAST
    """
    before = sql
    sql = _fix_unnest_with_aggregation(sql)
    sql = _fix_decimal_stripping(sql)
    sql = _fix_unnest_in_where(sql)
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
            "num_ctx":     8192,
        },
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )

    if resp.status_code != 200:
        raise Exception(
            f"Ollama returned HTTP {resp.status_code}. "
            f"Model requested: '{SQL_MODEL}'. "
            f"Ollama's exact error message: {resp.text}"
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
    """
    Generate a valid DuckDB SELECT query for *prompt*.

    Returns:
        {"success": True,  "sql": "<query>"}
        {"success": False, "error": "<plain-English user-facing message>"}

    All "error" values are plain-English strings from _humanize_error().
    Raw technical text is never returned in the error field.
    """
    conn       = db_session
    all_tables = [t.strip() for t in table_name.split(",") if t.strip()] if table_name else []

    print(
        f"\n{_BOLD}{'═'*55}{_R}\n"
        f"  {_BLUE}{_BOLD}ROUTE B — DATA ENGINE (SQL){_R}\n"
        f"  tables={_CYAN}{table_name}{_R}  prompt='{prompt[:60]}'\n"
        f"{_BOLD}{'═'*55}{_R}",
        flush=True,
    )

    # ── Build column dictionary ───────────────────────────────────────────────
    if conn and all_tables:
        column_dict = _build_column_dict(conn, all_tables)
    else:
        column_dict = f"(DDL schema — use column names from this only):\n{ddl_schema}"

    user_payload = _USER_TEMPLATE.format(column_dict=column_dict, prompt=prompt)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_payload},
    ]

    print(
        f"\n{_BOLD}{_BLUE}┌── Route B · Stage 1: SQL Generation{_R}\n"
        f"   model={_DIM}{SQL_MODEL}{_R}",
        flush=True,
    )

    # ── First attempt ─────────────────────────────────────────────────────────
    try:
        raw = _call_ollama(messages)
    except requests.exceptions.ConnectionError as e:
        technical = f"Cannot connect to Ollama at {OLLAMA_BASE_URL}: {e}"
        logger.error(technical)
        _print_fail(technical)
        return {"success": False, "error": _humanize_error(
            technical, prompt=prompt,
            context="connecting to the AI model server",
        )}
    except requests.exceptions.Timeout as e:
        technical = f"Ollama took longer than {OLLAMA_TIMEOUT}s to respond: {e}"
        logger.error(technical)
        _print_fail(technical)
        return {"success": False, "error": _humanize_error(
            technical, prompt=prompt,
            context="waiting for the AI model to respond",
        )}
    except Exception as e:
        technical = f"Ollama call failed: {e}"
        logger.error(technical)
        _print_fail(technical)
        return {"success": False, "error": _humanize_error(
            technical, prompt=prompt,
            context="calling the AI model to generate a query",
        )}

    sql = _extract_sql(raw)
    if not sql:
        technical = "Model response contained no valid SQL."
        logger.warning("%s Raw response (first 200 chars): %s", technical, raw[:200])
        _print_fail(technical)
        return {"success": False, "error": _humanize_error(
            technical, prompt=prompt,
            context="asking the AI model to generate a SQL query for your question",
        )}

    sql = _harden_sql(sql)
    logger.info("SQL first attempt (%d chars): %s…", len(sql), sql[:100])
    print(f"   sql={_DIM}{sql[:120]}…{_R}\n{_BOLD}{_BLUE}└──{_R}", flush=True)

    # ── Self-repair loop (up to 3 attempts via EXPLAIN) ───────────────────────
    if conn:
        print(
            f"\n{_BOLD}{_BLUE}┌── Route B · Stage 2: Validation & Repair{_R}",
            flush=True,
        )

        max_repairs      = 3
        repair_messages  = list(messages)
        last_explain_err = None

        for attempt in range(1, max_repairs + 1):
            try:
                conn.execute(f"EXPLAIN {sql}")
                if attempt == 1:
                    logger.info("SQL passed EXPLAIN on first attempt.")
                    print(
                        f"   {_GREEN}EXPLAIN PASS{_R}: SQL is valid\n"
                        f"{_BOLD}{_BLUE}└──{_R}",
                        flush=True,
                    )
                else:
                    logger.info("Self-repair succeeded (attempt %d): %s…", attempt, sql[:100])
                    print(
                        f"   {_GREEN}Self-repair succeeded (Attempt {attempt}/{max_repairs}){_R}\n"
                        f"{_BOLD}{_BLUE}└──{_R}",
                        flush=True,
                    )
                last_explain_err = None
                break

            except Exception as explain_err:
                last_explain_err = str(explain_err)

                if attempt == max_repairs:
                    logger.warning(
                        "Self-repair exhausted after %d attempts (%s) — "
                        "passing to validator.", max_repairs, explain_err,
                    )
                    print(
                        f"   {_RED}Self-repair failed after {max_repairs} attempts — "
                        f"passing original to validator{_R}\n{_BOLD}{_BLUE}└──{_R}",
                        flush=True,
                    )
                    break

                if attempt == 1:
                    logger.warning("SQL failed EXPLAIN (%s) — attempting self-repair…", explain_err)
                    print(
                        f"   {_YELLOW}EXPLAIN FAIL ({explain_err}) — "
                        f"attempting self-repair (Attempt 1/{max_repairs})…{_R}",
                        flush=True,
                    )
                else:
                    logger.warning(
                        "Self-repair failed EXPLAIN (%s) — attempt %d/%d…",
                        explain_err, attempt, max_repairs,
                    )
                    print(
                        f"   {_YELLOW}Self-repair failed ({explain_err}) — "
                        f"attempting self-repair (Attempt {attempt}/{max_repairs})…{_R}",
                        flush=True,
                    )

                repair_payload = _REPAIR_TEMPLATE.format(error=str(explain_err), sql=sql)
                repair_messages.append({"role": "assistant", "content": sql})
                repair_messages.append({"role": "user",      "content": repair_payload})

                try:
                    raw_repair = _call_ollama(repair_messages, max_tokens=1024)
                    repaired   = _extract_sql(raw_repair)
                    if repaired:
                        sql = _harden_sql(repaired)
                    else:
                        logger.warning("Self-repair produced no valid SQL.")
                        print(
                            f"   {_RED}Self-repair produced no valid SQL{_R}\n"
                            f"{_BOLD}{_BLUE}└──{_R}",
                            flush=True,
                        )
                        break
                except Exception as repair_exc:
                    logger.warning("Self-repair Ollama call failed: %s", repair_exc)
                    print(
                        f"   {_RED}Self-repair Ollama call failed{_R}\n"
                        f"{_BOLD}{_BLUE}└──{_R}",
                        flush=True,
                    )
                    break

    print(
        f"\n{_BOLD}{_GREEN}{'═'*55}{_R}\n"
        f"  {_GREEN}{_BOLD}ROUTE B COMPLETE ✓{_R}\n"
        f"{_BOLD}{_GREEN}{'═'*55}{_R}\n",
        flush=True,
    )

    logger.info("generate_safe_sql returning: %s…", sql[:120])
    return {"success": True, "sql": sql}