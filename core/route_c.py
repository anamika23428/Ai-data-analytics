# ─────────────────────────────────────────────────────────────────────────────
# core/route_c.py  –  Route C: Metadata / Schema + Keyword Answers
#
# Exclusively handles:
#   1. Schema/structural questions  — row counts, column names, data types, DDL
#   2. Keyword-matching queries     — unique values, top/frequent items,
#                                     value existence checks, distinct counts
#
# No visualization, no aggregation math, no LLM SQL generation.
# All answers come straight from DuckDB metadata or simple GROUP BY queries.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from core.ddl_utils import generate_privacy_safe_ddl


@dataclass
class RouteCResult:
    success: bool
    answer: str = ""
    # Optional dataframe for keyword-match results (e.g. unique value lists)
    dataframe: pd.DataFrame | None = None
    # For the app.py metadata renderer — list of {name, ddl, info_df}
    tables: list[dict] | None = None
    details: dict = field(default_factory=dict)
    error: str | None = None


# ─── helpers ─────────────────────────────────────────────────────────────────

def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _normalize(s: str) -> str:
    return re.sub(r"[\s_]+", "_", s.strip().lower())


def _describe_table(conn, table_name: str) -> dict:
    info_df  = conn.execute(f"DESCRIBE {_qident(table_name)}").df()
    row_count = int(conn.execute(f"SELECT COUNT(*) FROM {_qident(table_name)}").fetchone()[0])
    columns  = list(info_df.iloc[:, 0].astype(str))
    dtypes   = list(info_df.iloc[:, 1].astype(str))
    return {
        "table":        table_name,
        "row_count":    row_count,
        "column_count": len(columns),
        "columns":      columns,
        "dtypes":       dict(zip(columns, dtypes)),
        "info_df":      info_df,
    }


def _extract_column_hint(prompt: str) -> str | None:
    """
    Pull a specific column name from phrases like:
      - column price / field revenue
      - "unit price" column / `order_id` field
      - does X exist / is there a X column
    """
    # quoted multi-word before keyword:  "unit price" column
    m = re.search(r"[`\"']([a-zA-Z0-9_ ]+?)[`\"']\s+(?:column|field)\b", prompt, re.I)
    if m: return m.group(1).strip().lower()

    # keyword before quoted multi-word:  column "unit price"
    m = re.search(r"\b(?:column|field)\s+[`\"']([a-zA-Z0-9_ ]+?)[`\"']", prompt, re.I)
    if m: return m.group(1).strip().lower()

    # keyword before single word:  column price
    m = re.search(r"\b(?:column|field)\s+([a-zA-Z0-9_]+)\b", prompt, re.I)
    if m: return m.group(1).strip().lower()

    # single word before keyword:  price column
    m = re.search(r"\b([a-zA-Z0-9_]+)\s+(?:column|field)\b", prompt, re.I)
    if m: return m.group(1).strip().lower()

    # bare backtick / quote:  `price`
    m = re.search(r"[`\"']([a-zA-Z0-9_ ]+)[`\"']", prompt)
    if m: return m.group(1).strip().lower()

    return None


def _best_column_match(hint: str, summaries: list[dict]) -> list[tuple[str, str, str]]:
    """Return [(table, col_name, dtype)] for every column whose normalised name == hint."""
    norm = _normalize(hint)
    matches = []
    for s in summaries:
        for col in s["columns"]:
            if _normalize(col) == norm:
                matches.append((s["table"], col, s["dtypes"].get(col, "unknown")))
    return matches


def _guess_target_column(prompt: str, summaries: list[dict]) -> tuple[str | None, str | None]:
    """
    Best-effort: find the table + column the user is asking about.
    Used for keyword-matching queries like 'list unique categories'.
    Returns (table_name, column_name) or (None, None).
    """
    words = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", prompt.lower()))

    # 1. Try exact column name mentioned in prompt
    for s in summaries:
        for col in s["columns"]:
            if col.lower() in words:
                return s["table"], col

    # 2. Try partial match (prompt word is a substring of a column name or vice-versa)
    for s in summaries:
        for col in s["columns"]:
            col_l = col.lower()
            if any(w in col_l or col_l in w for w in words if len(w) > 3):
                return s["table"], col

    # 3. Fall back to the first categorical column of the first table
    for s in summaries:
        non_numeric = [
            col for col, dtype in s["dtypes"].items()
            if not any(t in dtype.lower() for t in ["int", "float", "double", "decimal", "numeric", "real"])
        ]
        if non_numeric:
            return s["table"], non_numeric[0]

    # 4. Absolute fallback — first column of first table
    if summaries:
        return summaries[0]["table"], summaries[0]["columns"][0]

    return None, None


# ─── sub-handlers ─────────────────────────────────────────────────────────────

def _handle_schema(summaries: list[dict], conn) -> RouteCResult:
    """Build the tables payload that app.py's metadata renderer expects."""
    tables_payload = []
    for s in summaries:
        ddl = generate_privacy_safe_ddl(conn, s["table"], redact=False)
        tables_payload.append({"name": s["table"], "ddl": ddl, "info_df": s["info_df"]})

    lines = ["Schema summary:"]
    for s in summaries:
        col_list = ", ".join(f"`{c}`" for c in s["columns"])
        lines.append(f"- **{s['table']}**: {s['column_count']} columns, {s['row_count']:,} rows")
        lines.append(f"  Columns: {col_list}")

    return RouteCResult(
        success=True,
        answer="\n".join(lines),
        tables=tables_payload,
    )


def _handle_row_count(summaries: list[dict]) -> RouteCResult:
    lines = ["Row count:"]
    total = 0
    for s in summaries:
        total += s["row_count"]
        lines.append(f"- **{s['table']}**: {s['row_count']:,} rows")
    if len(summaries) > 1:
        lines.append(f"- **Total**: {total:,} rows")
    return RouteCResult(success=True, answer="\n".join(lines))


def _handle_column_existence(hint: str, summaries: list[dict]) -> RouteCResult:
    matches = _best_column_match(hint, summaries)
    if matches:
        lines = [f"Yes — **{hint}** was found in:"]
        lines += [f"- **{t}** as `{col}` ({dtype})" for t, col, dtype in matches]
        return RouteCResult(success=True, answer="\n".join(lines))
    return RouteCResult(
        success=True,
        answer=f"No column named **{hint}** was found in the loaded tables.",
    )


def _handle_unique_values(conn, table: str, col: str, prompt_lower: str) -> RouteCResult:
    """
    Return the distinct values for `col`.
    If the column looks like a pipe-delimited multi-value field (e.g. categories),
    we split on '|' so each sub-value is counted individually.
    """
    # Check whether values contain '|'
    sample = conn.execute(
        f"SELECT {_qident(col)} FROM {_qident(table)} WHERE {_qident(col)} IS NOT NULL LIMIT 200"
    ).df()
    pipe_delimited = sample.iloc[:, 0].astype(str).str.contains(r"\|", na=False).any()

    if pipe_delimited:
        sql = (
            f"SELECT DISTINCT TRIM(val) AS {_qident(col)} "
            f"FROM (SELECT UNNEST(STRING_SPLIT({_qident(col)}, '|')) AS val "
            f"FROM {_qident(table)} WHERE {_qident(col)} IS NOT NULL) sub "
            f"WHERE TRIM(val) <> '' "
            f"ORDER BY {_qident(col)}"
        )
    else:
        sql = (
            f"SELECT DISTINCT {_qident(col)} FROM {_qident(table)} "
            f"WHERE {_qident(col)} IS NOT NULL ORDER BY {_qident(col)}"
        )

    df = conn.execute(sql).df()
    count = len(df)
    answer = f"Found **{count}** unique value(s) in `{col}` from **{table}**."
    return RouteCResult(success=True, answer=answer, dataframe=df)


def _handle_top_frequent(conn, table: str, col: str, limit: int, prompt_lower: str) -> RouteCResult:
    """Most/least frequent values for `col`."""
    descending = not any(w in prompt_lower for w in ["least", "bottom", "lowest", "rare", "rarest"])
    order      = "DESC" if descending else "ASC"
    direction  = "most common" if descending else "least common"

    pipe_delimited = (
        conn.execute(
            f"SELECT {_qident(col)} FROM {_qident(table)} WHERE {_qident(col)} IS NOT NULL LIMIT 200"
        ).df().iloc[:, 0].astype(str).str.contains(r"\|", na=False).any()
    )

    if pipe_delimited:
        sql = (
            f"SELECT TRIM(val) AS {_qident(col)}, COUNT(*) AS frequency "
            f"FROM (SELECT UNNEST(STRING_SPLIT({_qident(col)}, '|')) AS val "
            f"FROM {_qident(table)} WHERE {_qident(col)} IS NOT NULL) sub "
            f"WHERE TRIM(val) <> '' "
            f"GROUP BY TRIM(val) ORDER BY frequency {order} LIMIT {limit}"
        )
    else:
        sql = (
            f"SELECT {_qident(col)}, COUNT(*) AS frequency "
            f"FROM {_qident(table)} WHERE {_qident(col)} IS NOT NULL "
            f"GROUP BY {_qident(col)} ORDER BY frequency {order} LIMIT {limit}"
        )

    df = conn.execute(sql).df()
    answer = f"Top {limit} {direction} values in `{col}` from **{table}**."
    return RouteCResult(success=True, answer=answer, dataframe=df)


def _handle_value_existence(conn, table: str, col: str, prompt: str) -> RouteCResult:
    """Does a specific value exist in a column?"""
    # Extract quoted value first, then fallback to last 1-2 words
    m = re.search(r"[\"']([^\"']+)[\"']", prompt)
    if m:
        value = m.group(1)
    else:
        # heuristic: last meaningful word after "exist"/"contains"/"have"
        words = re.findall(r"[a-zA-Z0-9_]+", prompt)
        value = words[-1] if words else ""

    count = conn.execute(
        f"SELECT COUNT(*) FROM {_qident(table)} "
        f"WHERE LOWER(CAST({_qident(col)} AS VARCHAR)) = LOWER('{value.replace(chr(39), chr(39)*2)}')"
    ).fetchone()[0]

    if count:
        return RouteCResult(
            success=True,
            answer=f"Yes — **{value}** appears **{count:,}** time(s) in `{col}` of **{table}**.",
        )
    return RouteCResult(
        success=True,
        answer=f"No — **{value}** was not found in `{col}` of **{table}**.",
    )


def _handle_distinct_count(conn, table: str, col: str) -> RouteCResult:
    count = conn.execute(
        f"SELECT COUNT(DISTINCT {_qident(col)}) FROM {_qident(table)} WHERE {_qident(col)} IS NOT NULL"
    ).fetchone()[0]
    return RouteCResult(
        success=True,
        answer=f"There are **{count:,}** distinct value(s) in `{col}` of **{table}**.",
    )


# ─── public entry point ───────────────────────────────────────────────────────

def run(conn, tables: list[str], prompt: str) -> RouteCResult:
    """
    Answer metadata and keyword-matching questions directly from the loaded tables.

    Decision tree (first match wins):
      1. Schema/structure/column-list/data-type queries  → DDL + describe
      2. Row-count queries                               → COUNT(*)
      3. Explicit column existence check                 → column scan
      4. Unique / distinct value listing                → DISTINCT query
      5. Distinct count                                  → COUNT(DISTINCT …)
      6. Most/least frequent value queries               → GROUP BY … ORDER BY
      7. Value existence ("does X exist in column Y")    → filtered COUNT
      8. Default: loaded-table overview                  → row/col counts
    """
    if not tables:
        return RouteCResult(success=False, error="No tables are loaded.")

    pl = prompt.lower()
    summaries = [_describe_table(conn, t) for t in tables]

    # ── 1. Schema / structure ────────────────────────────────────────────────
    if any(phrase in pl for phrase in [
        "schema", "structure", "data type", "datatype", "what columns",
        "list columns", "list fields", "column names", "field names",
        "how many columns", "number of columns", "column count",
        "describe table", "describe dataset", "describe data",
        "show columns", "show fields", "show schema", "show structure",
    ]):
        return _handle_schema(summaries, conn)

    # ── 2. Row count ─────────────────────────────────────────────────────────
    if any(phrase in pl for phrase in [
        "how many rows", "row count", "number of rows", "how many records",
        "total rows", "total records",
    ]):
        return _handle_row_count(summaries)

    # ── 3. Column existence check ─────────────────────────────────────────────
    col_hint = _extract_column_hint(prompt)
    if col_hint and any(phrase in pl for phrase in [
        "does", "exist", "is there", "do you have", "contain", "have a",
    ]):
        return _handle_column_existence(col_hint, summaries)

    # ── 4–7. Keyword-matching queries ─────────────────────────────────────────
    table, col = _guess_target_column(prompt, summaries)
    if table and col:

        # 4. Unique / distinct values listing
        if any(phrase in pl for phrase in [
            "unique", "distinct values", "list all", "all values",
            "what are the", "show all", "all unique", "list unique",
            "what values", "possible values", "available values",
        ]):
            # Extract an explicit limit if given (e.g. "list top 20 categories")
            lm = re.search(r"\btop\s+(\d+)\b", pl)
            if lm:
                lim = min(int(lm.group(1)), 500)
                return _handle_top_frequent(conn, table, col, lim, pl)
            return _handle_unique_values(conn, table, col, pl)

        # 5. Distinct count
        if any(phrase in pl for phrase in [
            "how many unique", "how many distinct", "count unique",
            "count distinct", "number of unique", "number of distinct",
        ]):
            return _handle_distinct_count(conn, table, col)

        # 6. Most / least frequent
        lm = re.search(r"\btop\s+(\d+)\b", pl)
        limit = min(int(lm.group(1)), 100) if lm else 10
        if any(phrase in pl for phrase in [
            "most common", "most frequent", "top", "highest frequency",
            "least common", "least frequent", "bottom", "rarest", "rarely",
            "popular", "popularity",
        ]):
            return _handle_top_frequent(conn, table, col, limit, pl)

        # 7. Value existence
        if any(phrase in pl for phrase in [
            "does", "exist", "is there", "contains", "have", "find",
        ]):
            return _handle_value_existence(conn, table, col, prompt)

    # ── 8. Default: table overview ────────────────────────────────────────────
    lines = ["Loaded table overview:"]
    for s in summaries:
        lines.append(f"- **{s['table']}**: {s['row_count']:,} rows, {s['column_count']} columns")
    return RouteCResult(success=True, answer="\n".join(lines))
