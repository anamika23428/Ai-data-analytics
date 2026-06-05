# ─────────────────────────────────────────────
#  core/transformer.py  –  Clean & profile a table
#
#  Fixes:
#    1. Duplicate column names after sanitise → deduplicate with suffix
#    2. SELECT DISTINCT * fails on LIST/STRUCT → fallback to ROW_NUMBER dedup
# ─────────────────────────────────────────────

import re
import duckdb
from collections import Counter


def clean_and_profile(conn: duckdb.DuckDBPyConnection, table_name: str) -> dict:

    # ── Step 1: Row count before dedup ────────────────────────
    original_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    # ── Step 2: Deduplicate inside DuckDB ─────────────────────
    # SELECT DISTINCT * fails on LIST/STRUCT columns, so we try it
    # and fall back to ROW_NUMBER() if it errors.
    duplicates_removed = 0
    try:
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT DISTINCT * FROM {table_name}")
        clean_rows         = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        duplicates_removed = original_rows - clean_rows
    except Exception:
        # Fallback: assign a row number and keep only the first occurrence
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * EXCLUDE (__row_num) FROM (
                SELECT *, ROW_NUMBER() OVER () AS __row_num FROM {table_name}
            ) WHERE __row_num = (
                SELECT MIN(t2.__row_num)
                FROM (SELECT *, ROW_NUMBER() OVER () AS __row_num FROM {table_name}) t2
            )
        """)
        clean_rows         = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        duplicates_removed = original_rows - clean_rows

    # ── Step 3: Get column names + types ──────────────────────
    col_info  = conn.execute(f"DESCRIBE {table_name}").fetchall()
    columns   = [row[0] for row in col_info]
    col_types = [row[1] for row in col_info]

    # ── Step 4: Sanitise column names (deduplicate clashes) ───
    new_columns = _sanitise_all_names(columns)
    for old, new in zip(columns, new_columns):
        if old != new:
            try:
                conn.execute(f'ALTER TABLE {table_name} RENAME "{old}" TO "{new}"')
            except Exception:
                pass  # already renamed or identical

    # ── Step 5: Null counts ────────────────────────────────────
    null_count_sql = ", ".join(
        f'COUNT(*) FILTER (WHERE "{c}" IS NULL) AS "{c}"'
        for c in new_columns
    )
    null_row    = conn.execute(f"SELECT {null_count_sql} FROM {table_name}").fetchone()
    null_counts = dict(zip(new_columns, null_row))

    # ── Step 6: Fill nulls with type-safe defaults ─────────────
    fill_parts = []
    for col, dtype in zip(new_columns, col_types):
        default = _null_default(dtype)
        fill_parts.append(f'COALESCE("{col}", {default}) AS "{col}"')

    conn.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT {", ".join(fill_parts)} FROM {table_name}
    """)

    return {
        "original_rows":      original_rows,
        "clean_rows":         clean_rows,
        "duplicates_removed": duplicates_removed,
        "columns":            new_columns,
        "null_counts":        null_counts,
        "dtypes":             dict(zip(new_columns, col_types)),
        "coerced_to_numeric": [],
    }


# ── Helpers ───────────────────────────────────

def _sanitise_all_names(columns: list[str]) -> list[str]:
    """
    Sanitise all column names and resolve duplicates by appending _1, _2 etc.

    Example:
      ["First Name", "First-Name", "Age"]
      → ["first_name", "first_name_1", "age"]
    """
    sanitised = [_sanitise_name(c) for c in columns]

    # Count how many times each name appears
    counts = Counter(sanitised)
    # Track how many times we've seen each name so far
    seen   = Counter()
    result = []

    for name in sanitised:
        if counts[name] > 1:
            # This name clashes — append a counter suffix
            seen[name] += 1
            if seen[name] == 1:
                result.append(name)           # first occurrence keeps original
            else:
                result.append(f"{name}_{seen[name] - 1}")
        else:
            result.append(name)               # unique — no suffix needed

    return result


def _null_default(dtype: str) -> str:
    """
    Return a SQL literal that matches the column type for COALESCE.
    DuckDB rejects mixing types (e.g. DATE and 0).
    """
    d = dtype.upper()
    if "VARCHAR" in d or "TEXT" in d or "CHAR" in d or "STRING" in d:
        return "'unknown'"
    if "TIMESTAMP" in d:
        return "TIMESTAMP '1970-01-01 00:00:00'"
    if "DATE" in d:
        return "DATE '1970-01-01'"
    if "TIME" in d:
        return "TIME '00:00:00'"
    if "BOOLEAN" in d or "BOOL" in d:
        return "false"
    if "FLOAT" in d or "DOUBLE" in d or "REAL" in d or "DECIMAL" in d or "NUMERIC" in d:
        return "0.0"
    return "0"


def _sanitise_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    if name and name[0].isdigit():
        name = "col_" + name
    return name or "unnamed"