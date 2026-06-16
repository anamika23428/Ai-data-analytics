import re
import duckdb
from collections import Counter


def clean_and_profile(conn: duckdb.DuckDBPyConnection, table_name: str) -> dict:

    # ── Step 1: Row count before dedup ────────────────────────
    original_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

    # ── Step 2: Deduplicate inside DuckDB ─────────────────────
    # SELECT DISTINCT * fails on LIST/STRUCT columns, so we try it.
    # If DuckDB cannot deduplicate safely, we keep the original rows.
    duplicates_removed = 0
    try:
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT DISTINCT * FROM {table_name}")
        clean_rows         = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        duplicates_removed = original_rows - clean_rows
    except Exception:
        clean_rows = original_rows
        duplicates_removed = 0

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

    # ── Step 6: Keep nulls and report missingness ─────────────
    total_nulls = sum(null_counts.values())

    return {
        "original_rows":      original_rows,
        "clean_rows":         clean_rows,
        "duplicates_removed": duplicates_removed,
        "columns":            new_columns,
        "null_counts":        null_counts,
        "dtypes":             dict(zip(new_columns, col_types)),
        "coerced_to_numeric": [],
        "missing_cells":      total_nulls,
        "nulls_preserved":    True,
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


def _sanitise_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    if name and name[0].isdigit():
        name = "col_" + name
    return name or "unnamed"