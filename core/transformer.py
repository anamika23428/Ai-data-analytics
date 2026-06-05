# ─────────────────────────────────────────────
#  core/transformer.py  –  Clean & profile the data
#
#  Once the file is in DuckDB we pull it into pandas,
#  tidy it up, push it back, and produce a short
#  "quality report" the UI can show the user.
#
#  Steps:
#    1. Sanitise column names   (no spaces, lowercase)
#    2. Drop exact duplicate rows
#    3. Fill or flag nulls
#    4. Coerce obvious numeric columns
#    5. Return a quality report dict
# ─────────────────────────────────────────────

import re
import duckdb
import pandas as pd


def clean_and_profile(conn: duckdb.DuckDBPyConnection) -> dict:
    """
    Clean the "data" table inside the given DuckDB connection.

    Returns a quality_report dict with before/after stats.
    """

    # Pull everything into pandas so we can use its cleaning tools
    df = conn.execute("SELECT * FROM data").df()

    original_rows    = len(df)
    original_columns = list(df.columns)

    # ── Step 1: Sanitise column names ────────────────────
    # "First Name " → "first_name"
    df.columns = [_sanitise_name(col) for col in df.columns]

    # ── Step 2: Drop exact duplicate rows ────────────────
    df = df.drop_duplicates()
    duplicates_removed = original_rows - len(df)

    # ── Step 3: Track nulls before we fill them ──────────
    null_counts_before = df.isnull().sum().to_dict()

    # Fill nulls: numbers → 0,  text → "unknown"
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(0)

    # ── Step 4: Try to coerce text columns that look numeric ──
    coerced_columns = []
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only switch to numeric if at least 80 % of values converted
        success_rate = converted.notna().mean()
        if success_rate >= 0.8:
            df[col] = converted.fillna(0)
            coerced_columns.append(col)

    # ── Step 5: Write clean data back into DuckDB ─────────
    # Drop the old table, register the clean DataFrame
    conn.execute("DROP TABLE IF EXISTS data")
    conn.register("data", df)
    conn.execute("CREATE TABLE data AS SELECT * FROM data")

    # ── Build the quality report ──────────────────────────
    quality_report = {
        "original_rows":      original_rows,
        "clean_rows":         len(df),
        "duplicates_removed": duplicates_removed,
        "columns":            list(df.columns),
        "original_columns":   original_columns,
        "null_counts":        null_counts_before,
        "coerced_to_numeric": coerced_columns,
        "dtypes":             df.dtypes.astype(str).to_dict(),
    }

    return quality_report


# ── Private helpers ───────────────────────────────────────

def _sanitise_name(name: str) -> str:
    """
    Turn any column name into a clean, SQL-safe identifier.
    Examples:
      "First Name "  →  "first_name"
      "Revenue ($)"  →  "revenue"
      "123abc"       →  "col_123abc"
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)   # non-alphanumeric → underscore
    name = name.strip("_")                     # remove leading/trailing underscores

    if name and name[0].isdigit():             # SQL names can't start with a digit
        name = "col_" + name

    return name or "unnamed"
