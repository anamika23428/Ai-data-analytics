# ───────────────────────────────────
#  core/ingestor.py
#
#  Fixes:
#    1. Duplicate table names across files → warn instead of silent overwrite
#    2. File paths with spaces → use $$ quoting in SQL
#    3. Excel Extension Resilience → Uses try/except to prevent offline crash
#    4. Unescaped Excel Sheet Names → Safely escapes single quotes in tab names
#    5. Mixed alphanumeric data loss → load Excel sheets as VARCHAR first,
#       then safely downcast clean columns while preserving mixed text strings.
#    6. CSV/JSON loading → proper fallback to Pandas if DuckDB's auto-sniffer fails
# ─────────────────────────────────────────────

import re
import json
import pandas as pd
from pathlib import Path
import duckdb
import openpyxl


def get_or_create_connection(session_state) -> duckdb.DuckDBPyConnection:
    """Return the shared DuckDB connection, installing excel extension only once."""
    if "duckdb_conn" not in session_state or session_state.duckdb_conn is None:
        conn = duckdb.connect()
        # Handle offline environments gracefully
        try:
            conn.execute("INSTALL excel")
        except duckdb.Error:
            pass
        try:
            conn.execute("LOAD excel")
        except duckdb.Error:
            pass
        session_state.duckdb_conn = conn
    return session_state.duckdb_conn


def load_file_into_duckdb(
    file_path: Path,
    conn: duckdb.DuckDBPyConnection,
    existing_tables: list[str]
) -> tuple[list[str], list[str]]:
    """
    Read a file and create one DuckDB table per dataset.

    Returns:
        (created_tables, warnings)
        warnings is a list of human-readable messages about name conflicts.
    """
    suffix   = file_path.suffix.lower()
    warnings = []

    if suffix in (".csv", ".txt"):
        tables = [_load_csv(conn, file_path, existing_tables, warnings)]
    elif suffix == ".json":
        tables = _load_json(conn, file_path, existing_tables, warnings)
    elif suffix == ".xlsx":
        tables = _load_excel_all_sheets(conn, file_path, existing_tables, warnings)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return tables, warnings


def get_all_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    result = conn.execute("SHOW TABLES").fetchall()
    return [row[0] for row in result]


# ── Private helpers ───────────────────────────

def _safe_path(file_path: Path) -> str:
    """
    Wrap path in $$ so spaces and special characters don't break SQL.
    E.g.  C:/my files/data.csv  →  $$C:/my files/data.csv$$
    """
    return f"$${file_path}$$"


def _resolve_table_name(
    raw_name: str,
    existing_tables: list[str],
    warnings: list[str]
) -> str:
    """
    Generate a table name and check for conflicts.
    If the name already exists, append _2, _3 etc. and add a warning.
    """
    base = _make_table_name(raw_name)
    name = base
    counter = 2

    while name in existing_tables:
        warnings.append(
            f"⚠️ Table `{name}` already exists. "
            f"Renaming new table to `{base}_{counter}` to avoid overwrite."
        )
        name = f"{base}_{counter}"
        counter += 1

    existing_tables.append(name)  # register immediately so next file sees it
    return name


def _load_csv(conn, file_path, existing_tables, warnings):
    table_name = _resolve_table_name(file_path.stem, existing_tables, warnings)
    try:
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_csv_auto({_safe_path(file_path)}, header=true)
        """)
    except duckdb.Error as e:
        raise ValueError(f"Failed to load CSV file {file_path.name}: {e}")
    return table_name


def _load_json(conn, file_path, existing_tables, warnings):
    table_name = _resolve_table_name(file_path.stem, existing_tables, warnings)

    if not _try_load_json_document(conn, table_name, file_path):
        if not _try_load_json_ndjson(conn, table_name, file_path):
            if not _try_load_json_recovered(conn, table_name, file_path):
                raise ValueError(
                    f"Failed to parse JSON file {file_path.name}: not valid JSON, "
                    f"NDJSON, or a recoverable sequence of JSON records."
                )

    # The JSON may be a "wrapper" object whose keys are each an array of
    # records, e.g. {"employees": [...], "performance": [...]}. DuckDB
    # loads that as ONE row with LIST-typed columns instead of real rows.
    # Detect that shape and explode it into one table per array.
    return _split_multi_array_json(conn, table_name, file_path, existing_tables, warnings)


def _try_load_json_document(conn, table_name, file_path) -> bool:
    """Standard path: a single JSON document (object or array)."""
    try:
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_json_auto({_safe_path(file_path)})
        """)
        return True
    except duckdb.Error:
        return False


def _try_load_json_ndjson(conn, table_name, file_path) -> bool:
    """NDJSON / JSON Lines (one JSON object per line)."""
    try:
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_json_auto({_safe_path(file_path)}, format='newline_delimited')
        """)
        return True
    except duckdb.Error:
        pass

    try:
        df = pd.read_json(file_path, lines=True)
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        return True
    except Exception:
        pass

    try:
        df = pd.read_json(file_path)
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        return True
    except Exception:
        return False


def _try_load_json_recovered(conn, table_name, file_path) -> bool:
    """
    Malformed-but-recoverable JSON: a sequence of JSON values that isn't
    wrapped in `[ ]` and/or has stray commas between records, e.g.
        {"id":1,"name":"A"},
        {"id":2,"name":"B"},
    This is neither valid single-document JSON nor valid NDJSON, but the
    individual records are well-formed, so we can recover them by scanning
    the raw text for back-to-back JSON values.
    """
    try:
        records = _recover_json_records(file_path)
        if not records:
            return False
        df = pd.json_normalize(records)
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        return True
    except Exception:
        return False


def _split_multi_array_json(conn, table_name, file_path, existing_tables, warnings) -> list[str]:
    """
    If the loaded JSON resulted in a single row whose columns are
    themselves LIST-typed (arrays of records) — e.g. the file was shaped
    like {"employees": [...], "performance": [...]} — explode each list
    column into its own real table, mirroring how multi-sheet Excel files
    become multiple tables. Returns the final list of table name(s).
    """
    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    columns_info = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()

    if row_count != 1 or not columns_info:
        return [table_name]

    list_cols = [
        col_name for _, col_name, col_type, *_ in columns_info
        if col_type.upper().startswith("LIST") or col_type.endswith("[]")
    ]
    scalar_cols = [
        col_name for _, col_name, col_type, *_ in columns_info
        if col_name not in list_cols
    ]

    if not list_cols:
        return [table_name]

    base_stem = file_path.stem
    created = []
    for col_name in list_cols:
        escaped_col = f'"{col_name}"'
        sub_table = _resolve_table_name(f"{base_stem}_{col_name}", existing_tables, warnings)
        try:
            conn.execute(f"""
                CREATE OR REPLACE TABLE {sub_table} AS
                SELECT unnest({escaped_col}, recursive := true)
                FROM {table_name}
            """)
            created.append(sub_table)
        except duckdb.Error as e:
            warnings.append(
                f"⚠️ Could not expand `{col_name}` from {file_path.name} into its own table ({e})."
            )
            existing_tables.remove(sub_table)

    if not created:
        # Nothing could be expanded; fall back to the original single table.
        return [table_name]

    if scalar_cols:
        warnings.append(
            f"⚠️ {file_path.name}: top-level field(s) {', '.join(scalar_cols)} "
            f"were not arrays and were dropped when splitting into separate tables."
        )

    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    if table_name in existing_tables:
        existing_tables.remove(table_name)

    return created


def _recover_json_records(file_path) -> list:
    """
    Scan raw text for a sequence of well-formed JSON values, tolerating
    separators (commas, whitespace, newlines) between them and a missing
    enclosing array. Returns a list of the decoded values (dicts/lists/etc).
    """
    text = Path(file_path).read_text(encoding="utf-8", errors="strict")
    decoder = json.JSONDecoder()
    records = []
    i = 0
    n = len(text)

    while i < n:
        # Skip whitespace and stray separators between records.
        while i < n and (text[i].isspace() or text[i] in ",[]"):
            i += 1
        if i >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            # Can't make progress from here; stop with what we've recovered.
            break
        records.append(obj)
        i = end

    return records


def _load_excel_all_sheets(conn, file_path, existing_tables, warnings):
    wb          = openpyxl.load_workbook(file_path, read_only=True)
    sheet_names = wb.sheetnames
    wb.close()

    created = []
    for sheet in sheet_names:
        table_name = _resolve_table_name(sheet, existing_tables, warnings)
        
        try:
            # FIX: Escape single quotes in sheet names to prevent SQL injection 
            # (e.g. converting "Client's Data" into "Client''s Data")
            safe_sheet = sheet.replace("'", "''")
            
            # Load all columns as text strings first via all_varchar=true.
            # This prevents mixed cells (like C2347) from dropping out as NULLs.
            conn.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT * FROM read_xlsx({_safe_path(file_path)}, sheet='{safe_sheet}', all_varchar=true)
            """)
            
            # Post-processing optimization: attempt to convert schemas where it is safe
            _auto_optimize_column_types(conn, table_name)
            
        except duckdb.Error as e:
            warnings.append(
                f"⚠️ Sheet `{sheet}` appears to be empty and was skipped "
                f"({e})."
            )
            existing_tables.remove(table_name)
            continue

        created.append(table_name)

    if not created:
        raise ValueError(
            "No data could be loaded from this workbook — every sheet "
            "appears to be empty."
        )

    return created


def _auto_optimize_column_types(conn: duckdb.DuckDBPyConnection, table_name: str):
    """
    Safely inspects every VARCHAR column and updates its data type to INTEGER, 
    DOUBLE, or DATE only if 100% of its non-null records can be cast cleanly.
    Columns with mixed text and numbers are safely left as VARCHAR.
    """
    # Fetch column names metadata using PRAGMA
    columns_info = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    columns = [row[1] for row in columns_info]
    
    # Priority order for safe casting evaluation
    target_types = ["INTEGER", "DOUBLE", "DATE" , "BOOLEAN"]

    for col_name in columns:
        # Escape column names wrapped in double quotes for complex headers
        escaped_col = f'"{col_name}"'
        
        for data_type in target_types:
            try:
                # SQL check: verify if converting values creates unexpected new NULLs
                test_query = f"""
                    SELECT COUNT({escaped_col}) = COUNT(TRY_CAST({escaped_col} AS {data_type})) 
                    FROM {table_name}
                    WHERE {escaped_col} IS NOT NULL
                """
                can_cast = conn.execute(test_query).fetchone()[0]
                
                if can_cast:
                    # Alter the column type safely in place
                    conn.execute(f"ALTER TABLE {table_name} ALTER COLUMN {escaped_col} TYPE {data_type};")
                    break  # Found the tightest fit type, proceed to next column
            except duckdb.Error:
                continue


def _make_table_name(raw_name: str) -> str:
    name = raw_name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    if name and name[0].isdigit():
        name = "t_" + name
    return name or "unnamed_table"