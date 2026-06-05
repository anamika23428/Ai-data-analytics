# ─────────────────────────────────────────────
#  core/ingestor.py
#
#  Fixes:
#    1. Duplicate table names across files → warn instead of silent overwrite
#    2. File paths with spaces → use $$ quoting in SQL
# ─────────────────────────────────────────────

import re
from pathlib import Path
import duckdb
import openpyxl


def get_or_create_connection(session_state) -> duckdb.DuckDBPyConnection:
    """Return the shared DuckDB connection, installing excel extension only once."""
    if "duckdb_conn" not in session_state or session_state.duckdb_conn is None:
        conn = duckdb.connect()
        conn.execute("INSTALL excel")
        conn.execute("LOAD excel")
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
        tables = [_load_json(conn, file_path, existing_tables, warnings)]
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
    conn.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM read_csv_auto({_safe_path(file_path)}, header=true)
    """)
    return table_name


def _load_json(conn, file_path, existing_tables, warnings):
    table_name = _resolve_table_name(file_path.stem, existing_tables, warnings)
    conn.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM read_json_auto({_safe_path(file_path)})
    """)
    return table_name


def _load_excel_all_sheets(conn, file_path, existing_tables, warnings):
    wb          = openpyxl.load_workbook(file_path, read_only=True)
    sheet_names = wb.sheetnames
    wb.close()

    created = []
    for sheet in sheet_names:
        table_name = _resolve_table_name(sheet, existing_tables, warnings)
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_xlsx({_safe_path(file_path)}, sheet='{sheet}')
        """)
        created.append(table_name)

    return created


def _make_table_name(raw_name: str) -> str:
    name = raw_name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    if name and name[0].isdigit():
        name = "t_" + name
    return name or "unnamed_table"