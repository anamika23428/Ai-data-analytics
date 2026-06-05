# ─────────────────────────────────────────────
#  core/ingestor.py  –  "How do we read this file?"
#
#  Router logic:
#    CSV / JSON / TXT  →  DuckDB reads directly
#    XLSX              →  openpyxl converts to DataFrame first,
#                         then we hand it to DuckDB
#
#  Either way, the output is always a DuckDB connection
#  with a table called  "data"  ready to query.
# ─────────────────────────────────────────────

from pathlib import Path
import duckdb
import pandas as pd
import openpyxl


def load_file_into_duckdb(file_path: Path) -> duckdb.DuckDBPyConnection:
    """
    Read a file and load it into an in-memory DuckDB table called "data".

    Returns a DuckDB connection so the caller can run SQL on it.
    """
    suffix = file_path.suffix.lower()
    conn = duckdb.connect()   # fresh in-memory database

    if suffix in (".csv", ".txt"):
        _load_csv(conn, file_path)

    elif suffix == ".json":
        _load_json(conn, file_path)

    elif suffix == ".xlsx":
        _load_excel(conn, file_path)

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return conn


# ── Private helpers ───────────────────────────────────────

def _load_csv(conn: duckdb.DuckDBPyConnection, file_path: Path):
    """
    DuckDB can read CSV and TXT files natively with read_csv_auto().
    It figures out column names, separators, and types on its own.
    """
    sql = f"CREATE TABLE data AS SELECT * FROM read_csv_auto('{file_path}', header=true)"
    conn.execute(sql)


def _load_json(conn: duckdb.DuckDBPyConnection, file_path: Path):
    """
    DuckDB can read JSON files natively with read_json_auto().
    Works for both JSON arrays and newline-delimited JSON.
    """
    sql = f"CREATE TABLE data AS SELECT * FROM read_json_auto('{file_path}')"
    conn.execute(sql)


def _load_excel(conn: duckdb.DuckDBPyConnection, file_path: Path):
    """
    DuckDB doesn't speak XLSX, so we use openpyxl to open the workbook,
    turn the first sheet into a pandas DataFrame, then hand that
    DataFrame to DuckDB as a virtual table.
    """
    workbook = openpyxl.load_workbook(file_path, data_only=True)
    sheet    = workbook.active                         # first / active sheet

    # Pull all rows out of the sheet
    rows = list(sheet.iter_rows(values_only=True))

    if not rows:
        raise ValueError("The Excel file appears to be empty.")

    headers = [str(cell) if cell is not None else f"col_{i}"
               for i, cell in enumerate(rows[0])]

    df = pd.DataFrame(rows[1:], columns=headers)      # row 0 = headers

    # Register the DataFrame with DuckDB so we can query it like a table
    conn.register("data", df)
