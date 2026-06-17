# ─────────────────────────────────────────────
# core/ddl_utils.py  – generate privacy-safe DDL summaries
#
# Key improvement: for low-cardinality VARCHAR columns (≤ 20 distinct values),
# sample values are appended as a comment in the DDL. This lets the LLM pick
# the correct table by matching query *values* (e.g. "OFFSHORE", "ONSITE")
# to actual column contents — not by matching query words to table names.
# ─────────────────────────────────────────────

from collections import Counter
from typing import List


# Max distinct values to show inline per categorical column.
# Keeps prompt size small while still giving the LLM enough signal.
_SAMPLE_MAX_DISTINCT = 20
# Only sample columns whose total row count is below this threshold —
# avoids full-scanning million-row tables just for DDL hints.
_SAMPLE_ROW_LIMIT = 50_000


def generate_privacy_safe_ddl(
    conn,
    table_name: str,
    redact: bool = False,
    max_columns: int = 50,
    include_samples: bool = True,
) -> str:
    """
    Produce a privacy-safe DDL-like string for `table_name`.

    When include_samples=True (default), VARCHAR/TEXT columns with
    ≤ _SAMPLE_MAX_DISTINCT distinct values get an inline comment showing
    those values, e.g.:
        ONSITE_OFFSHORE VARCHAR  -- values: OFFSHORE, ONSITE

    This helps the LLM select the correct table based on what values exist
    in columns, not just column names or table names.

    - redact=True replaces column names with col_1, col_2, … (disables sampling)
    - max_columns: if table exceeds this, schema is compressed (head+tail+summary)
    """
    col_info = conn.execute(f"DESCRIBE {table_name}").fetchall()
    columns  = [row[0] for row in col_info]
    types    = [row[1] for row in col_info]
    n        = len(columns)

    # ── Optionally collect distinct-value hints for categorical columns ───────
    value_hints: dict[str, str] = {}
    if include_samples and not redact:
        try:
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            if row_count <= _SAMPLE_ROW_LIMIT:
                for col, typ in zip(columns, types):
                    # Only VARCHAR / TEXT columns are useful for value hints
                    if not any(t in typ.upper() for t in ("VARCHAR", "TEXT", "CHAR")):
                        continue
                    escaped = f'"{col}"'
                    try:
                        rows = conn.execute(
                            f"SELECT DISTINCT {escaped} FROM {table_name} "
                            f"WHERE {escaped} IS NOT NULL "
                            f"LIMIT {_SAMPLE_MAX_DISTINCT + 1}"
                        ).fetchall()
                        vals = [str(r[0]) for r in rows if r[0] is not None]
                        if 1 < len(vals) <= _SAMPLE_MAX_DISTINCT:
                            value_hints[col] = ", ".join(vals)
                        # If > _SAMPLE_MAX_DISTINCT distinct values → high cardinality
                        # (free text / IDs) — skip, not useful for table selection
                    except Exception:
                        continue
        except Exception:
            pass  # sampling is best-effort; never break DDL generation

    # ── Format column lines ───────────────────────────────────────────────────
    def _col_line(name: str, typ: str, display_name: str) -> str:
        hint = value_hints.get(name)
        if hint:
            return f"  {display_name} {typ}  -- values: {hint}"
        return f"  {display_name} {typ}"

    if redact:
        display_names = [f"col_{i+1}" for i in range(n)]
    else:
        display_names = columns

    pairs = list(zip(columns, types, display_names))

    if n <= max_columns:
        lines = [f"CREATE TABLE {table_name} ("]
        lines += [_col_line(col, typ, disp) for col, typ, disp in pairs]
        lines += [");"]
        return "\n".join(lines)

    # Compressed schema: head + summary + tail
    head_count = 10
    tail_count = 10
    head   = pairs[:head_count]
    middle = pairs[head_count:-tail_count]
    tail   = pairs[-tail_count:]

    middle_types = [t for (_, t, _) in middle]
    type_counts  = Counter(middle_types)
    type_summary = ", ".join(f"{cnt}x {typ}" for typ, cnt in type_counts.items()) or "none"

    lines  = [f"CREATE TABLE {table_name} ("]
    lines += [_col_line(col, typ, disp) for col, typ, disp in head]
    lines += [f"  -- ... {len(middle)} more columns summarized below ..."]
    lines += [f"  -- Summary of omitted columns by type: {type_summary}"]
    lines += [_col_line(col, typ, disp) for col, typ, disp in tail]
    lines += [");", f"-- Total columns: {n} (head {head_count} + middle {len(middle)} + tail {tail_count})"]

    return "\n".join(lines)


def generate_multi_table_ddl(
    conn,
    table_names: List[str],
    redact: bool = False,
    max_columns: int = 50,
    include_samples: bool = True,
) -> str:
    """
    Produce a combined DDL string for all tables in table_names.
    Each table block is separated by a blank line.
    """
    blocks = []
    for table_name in table_names:
        try:
            blocks.append(
                generate_privacy_safe_ddl(
                    conn,
                    table_name,
                    redact=redact,
                    max_columns=max_columns,
                    include_samples=include_samples,
                )
            )
        except Exception as exc:
            blocks.append(f"-- Could not describe table {table_name}: {exc}")
    return "\n\n".join(blocks)