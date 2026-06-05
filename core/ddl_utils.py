# ─────────────────────────────────────────────
# core/ddl_utils.py  – generate privacy-safe DDL summaries
#
# Exposes `generate_privacy_safe_ddl(conn, table_name, redact=False, max_columns=50)`
# which returns a DDL-like string suitable for sending to an LLM. If a table has
# more than `max_columns`, the schema is compressed into a human-readable
# summary (first/last columns + type counts) to reduce token usage and avoid
# leaking large schemas.
# ─────────────────────────────────────────────

from collections import Counter
from typing import List


def generate_privacy_safe_ddl(conn, table_name: str, redact: bool = False, max_columns: int = 50) -> str:
    """
    Produce a privacy-safe DDL-like string for `table_name` using
    DuckDB's `DESCRIBE` output.

    - If `redact` is True column names are replaced with generic names
      (`col_1`, `col_2`, ...).
    - If the table has more than `max_columns` columns the output is
      compressed: show the first 10 and last 10 columns, and include a
      summary of the remaining columns by type counts.

    Returns a string (multi-line) suitable for inclusion in an LLM prompt.
    """
    col_info = conn.execute(f"DESCRIBE {table_name}").fetchall()
    columns = [row[0] for row in col_info]
    types = [row[1] for row in col_info]

    n = len(columns)
    # Helper to format a list of (name,type) pairs
    def _format_pairs(pairs: List[tuple]) -> List[str]:
        return [f"  {name} {typ}" for name, typ in pairs]

    # Optionally redact names
    if redact:
        display_names = [f"col_{i+1}" for i in range(n)]
    else:
        display_names = columns

    pairs = list(zip(display_names, types))

    if n <= max_columns:
        lines = [f"CREATE TABLE {table_name} ("] + _format_pairs(pairs) + [");"]
        return "\n".join(lines)

    # Compress schema: show head/tail + summary
    head_count = 10
    tail_count = 10
    head = pairs[:head_count]
    tail = pairs[-tail_count:]
    middle = pairs[head_count:-tail_count]

    # Type summary for middle columns
    middle_types = [t for (_, t) in middle]
    type_counts = Counter(middle_types)
    type_summary = ", ".join(f"{cnt}x {typ}" for typ, cnt in type_counts.items()) if type_counts else "none"

    lines = [f"CREATE TABLE {table_name} ("]
    lines += _format_pairs(head)
    lines += [f"  -- ... {len(middle)} more columns summarized below ..."]
    lines += [f"  -- Summary of omitted columns by type: {type_summary}"]
    lines += _format_pairs(tail)
    lines += [") ;", f"-- Total columns: {n} (head {head_count} + middle {len(middle)} + tail {tail_count})"]

    return "\n".join(lines)
