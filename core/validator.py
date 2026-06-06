# ─────────────────────────────────────────────
#  core/validator.py  –  "Is this file safe to use?"
#
#  Five layered checks for plain-text files (CSV, JSON, TXT):
#    1. File size
#    2. Extension whitelist
#    3. Known binary magic bytes (EXE, PDF, PNG, JPEG, ZIP, ELF etc.)
#    4. Null byte check  (binaries almost always have null bytes)
#    5. Printability / entropy check  (real text is >95% printable)
#    6. Format-level check (JSON must parse, CSV must have structure)
#
#  For XLSX:
#    1. File size
#    2. Extension whitelist
#    3. filetype.guess() byte verification
# ─────────────────────────────────────────────

import json
import csv
import io
import re
import filetype
from pathlib import Path
from config import MAX_FILE_SIZE_BYTES, ALLOWED_EXTENSIONS

# How many bytes we read for all checks
SAMPLE_SIZE = 8192   # 8 KB — large enough to catch most disguised binaries

# Minimum ratio of printable bytes to consider a file "text-like"
PRINTABLE_THRESHOLD = 0.95

# XLSX expected MIME from filetype
EXPECTED_MIME = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

# Known dangerous magic byte signatures
# These are the first bytes of common binary formats
DANGEROUS_SIGNATURES = [
    (b"MZ",            "Windows executable (EXE/DLL)"),
    (b"\x7fELF",       "Linux binary (ELF)"),
    (b"%PDF",          "PDF document"),
    (b"\x89PNG",       "PNG image"),
    (b"\xff\xd8\xff",  "JPEG image"),
    (b"GIF8",          "GIF image"),
    (b"BM",            "BMP image"),
    (b"RIFF",          "WAV/AVI media file"),
    (b"\x1f\x8b",      "GZIP archive"),
    (b"7z\xbc\xaf",    "7-Zip archive"),
    (b"Rar!",          "RAR archive"),
    (b"\xca\xfe\xba\xbe", "Java class file"),
    (b"\xfe\xed\xfa",  "macOS binary"),
]

_READ_ONLY_PREFIX = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_DANGEROUS_SQL_TOKENS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|copy|pragma|call|execute|vacuum|analyze|transaction|begin|commit|rollback|export|import|install|load)\b",
    re.IGNORECASE,
)
_DANGEROUS_FUNCTIONS = re.compile(
    r"\b(read_csv_auto|read_csv|read_json_auto|read_json|read_parquet|read_xlsx|write_csv|read_blob|glob|system|shell)\s*\(",
    re.IGNORECASE,
)


def validate_file(uploaded_file) -> tuple[bool, str]:
    """
    Run all validation checks on an uploaded Streamlit file.

    Returns:
        (True,  "")            → file is safe to process
        (False, "reason why")  → file was rejected
    """

    # ── Check 1: File size ────────────────────────────────────
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        return False, f"File is too large. Maximum allowed size is {max_mb} MB."

    # ── Check 2: Extension whitelist ──────────────────────────
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return False, (
            f"'{suffix}' is not allowed. "
            "Please upload a CSV, XLSX, JSON, or TXT file."
        )

    # ── Read sample bytes (used by all remaining checks) ──────
    file_bytes = uploaded_file.read(SAMPLE_SIZE)
    uploaded_file.seek(0)   # always rewind so later code can read the file

    # ── XLSX: use filetype byte verification ──────────────────
    if suffix == ".xlsx":
        return _validate_xlsx(file_bytes, suffix)

    # ── Plain text files: run all 4 text checks ───────────────
    return _validate_text_file(file_bytes, suffix, uploaded_file)


def validate_sql_query(conn, sql: str, allowed_tables: list[str] | None = None) -> tuple[bool, str]:
    """
    Validate generated SQL before execution.

    The query must be a single read-only statement, must not use dangerous
    functions or DDL/DML tokens, and must parse successfully in DuckDB.
    """
    candidate = _normalize_sql(sql)
    if not candidate:
        return False, "Generated SQL is empty."

    if not _READ_ONLY_PREFIX.match(candidate):
        return False, "Only SELECT and WITH queries are allowed."

    statement, remainder = _split_single_statement(candidate)
    if statement is None or remainder:
        return False, "Only a single SQL statement is allowed."

    stripped = _strip_sql_comments(statement)
    if _DANGEROUS_SQL_TOKENS.search(stripped):
        return False, "Generated SQL contains a disallowed keyword."
    if _DANGEROUS_FUNCTIONS.search(stripped):
        return False, "Generated SQL uses a disallowed file or system function."

    if allowed_tables:
        allowed = {table.lower() for table in allowed_tables}
        referenced = _extract_table_candidates(stripped)
        unknown = sorted({name for name in referenced if name.lower() not in allowed})
        if unknown:
            return False, f"Generated SQL references unknown table(s): {', '.join(unknown)}"

    try:
        conn.execute(f"EXPLAIN {statement}")
    except Exception as exc:
        return False, f"Generated SQL could not be parsed or validated by DuckDB: {exc}"

    return True, ""


# ══════════════════════════════════════════════
#  XLSX validator
# ══════════════════════════════════════════════

def _validate_xlsx(file_bytes: bytes, suffix: str) -> tuple[bool, str]:
    """Verify XLSX files using filetype magic byte detection."""
    kind = filetype.guess(file_bytes)
    expected = EXPECTED_MIME[suffix]

    if kind is None or kind.mime != expected:
        detected = kind.mime if kind else "unknown format"
        return False, (
            f"File content looks like '{detected}' but extension is '{suffix}'. "
            "Please upload a real XLSX file."
        )
    return True, ""


# ══════════════════════════════════════════════
#  Plain-text file validators (CSV, JSON, TXT)
# ══════════════════════════════════════════════

def _validate_text_file(
    file_bytes: bytes,
    suffix: str,
    uploaded_file
) -> tuple[bool, str]:
    """
    Run four checks in order for CSV, JSON, TXT files.
    Stops and returns the first failure found.
    """

    # ── Check 3: Known binary magic bytes ─────────────────────
    for signature, description in DANGEROUS_SIGNATURES:
        if file_bytes.startswith(signature):
            return False, (
                f"'{uploaded_file.name}' appears to be a {description}, "
                f"not a valid {suffix[1:].upper()} file."
            )

    # ── Check 4: Null byte check ──────────────────────────────
    # Real text files almost never contain null bytes.
    # Binaries almost always do — even ones without known magic bytes.
    if b"\x00" in file_bytes:
        return False, (
            f"'{uploaded_file.name}' contains binary content (null bytes). "
            f"Please upload a real {suffix[1:].upper()} file."
        )

    # ── Check 5: Printability / entropy check ─────────────────
    # Measure what fraction of bytes are normal text characters.
    # Tabs (9), newlines (10, 13), and printable ASCII (32-126) count as text.
    if len(file_bytes) > 0:
        printable_count = sum(
            1 for b in file_bytes
            if b == 9 or b == 10 or b == 13 or 32 <= b <= 126
        )
        printable_ratio = printable_count / len(file_bytes)

        if printable_ratio < PRINTABLE_THRESHOLD:
            return False, (
                f"'{uploaded_file.name}' looks like a binary file "
                f"({printable_ratio:.0%} printable characters, expected ≥{PRINTABLE_THRESHOLD:.0%}). "
                f"Please upload a real {suffix[1:].upper()} file."
            )

    # ── Check 6: Format-level structural check ─────────────────
    if suffix == ".json":
        return _check_json_structure(file_bytes, uploaded_file.name)

    if suffix in (".csv", ".txt"):
        return _check_csv_structure(file_bytes, uploaded_file.name, suffix)

    # TXT with no further structure check — passes if it got here
    return True, ""


def _check_json_structure(file_bytes: bytes, filename: str) -> tuple[bool, str]:
    """
    Attempt to decode and parse the JSON sample.
    A renamed binary that passes byte checks will still fail here.
    """
    try:
        text = file_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False, (
            f"'{filename}' is not valid UTF-8 text. "
            "Please upload a real JSON file."
        )

    # We only have a sample (first 8KB), so the JSON may be truncated.
    # Try parsing the full sample; if it fails, try just the first line
    # (newline-delimited JSON). Only reject if both clearly fail.
    text_stripped = text.strip()

    # Try as full JSON
    try:
        json.loads(text_stripped)
        return True, ""
    except json.JSONDecodeError:
        pass

    # Try first non-empty line (newline-delimited JSON)
    for line in text_stripped.splitlines():
        line = line.strip()
        if line:
            try:
                json.loads(line)
                return True, ""  # at least one line parses → file looks like NDJSON
            except json.JSONDecodeError:
                break   # first line doesn't parse → not JSON

    return False, (
        f"'{filename}' does not appear to contain valid JSON. "
        "Please check the file and try again."
    )


def _check_csv_structure(
    file_bytes: bytes,
    filename: str,
    suffix: str
) -> tuple[bool, str]:
    """
    Check that the file has consistent row/column structure.
    A binary that passes byte checks won't have structured rows.
    """
    try:
        text = file_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        # Try latin-1 as fallback (common in older Excel exports)
        try:
            text = file_bytes.decode("latin-1", errors="strict")
        except Exception:
            return False, (
                f"'{filename}' could not be decoded as text. "
                f"Please upload a real {suffix[1:].upper()} file."
            )

    # Use Python's csv sniffer to detect the delimiter
    try:
        sample = text[:2048]
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except csv.Error:
        # Sniffer couldn't detect a delimiter — still might be a single-column file
        dialect = csv.excel   # fall back to comma

    # Read up to 5 rows and check column consistency
    try:
        reader = csv.reader(io.StringIO(text), dialect)
        rows   = [row for row, _ in zip(reader, range(5)) if row]
    except Exception:
        return False, (
            f"'{filename}' could not be parsed as a delimited file. "
            f"Please upload a real {suffix[1:].upper()} file."
        )

    if not rows:
        return False, f"'{filename}' appears to be empty."

    # Check that all sampled rows have the same number of columns
    col_counts = [len(row) for row in rows]
    if len(set(col_counts)) > 1 and max(col_counts) > 1:
        # Allow minor variation (1 col difference) for files with trailing commas
        if max(col_counts) - min(col_counts) > 1:
            return False, (
                f"'{filename}' has inconsistent column counts across rows "
                f"({min(col_counts)}–{max(col_counts)} columns). "
                "It may be corrupted or not a real CSV file."
            )

    return True, ""


def _normalize_sql(sql: str) -> str:
    return sql.strip().strip("\ufeff")


def _strip_sql_comments(sql: str) -> str:
    result = []
    i = 0
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False

    while i < len(sql):
        char = sql[i]
        next_char = sql[i + 1] if i + 1 < len(sql) else ""

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
                result.append(char)
            i += 1
            continue

        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if not in_single and not in_double and char == "-" and next_char == "-":
            in_line_comment = True
            i += 2
            continue
        if not in_single and not in_double and char == "/" and next_char == "*":
            in_block_comment = True
            i += 2
            continue

        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double

        result.append(char)
        i += 1

    return "".join(result)


def _split_single_statement(sql: str) -> tuple[str | None, str]:
    statement = []
    in_single = False
    in_double = False

    for index, char in enumerate(sql):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == ";" and not in_single and not in_double:
            remainder = sql[index + 1 :].strip()
            current = "".join(statement).strip()
            return current, remainder
        statement.append(char)

    return "".join(statement).strip(), ""


def _extract_table_candidates(sql: str) -> list[str]:
    pattern = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w\.]*)", re.IGNORECASE)
    names = []
    for match in pattern.finditer(sql):
        name = match.group(1).split(".")[-1]
        if name.lower() not in {"select", "with"}:
            names.append(name)
    return names