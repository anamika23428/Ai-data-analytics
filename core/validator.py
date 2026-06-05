# ─────────────────────────────────────────────
#  core/validator.py  –  "Is this file safe to use?"
#
#  Two checks:
#    1. File size  – reject anything too large
#    2. File type  – reject anything that isn't
#                   CSV / XLSX / JSON / TXT
#
#  Uses 'filetype' (pure Python, works on Windows)
#  instead of python-magic which needs a C library.
# ─────────────────────────────────────────────

import filetype  # pure-Python, works on Windows with no extra setup
from pathlib import Path
from config import MAX_FILE_SIZE_BYTES, ALLOWED_EXTENSIONS

# XLSX is a binary format — filetype can verify it from its magic bytes.
# CSV, JSON, TXT are plain text — they have no magic bytes, so we trust the extension.
BINARY_EXTENSIONS = {".xlsx"}

# What MIME type filetype should detect for each binary extension
EXPECTED_MIME = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def validate_file(uploaded_file) -> tuple[bool, str]:
    """
    Check an uploaded Streamlit file object.

    Returns:
        (True,  "")            → file is fine
        (False, "reason why")  → file was rejected
    """

    # ── 1. Size check ────────────────────────────────
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
        return False, f"File is too large. Maximum allowed size is {max_mb} MB."

    # ── 2. Extension check ───────────────────────────
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return False, f"'{suffix}' is not allowed. Please upload a CSV, XLSX, JSON, or TXT file."

    # ── 3. Byte-level check for XLSX only ────────────
    # filetype returns None for plain-text files (CSV/JSON/TXT) because they
    # have no binary signature — that's normal, not an error.
    if suffix in BINARY_EXTENSIONS:
        file_bytes = uploaded_file.read(2048)
        uploaded_file.seek(0)  # rewind so the rest of the app can still read it

        kind = filetype.guess(file_bytes)
        expected = EXPECTED_MIME[suffix]

        if kind is None or kind.mime != expected:
            detected = kind.mime if kind else "unknown binary format"
            return False, (
                f"File content looks like '{detected}' but the extension is '{suffix}'. "
                "Please upload a real XLSX file."
            )
    else:
        uploaded_file.seek(0)  # rewind for consistency

    return True, ""