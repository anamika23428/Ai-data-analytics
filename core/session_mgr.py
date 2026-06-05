# ─────────────────────────────────────────────
#  core/session_mgr.py  –  Temp folder management
#
#  Fixes:
#    1. TTL now based on last-accessed time, not creation time
#       so active sessions are never deleted mid-use
#    2. Windows PermissionError on rmtree is now surfaced as a warning
#       instead of silently swallowed
# ─────────────────────────────────────────────

import uuid
import shutil
import time
import os
from pathlib import Path
from config import TMP_BASE_DIR, SESSION_TTL_MINUTES


def create_session() -> tuple[str, Path]:
    """Create a new private session folder. Returns (session_id, folder_path)."""
    session_id  = str(uuid.uuid4())
    session_dir = Path(TMP_BASE_DIR) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    _touch_session(session_dir)   # write initial timestamp
    return session_id, session_dir


def get_session_dir(session_id: str) -> Path | None:
    session_dir = Path(TMP_BASE_DIR) / session_id
    if session_dir.exists():
        _touch_session(session_dir)   # update last-accessed time on every lookup
        return session_dir
    return None


def save_uploaded_file(session_dir: Path, uploaded_file) -> Path:
    """Save the uploaded Streamlit file to disk. Returns the saved path."""
    destination = session_dir / uploaded_file.name
    destination.write_bytes(uploaded_file.read())
    uploaded_file.seek(0)
    _touch_session(session_dir)   # accessing the session — refresh TTL
    return destination


def cleanup_old_sessions() -> list[str]:
    """
    Delete session folders that have been idle for longer than SESSION_TTL_MINUTES.
    Uses last-accessed time (not creation time) so active sessions are safe.

    Returns a list of any folders that could NOT be deleted (e.g. Windows lock).
    """
    base = Path(TMP_BASE_DIR)
    if not base.exists():
        return []

    cutoff   = time.time() - (SESSION_TTL_MINUTES * 60)
    failures = []

    for session_dir in base.iterdir():
        if not session_dir.is_dir():
            continue

        timestamp_file = session_dir / ".last_accessed"

        # No timestamp file → treat as expired
        if not timestamp_file.exists():
            _try_delete(session_dir, failures)
            continue

        last_accessed = float(timestamp_file.read_text())
        if last_accessed < cutoff:
            _try_delete(session_dir, failures)

    return failures


# ── Private helpers ───────────────────────────

def _touch_session(session_dir: Path):
    """Write the current timestamp as the last-accessed marker."""
    (session_dir / ".last_accessed").write_text(str(time.time()))


def _try_delete(session_dir: Path, failures: list[str]):
    """
    Try to delete a session folder.
    On Windows, DuckDB may still hold file handles open → catches PermissionError
    and records the path in failures instead of silently ignoring it.
    """
    try:
        shutil.rmtree(session_dir)
    except PermissionError:
        failures.append(str(session_dir))   # caller can log or warn the user
    except Exception:
        failures.append(str(session_dir))