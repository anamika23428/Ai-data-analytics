# ─────────────────────────────────────────────
# core/session_mgr.py  –  Temp folder management
# ─────────────────────────────────────────────

import uuid
import shutil
import time
import os
import threading
from pathlib import Path
from config import TMP_BASE_DIR, SESSION_TTL_MINUTES
import logging

# ── Logger setup ──────────────────────────────

_cleanup_daemon_started = False

logger = logging.getLogger("core.session_mgr")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(
        "[SESSION] %(asctime)s %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(ch)


# ── Public API ────────────────────────────────

def create_session() -> tuple[str, Path]:
    """Create a new private session folder."""
    session_id = str(uuid.uuid4())
    session_dir = Path(TMP_BASE_DIR) / session_id

    session_dir.mkdir(parents=True, exist_ok=True)
    _touch_session(session_dir)

    logger.info(f"SESSION CREATED   id={session_id} path={session_dir}")

    return session_id, session_dir


def get_session_dir(session_id: str) -> Path | None:
    """Return session directory if exists, else None."""
    session_dir = Path(TMP_BASE_DIR) / session_id

    if session_dir.exists():
        _touch_session(session_dir)
        return session_dir

    return None


def save_uploaded_file(session_dir: Path, uploaded_file) -> Path:
    """Save uploaded file (Streamlit) to session folder."""
    destination = session_dir / uploaded_file.name

    destination.write_bytes(uploaded_file.read())
    uploaded_file.seek(0)

    _touch_session(session_dir)

    logger.debug(f"FILE SAVED        path={destination}")

    return destination


def delete_file_from_session(
    session_dir: Path,
    filename: str,
    conn,                       # duckdb connection (optional)
    table_names: list[str],
) -> list[str]:
    """
    Delete a file from session folder and drop its DuckDB tables.
    Returns list of successfully dropped tables.
    """
    dropped_tables = []

    file_path = session_dir / filename

    # Delete file
    if file_path.exists():
        try:
            file_path.unlink()
            logger.info(f"FILE DELETED      path={file_path}")
        except Exception as e:
            logger.exception(f"Error deleting file {file_path}: {e}")

    # Drop tables in DuckDB
    if conn:
        for table in table_names:
            try:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
                dropped_tables.append(table)
                logger.info(f"TABLE DROPPED     table={table}")
            except Exception as e:
                logger.exception(f"Error dropping table {table}: {e}")

    return dropped_tables


def delete_session(session_id: str, conn) -> bool:
    """
    Completely delete a session: close DuckDB connection and delete folder.
    """
    session_dir = Path(TMP_BASE_DIR) / session_id

    # Close DB connection
    if conn:
        try:
            conn.close()
            logger.info(f"DB CONNECTION CLOSED for session={session_id}")
        except Exception as e:
            logger.exception(f"Error closing DB connection: {e}")

    # Delete folder
    if session_dir.exists():
        try:
            shutil.rmtree(session_dir)
            logger.info(f"SESSION DELETED    id={session_id} path={session_dir}")
            return True
        except Exception as e:
            logger.exception(f"Error deleting session {session_id}: {e}")
            return False

    return False


def cleanup_old_sessions() -> list[str]:
    """
    Delete session folders inactive for longer than TTL.
    Returns folders that could NOT be deleted.
    """
    failures = []
    base = Path(TMP_BASE_DIR)

    if not base.exists():
        return []

    now = time.time()
    ttl_seconds = SESSION_TTL_MINUTES * 60

    for session_dir in base.iterdir():
        if not session_dir.is_dir():
            continue

        access_file = session_dir / ".last_accessed"

        try:
            if access_file.exists():
                last_access = float(access_file.read_text().strip())
            else:
                # fallback: use folder modified time
                last_access = session_dir.stat().st_mtime

            age = now - last_access

            if age > ttl_seconds:
                logger.info(f"CLEANUP: deleting expired session {session_dir}")
                _try_delete(session_dir, failures)

        except Exception as e:
            failures.append(str(session_dir))
            logger.exception(f"Error checking session {session_dir}: {e}")

    return failures


def start_cleanup_daemon(interval_seconds: int = 60):
    """
    Start background cleanup thread.
    Runs periodically forever.
    """
    global _cleanup_daemon_started

    if _cleanup_daemon_started:
        return

    def _worker():
        while True:
            try:
                failures = cleanup_old_sessions()
                if failures:
                    logger.warning(f"CLEANUP FAILURES: {failures}")
            except Exception as e:
                logger.exception(f"Cleanup daemon error: {e}")

            time.sleep(interval_seconds)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    _cleanup_daemon_started = True

    logger.info("CLEANUP DAEMON STARTED")


# ── Private helpers ───────────────────────────

def _touch_session(session_dir: Path):
    """Update last accessed timestamp."""
    try:
        (session_dir / ".last_accessed").write_text(str(time.time()))
    except Exception as e:
        logger.exception(f"Error touching session {session_dir}: {e}")


def _try_delete(session_dir: Path, failures: list[str]):
    """Try deleting a folder safely."""
    try:
        shutil.rmtree(session_dir)
        logger.info(f"FOLDER REMOVED    path={session_dir}")
    except PermissionError:
        failures.append(str(session_dir))
        logger.warning(f"PermissionError removing {session_dir}")
    except Exception as exc:
        failures.append(str(session_dir))
        logger.exception(f"Error removing {session_dir}: {exc}")