# ─────────────────────────────────────────────
#  core/session_mgr.py  –  Temp folder management
#
#  Features:
#    1. TTL based on last-accessed time → active sessions are never deleted mid-use
#    2. Windows PermissionError on rmtree surfaced as warning
#    3. delete_session()   → wipe entire session folder + DuckDB conn
#    4. delete_file_from_session() → remove one file + drop its DuckDB tables
#    5. Full terminal logging for every create / delete event (stdout + file)
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
# Writes to both the terminal (stdout via StreamHandler) and a persistent log
# file under TMP_BASE_DIR so operators can watch `tail -f` in production.

_cleanup_daemon_started = False

logger = logging.getLogger("core.session_mgr")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    # Console handler — visible in the terminal where `streamlit run` is executed
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(
        "[SESSION] %(asctime)s %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(ch)

    # File handler — persistent log under the session base dir
    try:
        Path(TMP_BASE_DIR).mkdir(parents=True, exist_ok=True)
        log_path = Path(TMP_BASE_DIR) / "session_cleanup.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)
    except Exception:
        pass  # if we can't write the log file, terminal output is still active


# ── Public API ────────────────────────────────

def create_session() -> tuple[str, Path]:
    """Create a new private session folder. Returns (session_id, folder_path)."""
    session_id  = str(uuid.uuid4())
    session_dir = Path(TMP_BASE_DIR) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    _touch_session(session_dir)
    logger.info(f"SESSION CREATED   id={session_id}  path={session_dir}")
    return session_id, session_dir


def get_session_dir(session_id: str) -> Path | None:
    session_dir = Path(TMP_BASE_DIR) / session_id
    if session_dir.exists():
        _touch_session(session_dir)
        return session_dir
    return None


def save_uploaded_file(session_dir: Path, uploaded_file) -> Path:
    """Save the uploaded Streamlit file to disk. Returns the saved path."""
    destination = session_dir / uploaded_file.name
    destination.write_bytes(uploaded_file.read())
    uploaded_file.seek(0)
    _touch_session(session_dir)
    logger.debug(f"FILE SAVED        path={destination}")
    return destination


def delete_file_from_session(
    session_dir: Path,
    filename: str,
    conn,           # duckdb.DuckDBPyConnection | None
    table_names: list[str],
) -> list[str]:
    """
    Remove one uploaded file from the session folder and drop its DuckDB tables.

    Args:
        session_dir:  Path to the session directory.
        filename:     The original uploaded filename (e.g. "sales.csv").
        conn:         Live DuckDB connection (can be None).
        table_names:  List of DuckDB table names that were created from this file.

    Returns:
        List of table names that were successfully dropped.
    """
    dropped: list[str] = []

    # 1. Drop DuckDB tables
    if conn is not None:
        for table in table_names:
            try:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
                dropped.append(table)
                logger.info(f"TABLE DROPPED     table={table}  file={filename}")
            except Exception as exc:
                logger.warning(f"Could not drop table {table}: {exc}")

    # 2. Delete file from disk
    file_path = session_dir / filename
    if file_path.exists():
        try:
            file_path.unlink()
            logger.info(f"FILE DELETED      path={file_path}")
        except Exception as exc:
            logger.warning(f"Could not delete file {file_path}: {exc}")

    return dropped


def delete_session(session_id: str, conn) -> bool:
    """
    Completely destroy a session: close DuckDB connection and wipe the folder.

    Args:
        session_id: The UUID string of the session.
        conn:       The DuckDB connection to close (can be None).

    Returns:
        True if the folder was deleted successfully, False otherwise.
    """
    # 1. Close DuckDB connection first (avoids Windows file-lock)
    if conn is not None:
        try:
            conn.close()
            logger.info(f"DUCKDB CLOSED     session={session_id}")
        except Exception as exc:
            logger.warning(f"Could not close DuckDB for session {session_id}: {exc}")

    # 2. Remove session folder
    session_dir = Path(TMP_BASE_DIR) / session_id
    failures: list[str] = []
    _try_delete(session_dir, failures)

    if failures:
        logger.error(f"SESSION DELETE FAILED  id={session_id}  path={session_dir}")
        return False

    logger.info(f"SESSION DELETED   id={session_id}  path={session_dir}")
    return True


def cleanup_old_sessions() -> list[str]:
    """
    Delete session folders that have been idle for longer than SESSION_TTL_MINUTES.
    Returns a list of any folders that could NOT be deleted.
    """
    base = Path(TMP_BASE_DIR)
    if not base.exists():
        return []

    cutoff   = time.time() - (SESSION_TTL_MINUTES * 60)
    failures = []
    removed  = 0

    for session_dir in base.iterdir():
        if not session_dir.is_dir():
            continue

        timestamp_file = session_dir / ".last_accessed"

        if not timestamp_file.exists():
            _try_delete(session_dir, failures)
            removed += 1
            continue

        try:
            last_accessed = float(timestamp_file.read_text())
        except Exception:
            logger.warning(f"Could not read timestamp for {session_dir.name}; treating as expired")
            _try_delete(session_dir, failures)
            removed += 1
            continue

        if last_accessed < cutoff:
            idle_minutes = (time.time() - last_accessed) / 60
            logger.info(
                f"SESSION EXPIRED   id={session_dir.name}  "
                f"idle={idle_minutes:.1f}min  ttl={SESSION_TTL_MINUTES}min"
            )
            _try_delete(session_dir, failures)
            removed += 1

    if removed:
        logger.info(f"CLEANUP DONE      removed={removed}  failures={len(failures)}")
    else:
        logger.debug("CLEANUP RUN       no expired sessions found")

    return failures


def start_cleanup_daemon(interval_seconds: int = 60):
    """
    Start a background daemon thread that periodically runs cleanup_old_sessions().
    Safe to call multiple times — daemon is only started once per process.
    """
    global _cleanup_daemon_started
    if _cleanup_daemon_started:
        return

    def _loop():
        while True:
            try:
                cleanup_old_sessions()
            except Exception as exc:
                logger.exception(f"Cleanup daemon error: {exc}")
            time.sleep(interval_seconds)

    t = threading.Thread(target=_loop, daemon=True, name="session_cleanup")
    t.start()
    _cleanup_daemon_started = True
    logger.info(f"CLEANUP DAEMON    started  interval={interval_seconds}s")


# ── Private helpers ───────────────────────────

def _touch_session(session_dir: Path):
    """Write the current Unix timestamp as the last-accessed marker."""
    (session_dir / ".last_accessed").write_text(str(time.time()))


def _try_delete(session_dir: Path, failures: list[str]):
    try:
        shutil.rmtree(session_dir)
        logger.info(f"FOLDER REMOVED    path={session_dir}")
    except PermissionError:
        failures.append(str(session_dir))
        logger.warning(f"PermissionError removing {session_dir}")
    except Exception as exc:
        failures.append(str(session_dir))
        logger.exception(f"Error removing {session_dir}: {exc}")