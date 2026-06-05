# ─────────────────────────────────────────────
#  core/session_mgr.py  –  Temp folder management
#
#  Every user gets their own private folder under /tmp/
#  so their files never mix with anyone else's.
#
#  Folders are cleaned up automatically after the TTL
#  (30 minutes by default) has passed.
# ─────────────────────────────────────────────

import uuid
import shutil
import time
from pathlib import Path
from config import TMP_BASE_DIR, SESSION_TTL_MINUTES


def create_session() -> tuple[str, Path]:
    """
    Create a new session:
      - Generate a unique session ID
      - Create a private folder for this session at /tmp/analytics_sessions/<session_id>/
      - Return (session_id, folder_path)
    """
    session_id = str(uuid.uuid4())          # e.g. "3f2a1b4c-..."
    session_dir = Path(TMP_BASE_DIR) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Write a small timestamp file so we know when this session started
    (session_dir / ".created_at").write_text(str(time.time()))

    return session_id, session_dir


def get_session_dir(session_id: str) -> Path | None:
    """
    Look up an existing session folder.
    Returns the Path if it exists, or None if it has been deleted / never existed.
    """
    session_dir = Path(TMP_BASE_DIR) / session_id
    return session_dir if session_dir.exists() else None


def save_uploaded_file(session_dir: Path, uploaded_file) -> Path:
    """
    Write the uploaded Streamlit file to the session folder on disk.
    Returns the path to the saved file.
    """
    destination = session_dir / uploaded_file.name
    destination.write_bytes(uploaded_file.read())
    uploaded_file.seek(0)   # rewind in case something else needs to read it
    return destination


def cleanup_old_sessions():
    """
    Walk through all session folders and delete any that are older
    than SESSION_TTL_MINUTES.  Call this once at app startup.
    """
    base = Path(TMP_BASE_DIR)
    if not base.exists():
        return

    cutoff = time.time() - (SESSION_TTL_MINUTES * 60)   # timestamp N minutes ago

    for session_dir in base.iterdir():
        timestamp_file = session_dir / ".created_at"

        if not timestamp_file.exists():
            # No timestamp → treat as expired
            shutil.rmtree(session_dir, ignore_errors=True)
            continue

        created_at = float(timestamp_file.read_text())
        if created_at < cutoff:
            shutil.rmtree(session_dir, ignore_errors=True)
