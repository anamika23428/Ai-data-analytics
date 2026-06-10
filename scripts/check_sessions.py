#!/usr/bin/env python3
"""List current session folders and optionally run cleanup_old_sessions().

Usage:
  python scripts/check_sessions.py [--cleanup]

--cleanup   Run `cleanup_old_sessions()` after listing folders.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `from config` works when running
# this script directly (python scripts/check_sessions.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TMP_BASE_DIR
from core import session_mgr
import argparse


def list_sessions():
    base = Path(TMP_BASE_DIR)
    if not base.exists():
        print(f"Base dir does not exist: {base}")
        return

    entries = sorted([p for p in base.iterdir() if p.is_dir()])
    if not entries:
        print(f"No session folders in {base}")
        return

    print(f"Session folders in {base}:")
    for p in entries:
        ts_file = p / ".last_accessed"
        ts = ts_file.read_text().strip() if ts_file.exists() else "<missing>"
        print(f"- {p.name}  last_accessed: {ts}")


def run_cleanup():
    failures = session_mgr.cleanup_old_sessions()
    if failures:
        print("\nDeletion failures:")
        for f in failures:
            print(" -", f)
    else:
        print("\nCleanup completed successfully; no failures reported.")


def main():
    parser = argparse.ArgumentParser(description="Check session folders and run cleanup")
    parser.add_argument("--cleanup", action="store_true", help="Run cleanup_old_sessions() after listing")
    args = parser.parse_args()

    list_sessions()
    if args.cleanup:
        print("\nRunning cleanup...")
        run_cleanup()


if __name__ == "__main__":
    main()
