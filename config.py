# ─────────────────────────────────────────────
#  config.py  –  App-wide settings
#  Change values here; nothing else needs edits.
# ─────────────────────────────────────────────


# Your existing app.py code continues below...
import sys
from pathlib import Path
import socket

# How long (in minutes) we keep a user's temp folder alive
SESSION_TTL_MINUTES = 30

# Biggest file we'll accept (in megabytes)
MAX_FILE_SIZE_MB = 50

# Convert MB → bytes for easy comparison later
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# File types users are allowed to upload
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".json", ".txt"}

# The MIME types that match those extensions
ALLOWED_MIME_TYPES = {
    "text/csv",
    "text/plain",
    "application/json",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
}

# Where we store temp session folders on the server
# (If on Windows, Python automatically resolves this to a safe temporary path like C:\tmp\...)
TMP_BASE_DIR = "/tmp/analytics_sessions"

# Max allowed prompt length (characters) for NL→SQL requests
PROMPT_MAX_LENGTH = 2000

# When building a DDL for the LLM, if a table has more than this many
# columns we'll compress/summarise the schema for privacy and token cost.
DDL_MAX_COLUMNS = 50


# ─────────────────────────────────────────────
#  Ollama System Settings
# ─────────────────────────────────────────────

# Base URL for your local Ollama instance
OLLAMA_BASE_URL = "http://localhost:11434"

# Seconds before we give up waiting for Ollama to respond
OLLAMA_TIMEOUT = 180  # Increased slightly to give the Insight model time to read larger data

# ─────────────────────────────────────────────
#  Query Router (llm_router.py) settings
#  Stage 1: regex keyword pre-filter (zero cost, ~1ms)
#  Stage 2: local Ollama LLM (data NEVER leaves your machine)
# ─────────────────────────────────────────────

# The model used by the query router (small, fast, local)
ROUTER_MODEL = "llama3.2:3b"

# Confidence threshold below which the UI shows a clarification gate
ROUTER_LOW_CONFIDENCE_THRESHOLD = "LOW"

# ─────────────────────────────────────────────
#  Route A (Visualization Pipeline) Settings
# ─────────────────────────────────────────────

# Stage 1 — Intent extraction (small, fast)
INTENT_MODEL = "llama3.2:3b"

# ─────────────────────────────────────────────
#  Route B (SQL & Analytics Pipeline) Settings
# ─────────────────────────────────────────────

# The heavy-duty coding model used to generate DuckDB SQL queries
SQL_MODEL = "qwen2.5-coder:7b"

# NEW: The analytical model used to convert SQL data into human-readable English answers
INSIGHT_MODEL = "llama3:8b"
