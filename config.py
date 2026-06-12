# ─────────────────────────────────────────────
#  config.py  –  App-wide settings
#  Change values here; nothing else needs edits.
# ─────────────────────────────────────────────

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
TMP_BASE_DIR = "/tmp/analytics_sessions"

# Max allowed prompt length (characters) for NL→SQL requests
PROMPT_MAX_LENGTH = 2000

# When building a DDL for the LLM, if a table has more than this many
# columns we'll compress/summarise the schema for privacy and token cost.
DDL_MAX_COLUMNS = 50

# ─────────────────────────────────────────────
#  Query Router (llm_router.py) settings
#  Stage 1: regex keyword pre-filter (zero cost, ~1ms)
#  Stage 2: local Ollama LLM (data NEVER leaves your machine)
# ─────────────────────────────────────────────

# Base URL for your local Ollama instance
OLLAMA_BASE_URL = "http://localhost:11434"

# The model used by the query router (small, fast, local)
# Recommended: llama3.2:3b  or  mistral:7b
ROUTER_MODEL = "llama3.2:3b"

# Seconds before we give up waiting for Ollama
OLLAMA_TIMEOUT = 30

# Confidence threshold below which the UI shows a clarification gate
# Set to 0.0 to disable the gate entirely
ROUTER_LOW_CONFIDENCE_THRESHOLD = "LOW"

# ─────────────────────────────────────────────
#  Route A model settings
# ─────────────────────────────────────────────

# Stage 1 — Intent extraction (small, fast)
INTENT_MODEL = "llama3.2:3b"

# Stage 2 — SQL generation (code-specialised)
# Pull with: ollama pull qwen2.5-coder:7b
# Lighter alternative: ollama pull qwen2.5-coder:1.5b
SQL_MODEL = "qwen2.5-coder:7b"