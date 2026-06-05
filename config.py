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
