# ─────────────────────────────────────────────
# core/nl2sql.py  – NL→SQL pipeline
#
# Backends (in priority order):
#   1. Anthropic Claude (ANTHROPIC_API_KEY set)  ← recommended
#   2. Local Ollama  (fallback / offline)
#
# Always returns:
#   (sql: str | None, llm_prompt: str, used_llm: bool)
# ─────────────────────────────────────────────

import json
import os
import re
import urllib.error
import urllib.request
from typing import Tuple
from config import DDL_MAX_COLUMNS
from core import ddl_utils


# ══════════════════════════════════════════════
#  Prompt builder
# ══════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are an expert SQL analyst working with DuckDB. Your job is to translate a \
natural-language question into a single, correct DuckDB SQL query.

## Rules
- Output ONLY the SQL query — no explanation, no markdown fences, no preamble.
- Use only SELECT or WITH statements. Never use INSERT, UPDATE, DELETE, DROP, \
CREATE, ALTER, or any data-modifying statement.
- Reference only the tables and columns provided in the schema below.
- Column names that contain spaces or special characters must be quoted with \
double-quotes (e.g. "First Name").
- When joining tables, always use explicit JOIN … ON syntax.
- Prefer readable aliases (e.g. SELECT o.order_id, c.name FROM orders o JOIN \
customers c ON …).
- If the question is ambiguous, make a sensible assumption and write the query \
that most users would expect.
- End the query with a semicolon.
"""

_USER_TEMPLATE = """\
## Available tables
{scope}

## Schema (privacy-safe DDL)
{ddl}

## Question
{question}

## SQL
"""


def _build_prompt(ddl: str, question: str, table_names: list[str]) -> str:
    scope = ", ".join(table_names)
    return _USER_TEMPLATE.format(scope=scope, ddl=ddl, question=question.strip())


def _build_schema_context(conn, table_names: list[str], redact: bool, max_columns: int) -> str:
    sections = []
    for table_name in table_names:
        ddl = ddl_utils.generate_privacy_safe_ddl(conn, table_name, redact=redact, max_columns=max_columns)
        sections.append(f"-- Table: {table_name}\n{ddl}")
    return "\n\n".join(sections)


def _normalize_table_names(table_names: str | list[str]) -> list[str]:
    if isinstance(table_names, str):
        return [table_names]
    return list(table_names)


# ══════════════════════════════════════════════
#  Backend: Anthropic Claude
# ══════════════════════════════════════════════

def _call_anthropic(user_prompt: str) -> tuple[str | None, str]:
    """
    Call Anthropic Claude (claude-sonnet-4-20250514) to generate SQL.

    Requires ANTHROPIC_API_KEY environment variable.

    Returns (sql_text, error_message).  On success error_message is "".
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None, "ANTHROPIC_API_KEY is not set."

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "system": _SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
    }).encode("utf-8")

    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))

        # Claude returns content as a list of blocks
        content_blocks = data.get("content", [])
        text = "".join(
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        ).strip()

        if text:
            return text, ""
        return None, "Claude returned an empty response."

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return None, f"Anthropic API HTTP {exc.code}: {body}"
    except urllib.error.URLError as exc:
        return None, f"Anthropic API unreachable: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"Anthropic API returned invalid JSON: {exc}"
    except Exception as exc:
        return None, f"Anthropic API call failed: {exc}"


# ══════════════════════════════════════════════
#  Backend: local Ollama (fallback)
# ══════════════════════════════════════════════

def _call_ollama(prompt: str) -> tuple[str | None, str]:
    host  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.1")

    payload = json.dumps({
        "model": model,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }).encode("utf-8")

    request = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data.get("message", {}).get("content", "").strip()
        if content:
            return content, ""
        return None, "Ollama returned an empty response."
    except urllib.error.URLError as exc:
        return None, f"Ollama is not available at {host}: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"Ollama returned invalid JSON: {exc}"
    except Exception as exc:
        return None, f"Ollama call failed: {exc}"


# ══════════════════════════════════════════════
#  Public entry point
# ══════════════════════════════════════════════

def generate_sql_from_prompt(
    conn,
    table_name: str | list[str],
    prompt: str,
    redact: bool = True,
    max_columns: int = DDL_MAX_COLUMNS,
    max_prompt_length: int = 2000,
    llm_provider: str = "auto",          # "auto" | "anthropic" | "ollama"
    openai_api_key: str | None = None,   # kept for backward-compat, ignored
) -> Tuple[str | None, str, bool]:
    """
    Build a privacy-safe DDL and call an LLM to generate a DuckDB SQL query.

    Priority:
      - "anthropic" or "auto" + ANTHROPIC_API_KEY set  → Claude
      - "ollama"   or "auto" + no API key              → local Ollama

    Returns:
        (sql_or_none, llm_prompt, used_llm)
    """
    table_names = _normalize_table_names(table_name)
    ddl         = _build_schema_context(conn, table_names, redact=redact, max_columns=max_columns)
    user_prompt = _build_prompt(ddl, prompt, table_names)

    if len(user_prompt) > max_prompt_length:
        msg = f"Prompt exceeds {max_prompt_length} characters. Please shorten your question."
        return None, user_prompt + f"\n\n-- ERROR: {msg}", False

    # ── Choose backend ─────────────────────────────
    use_anthropic = (
        llm_provider == "anthropic"
        or (llm_provider in ("auto", "ollama") and os.environ.get("ANTHROPIC_API_KEY", "").strip())
    )

    if use_anthropic:
        content, error = _call_anthropic(user_prompt)
    else:
        content, error = _call_ollama(user_prompt)

    if content:
        sql = _extract_sql(content)
        return sql, user_prompt, True
    else:
        return None, user_prompt + f"\n\n-- ERROR: {error}", False


# ══════════════════════════════════════════════
#  SQL extractor
# ══════════════════════════════════════════════

def _extract_sql(text: str) -> str | None:
    """
    Pull a clean SELECT/WITH statement out of raw LLM output.
    Handles Markdown fences, leading chatter, trailing explanations.
    """
    # Strip Markdown fences
    text = re.sub(r"```(sql)?\n?", "", text, flags=re.I)
    text = text.replace("```", "").strip()

    # Find the first SELECT or WITH keyword
    m = re.search(r"(SELECT|WITH)\b[\s\S]*", text, flags=re.I)
    if m:
        sql = m.group(0).strip()
        # Trim anything after the first semicolon
        if ";" in sql:
            sql = sql[:sql.index(";") + 1]
        elif not sql.endswith(";"):
            sql += ";"
        return sql

    return None