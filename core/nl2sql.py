# ─────────────────────────────────────────────
# core/nl2sql.py  – Minimal NL→SQL pipeline with local Ollama only
#
# Provides `generate_sql_from_prompt(conn, table_names, prompt, redact=True, max_columns=50, max_prompt_length=2000)`
# which builds a privacy-safe schema description and calls a local Ollama model.
#
# The function always returns a tuple: (sql: str|None, llm_prompt: str, used_llm: bool)
# - `sql` will be a generated SQL string when `used_llm` is True and Ollama
#   returned something; otherwise None.
# - `llm_prompt` is the exact prompt sent to Ollama.
# - `used_llm` indicates whether an Ollama call was attempted.
#
# Note: This module does NOT execute SQL. Calling code should validate and run
# only safe `SELECT`/`WITH` queries.
# ─────────────────────────────────────────────

import json
import os
import re
import urllib.error
import urllib.request
from typing import Tuple
from config import DDL_MAX_COLUMNS
from core import ddl_utils


def _normalize_table_names(table_names: str | list[str]) -> list[str]:
    """Ensure table_names is always a list."""
    if isinstance(table_names, str):
        return [table_names]
    return list(table_names)


def _build_prompt(ddl: str, question: str, table_names: list[str]) -> str:
    """Construct the prompt sent to the LLM."""
    scope = ", ".join(table_names)
    instructions = (
        "You are a SQL generation assistant. Given a DuckDB table schema in DDL-like\n"
        "format and a natural language question, generate a single valid SQL query\n"
        "that answers the question. Only produce the SQL query and no surrounding\n"
        "explanation. The SQL should be read-only (SELECT or WITH) and must not\n"
        "include any data-modifying statements (INSERT/UPDATE/DELETE/DROP).\n"
        "If multiple tables are available, you may use joins or unions across them\n"
        "when needed.\n"
    )

    prompt = (
        instructions
        + f"\n-- Tables in scope: {scope}\n"
        + "\n-- Schema (privacy-safe DDL):\n"
        + ddl
        + "\n\n-- Question:\n"
        + question.strip()
        + "\n\n-- SQL:"
    )
    return prompt


def _build_schema_context(conn, table_names: list[str], redact: bool, max_columns: int) -> str:
    """Build privacy-safe DDL for all tables."""
    sections: list[str] = []
    for table_name in table_names:
        ddl = ddl_utils.generate_privacy_safe_ddl(conn, table_name, redact=redact, max_columns=max_columns)
        sections.append(f"-- Table: {table_name}\n{ddl}")
    return "\n\n".join(sections)


def _call_ollama(prompt: str) -> tuple[str | None, str]:
    """
    Call local Ollama instance.
    
    Returns:
        (content, error_message) or (None, error_message) on failure
    """
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode("utf-8")

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
        return None, "Ollama returned empty response."
    except urllib.error.URLError as exc:
        return None, f"Ollama is not available at {host}: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"Ollama returned invalid JSON: {exc}"
    except Exception as exc:
        return None, f"Ollama call failed: {exc}"


def generate_sql_from_prompt(
    conn,
    table_name: str | list[str],
    prompt: str,
    redact: bool = True,
    max_columns: int = DDL_MAX_COLUMNS,
    max_prompt_length: int = 2000,
) -> Tuple[str | None, str, bool]:
    """
    Build a privacy-safe DDL and call local Ollama to generate SQL.

    Args:
        conn: DuckDB connection
        table_name: Single table name or list of table names
        prompt: User's natural language question
        redact: If True (default), hide real column names for privacy
        max_columns: Maximum columns per table to include in DDL
        max_prompt_length: Maximum length of prompt before rejection

    Returns:
        (sql_or_none, llm_prompt, used_ollama)
        - sql: Generated SQL string or None if call failed
        - llm_prompt: Exact prompt sent to Ollama (for debugging)
        - used_ollama: Boolean indicating if Ollama was called
    """
    # ── Step 1: Normalize table names to a list ────────────────
    table_names = _normalize_table_names(table_name)

    # ── Step 2: Build privacy-safe DDL for all tables ──────────
    ddl = _build_schema_context(conn, table_names, redact=redact, max_columns=max_columns)

    # ── Step 3: Construct full prompt ──────────────────────────
    llm_prompt = _build_prompt(ddl, prompt, table_names)
    if len(llm_prompt) > max_prompt_length:
        error_msg = f"Prompt exceeds {max_prompt_length} characters. Please shorten your question."
        return None, llm_prompt + f"\n\n-- ERROR: {error_msg}", False

    # ── Step 4: Call local Ollama ─────────────────────────────
    content, error = _call_ollama(llm_prompt)
    
    if content:
        sql = _extract_sql(content)
        return sql, llm_prompt, True
    else:
        # Ollama failed — return error message in prompt
        return None, llm_prompt + f"\n\n-- ERROR: {error}", False


def _extract_sql(text: str) -> str | None:
    """
    Heuristically extract a single SQL statement from LLM output.
    
    Handles Markdown fences, backticks, and leading chatter.
    """
    # Remove Markdown fences
    text = re.sub(r"```(sql)?\n", "", text, flags=re.I)
    text = text.replace("```", "")
    
    # Strip leading non-SQL chatter
    m = re.search(r"(SELECT|WITH)\b[\s\S]*;?", text, flags=re.I)
    if m:
        sql = m.group(0).strip()
        # Ensure it ends with a semicolon
        if not sql.endswith(";"):
            sql += ";"
        return sql
    
    return None