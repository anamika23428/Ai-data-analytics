# ─────────────────────────────────────────────
# core/nl2sql.py  – Minimal NL→SQL pipeline with optional LLM integration
#
# Provides `generate_sql_from_prompt(conn, table_names, prompt, redact=True, max_columns=50, max_prompt_length=2000, llm_provider="ollama", openai_api_key=None)`
# which will build a privacy-safe schema description and prefer a local Ollama
# model by default. An OpenAI fallback is available only when explicitly
# requested.
#
# The function always returns a tuple: (sql: str|None, llm_prompt: str, used_llm: bool)
# - `sql` will be a generated SQL string when `used_llm` is True and the LLM
#   returned something; otherwise None.
# - `llm_prompt` is the exact prompt sent (or that should be sent) to the LLM.
# - `used_llm` indicates whether an LLM call was attempted.
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
    if isinstance(table_names, str):
        return [table_names]
    return list(table_names)


def _build_prompt(ddl: str, question: str, table_names: list[str]) -> str:
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
    sections: list[str] = []
    for table_name in table_names:
        ddl = ddl_utils.generate_privacy_safe_ddl(conn, table_name, redact=redact, max_columns=max_columns)
        sections.append(f"-- Table: {table_name}\n{ddl}")
    return "\n\n".join(sections)


def _call_ollama(prompt: str) -> tuple[str | None, str | None]:
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
        return content or None, None
    except urllib.error.URLError as exc:
        return None, f"Ollama is not available at {host}: {exc}"
    except Exception as exc:
        return None, f"Ollama call failed: {exc}"


def generate_sql_from_prompt(
    conn,
    table_name: str | list[str],
    prompt: str,
    redact: bool = True,
    max_columns: int = DDL_MAX_COLUMNS,
    max_prompt_length: int = 2000,
    llm_provider: str = "ollama",
    openai_api_key: str | None = None,
) -> Tuple[str | None, str, bool]:
    """
    Build a privacy-safe DDL and either call a local Ollama model or, if
    explicitly requested, an OpenAI ChatCompletion to get SQL.

    Returns (sql_or_none, llm_prompt, used_llm).
    """
    # 1) Build DDL
    table_names = _normalize_table_names(table_name)
    ddl = _build_schema_context(conn, table_names, redact=redact, max_columns=max_columns)

    # 2) Create LLM prompt and validate prompt length
    llm_prompt = _build_prompt(ddl, prompt, table_names)
    if len(llm_prompt) > max_prompt_length:
        # The prompt is too long to safely send — caller should shorten the question
        return None, llm_prompt + f"\n\n-- ERROR: prompt exceeds {max_prompt_length} characters.", False

    provider = (llm_provider or "ollama").strip().lower()

    # 3) Prefer local Ollama by default
    if provider in {"ollama", "local", "local-ollama", "auto"}:
        content, error = _call_ollama(llm_prompt)
        if content:
            sql = _extract_sql(content)
            return sql, llm_prompt, True
        if provider != "auto":
            return None, llm_prompt + f"\n\n-- ERROR: {error}", False

    # 4) Explicit OpenAI fallback only when requested
    if provider not in {"openai", "auto"}:
        return None, llm_prompt, False

    if not openai_api_key:
        return None, llm_prompt + "\n\n-- ERROR: OPENAI_API_KEY is not configured.", False

    try:
        import openai  # type: ignore[import-not-found]
    except Exception:
        return None, llm_prompt + "\n\n-- ERROR: openai package not installed.", False

    openai.api_key = openai_api_key

    try:
        # Use ChatCompletion only when explicitly requested.
        resp = openai.ChatCompletion.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": llm_prompt}],
            max_tokens=1024,
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        # LLMs sometimes wrap SQL with backticks or fences — try to extract the SQL
        sql = _extract_sql(content)
        return sql, llm_prompt, True
    except Exception as e:
        return None, llm_prompt + f"\n\n-- ERROR: LLM call failed: {e}", True


def _extract_sql(text: str) -> str | None:
    """
    Heuristically extract a single SQL statement from LLM output.
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
