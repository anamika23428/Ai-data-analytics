# ─────────────────────────────────────────────
# core/nl2sql.py  – Minimal NL→SQL pipeline with optional LLM integration
#
# Provides `generate_sql_from_prompt(conn, table_name, prompt, redact=False, max_columns=50, max_prompt_length=2000, openai_api_key=None)`
# which will build a privacy-safe schema description and either call an OpenAI
# ChatCompletion (if `openai_api_key` is provided and the `openai` package is
# installed) or return the constructed prompt so calling code can invoke an LLM.
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

import os
import re
from typing import Tuple
from config import DDL_MAX_COLUMNS
from core import ddl_utils


def _build_prompt(ddl: str, question: str) -> str:
    instructions = (
        "You are a SQL generation assistant. Given a DuckDB table schema in DDL-like\n"
        "format and a natural language question, generate a single valid SQL query\n"
        "that answers the question. Only produce the SQL query and no surrounding\n"
        "explanation. The SQL should be read-only (SELECT or WITH) and must not\n"
        "include any data-modifying statements (INSERT/UPDATE/DELETE/DROP).\n"
    )

    prompt = (
        instructions
        + "\n-- Schema (privacy-safe DDL):\n"
        + ddl
        + "\n\n-- Question:\n"
        + question.strip()
        + "\n\n-- SQL:" 
    )
    return prompt


def generate_sql_from_prompt(
    conn,
    table_name: str,
    prompt: str,
    redact: bool = False,
    max_columns: int = DDL_MAX_COLUMNS,
    max_prompt_length: int = 2000,
    openai_api_key: str | None = None,
) -> Tuple[str | None, str, bool]:
    """
    Build a privacy-safe DDL and either call an OpenAI ChatCompletion to get SQL,
    or return the constructed prompt so the caller can send it to an LLM.

    Returns (sql_or_none, llm_prompt, used_llm).
    """
    # 1) Build DDL
    ddl = ddl_utils.generate_privacy_safe_ddl(conn, table_name, redact=redact, max_columns=max_columns)

    # 2) Create LLM prompt and validate prompt length
    llm_prompt = _build_prompt(ddl, prompt)
    if len(llm_prompt) > max_prompt_length:
        # The prompt is too long to safely send — caller should shorten the question
        return None, llm_prompt + f"\n\n-- ERROR: prompt exceeds {max_prompt_length} characters.", False

    # 3) If no API key provided, return prompt for caller use
    if not openai_api_key:
        return None, llm_prompt, False

    # 4) Attempt to call OpenAI if package is available
    try:
        import openai
    except Exception:
        return None, llm_prompt + "\n\n-- ERROR: openai package not installed.", False

    openai.api_key = openai_api_key

    try:
        # Use ChatCompletion (chat format) – prefer gpt-4o or gpt-4 if available to user
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
