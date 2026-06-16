# sql_engine.py
import logging
import requests
from config import OLLAMA_BASE_URL, SQL_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

def generate_safe_sql(prompt: str, ddl_schema: str, table_name: str, db_session=None) -> dict:
    """
    Generates a clean, executable DuckDB SQL query based on the user prompt and DDL schema.
    """
    
    # ── SYSTEM PROMPT: This is what qwen2.5-coder pays maximum attention to ──
    system_prompt = (
        "You are an expert DuckDB SQL developer.\n"
        "Your task is to write a single, optimized, valid DuckDB SQL query that answers the user's request.\n\n"
        "CRITICAL RULES:\n"
        "- Respond with ONLY the raw SQL code. Do NOT wrap it in markdown code blocks like ```sql.\n"
        "- Do NOT add any explanations, introductory text, or trailing comments.\n"
        "- Ensure all table names and column names match the DDL schema exactly.\n\n"
        "DATABASE SPECIFIC RULES (DUCKDB):\n"
        "- The column 'category' contains multiple sub-categories separated by a pipe symbol ('|') (e.g., 'Electronics|Computers|Mice').\n"
        "- If the user asks for 'unique categories', 'distinct categories', 'list categories', or a 'count of categories', you MUST split this column up.\n"
        "- Use this exact pattern to expand and split them: UNNEST(STRING_SPLIT(category, '|'))\n"
        "- Example for unique categories: SELECT DISTINCT UNNEST(STRING_SPLIT(category, '|')) AS unique_categories FROM amazon"
    )

    user_payload = (
        f"### TABLE SCHEMA:\n{ddl_schema}\n\n"
        f"### USER REQUEST:\n{prompt}\n\n"
        f"Provide the raw DuckDB SQL query below:"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": SQL_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload}
                ],
                "stream": False,
                "options": {"temperature": 0.0} # Absolute precision, no guessing
            },
            timeout=OLLAMA_TIMEOUT
        )
        resp.raise_for_status()
        sql_query = resp.json()["message"]["content"].strip()
        
        # Clean up any accidental markdown code blocks if the LLM slips up
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        return {"success": True, "sql": sql_query}

    except Exception as e:
        logger.error("SQL generation failed: %s", e)
        return {"success": False, "error": str(e)}