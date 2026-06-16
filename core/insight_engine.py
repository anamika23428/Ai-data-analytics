# insight_engine.py
import logging
import requests
import pandas as pd
from config import OLLAMA_BASE_URL, INSIGHT_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

def generate_natural_language_insight(prompt: str, ddl_schema: str, sql: str, df: pd.DataFrame) -> str:
    """
    Passes the user query, full schema, generated SQL, and raw data output 
    to Llama 3.1 to generate a clean, clear, natural language summary.
    """
    total_rows = len(df)
    
    # INCREASED LIMIT: We bumped this to 500 so it can actually see "all" the categories
    # Single-column text is very lightweight, so 500 rows will not crash the LLM's memory.
    limit = 500 
    data_str = df.head(limit).to_csv(index=False)
    
    system_prompt = (
        "You are a strict data extraction analyst.\n"
        "Your ONLY job is to answer the user's question based strictly on the raw data provided.\n\n"
        "CRITICAL RULES - NEVER BREAK THESE:\n"
        "1. NEVER summarize, group, or categorize the data using outside knowledge. Do NOT map sub-categories to parent categories.\n"
        "2. If the user asks you to 'list', 'name', or 'show' items, you MUST output a raw bulleted list of the EXACT values provided in the CSV data.\n"
        "3. Do not write SQL, and do not explain your methodology.\n"
        "4. If there are more items than shown, state 'Here are the top [X] items:' followed by the exact list."
    )
    
    user_payload = (
        f"### 1. USER QUESTION:\n{prompt}\n\n"
        f"### 2. DATASET SCHEMA:\n{ddl_schema}\n\n"
        f"### 3. GENERATED SQL QUERY:\n{sql}\n\n"
        f"### 4. RAW DATABASE RESULT (Showing {min(total_rows, limit)} out of {total_rows} total items):\n{data_str}\n\n"
        f"Provide the final natural language answer now:"
    )
    
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": INSIGHT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload}
                ],
                "stream": False,
                "options": {"temperature": 0.0} # 0.0 forces the AI to be robotic and literal
            },
            timeout=OLLAMA_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
        
    except Exception as e:
        logger.error("Insight generation failed: %s", e)
        raise e