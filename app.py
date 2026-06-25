import sys
from pathlib import Path
import socket

# ── FIX: Force Python to use IPv4 so 'localhost' doesn't crash on Windows ──
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo
# ───────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import uuid

from config import MAX_FILE_SIZE_MB, SESSION_TTL_MINUTES
from core.validator import validate_file, validate_sql_query
from core.session_mgr import (
    create_session,
    save_uploaded_file,
    cleanup_old_sessions,
    delete_session,
    delete_file_from_session,
    start_cleanup_daemon,
)
from core.ingestor    import get_or_create_connection, load_file_into_duckdb, get_all_tables
from core.transformer import clean_and_profile
import os
import re
import logging
from config import PROMPT_MAX_LENGTH
from core.llm_router import route_query
from core.ddl_utils import generate_privacy_safe_ddl, generate_multi_table_ddl
from core.route_a import run as run_route_a
from core.route_c import run as run_route_c
from core.route_d import run as run_route_d
from core.sql_engine import generate_safe_sql
# --- DISABLED INSIGHT ENGINE ---
# from core.insight_engine import generate_natural_language_insight
from config import DDL_MAX_COLUMNS

# ── Logging: route decisions visible in terminal ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)


st.set_page_config(page_title="Data Chatbot", page_icon="📊", layout="wide")

cleanup_old_sessions()
start_cleanup_daemon()

# ── Session state defaults ────────────────────────────────────────────────────
if "session_id"      not in st.session_state: st.session_state.session_id      = None
if "session_dir"     not in st.session_state: st.session_state.session_dir     = None
if "duckdb_conn"     not in st.session_state: st.session_state.duckdb_conn     = None
if "loaded_tables"   not in st.session_state: st.session_state.loaded_tables   = []
if "quality_reports" not in st.session_state: st.session_state.quality_reports = {}
if "processed_files" not in st.session_state: st.session_state.processed_files = set()
if "file_table_map"  not in st.session_state: st.session_state.file_table_map  = {}
if "chat_history"    not in st.session_state: st.session_state.chat_history    = []
if "uploader_key"    not in st.session_state: st.session_state.uploader_key    = str(uuid.uuid4())


def _on_uploader_change():
    current_names = {f.name for f in (st.session_state.get(st.session_state.uploader_key) or [])}
    removed = st.session_state.processed_files - current_names
    for fname in removed:
        _remove_file(fname)


def _reset_session_state():
    st.session_state.session_id      = None
    st.session_state.session_dir     = None

    if st.session_state.duckdb_conn:
        try:
            st.session_state.duckdb_conn.close()
        except Exception:
            pass

    st.session_state.duckdb_conn     = None
    st.session_state.loaded_tables   = []
    st.session_state.quality_reports = {}
    st.session_state.processed_files = set()
    st.session_state.file_table_map  = {}
    st.session_state.chat_history    = []
    st.session_state.uploader_key    = str(uuid.uuid4())


def _remove_file(filename: str):
    tables_for_file = st.session_state.file_table_map.get(filename, [])
    delete_file_from_session(
        session_dir=st.session_state.session_dir,
        filename=filename,
        conn=st.session_state.duckdb_conn,
        table_names=tables_for_file,
    )
    st.session_state.processed_files.discard(filename)
    st.session_state.file_table_map.pop(filename, None)
    for t in tables_for_file:
        st.session_state.quality_reports.pop(t, None)

    if st.session_state.duckdb_conn is not None:
        st.session_state.loaded_tables = get_all_tables(st.session_state.duckdb_conn)
    else:
        st.session_state.loaded_tables = []


# ═══════════════════════════════════════════════
#  LLM-powered friendly error explainer
# ═══════════════════════════════════════════════

def _explain_error_friendly(user_prompt: str, error_msg: str, sql: str | None = None) -> dict:
    """
    Ask the local LLM to explain a SQL/query error in plain English and
    suggest how the user might rephrase their question.

    Returns:
        {
          "explanation": str,   # plain-English reason the query failed
          "suggestion":  str,   # what the user might have meant
          "rephrasing":  str,   # concrete example of a better question
        }
    or falls back to a static dict on any Ollama failure.
    """
    from config import OLLAMA_BASE_URL, INTENT_MODEL, OLLAMA_TIMEOUT
    import requests as _requests

    sql_context = f"\n\nGenerated SQL:\n{sql}" if sql else ""

    system_prompt = """\
You are a helpful data assistant. A user asked a question about their data,
but the system could not generate a valid database query to answer it.

Your job is to:
1. Explain in ONE plain-English sentence why the question could not be answered
   (e.g. column not found, ambiguous question, no matching data).
2. Suggest what the user might have actually meant (one short sentence).
3. Provide ONE concrete rephrased question the user could try instead.

Respond ONLY with a JSON object — no markdown, no backticks, no extra text:
{
  "explanation": "<one sentence: why it failed>",
  "suggestion":  "<one sentence: what you think they meant>",
  "rephrasing":  "<one example of a better question>"
}"""

    user_content = (
        f"User question: {user_prompt}\n"
        f"Error: {error_msg}"
        f"{sql_context}"
    )

    try:
        resp = _requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model":   INTENT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                "stream":  False,
                "options": {"temperature": 0.3, "num_predict": 256},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        if resp.status_code == 200:
            raw = resp.json()["message"]["content"].strip()
            # strip ```json fences if model adds them
            raw = re.sub(r"```json\s*", "", raw, flags=re.I)
            raw = re.sub(r"```", "", raw).strip()
            import json as _json
            parsed = _json.loads(raw)
            # validate expected keys exist
            if all(k in parsed for k in ("explanation", "suggestion", "rephrasing")):
                return parsed
    except Exception:
        pass  # fall through to generic message

    # ── Generic message when LLM is unavailable ──────────────────────────────
    return {
        "explanation": "I wasn't able to generate a valid query for your question.",
        "suggestion":  "Your question may reference a column or value not present in the loaded data.",
        "rephrasing":  "Try rephrasing using the exact column names shown in the schema panel.",
    }


def _stamp_friendly_error(msg: dict) -> None:
    """
    Call _explain_error_friendly once at processing time and store the result
    in msg["friendly_error"] so the renderer never needs to call it again.
    """
    if msg.get("error") and "friendly_error" not in msg:
        with st.spinner("Figuring out what went wrong…"):
            msg["friendly_error"] = _explain_error_friendly(
                msg.get("user_prompt", "your question"),
                msg["error"],
                msg.get("sql"),
            )



def _render_assistant_message(msg: dict):
    route_icons = {
        "visualization": "📊",
        "sql_answer":    "🔢",
        "metadata":      "🗂️",
        "statistical":   "📉",
        "reasoning":     "💡",
    }
    icon        = route_icons.get(msg.get("route"), "❓")
    conf_colour = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(msg.get("confidence"), "⚪")

    with st.expander(
        f"{icon} Route: **{str(msg.get('route')).upper()}** "
        f"{conf_colour} {msg.get('confidence')}  ·  stage={msg.get('stage')}",
        expanded=False,
    ):
        if msg.get("router_parsed"):
            st.json(msg["router_parsed"])

    if msg.get("warning"):
        st.warning(msg["warning"])

    if msg.get("error"):
        # ── Friendly error response — use pre-computed result from processing ─
        err_raw  = msg["error"]
        sql_ctx  = msg.get("sql")
        friendly = msg.get("friendly_error", {})

        st.warning(
            f"**I couldn't answer that question.**\n\n"
            f"🔍 **What happened:** {friendly.get('explanation', err_raw)}\n\n"
            f"💡 **Are you referring to:** {friendly.get('suggestion', '')}\n\n"
            f"✏️ **Please rephrase your question — for example:**\n> {friendly.get('rephrasing', '')}"
        )

        with st.expander("🛠️ Technical details", expanded=False):
            st.caption(f"**Error:** {err_raw}")
            if sql_ctx:
                st.code(sql_ctx, language="sql")
            if msg.get("val_log"):
                for line in msg["val_log"]:
                    st.text(line)
        return

    if msg["route"] == "metadata":
        # ── Schema queries: DDL + DESCRIBE per table ──────────────────────────
        if msg.get("tables"):
            st.subheader("🗂️ Schema Information")
            for t_data in msg["tables"]:
                with st.expander(f"Table: `{t_data['name']}`", expanded=True):
                    st.code(t_data["ddl"], language="sql")
                    if t_data.get("info_df") is not None:
                        st.dataframe(t_data["info_df"], width="stretch")
        # ── Keyword-match queries: plain answer + optional result table ────────
        if msg.get("answer"):
            st.markdown(msg["answer"])
        if msg.get("df") is not None:
            st.dataframe(msg["df"], use_container_width=True)
            st.download_button(
                "⬇️ Download CSV",
                data=msg["df"].to_csv(index=False).encode("utf-8"),
                file_name="metadata_result.csv",
                mime="text/csv",
                key=f"dl_meta_{msg['id']}",
            )

    elif msg["route"] in ("statistical", "reasoning"):
        # ── Route D: AI observation + SQL expander + result table ─────────────
        if msg.get("answer"):
            st.markdown(f"**📊 AI Observation:** {msg['answer']}")
            st.divider()
        if msg.get("sql"):
            with st.expander("🔎 Generated SQL", expanded=False):
                st.code(msg["sql"], language="sql")
        if msg.get("df") is not None:
            total_rows = len(msg["df"])
            if total_rows > 50:
                st.subheader(f"📋 Statistical Results (Showing first 50 of {total_rows} rows)")
                st.dataframe(msg["df"].head(50), use_container_width=True)
            else:
                st.subheader(f"📋 Statistical Results ({total_rows} rows)")
                st.dataframe(msg["df"], use_container_width=True)
            st.download_button(
                "⬇️ Download CSV",
                data=msg["df"].to_csv(index=False).encode("utf-8"),
                file_name="statistical_result.csv",
                mime="text/csv",
                key=f"dl_stat_{msg['id']}",
            )

    elif msg["route"] == "visualization":
        with st.expander("🔎 Generated SQL", expanded=False):
            st.code(msg["sql"], language="sql")

        st.subheader(f"📊 {msg['intent'].get('title', 'Visualization')}")
        st.plotly_chart(msg["fig"], width="stretch")

        col_csv, col_html = st.columns(2)
        with col_csv:
            st.download_button(
                "⬇️ Download CSV",
                data=msg["df"].to_csv(index=False).encode("utf-8"),
                file_name="chart_data.csv",
                mime="text/csv",
                key=f"dl_csv_{msg['id']}",
            )
        with col_html:
            st.download_button(
                "⬇️ Download HTML",
                data=msg["fig"].to_html(include_plotlyjs="cdn").encode("utf-8"),
                file_name="chart.html",
                mime="text/html",
                key=f"dl_html_{msg['id']}",
            )

        with st.expander("📋 Raw query result", expanded=False):
            st.dataframe(msg["df"].head(50), width="stretch")

    else:
        # ── Route B: sql_answer ───────────────────────────────────────────────
        if msg.get("sql"):
            st.subheader("🔎 Generated SQL")
            st.code(msg["sql"], language="sql")
        if msg.get("df") is not None:
            total_rows = len(msg["df"])
            if total_rows > 50:
                st.subheader(f"📋 Query result (Showing first 50 of {total_rows} total rows)")
                st.dataframe(msg["df"].head(50), use_container_width=True)
            else:
                st.subheader(f"📋 Query result ({total_rows} rows)")
                st.dataframe(msg["df"], use_container_width=True)
            st.download_button(
                "Download CSV",
                data=msg["df"].to_csv(index=False).encode("utf-8"),
                file_name="query_result.csv",
                mime="text/csv",
                key=f"dl_sql_{msg['id']}",
            )


# ═══════════════════════════════════════════════
#  SIDEBAR: File Management
# ═══════════════════════════════════════════════
with st.sidebar:
    st.title("📊 Data Chatbot")
    st.caption(f"Max size: {MAX_FILE_SIZE_MB} MB  |  Timeout: {SESSION_TTL_MINUTES} min")
    st.divider()

    st.subheader("1. Upload Data")
    uploaded_files = st.file_uploader(
        label="Drop CSV, XLSX, JSON, or TXT",
        type=["csv", "xlsx", "json", "txt"],
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
        on_change=_on_uploader_change,
        label_visibility="collapsed",
    )
    run_button = st.button(
        "▶  Load Files",
        type="primary",
        disabled=(not uploaded_files),
        use_container_width=True, 
    )

    if st.session_state.duckdb_conn and st.session_state.loaded_tables:
        st.divider()
        st.subheader("📁 Loaded Context")

        for fname in sorted(st.session_state.processed_files):
            file_tables = st.session_state.file_table_map.get(fname, [])
            st.markdown(f"**{fname}**")
            st.caption(f"Tables: `{', '.join(file_tables)}`")

        with st.expander("👀 Preview Data", expanded=False):
            for t in st.session_state.loaded_tables:
                st.markdown(f"**`{t}`**")
                st.dataframe(
                    st.session_state.duckdb_conn.execute(f"SELECT * FROM {t} LIMIT 5").df(),
                    width="stretch",
                )

        st.divider()
        if st.button("🗑️ Delete Session", type="secondary", use_container_width=True):
            if st.session_state.session_id:
                delete_session(
                    session_id=st.session_state.session_id,
                    conn=st.session_state.duckdb_conn,
                )
            _reset_session_state()
            st.rerun()


# ═══════════════════════════════════════════════
#  MAIN SCREEN: Logic & Chatbot
# ═══════════════════════════════════════════════
conn   = st.session_state.duckdb_conn
tables = st.session_state.loaded_tables

if run_button and uploaded_files:
    if st.session_state.session_id is None:
        session_id, session_dir = create_session()
        st.session_state.session_id  = session_id
        st.session_state.session_dir = session_dir

    conn = get_or_create_connection(st.session_state)

    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.processed_files:
            continue

        with st.status(f"Cleaning **{uploaded_file.name}**…", expanded=True) as status:
            ok, reason = validate_file(uploaded_file)
            if not ok:
                status.update(label=f"❌ {uploaded_file.name} rejected", state="error")
                st.error(reason)
                continue

            file_path = save_uploaded_file(st.session_state.session_dir, uploaded_file)
            try:
                existing = list(st.session_state.loaded_tables)
                new_tables, warnings = load_file_into_duckdb(file_path, conn, existing)
            except Exception as e:
                status.update(label=f"❌ Error loading {uploaded_file.name}", state="error")
                st.error(str(e))
                continue

            for table_name in new_tables:
                report = clean_and_profile(conn, table_name)
                st.session_state.quality_reports[table_name] = report

            st.session_state.processed_files.add(uploaded_file.name)
            st.session_state.loaded_tables = get_all_tables(conn)
            st.session_state.file_table_map[uploaded_file.name] = new_tables
            status.update(label=f"✅ {uploaded_file.name} ready!", state="complete", expanded=False)

    st.rerun()


# ── Render the Chat Interface ─────────────────────────────────────────────────
if not tables:
    st.title("Welcome to Chat-to-Data 🤖")
    st.info("👈 Please upload and load a data file from the sidebar to begin chatting.")
else:
    st.title("Chat-to-Data")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                _render_assistant_message(msg)

    if prompt := st.chat_input(f"Ask anything about {', '.join(tables)}..."):
        if len(prompt) > PROMPT_MAX_LENGTH:
            st.error(f"Prompt is too long (max {PROMPT_MAX_LENGTH} characters).")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):

                    ddl_for_router = (
                        generate_multi_table_ddl(
                            conn, tables, redact=False, max_columns=DDL_MAX_COLUMNS
                        )
                        if tables else ""
                    )

                    routing_result = route_query(
                        user_prompt=prompt,
                        ddl_schema=ddl_for_router,
                        table_name=", ".join(tables) if tables else "",
                        sample_rows=None,
                        print_to_terminal=True,
                    )

                    parsed_route = routing_result.get("parsed") or {}
                    route_label  = parsed_route.get("route")
                    confidence   = parsed_route.get("confidence", "HIGH")
                    route_stage  = routing_result.get("stage", "?")

                    msg_id = str(uuid.uuid4())
                    assistant_msg = {
                        "role":          "assistant",
                        "id":            msg_id,
                        "route":         route_label,
                        "confidence":    confidence,
                        "stage":         route_stage,
                        "router_parsed": parsed_route,
                        "user_prompt":   prompt,
                    }

                    if not routing_result.get("success"):
                        assistant_msg["error"] = (
                            f"Query routing failed: {routing_result.get('error')}"
                        )

                    elif not route_label:
                        assistant_msg["error"] = (
                            "The system could not determine how to handle your question. "
                            "Please try rephrasing it."
                        )

                    elif route_label == "reasoning":
                        route_d_result = run_route_d(
                            conn=conn, tables=tables, prompt=prompt,
                            route_label=route_label,
                        )
                        if not route_d_result.success:
                            assistant_msg["error"] = route_d_result.error or "Route D failed."
                            assistant_msg["sql"]   = route_d_result.sql
                        else:
                            assistant_msg.update({
                                "answer": route_d_result.answer,
                                "sql":    route_d_result.sql,
                                "df":     route_d_result.dataframe,
                            })

                    elif route_label == "metadata":
                        route_c_result = run_route_c(
                            conn=conn, tables=tables, prompt=prompt
                        )
                        if not route_c_result.success:
                            assistant_msg["error"] = route_c_result.error or "Route C failed."
                        else:
                            assistant_msg["answer"] = route_c_result.answer
                            if route_c_result.tables:
                                assistant_msg["tables"] = route_c_result.tables
                            if route_c_result.dataframe is not None:
                                assistant_msg["df"] = route_c_result.dataframe

                    elif route_label == "visualization":
                        route_a_result = run_route_a(
                            conn=conn, tables=tables, prompt=prompt,
                            router_intent=parsed_route,
                        )
                        if not route_a_result.success:
                            assistant_msg["error"] = (
                                f"Visualization failed at stage "
                                f"{route_a_result.stage_reached}: {route_a_result.error}"
                            )
                            assistant_msg["val_log"] = route_a_result.validation_log
                            assistant_msg["sql"]     = route_a_result.sql
                        else:
                            assistant_msg.update({
                                "intent":  route_a_result.intent,
                                "sql":     route_a_result.sql,
                                "val_log": route_a_result.validation_log,
                                "df":      route_a_result.df,
                                "fig":     route_a_result.fig,
                            })

                    elif route_label == "statistical":
                        route_d_result = run_route_d(
                            conn=conn, tables=tables, prompt=prompt,
                            route_label=route_label,
                        )
                        if not route_d_result.success:
                            assistant_msg["error"] = route_d_result.error or "Route D failed."
                            assistant_msg["sql"]   = route_d_result.sql
                        else:
                            assistant_msg.update({
                                "answer": route_d_result.answer,
                                "sql":    route_d_result.sql,
                                "df":     route_d_result.dataframe,
                            })

                    else:  # sql_answer
                        sql_result = generate_safe_sql(
                            prompt=prompt,
                            ddl_schema=ddl_for_router,
                            table_name=", ".join(tables) if tables else "",
                            db_session=conn,
                        )
                        if not sql_result["success"]:
                            assistant_msg["error"] = (
                                f"LLM failed to generate valid SQL: {sql_result['error']}"
                            )
                            assistant_msg["sql"] = sql_result.get("sql")
                        else:
                            sql = sql_result["sql"]
                            ok, reason = validate_sql_query(conn, sql, tables)
                            if not ok:
                                assistant_msg["error"] = reason
                                assistant_msg["sql"]   = sql
                            else:
                                try:
                                    df = conn.execute(sql).df()
                                    assistant_msg.update({"sql": sql, "df": df})
                                except Exception as e:
                                    assistant_msg["error"] = f"Execution error: {e}"
                                    assistant_msg["sql"]   = sql

                    # Compute friendly error once while spinner is still showing
                    _stamp_friendly_error(assistant_msg)

                # Spinner is now closed — render is visible to the user
                _render_assistant_message(assistant_msg)

            st.session_state.chat_history.append(assistant_msg)
            st.rerun()