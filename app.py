import os
# Force the base theme to light mode
os.environ["STREAMLIT_THEME_BASE"] = "light"
# Optional: Hardcode exact background shades if you want pure white
os.environ["STREAMLIT_THEME_BACKGROUND_COLOR"] = "#FFFFFF"
os.environ["STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR"] = "#F8F9FA"

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
import json
import re
import logging

from config import MAX_FILE_SIZE_MB, SESSION_TTL_MINUTES, PROMPT_MAX_LENGTH, DDL_MAX_COLUMNS
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
from core.llm_router  import route_query
from core.ddl_utils   import generate_privacy_safe_ddl, generate_multi_table_ddl
from core.route_a     import run as run_route_a
from core.route_c     import run as run_route_c
from core.route_d     import run as run_route_d
from core.sql_engine  import generate_safe_sql, _humanize_error

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
if "upload_errors"   not in st.session_state: st.session_state.upload_errors   = []


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
#  Error handling helpers
# ═══════════════════════════════════════════════

def _to_friendly_dict(plain_english_msg: str, prompt: str = "") -> dict:
    """
    Wrap a plain-English error into the friendly_error dict.

    Generates contextual suggestion and rephrasing text by pattern-matching
    the already-humanized message — no extra LLM call needed.
    This is intentionally cheap: the humanization already happened in
    _humanize_error(). This function just formats the display fields.
    """
    msg_lower = plain_english_msg.lower()

    # ── Column not found ──────────────────────────────────────────────────────
    if "column" in msg_lower and any(
        w in msg_lower for w in ("doesn't have", "don't have", "does not exist", "not found")
    ):
        col_match = re.search(r'"([^"]+)"', plain_english_msg)
        bad_col   = col_match.group(1) if col_match else "that column"
        return {
            "explanation": plain_english_msg,
            "suggestion": (
                f'"{bad_col}" is not a column in your dataset — '
                f"the column name may be different from what you mentioned."
            ),
            "rephrasing": (
                "Open the schema panel → find the correct column name → "
                f"then try: \"{prompt.replace(bad_col, '<correct column name>')}\"."
                if prompt and col_match else
                "Open the schema panel on the left and use one of the exact "
                "column names shown there in your question."
            ),
        }

    # ── Math on text column ───────────────────────────────────────────────────
    if "text instead of numbers" in msg_lower or (
        "calculation" in msg_lower and "text" in msg_lower
    ):
        return {
            "explanation": plain_english_msg,
            "suggestion": (
                "You may be trying to calculate something (total, average, sum) "
                "on a column that stores names or categories, not numbers."
            ),
            "rephrasing": (
                "Check the schema panel for columns with a BIGINT or DOUBLE type — "
                "those are the numeric ones. Then ask: "
                "\"What is the total [numeric column] grouped by [category column]?\""
            ),
        }

    # ── Session / table not found ─────────────────────────────────────────────
    if "session" in msg_lower and "expired" in msg_lower:
        return {
            "explanation": plain_english_msg,
            "suggestion":  "Your session may have timed out while you were away.",
            "rephrasing":  "Re-upload your file using the sidebar, then ask again.",
        }

    # ── Ambiguous column ──────────────────────────────────────────────────────
    if "more than one table" in msg_lower or "appears in" in msg_lower:
        return {
            "explanation": plain_english_msg,
            "suggestion":  "Try naming the table in your question to be more specific.",
            "rephrasing": (
                f"Try: \"From the [table name], {prompt.lower()}\""
                if prompt else
                "Specify which table you mean alongside the column name."
            ),
        }

    # ── Generic / unknown ─────────────────────────────────────────────────────
    return {
        "explanation": plain_english_msg,
        "suggestion": (
            "Try rephrasing your question using the exact column names "
            "shown in the schema panel."
        ),
        "rephrasing": (
            f"Try asking: \"{prompt}\" but replace any column names with "
            "the exact names shown in the schema panel."
            if prompt else
            "Check the schema panel for the available column names and "
            "rephrase your question around those."
        ),
    }


def _explain_error_with_llm(
    user_prompt: str,
    error_msg: str,
    sql: str | None = None,
    column_context: str = "",
) -> dict:
    """
    Ask llama3.2:3b (INTENT_MODEL / ROUTER_MODEL — already warm from routing)
    to explain a technical error in plain English and suggest a rephrasing.

    Used ONLY for errors that don't already go through _humanize_error:
    - Route A (visualization pipeline) errors
    - validate_sql_query rejections
    - DuckDB execution errors in the sql_answer path

    Routes B and D already produce humanized errors via _humanize_error —
    those are wrapped with _to_friendly_dict instead to avoid double LLM calls.
    """
    from config import OLLAMA_BASE_URL, INTENT_MODEL, OLLAMA_TIMEOUT
    import requests as _requests

    sql_context    = f"\n\nGenerated SQL:\n{sql}" if sql else ""
    column_section = f"\n\nDataset columns and types:\n{column_context}" if column_context else ""

    system_prompt = """\
You are a helpful data assistant. A user asked a question about their data,
but the system could not generate a valid database query to answer it.

Your job is to:
1. Explain in ONE plain-English sentence why the question could not be answered.
2. Suggest what the user might have actually meant (one short sentence).
3. Provide ONE concrete rephrased question the user could try instead.

NAME THE SPECIFIC PROBLEM, DON'T GENERALIZE:
- If the error or the dataset's column list shows the user tried to do MATH
  (sum, average, total, divide, multiply) on a column that holds TEXT data
  (names, categories, IDs, descriptions, statuses) — say so explicitly.
  Example: "You can't sum up employee names because that column holds text,
  not numbers."
- If a column name in the error doesn't exist, and a similarly-named real
  column appears in the dataset columns list, name that real column in your
  suggestion instead of speaking generically about "a missing column."
- Only name a specific cause when the error message or column list actually
  supports it — don't guess wildly. A generic explanation is fine when no
  specific cause is inferable, but always prefer specific over generic.

Respond ONLY with a JSON object — no markdown, no backticks, no extra text:
{
  "explanation": "<one sentence: why it failed, specific if inferable>",
  "suggestion":  "<one sentence: what you think they meant>",
  "rephrasing":  "<one example of a better question>"
}"""

    user_content = (
        f"User question: {user_prompt}\n"
        f"Error: {error_msg}"
        f"{sql_context}"
        f"{column_section}"
    )

    try:
        resp = _requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model":    INTENT_MODEL,
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
            raw = re.sub(r"```json\s*", "", raw, flags=re.I)
            raw = re.sub(r"```", "", raw).strip()
            parsed = json.loads(raw)
            if all(k in parsed for k in ("explanation", "suggestion", "rephrasing")):
                return parsed
    except Exception:
        pass

    return {
        "explanation": "I wasn't able to generate a valid query for your question.",
        "suggestion":  "Your question may reference a column or value not present in the loaded data.",
        "rephrasing":  "Try rephrasing using the exact column names shown in the schema panel.",
    }


def _stamp_friendly_error(msg: dict, ddl_schema: str = "") -> None:
    """
    Attach a friendly_error dict to msg if it has an error and doesn't
    already have one. Chooses the cheapest path:

    - Routes B and D: errors are already plain English from _humanize_error.
      Wrap with _to_friendly_dict — no extra LLM call needed.
    - Route A and validator/execution errors in sql_answer: use the full
      _explain_error_with_llm to get a richer explanation with rephrasing.
    - Route C: errors are plain-English strings written directly in route_c.py.
      Wrap with _to_friendly_dict — no extra LLM call needed.
    - Router failures: wrap with _to_friendly_dict — the message is already
      readable (e.g. "Cannot connect to Ollama").
    """
    if not msg.get("error") or "friendly_error" in msg:
        return

    route  = msg.get("route", "")
    prompt = msg.get("user_prompt", "")
    error  = msg["error"]

    # Routes whose errors are already humanized — just wrap, no LLM call
    if route in ("sql_answer", "statistical", "metadata") or msg.get("_error_pre_humanized"):
        msg["friendly_error"] = _to_friendly_dict(error, prompt)
        return

    # Route A and any other route — use the full LLM explainer
    with st.spinner("Figuring out what went wrong…"):
        msg["friendly_error"] = _explain_error_with_llm(
            user_prompt=prompt,
            error_msg=error,
            sql=msg.get("sql"),
            column_context=ddl_schema,
        )


# ═══════════════════════════════════════════════
#  Chat Message Renderer
# ═══════════════════════════════════════════════

def _render_assistant_message(msg: dict):
    route_icons = {
        "visualization": "📊",
        "sql_answer":    "🔢",
        "metadata":      "🗂️",
        "statistical":   "📉",
    }
    route       = msg.get("route") or "sql_answer"
    icon        = route_icons.get(route, "❓")
    conf_colour = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(msg.get("confidence"), "⚪")

    with st.expander(
        f"{icon} Route: **{str(route).upper()}** "
        f"{conf_colour} {msg.get('confidence')}  ·  stage={msg.get('stage')}",
        expanded=False,
    ):
        if msg.get("router_parsed"):
            st.json(msg["router_parsed"])

    if msg.get("warning"):
        st.warning(msg["warning"])

    # ── Error rendering ───────────────────────────────────────────────────────
    if msg.get("error"):
        friendly = msg.get("friendly_error", {})
        err_raw  = msg["error"]

        st.warning(
            f"**I couldn't answer that question.**\n\n"
            f"🔍 **What happened:** {friendly.get('explanation', err_raw)}\n\n"
            f"💡 **Are you referring to:** {friendly.get('suggestion', '')}\n\n"
            f"✏️ **Try rephrasing — for example:**\n> {friendly.get('rephrasing', '')}"
        )
        with st.expander("🛠️ Technical details", expanded=False):
            st.caption(f"**Error:** {err_raw}")
            if msg.get("sql"):
                st.code(msg["sql"], language="sql")
            if msg.get("val_log"):
                for line in msg["val_log"]:
                    st.text(line)
        return

    # ── Success rendering by route ────────────────────────────────────────────
    if route == "metadata":
        if msg.get("tables"):
            st.subheader("🗂️ Schema Information")
            for t_data in msg["tables"]:
                with st.expander(f"Table: `{t_data['name']}`", expanded=True):
                    st.code(t_data["ddl"], language="sql")
                    if t_data.get("info_df") is not None:
                        st.dataframe(t_data["info_df"], use_container_width=True)
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

    elif route == "statistical":
        if msg.get("answer"):
            st.markdown(f"**📊 AI Observation:** {msg['answer']}")
            st.divider()
        if msg.get("sql"):
            with st.expander("🔎 Generated SQL", expanded=False):
                st.code(msg["sql"], language="sql")
        if msg.get("df") is not None:
            total_rows = len(msg["df"])
            shown      = min(total_rows, 50)
            st.subheader(
                f"📋 Statistical Results ({shown} of {total_rows} rows)"
                if total_rows > 50 else
                f"📋 Statistical Results ({total_rows} rows)"
            )
            st.dataframe(msg["df"].head(50), use_container_width=True)
            st.download_button(
                "⬇️ Download CSV",
                data=msg["df"].to_csv(index=False).encode("utf-8"),
                file_name="statistical_result.csv",
                mime="text/csv",
                key=f"dl_stat_{msg['id']}",
            )

    elif route == "visualization":
        with st.expander("🔎 Generated SQL", expanded=False):
            st.code(msg["sql"], language="sql")
        st.subheader(f"📊 {msg['intent'].get('title', 'Visualization')}")
        st.plotly_chart(msg["fig"], use_container_width=True)
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
            st.dataframe(msg["df"].head(50), use_container_width=True)

    else:  # sql_answer
        if msg.get("sql"):
            # Collapsed by default — click to reveal, same pattern as the
            # Route badge expander and every other route's SQL display.
            with st.expander("🔎 Generated SQL", expanded=False):
                st.code(msg["sql"], language="sql")
        if msg.get("df") is not None:
            total_rows = len(msg["df"])
            shown      = min(total_rows, 50)
            st.subheader(
                f"📋 Query result (showing {shown} of {total_rows} rows)"
                if total_rows > 50 else
                f"📋 Query result ({total_rows} rows)"
            )
            st.dataframe(msg["df"].head(50), use_container_width=True)
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
                    use_container_width=True,
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
    st.session_state.upload_errors = []  # clear stale errors from any previous run

    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.processed_files:
            continue

        with st.status(f"Cleaning **{uploaded_file.name}**…", expanded=True) as status:
            ok, reason = validate_file(uploaded_file)
            if not ok:
                status.update(label=f"❌ {uploaded_file.name} rejected", state="error")
                # NOTE: st.error() here is wiped out immediately by the
                # st.rerun() below (buttons reset on rerun, so this whole
                # block never runs again to re-draw it). Persist the
                # message in session_state so it survives the rerun and
                # actually gets shown to the user — see rendering block
                # right before "Render the Chat Interface" further down.
                st.session_state.upload_errors.append((uploaded_file.name, reason))
                continue

            file_path = save_uploaded_file(st.session_state.session_dir, uploaded_file)
            try:
                existing   = list(st.session_state.loaded_tables)
                new_tables, warnings = load_file_into_duckdb(file_path, conn, existing)
            except Exception as e:
                status.update(label=f"❌ Error loading {uploaded_file.name}", state="error")
                st.session_state.upload_errors.append((uploaded_file.name, str(e)))
                continue

            for table_name in new_tables:
                report = clean_and_profile(conn, table_name)
                st.session_state.quality_reports[table_name] = report

            st.session_state.processed_files.add(uploaded_file.name)
            st.session_state.loaded_tables = get_all_tables(conn)
            st.session_state.file_table_map[uploaded_file.name] = new_tables
            status.update(label=f"✅ {uploaded_file.name} ready!", state="complete", expanded=False)

    st.rerun()


# ── Show any upload/validation errors from the last run ───────────────────────
# These were stashed in session_state because st.rerun() (called right after
# processing uploads) wipes out any st.error()/st.warning() that was drawn
# during the run that triggered it — the code path that drew them doesn't
# execute again on the rerun, so the message never reaches the user.
if st.session_state.upload_errors:
    for fname, reason in st.session_state.upload_errors:
        st.error(f"**{fname}**: {reason}")
    st.session_state.upload_errors = []

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
                    route_label  = parsed_route.get("route") or "sql_answer"
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

                    # ── Router-level failures ─────────────────────────────────
                    if not routing_result.get("success"):
                        technical = f"Query routing failed: {routing_result.get('error')}"
                        assistant_msg["error"] = _humanize_error(
                            technical, prompt=prompt,
                            context="routing your question to the right handler",
                        )
                        assistant_msg["_error_pre_humanized"] = True

                    elif confidence == "LOW":
                        assistant_msg["warning"] = (
                            "⚠️ The system wasn't fully confident about how to interpret "
                            "your question. If the answer looks wrong, try rephrasing it."
                        )

                    # ── Route: metadata → Route C ─────────────────────────────
                    if not assistant_msg.get("error") and route_label == "metadata":
                        route_c_result = run_route_c(conn=conn, tables=tables, prompt=prompt)
                        if not route_c_result.success:
                            assistant_msg["error"] = route_c_result.error or "Could not retrieve metadata."
                            assistant_msg["_error_pre_humanized"] = True
                        else:
                            assistant_msg["answer"] = route_c_result.answer
                            if route_c_result.tables:
                                assistant_msg["tables"] = route_c_result.tables
                            if route_c_result.dataframe is not None:
                                assistant_msg["df"] = route_c_result.dataframe

                    # ── Route: visualization → Route A ────────────────────────
                    elif not assistant_msg.get("error") and route_label == "visualization":
                        route_a_result = run_route_a(
                            conn=conn, tables=tables, prompt=prompt,
                            router_intent=parsed_route,
                        )
                        if not route_a_result.success:
                            assistant_msg["error"]   = route_a_result.error or "Visualization failed."
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

                    # ── Route: statistical → Route D ──────────────────────────
                    elif not assistant_msg.get("error") and route_label == "statistical":
                        route_d_result = run_route_d(
                            conn=conn, tables=tables, prompt=prompt,
                            route_label=route_label,
                        )
                        if not route_d_result.success:
                            assistant_msg["error"] = route_d_result.error or "Statistical analysis failed."
                            assistant_msg["sql"]   = route_d_result.sql
                            assistant_msg["_error_pre_humanized"] = True
                        else:
                            assistant_msg.update({
                                "answer": route_d_result.answer,
                                "sql":    route_d_result.sql,
                                "df":     route_d_result.dataframe,
                            })

                    # ── Route: sql_answer → Route B ───────────────────────────
                    elif not assistant_msg.get("error"):
                        sql_result = generate_safe_sql(
                            prompt=prompt,
                            ddl_schema=ddl_for_router,
                            table_name=", ".join(tables) if tables else "",
                            db_session=conn,
                        )
                        if not sql_result["success"]:
                            assistant_msg["error"] = sql_result["error"]
                            assistant_msg["sql"]   = sql_result.get("sql")
                            assistant_msg["_error_pre_humanized"] = True
                        else:
                            sql = sql_result["sql"]
                            ok, reason = validate_sql_query(conn, sql, tables)
                            if not ok:
                                assistant_msg["error"] = _humanize_error(
                                    reason, prompt=prompt,
                                    context="validating the generated query before running it",
                                )
                                assistant_msg["sql"]   = sql
                                assistant_msg["_error_pre_humanized"] = True
                            else:
                                try:
                                    df = conn.execute(sql).df()
                                    assistant_msg.update({"sql": sql, "df": df})
                                except Exception as e:
                                    technical = str(e)
                                    assistant_msg["error"] = _humanize_error(
                                        technical, prompt=prompt,
                                        context="running the query against your data",
                                    )
                                    assistant_msg["sql"] = sql
                                    assistant_msg["_error_pre_humanized"] = True

                    # ── Attach friendly_error for any remaining error ──────────
                    _stamp_friendly_error(assistant_msg, ddl_schema=ddl_for_router)

                _render_assistant_message(assistant_msg)

            st.session_state.chat_history.append(assistant_msg)
            st.rerun()
