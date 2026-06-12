# ─────────────────────────────────────────────
#  app.py  –  Main Streamlit UI
#
#  Features:
#    1. Duplicate file upload → detected and skipped with warning
#    2. Re-clicking Analyse → stale quality reports are refreshed
#    3. ✕ button next to each file → removes file from disk + drops DuckDB tables
#    4. "🗑 Delete Session" button → wipes entire session and resets UI
#    5. Terminal logs for every session create / delete / expire event
#    6. NL→SQL prompt uses Anthropic Claude API (claude-sonnet-4-20250514)
# ─────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.express as px

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
from config import DDL_MAX_COLUMNS

# ── Logging: route decisions visible in terminal ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)


st.set_page_config(page_title="Analytics App", page_icon="📊", layout="wide")

cleanup_old_sessions()
start_cleanup_daemon()

# ── Session state defaults ────────────────────
if "session_id"         not in st.session_state: st.session_state.session_id         = None
if "session_dir"        not in st.session_state: st.session_state.session_dir        = None
if "duckdb_conn"        not in st.session_state: st.session_state.duckdb_conn        = None
if "loaded_tables"      not in st.session_state: st.session_state.loaded_tables      = []
if "quality_reports"    not in st.session_state: st.session_state.quality_reports    = {}
if "processed_files"    not in st.session_state: st.session_state.processed_files    = set()
# Maps filename → list[table_name] so we know which tables to drop on delete
if "file_table_map"     not in st.session_state: st.session_state.file_table_map     = {}


# ── Uploader change callback ──────────────────
# Called by Streamlit whenever the file uploader widget changes (including when
# the user clicks the native ✕ on a file).  We compare the widget's current
# file list against processed_files to find removed files and clean them up.
def _on_uploader_change():
    current_names = {f.name for f in (st.session_state.get("file_uploader_widget") or [])}
    removed = st.session_state.processed_files - current_names
    for fname in removed:
        _remove_file(fname)


# ═══════════════════════════════════════════════
#  Helper: fully reset Streamlit session state
# ═══════════════════════════════════════════════
def _reset_session_state():
    st.session_state.session_id      = None
    st.session_state.session_dir     = None
    st.session_state.duckdb_conn     = None
    st.session_state.loaded_tables   = []
    st.session_state.quality_reports = {}
    st.session_state.processed_files = set()
    st.session_state.file_table_map  = {}


# ═══════════════════════════════════════════════
#  Helper: remove one file + its DuckDB tables
# ═══════════════════════════════════════════════
def _remove_file(filename: str):
    tables_for_file = st.session_state.file_table_map.get(filename, [])

    delete_file_from_session(
        session_dir=st.session_state.session_dir,
        filename=filename,
        conn=st.session_state.duckdb_conn,
        table_names=tables_for_file,
    )

    # Update in-memory state
    st.session_state.processed_files.discard(filename)
    st.session_state.file_table_map.pop(filename, None)
    for t in tables_for_file:
        st.session_state.quality_reports.pop(t, None)

    # Re-query DuckDB for the real remaining tables
    if st.session_state.duckdb_conn is not None:
        st.session_state.loaded_tables = get_all_tables(st.session_state.duckdb_conn)
    else:
        st.session_state.loaded_tables = []


# ═══════════════════════════════════════════════
#  Chart helpers
# ═══════════════════════════════════════════════
def _infer_chart_spec(df: pd.DataFrame, question: str) -> tuple[str | None, dict]:
    if df.empty:
        return None, {}

    question_lower    = question.lower()
    numeric_cols      = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols     = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical_cols  = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

    if datetime_cols and numeric_cols and any(
        w in question_lower
        for w in ["trend", "over time", "timeline", "by date", "per day", "per month", "monthly", "daily"]
    ):
        return "line", {"x": datetime_cols[0], "y": numeric_cols[0], "color": categorical_cols[0] if categorical_cols else None}

    if categorical_cols and numeric_cols:
        return "bar", {"x": categorical_cols[0], "y": numeric_cols[0]}

    if len(numeric_cols) >= 2:
        return "scatter", {"x": numeric_cols[0], "y": numeric_cols[1]}

    return None, {}


def _render_chart_with_hint(
    df: pd.DataFrame,
    question: str,
    chart_type: str | None,
    x_col: str | None,
    y_col: str | None,
) -> None:
    """Render a chart using router-provided hints (chart_type, x/y axes)."""
    if df.empty:
        st.info("No data to display.")
        return

    # Validate router-suggested columns actually exist in df
    if x_col and x_col not in df.columns:
        x_col = None
    if y_col and y_col not in df.columns:
        y_col = None

    # Fall back to inference if hints are incomplete
    if not x_col or not y_col:
        _render_chart(df, question)
        return

    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col, title=f"{y_col} distribution")
        else:
            _render_chart(df, question)
            return

        st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "Download chart as HTML",
            data=fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
            file_name="chart.html",
            mime="text/html",
        )
    except Exception as e:
        st.warning(f"Router-guided chart failed ({e}), falling back to auto-detect.")
        _render_chart(df, question)


def _render_chart(df: pd.DataFrame, question: str) -> None:
    chart_kind, spec = _infer_chart_spec(df, question)
    if not chart_kind or not spec:
        st.info("No obvious chart mapping was detected for this result.")
        return

    chart_df = df.copy()
    if chart_kind == "line" and spec.get("x") in chart_df.columns:
        chart_df[spec["x"]] = pd.to_datetime(chart_df[spec["x"]], errors="coerce")
        chart_df = chart_df.dropna(subset=[spec["x"]])

    if chart_kind == "bar":
        x_col, y_col = spec["x"], spec["y"]
        chart_df = chart_df[[x_col, y_col]].dropna()
        chart_df = chart_df.groupby(x_col, as_index=False)[y_col].sum().sort_values(y_col, ascending=False).head(25)
        fig = px.bar(chart_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
    elif chart_kind == "line":
        fig = px.line(chart_df, x=spec["x"], y=spec["y"], color=spec.get("color"), title=f"{spec['y']} over {spec['x']}")
    else:
        fig = px.scatter(chart_df, x=spec["x"], y=spec["y"], title=f"{spec['y']} vs {spec['x']}")

    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "Download chart as HTML",
        data=fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
        file_name="chart.html",
        mime="text/html",
    )


# ═══════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════
st.title("📊 Analytics App")
st.caption(f"Max file size: {MAX_FILE_SIZE_MB} MB  ·  Session expires after {SESSION_TTL_MINUTES} min")
st.divider()


# ═══════════════════════════════════════════════
#  STEP 1 – Upload
# ═══════════════════════════════════════════════
st.subheader("1 · Upload your files")

uploaded_files = st.file_uploader(
    label="Drop one or more CSV, XLSX, JSON, or TXT files here",
    type=["csv", "xlsx", "json", "txt"],
    accept_multiple_files=True,
    key="file_uploader_widget",
    on_change=_on_uploader_change,
)

prompt = st.text_area(
    label="2 · What do you want to know about this data?",
    placeholder="e.g. Show total revenue by region for Q1",
    height=80,
)

run_button = st.button("▶  Load & Analyse", type="primary", disabled=(not uploaded_files))


# ═══════════════════════════════════════════════
#  STEP 2 – Process files
# ═══════════════════════════════════════════════
if run_button and uploaded_files:

    if st.session_state.session_id is None:
        session_id, session_dir = create_session()
        st.session_state.session_id  = session_id
        st.session_state.session_dir = session_dir

    conn = get_or_create_connection(st.session_state)

    for uploaded_file in uploaded_files:

        # ── Duplicate file check ──────────────────
        if uploaded_file.name in st.session_state.processed_files:
            st.warning(f"⚠️ **{uploaded_file.name}** was already loaded — skipping to avoid duplicate.")
            continue

        with st.status(f"Processing **{uploaded_file.name}**…", expanded=True) as status:

            # ── Validate ──────────────────────────
            st.write("🔍 Validating file…")
            ok, reason = validate_file(uploaded_file)
            if not ok:
                status.update(label=f"❌ {uploaded_file.name} rejected", state="error")
                st.error(reason)
                continue

            # ── Save to disk ──────────────────────
            st.write("💾 Saving to session folder…")
            file_path = save_uploaded_file(st.session_state.session_dir, uploaded_file)

            # ── Load into DuckDB ──────────────────
            st.write("🦆 Loading into DuckDB…")
            try:
                existing    = list(st.session_state.loaded_tables)
                new_tables, warnings = load_file_into_duckdb(file_path, conn, existing)
            except Exception as e:
                status.update(label=f"❌ Could not read {uploaded_file.name}", state="error")
                st.error(str(e))
                continue

            for w in warnings:
                st.warning(w)

            # ── Clean & profile ───────────────────
            for table_name in new_tables:
                st.write(f"🧹 Cleaning table `{table_name}`…")
                report = clean_and_profile(conn, table_name)
                st.session_state.quality_reports[table_name] = report

            # ── Register in state ─────────────────
            st.session_state.processed_files.add(uploaded_file.name)
            st.session_state.loaded_tables = get_all_tables(conn)
            # Track which tables belong to this file (for later deletion)
            st.session_state.file_table_map[uploaded_file.name] = new_tables

            label = f"✅ {uploaded_file.name} → " + ", ".join(f"`{t}`" for t in new_tables)
            status.update(label=label, state="complete", expanded=False)


# ═══════════════════════════════════════════════
#  STEP 3 – Loaded files list (native ✕ on uploader handles per-file removal)
# ═══════════════════════════════════════════════
conn   = st.session_state.duckdb_conn
tables = st.session_state.loaded_tables

if st.session_state.processed_files:
    st.divider()

    # ── Header row: title + Delete Session button ─────────────
    hdr_left, hdr_right = st.columns([3, 1])
    with hdr_left:
        st.subheader("📁 Loaded Files")
    with hdr_right:
        if st.button("🗑 Delete Session", type="secondary", use_container_width=True):
            if st.session_state.session_id:
                delete_session(
                    session_id=st.session_state.session_id,
                    conn=st.session_state.duckdb_conn,
                )
            _reset_session_state()
            st.rerun()

    # ── Read-only summary (removal is handled by the native ✕ in the uploader) ─
    for fname in sorted(st.session_state.processed_files):
        file_tables = st.session_state.file_table_map.get(fname, [])
        col_name, col_tables = st.columns([3, 5])
        with col_name:
            st.markdown(f"📄 **{fname}**")
        with col_tables:
            if file_tables:
                st.caption("Tables: " + ", ".join(f"`{t}`" for t in file_tables))


# ═══════════════════════════════════════════════
#  STEP 4 – Table previews + quality reports
# ═══════════════════════════════════════════════
if conn and tables:

    st.divider()
    st.subheader(f"📦 Loaded Tables  ({len(tables)} total)")

    tabs = st.tabs([f"🗂 {t}" for t in tables])

    for tab, table_name in zip(tabs, tables):
        with tab:

            report = st.session_state.quality_reports.get(table_name, {})

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows",               report.get("clean_rows", "—"))
            col2.metric("Duplicates removed", report.get("duplicates_removed", "—"))
            col3.metric("Columns",            len(report.get("columns", [])))

            with st.expander("Column details"):
                st.write("**Columns:**",    report.get("columns", []))
                st.write("**Data types:**", report.get("dtypes", {}))
                if report.get("coerced_to_numeric"):
                    st.write("**Text → numeric:**", report["coerced_to_numeric"])
                null_counts = {k: v for k, v in report.get("null_counts", {}).items() if v > 0}
                if null_counts:
                    st.write("**Nulls preserved:**", null_counts)
                if report.get("missing_cells") is not None:
                    st.write("**Missing cells detected:**", report["missing_cells"])

            st.markdown("**Preview (first 50 rows)**")
            preview = conn.execute(f"SELECT * FROM {table_name} LIMIT 50").df()
            st.dataframe(preview, use_container_width=True)

    # ═══════════════════════════════════════════════
    #  STEP 5 – NL → SQL query
    # ═══════════════════════════════════════════════
    st.divider()
    st.subheader("💬 Query")
    st.info(
        "**Tables ready to query:** " +
        "  |  ".join(f"`{t}`" for t in tables) +
        "\n\nThe query assistant can use all loaded tables."
    )
    if prompt.strip():
        if len(prompt) > PROMPT_MAX_LENGTH:
            st.error(f"Prompt is too long (max {PROMPT_MAX_LENGTH} characters). Please shorten it.")
        else:
            # ══════════════════════════════════════════
            #  QUERY ROUTER  (runs before NL→SQL)
            #  Stage 1: regex keyword (~1ms, zero LLM)
            #  Stage 2: local Ollama  (data stays local)
            # ══════════════════════════════════════════

            # Build a privacy-safe DDL covering EVERY loaded table, so the
            # router can correctly classify questions about any uploaded file.
            primary_table = tables[0] if tables else ""
            ddl_for_router = generate_multi_table_ddl(
                conn, tables, redact=False, max_columns=DDL_MAX_COLUMNS
            ) if tables else ""

            routing_result = route_query(
                user_prompt       = prompt,
                ddl_schema        = ddl_for_router,
                table_name        = ", ".join(tables) if tables else "",
                sample_rows       = None,   # No raw rows sent — privacy-safe
                print_to_terminal = True,   # Prints route decision to terminal
            )

            parsed_route = routing_result.get("parsed") or {}
            route_label  = parsed_route.get("route", "sql_answer")
            confidence   = parsed_route.get("confidence", "HIGH")
            chart_hint   = parsed_route.get("chart_type")
            route_stage  = routing_result.get("stage", "?")

            # ── Show routing badge in UI ──────────────
            route_icons = {
                "visualization": "📊",
                "sql_answer":    "🔢",
                "metadata":      "🗂️",
                "statistical":   "📉",
                "reasoning":     "💡",
            }
            icon = route_icons.get(route_label, "❓")
            conf_colour = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(confidence, "⚪")

            with st.expander(f"{icon} Route: **{route_label.upper()}**  {conf_colour} {confidence}  ·  stage={route_stage}", expanded=False):
                if routing_result.get("success"):
                    st.json(parsed_route)
                else:
                    st.warning(f"Router fallback — defaulting to sql_answer. Error: {routing_result.get('error')}")

            # ── Clarification gate for LOW confidence ─
            if confidence == "LOW" and routing_result.get("success"):
                st.warning(
                    "⚠️ The query router is unsure how to classify this question. "
                    "It will proceed as **sql_answer** — refine your question if the result looks wrong."
                )

            # ══════════════════════════════════════════
            #  Metadata route — answer from DDL directly
            #  (no SQL needed, no LLM for data queries)
            # ══════════════════════════════════════════
            if route_label == "metadata":
                st.subheader("🗂️ Schema Information")
                for t in tables:
                    ddl = generate_privacy_safe_ddl(conn, t, redact=False, max_columns=DDL_MAX_COLUMNS)
                    with st.expander(f"Table: `{t}`", expanded=True):
                        st.code(ddl, language="sql")
                        try:
                            info = conn.execute(f"DESCRIBE {t}").df()
                            st.dataframe(info, use_container_width=True)
                        except Exception:
                            pass

            elif route_label == "visualization":
                # ══════════════════════════════════════
                #  ROUTE A — Full Visualization Pipeline
                #  Intent → SQL → 3-layer Validation
                #  → DuckDB Execution → Plotly Chart
                # ══════════════════════════════════════
                with st.spinner("🎨 Running Route A visualization pipeline…"):
                    route_a_result = run_route_a(
                        conn          = conn,
                        tables        = tables,
                        prompt        = prompt,
                        router_intent = parsed_route,   # pass router hints in
                    )

                if not route_a_result.success:
                    st.error(f"Route A failed at stage **{route_a_result.stage_reached}**: {route_a_result.error}")

                    # Show validation log if available
                    if route_a_result.validation_log:
                        with st.expander("🔍 Validation log"):
                            for line in route_a_result.validation_log:
                                st.text(line)

                    # Show generated SQL even on failure so user can debug
                    if route_a_result.sql:
                        with st.expander("🔎 Generated SQL (failed)"):
                            st.code(route_a_result.sql, language="sql")
                else:
                    # ── Intent summary ────────────────────
                    with st.expander("🧠 Extracted Intent", expanded=False):
                        st.json(route_a_result.intent)

                    # ── SQL ───────────────────────────────
                    with st.expander("🔎 Generated SQL", expanded=False):
                        st.code(route_a_result.sql, language="sql")

                    # ── Validation log ────────────────────
                    with st.expander("✅ Validation log", expanded=False):
                        for line in route_a_result.validation_log:
                            st.text(line)

                    # ── Chart ─────────────────────────────
                    st.subheader(f"📊 {route_a_result.intent.get('title', 'Visualization')}")
                    st.plotly_chart(route_a_result.fig, use_container_width=True)

                    # ── Export buttons ────────────────────
                    col_csv, col_html = st.columns(2)
                    with col_csv:
                        st.download_button(
                            "⬇️ Download data as CSV",
                            data=route_a_result.df.to_csv(index=False).encode("utf-8"),
                            file_name="chart_data.csv",
                            mime="text/csv",
                        )
                    with col_html:
                        st.download_button(
                            "⬇️ Download chart as HTML",
                            data=route_a_result.fig.to_html(include_plotlyjs="cdn").encode("utf-8"),
                            file_name="chart.html",
                            mime="text/html",
                        )

                    # ── Raw data table ────────────────────
                    with st.expander("📋 Raw query result", expanded=False):
                        st.dataframe(route_a_result.df.head(50), use_container_width=True)

            else:
                # ══════════════════════════════════════
                #  All other routes → NL→SQL pipeline
                #  (sql_answer, statistical, reasoning)
                # ══════════════════════════════════════
                llm_provider = os.environ.get("LLM_PROVIDER", "ollama")
                api_key      = os.environ.get("OPENAI_API_KEY")
                sql, llm_prompt, used = nl2sql.generate_sql_from_prompt(
                    conn,
                    tables,
                    prompt,
                    redact=True,
                    max_prompt_length=PROMPT_MAX_LENGTH,
                    llm_provider=llm_provider,
                    openai_api_key=api_key,
                )

                if not used:
                    st.code("-- LLM not configured or prompt exceeds limit. Preview of constructed prompt:\n" + llm_prompt)
                    st.info(f"Constructed prompt length: {len(llm_prompt)} chars")
                else:
                    if sql is None:
                        st.error("LLM did not return a valid SQL statement. See prompt/response for details.")
                        st.code(llm_prompt)
                    else:
                        ok, reason = validate_sql_query(conn, sql, tables)
                        if not ok:
                            st.error(reason)
                            st.code(sql)
                        else:
                            st.subheader("🔎 Generated SQL")
                            st.code(sql)
                            try:
                                df = conn.execute(sql).df()
                                st.subheader("📋 Query result (first 50 rows)")
                                st.dataframe(df.head(50), use_container_width=True)
                                st.download_button(
                                    "Download result as CSV",
                                    data=df.to_csv(index=False).encode("utf-8"),
                                    file_name="query_result.csv",
                                    mime="text/csv",
                                )
                                st.subheader("📈 Visualization")
                                _render_chart(df, prompt)
                            except Exception as e:
                                st.error(f"Could not execute generated SQL: {e}")