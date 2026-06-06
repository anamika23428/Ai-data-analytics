# ─────────────────────────────────────────────
#  app.py  –  Main Streamlit UI
#
#  Fixes:
#    1. Duplicate file upload → detected and skipped with warning
#    2. Re-clicking Analyse → stale quality reports are refreshed
#    3. Warnings from ingestor (table name conflicts) shown to user
# ─────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from config import MAX_FILE_SIZE_MB, SESSION_TTL_MINUTES
from core.validator import validate_file, validate_sql_query
from core.session_mgr import create_session, save_uploaded_file, cleanup_old_sessions
from core.ingestor    import get_or_create_connection, load_file_into_duckdb, get_all_tables
from core.transformer import clean_and_profile
from core.session_mgr import start_cleanup_daemon
from core import nl2sql
import os
import re
from config import PROMPT_MAX_LENGTH


st.set_page_config(page_title="Analytics App", page_icon="📊", layout="wide")

cleanup_old_sessions()
start_cleanup_daemon()

# ── Session state defaults ────────────────────
if "session_id"      not in st.session_state: st.session_state.session_id      = None
if "session_dir"     not in st.session_state: st.session_state.session_dir     = None
if "duckdb_conn"     not in st.session_state: st.session_state.duckdb_conn     = None
if "loaded_tables"   not in st.session_state: st.session_state.loaded_tables   = []
if "quality_reports" not in st.session_state: st.session_state.quality_reports = {}
if "processed_files" not in st.session_state: st.session_state.processed_files = set()  # tracks filenames already loaded


def _infer_chart_spec(df: pd.DataFrame, question: str) -> tuple[str | None, dict]:
    if df.empty:
        return None, {}

    question_lower = question.lower()
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    categorical_cols = [col for col in df.columns if col not in numeric_cols and col not in datetime_cols]

    if datetime_cols and numeric_cols and any(word in question_lower for word in ["trend", "over time", "timeline", "by date", "per day", "per month", "monthly", "daily"]):
        return "line", {"x": datetime_cols[0], "y": numeric_cols[0], "color": categorical_cols[0] if categorical_cols else None}

    if categorical_cols and numeric_cols:
        return "bar", {"x": categorical_cols[0], "y": numeric_cols[0]}

    if len(numeric_cols) >= 2:
        return "scatter", {"x": numeric_cols[0], "y": numeric_cols[1]}

    return None, {}


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
        x_col = spec["x"]
        y_col = spec["y"]
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


# ── Header ────────────────────────────────────
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
                existing = list(st.session_state.loaded_tables)   # snapshot before load
                new_tables, warnings = load_file_into_duckdb(file_path, conn, existing)
            except Exception as e:
                status.update(label=f"❌ Could not read {uploaded_file.name}", state="error")
                st.error(str(e))
                continue

            # Show any table-name conflict warnings
            for w in warnings:
                st.warning(w)

            # ── Clean & profile ───────────────────
            for table_name in new_tables:
                st.write(f"🧹 Cleaning table `{table_name}`…")
                # Always regenerate — fixes stale reports on re-upload
                report = clean_and_profile(conn, table_name)
                st.session_state.quality_reports[table_name] = report

            # Mark file as processed so re-clicking doesn't duplicate it
            st.session_state.processed_files.add(uploaded_file.name)
            st.session_state.loaded_tables = get_all_tables(conn)

            label = f"✅ {uploaded_file.name} → " + ", ".join(f"`{t}`" for t in new_tables)
            status.update(label=label, state="complete", expanded=False)


# ═══════════════════════════════════════════════
#  STEP 3 – Show results
# ═══════════════════════════════════════════════
conn   = st.session_state.duckdb_conn
tables = st.session_state.loaded_tables

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

    st.divider()
    st.subheader("💬 Query")
    st.info(
        "**Tables ready to query:** " +
        "  |  ".join(f"`{t}`" for t in tables) +
        "\n\nThe query assistant can use all loaded tables."
    )
    if prompt.strip():
        # Prompt length validation
        if len(prompt) > PROMPT_MAX_LENGTH:
            st.error(f"Prompt is too long (max {PROMPT_MAX_LENGTH} characters). Please shorten it.")
        else:
            llm_provider = os.environ.get("LLM_PROVIDER", "ollama")
            api_key = os.environ.get("OPENAI_API_KEY")
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