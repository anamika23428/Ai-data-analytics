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

from config import MAX_FILE_SIZE_MB, SESSION_TTL_MINUTES
from core.validator   import validate_file
from core.session_mgr import create_session, save_uploaded_file, cleanup_old_sessions
from core.ingestor    import get_or_create_connection, load_file_into_duckdb, get_all_tables
from core.transformer import clean_and_profile


st.set_page_config(page_title="Analytics App", page_icon="📊", layout="wide")

cleanup_old_sessions()

# ── Session state defaults ────────────────────
if "session_id"      not in st.session_state: st.session_state.session_id      = None
if "session_dir"     not in st.session_state: st.session_state.session_dir     = None
if "duckdb_conn"     not in st.session_state: st.session_state.duckdb_conn     = None
if "loaded_tables"   not in st.session_state: st.session_state.loaded_tables   = []
if "quality_reports" not in st.session_state: st.session_state.quality_reports = {}
if "processed_files" not in st.session_state: st.session_state.processed_files = set()  # tracks filenames already loaded


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
                    st.write("**Nulls filled:**", null_counts)

            st.markdown("**Preview (first 50 rows)**")
            preview = conn.execute(f"SELECT * FROM {table_name} LIMIT 50").df()
            st.dataframe(preview, use_container_width=True)

    st.divider()
    st.subheader("💬 Query")
    st.info(
        "**Tables ready to query:** " +
        "  |  ".join(f"`{t}`" for t in tables) +
        "\n\n🚧 NL → SQL pipeline coming next."
    )
    if prompt.strip():
        st.code(f"-- Your prompt will generate SQL like:\nSELECT * FROM {tables[0]} LIMIT 10;")