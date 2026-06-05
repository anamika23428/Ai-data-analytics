# ─────────────────────────────────────────────
#  app.py  –  Main Streamlit UI
#
#  Run with:  streamlit run app.py
#
#  Page flow:
#    1. User uploads a file + types a prompt
#    2. We validate the file (size, MIME type)
#    3. Save it to a private session folder
#    4. Load it into DuckDB
#    5. Clean & profile the data
#    6. Show a quality report + preview table
#    7. (Placeholder) Run the user's prompt as SQL / NL query
# ─────────────────────────────────────────────

import sys
from pathlib import Path

# Make sure Python can find config.py and the core/ folder
# regardless of which directory Streamlit was launched from
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# Our own modules
from config import MAX_FILE_SIZE_MB, SESSION_TTL_MINUTES
from core.validator   import validate_file
from core.session_mgr import create_session, save_uploaded_file, cleanup_old_sessions
from core.ingestor    import load_file_into_duckdb
from core.transformer import clean_and_profile


# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="Analytics App",
    page_icon="📊",
    layout="wide",
)


# ── Clean up any expired sessions on each app start ──
cleanup_old_sessions()


# ── Session state defaults ────────────────────
# st.session_state persists across reruns (like Zustand did in React)
if "session_id"     not in st.session_state:   st.session_state.session_id   = None
if "session_dir"    not in st.session_state:   st.session_state.session_dir  = None
if "duckdb_conn"    not in st.session_state:   st.session_state.duckdb_conn  = None
if "quality_report" not in st.session_state:   st.session_state.quality_report = None


# ── Header ────────────────────────────────────
st.title("📊 Analytics App")
st.caption(f"Max file size: {MAX_FILE_SIZE_MB} MB  ·  Session expires after {SESSION_TTL_MINUTES} min")
st.divider()


# ═══════════════════════════════════════════════
#  STEP 1 – Upload panel
# ═══════════════════════════════════════════════
st.subheader("1 · Upload your file")

uploaded_file = st.file_uploader(
    label="Drop a CSV, XLSX, JSON, or TXT file here",
    type=["csv", "xlsx", "json", "txt"],
)

prompt = st.text_area(
    label="2 · What do you want to know about this data?",
    placeholder="e.g. Show me total revenue by region for Q1",
    height=80,
)

run_button = st.button("▶  Analyse", type="primary", disabled=(uploaded_file is None))


# ═══════════════════════════════════════════════
#  STEP 2 – Process the file when button is clicked
# ═══════════════════════════════════════════════
if run_button and uploaded_file:

    with st.status("Processing your file…", expanded=True) as status:

        # ── Validate ─────────────────────────────
        st.write("🔍 Checking file safety…")
        ok, reason = validate_file(uploaded_file)

        if not ok:
            status.update(label="❌ File rejected", state="error")
            st.error(reason)
            st.stop()

        # ── Create session folder ─────────────────
        st.write("📁 Creating private session folder…")
        session_id, session_dir = create_session()
        st.session_state.session_id  = session_id
        st.session_state.session_dir = session_dir

        # ── Save file to disk ─────────────────────
        st.write("💾 Saving file…")
        file_path = save_uploaded_file(session_dir, uploaded_file)

        # ── Load into DuckDB ──────────────────────
        st.write("🦆 Loading into DuckDB…")
        try:
            conn = load_file_into_duckdb(file_path)
        except Exception as e:
            status.update(label="❌ Could not read file", state="error")
            st.error(str(e))
            st.stop()

        # ── Clean & profile ───────────────────────
        st.write("🧹 Cleaning and profiling data…")
        quality_report = clean_and_profile(conn)

        # Save to session state so we don't re-run on every Streamlit rerun
        st.session_state.duckdb_conn    = conn
        st.session_state.quality_report = quality_report

        status.update(label="✅ Ready!", state="complete", expanded=False)


# ═══════════════════════════════════════════════
#  STEP 3 – Show results (if data is loaded)
# ═══════════════════════════════════════════════
conn   = st.session_state.duckdb_conn
report = st.session_state.quality_report

if conn and report:

    st.divider()

    # ── Quality report ────────────────────────
    st.subheader("📋 Data Quality Report")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows (after cleaning)", report["clean_rows"])
    col2.metric("Duplicates removed",    report["duplicates_removed"])
    col3.metric("Columns",              len(report["columns"]))

    with st.expander("Column details"):
        st.write("**Columns after name clean-up:**", report["columns"])
        st.write("**Data types:**", report["dtypes"])
        if report["coerced_to_numeric"]:
            st.write("**Text → numeric conversions:**", report["coerced_to_numeric"])
        if any(v > 0 for v in report["null_counts"].values()):
            st.write("**Null counts (before fill):**", report["null_counts"])

    # ── Data preview ─────────────────────────
    st.subheader("👀 Data Preview")
    preview_df = conn.execute("SELECT * FROM data LIMIT 50").df()
    st.dataframe(preview_df, use_container_width=True)

    # ── Prompt result (placeholder) ──────────
    st.divider()
    st.subheader("💬 Query Result")

    if prompt.strip():
        st.info(
            "🚧 NL→SQL pipeline coming next. "
            "The schema below will be sent to the LLM to generate a SQL query.\n\n"
            f"**Schema:** {report['dtypes']}"
        )
    else:
        st.caption("Type a question above and click Analyse to query your data.")