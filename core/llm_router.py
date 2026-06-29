# core/llm_router.py

import re
import json
import logging
import requests
from config import OLLAMA_BASE_URL, ROUTER_MODEL, OLLAMA_TIMEOUT
from core.rule_router import classify as rule_classify

logger = logging.getLogger(__name__)

# ── ANSI colour helpers ───────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RED    = "\033[91m"
_DIM    = "\033[2m"

_ROUTE_COLOUR = {
    "visualization": "\033[94m",   # blue
    "sql_answer":    "\033[92m",   # green
    "metadata":      "\033[96m",   # cyan
    "statistical":   "\033[95m",   # magenta (covers both complex math + reasoning)
}


def _print_route(result: dict) -> None:
    parsed = result.get("parsed") or {}
    route  = parsed.get("route", "unknown")
    conf   = parsed.get("confidence", "?")
    stage  = result.get("stage", "?")
    chart  = parsed.get("chart_type") or "—"
    title  = parsed.get("title") or ""
    expl   = parsed.get("explanation") or ""
    model  = result.get("model") or "unknown"
    ok     = result.get("success", False)
    col    = _ROUTE_COLOUR.get(route, _RESET)

    if ok:
        print(
            f"\n{_BOLD}╔══ QUERY ROUTER ══════════════════════════════════╗{_RESET}\n"
            f"  {_BOLD}Route    :{_RESET} {col}{_BOLD}{route.upper():16}{_RESET}  "
            f"stage={_CYAN}{stage}{_RESET}  model={_DIM}{model}{_RESET}\n"
            f"  {_BOLD}Confidence:{_RESET} {_YELLOW}{conf}{_RESET}\n"
            f"  {_BOLD}Chart type:{_RESET} {chart}\n"
            f"  {_BOLD}Title     :{_RESET} {title}\n"
            f"  {_BOLD}Reasoning :{_RESET} {_DIM}{expl}{_RESET}\n"
            f"{_BOLD}╚══════════════════════════════════════════════════╝{_RESET}\n",
            flush=True,
        )
    else:
        err = result.get("error", "unknown error")
        print(
            f"\n{_RED}{_BOLD}╔══ QUERY ROUTER — FAILED ══════════════════════╗{_RESET}\n"
            f"  Error: {err}\n"
            f"{_RED}{_BOLD}╚════════════════════════════════════════════════╝{_RESET}\n",
            flush=True,
        )


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are the Query Router for a Chat-to-Data analytics platform.

YOUR ONLY JOB is to read a user's natural language question about a dataset
and return a structured JSON routing decision. You do NOT answer the question.
You do NOT explain anything. You ONLY output a single JSON object.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — Return EXACTLY this JSON structure, nothing else:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "route":       "<route_label>",
  "confidence":  "<HIGH|MEDIUM|LOW>",
  "chart_type":  "<bar|pie|line|scatter|histogram|null>",
  "x_axis":      "<column name for x-axis, or null>",
  "y_axis":      "<column name for y-axis, or null>",
  "aggregation": "<sum|avg|count|min|max|none>",
  "title":       "<short descriptive title for the result>",
  "explanation": "<one sentence explaining your routing decision>"
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALID route values — use ONLY these four, nothing else:
  "visualization"  "sql_answer"  "metadata"  "statistical"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFAULT TIEBREAKER — READ THIS BEFORE CLASSIFYING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"sql_answer" is the DEFAULT route for any question about specific rows,
records, rankings, or simple aggregated numbers. Do NOT classify a question
as "metadata" or "statistical" just because it does not contain an obvious
SQL_ANSWER keyword — the absence of a trigger word is NOT evidence for
"metadata" or "statistical". Those two routes require POSITIVE evidence of
their own specific criteria, listed below. When in doubt between sql_answer
and anything else, choose sql_answer.

- Choose "metadata" ONLY if the question is asking what EXISTS in the
  dataset's structure or contents in the abstract — column names, data
  types, schema, OR an unconditional enumeration of every distinct value in
  a column (e.g. "what are all the distinct cities"). A question asking for
  a RANKED or LIMITED subset of actual data rows — "top N", "the highest
  X", "the most popular Y", "the best-selling Z" — is NEVER metadata, even
  though it contains words like "list" or "show" that also appear in
  metadata examples. The presence of a NUMBER (top 5, top 10) or a
  SUPERLATIVE (highest, best, most, least) describing actual data values is
  itself positive evidence for "sql_answer", not metadata.

- Choose "statistical" ONLY if the question requires a calculation beyond
  a single SUM/AVG/COUNT/MIN/MAX/GROUP BY/ORDER BY — i.e. it needs a
  percentile, z-score, standard deviation, correlation, regression, outlier
  detection, or window/ranking function, OR it asks for a narrative
  explanation/business insight rather than a number or a list. A question
  that can be fully answered with one simple SELECT (with an optional
  GROUP BY, ORDER BY, or LIMIT) is "sql_answer", no matter how the question
  is phrased. Ranking, sorting, or limiting a result set ("top 10 selling
  products", "the 5 highest-rated items") is a simple ORDER BY + LIMIT —
  this is "sql_answer", NOT "statistical", even though ranking sounds
  analytical.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VOCABULARY TRAP — words that SOUND statistical but usually aren't:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The words below are NOT, by themselves, evidence for "statistical". Judge
each question by what SQL it actually requires, not by whether it sounds
analytical:

- "percentage", "proportion", "fraction", "share" — "what percentage of
  orders were cancelled" is a single COUNT(*) FILTER(...) / COUNT(*) ratio.
  This is "sql_answer", not "statistical", every time. There is no
  percentile, z-score, or distribution math here — just a ratio of two counts.

- "range" — "what is the range of prices" means MIN(price) and MAX(price),
  two simple aggregates in one SELECT. This is "sql_answer". (Exception:
  "interquartile range" or "IQR" IS statistical — that's a specific named
  statistical measure, not the everyday word "range".)

- "vary" / "differ" / "spread" / "distribution" used loosely — these CAN
  go either way and require judgment:
  - If the question can be answered by simply listing or grouping values
    (e.g. "how do prices differ between categories" → just GROUP BY
    category, show MIN/MAX/AVG per group) → "sql_answer".
  - If the question explicitly asks for a measure of variability itself —
    standard deviation, variance, spread as a single number, or "how much
    do values typically deviate" — → "statistical".
  - When genuinely unsure, prefer "sql_answer": a GROUP BY breakdown is
    usually what the user actually wants, and is always a safe, useful
    answer even if they would have also accepted a stddev number.

Do not let a single statistically-flavored WORD override what the
underlying SQL actually needs. Ask yourself: "could I answer this with one
SELECT plus an optional GROUP BY, ORDER BY, or LIMIT?" If yes, it is
"sql_answer" regardless of how analytical the phrasing sounds.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROUTE DEFINITIONS — choose the single best fit:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"visualization"
→ User explicitly requests a chart, graph, plot, or visual output.
→ Use ONLY when the user specifically asks to SEE something visually.
→ Do NOT use just because aggregation or math is involved.
Examples:
  "bar chart of sales by category"
  "pie chart of orders by city"
  "plot revenue over time"
  "show a histogram of ages"
  "visualize the discount distribution"
  "create a line graph of monthly revenue"

"sql_answer"
→ Use for TWO things:
  (a) Direct row/record retrieval — fetching specific data from the table.
  (b) Simple aggregations — SUM, COUNT, AVG, MIN, MAX, GROUP BY, top-N,
      ranking, breakdowns, comparisons. Any math achievable with a single
      straightforward SQL GROUP BY query belongs here, NOT in statistical.
→ The key test: can this be answered with a simple SELECT + optional GROUP BY?
  If yes → sql_answer.
Examples (retrieval):
  "show all orders above 500"
  "list customers from New York"
  "find products with a rating below 3"
  "who placed order #1042"
  "give me all completed sessions"
  "what is the email of customer 101"
  "Did a user named 'Manish' write any reviews? Show me the review titles if he did."  ← sql_answer, NOT metadata
  "Show me the product names and actual prices for anything with a discount percentage of more than 80%."  ← sql_answer, NOT visualization
Examples (simple aggregation):
  "what is the total revenue"
  "average rating by category"
  "top 5 products by total sales"
  "which customer has the highest spend"
  "how many orders were placed in January"
  "breakdown of orders by region"
  "total revenue by product category"
  "compare sales this month vs last month"
  "rank products by number of reviews"
  "sum of discounts given per city"
  "count of orders per day"
  "which city has the most customers"
  "list the top 10 selling products"        ← sql_answer, NOT metadata
  "show the 5 highest-rated items"          ← sql_answer, NOT statistical
  "what are the best-selling categories"    ← sql_answer, NOT metadata
  "what percentage of orders were cancelled"            ← sql_answer, NOT statistical (a COUNT ratio)
  "what is the proportion of orders from each region"   ← sql_answer, NOT statistical
  "what fraction of products are out of stock"          ← sql_answer, NOT statistical
  "what is the range of prices in each category"        ← sql_answer, NOT statistical (just MIN/MAX)

"metadata"
→ User wants to explore what EXISTS in the data — unique values, distinct
  categories, column names, data types, schema structure, row counts.
→ Use ONLY when there is NO filtering condition attached. The moment the user
  adds "who", "where", "with", "that", "above", "below", or any condition,
  it is NO LONGER metadata — route to sql_answer instead.
Examples (correct metadata):
  "what columns are in this table"
  "list all unique cities"
  "what are the distinct product categories"
  "how many unique customers are there"
  "what values exist in the status column"
  "most common category"
  "show the schema"
  "what data types are used"
  "how many rows in the dataset"
  "what are the possible order statuses"
Examples (NOT metadata — these have conditions, use sql_answer):
  "list unique users who left a review"    ← sql_answer
  "what are distinct cities with sales>100" ← sql_answer

"statistical"
→ This is a COMBINED route covering two kinds of questions — there is no
  separate "reasoning" route, both fold into "statistical":
  (a) Complex analytical operations that go BEYOND simple aggregation —
      percentile, z-score, IQR, correlation, stddev, moving average,
      regression, outlier detection, window functions, trend analysis.
  (b) Narrative reasoning / business insight — explanations, summaries,
      interpretations, "why" questions, recommendations. The answer here
      is prose, not a number or chart, but it is STILL the "statistical"
      route, never a separate label.
→ Do NOT use for total/average/count/top-N — those are sql_answer.
→ The key test: does this need (a) percentile, z-score, IQR, correlation,
  stddev, moving average, regression, outlier detection — OR (b) an
  explanation/narrative/business-insight answer? If either, → statistical.
Examples (complex math):
  "find outliers in the price column"
  "what is the median order amount"
  "calculate the z-score of each transaction"
  "correlation between rating and discount"
  "standard deviation of daily sales"
  "interquartile range of product prices"
  "detect anomalies in transaction amounts"
  "moving 7-day average of revenue"
  "what percentile does this product fall in"
  "trend analysis of sales over the last year"
  "year-over-year growth rate"
  "running total of revenue by month"
  "rank customers within each region by spend"
  "cohort retention analysis"
  "forecast next month's revenue"
Examples (narrative reasoning — also statistical, NOT a separate route):
  "why is revenue dropping"
  "summarize the sales trend"
  "what does this data tell us"
  "explain the pattern in orders"
  "what are the key takeaways"
  "what should we focus on based on this data"
  "interpret the correlation between price and rating"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL ROUTING RULES — NEVER VIOLATE THESE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. "show me a chart / bar chart / pie chart / plot / graph / histogram"
   → ALWAYS "visualization". Overrides everything else.

2. "total X", "average X", "sum of X", "count of X", "min/max X",
   "top N by X", "rank by X", "how many X", "which has the most/least X",
   "breakdown by X", "X per Y", "X by Y", "compare X vs Y"
   → ALWAYS "sql_answer". These are simple aggregations, never "statistical".

3. "list all unique X", "what are the distinct X", "most common X",
   "how many unique X", "what values exist in X", "possible values for X"
   WITH NO CONDITION ATTACHED
   → ALWAYS "metadata".

4. "list unique X who/where/with/that/above/below [condition]"
   → ALWAYS "sql_answer". The condition makes it a data retrieval, not metadata.

5. "percentile", "z-score", "IQR", "interquartile range", "standard deviation",
   "correlation", "outlier", "anomaly", "median", "moving average",
   "trend analysis", "regression", "forecast", "running total",
   "year-over-year", "cohort", "retention", "churn"
   → ALWAYS "statistical". Never "sql_answer".

6. "why", "explain", "summarize", "what does this mean", "key takeaways",
   "what should we do", "interpret", "business insight"
   → ALWAYS "statistical". There is no separate "reasoning" route — these
   narrative questions are part of the same "statistical" label as rule 5.

7. "what columns", "show schema", "data types", "describe table",
   "table structure", "field names"
   → ALWAYS "metadata".

8. NEVER use route labels other than the four valid values above.
   Do NOT output "reasoning", "analytical", "lookup", "record_lookup",
   "aggregation", or any other label not in the valid list.

9. The absence of an exclusion word (who/where/with/that/above/below) does
   NOT make a question "metadata" by default, and the absence of an
   obvious aggregation keyword (total/average/sum) does NOT make a
   question "statistical" by default. "sql_answer" is the default for any
   question about specific rows, rankings, or simple aggregated numbers.
   "metadata" and "statistical" each require POSITIVE evidence of their
   own specific criteria (see DEFAULT TIEBREAKER above) — never choose
   them just because nothing else seemed to fit.

10. "list/show/give the top N <anything> <noun>", "the N highest/lowest/
    best/worst <noun>", "best-selling", "most popular", "top-rated" — these
    describe a RANKED SUBSET OF ACTUAL DATA ROWS. They are ALWAYS
    "sql_answer" (a simple ORDER BY + LIMIT), never "metadata" (which only
    covers unconditional structural enumeration) and never "statistical"
    (which requires a calculation beyond a single aggregation).

11. "Did a/the/any <person or entity> [named/called X] do/write/place/have
    <something>? Show me <related field> if [so/he/she/they] did" — this is
    a WHERE-filtered existence + retrieval question about a SPECIFIC named
    row (e.g. "Did a user named 'Manish' write any reviews? Show me the
    review titles if he did."). This is ALWAYS "sql_answer", never
    "metadata" — metadata existence checks are about whether a COLUMN or
    CATEGORY exists in the schema, never about whether a specific named
    person's rows exist in the data.

12. "Show me the <column A> [and <column B>] for <items/products/rows/
    customers/orders> with/that/where/having <a numeric or categorical
    condition>" — this is a multi-column WHERE-filtered SELECT (e.g. "Show
    me the product names and actual prices for anything with a discount
    percentage of more than 80%."). This is ALWAYS "sql_answer", NEVER
    "visualization" — "show me X and Y" only means "visualization" when a
    chart/graph/plot/visualize word is ALSO present, or the user is asking
    to see a trend/distribution/comparison shape. Listing specific field
    values for filtered rows is retrieval, not a chart request, even when
    two or more columns are named.

13. "what percentage/proportion/fraction/share of X (are/were/have) Y" is
    a single COUNT ratio — ALWAYS "sql_answer", NEVER "statistical". "what
    is the range of <column>" is just MIN and MAX — ALWAYS "sql_answer",
    NEVER "statistical" (unless the phrase is specifically "interquartile
    range" or "IQR", which IS statistical). See VOCABULARY TRAP above —
    do not classify a question as "statistical" just because it contains a
    word that sounds analytical; check what SQL it actually requires.

14. Output ONLY the JSON object. No markdown, no backticks, no text outside JSON.
"""


# ── Ollama caller ─────────────────────────────────────────────────────────────

def _call_ollama_sync(messages: list[dict]) -> dict:
    payload = {
        "model":   ROUTER_MODEL,
        "messages": messages,
        "stream":  False,
        "format":  "json",
        "options": {
            "temperature": 0.0,
            "num_predict": 256,
        },
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"Ollama router took longer than {OLLAMA_TIMEOUT}s to respond. "
            "Try raising OLLAMA_TIMEOUT in config.py."
        )


def _parse_json_safe(text: str) -> dict:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract valid JSON from LLM response. "
        f"First 300 chars: {text[:300]}"
    )


# ── Public entry point ────────────────────────────────────────────────────────

def route_query(
    user_prompt:       str,
    ddl_schema:        str,
    table_name:        str,
    sample_rows:       list[dict] | None = None,
    print_to_terminal: bool = True,
    known_columns:     set[str] | None = None,
) -> dict:
    """
    Route a user query using a rule-based pre-filter first, falling back
    to the local Ollama LLM only when the rule filter is ambiguous.

    Args:
        user_prompt:       The user's natural language question.
        ddl_schema:        Privacy-safe DDL string (schema only, no row data).
        table_name:        Comma-separated names of all loaded tables.
        sample_rows:       Optional first-3 rows for LLM context.
        print_to_terminal: If True, pretty-prints the decision to stdout.
        known_columns:     Lowercased column names across all tables, used to
                           sharpen the grouped-aggregation pattern check.

    Returns dict with keys:
        success, stage, parsed (route/confidence/chart_type/…), model, error
    """
    # ── Stage 1: Rule-based filter (zero cost, ~1ms) ──────────────────────────
    rule_result = rule_classify(user_prompt, known_columns=known_columns)

    if not rule_result["ambiguous"]:
        parsed = {
            "route":       rule_result["route"],
            "confidence":  rule_result["confidence"],
            "chart_type":  None,
            "x_axis":      None,
            "y_axis":      None,
            "aggregation": "none",
            "title":       "",
            "explanation": f"Rule-based match: {rule_result['matches']}",
        }
        result = {
            "success":      True,
            "stage":        "rule",
            "parsed":       parsed,
            "model":        "rule_router",
            "error":        None,
            "user_message": None,
        }
        if print_to_terminal:
            _print_route(result)
        return result

    # ── Stage 2: LLM fallback for ambiguous queries ───────────────────────────
    candidate_hint = ""
    if rule_result["candidates"]:
        candidate_hint = (
            f"\nA preliminary rule scan flagged this question as possibly "
            f"matching MULTIPLE routes: {', '.join(rule_result['candidates'])}. "
            f"Choose the single best route from that shortlist unless neither "
            f"fits — in that case choose whichever route is correct.\n"
        )
    else:
        # Zero rule matches: the keyword scan found NO obvious signal for any
        # route. This does NOT mean the question is ambiguous or unanswerable
        # — it usually means the question is phrased in a way the keyword
        # patterns didn't anticipate. Most zero-match questions turn out to
        # be ordinary row/aggregate retrieval (sql_answer), so bias toward it
        # — but only when there's truly no visualization, metadata, or
        # statistical signal either. This keeps "plot it" routable as
        # visualization and "what insights can you give me" routable as
        # statistical, instead of forcing every zero-match case into
        # sql_answer regardless of actual content.
        candidate_hint = (
            f"\nA preliminary rule scan found no obvious keyword match for this "
            f"question. In the absence of a clear chart/plot/visualize request, "
            f"a clear schema/structure question, or a clear complex-math/"
            f"narrative-insight request, default to \"sql_answer\" — most "
            f"unmatched questions are ordinary row or aggregate retrieval "
            f"phrased in a way the keyword scan didn't anticipate.\n"
        )

    sample_section = ""
    if sample_rows:
        sample_section = (
            "\nSample Rows (first 3 rows — for column value context only):\n"
            + json.dumps(sample_rows, indent=2, default=str)
        )

    user_message = (
        f"Database Schema:\n{ddl_schema}\n"
        f"{sample_section}\n"
        f"Available Tables: {table_name}\n"
        f"{candidate_hint}\n"
        f"User Question: {user_prompt}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    logger.info(
        "Ambiguous (%s) — LLM routing — model=%s table=%s prompt='%s'",
        rule_result["candidates"], ROUTER_MODEL, table_name, user_prompt[:80],
    )

    try:
        raw_response = _call_ollama_sync(messages)
        raw_text     = raw_response["message"]["content"]
        parsed       = _parse_json_safe(raw_text)

        parsed["confidence"] = str(parsed.get("confidence", "MEDIUM")).upper()
        if parsed["confidence"] not in ("HIGH", "MEDIUM", "LOW"):
            parsed["confidence"] = "MEDIUM"

        # Validate route label — reject hallucinated labels
        valid_routes = {"visualization", "sql_answer", "metadata", "statistical"}
        if parsed.get("route") not in valid_routes:
            logger.warning(
                "LLM returned invalid route '%s' — falling back to sql_answer",
                parsed.get("route"),
            )
            parsed["route"]      = "sql_answer"
            parsed["confidence"] = "LOW"

        result = {
            "success":      True,
            "stage":        "llm",
            "parsed":       parsed,
            "model":        ROUTER_MODEL,
            "error":        None,
            "user_message": user_message,
        }
        if print_to_terminal:
            _print_route(result)
        return result

    except (ConnectionError, TimeoutError) as exc:
        msg = str(exc)
        logger.error(msg)
        result = {
            "success": False, "stage": "llm", "parsed": None,
            "model": ROUTER_MODEL, "error": msg, "user_message": user_message,
        }
        if print_to_terminal:
            _print_route(result)
        return result

    except Exception as exc:
        logger.error("LLM routing failed: %s", exc, exc_info=True)
        result = {
            "success": False, "stage": "llm", "parsed": None,
            "model": ROUTER_MODEL, "error": str(exc), "user_message": user_message,
        }
        if print_to_terminal:
            _print_route(result)
        return result