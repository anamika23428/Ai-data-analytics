# core/rule_router.py
#
# Deterministic, rule-based pre-filter for query routing.
#
# Design principles:
#   1. Match multi-word PHRASES and tightly-scoped terms — never a bare
#      single word that can show up incidentally in an unrelated sentence.
#   2. Score every route category instead of "first match wins". If more
#      than one category lights up, that IS the ambiguity signal — it's
#      handed to the LLM, not resolved by guessing.
#   3. Only return a confident decision when exactly one category matched.
#      Zero matches → LLM escalation (never silent default).
#      Multiple matches → LLM escalation with candidate hint —
#      EXCEPT when "visualization" is one of the candidates, since the
#      system prompt's rule #1 says visualization always overrides every
#      other route. That case is resolved here directly, with no LLM call.
#   4. Never guesses chart_type / x_axis / y_axis. Column selection is
#      left entirely to route_a.py's own schema-grounded intent stage.
#
# Route boundaries (must match llm_router.py's system prompt exactly) —
# FOUR routes total, not five:
#   Route A (visualization)  — user explicitly asks for a chart/plot/graph
#   Route B (sql_answer)     — row retrieval + simple aggregations
#                              (SUM, AVG, COUNT, MIN, MAX, GROUP BY, top-N)
#   Route C (metadata)       — pure structural enumeration: unique values,
#                              distinct counts, schema, column names, data types
#                              NO conditions, NO math
#   Route D (statistical)    — combined complex-analytics + reasoning route.
#                              Covers BOTH:
#                                (a) complex math — percentile, IQR, z-score,
#                                    correlation, stddev, trend analysis,
#                                    outliers, moving averages, regression,
#                                    window functions, AND
#                                (b) narrative reasoning — explanation,
#                                    business insight, "why", "summarize",
#                                    "what does this mean", recommendations.
#                              Internally still scored as two pattern groups
#                              ("statistical" and "reasoning" candidate
#                              labels below) so ambiguity-detection logic
#                              keeps working, but both groups are folded
#                              into the single "statistical" route label
#                              before being returned to the caller.

import re

# ── Route A: Visualization ────────────────────────────────────────────────────
_VISUALIZATION_PATTERNS = [
    r"\bchart\s+of\b",
    r"\bchart\s+the\b",
    r"\bbar\s*chart\b",
    r"\bpie\s*chart\b",
    r"\bline\s*chart\b",
    r"\bline\s*graph\b",
    r"\bscatter\s*plot\b",
    r"\bheat\s*map\b",
    r"\bbar\s*graph\b",
    r"\bpie\s*graph\b",
    r"\bhistogram\b",
    r"\bplot\s+(the|a|my|me)\b",
    r"\bgraph\s+(the|a|my|me)\b",
    r"\bvisuali[sz]e\b",
    r"\bshow\s+(me\s+)?a\s+(chart|graph|plot)\b",
    r"\bcreate\s+(a\s+)?(chart|graph|plot|visual)\b",
    r"\bdraw\s+(a\s+)?(chart|graph|plot)\b",
]

# ── Route B: SQL Answer (row retrieval + simple aggregation) ──────────────────
_SQL_ANSWER_PATTERNS = [
    # ── Row retrieval ──────────────────────────────────────────────────────────
    r"\bshow\s+(me\s+)?(all\s+)?(the\s+)?(rows?|records?|data|details?|entries|items?)\b",
    r"\bfind\s+(all\s+)?(the\s+)?(rows?|records?|orders?|customers?|users?|products?|entries|items?|people|employees?)\b",
    r"\bget\s+(me\s+)?(all\s+)?(the\s+)?(rows?|records?|data|details?|entries|items?)\b",
    r"\bgive\s+me\s+(all\s+)?(the\s+)?(rows?|records?|data|details?|list|entries|items?)\b",
    r"\bfetch\s+(all\s+)?(the\s+)?(rows?|records?|data|entries)\b",
    r"\bshow\s+(me\s+)?(all\s+)?(the\s+)?\w+\s+(where|with|that|whose|above|below|greater|less|between|from|in)\b",
    r"\blist\s+(all\s+)?(the\s+)?(?!unique|distinct|columns|tables|fields)\w+\s+(where|with|that|whose|who|above|below)\b",
    # "show/list/give me the names/list/details of [all] [unique] X who/that/whose <condition>"
    # Bridges the gap left by the single-\w+ patterns above when extra words
    # (e.g. "names of all", "list of") sit between the verb and the noun.
    # The relative clause (who/that/whose/...) is what makes this a filtered
    # retrieval — it must always win over a metadata "list unique X" read,
    # which only applies when NO condition follows.
    # In _SQL_ANSWER_PATTERNS, update the list pattern:
    r"\blist\s+(all\s+)?(the\s+)?(?!unique|distinct|columns|tables)\w+\s+"
    r"(where|with|that|whose|who|above|below|working|from|in|having|earning|named|called)\b",
    r"\blist\s+(all\s+)?employees\b",
    r"\blist\s+(all\s+)?customers\b",
    r"\blist\s+(all\s+)?products\b",
    r"\blist\s+(all\s+)?orders\b",
    r"\blist\s+(all\s+)?users\b",
    r"\balong\s+with\s+(their|the)\b",   # "along with their job titles" → retrieval
    r"\bworking\s+in\s+(the\s+)?\w+\s+department\b",
    r"\b(show|list|give|get|fetch)\s+(me\s+)?(the\s+)?(names?|list|details?|emails?|ids?)\s+of\s+(all\s+)?(the\s+)?(unique\s+)?\w+\s+(who|that|whose|which)\b",
    # "list/show/give [all] unique/distinct X who/that/which <condition>"
    # Same idea but without a leading "names/list/details of" — e.g.
    # "list unique users who left a review". The presence of unique/distinct
    # does NOT make this metadata once a relative clause filters the result;
    # it's a SELECT DISTINCT ... WHERE query, which is still Route B.
    r"\b(list|show|give|get|fetch)\s+(me\s+)?(all\s+)?(the\s+)?(unique|distinct)\s+\w+\s+(who|that|whose|which)\b",
    # Entity lookups — "who is/are/was/were" or "who has/have" but NOT
    # "who has the most/highest" (that's a ranking superlative, handled
    # separately by the "which X has the most/least" pattern below).
    r"\bwho\s+(is|are|was|were)\b",
    r"\bwho\s+(has|have)\b(?!\s+the\s+(most|least|highest|lowest|best|worst|maximum|minimum))",
    r"\bwhich\s+(specific\s+)?(person|employee|customer|user|product|order|item|record)s?\b",
    r"\bwhat\s+(is|are)\s+the\s+(email|name|id|address|status|phone|number|code|description)\b",
    r"\btell\s+me\s+about\s+(the\s+)?(specific|individual|particular)\b",
    # "Did a/the/any <noun> [named/called X] <verb> any <noun2>" — a named-
    # entity existence + retrieval question, e.g. "Did a user named 'Manish'
    # write any reviews? Show me the review titles if he did." This is a
    # WHERE name = 'X' filter, not a structural metadata existence check
    # (those ask whether a COLUMN exists, not whether a specific person's
    # ROWS exist) — without this pattern the whole question fell through
    # with zero candidates and was cold-routed to the LLM, which tends to
    # mistake the "did...any" existence phrasing for a metadata check.
    r"\bdid\s+(a|the|any)\s+\w+(\s+(named|called)\s+[\"']?[\w\s]+?[\"']?)?\s+\w+\s+any\b",
    # "show me the X [and Y] for anything/items/products/rows/entries with/
    # that/where/having <condition>" — a multi-column filtered retrieval,
    # e.g. "Show me the product names and actual prices for anything with a
    # discount percentage of more than 80%." The leading "show me the X and
    # Y" superficially resembles a visualization request ("show me X and Y"
    # plotted together), but the trailing "for <noun> with/that <condition>"
    # makes this a WHERE-filtered SELECT of multiple columns, not a chart —
    # there is no chart/graph/plot/visualize keyword anywhere in the prompt.
    r"\bshow\s+(me\s+)?(the\s+)?[\w\s,]+?\s+for\s+(anything|items?|products?|rows?|records?|entries|orders?|customers?|users?)\s+(with|that|where|having)\b",

    # ── Simple aggregations (Route B, NOT Route D) ────────────────────────────
    # Single-value aggregations
    r"\bwhat\s+is\s+the\s+(total|sum|average|avg|mean|max|maximum|min|minimum)\b",
    r"\bwhat\s+(is|are)\s+the\s+(highest|lowest|most|least|best|worst)\b",
    r"\bhow\s+many\s+(?!unique\b|distinct\b)\w",
    r"\bhow\s+much\s+(total\s+)?(revenue|sales?|profit|spend|cost|amount)\b",
    r"\bcount\s+(of\s+)?(all\s+)?(the\s+)?(rows?|records?|orders?|customers?|users?|products?|entries|items?)\b",

    # Group-level simple aggregations
    r"\btotal\s+(revenue|sales?|spend|orders?|count|amount|quantity|profit|cost|income)\b",
    r"\baverage\s+(rating|price|salary|age|score|amount|revenue|cost|discount|spend)\b",
    r"\bsum\s+of\b",
    r"\bbreakdown\s+of\b",
    r"\bwhich\s+.{0,50}\bhas\s+the\s+(most|least|highest|lowest|maximum|minimum)\b",
    # "who has/have the most/least/highest/lowest ..." — the superlative form
    # of an entity lookup. The plain "who has" pattern above deliberately
    # excludes this via its lookahead (to avoid double-firing on every
    # superlative), so it needs its own explicit pattern here instead of
    # falling through to zero candidates.
    r"\bwho\s+(has|have)\s+the\s+(most|least|highest|lowest|best|worst|maximum|minimum)\b",

    # Top-N / ranking (simple ORDER BY, not window-function ranking)
    r"\btop\s+\d+\s+(customers?|products?|orders?|employees?|cities|categories|items?|regions?|brands?|sellers?)\b",
    r"\bbottom\s+\d+\s+(customers?|products?|orders?|employees?|cities|categories|items?|regions?|brands?|sellers?)\b",
    r"\brank(ed|ing)?\s+(the\s+|all\s+|by\s+|products?|customers?|orders?|employees?|cities|categories)\b",
    r"\bsort(ed)?\s+by\b",
    r"\border(ed)?\s+by\b",

    # Comparison (simple, not statistical)
    r"\bcompare\s+.{0,50}\b(vs\.?|versus|against|and)\b",
    # "percentage/proportion/fraction/share of X (that are/were/have) Y" —
    # this is a COUNT(*) FILTER (...) / COUNT(*) ratio, a single SELECT with
    # no GROUP BY or window function needed. These words sound statistical
    # in isolation, but the underlying SQL is exactly as simple as a plain
    # COUNT — without this pattern the question zero-matches the rule layer
    # entirely and gets cold-routed to the LLM, which strongly over-
    # associates "percentage/proportion/fraction" with the statistical
    # route on vocabulary alone, even when no real statistics are involved.
    r"\b(percentage|proportion|fraction|share)\s+of\s+\w+",
    # "(what is the) range of <column>" — MIN(col) and MAX(col), two simple
    # aggregates in one SELECT. Distinct from "how does X vary/differ" (left
    # unmatched here on purpose — that phrasing is genuinely ambiguous
    # between "give me the min/max" and "give me the standard deviation",
    # and is better resolved by the LLM with full context than forced here).
    # Negative lookbehind excludes "interquartile range" — that specific
    # phrase is unambiguously statistical (IQR) and is already matched by
    # _STATISTICAL_PATTERNS; without this exclusion it would double-match
    # both routes and create a false ambiguity for a query that should
    # cleanly resolve to statistical.
    r"(?<!interquartile )\brange\s+of\s+\w+",
]

# ── Route C: Metadata (pure structural enumeration, NO conditions) ────────────
# A master negative lookahead that aborts the Route C match if ANY filtering,
# threshold, or conditional word appears later in the user's prompt.
# NOTE: "order" alone is intentionally excluded from the word list — it's a
# common domain noun (order status, order type, order ID) and would
# false-positive on clean metadata questions like "what are the distinct
# order statuses". Only the actual SQL clause "order by" should trigger
# the block, handled by the second lookahead below.
_NO_FILTER = (
    r"(?!.*\b(who|that|which|where|with|having|from|after|before|above|below|"
    r"greater|less|more|between|under|over|except|costing|priced|rated|bought|"
    r"sold|by|like|not|only|when|if|limit)\b)"
    r"(?!.*\border\s+by\b)"
)

# ── Route C: Metadata (pure structural enumeration, NO conditions) ────────────
_METADATA_PATTERNS = [
    # ── Schema / structure — unambiguously structural, never need SQL ──────
    r"\\bwhat\\s+columns\\b",
    r"\\bcolumn\\s+names\\b",
    r"\\blist\\s+(the\\s+)?columns\\b",
    r"\\bshow\\s+(me\\s+)?(the\\s+)?columns\\b",
    r"\\bdata\\s+types?\\b",
    r"\\bschema\\b",
    r"\\bdescribe\\s+the\\s+(table|dataset|data)\\b",
    r"\\btable\\s+structure\\b",
    r"\\bhow\\s+many\\s+columns\\b",
    r"\\bwhat\\s+tables\\b",
    r"\\bfield\\s+names\\b",
    r"\\bstructure\\s+of\\s+(the\\s+)?(data|table|dataset)\\b",
    r"\\bshow\\s+(me\\s+)?(the\\s+)?schema\\b",
    r"\\bwhat\\s+fields\\b",
    # ── Strict unconditional unique/distinct enumeration only ────────────
    # These require NO condition after them (_NO_FILTER enforces this).
    # "most/least common/frequent" and "top N popular" are removed —
    # those are ranked aggregations (Route B), not structural enumeration.
    rf"\\blist\\s+(all\\s+)?(unique|distinct)\\b{_NO_FILTER}",
    rf"\\bwhat\\s+are\\s+the\\s+(unique|distinct|possible|different)\\b{_NO_FILTER}",
    rf"\\bunique\\s+values?\\s+(in|of|for)\\b{_NO_FILTER}",
    rf"\\bdistinct\\s+values?\\s+(in|of|for)\\b{_NO_FILTER}",
    rf"\\bshow\\s+(all\\s+)?(unique|distinct)\\b{_NO_FILTER}",
    rf"\\bwhat\\s+(categories|types?|statuses?|options?)\\s+(exist|are\\s+(there|available|in))\\b{_NO_FILTER}",
    rf"\\bhow\\s+many\\s+(unique|distinct)\\b{_NO_FILTER}",
    r"\\ball\\s+(unique|distinct)\\s+\\w+\\s*$",
]
# ── Route D: Statistical (complex analytics only) ─────────────────────────────
_STATISTICAL_PATTERNS = [
    # Outlier / anomaly detection
    r"\boutliers?\b",
    r"\banomal(y|ies)\b",
    r"\bdeviat(e|es|ion|ions)\s+(from|above|below)\b",

    # Distribution / spread
    r"\bstandard\s+deviation\b", r"\bstddev\b",
    r"\bvariance\b",
    r"\bskew(ness)?\b",
    r"\bkurtosis\b",
    r"\bnormal\s+distribution\b",
    r"\bprobability\s+distribution\b",
    r"\bdistribution\s+of\b",
    r"\bfrequency\s+distribution\b",

    # Percentile / quantile
    r"\bpercentile\b",
    r"\bquartile\b",
    r"\binterquartile\b", r"\biqr\b",
    r"\bntile\b",
    r"\bpercent_rank\b",
    r"\bcumulative\s+(distribution|frequency|percent)\b",

    # Central tendency (non-trivial)
    r"\bmedian\b",
    r"\bweighted\s+average\b", r"\bweighted\s+mean\b",

    # Statistical scores
    r"\bz[\s-]?score\b",
    r"\bstandard\s+score\b",
    r"\bt[\s-]?test\b",
    r"\bchi[\s-]?square\b",

    # Relationships
    r"\bcorrelation\b",
    r"\bcovarian[ct]e\b",
    r"\bregression\b",
    r"\blinear\s+(model|fit|relationship)\b",

    # Time series / trend analysis
    r"\btrend\s+analysis\b",
    r"\btime[\s-]?series\b",
    r"\bmoving\s+average\b", r"\brolling\s+average\b",
    r"\brolling\s+(sum|count|min|max)\b",
    r"\bseasonalit(y|ies)\b",
    r"\bforecast(ing)?\b",
    r"\byear[\s-]?over[\s-]?year\b", r"\byoy\b",
    r"\bmonth[\s-]?over[\s-]?month\b", r"\bmom\b",

    # Window / ranking functions (advanced, not simple ORDER BY)
    r"\bwindow\s+function\b",
    r"\bdense_rank\b", r"\brow_number\b",
    r"\brank\s+within\b",
    r"\bcumulative\s+(sum|total|revenue|count)\b",
    r"\brunning\s+(total|sum|count)\b",

    # Confidence / significance
    r"\bconfidence\s+interval\b",
    r"\bstatistical(ly)?\s+significant\b",
    r"\bp[\s-]?value\b",
    r"\bmargin\s+of\s+error\b",

    # Misc advanced
    r"\bcluster(ing)?\b",
    r"\bsegment(ation)?\b",
    r"\bprincipal\s+component\b", r"\bpca\b",
    r"\bcohort\s+analysis\b",
    r"\bfunnel\s+analysis\b",
    r"\bretention\s+(rate|analysis)\b",
    r"\bchurn\s+(rate|analysis|prediction)\b",
]

# ── Route D (reasoning half): narrative insight / explanation ────────────────
# NOTE: this is NOT a separate route. It is one of the two pattern groups
# that both fold into the single "statistical" route label (= Route D) in
# classify() below. Kept as its own list purely so the ambiguity-detection
# logic can still tell "this looks like complex math" apart from "this looks
# like a narrative question" — useful for logging/debugging — without ever
# exposing a fifth route label to callers.
_REASONING_PATTERNS = [
    r"\bwhy\s+(is|did|does|are|was|were)\b",
    r"\bexplain\s+(why|how|what|the|this)\b",
    r"\bsummari[sz]e\b",
    r"\bsummary\s+of\b",
    r"\bkey\s+(insights?|takeaways?|findings?)\b",
    r"\bwhat\s+does\s+this\s+(mean|tell|show|suggest|indicate)\b",
    r"\binterpret\b",
    r"\bwhat\s+can\s+(you|we|i)\s+tell\s+(me\s+)?about\b",
    r"\bwhat\s+(conclusions?|inferences?)\b",
    r"\bwhat\s+is\s+(happening|going\s+on)\b",
    r"\bnarrative\b",
    r"\bstory\s+(behind|of|about)\b",
    r"\bbusiness\s+(insight|implication|meaning|impact)\b",
    r"\bwhat\s+should\s+(i|we|the\s+business)\b",
    r"\brecommend(ation)?\b",
]

# ── Compiled patterns ─────────────────────────────────────────────────────────
_COMPILED = {
    "visualization": [re.compile(p, re.I) for p in _VISUALIZATION_PATTERNS],
    "sql_answer":    [re.compile(p, re.I) for p in _SQL_ANSWER_PATTERNS],
    "metadata":      [re.compile(p, re.I) for p in _METADATA_PATTERNS],
    "statistical":   [re.compile(p, re.I) for p in _STATISTICAL_PATTERNS],
    "reasoning":     [re.compile(p, re.I) for p in _REASONING_PATTERNS],
}

# ── Grouped aggregation check ─────────────────────────────────────────────────
# "average X by Y", "total X per Y", "count X by Y" etc.
# These are simple GROUP BY queries → Route B, not Route D.
_AGG_BY_PATTERN = re.compile(
    r"\b(average|avg|total|sum|mean|count|max|min|maximum|minimum)\b"
    r".{0,50}?\b(?:by|per|across|for\s+each)\s+([a-zA-Z_][\w\s]{0,30}?)\b",
    re.I,
)


def _has_grouped_aggregation(prompt: str, known_columns: set[str] | None) -> bool:
    match = _AGG_BY_PATTERN.search(prompt)
    if not match:
        return False
    if not known_columns:
        return True
    grouping_phrase = match.group(2).lower().strip()
    return any(
        col.replace("_", " ") in grouping_phrase
        or grouping_phrase in col.replace("_", " ")
        for col in known_columns
    )


def classify(prompt: str, known_columns: set[str] | None = None) -> dict:
    """
    Score `prompt` against every route category's patterns.

    `known_columns` (optional): flattened set of column names across all
    loaded tables, lowercased. When supplied, sharpens the grouped-aggregation
    check against real column names instead of accepting on pattern alone.

    Returns:
        {
            "route":      <route label, or None if ambiguous>,
            "confidence": "HIGH" | "MEDIUM" | "LOW",
            "ambiguous":  bool,
            "candidates": [route labels with at least one match],
            "matches":    {route: [matched pattern strings]},
        }

    NOTE: only four route labels are ever returned — "visualization",
    "sql_answer", "metadata", "statistical". The "reasoning" pattern group
    is scored internally for debugging clarity but always folds into
    "statistical" before this function returns, since the actual system
    has no separate reasoning route.
    """
    matches: dict[str, list[str]] = {}

    for route, patterns in _COMPILED.items():
        hits = [p.pattern for p in patterns if p.search(prompt)]
        if hits:
            matches[route] = hits

    # "average X by Y" / "total X per Y" → Route B (simple GROUP BY)
    if _has_grouped_aggregation(prompt, known_columns):
        matches.setdefault("sql_answer", []).append("aggregation_by_per")

    # ── Fold "reasoning" into "statistical" — there is no separate Route E ───
    # The system has exactly four routes: visualization, sql_answer, metadata,
    # statistical. Complex-math patterns and narrative-reasoning patterns are
    # scored as two internal buckets above (so a debugger can see which kind
    # of statistical/reasoning language triggered the match), but only ever
    # surface as a single combined "statistical" route label here.
    if "reasoning" in matches:
        combined_hits = matches.pop("reasoning")
        matches.setdefault("statistical", []).extend(combined_hits)

    candidates = list(matches.keys())

    # ── Tie-breaker: visualization always wins ────────────────────────────────
    # System prompt rule #1: "show me a chart/bar chart/.../graph" ALWAYS
    # routes to visualization, overriding every other route. If the prompt
    # explicitly matched a visualization pattern, there's no real ambiguity
    # to resolve — encode the override here instead of paying an LLM round
    # trip for a decision the rules already make for free.
    if "visualization" in candidates:
        return {
            "route":      "visualization",
            "confidence": "HIGH",
            "ambiguous":  False,
            "candidates": candidates,
            "matches":    matches,
        }

    if len(candidates) == 1:
        return {
            "route":      candidates[0],
            "confidence": "HIGH",
            "ambiguous":  False,
            "candidates": candidates,
            "matches":    matches,
        }

    if not candidates:
        # No pattern matched — escalate to LLM rather than silently defaulting.
        return {
            "route":      None,
            "confidence": "LOW",
            "ambiguous":  True,
            "candidates": [],
            "matches":    matches,
        }

    # Two or more categories fired → genuine ambiguity, escalate to LLM.
    return {
        "route":      None,
        "confidence": "LOW",
        "ambiguous":  True,
        "candidates": candidates,
        "matches":    matches,
    }