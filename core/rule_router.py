# core/rule_router.py
#
# Deterministic, rule-based pre-filter for query routing.
#
# Design, replacing the old flat keyword-list approach:
#   1. Match multi-word PHRASES and tightly-scoped terms — never a bare
#      single word that can show up incidentally in an unrelated sentence.
#   2. Score every route category instead of "first match wins". If more
#      than one category lights up, that IS the ambiguity signal — it's
#      handed to the LLM, not resolved by guessing.
#   3. Only return a confident decision when exactly one category matched.
#      Zero matches defaults to sql_answer (the safe generic fallback).
#      Multiple matches go to the LLM, with candidates passed as a hint.
#   4. Never guesses chart_type / x_axis / y_axis. Column selection is
#      left entirely to route_a.py's own schema-grounded intent stage.

import re

_VISUALIZATION_PATTERNS = [
    r"\bbar\s*chart\b", r"\bpie\s*chart\b", r"\bline\s*chart\b",
    r"\bline\s*graph\b", r"\bscatter\s*plot\b", r"\bhistogram\b",
    r"\bheat\s*map\b", r"\bbar\s*graph\b", r"\bpie\s*graph\b",
    r"\bplot\s+(the|a|my|me)\b", r"\bgraph\s+(the|a|my|me)\b",
    r"\bvisuali[sz]e\b", r"\bchart\s+of\b", r"\bchart\s+the\b",
    r"\bshow\s+(me\s+)?a\s+(chart|graph|plot)\b",
    r"\btrend\s+(over\s+time|of)\b",
]

_METADATA_PATTERNS = [
    r"\bwhat\s+columns\b", r"\bcolumn\s+names\b", r"\blist\s+(the\s+)?columns\b",
    r"\bshow\s+(me\s+)?(the\s+)?columns\b", r"\bdata\s+types?\b",
    r"\bschema\b", r"\bdescribe\s+the\s+table\b", r"\btable\s+structure\b",
    r"\bhow\s+many\s+columns\b", r"\bwhat\s+tables\b", r"\bfield\s+names\b",
    r"\bstructure\s+of\s+(the\s+)?(data|table)\b",
    # NOTE: deliberately does NOT match "how many rows" — that's sql_answer,
    # per the existing system prompt's own rule.
]

_STATISTICAL_PATTERNS = [
    r"\boutliers?\b", r"\banomal(y|ies)\b", r"\bpercentile\b",
    r"\bz[\s-]?score\b", r"\bstandard\s+deviation\b", r"\bcorrelation\b",
    r"\bvariance\b", r"\bmedian\b", r"\bquartile\b", r"\bskew(ness)?\b",
]

_REASONING_PATTERNS = [
    r"\bwhy\s+(is|did|does|are)\b", r"\bexplain\b", r"\bsummari[sz]e\b",
    r"\bsummary\s+of\b", r"\binsight\b", r"\bwhat\s+does\s+this\s+mean\b",
    r"\binterpret\b", r"\bwhat\s+can\s+you\s+tell\s+me\s+about\b",
]

_COMPILED = {
    "visualization": [re.compile(p, re.I) for p in _VISUALIZATION_PATTERNS],
    "metadata":      [re.compile(p, re.I) for p in _METADATA_PATTERNS],
    "statistical":   [re.compile(p, re.I) for p in _STATISTICAL_PATTERNS],
    "reasoning":     [re.compile(p, re.I) for p in _REASONING_PATTERNS],
}

# Aggregation word followed (within a short window) by "by"/"per" is treated
# as a grouped-breakdown visualization request, matching the existing system
# prompt's stated philosophy ("average salary by department" -> visualization).
# This is handled separately from the main category loop because it benefits
# from an optional schema cross-check to cut down false positives (e.g.
# "what's the average rating, by the way" should NOT match).
_AGG_BY_PATTERN = re.compile(
    r"\b(average|avg|total|sum|mean|count)\b.{0,40}?\b(?:by|per)\s+"
    r"([a-zA-Z_][\w\s]{0,30}?)\b",
    re.I,
)


def _has_grouped_aggregation(prompt: str, known_columns: set[str] | None) -> bool:
    match = _AGG_BY_PATTERN.search(prompt)
    if not match:
        return False
    if not known_columns:
        # No schema available to cross-check against — accept on pattern alone.
        return True
    grouping_phrase = match.group(2).lower().strip()
    return any(
        col.replace("_", " ") in grouping_phrase or grouping_phrase in col.replace("_", " ")
        for col in known_columns
    )


def classify(prompt: str, known_columns: set[str] | None = None) -> dict:
    """
    Score `prompt` against every route category's patterns.

    `known_columns` (optional): flattened set of column names across all
    loaded tables, lowercased. When supplied, sharpens the aggregation
    "by/per" check against real grouping columns instead of accepting it
    on pattern alone.

    Returns:
        {
            "route":      <route label, or None if ambiguous>,
            "confidence": "HIGH" | "MEDIUM" | "LOW",
            "ambiguous":  bool,
            "candidates": [route labels with at least one match],
            "matches":    {route: [matched pattern strings]},
        }
    """
    matches: dict[str, list[str]] = {}

    for route, patterns in _COMPILED.items():
        hits = [p.pattern for p in patterns if p.search(prompt)]
        if hits:
            matches[route] = hits

    if _has_grouped_aggregation(prompt, known_columns):
        matches.setdefault("visualization", []).append("aggregation_by_per")

    candidates = list(matches.keys())

    if len(candidates) == 1:
        return {
            "route":      candidates[0],
            "confidence": "HIGH",
            "ambiguous":  False,
            "candidates": candidates,
            "matches":    matches,
        }

    if not candidates:
        # No signal at all -> default to the safe, generic fallback route
        # rather than spending an LLM call on it. This is a deliberate
        # speed/safety tradeoff: worst case is a table instead of a chart,
        # never a wrong or hallucinated answer.
        return {
            "route":      "sql_answer",
            "confidence": "MEDIUM",
            "ambiguous":  False,
            "candidates": [],
            "matches":    matches,
        }

    # Two or more categories fired -> genuine ambiguity.
    return {
        "route":      None,
        "confidence": "LOW",
        "ambiguous":  True,
        "candidates": candidates,
        "matches":    matches,
    }