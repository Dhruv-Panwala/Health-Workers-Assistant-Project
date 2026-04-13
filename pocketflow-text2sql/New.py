import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.call_llm import (
    CHAT_SYSTEM_PROMPT,
    call_chat_llm,
    chat_model_unavailable_reason,
    is_chat_model_available,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_ROOT / "dhis2.sqlite"

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

GENERIC_METRIC_TOKENS = {
    "case", "cases", "data", "dose", "doses", "given", "patient", "patients", "record",
    "records", "related", "result", "results", "show", "what", "chance", "chances",
    "risk", "risks", "probability", "probabilities", "odds", "likelihood",
    "difference", "different", "between", "compare", "comparison", "matter", "matters",
    "versus", "vs", "test", "tests",
    "treat", "treated", "treating", "treatment", "therapy", "therapies", "manage",
    "management", "medicine", "medicines", "drug", "drugs", "serious", "severe",
    "our", "your", "their", "this", "these", "those",
}

ORGUNIT_SUFFIX_TOKENS = {
    "center", "centre", "clinic", "facility", "hospital", "mchp", "phu", "unit",
}

EXPLAINABLE_STARTERS = (
    "what is",
    "what are",
    "what medicine",
    "what treatment",
    "what drug",
    "what does",
    "what do",
    "which medicine",
    "which treatment",
    "which drug",
    "who is",
    "how to",
    "how do",
    "how should",
    "how can",
    "tell me about",
    "explain",
    "describe",
    "why",
    "why is",
    "why was",
    "why were",
    "why did",
)

EXPLAINABLE_STOP_WORDS = {
    "a", "about", "an", "and", "are", "chance", "chances", "describe", "did", "do", "does",
    "explain", "give", "how", "in", "is", "likelihood", "me", "odds", "of", "patient", "patients",
    "probability", "risk", "tell", "the", "there", "was", "were", "what", "who", "why",
    "to",
}

COMPARISON_QUALIFIER_TOKENS = {
    "case", "cases", "death", "deaths", "dose", "doses", "negative", "positive", "result", "results",
}

COMPARISON_NOISE_TOKENS = {
    "case", "category", "categories",
    "align", "aligned", "alignment", "between", "compare", "compared", "comparison", "data",
    "difference", "different", "evidence", "line", "lines", "lineup", "matter", "matters", "mean",
    "means", "result", "show", "shows", "symptom", "symptoms", "sign", "signs", "test", "tests",
    "versus", "vs", "treat", "treatment", "therapy", "therapies",
}

COMPARISON_LEFT_BOUNDARY_TOKENS = EXPLAINABLE_STOP_WORDS | COMPARISON_NOISE_TOKENS | {
    "about", "our", "their", "this", "those", "your",
}

COMPARISON_RIGHT_BOUNDARY_TOKENS = EXPLAINABLE_STOP_WORDS | COMPARISON_NOISE_TOKENS | {
    "and", "because", "but", "during", "for", "from", "how", "if", "in", "that", "when", "where",
    "which", "while",
}

METRIC_REQUIRED_INTENTS = {"summary", "ranking", "peak", "comparison"}

FACT_TABLE_NAME = "assistant_fact_values"
MONTHLY_SUMMARY_TABLE_NAME = "assistant_monthly_summary"
EXPLAINABLE_CACHE_TABLE_NAME = "assistant_explainable_cache"
METRIC_LOOKUP_TABLE_NAME = "assistant_metric_lookup"
ORGUNIT_LOOKUP_TABLE_NAME = "assistant_orgunit_lookup"
RUNTIME_METADATA_TABLE_NAME = "assistant_runtime_metadata"

LEGACY_BASE_CTE_SQL = """
WITH base AS (
    SELECT
        dv.dataelementid,
        dv.sourceid,
        dv.periodid,
        TRIM(de.name) AS dataelement_name,
        TRIM(ou.name) AS orgunit_name,
        p.startdate,
        p.enddate,
        TRIM(COALESCE(pt.name, '')) AS period_type,
        dv.value,
        CASE
            WHEN REGEXP('^[-]?\\d+(\\.\\d+)?$', TRIM(CAST(dv.value AS TEXT)))
            THEN CAST(dv.value AS REAL)
        END AS value_num,
        CASE WHEN COALESCE(dv.followup, 0) <> 0 THEN 1 ELSE 0 END AS followup
    FROM datavalue AS dv
    JOIN dataelement AS de ON dv.dataelementid = de.dataelementid
    JOIN organisationunit AS ou ON dv.sourceid = ou.organisationunitid
    JOIN period AS p ON dv.periodid = p.periodid
    LEFT JOIN periodtype AS pt ON p.periodtypeid = pt.periodtypeid
)
""".strip()

OPTIMIZED_BASE_CTE_SQL = f"""
WITH base AS (
    SELECT
        dataelementid,
        sourceid,
        periodid,
        dataelement_name,
        orgunit_name,
        startdate,
        enddate,
        period_type,
        value,
        value_num,
        followup
    FROM {FACT_TABLE_NAME}
)
""".strip()


@dataclass
class QueryPlan:
    question: str
    intent: str
    orgunits: List[str] = field(default_factory=list)
    orgunit_ids: List[int] = field(default_factory=list)
    metric_groups: List[List[str]] = field(default_factory=list)
    metric_group_ids: List[List[int]] = field(default_factory=list)
    period_type: Optional[str] = None
    period_type_id: Optional[int] = None
    followup: Optional[bool] = None
    value_filter: Optional[Tuple[str, float, Optional[float]]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    order_direction: str = "DESC"
    limit: int = 100
    page: int = 1
    page_size: int = 100
    debug: bool = False


def serialize_query_plan(plan: QueryPlan) -> Dict[str, Any]:
    return make_json_safe(asdict(plan))


def deserialize_query_plan(payload: Dict[str, Any]) -> QueryPlan:
    return QueryPlan(
        question=str(payload.get("question") or ""),
        intent=str(payload.get("intent") or "records"),
        orgunits=[str(value) for value in (payload.get("orgunits") or [])],
        orgunit_ids=[int(value) for value in (payload.get("orgunit_ids") or [])],
        metric_groups=[
            [str(token) for token in group]
            for group in (payload.get("metric_groups") or [])
        ],
        metric_group_ids=[
            [int(value) for value in group]
            for group in (payload.get("metric_group_ids") or [])
        ],
        period_type=str(payload["period_type"]) if payload.get("period_type") is not None else None,
        period_type_id=int(payload["period_type_id"]) if payload.get("period_type_id") is not None else None,
        followup=bool(payload["followup"]) if payload.get("followup") is not None else None,
        value_filter=tuple(payload["value_filter"]) if payload.get("value_filter") else None,
        start_date=str(payload["start_date"]) if payload.get("start_date") is not None else None,
        end_date=str(payload["end_date"]) if payload.get("end_date") is not None else None,
        order_direction=str(payload.get("order_direction") or "DESC"),
        limit=int(payload.get("limit") or payload.get("page_size") or 100),
        page=max(1, int(payload.get("page") or 1)),
        page_size=max(1, int(payload.get("page_size") or 100)),
        debug=bool(payload.get("debug", False)),
    )


def build_resolved_plan(kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": kind, **make_json_safe(payload)}


def normalize_resolved_plan_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        cleaned = payload.strip()
        if not cleaned:
            return None
        return json.loads(cleaned)
    raise ValueError("resolved_plan must be a JSON object or JSON string")


def sqlite_regexp(pattern: str, value: Any) -> int:
    if value is None:
        return 0
    try:
        return 1 if re.search(pattern, str(value)) else 0
    except re.error:
        return 0


def get_sqlite_connection(db_path: str) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.create_function("REGEXP", 2, sqlite_regexp)
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def runtime_table_exists(db_path: str, table_name: str) -> bool:
    with get_sqlite_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type IN ('table', 'view') AND name = ?
            """,
            (table_name,),
        ).fetchone()
    return bool(row)


@lru_cache(maxsize=32)
def runtime_metadata(db_path: str) -> Dict[str, str]:
    if not runtime_table_exists(db_path, RUNTIME_METADATA_TABLE_NAME):
        return {}
    with get_sqlite_connection(db_path) as connection:
        rows = connection.execute(
            f"SELECT key, value FROM {RUNTIME_METADATA_TABLE_NAME}"
        ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def can_use_fact_table_for_plan(db_path: str, plan: "QueryPlan") -> bool:
    if not runtime_table_exists(db_path, FACT_TABLE_NAME):
        return False

    metadata = runtime_metadata(db_path)
    mode = metadata.get("fact_table_mode", "").strip().lower()
    if mode in {"", "full"}:
        return True
    if mode != "partial_recent":
        return False

    start_boundary = metadata.get("fact_table_start_date", "").strip()
    if not start_boundary:
        return False

    # Preserve correctness: only use the partial fact table when the query
    # is explicitly bounded to the covered period range.
    if not plan.start_date:
        return False
    return plan.start_date >= start_boundary


def build_base_cte_sql(db_path: str, plan: Optional["QueryPlan"] = None) -> str:
    if plan is not None and can_use_fact_table_for_plan(db_path, plan):
        return OPTIMIZED_BASE_CTE_SQL
    if plan is None and runtime_table_exists(db_path, FACT_TABLE_NAME):
        return OPTIMIZED_BASE_CTE_SQL
    return LEGACY_BASE_CTE_SQL


def is_month_aligned_boundary(iso_value: Optional[str]) -> bool:
    if not iso_value:
        return True
    try:
        return date.fromisoformat(iso_value).day == 1
    except ValueError:
        return False


def should_use_monthly_summary(db_path: str, plan: "QueryPlan") -> bool:
    if not runtime_table_exists(db_path, MONTHLY_SUMMARY_TABLE_NAME):
        return False
    if plan.value_filter:
        return False
    if not is_month_aligned_boundary(plan.start_date):
        return False
    if not is_month_aligned_boundary(plan.end_date):
        return False
    return True


def build_conn_str_from_parts() -> str:
    configured = os.environ.get("SQLITE_DB_PATH", "").strip()
    if configured:
        return configured
    return str(DEFAULT_DB_PATH)


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def record_timing(timings: Optional[Dict[str, float]], key: str, started_at: float) -> None:
    if timings is None:
        return
    timings[key] = elapsed_ms(started_at)


def attach_debug_trace_timings(result: Dict[str, Any], timings: Optional[Dict[str, float]]) -> Dict[str, Any]:
    if not timings:
        return result

    debug_trace = result.get("debug_trace")
    if not isinstance(debug_trace, dict):
        debug_trace = {}
        result["debug_trace"] = debug_trace

    trace_timings = debug_trace.get("timings")
    if not isinstance(trace_timings, dict):
        trace_timings = {}
        debug_trace["timings"] = trace_timings

    trace_timings.update(make_json_safe(timings))
    return result


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def tokenize(text: str) -> List[str]:
    return [token for token in normalize_text(text).split() if token]


def singularize(token: str) -> str:
    if token in {"die", "dies", "died", "dying", "fatal", "fatality", "fatalities", "mortality"}:
        return "death"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def add_months(year: int, month: int, offset: int) -> Tuple[int, int]:
    month_index = (year * 12 + (month - 1)) + offset
    new_year, new_month_index = divmod(month_index, 12)
    return new_year, new_month_index + 1


def month_number(token: str) -> Optional[int]:
    return MONTHS.get(normalize_text(token))


def month_start(year: int, month: int) -> date:
    return date(year, month, 1)


def to_iso(value: Optional[date]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()


def contains_normalized_phrase(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return re.search(rf"(?:^| ){re.escape(needle)}(?:$| )", haystack) is not None


def parse_date_range(question: str) -> Tuple[Optional[str], Optional[str], str]:
    q = question.lower()
    order_direction = "DESC"
    if re.search(r"\b(earliest|oldest|ascending)\b", q):
        order_direction = "ASC"

    exact_day_range = re.search(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+of\s+([a-z]+)\s+to\s+(\d{1,2})(?:st|nd|rd|th)?\s+of\s+([a-z]+)\s+(20\d{2})",
        q,
    )
    if exact_day_range:
        start_day, start_month_text, end_day, end_month_text, year_text = exact_day_range.groups()
        start_month = month_number(start_month_text)
        end_month = month_number(end_month_text)
        if start_month and end_month:
            start_value = date(int(year_text), start_month, int(start_day))
            end_value = date(int(year_text), end_month, int(end_day))
            return to_iso(start_value), to_iso(end_value.fromordinal(end_value.toordinal() + 1)), order_direction

    month_range = re.search(r"between\s+([a-z]+)\s+and\s+([a-z]+)\s+(20\d{2})", q)
    if month_range:
        left_month, right_month, year_text = month_range.groups()
        left_number = month_number(left_month)
        right_number = month_number(right_month)
        if left_number and right_number:
            start_value = month_start(int(year_text), left_number)
            next_year, next_month = add_months(int(year_text), right_number, 1)
            end_value = month_start(next_year, next_month)
            return to_iso(start_value), to_iso(end_value), order_direction

    first_months = re.search(r"first\s+(\d+)\s+months?\s+of\s+(20\d{2})", q)
    if first_months:
        month_count = max(1, min(12, int(first_months.group(1))))
        year_value = int(first_months.group(2))
        start_value = month_start(year_value, 1)
        end_year, end_month = add_months(year_value, 1, month_count)
        return to_iso(start_value), to_iso(month_start(end_year, end_month)), order_direction

    last_months = re.search(r"last\s+(\d+)\s+months?\s+of\s+(20\d{2})", q)
    if last_months:
        month_count = max(1, min(12, int(last_months.group(1))))
        year_value = int(last_months.group(2))
        start_month_value = 13 - month_count
        start_value = month_start(year_value, start_month_value)
        end_value = month_start(year_value + 1, 1)
        return to_iso(start_value), to_iso(end_value), order_direction

    last_few_months = re.search(r"last\s+few\s+months?\s+of\s+(20\d{2})", q)
    if last_few_months:
        year_value = int(last_few_months.group(1))
        start_value = month_start(year_value, 10)
        end_value = month_start(year_value + 1, 1)
        return to_iso(start_value), to_iso(end_value), order_direction

    year_between = re.search(r"\bbetween\s+(20\d{2})\s+and\s+(20\d{2})\b", q)
    if year_between:
        start_year, end_year = sorted([int(year_between.group(1)), int(year_between.group(2))])
        start_value = month_start(start_year, 1)
        end_value = month_start(end_year + 1, 1)
        return to_iso(start_value), to_iso(end_value), order_direction

    year_range = re.search(r"\b(20\d{2})\s*(?:to|-|and)\s*(20\d{2})\b", q)
    if year_range:
        start_year, end_year = sorted([int(year_range.group(1)), int(year_range.group(2))])
        start_value = month_start(start_year, 1)
        end_value = month_start(end_year + 1, 1)
        return to_iso(start_value), to_iso(end_value), order_direction

    single_year = re.search(r"\b(20\d{2})\b", q)
    if single_year:
        year_value = int(single_year.group(1))
        start_value = month_start(year_value, 1)
        end_value = month_start(year_value + 1, 1)
        return to_iso(start_value), to_iso(end_value), order_direction

    return None, None, order_direction


def extract_value_filter(question: str) -> Optional[Tuple[str, float, Optional[float]]]:
    q = question.lower()

    between_match = re.search(
        r"value\s+(?:between|from)\s+(-?\d+(?:\.\d+)?)\s+(?:and|to)\s+(-?\d+(?:\.\d+)?)",
        q,
    )
    if between_match:
        left_value = float(between_match.group(1))
        right_value = float(between_match.group(2))
        return ("between", min(left_value, right_value), max(left_value, right_value))

    comparator_patterns = [
        (r"value\s+(?:more than|greater than|above|over)\s+(-?\d+(?:\.\d+)?)", ">"),
        (r"value\s+(?:less than|below|under)\s+(-?\d+(?:\.\d+)?)", "<"),
        (r"value\s+(?:at least|minimum of)\s+(-?\d+(?:\.\d+)?)", ">="),
        (r"value\s+(?:at most|maximum of)\s+(-?\d+(?:\.\d+)?)", "<="),
        (r"value\s+(?:equal to|equals|is)\s+(-?\d+(?:\.\d+)?)", "="),
        (r"(?:with|has|having)\s+value\s+(-?\d+(?:\.\d+)?)", "="),
    ]

    for pattern, operator in comparator_patterns:
        match = re.search(pattern, q)
        if match:
            return (operator, float(match.group(1)), None)

    return None


def extract_followup(question: str) -> Optional[bool]:
    q = normalize_text(question)
    if "follow up" not in q and "followup" not in q:
        return None

    if re.search(r"follow(?:\s|-)?up\s+(?:is\s+)?(?:0|false|no)\b", q):
        return False
    if re.search(r"(?:0|false|no)\s+follow(?:\s|-)?up\b", q):
        return False
    if re.search(r"without\s+follow(?:\s|-)?up\b", q):
        return False
    if re.search(r"follow(?:\s|-)?up\s+(?:is\s+)?(?:1|true|yes)\b", q):
        return True
    if re.search(r"(?:1|true|yes)\s+follow(?:\s|-)?up\b", q):
        return True

    return None


@lru_cache(maxsize=4)
def get_orgunit_names(db_path: str) -> Tuple[str, ...]:
    table_name = ORGUNIT_LOOKUP_TABLE_NAME if runtime_table_exists(db_path, ORGUNIT_LOOKUP_TABLE_NAME) else "organisationunit"
    with get_sqlite_connection(db_path) as connection:
        if table_name == ORGUNIT_LOOKUP_TABLE_NAME:
            rows = connection.execute(
                f"SELECT DISTINCT name FROM {ORGUNIT_LOOKUP_TABLE_NAME} WHERE trim(coalesce(name, '')) <> '' ORDER BY name"
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT DISTINCT TRIM(name) FROM organisationunit WHERE trim(coalesce(name, '')) <> '' ORDER BY TRIM(name)"
            ).fetchall()
    return tuple(row[0] for row in rows)


@lru_cache(maxsize=4)
def get_period_types(db_path: str) -> Tuple[str, ...]:
    with get_sqlite_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT DISTINCT TRIM(name) FROM periodtype WHERE trim(coalesce(name, '')) <> '' ORDER BY TRIM(name)"
        ).fetchall()
    return tuple(row[0] for row in rows)


@lru_cache(maxsize=64)
def resolve_orgunit_ids_cached(db_path: str, orgunits: Tuple[str, ...]) -> Tuple[int, ...]:
    cleaned_orgunits = [orgunit.strip().lower() for orgunit in orgunits if orgunit and orgunit.strip()]
    if not cleaned_orgunits:
        return ()

    placeholders = ", ".join("?" for _ in cleaned_orgunits)
    table_name = ORGUNIT_LOOKUP_TABLE_NAME if runtime_table_exists(db_path, ORGUNIT_LOOKUP_TABLE_NAME) else "organisationunit"
    id_column = "organisationunitid"
    name_column = "normalized_name" if table_name == ORGUNIT_LOOKUP_TABLE_NAME else "lower(TRIM(name))"
    with get_sqlite_connection(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT {id_column}
            FROM {table_name}
            WHERE {name_column} IN ({placeholders})
            ORDER BY {id_column}
            """,
            cleaned_orgunits,
        ).fetchall()
    return tuple(int(row[0]) for row in rows)


@lru_cache(maxsize=32)
def resolve_period_type_id_cached(db_path: str, cleaned_period_type: str) -> Optional[int]:
    if not cleaned_period_type:
        return None

    with get_sqlite_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT periodtypeid
            FROM periodtype
            WHERE lower(TRIM(name)) = ?
            LIMIT 1
            """,
            (cleaned_period_type,),
        ).fetchone()
    return int(row[0]) if row else None


@lru_cache(maxsize=256)
def query_dataelement_match_count_cached(db_path: str, tokens: Tuple[str, ...]) -> int:
    if not tokens:
        return 0

    table_name = METRIC_LOOKUP_TABLE_NAME if runtime_table_exists(db_path, METRIC_LOOKUP_TABLE_NAME) else "dataelement"
    search_expression = "search_blob" if table_name == METRIC_LOOKUP_TABLE_NAME else "lower(name)"
    clauses = [f"{search_expression} LIKE ?" for _ in tokens]
    params = [f"%{token}%" for token in tokens]

    with get_sqlite_connection(db_path) as connection:
        row = connection.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()
    return int(row[0]) if row else 0


@lru_cache(maxsize=128)
def resolve_metric_group_ids_cached(db_path: str, metric_groups: Tuple[Tuple[str, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
    resolved_groups: List[Tuple[int, ...]] = []
    table_name = METRIC_LOOKUP_TABLE_NAME if runtime_table_exists(db_path, METRIC_LOOKUP_TABLE_NAME) else "dataelement"
    id_column = "dataelementid"
    search_expression = "search_blob" if table_name == METRIC_LOOKUP_TABLE_NAME else "lower(TRIM(name))"
    with get_sqlite_connection(db_path) as connection:
        for group in metric_groups:
            cleaned_group = [token for token in dedupe_tokens(group) if token]
            if not cleaned_group:
                resolved_groups.append(())
                continue

            clauses = [f"{search_expression} LIKE ?" for _ in cleaned_group]
            params = [f"%{token}%" for token in cleaned_group]
            rows = connection.execute(
                f"""
                SELECT DISTINCT {id_column}
                FROM {table_name}
                WHERE {' AND '.join(clauses)}
                ORDER BY {id_column}
                """,
                params,
            ).fetchall()
            resolved_groups.append(tuple(int(row[0]) for row in rows))
    return tuple(resolved_groups)


def orgunit_significant_tokens(orgunit_name: str) -> List[str]:
    name_tokens = tokenize(orgunit_name)
    significant_tokens = [token for token in name_tokens if token not in ORGUNIT_SUFFIX_TOKENS]
    return significant_tokens or name_tokens


def match_orgunits(question: str, db_path: str) -> List[str]:
    normalized_question = normalize_text(question)
    question_tokens = set(tokenize(question))
    phrase_matches: List[Tuple[int, str, Tuple[str, ...]]] = []
    matches: List[Tuple[int, str]] = []

    for orgunit_name in get_orgunit_names(db_path):
        normalized_name = normalize_text(orgunit_name)
        name_tokens = tokenize(orgunit_name)
        significant_tokens = orgunit_significant_tokens(orgunit_name)

        score = 0
        if normalized_name and contains_normalized_phrase(normalized_question, normalized_name):
            score = 100 + len(significant_tokens)
            phrase_matches.append((score, orgunit_name, tuple(significant_tokens)))
        elif len(significant_tokens) >= 2 and all(token in question_tokens for token in significant_tokens):
            score = 85 + len(significant_tokens)
        else:
            overlap_count = sum(token in question_tokens for token in significant_tokens)
            if overlap_count >= 2:
                score = 50 + overlap_count
            elif overlap_count == 1 and len(significant_tokens) == 1 and len(name_tokens) == 1:
                score = 45

        if score > 0:
            matches.append((score, orgunit_name))

    if phrase_matches:
        phrase_matches.sort(key=lambda item: (-item[0], item[1]))
        filtered_phrase_matches: List[Tuple[int, str, Tuple[str, ...]]] = []
        for score, orgunit_name, token_group in phrase_matches:
            token_set = set(token_group)
            if any(token_set < set(existing_group) for _, _, existing_group in filtered_phrase_matches):
                continue
            filtered_phrase_matches.append((score, orgunit_name, token_group))
        return [orgunit_name for _, orgunit_name, _ in filtered_phrase_matches]

    matches.sort(key=lambda item: (-item[0], item[1]))
    if matches and matches[0][0] >= 80:
        best_score = matches[0][0]
        return [name for score, name in matches if score == best_score]
    return [matches[0][1]] if matches else []


def match_period_type(question: str, db_path: str) -> Optional[str]:
    normalized_question = normalize_text(question)
    if "period type" not in normalized_question and not re.search(r"\bmonthly\b|\bweekly\b|\byearly\b|\bquarterly\b", normalized_question):
        return None

    for period_type in get_period_types(db_path):
        if normalize_text(period_type) in normalized_question:
            return period_type
    return None


def resolve_orgunit_ids(db_path: str, orgunits: Sequence[str]) -> List[int]:
    return list(resolve_orgunit_ids_cached(db_path, tuple(orgunits)))


def resolve_period_type_id(db_path: str, period_type: Optional[str]) -> Optional[int]:
    cleaned_period_type = (period_type or "").strip().lower()
    return resolve_period_type_id_cached(db_path, cleaned_period_type)


def resolve_metric_group_ids(db_path: str, metric_groups: Sequence[Sequence[str]]) -> List[List[int]]:
    cached = resolve_metric_group_ids_cached(
        db_path,
        tuple(tuple(token for token in group) for group in metric_groups),
    )
    return [list(group) for group in cached]


def query_dataelement_match_count(db_path: str, tokens: Sequence[str]) -> int:
    return query_dataelement_match_count_cached(db_path, tuple(tokens))


def dedupe_tokens(tokens: Sequence[str]) -> List[str]:
    cleaned_tokens: List[str] = []
    seen_tokens = set()
    for token in tokens:
        singular_token = singularize(token)
        if not singular_token or singular_token in seen_tokens:
            continue
        cleaned_tokens.append(singular_token)
        seen_tokens.add(singular_token)
    return cleaned_tokens


def resolve_metric_tokens(db_path: str, tokens: Sequence[str]) -> List[str]:
    normalized_tokens = dedupe_tokens([token for token in tokens if token])
    if not normalized_tokens:
        return []

    if query_dataelement_match_count(db_path, normalized_tokens) > 0:
        return normalized_tokens

    relaxed_tokens = list(normalized_tokens)
    removable_tokens = {singularize(token) for token in GENERIC_METRIC_TOKENS}
    prioritized_pools = [
        [token for token in relaxed_tokens if token not in removable_tokens],
        relaxed_tokens,
    ]

    for pool in prioritized_pools:
        candidate_pool = dedupe_tokens(pool)
        if not candidate_pool:
            continue

        best_group: List[str] = []
        best_count = 0
        max_size = min(len(candidate_pool), 4)
        for group_size in range(max_size, 0, -1):
            for token_group in combinations(candidate_pool, group_size):
                match_count = query_dataelement_match_count(db_path, token_group)
                if match_count <= 0:
                    continue
                if not best_group or group_size > len(best_group) or (
                    group_size == len(best_group) and match_count > best_count
                ):
                    best_group = list(token_group)
                    best_count = match_count
            if best_group:
                return best_group

    fallback_tokens = [token for token in relaxed_tokens if token not in removable_tokens] or relaxed_tokens
    if len(fallback_tokens) == 1:
        return fallback_tokens

    best_single = max(
        fallback_tokens,
        key=lambda token: (query_dataelement_match_count(db_path, [token]), -fallback_tokens.index(token)),
    )
    return [best_single]


def spans_overlap(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def extract_metric_groups(question: str, db_path: str) -> List[List[str]]:
    lower_question = question.lower()
    matched_spans: List[Tuple[int, int]] = []
    metric_groups: List[List[str]] = []

    patterns = [
        (r"\bmalaria\s+positive\b", ["malaria", "positive"]),
        (r"\bmalaria\s+negative\b", ["malaria", "negative"]),
        (r"\bdeaths?\s+(?:due\s+to\s+)?malaria\b", ["malaria", "death"]),
        (r"\bmalaria\s+deaths?\b", ["malaria", "death"]),
        (r"\bmeasles\s+doses?\b", ["measles", "dose"]),
        (r"\btt3\s+doses?\b", ["tt3", "dose"]),
        (r"\bbcg\s+doses?\b", ["bcg", "dose"]),
        (r"\bmalaria\s+cases?\b", ["malaria", "case"]),
        (r"\bmeasles\s+cases?\b", ["measles", "case"]),
    ]

    for pattern, tokens in patterns:
        for match in re.finditer(pattern, lower_question):
            span = match.span()
            if any(spans_overlap(span, existing_span) for existing_span in matched_spans):
                continue
            matched_spans.append(span)
            metric_groups.append(resolve_metric_tokens(db_path, tokens))

    generic_tokens = ["malaria", "measles", "tt3", "bcg"]
    qualifier_tokens = ["positive", "negative", "death", "deaths", "dose", "doses", "case", "cases"]

    for generic_token in generic_tokens:
        for match in re.finditer(rf"\b{generic_token}\b", lower_question):
            span = match.span()
            if any(spans_overlap(span, existing_span) for existing_span in matched_spans):
                continue

            group_tokens = [generic_token]
            window = lower_question[max(0, span[0] - 24): min(len(lower_question), span[1] + 24)]
            for qualifier_token in qualifier_tokens:
                if re.search(rf"\b{qualifier_token}\b", window):
                    group_tokens.append(qualifier_token)
                    break

            resolved_tokens = resolve_metric_tokens(db_path, group_tokens)
            if resolved_tokens:
                metric_groups.append(resolved_tokens)
                matched_spans.append(span)

    deduped_groups: List[List[str]] = []
    seen_signatures = set()
    for group in metric_groups:
        signature = tuple(group)
        if not group or signature in seen_signatures:
            continue
        deduped_groups.append(group)
        seen_signatures.add(signature)

    return deduped_groups


def sanitize_comparison_phrase_tokens(tokens: Sequence[str]) -> List[str]:
    cleaned_tokens = [
        singularize(token)
        for token in tokens
        if token
        and not token.isdigit()
        and singularize(token) not in EXPLAINABLE_STOP_WORDS
        and singularize(token) not in COMPARISON_NOISE_TOKENS
        and singularize(token) not in {"our", "their", "this", "those", "your"}
    ]
    cleaned_tokens = dedupe_tokens(cleaned_tokens)
    if cleaned_tokens:
        return cleaned_tokens

    fallback_tokens = [
        singularize(token)
        for token in tokens
        if token and not token.isdigit() and singularize(token) not in EXPLAINABLE_STOP_WORDS
    ]
    return dedupe_tokens(fallback_tokens)


def collect_left_comparison_tokens(tokens: Sequence[str], start_index: int) -> List[str]:
    collected: List[str] = []
    cursor = start_index
    while cursor >= 0:
        singular_token = singularize(tokens[cursor])
        if singular_token in COMPARISON_LEFT_BOUNDARY_TOKENS:
            break
        collected.insert(0, tokens[cursor])
        cursor -= 1
    return collected


def collect_right_comparison_tokens(tokens: Sequence[str], start_index: int) -> List[str]:
    collected: List[str] = []
    cursor = start_index
    while cursor < len(tokens):
        singular_token = singularize(tokens[cursor])
        if singular_token in COMPARISON_RIGHT_BOUNDARY_TOKENS:
            break
        collected.append(tokens[cursor])
        cursor += 1
    return collected


def inherit_comparison_subject(tokens: Sequence[str], shared_subject: Sequence[str]) -> List[str]:
    cleaned_tokens = dedupe_tokens(tokens)
    if not cleaned_tokens:
        return []
    if shared_subject and all(token in COMPARISON_QUALIFIER_TOKENS for token in cleaned_tokens):
        return dedupe_tokens([*shared_subject, *cleaned_tokens])
    return cleaned_tokens


def extract_comparison_target_groups(
    question: str,
    db_path: Optional[str] = None,
    seed_groups: Optional[Sequence[Sequence[str]]] = None,
) -> List[List[str]]:
    if not is_difference_question(question):
        return []

    tokens = tokenize(question)
    raw_pairs: List[Tuple[List[str], List[str]]] = []

    for index, token in enumerate(tokens):
        if token != "between":
            continue
        and_index = next((cursor for cursor in range(index + 1, len(tokens)) if tokens[cursor] == "and"), None)
        if and_index and and_index > index + 1:
            raw_pairs.append((list(tokens[index + 1:and_index]), collect_right_comparison_tokens(tokens, and_index + 1)))
            break

    if not raw_pairs:
        for connector in ("versus", "vs"):
            if connector not in tokens:
                continue
            connector_index = tokens.index(connector)
            raw_pairs.append(
                (
                    collect_left_comparison_tokens(tokens, connector_index - 1),
                    collect_right_comparison_tokens(tokens, connector_index + 1),
                )
            )
            break

    if not raw_pairs and "compare" in tokens:
        compare_index = tokens.index("compare")
        connector_index = next(
            (
                cursor
                for cursor in range(compare_index + 1, len(tokens))
                if tokens[cursor] in {"and", "with", "to", "versus", "vs"}
            ),
            None,
        )
        if connector_index and connector_index > compare_index + 1:
            raw_pairs.append(
                (
                    list(tokens[compare_index + 1:connector_index]),
                    collect_right_comparison_tokens(tokens, connector_index + 1),
                )
            )

    groups: List[List[str]] = []
    for raw_left, raw_right in raw_pairs:
        left_tokens = sanitize_comparison_phrase_tokens(raw_left)
        right_tokens = sanitize_comparison_phrase_tokens(raw_right)

        left_subject = [token for token in left_tokens if token not in COMPARISON_QUALIFIER_TOKENS]
        right_subject = [token for token in right_tokens if token not in COMPARISON_QUALIFIER_TOKENS]

        if left_subject and not right_subject:
            right_tokens = inherit_comparison_subject(right_tokens, left_subject)
        elif right_subject and not left_subject:
            left_tokens = inherit_comparison_subject(left_tokens, right_subject)

        for token_group in (left_tokens, right_tokens):
            if not token_group:
                continue
            resolved_group = resolve_metric_tokens(db_path, token_group) if db_path else dedupe_tokens(token_group)
            if resolved_group:
                groups.append(resolved_group)

    if len(groups) < 2 and seed_groups:
        groups.extend([list(group) for group in seed_groups if group])

    deduped_groups = dedupe_token_groups(groups)
    return deduped_groups[:2]


def detect_intent(question: str) -> str:
    q = question.lower().strip()

    if re.search(r"\bdifference\b|\bcompare\b|\bversus\b|\bvs\b", q):
        return "comparison"
    if re.search(r"\bwhen\b", q) and re.search(r"\b(peak|highest|max|maximum|most)\b", q):
        return "peak"
    if re.search(r"\bwhich\b", q) and re.search(r"\b(organisation|organization)\b", q) and re.search(r"\b(most|least|highest|lowest)\b", q):
        return "ranking"
    if re.search(r"\bwhich\s+organisation\b|\bwhich\s+organization\b", q):
        return "ranking"
    if re.search(r"\b(how many|total|sum|overall|number of|count)\b", q):
        return "summary"
    return "records"


def is_orgunit_metadata_question(question: str) -> bool:
    normalized_question = normalize_text(question)
    mentions_orgunit = any(
        phrase in normalized_question
        for phrase in ("organisation unit", "organisation units", "organization unit", "organization units", "facility", "facilities")
    )
    asks_orgunit_metadata = any(
        term in normalized_question
        for term in ("opened", "opening date", "openingdate", "closed", "closed date", "closeddate", "hierarchy level", "hierarchylevel", "code", "codes")
    )
    return mentions_orgunit and asks_orgunit_metadata


def build_query_plan(
    question: str,
    db_path: str,
    page: int,
    page_size: int,
    row_limit: int,
    debug: bool,
    timings: Optional[Dict[str, float]] = None,
) -> QueryPlan:
    date_parse_started_at = perf_counter()
    start_date, end_date, order_direction = parse_date_range(question)
    record_timing(timings, "date_parse_ms", date_parse_started_at)

    period_match_started_at = perf_counter()
    period_type = match_period_type(question, db_path)
    record_timing(timings, "period_type_match_ms", period_match_started_at)

    followup_parse_started_at = perf_counter()
    followup = extract_followup(question)
    value_filter = extract_value_filter(question)
    record_timing(timings, "filter_parse_ms", followup_parse_started_at)

    metric_extract_started_at = perf_counter()
    metric_groups = [group for group in extract_metric_groups(question, db_path) if group]
    record_timing(timings, "metric_group_extract_ms", metric_extract_started_at)

    orgunit_match_started_at = perf_counter()
    orgunits = match_orgunits(question, db_path)
    record_timing(timings, "orgunit_match_ms", orgunit_match_started_at)

    metric_id_resolution_started_at = perf_counter()
    metric_group_ids = resolve_metric_group_ids(db_path, metric_groups)
    record_timing(timings, "metric_id_resolution_ms", metric_id_resolution_started_at)

    orgunit_id_resolution_started_at = perf_counter()
    orgunit_ids = resolve_orgunit_ids(db_path, orgunits)
    record_timing(timings, "orgunit_id_resolution_ms", orgunit_id_resolution_started_at)

    period_type_id_started_at = perf_counter()
    period_type_id = resolve_period_type_id(db_path, period_type)
    record_timing(timings, "period_type_id_resolution_ms", period_type_id_started_at)

    limit = max(1, min(row_limit, page_size))
    intent = detect_intent(question)

    if intent == "comparison":
        comparison_group_started_at = perf_counter()
        comparison_groups = extract_comparison_target_groups(question, db_path, seed_groups=metric_groups)
        record_timing(timings, "comparison_group_extract_ms", comparison_group_started_at)
        if comparison_groups:
            metric_groups = [group for group in comparison_groups if group]
            metric_rebuild_started_at = perf_counter()
            metric_group_ids = resolve_metric_group_ids(db_path, metric_groups)
            record_timing(timings, "comparison_metric_id_resolution_ms", metric_rebuild_started_at)

    return QueryPlan(
        question=question,
        intent=intent,
        orgunits=orgunits,
        orgunit_ids=orgunit_ids,
        metric_groups=metric_groups,
        metric_group_ids=metric_group_ids,
        period_type=period_type,
        period_type_id=period_type_id,
        followup=followup,
        value_filter=value_filter,
        start_date=start_date,
        end_date=end_date,
        order_direction=order_direction,
        limit=limit,
        page=max(1, page),
        page_size=limit,
        debug=debug,
    )


def is_explainable_question(question: str) -> bool:
    normalized_question = normalize_text(question)
    if re.match(r"^(what|which) ", normalized_question) and re.search(r"\b(medicine|treatment|drug|therapy|therapies)\b", normalized_question):
        return True
    return any(normalized_question.startswith(starter) for starter in EXPLAINABLE_STARTERS)


def resolved_standard_plan_payload(plan: QueryPlan) -> Dict[str, Any]:
    return build_resolved_plan("standard", {"plan": serialize_query_plan(plan)})


def resolved_explainable_plan_payload(plan: QueryPlan) -> Dict[str, Any]:
    return build_resolved_plan("explainable", {"plan": serialize_query_plan(plan)})


def resolved_orgunit_plan_payload(question: str) -> Dict[str, Any]:
    return build_resolved_plan("orgunit_metadata", {"question": question})


def resolve_runtime_plan(
    question: str,
    db_path: str,
    page: int,
    page_size: int,
    row_limit: int,
    debug: bool,
    timings: Optional[Dict[str, float]] = None,
    resolved_plan: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[QueryPlan], Optional[Dict[str, Any]], str]:
    normalized_resolved_plan = normalize_resolved_plan_payload(resolved_plan)
    if normalized_resolved_plan:
        resolved_kind = str(normalized_resolved_plan.get("kind") or "").strip().lower()
        if resolved_kind == "standard":
            cached_plan = deserialize_query_plan(normalized_resolved_plan.get("plan") or {})
            plan = replace(
                cached_plan,
                page=max(1, page),
                page_size=max(1, min(row_limit, page_size)),
                limit=max(1, min(row_limit, page_size)),
                debug=debug,
            )
            return "standard", plan, normalized_resolved_plan, question or plan.question
        if resolved_kind == "explainable":
            cached_plan = deserialize_query_plan(normalized_resolved_plan.get("plan") or {})
            plan = replace(
                cached_plan,
                page=max(1, page),
                page_size=max(1, min(row_limit, page_size)),
                limit=max(1, min(row_limit, page_size)),
                debug=debug,
            )
            return "explainable", plan, normalized_resolved_plan, question or plan.question
        if resolved_kind == "orgunit_metadata":
            resolved_question = str(normalized_resolved_plan.get("question") or question or "").strip()
            return "orgunit_metadata", None, normalized_resolved_plan, resolved_question

    resolved_question = (question or "").strip()
    if is_explainable_question(resolved_question):
        planning_started_at = perf_counter()
        plan = build_query_plan(
            question=resolved_question,
            db_path=db_path,
            page=page,
            page_size=page_size,
            row_limit=row_limit,
            debug=debug,
            timings=timings,
        )
        record_timing(timings, "query_planning_ms", planning_started_at)
        return "explainable", plan, resolved_explainable_plan_payload(plan), resolved_question

    if is_orgunit_metadata_question(resolved_question):
        return "orgunit_metadata", None, resolved_orgunit_plan_payload(resolved_question), resolved_question

    planning_started_at = perf_counter()
    plan = build_query_plan(
        question=resolved_question,
        db_path=db_path,
        page=page,
        page_size=page_size,
        row_limit=row_limit,
        debug=debug,
        timings=timings,
    )
    record_timing(timings, "query_planning_ms", planning_started_at)
    return "standard", plan, resolved_standard_plan_payload(plan), resolved_question


def dedupe_token_groups(groups: Sequence[Sequence[str]]) -> List[List[str]]:
    cleaned_groups: List[List[str]] = []
    seen_signatures = set()
    for group in groups:
        cleaned_group: List[str] = []
        seen_tokens = set()
        for token in group:
            if not token or token in seen_tokens:
                continue
            cleaned_group.append(token)
            seen_tokens.add(token)
        signature = tuple(cleaned_group)
        if not cleaned_group or signature in seen_signatures:
            continue
        cleaned_groups.append(cleaned_group)
        seen_signatures.add(signature)

    return cleaned_groups


def drop_subset_token_groups(groups: Sequence[Sequence[str]]) -> List[List[str]]:
    cleaned_groups = dedupe_token_groups(groups)

    filtered_groups: List[List[str]] = []
    group_sets = [set(group) for group in cleaned_groups]
    for index, group in enumerate(cleaned_groups):
        group_set = group_sets[index]
        if any(group_set < other_group_set for other_group_set in group_sets):
            continue
        filtered_groups.append(group)
    return filtered_groups


def expand_explainable_comparison_groups(
    question: str,
    db_path: str,
    seed_groups: Sequence[Sequence[str]],
) -> List[List[str]]:
    return extract_comparison_target_groups(question, db_path, seed_groups=seed_groups)


def extract_explainable_token_groups(
    question: str,
    db_path: str,
    seed_groups: Optional[Sequence[Sequence[str]]] = None,
) -> List[List[str]]:
    token_groups = [list(group) for group in (seed_groups or []) if group]
    token_groups.extend(extract_metric_groups(question, db_path))
    token_groups.extend(expand_explainable_comparison_groups(question, db_path, token_groups))
    removable_tokens = EXPLAINABLE_STOP_WORDS | {singularize(token) for token in GENERIC_METRIC_TOKENS}

    topic_tokens = [
        singularize(token)
        for token in tokenize(question)
        if singularize(token) not in removable_tokens and not token.isdigit()
    ]
    if topic_tokens:
        if len(topic_tokens) > 1:
            token_groups.append(topic_tokens[:3])
        resolved_tokens = resolve_metric_tokens(db_path, topic_tokens)
        if resolved_tokens:
            token_groups.append(resolved_tokens)
        else:
            token_groups.append(topic_tokens[:3])

    token_groups = dedupe_token_groups(token_groups)
    matching_groups = [group for group in token_groups if query_dataelement_match_count(db_path, group) > 0]
    if matching_groups:
        return drop_subset_token_groups(matching_groups)
    return drop_subset_token_groups(token_groups)


def build_dataelement_keyword_clause(token_groups: Sequence[Sequence[str]]) -> Tuple[Optional[str], List[Any]]:
    if not token_groups:
        return None, []

    group_clauses = []
    params: List[Any] = []
    searchable_fields = (
        "lower(TRIM(COALESCE(de.name, '')))",
        "lower(TRIM(COALESCE(de.shortname, '')))",
        "lower(TRIM(COALESCE(de.code, '')))",
        "lower(TRIM(COALESCE(de.description, '')))",
    )

    for group in token_groups:
        token_clauses = []
        for token in group:
            field_clauses = [f"{field} LIKE ?" for field in searchable_fields]
            token_clauses.append("(" + " OR ".join(field_clauses) + ")")
            params.extend([f"%{token}%"] * len(searchable_fields))
        group_clauses.append("(" + " AND ".join(token_clauses) + ")")

    return "(" + " OR ".join(group_clauses) + ")", params


def can_use_explainable_cache(plan: QueryPlan, db_path: str) -> bool:
    if not runtime_table_exists(db_path, EXPLAINABLE_CACHE_TABLE_NAME):
        return False
    if plan.orgunits or plan.orgunit_ids:
        return False
    if plan.period_type or plan.period_type_id is not None:
        return False
    if plan.followup is not None or plan.value_filter is not None:
        return False
    if plan.start_date or plan.end_date:
        return False
    return True


def build_explainable_cache_keyword_clause(token_groups: Sequence[Sequence[str]]) -> Tuple[Optional[str], List[Any]]:
    if not token_groups:
        return None, []

    searchable_fields = (
        "lower(TRIM(COALESCE(dataelement_name, '')))",
        "lower(TRIM(COALESCE(shortname, '')))",
        "lower(TRIM(COALESCE(code, '')))",
        "lower(TRIM(COALESCE(description, '')))",
        "lower(TRIM(COALESCE(latest_text_value, '')))",
        "lower(TRIM(COALESCE(latest_comment, '')))",
    )
    group_clauses = []
    params: List[Any] = []

    for group in token_groups:
        token_clauses = []
        for token in group:
            field_clauses = [f"{field} LIKE ?" for field in searchable_fields]
            token_clauses.append("(" + " OR ".join(field_clauses) + ")")
            params.extend([f"%{token}%"] * len(searchable_fields))
        group_clauses.append("(" + " AND ".join(token_clauses) + ")")

    return "(" + " OR ".join(group_clauses) + ")", params


def build_cached_explainable_evidence_query(
    plan: QueryPlan,
    token_groups: Sequence[Sequence[str]],
) -> Tuple[str, List[Any], str]:
    keyword_clause, keyword_params = build_explainable_cache_keyword_clause(token_groups)
    if not keyword_clause:
        return "", [], "none"

    limit = 10
    sql = f"""
SELECT
    dataelement_name AS name,
    description,
    valuetype,
    aggregationtype,
    data_points,
    total_value,
    latest_value,
    latest_period_start,
    latest_period_end,
    first_period_start,
    last_period_end,
    latest_text_value,
    latest_comment
FROM {EXPLAINABLE_CACHE_TABLE_NAME}
WHERE {keyword_clause}
ORDER BY
    CASE
        WHEN TRIM(COALESCE(description, '')) <> '' THEN 0
        WHEN TRIM(COALESCE(latest_text_value, '')) <> '' THEN 1
        ELSE 2
    END,
    COALESCE(total_value, 0) DESC,
    COALESCE(data_points, 0) DESC,
    dataelement_name ASC
LIMIT {limit}
""".strip()
    return sql, keyword_params, "cache"


def is_definition_style_explainable_question(question: str) -> bool:
    normalized_question = normalize_text(question)
    starters = (
        "what is",
        "what are",
        "what does",
        "what do",
        "who is",
        "tell me about",
        "explain",
        "describe",
        "why",
        "why is",
        "why was",
        "why were",
        "why did",
    )
    return any(normalized_question.startswith(starter) for starter in starters)


def should_use_text_first_explainable_query(plan: QueryPlan) -> bool:
    if is_difference_question(plan.question):
        return False
    if plan.orgunits or plan.period_type:
        return False
    if plan.followup is not None or plan.value_filter is not None:
        return False
    if plan.start_date or plan.end_date:
        return False
    return is_definition_style_explainable_question(plan.question)


def build_explainable_scope_sql(plan: QueryPlan) -> Tuple[str, str, List[Any]]:
    join_clauses: List[str] = []
    filters: List[str] = []
    params: List[Any] = []

    if plan.orgunit_ids:
        orgunit_placeholders = ", ".join("?" for _ in plan.orgunit_ids)
        filters.append(f"dv.sourceid IN ({orgunit_placeholders})")
        params.extend(plan.orgunit_ids)
    elif plan.orgunits:
        join_clauses.append("JOIN organisationunit AS ou ON dv.sourceid = ou.organisationunitid")
        orgunit_clause = " OR ".join("TRIM(ou.name) = ? COLLATE NOCASE" for _ in plan.orgunits)
        filters.append("(" + orgunit_clause + ")")
        params.extend(plan.orgunits)

    if plan.period_type_id is not None:
        filters.append("p.periodtypeid = ?")
        params.append(plan.period_type_id)
    elif plan.period_type:
        join_clauses.append("LEFT JOIN periodtype AS pt ON p.periodtypeid = pt.periodtypeid")
        filters.append("TRIM(COALESCE(pt.name, '')) = ? COLLATE NOCASE")
        params.append(plan.period_type)

    if plan.start_date:
        filters.append("p.startdate >= ?")
        params.append(plan.start_date)
    if plan.end_date:
        filters.append("p.startdate < ?")
        params.append(plan.end_date)
    if plan.followup is not None:
        filters.append("COALESCE(dv.followup, 0) = ?")
        params.append(1 if plan.followup else 0)

    return "\n".join(join_clauses), " AND ".join(filters) if filters else "1=1", params


def build_explainable_numeric_filter_sql(plan: QueryPlan) -> Tuple[str, List[Any]]:
    if not plan.value_filter:
        return "1=1", []

    operator, left_value, right_value = plan.value_filter
    filters = ["numeric_value IS NOT NULL"]
    params: List[Any] = []
    if operator == "between" and right_value is not None:
        filters.append("numeric_value BETWEEN ? AND ?")
        params.extend([left_value, right_value])
    else:
        filters.append(f"numeric_value {operator} ?")
        params.append(left_value)

    return " AND ".join(filters), params


def build_numeric_text_explainable_evidence_query(
    plan: QueryPlan,
    keyword_clause: str,
    keyword_params: Sequence[Any],
) -> Tuple[str, List[Any]]:
    scoped_join_sql, scoped_where_sql, scoped_params = build_explainable_scope_sql(plan)
    numeric_filter_sql, numeric_filter_params = build_explainable_numeric_filter_sql(plan)
    evidence_limit = 12 if is_difference_question(plan.question) else 6
    candidate_limit = 16 if is_difference_question(plan.question) else 8
    numeric_value_expression = (
        "CASE "
        "WHEN REGEXP('^[-]?\\d+(\\.\\d+)?$', TRIM(CAST(dv.value AS TEXT))) "
        "THEN CAST(dv.value AS REAL) "
        "END"
    )

    sql = f"""
WITH matched_elements AS (
    SELECT
        de.dataelementid,
        TRIM(de.name) AS name,
        TRIM(COALESCE(de.description, '')) AS description,
        TRIM(COALESCE(de.valuetype, '')) AS valuetype,
        TRIM(COALESCE(de.aggregationtype, '')) AS aggregationtype
    FROM dataelement AS de
    WHERE {keyword_clause}
),
ranked_elements AS (
    SELECT
        dataelementid,
        name,
        description,
        valuetype,
        aggregationtype
    FROM matched_elements
    ORDER BY
        CASE
            WHEN TRIM(COALESCE(description, '')) <> '' THEN 0
            WHEN lower(name) LIKE '%narrative%' THEN 1
            ELSE 2
        END,
        name ASC
    LIMIT {candidate_limit}
),
typed_values AS (
    SELECT
        dv.dataelementid,
        {numeric_value_expression} AS numeric_value,
        NULLIF(TRIM(CAST(dv.value AS TEXT)), '') AS raw_text_value,
        NULLIF(TRIM(COALESCE(dv.comment, '')), '') AS comment_value,
        p.startdate,
        p.enddate,
        dv.lastupdated,
        dv.created
    FROM ranked_elements AS me
    CROSS JOIN datavalue AS dv INDEXED BY idx_datavalue_dataelementid
    JOIN period AS p ON dv.periodid = p.periodid
    {scoped_join_sql}
    WHERE dv.dataelementid = me.dataelementid
      AND {scoped_where_sql}
),
filtered_values AS (
    SELECT
        dataelementid,
        numeric_value,
        CASE
            WHEN numeric_value IS NULL THEN raw_text_value
            ELSE ''
        END AS text_value,
        comment_value,
        startdate,
        enddate,
        ROW_NUMBER() OVER (
            PARTITION BY dataelementid
            ORDER BY startdate DESC, enddate DESC, lastupdated DESC, created DESC
        ) AS recency_rank
    FROM typed_values
    WHERE {numeric_filter_sql}
),
numeric_rollup AS (
    SELECT
        dataelementid,
        COUNT(numeric_value) AS data_points,
        COALESCE(SUM(numeric_value), 0) AS total_value,
        MAX(CASE WHEN recency_rank = 1 THEN numeric_value END) AS latest_value,
        MAX(CASE WHEN recency_rank = 1 THEN startdate END) AS latest_period_start,
        MAX(CASE WHEN recency_rank = 1 THEN enddate END) AS latest_period_end,
        MIN(startdate) AS first_period_start,
        MAX(enddate) AS last_period_end
    FROM filtered_values
    GROUP BY dataelementid
),
text_rollup AS (
    SELECT
        dataelementid,
        MAX(CASE WHEN recency_rank = 1 THEN text_value END) AS latest_text_value,
        MAX(CASE WHEN recency_rank = 1 THEN comment_value END) AS latest_comment
    FROM filtered_values
    GROUP BY dataelementid
)
SELECT
    me.name,
    me.description,
    me.valuetype,
    me.aggregationtype,
    nr.data_points,
    nr.total_value,
    nr.latest_value,
    nr.latest_period_start,
    nr.latest_period_end,
    nr.first_period_start,
    nr.last_period_end,
    tr.latest_text_value,
    tr.latest_comment
FROM ranked_elements AS me
LEFT JOIN numeric_rollup AS nr ON nr.dataelementid = me.dataelementid
LEFT JOIN text_rollup AS tr ON tr.dataelementid = me.dataelementid
ORDER BY
    CASE
        WHEN TRIM(COALESCE(me.description, '')) <> '' THEN 0
        WHEN TRIM(COALESCE(tr.latest_text_value, '')) <> '' THEN 1
        WHEN lower(me.name) LIKE '%narrative%' THEN 2
        ELSE 3
    END,
    COALESCE(nr.total_value, 0) DESC,
    COALESCE(nr.data_points, 0) DESC,
    me.name ASC
LIMIT {evidence_limit}
""".strip()

    return sql, [*keyword_params, *scoped_params, *numeric_filter_params]


def build_text_first_explainable_evidence_query(
    plan: QueryPlan,
    keyword_clause: str,
    keyword_params: Sequence[Any],
) -> Tuple[str, List[Any]]:
    scoped_join_sql, scoped_where_sql, scoped_params = build_explainable_scope_sql(plan)

    sql = f"""
WITH matched_elements AS (
    SELECT
        de.dataelementid,
        TRIM(de.name) AS name,
        TRIM(COALESCE(de.description, '')) AS description,
        TRIM(COALESCE(de.valuetype, '')) AS valuetype,
        TRIM(COALESCE(de.aggregationtype, '')) AS aggregationtype
    FROM dataelement AS de
    WHERE {keyword_clause}
),
ranked_elements AS (
    SELECT
        dataelementid,
        name,
        description,
        valuetype,
        aggregationtype
    FROM matched_elements
    ORDER BY
        CASE
            WHEN TRIM(COALESCE(description, '')) <> '' THEN 0
            WHEN lower(valuetype) LIKE '%text%' THEN 1
            WHEN lower(name) LIKE '%narrative%' THEN 2
            ELSE 3
        END,
        name ASC
    LIMIT 6
),
text_candidate_elements AS (
    SELECT
        dataelementid
    FROM ranked_elements
    WHERE
        TRIM(COALESCE(description, '')) = ''
        OR lower(valuetype) LIKE '%text%'
        OR lower(name) LIKE '%narrative%'
),
textual_values AS (
    SELECT
        dv.dataelementid,
        NULLIF(TRIM(CAST(dv.value AS TEXT)), '') AS text_value,
        NULLIF(TRIM(COALESCE(dv.comment, '')), '') AS comment_value,
        p.startdate,
        p.enddate,
        dv.lastupdated,
        dv.created,
        ROW_NUMBER() OVER (
            PARTITION BY dv.dataelementid
            ORDER BY p.startdate DESC, p.enddate DESC, dv.lastupdated DESC, dv.created DESC
        ) AS recency_rank
    FROM text_candidate_elements AS tce
    CROSS JOIN datavalue AS dv INDEXED BY idx_datavalue_dataelementid
    JOIN period AS p ON dv.periodid = p.periodid
    {scoped_join_sql}
    WHERE dv.dataelementid = tce.dataelementid
      AND {scoped_where_sql}
      AND (
        TRIM(COALESCE(dv.comment, '')) <> ''
        OR TRIM(CAST(dv.value AS TEXT)) GLOB '*[A-Za-z]*'
      )
),
period_rollup AS (
    SELECT
        dataelementid,
        MAX(CASE WHEN recency_rank = 1 THEN startdate END) AS latest_period_start,
        MAX(CASE WHEN recency_rank = 1 THEN enddate END) AS latest_period_end,
        MIN(startdate) AS first_period_start,
        MAX(enddate) AS last_period_end
    FROM textual_values
    GROUP BY dataelementid
),
text_rollup AS (
    SELECT
        dataelementid,
        MAX(CASE WHEN recency_rank = 1 THEN text_value END) AS latest_text_value,
        MAX(CASE WHEN recency_rank = 1 THEN comment_value END) AS latest_comment
    FROM textual_values
    GROUP BY dataelementid
)
SELECT
    me.name,
    me.description,
    me.valuetype,
    me.aggregationtype,
    NULL AS data_points,
    NULL AS total_value,
    NULL AS latest_value,
    pr.latest_period_start,
    pr.latest_period_end,
    pr.first_period_start,
    pr.last_period_end,
    tr.latest_text_value,
    tr.latest_comment
FROM ranked_elements AS me
LEFT JOIN period_rollup AS pr ON pr.dataelementid = me.dataelementid
LEFT JOIN text_rollup AS tr ON tr.dataelementid = me.dataelementid
ORDER BY
    CASE
        WHEN TRIM(COALESCE(me.description, '')) <> '' THEN 0
        WHEN TRIM(COALESCE(tr.latest_text_value, '')) <> '' THEN 1
        WHEN lower(me.name) LIKE '%narrative%' THEN 2
        ELSE 3
    END,
    me.name ASC
LIMIT 6
""".strip()

    return sql, [*keyword_params, *scoped_params]


def build_explainable_evidence_query(
    plan: QueryPlan,
    db_path: str,
    token_groups: Sequence[Sequence[str]],
) -> Tuple[str, List[Any], str]:
    if can_use_explainable_cache(plan, db_path):
        sql, params, query_mode = build_cached_explainable_evidence_query(plan, token_groups)
        if sql:
            return sql, params, query_mode
    keyword_clause, keyword_params = build_dataelement_keyword_clause(token_groups)
    if not keyword_clause:
        return "", [], "none"

    if should_use_text_first_explainable_query(plan):
        sql, params = build_text_first_explainable_evidence_query(plan, keyword_clause, keyword_params)
        return sql, params, "text_first"

    sql, params = build_numeric_text_explainable_evidence_query(plan, keyword_clause, keyword_params)
    return sql, params, "numeric_text"


def fetch_explainable_evidence(
    plan: QueryPlan,
    db_path: str,
    timings: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], str, List[Any], str]:
    token_groups = extract_explainable_token_groups(plan.question, db_path, seed_groups=plan.metric_groups)
    sql, params, query_mode = build_explainable_evidence_query(plan, db_path, token_groups)
    if not sql:
        return [], "", [], query_mode

    sql_started_at = perf_counter()
    columns, rows = run_sql(db_path, sql, params)
    record_timing(timings, "explainable_sql_execution_ms", sql_started_at)
    evidence_rows: List[Dict[str, Any]] = []
    for row in rows:
        evidence_rows.append({column: format_number(row_value(row, column)) for column in columns})
    return select_explainable_evidence_rows(plan.question, evidence_rows, limit=10), sql, list(params), query_mode


def extract_explainable_topic(question: str) -> str:
    cleaned_question = question.strip().rstrip(" ?")
    match = re.match(
        r"(?i)^\s*(what is|what are|what medicine|what treatment|what drug|what does|what do|which medicine|which treatment|which drug|who is|how to|how do|how should|how can|tell me about|explain|describe|why(?:\s+(?:is|was|were|did))?)\s+(.*)$",
        cleaned_question,
    )
    if match:
        topic = match.group(2).strip()
        topic = re.sub(r"(?i)^there\s+", "", topic)
        topic = re.sub(r"(?i)^(?:a|an|the)\s+", "", topic)
        if re.match(r"(?i)^\s*(what medicine|what treatment|what drug|which medicine|which treatment|which drug)\b", cleaned_question):
            topic = re.sub(r"(?i)^(?:to\s+)?(?:give|use|for)\s+", "", topic)
        if re.match(r"(?i)^\s*(how to|how do|how should|how can)\b", cleaned_question):
            topic = re.sub(r"(?i)^(?:treat|manage|cure)\s+", "", topic)
        return topic.strip()
    return cleaned_question


def clean_explainable_text(text: Any) -> str:
    cleaned = re.sub(r"\[[0-9]+\]", "", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def explainable_text_excerpt(text: Any, max_chars: int = 220) -> str:
    cleaned = clean_explainable_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    shortened = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return (shortened or cleaned[:max_chars]).rstrip(".,;:") + "..."


def summarize_explainable_text(text: Any, max_sentences: int = 3, max_chars: int = 420) -> str:
    cleaned = clean_explainable_text(text)
    if not cleaned:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    selected: List[str] = []
    total_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        projected_length = total_length + len(sentence) + (1 if selected else 0)
        if selected and projected_length > max_chars:
            break
        selected.append(sentence)
        total_length = projected_length
        if len(selected) >= max_sentences:
            break

    if selected:
        return " ".join(selected)
    if len(cleaned) <= max_chars:
        return cleaned
    return explainable_text_excerpt(cleaned, max_chars=max_chars)


def explainable_row_text(row: Dict[str, Any]) -> str:
    return (
        clean_explainable_text(row.get("description"))
        or clean_explainable_text(row.get("latest_text_value"))
        or clean_explainable_text(row.get("latest_comment"))
    )


def is_substantive_explainable_text(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    if normalized in {"follow up", "followup", "narrative"}:
        return False
    return len(text) >= 40


def row_has_substantive_text(row: Dict[str, Any]) -> bool:
    narrative_text = clean_explainable_text(row.get("latest_text_value"))
    if is_substantive_explainable_text(narrative_text):
        return True

    comment_text = clean_explainable_text(row.get("latest_comment"))
    if is_substantive_explainable_text(comment_text):
        return True

    description_text = clean_explainable_text(row.get("description"))
    valuetype = normalize_text(str(row.get("valuetype", "")))
    name = normalize_text(str(row.get("name", "")))
    if not is_substantive_explainable_text(description_text):
        return False
    return "narrative" in name or "text" in valuetype


def select_textual_evidence_row(
    evidence_rows: Sequence[Dict[str, Any]],
    question: str = "",
) -> Optional[Dict[str, Any]]:
    normalized_question = normalize_text(question)
    scored_rows: List[Tuple[int, Dict[str, Any]]] = []
    for row in evidence_rows:
        if not row_has_substantive_text(row):
            continue

        text_value = clean_explainable_text(row.get("latest_text_value"))
        description = clean_explainable_text(row.get("description"))
        row_blob = explainable_row_blob(row)

        score = 0
        if is_substantive_explainable_text(description):
            score += 3
        if is_substantive_explainable_text(text_value):
            score += 2
        if "narrative" in normalize_text(str(row.get("name", ""))):
            score += 2
        if is_treatment_question(question):
            if any(term in row_blob for term in ("treat", "treatment", "therapy", "act", "artemisinin", "medicine", "drug")):
                score += 6
        if is_symptom_question(question):
            if any(term in row_blob for term in ("symptom", "fever", "headache", "vomiting", "chills", "nausea")):
                score += 6
        if is_probability_question(question):
            if any(term in row_blob for term in ("risk", "probability", "chance", "odds", "likelihood")):
                score += 4
        if normalized_question and any(token in row_blob for token in explainable_eval_tokens(question)[:4]):
            score += 2
        scored_rows.append((score, row))

    if not scored_rows:
        return None

    scored_rows.sort(
        key=lambda item: (
            -item[0],
            -float(item[1].get("total_value") or 0),
            -int(item[1].get("data_points") or 0),
            str(item[1].get("name") or ""),
        )
    )
    return scored_rows[0][1]


def explainable_row_blob(row: Dict[str, Any]) -> str:
    return normalize_text(
        " ".join(
            str(row.get(field) or "")
            for field in ("name", "description", "latest_text_value", "latest_comment")
        )
    )


def row_matches_token_group(row: Dict[str, Any], token_group: Sequence[str]) -> bool:
    row_blob = explainable_row_blob(row)
    return all(contains_normalized_phrase(row_blob, singularize(token)) for token in token_group if token)


def prioritize_explainable_evidence_rows(question: str, evidence_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_question = normalize_text(question)
    text_row = select_textual_evidence_row(evidence_rows, question=question)
    comparison_groups = extract_comparison_target_groups(question)
    polarity_comparison = (
        len(comparison_groups) >= 2
        and any("positive" in group for group in comparison_groups[:2])
        and any("negative" in group for group in comparison_groups[:2])
    )

    def score(row: Dict[str, Any]) -> Tuple[int, float, int, str]:
        points = 0
        if text_row and row.get("name") == text_row.get("name"):
            points += 100
        if row_has_substantive_text(row):
            points += 20

        row_name = normalize_text(str(row.get("name", "")))
        row_text = normalize_text(explainable_row_text(row))
        if is_difference_question(question):
            for group in comparison_groups[:2]:
                if row_matches_token_group(row, group):
                    points += 30
            if row_has_substantive_text(row) and any(row_matches_token_group(row, group) for group in comparison_groups[:2]):
                points += 15
            if polarity_comparison and ("positive" in row_name or "negative" in row_name):
                points += 20
            if polarity_comparison and any(term in row_name for term in ("rapid diagnostic test", "diagnostic", "test")):
                points += 12
        if re.search(r"\b(symptom|symptoms)\b", normalized_question):
            if "narrative" in row_name:
                points += 15
            if any(term in row_text for term in ("symptom", "fever", "headache", "vomiting", "chills")):
                points += 10
        if re.search(r"\b(treat|treatment|manage|therapy|therapies)\b", normalized_question):
            if "narrative" in row_name:
                points += 15
            if any(term in row_text for term in ("treatment", "treat", "intervention", "therapy", "artemisinin", "bednet")):
                points += 10

        return (
            -points,
            -float(row.get("total_value") or 0),
            -int(row.get("data_points") or 0),
            str(row.get("name") or ""),
        )

    return sorted(list(evidence_rows), key=score)


def select_explainable_evidence_rows(
    question: str,
    evidence_rows: Sequence[Dict[str, Any]],
    limit: int = 6,
) -> List[Dict[str, Any]]:
    prioritized_rows = prioritize_explainable_evidence_rows(question, evidence_rows)
    if not is_difference_question(question):
        return prioritized_rows[:limit]

    comparison_groups = extract_comparison_target_groups(question)
    if len(comparison_groups) < 2:
        return prioritized_rows[:limit]

    selected_rows: List[Dict[str, Any]] = []
    seen_names = set()

    for group in comparison_groups[:2]:
        matching_rows = [row for row in prioritized_rows if row_matches_token_group(row, group)]
        if not matching_rows:
            continue

        preferred_row = next((row for row in matching_rows if row_has_substantive_text(row)), None)
        if preferred_row is None:
            preferred_row = next((row for row in matching_rows if row_has_numeric_signal(row)), None)
        preferred_row = preferred_row or matching_rows[0]

        preferred_name = preferred_row.get("name")
        if preferred_name not in seen_names:
            selected_rows.append(preferred_row)
            seen_names.add(preferred_name)

        secondary_row = next(
            (
                row
                for row in matching_rows
                if row.get("name") not in seen_names and (row_has_numeric_signal(row) or row_has_substantive_text(row))
            ),
            None,
        )
        if secondary_row and len(selected_rows) < limit:
            selected_rows.append(secondary_row)
            seen_names.add(secondary_row.get("name"))

    for row in prioritized_rows:
        row_name = row.get("name")
        if row_name in seen_names:
            continue
        selected_rows.append(row)
        seen_names.add(row_name)
        if len(selected_rows) >= limit:
            break

    return selected_rows[:limit]


def format_explainable_evidence(question: str, evidence_rows: Sequence[Dict[str, Any]]) -> str:
    lines = []
    prioritized_rows = prioritize_explainable_evidence_rows(question, evidence_rows)
    for index, row in enumerate(prioritized_rows[:3], 1):
        evidence_text = explainable_text_excerpt(explainable_row_text(row))
        period_end = row.get("last_period_end") or row.get("latest_period_end") or "n/a"
        line = (
            f"{index}. metric={row.get('name', '')}; "
            f"data_points={row.get('data_points', 0)}; "
            f"total_value={row.get('total_value', 0)}; "
            f"period_end={period_end}; "
            f"text={evidence_text or '(none)'}"
        )
        lines.append(line)
    return "\n".join(lines)


@lru_cache(maxsize=128)
def cached_explainable_chat_completion(
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    return (
        call_chat_llm(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        or ""
    ).strip()


def is_why_question(question: str) -> bool:
    return normalize_text(question).startswith("why")


def build_explainable_fallback_answer(question: str, evidence_rows: Sequence[Dict[str, Any]]) -> str:
    topic = extract_explainable_topic(question)
    if not evidence_rows:
        return (
            f"I could not find database evidence for '{topic}'. "
            "The current DHIS2 data does not show matching indicators or recorded values for that topic."
        )

    prioritized_rows = prioritize_explainable_evidence_rows(question, evidence_rows)
    names = [row.get("name") for row in prioritized_rows[:3] if row.get("name")]
    top_row = prioritized_rows[0]
    text_row = select_textual_evidence_row(evidence_rows, question=question)
    row_for_text = text_row or top_row
    evidence_text = summarize_explainable_text(explainable_row_text(row_for_text))
    primary_row = text_row or top_row

    answer_parts = []
    if is_difference_question(question):
        comparison_groups = extract_comparison_target_groups(question)
        if len(comparison_groups) >= 2:
            left_group, right_group = comparison_groups[:2]
            left_row = select_group_evidence_row(prioritized_rows, left_group)
            right_row = select_group_evidence_row(prioritized_rows, right_group)
            if left_row and right_row:
                left_label = format_metric_group_label(left_group)
                right_label = format_metric_group_label(right_group)
                answer_parts.append(
                    f"The retrieved data matches separate evidence for {left_label} and {right_label}."
                )
                if comparison_groups_are_polarity_pair(comparison_groups):
                    answer_parts.append(
                        f"'{left_row['name']}' reflects one diagnostic result category, while '{right_row['name']}' reflects the opposite diagnostic result category."
                    )
                else:
                    answer_parts.append(
                        f"For {left_label}, the closest matched indicator is '{left_row['name']}', while for {right_label}, it is '{right_row['name']}'."
                    )
                answer_parts.append(
                    f"In the matched data, {describe_evidence_row(left_row, allow_narrative_only=True)}, while {describe_evidence_row(right_row, allow_narrative_only=True)}."
                )
                start, end = evidence_period_span(prioritized_rows)
                if start or end:
                    answer_parts.append(
                        f"The available evidence spans from {start or 'n/a'} to {end or 'n/a'}."
                    )
                return " ".join(answer_parts)

    if is_probability_question(question):
        answer_parts.append(
            f"The dataset does not provide a patient-level probability or chance for {topic}."
        )
        if names:
            answer_parts.append(
                f"It does contain related indicators such as {', '.join(names)}."
            )
    elif text_row and evidence_text and is_why_question(question):
        answer_parts.append(
            f"The database does not record a direct causal explanation for {topic}."
        )
        answer_parts.append(
            f"The closest narrative evidence comes from '{text_row['name']}': {evidence_text}"
        )
    elif text_row and evidence_text:
        answer_parts.append(evidence_text)
        answer_parts.append(
            f"This explanation comes from '{text_row['name']}' in the DHIS2 data."
        )
    elif row_has_substantive_text(top_row):
        answer_parts.append(
            f"The database does not give a full medical definition of {topic}, but the closest matched indicator is "
            f"'{top_row['name']}', described as '{clean_explainable_text(top_row.get('description'))}'."
        )
    else:
        if is_why_question(question):
            answer_parts.append(
                f"The database does not record a direct causal explanation for {topic}."
            )
        elif is_treatment_question(question):
            answer_parts.append(
                f"The database does not record a direct treatment protocol for {topic}."
            )
        elif is_symptom_question(question):
            answer_parts.append(
                f"The database does not record a direct symptom description for {topic}."
            )
        else:
            answer_parts.append(
                f"The database does not contain a direct medical definition of {topic}."
            )
        if names:
            answer_parts.append(
                f"It does contain related indicators such as {', '.join(names)}."
            )

    if primary_row.get("name"):
        if row_has_numeric_signal(primary_row):
            answer_parts.append(
                f"The strongest database match is '{primary_row['name']}' with total value {format_answer_number(primary_row.get('total_value'))} "
                f"across {format_answer_number(primary_row.get('data_points'))} data points."
            )
        else:
            answer_parts.append(
                f"The strongest database match is '{primary_row['name']}'."
            )

    if primary_row.get("first_period_start") or primary_row.get("last_period_end"):
        answer_parts.append(
            f"The available evidence spans from {primary_row.get('first_period_start', 'n/a')} "
            f"to {primary_row.get('last_period_end', 'n/a')}."
        )

    return " ".join(answer_parts)


def explainable_evidence_has_descriptions(evidence_rows: Sequence[Dict[str, Any]]) -> bool:
    return any(row_has_substantive_text(row) for row in evidence_rows[:5])


def is_probability_question(question: str) -> bool:
    return re.search(r"\b(chance|chances|risk|probability|odds|likelihood)\b", normalize_text(question)) is not None


def is_treatment_question(question: str) -> bool:
    return re.search(
        r"\b(treat|treatment|therapy|manage|management|cure|medicine|medicines|drug|drugs)\b",
        normalize_text(question),
    ) is not None


def is_symptom_question(question: str) -> bool:
    return re.search(r"\b(symptom|symptoms|sign|signs)\b", normalize_text(question)) is not None


def is_difference_question(question: str) -> bool:
    return re.search(r"\b(difference|different|compare|versus|vs)\b", normalize_text(question)) is not None


def explainable_eval_tokens(text: str) -> List[str]:
    extra_stop_words = {
        "data", "evidence", "indicator", "indicators", "line", "say", "says",
        "align", "aligned", "alignment", "lineup", "lines", "does", "it", "with",
    }
    tokens: List[str] = []
    for token in tokenize(text):
        singular_token = singularize(token)
        if singular_token in EXPLAINABLE_STOP_WORDS or singular_token in extra_stop_words:
            continue
        if singular_token not in tokens:
            tokens.append(singular_token)
    return tokens


def explainable_answer_mentions_evidence(answer: str, evidence_rows: Sequence[Dict[str, Any]]) -> bool:
    normalized_answer = normalize_text(answer)
    evidence_names = [normalize_text(row.get("name", "")) for row in evidence_rows[:3] if row.get("name")]
    return any(name and contains_normalized_phrase(normalized_answer, name) for name in evidence_names)


def explainable_answer_references_dataset(answer: str) -> bool:
    normalized_answer = normalize_text(answer)
    dataset_phrases = (
        "retrieved data",
        "matched data",
        "matched evidence",
        "database evidence",
        "dhis2",
        "dataset",
        "indicator",
        "indicators",
        "data point",
        "data points",
        "total value",
        "latest value",
        "evidence period",
    )
    return any(phrase in normalized_answer for phrase in dataset_phrases)


def grounded_answer_supported_by_evidence(question: str, answer: str, evidence_rows: Sequence[Dict[str, Any]]) -> bool:
    normalized_answer = normalize_text(answer)
    if not normalized_answer:
        return False

    evidence_blob = " ".join(
        normalize_text(
            " ".join(
                str(row.get(field) or "")
                for field in ("name", "description", "latest_text_value", "latest_comment")
            )
        )
        for row in evidence_rows[:5]
    )
    normalized_question = normalize_text(question)
    answer_tokens = set(explainable_eval_tokens(answer))
    question_tokens = set(explainable_eval_tokens(question))
    evidence_tokens = set(explainable_eval_tokens(evidence_blob))
    topical_overlap = question_tokens & answer_tokens
    evidence_overlap = evidence_tokens & answer_tokens
    mentions_dataset_context = explainable_answer_references_dataset(answer)
    mentions_evidence_name = explainable_answer_mentions_evidence(answer, evidence_rows)

    if is_difference_question(question):
        comparison_groups = extract_comparison_target_groups(question)
        if len(comparison_groups) >= 2:
            left_group, right_group = comparison_groups[:2]
            answer_has_left = any(token in answer_tokens or contains_normalized_phrase(normalized_answer, token) for token in left_group)
            answer_has_right = any(token in answer_tokens or contains_normalized_phrase(normalized_answer, token) for token in right_group)
            evidence_has_left = any(contains_normalized_phrase(evidence_blob, token) for token in left_group)
            evidence_has_right = any(contains_normalized_phrase(evidence_blob, token) for token in right_group)
            left_row = select_group_evidence_row(evidence_rows, left_group)
            right_row = select_group_evidence_row(evidence_rows, right_group)

            if not (answer_has_left and answer_has_right and evidence_has_left and evidence_has_right):
                return False

            if comparison_groups_are_polarity_pair(comparison_groups):
                has_detection_language = any(
                    phrase in normalized_answer
                    for phrase in (
                        "detected",
                        "not detected",
                        "presence",
                        "absence",
                        "present",
                        "absent",
                        "test is positive",
                        "test is negative",
                    )
                )
                left_indicator_name = normalize_text(str(left_row.get("name", ""))) if left_row else ""
                right_indicator_name = normalize_text(str(right_row.get("name", ""))) if right_row else ""
                mentions_left_indicator = bool(left_indicator_name and contains_normalized_phrase(normalized_answer, left_indicator_name))
                mentions_right_indicator = bool(right_indicator_name and contains_normalized_phrase(normalized_answer, right_indicator_name))
                has_numeric_comparison = any(
                    phrase in normalized_answer
                    for phrase in (
                        "total",
                        "totals",
                        "total value",
                        "data point",
                        "data points",
                        "latest value",
                        "higher than",
                        "lower than",
                        "significantly higher",
                    )
                )
                return (
                    has_detection_language and (
                        mentions_dataset_context or mentions_left_indicator or mentions_right_indicator or has_numeric_comparison
                    )
                ) or (
                    mentions_left_indicator and mentions_right_indicator and has_numeric_comparison
                )

            left_indicator_name = normalize_text(str(left_row.get("name", ""))) if left_row else ""
            right_indicator_name = normalize_text(str(right_row.get("name", ""))) if right_row else ""
            mentions_left_indicator = bool(left_indicator_name and contains_normalized_phrase(normalized_answer, left_indicator_name))
            mentions_right_indicator = bool(right_indicator_name and contains_normalized_phrase(normalized_answer, right_indicator_name))
            if left_row and right_row and (row_has_substantive_text(left_row) or row_has_substantive_text(right_row)):
                return mentions_left_indicator and mentions_right_indicator

            has_comparison_language = any(
                phrase in normalized_answer
                for phrase in ("difference", "different", "while", "whereas", "compared", "compare", "both")
            )
            return has_comparison_language or bool(evidence_overlap)

    if mentions_evidence_name:
        return True

    if topical_overlap and (mentions_dataset_context or len(evidence_overlap) >= 2):
        return True

    if is_treatment_question(question):
        has_topic = "malaria" in normalized_answer if "malaria" in normalized_question else True
        has_treatment_terms = any(term in normalized_answer for term in ("treat", "treatment", "therapy", "follow up", "follow-up", "act"))
        evidence_support = any(term in evidence_blob for term in ("treat", "treatment", "therapy", "act"))
        return has_topic and has_treatment_terms and evidence_support

    if is_symptom_question(question):
        has_topic = "malaria" in normalized_answer if "malaria" in normalized_question else True
        has_symptom_terms = any(term in normalized_answer for term in ("symptom", "fever", "headache", "vomiting", "chills", "nausea"))
        evidence_support = any(term in evidence_blob for term in ("symptom", "fever", "headache", "vomiting", "chills", "nausea"))
        return has_topic and has_symptom_terms and evidence_support

    if explainable_evidence_has_descriptions(evidence_rows):
        asks_for_narrative_alignment = any(
            phrase in normalized_question
            for phrase in ("narrative", "line up", "lineup", "align", "evidence")
        )
        has_alignment_language = any(
            phrase in normalized_answer
            for phrase in ("narrative", "indicator", "indicators", "align", "aligned", "consistent", "line up")
        )
        if topical_overlap and len(evidence_overlap) >= 2:
            return True
        if asks_for_narrative_alignment and topical_overlap and (has_alignment_language or evidence_overlap):
            return True

    return False


def numeric_value(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if not text or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_answer_number(value: Any) -> str:
    number = numeric_value(value)
    if number is None:
        return "n/a"
    if abs(number - round(number)) < 1e-9:
        return f"{int(round(number)):,}"
    return f"{number:,.2f}".rstrip("0").rstrip(".")


def evidence_period_span(evidence_rows: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    starts = [str(row.get("first_period_start")).strip() for row in evidence_rows if row.get("first_period_start")]
    ends = [str(row.get("last_period_end")).strip() for row in evidence_rows if row.get("last_period_end")]
    start = min(starts) if starts else None
    end = max(ends) if ends else None
    return start, end


def row_has_numeric_signal(row: Dict[str, Any]) -> bool:
    total_value = numeric_value(row.get("total_value"))
    data_points = numeric_value(row.get("data_points"))
    latest_value = numeric_value(row.get("latest_value"))
    return any(value not in (None, 0) for value in (total_value, data_points, latest_value))


def describe_evidence_row(row: Dict[str, Any], allow_narrative_only: bool = True) -> str:
    name = row.get("name") or "the matched indicator"
    total_value = numeric_value(row.get("total_value"))
    data_points = numeric_value(row.get("data_points"))
    latest_value = numeric_value(row.get("latest_value"))

    if total_value not in (None, 0) and data_points not in (None, 0):
        return f"'{name}' totals {format_answer_number(total_value)} across {format_answer_number(data_points)} data points"
    if total_value not in (None, 0):
        return f"'{name}' has total value {format_answer_number(total_value)}"
    if latest_value not in (None, 0):
        return f"'{name}' has latest value {format_answer_number(latest_value)}"
    if allow_narrative_only and row_has_substantive_text(row):
        return f"'{name}' provides the closest matched narrative evidence"
    return f"'{name}' is one of the closest matched indicators"


def format_metric_group_label(group: Sequence[str]) -> str:
    return " ".join(group).strip()


def comparison_groups_are_polarity_pair(groups: Sequence[Sequence[str]]) -> bool:
    if len(groups) < 2:
        return False
    flattened_tokens = {token for group in groups[:2] for token in group}
    return "positive" in flattened_tokens and "negative" in flattened_tokens


def select_group_evidence_row(evidence_rows: Sequence[Dict[str, Any]], token_group: Sequence[str]) -> Optional[Dict[str, Any]]:
    matching_rows = [row for row in evidence_rows if row_matches_token_group(row, token_group)]
    if not matching_rows:
        return None
    return next((row for row in matching_rows if row_has_substantive_text(row)), None) or next(
        (row for row in matching_rows if row_has_numeric_signal(row)),
        None,
    ) or matching_rows[0]


def build_retrieved_data_summary(question: str, evidence_rows: Sequence[Dict[str, Any]]) -> str:
    prioritized_rows = prioritize_explainable_evidence_rows(question, evidence_rows)
    if not prioritized_rows:
        return ""

    start, end = evidence_period_span(prioritized_rows)
    span_text = ""
    if start and end:
        span_text = f" The matched evidence spans {start} to {end}."
    elif start:
        span_text = f" The matched evidence starts at {start}."
    elif end:
        span_text = f" The matched evidence runs through {end}."

    if is_difference_question(question):
        comparison_groups = extract_comparison_target_groups(question)
        if len(comparison_groups) >= 2:
            left_group, right_group = comparison_groups[:2]
            left_row = select_group_evidence_row(prioritized_rows, left_group)
            right_row = select_group_evidence_row(prioritized_rows, right_group)
            if left_row and right_row:
                left_label = format_metric_group_label(left_group)
                right_label = format_metric_group_label(right_group)
                return (
                    f"In the retrieved data, for {left_label}, {describe_evidence_row(left_row)}; "
                    f"for {right_label}, {describe_evidence_row(right_row)}."
                    f"{span_text}"
                ).strip()

    primary_row = prioritized_rows[0]
    text_row = select_textual_evidence_row(prioritized_rows)
    numeric_row = next((row for row in prioritized_rows if row_has_numeric_signal(row)), None)

    summary_parts: List[str] = []
    if text_row and text_row.get("name"):
        summary_parts.append(
            f"The retrieved DHIS2 evidence includes descriptive text under '{text_row['name']}'."
        )

    focus_row = numeric_row or primary_row
    if focus_row:
        focus_sentence = f"In the matched data, {describe_evidence_row(focus_row)}."
        if not text_row or focus_row.get("name") != text_row.get("name"):
            summary_parts.append(focus_sentence)
        elif not row_has_substantive_text(text_row):
            summary_parts.append(focus_sentence)

    if not summary_parts:
        summary_parts.append(
            f"The closest retrieved evidence is {describe_evidence_row(primary_row)}."
        )

    return " ".join(summary_parts).strip() + span_text


def attach_evidence_context(answer: str, question: str, evidence_rows: Sequence[Dict[str, Any]]) -> str:
    cleaned_answer = (answer or "").strip()
    if not cleaned_answer or not evidence_rows:
        return cleaned_answer

    evidence_note = build_retrieved_data_summary(question, evidence_rows)
    if not evidence_note:
        return cleaned_answer
    if normalize_text(evidence_note) in normalize_text(cleaned_answer):
        return cleaned_answer
    return f"{cleaned_answer} {evidence_note}"


ANDROID_RUNTIME_ENABLED = {"1", "true", "yes", "on"}


def should_use_sqlcoder_first(prefer_sqlcoder_first: Optional[bool] = None) -> bool:
    if prefer_sqlcoder_first is not None:
        return bool(prefer_sqlcoder_first)
    return os.environ.get("ANDROID_LLM_BRIDGE", "").strip().lower() in ANDROID_RUNTIME_ENABLED


def run_sqlcoder_first_pass(
    question: str,
    db_path: str,
    *,
    debug: bool,
    force_sql_model: bool,
) -> Dict[str, Any]:
    result = {
        "sql_model_attempted": True,
        "sql_model_used": False,
        "sql_model_raw_output": None,
        "sql_model_error": None,
        "generated_sql": None,
        "analytics_payload": None,
        "analytics_context": "",
        "final_error": None,
    }
    try:
        from main import run_text_to_sql

        shared = run_text_to_sql(
            question,
            db_path=db_path,
            max_debug_retries=3,
            force_sql_model=force_sql_model,
        )
        sql_trace = shared.get("sql_trace") or {}
        result["sql_model_used"] = bool(sql_trace.get("model_used"))
        result["sql_model_raw_output"] = sql_trace.get("raw_output")
        result["generated_sql"] = shared.get("generated_sql")
        result["analytics_payload"] = shared.get("analytics_payload")
        result["analytics_context"] = shared.get("analytics_context") or ""
        result["final_error"] = shared.get("final_error")
        result["sql_model_error"] = shared.get("final_error")
    except Exception as exc:
        if debug:
            print(f"SQLCoder-first pass failed before explanation stage. Error: {exc}")
        result["final_error"] = str(exc)
        result["sql_model_error"] = str(exc)
    return result


def build_sqlcoder_prompt_context(sqlcoder_result: Optional[Dict[str, Any]]) -> str:
    if not sqlcoder_result:
        return ""

    payload = sqlcoder_result.get("analytics_payload") or {}
    sql_text = (payload.get("sql") or sqlcoder_result.get("generated_sql") or "").strip()
    highlights = payload.get("highlights") or []
    preview_rows = payload.get("preview_rows") or []
    preview_text = json.dumps(preview_rows[:5], indent=2, default=str) if preview_rows else "[]"

    parts = []
    if sql_text:
        parts.append("Executed SQL:\n" + sql_text)
    if highlights:
        parts.append("SQL result highlights:\n" + "\n".join(f"- {highlight}" for highlight in highlights))
    if preview_rows:
        parts.append("SQL preview rows (JSON):\n" + preview_text)
    return "\n\n".join(part for part in parts if part).strip()


def build_explainable_debug_trace(
    *,
    query_path: str,
    sqlcoder_result: Optional[Dict[str, Any]],
    answer_options: Dict[str, Any],
    evidence_query_mode: str,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    return {
        "query_path": query_path,
        "evidence_query_mode": evidence_query_mode,
        "sql_model_attempted": bool((sqlcoder_result or {}).get("sql_model_attempted")),
        "sql_model_used": bool((sqlcoder_result or {}).get("sql_model_used")),
        "sql_model_raw_output": (sqlcoder_result or {}).get("sql_model_raw_output"),
        "sql_model_error": (sqlcoder_result or {}).get("sql_model_error"),
        "cure_med_grounded_raw_output": answer_options.get("raw_grounded_model_answer"),
        "fallback_answer": answer_options.get("fallback_answer"),
        "answer_source": answer_options.get("answer_source"),
        "model_reason": answer_options.get("model_reason"),
        "final_answer": answer_options.get("answer"),
        "timings": make_json_safe(timings or {}),
    }


GENERAL_CHAT_SYSTEM_PROMPT = (
    "You are Uliza, a helpful health assistant. "
    "Answer the user's question directly in 3 to 5 sentences. "
    "You may use general medical knowledge to explain the concept clearly. "
    "Do not mention prompts, SQL, models, or database evidence unless the user asks. "
    "Do not claim anything about the user's dataset, records, or retrieved evidence unless it is explicitly provided."
)


PATCH_CHAT_SYSTEM_PROMPT = (
    "You are Uliza, a grounded health assistant. "
    "You will receive a general medical answer and a database-grounded answer. "
    "Produce one concise final answer that combines medical explanation with the retrieved DHIS2 evidence. "
    "Prefer the database-grounded answer for dataset-specific claims, but keep accurate medical context from the general answer when it helps the user. "
    "Answer the user's question directly. "
    "Do not mention prompts, SQL, models, the merge process, or phrases like general answer or database-grounded answer."
)


def generate_general_explainable_answer(question: str) -> str:
    prompt = f"""
Question: {question}

Write a short direct answer for the user in 3 to 5 sentences.
Use your medical knowledge to explain the concept plainly and accurately.
If the question asks for a probability, chance, or risk that is not directly knowable, say that clearly instead of inventing a number.
Use plain sentences only, with no markdown, asterisks, or bullet points.
""".strip()
    return (call_chat_llm(prompt, system_prompt=GENERAL_CHAT_SYSTEM_PROMPT) or "").strip()


def generate_grounded_explainable_answer(
    question: str,
    evidence_rows: Sequence[Dict[str, Any]],
    sql_analytics_context: str = "",
) -> str:
    prioritized_rows = prioritize_explainable_evidence_rows(question, evidence_rows)
    key_text_row = select_textual_evidence_row(prioritized_rows, question=question)
    key_text = summarize_explainable_text(explainable_row_text(key_text_row)) if key_text_row else ""
    sql_context_block = (
        f"\nExecuted SQL analytics context:\n{sql_analytics_context.strip()}\n"
        if sql_analytics_context.strip()
        else ""
    )
    if is_difference_question(question):
        comparison_groups = extract_comparison_target_groups(question)
        left_label = format_metric_group_label(comparison_groups[0]) if len(comparison_groups) >= 1 else "the first topic"
        right_label = format_metric_group_label(comparison_groups[1]) if len(comparison_groups) >= 2 else "the second topic"
        prompt = f"""
Question: {question}
{sql_context_block}

Database evidence:
{format_explainable_evidence(question, prioritized_rows)}

Write 2 to 4 short sentences.
Explain the medical difference between {left_label} and {right_label}, then connect it to the matched evidence.
Mention at least one matched indicator for each side when available.
Copy retrieved numbers exactly if used. Do not invent unsupported dataset claims.
""".strip()
        return cached_explainable_chat_completion(
            prompt,
            CHAT_SYSTEM_PROMPT,
            192,
            0.15,
        )

    prompt = f"""
Question: {question}
{sql_context_block}

Database evidence:
{format_explainable_evidence(question, prioritized_rows)}

Primary textual evidence:
metric={key_text_row.get('name') if key_text_row else '(none)'}
text={key_text or '(none)'}

Write 2 to 4 short sentences for the user.
Answer medically, then connect the explanation to the matched DHIS2 evidence.
Mention at least one matched indicator from the evidence.
If the evidence is indirect, say that briefly.
Copy retrieved numbers exactly if used. Do not invent unsupported dataset claims.
    """.strip()
    return cached_explainable_chat_completion(
        prompt,
        CHAT_SYSTEM_PROMPT,
        160,
        0.15,
    )


def patch_explainable_answers(
    question: str,
    general_answer: str,
    grounded_answer: str,
    sql_analytics_context: str = "",
) -> str:
    sql_context_block = (
        f"\nExecuted SQL analytics context:\n{sql_analytics_context.strip()}\n"
        if sql_analytics_context.strip()
        else ""
    )
    if is_difference_question(question):
        comparison_groups = extract_comparison_target_groups(question)
        left_label = format_metric_group_label(comparison_groups[0]) if len(comparison_groups) >= 1 else "the first topic"
        right_label = format_metric_group_label(comparison_groups[1]) if len(comparison_groups) >= 2 else "the second topic"
        prompt = f"""
Question: {question}
{sql_context_block}

General answer:
{general_answer or '(none)'}

Database-grounded answer:
{grounded_answer or '(none)'}

Write one final answer in 3 to 5 sentences.
Answer the question directly as a comparison between {left_label} and {right_label}.
Use the executed SQL analytics context first for dataset-specific claims when it is provided.
If the comparison is about positive versus negative test results, explain that distinction medically and then relate it to the matched test evidence.
Prefer the database-grounded answer whenever there is any conflict about what the dataset shows.
Keep useful medical explanation from the general answer when it helps clarify the terms.
Keep any concrete retrieved details such as indicator names, totals, data-point counts, or evidence periods when they are available.
If you mention a retrieved number, copy it exactly from the evidence-aware answer.
Do not infer prevalence, treatment rates, burden, or trends from the data unless the evidence-aware answer explicitly supports them.
Do not speculate about causes, trends, geography, interventions, or public-health drivers unless the evidence-aware answer explicitly supports them.
Use plain sentences only, with no markdown, asterisks, or bullet points.
Make sure both sides of the comparison are addressed.
Do not invent dataset values or unsupported retrieved findings.
Do not mention the two candidate answers or the merge process.
""".strip()
        return (call_chat_llm(prompt, system_prompt=PATCH_CHAT_SYSTEM_PROMPT) or "").strip()

    prompt = f"""
Question: {question}
{sql_context_block}

General answer:
{general_answer or '(none)'}

Database-grounded answer:
{grounded_answer or '(none)'}

Write one final answer in 3 to 5 sentences.
Keep the response medically sensible, and use the general answer for medical background when it helps.
Use the executed SQL analytics context first for dataset-specific claims when it is provided.
Prefer the database-grounded answer when there is any conflict about what the dataset shows.
If the grounded answer says the dataset does not provide a specific measure, preserve that limitation clearly.
When the dataset includes concrete retrieved details, mention them briefly in the final answer.
If you mention a retrieved number, copy it exactly from the evidence-aware answer.
Do not speculate about causes, trends, geography, interventions, or public-health drivers unless the evidence-aware answer explicitly supports them.
Use plain sentences only, with no markdown, asterisks, or bullet points.
Do not talk about the two candidate answers.
Do not say things like "the database-grounded answer", "the general answer", "this answer", or "the response".
Answer the original question directly for the end user.
""".strip()
    return (call_chat_llm(prompt, system_prompt=PATCH_CHAT_SYSTEM_PROMPT) or "").strip()


def is_meta_answer(answer: str) -> bool:
    normalized_answer = normalize_text(answer)
    meta_phrases = (
        "general answer",
        "database grounded answer",
        "grounded answer",
        "this answer",
        "the answer",
        "the response",
        "candidate answer",
        "maintains the specificity",
        "providing a more detailed explanation",
    )
    return any(phrase in normalized_answer for phrase in meta_phrases)


def model_answer_failed(answer: str) -> bool:
    normalized_answer = normalize_text(answer)
    if not normalized_answer or len(normalized_answer.split()) < 5:
        return True
    failure_phrases = (
        "i do not know",
        "i don't know",
        "cannot answer",
        "can not answer",
        "no answer",
        "not enough information",
        "insufficient information",
        "unable to answer",
    )
    return any(phrase in normalized_answer for phrase in failure_phrases)


def strip_untrusted_evidence_claims(answer: str, evidence_rows: Sequence[Dict[str, Any]]) -> str:
    cleaned_answer = (answer or "").strip()
    if not cleaned_answer or not evidence_rows:
        return cleaned_answer

    evidence_names = [normalize_text(row.get("name", "")) for row in evidence_rows[:5] if row.get("name")]
    dataset_markers = (
        "dataset",
        "database",
        "evidence",
        "retrieved data",
        "matched data",
        "dhis2",
        "indicator",
        "indicators",
        "case",
        "cases",
        "result",
        "results",
        "data point",
        "data points",
        "total",
        "latest",
        "latest value",
        "latest period",
    )
    dataset_claim_phrases = (
        "dataset shows",
        "database shows",
        "evidence shows",
        "retrieved data shows",
        "matched data shows",
        "according to the data",
        "according to the dataset",
        "this data",
        "the data",
        "these data",
        "available evidence",
        "key indicators",
        "indicators for",
    )

    kept_sentences = []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned_answer)
    for sentence in sentences:
        normalized_sentence = normalize_text(sentence)
        if not normalized_sentence:
            continue
        has_digits = re.search(r"\d", sentence) is not None
        mentions_evidence_name = any(
            name and contains_normalized_phrase(normalized_sentence, name)
            for name in evidence_names
        )
        has_dataset_marker = any(marker in normalized_sentence for marker in dataset_markers)
        has_dataset_claim = any(phrase in normalized_sentence for phrase in dataset_claim_phrases)

        if mentions_evidence_name:
            continue
        if has_dataset_marker or has_dataset_claim:
            continue
        if has_digits and any(term in normalized_sentence for term in ("case", "cases", "result", "results", "latest", "total")):
            continue
        kept_sentences.append(sentence.strip())

    return " ".join(sentence for sentence in kept_sentences if sentence).strip()


def prepare_explainable_model_answer(
    question: str,
    answer: str,
    evidence_rows: Sequence[Dict[str, Any]],
) -> str:
    cleaned_answer = refine_explainable_model_answer(question, answer)
    cleaned_answer = strip_untrusted_evidence_claims(cleaned_answer, evidence_rows)
    return cleaned_answer


def best_effort_explainable_model_answer(
    question: str,
    evidence_rows: Sequence[Dict[str, Any]],
    *,
    general_answer: str = "",
    grounded_answer: str = "",
    patched_answer: str = "",
) -> Tuple[Optional[str], Optional[str]]:
    candidates = (
        ("patched_model", patched_answer),
        ("grounded_model", grounded_answer),
        ("general_model", general_answer),
    )

    for source, raw_answer in candidates:
        cleaned_answer = prepare_explainable_model_answer(question, raw_answer, evidence_rows)
        if not cleaned_answer or model_answer_failed(cleaned_answer) or is_meta_answer(cleaned_answer):
            continue
        if source != "general_model" and evidence_rows:
            return attach_evidence_context(cleaned_answer, question, evidence_rows), source
        if source == "general_model" and evidence_rows:
            return attach_evidence_context(cleaned_answer, question, evidence_rows), source
        return cleaned_answer, source

    return None, None


def refine_explainable_model_answer(question: str, answer: str) -> str:
    cleaned_answer = re.sub(r"\s+", " ", (answer or "").strip())
    cleaned_answer = re.sub(r"[*_`]+", "", cleaned_answer)
    cleaned_answer = re.sub(r"(?i)^the question asks:[^.?!]*[.?!]\s*", "", cleaned_answer)
    cleaned_answer = re.sub(r'(?<!\d)(?:\d+\.\s*){2,}', "", cleaned_answer)
    if not cleaned_answer:
        return ""

    speculation_phrases = (
        "possibly due",
        "may be due",
        "might be due",
        "likely due",
        "suggesting an increase",
        "suggests an increase",
        "recent increase",
        "climate change",
        "vector breeding",
        "because of changing weather",
    )
    sentences = re.split(r"(?<=[.!?])\s+", cleaned_answer)
    kept_sentences = []
    for sentence in sentences:
        normalized_sentence = normalize_text(sentence)
        if any(phrase in normalized_sentence for phrase in speculation_phrases):
            continue
        sentence = re.sub(r"\s+[—-]\s+([^.;!?]+)", "", sentence).strip()
        if sentence:
            kept_sentences.append(sentence)
    cleaned_answer = " ".join(sentence for sentence in kept_sentences if sentence).strip()
    if not cleaned_answer:
        return ""

    if is_difference_question(question):
        comparison_groups = extract_comparison_target_groups(question)
        topical_terms = {token for group in comparison_groups for token in group}
        banned_phrases = (
            "exposed to the disease",
            "not yet contracted it",
            "control programs",
            "effectiveness of interventions",
            "progression of the disease",
            "clinical decision making process",
            "clinical decision-making process",
            "appropriate treatment regimen",
        )
        sentences = re.split(r"(?<=[.!?])\s+", cleaned_answer)
        kept_sentences = []
        for sentence in sentences:
            normalized_sentence = normalize_text(sentence)
            if any(phrase in normalized_sentence for phrase in banned_phrases):
                continue
            kept_sentences.append(sentence.strip())

        comparison_sentences = [
            sentence
            for sentence in kept_sentences
            if any(
                term in normalize_text(sentence)
                for term in (
                    *topical_terms,
                    "retrieved data",
                    "matched evidence",
                )
            )
        ]
        if comparison_sentences:
            return " ".join(comparison_sentences).strip()
        return " ".join(kept_sentences).strip()

    return cleaned_answer


def should_skip_grounded_explainable_model(
    question: str,
    evidence_rows: Sequence[Dict[str, Any]],
) -> Optional[str]:
    if not evidence_rows:
        return "no_evidence"
    return None


def generate_explainable_answer_options(
    question: str,
    evidence_rows: Sequence[Dict[str, Any]],
    debug: bool,
    sql_analytics_context: str = "",
    prefer_sqlcoder_first: bool = False,
) -> Dict[str, Any]:
    fallback_answer = build_explainable_fallback_answer(question, evidence_rows)
    result = {
        "fallback_answer": fallback_answer,
        "model_answer": None,
        "general_model_answer": None,
        "grounded_model_answer": None,
        "patched_model_answer": None,
        "raw_general_model_answer": None,
        "raw_grounded_model_answer": None,
        "raw_patched_model_answer": None,
        "answer": fallback_answer,
        "answer_source": "fallback",
        "model_attempted": False,
        "model_reason": None,
    }

    if not is_chat_model_available():
        if debug:
            print(f"Chat model unavailable. Using fallback answer. Reason: {chat_model_unavailable_reason()}")
        result["model_reason"] = chat_model_unavailable_reason()
        return result

    skip_reason = should_skip_grounded_explainable_model(question, evidence_rows)
    if skip_reason:
        result["model_reason"] = skip_reason
        return result

    result["model_attempted"] = True
    grounded_model_answer = ""
    try:
        raw_grounded_model_answer = generate_grounded_explainable_answer(
            question,
            evidence_rows,
            sql_analytics_context=sql_analytics_context,
        )
        result["raw_grounded_model_answer"] = raw_grounded_model_answer or None
        grounded_model_answer = raw_grounded_model_answer
        grounded_model_answer = prepare_explainable_model_answer(question, grounded_model_answer, evidence_rows)
    except Exception as exc:
        if debug:
            print(f"Grounded explainable generation failed. Using fallback answer. Error: {exc}")
        result["model_reason"] = f"grounded_model_error: {exc}"
        return result

    result["grounded_model_answer"] = grounded_model_answer or None
    grounded_supported = bool(
        grounded_model_answer and grounded_answer_supported_by_evidence(question, grounded_model_answer, evidence_rows)
    )
    if grounded_model_answer and not grounded_supported and debug:
        print("Grounded model answer was not sufficiently grounded in the matched evidence. Continuing because the model still answered the question.")

    grounded_with_context = attach_evidence_context(grounded_model_answer, question, evidence_rows) if grounded_model_answer else None
    grounded_with_context_supported = bool(
        grounded_with_context and grounded_answer_supported_by_evidence(question, grounded_with_context, evidence_rows)
    )
    selected_grounded_answer = grounded_with_context if grounded_with_context_supported else grounded_model_answer

    if grounded_model_answer and model_answer_failed(grounded_model_answer):
        if debug:
            print("Grounded model answer failed to answer the question. Using fallback answer.")
        result["model_reason"] = "failed_grounded_model_answer"
        return result

    result["model_answer"] = selected_grounded_answer or grounded_model_answer or None
    if prefer_sqlcoder_first:
        if selected_grounded_answer and grounded_with_context_supported:
            result["grounded_model_answer"] = selected_grounded_answer
        else:
            result["grounded_model_answer"] = grounded_model_answer or None
        if result["model_answer"]:
            result["answer"] = result["model_answer"]
            result["answer_source"] = "grounded_model"
            result["model_reason"] = "grounded_model_used_after_sqlcoder"
            return result
        result["model_reason"] = "sqlcoder_first_no_grounded_model_answer"
        return result

    if selected_grounded_answer and grounded_with_context_supported:
        result["grounded_model_answer"] = selected_grounded_answer
        result["model_answer"] = selected_grounded_answer
        result["answer"] = selected_grounded_answer
    else:
        result["grounded_model_answer"] = grounded_model_answer
        result["model_answer"] = grounded_model_answer
        result["answer"] = grounded_model_answer
    result["answer_source"] = "grounded_model"
    result["model_reason"] = "accepted_grounded_only"
    return result


def generate_explainable_answer(question: str, evidence_rows: Sequence[Dict[str, Any]], debug: bool) -> str:
    return generate_explainable_answer_options(question, evidence_rows, debug)["answer"]


def build_unmatched_result(plan: QueryPlan, include_insights: bool, message: str) -> Dict[str, Any]:
    insights = None
    if include_insights:
        insights = build_basic_insights(plan, plan.intent, [], [], 0)
        insights["message"] = message

    return make_json_safe(
        {
            "view": plan.intent,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "message": message,
            "insights": insights,
            "insights_available": False,
            "sql_queries": [],
        }
    )


def explainable_rows_to_frontend(evidence_rows: Sequence[Dict[str, Any]], include_rows: bool) -> Tuple[List[str], List[List[Any]]]:
    if not evidence_rows:
        return [], []

    prioritized_rows = prioritize_explainable_evidence_rows("", evidence_rows)

    column_order = [
        "name",
        "description",
        "valuetype",
        "aggregationtype",
        "data_points",
        "total_value",
        "latest_value",
        "latest_period_start",
        "latest_period_end",
        "first_period_start",
        "last_period_end",
        "latest_text_value",
        "latest_comment",
    ]
    rename_map = {
        "name": "Metric",
        "description": "Description",
        "valuetype": "Value Type",
        "aggregationtype": "Aggregation",
        "data_points": "Data Points",
        "total_value": "Total Value",
        "latest_value": "Latest Value",
        "latest_period_start": "Latest Period Start",
        "latest_period_end": "Latest Period End",
        "first_period_start": "First Period Start",
        "last_period_end": "Last Period End",
        "latest_text_value": "Latest Text Value",
        "latest_comment": "Latest Comment",
    }

    available_columns = [column for column in column_order if any(row.get(column) not in (None, "") for row in evidence_rows)]
    if not include_rows:
        return [rename_map[column] for column in available_columns], []

    frontend_rows: List[List[Any]] = []
    for row in prioritized_rows:
        frontend_rows.append([format_number(row.get(column)) for column in available_columns])
    return [rename_map[column] for column in available_columns], frontend_rows


def answer_explainable_question(
    question: str,
    db_path: str,
    row_limit: int,
    debug: bool = False,
    page: int = 1,
    page_size: int = 100,
    include_insights: bool = False,
    include_rows: bool = True,
    include_debug_trace: bool = False,
    prefer_sqlcoder_first: Optional[bool] = None,
    timings: Optional[Dict[str, float]] = None,
    plan: Optional[QueryPlan] = None,
    resolved_plan_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sqlcoder_first_enabled = should_use_sqlcoder_first(prefer_sqlcoder_first)
    query_path = "android_sqlcoder_first_explainable" if sqlcoder_first_enabled else "explainable"
    sqlcoder_result = None
    sql_analytics_context = ""
    evidence_query_mode = "none"

    if plan is None:
        planning_started_at = perf_counter()
        plan = build_query_plan(
            question=question,
            db_path=db_path,
            page=page,
            page_size=page_size,
            row_limit=row_limit,
            debug=debug,
            timings=timings,
        )
        record_timing(timings, "query_planning_ms", planning_started_at)
    else:
        plan = replace(
            plan,
            question=question or plan.question,
            page=max(1, page),
            page_size=max(1, min(row_limit, page_size)),
            limit=max(1, min(row_limit, page_size)),
            debug=debug,
        )

    if sqlcoder_first_enabled:
        sqlcoder_started_at = perf_counter()
        sqlcoder_result = run_sqlcoder_first_pass(
            question,
            db_path,
            debug=debug,
            force_sql_model=True,
        )
        record_timing(timings, "sqlcoder_first_pass_ms", sqlcoder_started_at)
        sql_analytics_context = build_sqlcoder_prompt_context(sqlcoder_result)

    evidence_rows, evidence_sql, evidence_params, evidence_query_mode = fetch_explainable_evidence(
        plan,
        db_path,
        timings=timings,
    )

    if debug:
        if evidence_sql:
            print("\n===== EXPLAINABLE SQL =====")
            print(evidence_sql)
            print("\n===== EXPLAINABLE PARAMS =====")
            print(json.dumps(make_json_safe(evidence_params), indent=2))
        print("\n===== EXPLAINABLE EVIDENCE =====")
        print(json.dumps(make_json_safe(evidence_rows[:5]), indent=2))
        print("================================\n")

    model_started_at = perf_counter()
    answer_options = generate_explainable_answer_options(
        question,
        evidence_rows,
        debug=debug,
        sql_analytics_context=sql_analytics_context,
        prefer_sqlcoder_first=sqlcoder_first_enabled,
    )
    record_timing(timings, "model_generation_ms", model_started_at)
    frontend_columns, frontend_rows = explainable_rows_to_frontend(evidence_rows, include_rows=include_rows)
    insights = None
    insights_available = bool(evidence_rows)
    if include_insights:
        insight_started_at = perf_counter()
        preview_columns, preview_rows = explainable_rows_to_frontend(evidence_rows, include_rows=True)
        insights = (
            build_basic_insights(plan, "records", preview_columns, preview_rows, len(evidence_rows))
            if preview_rows
            else {"mode": "none"}
        )
        record_timing(timings, "insight_query_ms", insight_started_at)
        insights_available = bool(preview_rows)
    sql_queries = []
    if sqlcoder_result and sqlcoder_result.get("generated_sql"):
        sql_queries.append(
            {
                "label": "sqlcoder_generated_query",
                "sql": sqlcoder_result.get("generated_sql"),
                "params": [],
            }
        )
    if evidence_sql:
        sql_queries.append(
            {
                "label": "explainable_evidence_query",
                "sql": evidence_sql,
                "params": evidence_params,
            }
        )
    result = {
        "view": "explainable",
        "answer": answer_options["answer"],
        "answer_source": answer_options["answer_source"],
        "fallback_answer": answer_options["fallback_answer"],
        "model_answer": answer_options["model_answer"],
        "general_model_answer": answer_options["general_model_answer"],
        "grounded_model_answer": answer_options["grounded_model_answer"],
        "patched_model_answer": answer_options["patched_model_answer"],
        "model_attempted": answer_options["model_attempted"],
        "model_reason": answer_options["model_reason"],
        "columns": frontend_columns,
        "rows": frontend_rows,
        "row_count": len(evidence_rows),
        "insights": insights,
        "insights_available": insights_available,
        "sql_queries": make_json_safe(sql_queries),
        "resolved_plan": make_json_safe(resolved_plan_payload or resolved_explainable_plan_payload(plan)),
    }
    if include_debug_trace:
        result["debug_trace"] = build_explainable_debug_trace(
            query_path=query_path,
            sqlcoder_result=sqlcoder_result,
            answer_options=answer_options,
            evidence_query_mode=evidence_query_mode,
            timings=timings,
        )
    return make_json_safe(result)


def build_sql_query_entries(plan: QueryPlan, db_path: str) -> List[Dict[str, Any]]:
    where_sql, params = build_where_clause(plan)
    offset = (plan.page - 1) * plan.page_size

    if plan.intent == "summary":
        return [
            {
                "label": "summary_query",
                "sql": build_summary_sql(db_path, plan, where_sql),
                "params": list(params),
            }
        ]

    if plan.intent == "ranking":
        least_first = bool(re.search(r"\b(least|lowest)\b", plan.question.lower()))
        data_sql, count_sql = build_ranking_sql(db_path, plan, where_sql, least_first=least_first)
        return [
            {
                "label": "ranking_count_query",
                "sql": count_sql,
                "params": list(params),
            },
            {
                "label": "ranking_data_query",
                "sql": data_sql,
                "params": [*params, plan.page_size, offset],
            },
        ]

    if plan.intent == "peak":
        data_sql, count_sql = build_peak_sql(db_path, plan, where_sql)
        summary_sql = build_summary_sql(db_path, plan, where_sql)
        return [
            {
                "label": "peak_count_query",
                "sql": count_sql,
                "params": list(params),
            },
            {
                "label": "peak_summary_query",
                "sql": summary_sql,
                "params": list(params),
            },
            {
                "label": "peak_data_query",
                "sql": data_sql,
                "params": [*params, plan.page_size, offset],
            },
        ]

    if plan.intent == "comparison":
        sql, sql_params = build_comparison_sql(plan, db_path, where_sql, params)
        return [
            {
                "label": "comparison_query",
                "sql": sql,
                "params": list(sql_params),
            }
        ]

    data_sql, count_sql = build_records_sql(db_path, plan, where_sql)
    return [
        {
            "label": "records_count_query",
            "sql": count_sql,
            "params": list(params),
        },
        {
            "label": "records_data_query",
            "sql": data_sql,
            "params": [*params, plan.page_size, offset],
        },
    ]


def build_metric_group_condition(
    metric_group: Sequence[str],
    metric_ids: Sequence[int],
    id_expression: str = "dataelementid",
    name_expression: str = "lower(dataelement_name)",
) -> Tuple[Optional[str], List[Any]]:
    if metric_ids:
        placeholders = ", ".join("?" for _ in metric_ids)
        return f"{id_expression} IN ({placeholders})", [*metric_ids]

    cleaned_group = [token for token in metric_group if token]
    if not cleaned_group:
        return None, []

    params: List[Any] = []
    token_clauses = []
    for token in cleaned_group:
        token_clauses.append(f"{name_expression} LIKE ?")
        params.append(f"%{token}%")
    return "(" + " AND ".join(token_clauses) + ")", params


def build_metric_clause(
    metric_groups: Sequence[Sequence[str]],
    metric_group_ids: Optional[Sequence[Sequence[int]]] = None,
    id_expression: str = "dataelementid",
    name_expression: str = "lower(dataelement_name)",
) -> Tuple[Optional[str], List[Any]]:
    if not metric_groups:
        return None, []

    group_clauses = []
    params: List[Any] = []
    metric_group_ids = metric_group_ids or []
    for index, group in enumerate(metric_groups):
        group_ids = metric_group_ids[index] if index < len(metric_group_ids) else []
        group_clause, group_params = build_metric_group_condition(
            group,
            group_ids,
            id_expression=id_expression,
            name_expression=name_expression,
        )
        if not group_clause:
            continue
        group_clauses.append("(" + group_clause + ")")
        params.extend(group_params)

    if not group_clauses:
        return None, []
    return "(" + " OR ".join(group_clauses) + ")", params


def build_orgunit_metadata_sql(question: str, page_size: int, page: int) -> Tuple[str, List[Any], str, List[Any], Dict[str, str]]:
    normalized_question = normalize_text(question)
    asks_opening = any(term in normalized_question for term in ("opened", "opening date", "openingdate"))
    asks_closing = any(term in normalized_question for term in ("closed", "closed date", "closeddate"))
    asks_hierarchy = any(term in normalized_question for term in ("hierarchy level", "hierarchylevel", "level"))
    asks_code = "code" in normalized_question or "codes" in normalized_question

    date_column = "openingdate" if asks_opening or not asks_closing else "closeddate"
    sort_direction = "MIN"
    if any(term in normalized_question for term in ("latest", "newest", "most recent", "last")):
        sort_direction = "MAX"

    select_columns = ["name"]
    rename_map = {"name": "Organisation Unit"}
    if asks_code:
        select_columns.append("code")
        rename_map["code"] = "Code"
    if asks_hierarchy:
        select_columns.append("hierarchylevel")
        rename_map["hierarchylevel"] = "Hierarchy Level"
    if asks_opening or asks_closing:
        select_columns.append(date_column)
        rename_map[date_column] = "Opening Date" if date_column == "openingdate" else "Closed Date"

    select_sql = ",\n  ".join(select_columns)
    offset = max(0, (page - 1) * page_size)

    if asks_opening or asks_closing:
        predicate = f"""
trim(coalesce({date_column}, '')) <> ''
  AND {date_column} = (
    SELECT {sort_direction}({date_column})
    FROM organisationunit
    WHERE trim(coalesce({date_column}, '')) <> ''
  )
""".strip()
    elif asks_hierarchy:
        hierarchy_operator = "MAX"
        if any(term in normalized_question for term in ("lowest", "smallest", "minimum", "min")):
            hierarchy_operator = "MIN"
        predicate = f"""
hierarchylevel IS NOT NULL
  AND hierarchylevel = (
    SELECT {hierarchy_operator}(hierarchylevel)
    FROM organisationunit
    WHERE hierarchylevel IS NOT NULL
  )
""".strip()
    else:
        predicate = "trim(coalesce(name, '')) <> ''"

    data_sql = f"""
SELECT
  {select_sql}
FROM organisationunit
WHERE {predicate}
ORDER BY name ASC
LIMIT ? OFFSET ?
""".strip()

    count_sql = f"""
SELECT COUNT(*) AS total_count
FROM organisationunit
WHERE {predicate}
""".strip()

    return data_sql, [page_size, offset], count_sql, [], rename_map


def answer_orgunit_metadata_question(
    question: str,
    db_path: str,
    page: int,
    page_size: int,
    include_insights: bool,
    include_rows: bool,
    timings: Optional[Dict[str, float]] = None,
    resolved_plan_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data_sql, data_params, count_sql, count_params, rename_map = build_orgunit_metadata_sql(question, page_size, page)
    count_started_at = perf_counter()
    count_columns, count_rows = run_sql(db_path, count_sql, count_params)
    record_timing(timings, "row_count_query_ms", count_started_at)
    row_count = int(count_rows[0][count_columns[0]]) if count_rows else 0

    if include_rows:
        data_started_at = perf_counter()
        columns, rows = run_sql(db_path, data_sql, data_params)
        record_timing(timings, "data_query_ms", data_started_at)
        frontend_columns, frontend_rows = grouped_rows_to_frontend(columns, rows, rename_map)
    else:
        frontend_columns, frontend_rows = [], []

    temp_plan = QueryPlan(question=question, intent="records", page=page, page_size=page_size)
    insights = None
    if include_insights:
        insight_started_at = perf_counter()
        insights = build_basic_insights(temp_plan, "records", frontend_columns, frontend_rows, row_count)
        record_timing(timings, "insight_query_ms", insight_started_at)

    return make_json_safe(
        {
            "view": "records",
            "columns": frontend_columns,
            "rows": frontend_rows,
            "row_count": row_count,
            "insights": insights,
            "insights_available": True,
            "resolved_plan": make_json_safe(resolved_plan_payload or resolved_orgunit_plan_payload(question)),
            "sql_queries": [
                {"label": "organisationunit_count_query", "sql": count_sql, "params": count_params},
                {"label": "organisationunit_data_query", "sql": data_sql, "params": data_params},
            ],
        }
    )


def build_where_clause(plan: QueryPlan) -> Tuple[str, List[Any]]:
    clauses: List[str] = []
    params: List[Any] = []

    if plan.start_date:
        clauses.append("startdate >= ?")
        params.append(plan.start_date)
    if plan.end_date:
        clauses.append("startdate < ?")
        params.append(plan.end_date)

    if plan.orgunits:
        if plan.orgunit_ids:
            placeholders = ", ".join("?" for _ in plan.orgunit_ids)
            clauses.append(f"sourceid IN ({placeholders})")
            params.extend(plan.orgunit_ids)
        else:
            orgunit_clauses = ["orgunit_name = ? COLLATE NOCASE" for _ in plan.orgunits]
            clauses.append("(" + " OR ".join(orgunit_clauses) + ")")
            params.extend(plan.orgunits)

    metric_clause, metric_params = build_metric_clause(plan.metric_groups, plan.metric_group_ids)
    if metric_clause:
        clauses.append(metric_clause)
        params.extend(metric_params)

    if plan.period_type:
        clauses.append("period_type = ? COLLATE NOCASE")
        params.append(plan.period_type)

    if plan.followup is not None:
        clauses.append("followup = ?")
        params.append(1 if plan.followup else 0)

    if plan.value_filter:
        operator, left_value, right_value = plan.value_filter
        clauses.append("value_num IS NOT NULL")
        if operator == "between" and right_value is not None:
            clauses.append("value_num BETWEEN ? AND ?")
            params.extend([left_value, right_value])
        else:
            clauses.append(f"value_num {operator} ?")
            params.append(left_value)

    return " AND ".join(clauses) if clauses else "TRUE", params


def run_sql(db_path: str, sql: str, params: Sequence[Any]) -> Tuple[List[str], List[sqlite3.Row]]:
    with get_sqlite_connection(db_path) as connection:
        cursor = connection.execute(sql, params)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description] if cursor.description else []
    return columns, rows


def build_records_sql(db_path: str, plan: QueryPlan, where_sql: str) -> Tuple[str, str]:
    base_cte_sql = build_base_cte_sql(db_path, plan)
    data_sql = f"""
{base_cte_sql}
SELECT
    orgunit_name,
    dataelement_name,
    startdate,
    enddate,
    period_type,
    value,
    value_num,
    followup
FROM base
WHERE {where_sql}
ORDER BY startdate DESC, orgunit_name ASC, dataelement_name ASC
LIMIT ? OFFSET ?
""".strip()

    count_sql = f"""
{base_cte_sql}
SELECT COUNT(*) AS total_count
FROM base
WHERE {where_sql}
""".strip()
    return data_sql, count_sql


def build_summary_sql(db_path: str, plan: QueryPlan, where_sql: str) -> str:
    base_cte_sql = build_base_cte_sql(db_path, plan)
    return f"""
{base_cte_sql}
SELECT
    COALESCE(SUM(value_num), 0) AS total_value,
    COUNT(*) AS record_count
FROM base
WHERE {where_sql}
""".strip()


def build_ranking_sql(db_path: str, plan: QueryPlan, where_sql: str, least_first: bool) -> Tuple[str, str]:
    base_cte_sql = build_base_cte_sql(db_path, plan)
    order_direction = "ASC" if least_first else "DESC"
    grouped_cte = f"""
{base_cte_sql},
ranked AS (
    SELECT
        orgunit_name,
        COALESCE(SUM(value_num), 0) AS total_value
    FROM base
    WHERE {where_sql}
    GROUP BY orgunit_name
)
""".strip()

    data_sql = f"""
{grouped_cte}
SELECT
    orgunit_name,
    total_value
FROM ranked
ORDER BY total_value {order_direction}, orgunit_name ASC
LIMIT ? OFFSET ?
""".strip()

    count_sql = f"""
{grouped_cte}
SELECT COUNT(*) AS total_count
FROM ranked
""".strip()

    return data_sql, count_sql


def build_peak_sql(db_path: str, plan: QueryPlan, where_sql: str) -> Tuple[str, str]:
    base_cte_sql = build_base_cte_sql(db_path, plan)
    grouped_cte = f"""
{base_cte_sql},
peaks AS (
    SELECT
        dataelement_name,
        startdate,
        enddate,
        COALESCE(SUM(value_num), 0) AS total_value
    FROM base
    WHERE {where_sql}
    GROUP BY dataelement_name, startdate, enddate
)
""".strip()

    data_sql = f"""
{grouped_cte}
SELECT
    dataelement_name,
    startdate,
    enddate,
    total_value
FROM peaks
ORDER BY total_value DESC, startdate ASC, dataelement_name ASC
LIMIT ? OFFSET ?
""".strip()

    count_sql = f"""
{grouped_cte}
SELECT COUNT(*) AS total_count
FROM peaks
""".strip()

    return data_sql, count_sql


def build_comparison_sql(plan: QueryPlan, db_path: str, where_sql: str, base_params: Sequence[Any]) -> Tuple[str, List[Any]]:
    base_cte_sql = build_base_cte_sql(db_path, plan)
    left_group = plan.metric_groups[0] if plan.metric_groups else ["positive"]
    right_group = plan.metric_groups[1] if len(plan.metric_groups) > 1 else ["negative"]
    left_ids = plan.metric_group_ids[0] if plan.metric_group_ids else []
    right_ids = plan.metric_group_ids[1] if len(plan.metric_group_ids) > 1 else []

    left_clause, left_params = build_metric_group_condition(left_group, left_ids)
    right_clause, right_params = build_metric_group_condition(right_group, right_ids)
    if not left_clause:
        left_clause = "1=0"
        left_params = []
    if not right_clause:
        right_clause = "1=0"
        right_params = []

    sql = f"""
{base_cte_sql}
SELECT
    ? AS left_term,
    COALESCE(SUM(CASE WHEN {left_clause} THEN value_num ELSE 0 END), 0) AS left_total,
    ? AS right_term,
    COALESCE(SUM(CASE WHEN {right_clause} THEN value_num ELSE 0 END), 0) AS right_total,
    COALESCE(SUM(CASE WHEN {left_clause} THEN value_num ELSE 0 END), 0) -
    COALESCE(SUM(CASE WHEN {right_clause} THEN value_num ELSE 0 END), 0) AS difference
FROM base
WHERE {where_sql}
""".strip()

    params = [
        " ".join(left_group),
        *left_params,
        " ".join(right_group),
        *right_params,
        *left_params,
        *right_params,
        *base_params,
    ]
    return sql, params


def format_number(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def row_value(row: sqlite3.Row, key: str) -> Any:
    return row[key]


def rows_to_dicts(columns: Sequence[str], rows: Sequence[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [{column: row_value(row, column) for column in columns} for row in rows]


def rows_to_frontend_table(columns: Sequence[str], rows: Sequence[sqlite3.Row]) -> Tuple[List[str], List[List[Any]]]:
    if not rows:
        return [], []

    frontend_columns = ["Organisation Unit", "Metric", "Start Date", "End Date", "Period Type", "Value", "Follow-up"]
    frontend_rows: List[List[Any]] = []

    for row in rows:
        numeric_value = row_value(row, "value_num")
        raw_value = row_value(row, "value")
        displayed_value = numeric_value if numeric_value is not None else raw_value

        frontend_rows.append(
            [
                row_value(row, "orgunit_name"),
                row_value(row, "dataelement_name"),
                row_value(row, "startdate"),
                row_value(row, "enddate"),
                row_value(row, "period_type"),
                format_number(displayed_value),
                bool(row_value(row, "followup")),
            ]
        )

    return frontend_columns, frontend_rows


def grouped_rows_to_frontend(columns: Sequence[str], rows: Sequence[sqlite3.Row], rename_map: Dict[str, str]) -> Tuple[List[str], List[List[Any]]]:
    frontend_columns = [rename_map.get(column, column) for column in columns]
    frontend_rows: List[List[Any]] = []
    for row in rows:
        frontend_rows.append([format_number(row_value(row, column)) for column in columns])
    return frontend_columns, frontend_rows


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return format_number(obj)


def build_chart_summary_where_clause(plan: QueryPlan) -> Tuple[str, List[Any]]:
    clauses: List[str] = []
    params: List[Any] = []

    if plan.start_date:
        clauses.append("period_month >= ?")
        params.append(plan.start_date)
    if plan.end_date:
        clauses.append("period_month < ?")
        params.append(plan.end_date)

    if plan.orgunits:
        if plan.orgunit_ids:
            placeholders = ", ".join("?" for _ in plan.orgunit_ids)
            clauses.append(f"sourceid IN ({placeholders})")
            params.extend(plan.orgunit_ids)
        else:
            orgunit_clauses = ["orgunit_name = ? COLLATE NOCASE" for _ in plan.orgunits]
            clauses.append("(" + " OR ".join(orgunit_clauses) + ")")
            params.extend(plan.orgunits)

    metric_clause, metric_params = build_metric_clause(plan.metric_groups, plan.metric_group_ids)
    if metric_clause:
        clauses.append(metric_clause)
        params.extend(metric_params)

    if plan.period_type:
        clauses.append("period_type = ? COLLATE NOCASE")
        params.append(plan.period_type)

    if plan.followup is not None:
        clauses.append("followup = ?")
        params.append(1 if plan.followup else 0)

    return " AND ".join(clauses) if clauses else "TRUE", params


def build_chart_query(
    db_path: str,
    plan: QueryPlan,
    where_sql: str,
    params: Sequence[Any],
) -> Tuple[str, List[Any]]:
    if should_use_monthly_summary(db_path, plan):
        summary_where_sql, summary_params = build_chart_summary_where_clause(plan)
        return (
            f"""
SELECT
    period_month AS period,
    orgunit_name,
    dataelement_name,
    total_value
FROM {MONTHLY_SUMMARY_TABLE_NAME}
WHERE {summary_where_sql}
ORDER BY period ASC
""".strip(),
            summary_params,
        )

    return (
        f"""
{build_base_cte_sql(db_path, plan)}
SELECT
    date(startdate, 'start of month') AS period,
    orgunit_name,
    dataelement_name,
    COALESCE(SUM(value_num), 0) AS total_value
FROM base
WHERE {where_sql}
GROUP BY 1, 2, 3
ORDER BY 1 ASC
""".strip(),
        list(params),
    )


def numeric_total(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def unique_non_blank(values: Sequence[Any]) -> List[str]:
    seen = set()
    unique_values: List[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique_values.append(cleaned)
    return unique_values


def aggregate_totals(chart_rows: Sequence[Dict[str, Any]], key: str) -> List[Tuple[str, float]]:
    totals: Dict[str, float] = {}
    for row in chart_rows:
        label = str(row.get(key) or "").strip()
        if not label:
            continue
        totals[label] = totals.get(label, 0.0) + numeric_total(row.get("total_value"))
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))


def aggregate_by_period(chart_rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, float]]:
    totals: Dict[str, float] = {}
    for row in chart_rows:
        period = str(row.get("period") or "").strip()
        if not period:
            continue
        totals[period] = totals.get(period, 0.0) + numeric_total(row.get("total_value"))
    return sorted(totals.items(), key=lambda item: item[0])


def build_dashboard_cards(chart_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total": format_number(sum(numeric_total(row.get("total_value")) for row in chart_rows)),
        "metrics": len(unique_non_blank(row.get("dataelement_name") for row in chart_rows)),
        "orgs": len(unique_non_blank(row.get("orgunit_name") for row in chart_rows)),
    }


def build_metric_breakdown(chart_rows: Sequence[Dict[str, Any]], top_n: int = 10) -> Dict[str, Any]:
    top_metrics = aggregate_totals(chart_rows, "dataelement_name")[:top_n]
    return {
        "type": "bar_metrics",
        "title": "Top Metrics",
        "data": [{"name": name, "total": format_number(total)} for name, total in top_metrics],
    }


def build_org_breakdown(chart_rows: Sequence[Dict[str, Any]], top_n: int = 8) -> Dict[str, Any]:
    org_totals = aggregate_totals(chart_rows, "orgunit_name")
    top_orgs = org_totals[:top_n]
    other_total = sum(total for _, total in org_totals[top_n:])
    data = [{"name": name, "total": format_number(total)} for name, total in top_orgs]
    if other_total > 0:
        data.append({"name": "Other", "total": format_number(other_total)})
    return {
        "type": "bar_orgs",
        "title": "Top Organisations",
        "data": data,
    }


def build_trend(chart_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    trend_points = aggregate_by_period(chart_rows)
    return {
        "type": "line_trend",
        "title": "Trend Over Time",
        "data": [{"date": period, "total": format_number(total)} for period, total in trend_points],
    }


def build_metric_trend(chart_rows: Sequence[Dict[str, Any]], top_n: int = 5) -> Dict[str, Any]:
    top_metric_names = [name for name, _ in aggregate_totals(chart_rows, "dataelement_name")[:top_n]]
    grouped: Dict[str, Dict[str, float]] = {}

    for row in chart_rows:
        metric_name = str(row.get("dataelement_name") or "").strip()
        period = str(row.get("period") or "").strip()
        if metric_name not in top_metric_names or not period:
            continue
        metric_periods = grouped.setdefault(metric_name, {})
        metric_periods[period] = metric_periods.get(period, 0.0) + numeric_total(row.get("total_value"))

    series = []
    for metric_name in top_metric_names:
        points = grouped.get(metric_name, {})
        if not points:
            continue
        series.append(
            {
                "metric": metric_name,
                "data": [
                    {"date": period, "total": format_number(total)}
                    for period, total in sorted(points.items(), key=lambda item: item[0])
                ],
            }
        )

    return {
        "type": "multi_line_metric_trend",
        "title": "Metric Trends Over Time",
        "series": series,
    }


def build_dashboard_insights_from_rows(chart_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not chart_rows:
        return {"mode": "none"}

    insights = {
        "mode": "dashboard",
        "cards": build_dashboard_cards(chart_rows),
        "charts": [],
    }

    periods = unique_non_blank(row.get("period") for row in chart_rows)
    metric_names = unique_non_blank(row.get("dataelement_name") for row in chart_rows)
    org_names = unique_non_blank(row.get("orgunit_name") for row in chart_rows)

    has_time = len(periods) > 1
    multi_metric = len(metric_names) > 1
    multi_org = len(org_names) > 1

    if has_time:
        if multi_metric:
            metric_trend = build_metric_trend(chart_rows)
            if metric_trend["series"]:
                insights["charts"].append(metric_trend)
        else:
            insights["charts"].append(build_trend(chart_rows))

    if multi_org:
        org_breakdown = build_org_breakdown(chart_rows)
        if org_breakdown["data"]:
            insights["charts"].append(org_breakdown)

    if multi_metric:
        metric_breakdown = build_metric_breakdown(chart_rows)
        if metric_breakdown["data"]:
            insights["charts"].append(metric_breakdown)

    if not insights["charts"] and not chart_rows:
        return {"mode": "none"}

    return insights


def build_dashboard_insights(
    db_path: str,
    plan: QueryPlan,
    where_sql: str,
    params: Sequence[Any],
) -> Dict[str, Any]:
    chart_sql, chart_params = build_chart_query(db_path, plan, where_sql, params)
    chart_columns, chart_rows = run_sql(db_path, chart_sql, chart_params)
    chart_dict_rows = rows_to_dicts(chart_columns, chart_rows)
    return build_dashboard_insights_from_rows(chart_dict_rows)


def build_basic_insights(plan: QueryPlan, view: str, columns: Sequence[str], rows: Sequence[Sequence[Any]], row_count: int) -> Dict[str, Any]:
    insight = {
        "mode": view,
        "matched_rows": row_count,
        "filters": {
            "orgunits": plan.orgunits,
            "metric_groups": plan.metric_groups,
            "period_type": plan.period_type,
            "followup": plan.followup,
            "value_filter": plan.value_filter,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
        },
    }

    if rows and columns:
        insight["preview"] = [dict(zip(columns, row)) for row in rows[:5]]

    return insight


def build_relaxed_plan_for_empty_results(plan: QueryPlan, db_path: str) -> Optional[QueryPlan]:
    if not plan.metric_groups:
        return None

    removable_tokens = {singularize(token) for token in GENERIC_METRIC_TOKENS}
    relaxed_groups: List[List[str]] = []
    changed = False

    for group in plan.metric_groups:
        reduced_group = [token for token in group if singularize(token) not in removable_tokens]
        if not reduced_group:
            reduced_group = [group[0]]
        resolved_group = resolve_metric_tokens(db_path, reduced_group)
        if resolved_group != group:
            changed = True
        relaxed_groups.append(resolved_group)

    if not changed:
        return None

    return replace(
        plan,
        metric_groups=relaxed_groups,
        metric_group_ids=resolve_metric_group_ids(db_path, relaxed_groups),
    )


def execute_records_view(
    plan: QueryPlan,
    db_path: str,
    where_sql: str,
    params: Sequence[Any],
    include_insights: bool,
    include_rows: bool,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    data_sql, count_sql = build_records_sql(db_path, plan, where_sql)
    offset = (plan.page - 1) * plan.page_size

    count_started_at = perf_counter()
    count_columns, count_rows = run_sql(db_path, count_sql, params)
    record_timing(timings, "row_count_query_ms", count_started_at)
    row_count = int(count_rows[0][count_columns[0]]) if count_rows else 0

    if include_rows:
        data_started_at = perf_counter()
        columns, rows = run_sql(db_path, data_sql, [*params, plan.page_size, offset])
        record_timing(timings, "data_query_ms", data_started_at)
        frontend_columns, frontend_rows = rows_to_frontend_table(columns, rows)
    else:
        frontend_columns, frontend_rows = [], []

    insights = None
    if include_insights:
        insight_started_at = perf_counter()
        insights = build_dashboard_insights(db_path, plan, where_sql, params)
        record_timing(timings, "insight_query_ms", insight_started_at)
    insights_available = (
        row_count > 0 if not include_insights else (insights or {}).get("mode") != "none"
    )
    return make_json_safe(
        {
            "view": "records",
            "columns": frontend_columns,
            "rows": frontend_rows,
            "row_count": row_count,
            "insights": insights,
            "insights_available": insights_available,
        }
    )


def execute_summary_view(
    plan: QueryPlan,
    db_path: str,
    where_sql: str,
    params: Sequence[Any],
    include_insights: bool,
    include_rows: bool,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    sql = build_summary_sql(db_path, plan, where_sql)
    data_started_at = perf_counter()
    columns, rows = run_sql(db_path, sql, params)
    record_timing(timings, "data_query_ms", data_started_at)

    total_value = format_number(rows[0][columns[0]]) if rows else 0
    record_count = format_number(rows[0][columns[1]]) if rows else 0

    frontend_columns = ["Total", "Records"] if include_rows else []
    frontend_rows = [[total_value, record_count]] if include_rows else []
    insights = None
    if include_insights:
        insight_started_at = perf_counter()
        insights = build_dashboard_insights(db_path, plan, where_sql, params)
        record_timing(timings, "insight_query_ms", insight_started_at)
    insights_available = (
        record_count > 0 if not include_insights else (insights or {}).get("mode") != "none"
    )

    return make_json_safe(
        {
            "view": "summary",
            "columns": frontend_columns,
            "rows": frontend_rows,
            "row_count": 1,
            "insights": insights,
            "insights_available": insights_available,
        }
    )


def execute_ranking_view(
    plan: QueryPlan,
    db_path: str,
    where_sql: str,
    params: Sequence[Any],
    include_insights: bool,
    include_rows: bool,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    least_first = bool(re.search(r"\b(least|lowest)\b", plan.question.lower()))
    data_sql, count_sql = build_ranking_sql(db_path, plan, where_sql, least_first=least_first)
    offset = (plan.page - 1) * plan.page_size

    count_started_at = perf_counter()
    count_columns, count_rows = run_sql(db_path, count_sql, params)
    record_timing(timings, "row_count_query_ms", count_started_at)
    row_count = int(count_rows[0][count_columns[0]]) if count_rows else 0

    if include_rows:
        data_started_at = perf_counter()
        columns, rows = run_sql(db_path, data_sql, [*params, plan.page_size, offset])
        record_timing(timings, "data_query_ms", data_started_at)
        frontend_columns, frontend_rows = grouped_rows_to_frontend(
            columns,
            rows,
            {"orgunit_name": "Organisation Unit", "total_value": "Total"},
        )
    else:
        frontend_columns, frontend_rows = [], []

    insights = None
    if include_insights:
        insight_started_at = perf_counter()
        insights = build_dashboard_insights(db_path, plan, where_sql, params)
        record_timing(timings, "insight_query_ms", insight_started_at)
    insights_available = (
        row_count > 0 if not include_insights else (insights or {}).get("mode") != "none"
    )
    return make_json_safe(
        {
            "view": "records",
            "columns": frontend_columns,
            "rows": frontend_rows,
            "row_count": row_count,
            "insights": insights,
            "insights_available": insights_available,
        }
    )


def execute_peak_view(
    plan: QueryPlan,
    db_path: str,
    where_sql: str,
    params: Sequence[Any],
    include_insights: bool,
    include_rows: bool,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    data_sql, count_sql = build_peak_sql(db_path, plan, where_sql)
    summary_sql = build_summary_sql(db_path, plan, where_sql)
    offset = (plan.page - 1) * plan.page_size

    count_started_at = perf_counter()
    count_columns, count_rows = run_sql(db_path, count_sql, params)
    record_timing(timings, "row_count_query_ms", count_started_at)
    row_count = int(count_rows[0][count_columns[0]]) if count_rows else 0
    summary_started_at = perf_counter()
    summary_columns, summary_rows = run_sql(db_path, summary_sql, params)
    record_timing(timings, "summary_query_ms", summary_started_at)
    overall_total = format_number(summary_rows[0][summary_columns[0]]) if summary_rows else 0
    overall_record_count = format_number(summary_rows[0][summary_columns[1]]) if summary_rows else 0
    data_started_at = perf_counter()
    columns, rows = run_sql(db_path, data_sql, [*params, plan.page_size, offset])
    record_timing(timings, "data_query_ms", data_started_at)

    if include_rows:
        frontend_columns, frontend_rows = grouped_rows_to_frontend(
            columns,
            rows,
            {
                "dataelement_name": "Metric",
                "startdate": "Start Date",
                "enddate": "End Date",
                "total_value": "Total",
            },
        )
    else:
        frontend_columns, frontend_rows = [], []

    insights = None
    if include_insights:
        insight_started_at = perf_counter()
        insights = build_dashboard_insights(db_path, plan, where_sql, params)
        record_timing(timings, "insight_query_ms", insight_started_at)
    insights_available = (
        row_count > 0 if not include_insights else (insights or {}).get("mode") != "none"
    )
    answer = None
    if rows:
        top_row = rows[0]
        metric_name = row_value(top_row, "dataelement_name")
        start_date = row_value(top_row, "startdate")
        end_date = row_value(top_row, "enddate")
        peak_total = format_number(row_value(top_row, "total_value"))
        period_text = f"{start_date} to {end_date}" if start_date and end_date else (start_date or end_date or "the matched period")
        answer = (
            f"The peak for '{metric_name}' was {peak_total} during {period_text}. "
            f"Across all matched records in this filtered range, the total was {overall_total} from {overall_record_count} reported rows."
        )

    return make_json_safe(
        {
            "view": "records",
            "answer": answer,
            "columns": frontend_columns,
            "rows": frontend_rows,
            "row_count": row_count,
            "message": answer,
            "insights": insights,
            "insights_available": insights_available,
        }
    )


def execute_comparison_view(
    plan: QueryPlan,
    db_path: str,
    where_sql: str,
    params: Sequence[Any],
    include_insights: bool,
    include_rows: bool,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    sql, sql_params = build_comparison_sql(plan, db_path, where_sql, params)
    data_started_at = perf_counter()
    columns, rows = run_sql(db_path, sql, sql_params)
    record_timing(timings, "data_query_ms", data_started_at)

    if include_rows:
        frontend_columns, frontend_rows = grouped_rows_to_frontend(
            columns,
            rows,
            {
                "left_term": "Left Term",
                "left_total": "Left Total",
                "right_term": "Right Term",
                "right_total": "Right Total",
                "difference": "Difference",
            },
        )
    else:
        frontend_columns, frontend_rows = [], []

    insights = None
    if include_insights:
        insight_started_at = perf_counter()
        insights = build_dashboard_insights(db_path, plan, where_sql, params)
        record_timing(timings, "insight_query_ms", insight_started_at)
    insights_available = (
        True if not include_insights else (insights or {}).get("mode") != "none"
    )
    return make_json_safe(
        {
            "view": "records",
            "columns": frontend_columns,
            "rows": frontend_rows,
            "row_count": 1,
            "insights": insights,
            "insights_available": insights_available,
        }
    )


def execute_query_plan_once(
    plan: QueryPlan,
    db_path: str,
    include_insights: bool,
    include_rows: bool,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    where_sql, params = build_where_clause(plan)

    if plan.debug:
        print("\n===== QUERY PLAN =====")
        print(json.dumps(make_json_safe(plan.__dict__), indent=2))
        print("\n===== WHERE SQL =====")
        print(where_sql)
        print("\n===== PARAMS =====")
        print(json.dumps(make_json_safe(list(params)), indent=2))
        print("======================\n")

    if plan.intent == "summary":
        return execute_summary_view(plan, db_path, where_sql, params, include_insights, include_rows, timings=timings)
    if plan.intent == "ranking":
        return execute_ranking_view(plan, db_path, where_sql, params, include_insights, include_rows, timings=timings)
    if plan.intent == "peak":
        return execute_peak_view(plan, db_path, where_sql, params, include_insights, include_rows, timings=timings)
    if plan.intent == "comparison":
        return execute_comparison_view(plan, db_path, where_sql, params, include_insights, include_rows, timings=timings)
    return execute_records_view(plan, db_path, where_sql, params, include_insights, include_rows, timings=timings)


def execute_query_plan(
    plan: QueryPlan,
    db_path: str,
    include_insights: bool,
    include_rows: bool,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    result = execute_query_plan_once(plan, db_path, include_insights, include_rows, timings=timings)
    if result.get("row_count", 0):
        result["sql_queries"] = make_json_safe(build_sql_query_entries(plan, db_path))
        return make_json_safe(result)

    relaxed_plan = build_relaxed_plan_for_empty_results(plan, db_path)
    if relaxed_plan is None:
        result["sql_queries"] = make_json_safe(build_sql_query_entries(plan, db_path))
        return make_json_safe(result)

    if plan.debug:
        print("Retrying with relaxed metric groups after empty result.\n")
    relaxed_result = execute_query_plan_once(relaxed_plan, db_path, include_insights, include_rows, timings=timings)
    relaxed_result["sql_queries"] = make_json_safe(build_sql_query_entries(relaxed_plan, db_path))
    return make_json_safe(relaxed_result)


def answer_question(
    question: str,
    debug: bool = False,
    page: int = 1,
    page_size: int = 100,
    include_insights: bool = False,
    include_rows: bool = True,
    include_debug_trace: bool = False,
    prefer_sqlcoder_first: Optional[bool] = None,
    resolved_plan: Optional[Any] = None,
) -> Dict[str, Any]:
    request_started_at = perf_counter()
    timings: Optional[Dict[str, float]] = {} if include_debug_trace else None
    db_path = Path(build_conn_str_from_parts()).expanduser().resolve()
    if not db_path.exists():
        raise ValueError(f"SQLite database not found: {db_path}")

    question = (question or "").strip()
    normalized_resolved_plan = normalize_resolved_plan_payload(resolved_plan)
    if not question and not normalized_resolved_plan:
        raise ValueError("No question provided.")

    row_limit = int(os.environ.get("ROW_LIMIT", "400"))
    route_resolution_started_at = perf_counter()
    route_kind, plan, resolved_plan_payload, resolved_question = resolve_runtime_plan(
        question=question,
        db_path=str(db_path),
        page=page,
        page_size=page_size,
        row_limit=row_limit,
        debug=debug,
        timings=timings,
        resolved_plan=normalized_resolved_plan,
    )
    record_timing(timings, "route_resolution_ms", route_resolution_started_at)
    if normalized_resolved_plan:
        record_timing(timings, "resolved_plan_reuse_ms", route_resolution_started_at)

    if route_kind == "explainable" and plan is not None:
        result = answer_explainable_question(
            question=resolved_question,
            db_path=str(db_path),
            row_limit=row_limit,
            debug=debug,
            page=page,
            page_size=page_size,
            include_insights=include_insights,
            include_rows=include_rows,
            include_debug_trace=include_debug_trace,
            prefer_sqlcoder_first=prefer_sqlcoder_first,
            timings=timings,
            plan=plan,
            resolved_plan_payload=resolved_plan_payload,
        )
        record_timing(timings, "python_total_request_ms", request_started_at)
        return attach_debug_trace_timings(result, timings)

    if route_kind == "orgunit_metadata":
        execution_started_at = perf_counter()
        result = answer_orgunit_metadata_question(
            question=resolved_question,
            db_path=str(db_path),
            page=page,
            page_size=page_size,
            include_insights=include_insights,
            include_rows=include_rows,
            timings=timings,
            resolved_plan_payload=resolved_plan_payload,
        )
        record_timing(timings, "query_execution_ms", execution_started_at)
        record_timing(timings, "python_total_request_ms", request_started_at)
        return attach_debug_trace_timings(result, timings)

    if plan is None:
        raise ValueError("No executable query plan could be resolved.")

    if plan.intent in METRIC_REQUIRED_INTENTS and not plan.metric_groups:
        message = (
            "I could not match that question to a supported DHIS2 metric or indicator. "
            "Returning a broad database total here would be misleading."
        )
        if debug:
            print(message)
        result = build_unmatched_result(plan, include_insights=include_insights, message=message)
        result["resolved_plan"] = make_json_safe(resolved_plan_payload or resolved_standard_plan_payload(plan))
        record_timing(timings, "python_total_request_ms", request_started_at)
        return attach_debug_trace_timings(result, timings)

    execution_started_at = perf_counter()
    result = execute_query_plan(
        plan,
        str(db_path),
        include_insights=include_insights,
        include_rows=include_rows,
        timings=timings,
    )
    result["resolved_plan"] = make_json_safe(resolved_plan_payload or resolved_standard_plan_payload(plan))
    record_timing(timings, "query_execution_ms", execution_started_at)
    record_timing(timings, "python_total_request_ms", request_started_at)
    return attach_debug_trace_timings(result, timings)


def print_table(columns: Sequence[str], rows: Sequence[Sequence[Any]], limit: int = 20) -> None:
    if not columns:
        print("(No rows)")
        return

    preview_rows = list(rows[:limit])
    widths = [len(str(column)) for column in columns]
    for row in preview_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    header = " | ".join(str(column).ljust(widths[index]) for index, column in enumerate(columns))
    divider = "-+-".join("-" * width for width in widths)
    print(header)
    print(divider)
    for row in preview_rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def print_sql_queries(sql_queries: Sequence[Dict[str, Any]]) -> None:
    if not sql_queries:
        print("(No SQL query recorded)")
        return

    for query in sql_queries:
        label = query.get("label", "query")
        sql_text = query.get("sql", "")
        params = query.get("params", [])
        print(f"\n[{label}]")
        print(sql_text or "(empty)")
        print("\nParams:")
        print(json.dumps(make_json_safe(list(params)), indent=2))


def build_cli_output_result(result: Dict[str, Any]) -> Dict[str, Any]:
    cli_result = dict(result)
    for key in (
        "fallback_answer",
        "model_answer",
        "general_model_answer",
        "grounded_model_answer",
        "patched_model_answer",
        "model_attempted",
    ):
        cli_result.pop(key, None)
    return cli_result


def main() -> None:
    question = input("\nEnter your question: ").strip()
    if not question:
        raise SystemExit("ERROR: No question provided.")

    debug_enabled = os.environ.get("NEW_DEBUG", "").strip().lower() in {"1", "true", "yes"}

    try:
        result = answer_question(question, debug=debug_enabled, page=1, page_size=200, include_insights=True)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print("\n================ API STYLE OUTPUT ================")
    print(json.dumps(make_json_safe(build_cli_output_result(result)), indent=2))

    print("\n================ SQL QUERIES ================")
    print_sql_queries(result.get("sql_queries", []))

    if result.get("message"):
        print("\n================ MESSAGE ================")
        print(result["message"])

    if result.get("view") == "explainable":
        print("\n================ EXPLAINABLE ANSWER ================")
        print(result.get("answer", ""))
        if result.get("columns") and result.get("rows"):
            print("\n================ RETRIEVED EVIDENCE (first 20) ================")
            print_table(result.get("columns", []), result.get("rows", []), limit=20)
    else:
        print("\n================ USER FRIENDLY TABLE (first 20) ================")
        print_table(result.get("columns", []), result.get("rows", []), limit=20)


if __name__ == "__main__":
    main()
