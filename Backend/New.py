import json
import os
import re
from calendar import monthrange
from datetime import date
from typing import Any, Dict, List, Tuple
from fastapi import params
import pandas as pd
import math
import numpy as np
from datetime import datetime, date
import psycopg2
from dateutil import parser as dateutil_parser
from openai import OpenAI
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

HF_TOKEN_FALLBACK = ""  # Optional: paste token here for local-only use.

# ---------------------------------------------------------------------
# Constants & schema mapping
# ---------------------------------------------------------------------

VALUE_KEYWORDS = (
    "value", "values", "case", "cases", "death", "deaths", "count", "counts",
    "test", "tests", "visits", "incidence", "coverage"
)
COMPARATORS = {">", "<", ">=", "<=", "="}

SUPPORTED_FILTERS: Dict[str, Dict[str, str]] = {
    "orgunit_name": {"column": "orgunit_name", "type": "text"},
    "orgunit_code": {"column": "orgunit_code", "type": "text"},
    "orgunit_uid": {"column": "orgunit_uid", "type": "text"},
    "hierarchylevel": {"column": "hierarchylevel", "type": "numeric"},
    "dataelement_name": {"column": "dataelement_name", "type": "text"},
    "dataelement_code": {"column": "dataelement_code", "type": "text"},
    "dataelement_uid": {"column": "dataelement_uid", "type": "text"},
    "period_type": {"column": "period_type", "type": "text"},
    "storedby": {"column": "storedby", "type": "text"},
    "followup": {"column": "followup", "type": "bool"},
    "value_num": {"column": "value_num", "type": "numeric"},
}

FILTER_ALIASES = {
    "org_unit": "orgunit_name",
    "orgunit": "orgunit_name",
    "organisation": "orgunit_name",
    "organisation_unit": "orgunit_name",
    "facility": "orgunit_name",
    "district": "orgunit_name",
    "ou": "orgunit_name",
    "unit": "orgunit_name",
    "data_element": "dataelement_name",
    "indicator": "dataelement_name",
    "indicator_code": "dataelement_code",
    "dataelement": "dataelement_name",
    "value": "value_num",
    "numeric_value": "value_num",
    "cases": "value_num",
    "counts": "value_num",
    "level": "hierarchylevel",
    "org_level": "hierarchylevel",
}

ORDERABLE_COLUMNS = {"startdate", "enddate", "lastupdated", "created"}

BASE_COLUMNS = {
    "dataelementid",
    "dataelement_name",
    "dataelement_code",
    "dataelement_uid",
    "organisationunitid",
    "orgunit_name",
    "orgunit_code",
    "orgunit_uid",
    "hierarchylevel",
    "periodid",
    "startdate",
    "enddate",
    "period_type",
    "value",
    "value_num",
    "comment",
    "storedby",
    "lastupdated",
    "created",
    "followup",
}

ALLOWED_SQL_KEYWORDS = {
    "and",
    "or",
    "not",
    "between",
    "ilike",
    "like",
    "in",
    "is",
    "null",
    "true",
    "false",
}

INDICATOR_SUFFIXES = {
    "cases",
    "case",
    "tests",
    "test",
    "visits",
    "visit",
    "clients",
    "treated",
    "given",
    "reported",
    "outbreak",
    "outbreaks"
}

def normalize_metrics(intent: dict) -> dict:
    """
    Removes indicator suffix words from metric list.
    Safe for production. Does not modify orgunit.
    """

    metrics = intent.get("metric") or []

    cleaned_metrics = []

    for metric in metrics:

        # split words
        words = metric.split()

        # remove suffix words
        filtered = [
            w for w in words
            if w.lower() not in INDICATOR_SUFFIXES
        ]

        cleaned = " ".join(filtered).strip()

        if cleaned:
            cleaned_metrics.append(cleaned)

    intent["metric"] = cleaned_metrics

    return intent

# ---------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------

def get_llm_client(token: str) -> OpenAI | None:
    if not token:
        return None
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

def run_sql_llm_prompt(prompt: str, token: str) -> str:
    client = get_llm_client(token)
    if client is None:
        raise ValueError("Missing HF_TOKEN for the Hugging Face router.")
    completion = client.chat.completions.create(
        model="defog/llama-3-sqlcoder-8b:featherless-ai",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    message = completion.choices[0].message
    return message.content or ""

def run_instruct_llm_prompt(prompt: str, token: str,payload: dict={},payload_include: bool=False) -> str:
    print("Entered run_instruct_llm_prompt")
    client = get_llm_client(token)
    if client is None:
        raise ValueError("Missing HF_TOKEN for the Hugging Face router.")
    
    if payload_include:
        response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload)}
        ],
        temperature=0
        )
        explanation = response.choices[0].message.content
        return explanation
    completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    message = completion.choices[0].message
    return message.content or ""

def preheat_database(conn_str: str):
    """
    A very small warm-up query to force PostgreSQL to load table metadata,
    cached plans, and buffer pages into memory.
    """
    get_cached_llm_plan.cache_clear()
    warm_sql = """
        WITH base AS (
            SELECT dv.periodid, dv.dataelementid
            FROM datavalue dv
            LIMIT 50
        )
        SELECT COUNT(*) FROM base;
    """
    try:
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(warm_sql)
    except Exception as e:
        print("⚠️ Database preheat skipped:", e)

# ---------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------

def looks_like_date_token(tok: str) -> bool:
    if not tok:
        return False
    # 2024 or 2024-05 or 2024-05-12
    if re.match(r"^20\d{2}([\/\-](0[1-9]|1[0-2])([\/\-](0[1-9]|[12]\d|3[01]))?)?$", tok):
        return True
    # 1/5/24, 01-05-2024 etc.
    if re.match(r"^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}$", tok):
        return True
    # Month name
    if re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
        tok,
        re.IGNORECASE,
    ):
        return True
    return False


def choose_date_column(question: str) -> str:
    if re.search(r"\b(updated|synced|entered|modified)\b", question, re.IGNORECASE):
        return "lastupdated"
    if re.search(r"\bend(?:\s|-)?date\b", question, re.IGNORECASE):
        return "enddate"
    return "startdate"


def normalize_date(dt: pd.Timestamp) -> pd.Timestamp:
    return dt.normalize()

def extract_date_range(question: str,) -> Tuple[str, pd.Timestamp | None, pd.Timestamp | None, str]:
    date_col = "startdate"

    q = question.lower()
    today = pd.Timestamp.today().normalize()

    start = None
    end = None

    LAST_WORDS = r"(?:last|previous|past|recent|trailing)"
    MONTH_WORDS = r"(?:month|months|mth|mths)"
    DAY_WORDS = r"(?:day|days)"

    m = re.search(r"(20\d{2})\s*(?:to|and|-)\s*(20\d{2})", q)
    if m:
        y1, y2 = sorted([int(m.group(1)), int(m.group(2))])
        start = pd.Timestamp(y1, 1, 1)
        end = pd.Timestamp(y2 + 1, 1, 1)
        return date_col, start, end, "DESC"

    m = re.search(rf"{LAST_WORDS}\s+(\d+)\s+{MONTH_WORDS}\s+of\s+(20\d{{2}})",q,)
    if m:
        print("Entered The date func")
        months = int(m.group(1))
        year = int(m.group(2))
        print(months)
        months = max(1, min(months, 12))

        start_month = 13 - months
        start = pd.Timestamp(year, start_month, 1)
        end = pd.Timestamp(year + 1, 1, 1)
        print(start,end)
        return date_col, start, end, "DESC"

    m = re.search(rf"{LAST_WORDS}\s+(\d+)\s+{MONTH_WORDS}\b", q)
    if m:
        months = int(m.group(2))
        end = today + pd.Timedelta(days=1)
        start = end - pd.DateOffset(months=months)
        return date_col, start, end, "DESC"

    m = re.search(rf"{LAST_WORDS}\s+(\d+)\s+{DAY_WORDS}\b", q)
    if m:
        days = int(m.group(2))
        end = today + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=days)
        return date_col, start, end, "DESC"

    m = re.search(r"(?:q|quarter\s*)([1-4])\s*(20\d{2})", q)
    if not m:
        m = re.search(r"([1-4])(st|nd|rd|th)?\s*quarter\s*(20\d{2})", q)

    if m:
        qtr = int(m.group(1))
        year = int(m.group(3) if len(m.groups()) == 3 else m.group(2))

        start_month = (qtr - 1) * 3 + 1
        start = pd.Timestamp(year, start_month, 1)
        end = start + pd.DateOffset(months=3)

        return date_col, start, end, "DESC"

    m = re.search(r"between\s+([a-z]+)\s+and\s+([a-z]+)\s+(20\d{2})", q)
    if m:
        m1 = pd.to_datetime(m.group(1), errors="coerce").month
        m2 = pd.to_datetime(m.group(2), errors="coerce").month
        year = int(m.group(3))

        start = pd.Timestamp(year, m1, 1)
        end = pd.Timestamp(year, m2, 1) + pd.DateOffset(months=1)

        return date_col, start, end, "DESC"

    m = re.search(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+of\s+([a-z]+)\s+to\s+(\d{1,2})(?:st|nd|rd|th)?\s+of\s+([a-z]+)\s+(20\d{2})",
        q,
    )
    if m:
        d1, mon1, d2, mon2, year = m.groups()

        start = pd.Timestamp(year, pd.to_datetime(mon1).month, int(d1))
        end = pd.Timestamp(year, pd.to_datetime(mon2).month, int(d2)) + pd.Timedelta(days=1)

        return date_col, start, end, "DESC"

    # --------------------------------------------------
    # 8️⃣ single year
    # --------------------------------------------------
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        year = int(m.group(1))
        start = pd.Timestamp(year, 1, 1)
        end = pd.Timestamp(year + 1, 1, 1)
        return date_col, start, end, "DESC"

    # --------------------------------------------------
    # sorting direction
    # --------------------------------------------------
    sort_dir = "DESC"
    if re.search(r"\b(oldest|earliest|ascending)\b", q):
        sort_dir = "ASC"

    return date_col, None, None, sort_dir


# ---------------------------------------------------------------------
# Numeric filters
# ---------------------------------------------------------------------

def detect_numeric_range(
    question: str, keywords: Tuple[str, ...]
) -> Tuple[str, float, float] | None:
    for kw in keywords:
        base = re.escape(kw).replace("\\ ", "[\\s_-]?")
        patterns = [
            rf"\b{base}\b.*?\b(?:range\s*(?:like\s*)?from)\s*(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)\b",
            rf"\b{base}\b\s*(?:from)\s*(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)\b",
            rf"\b{base}\b\s*(-?\d+(?:\.\d+)?)\s*[-]\s*(-?\d+(?:\.\d+)?)\b",
            rf"\b{base}\b\s*between\s*(-?\d+(?:\.\d+)?)\s*(?:and|-)\s*(-?\d+(?:\.\d+)?)\b",
            rf"\b{base}\s*(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)\b",
        ]
        for pat in patterns:
            m = re.search(pat, question, re.IGNORECASE)
            if m:
                low = float(m.group(1))
                high = float(m.group(2))
                if low > high:
                    low, high = high, low
                return ("range", low, high)
    return None


def detect_numeric_comparator(
    question: str, keywords: Tuple[str, ...]
) -> Tuple[str, float] | None:
    for kw in keywords:
        base = re.escape(kw).replace("\\ ", "[\\s_-]?")
        pattern = rf"\b{base}\b\s*(>=|<=|>|<|=)\s*(-?\d+(?:\.\d+)?)"
        m = re.search(pattern, question, re.IGNORECASE)
        if m:
            op = m.group(1)
            val = float(m.group(2))
            return (op, val)
    return None


# ---------------------------------------------------------------------
# Filter normalization
# ---------------------------------------------------------------------

def canonical_filter_name(name: str) -> str:
    key = name.lower()
    return FILTER_ALIASES.get(key, key)

def normalize_filter_value(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) == 3 and isinstance(value[0], str) and value[0].lower() == "range":
            return ("range", value[1], value[2])
        if len(value) == 2 and isinstance(value[0], str) and value[0] in COMPARATORS:
            return (value[0], value[1])
    if isinstance(value, dict):
        if {"min", "max"}.issubset(value.keys()):
            return ("range", value["min"], value["max"])
        if "op" in value and "value" in value and value["op"] in COMPARATORS:
            return (value["op"], value["value"])
    return value


def normalize_filters(raw_filters: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, val in raw_filters.items():
        if val in (None, ""):
            continue
        canonical = canonical_filter_name(str(key))
        if canonical not in SUPPORTED_FILTERS:
            continue
        normalized[canonical] = normalize_filter_value(val)
    return normalized


def serialize_filters_for_display(filters: Dict[str, Any]) -> Dict[str, Any]:
    display: Dict[str, Any] = {}
    for key, val in filters.items():
        if isinstance(val, tuple):
            display[key] = list(val)
        else:
            display[key] = val
    return display


# ---------------------------------------------------------------------
# WHERE clause builder
# ---------------------------------------------------------------------

def as_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes", "y"}
    return False


def build_where(
    filters: Dict[str, Any],
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
    date_col: str,
) -> Tuple[List[str], List[Any]]:
    where: List[str] = []
    params: List[Any] = []

    if start_dt is not None:
        where.append(f"{date_col} >= %s")
        params.append(start_dt.to_pydatetime())
    if end_dt is not None:
        where.append(f"{date_col} <= %s")
        params.append(end_dt.to_pydatetime())

    for key, val in filters.items():
        if key not in SUPPORTED_FILTERS:
            continue
        info = SUPPORTED_FILTERS[key]
        column = info["column"]
        dtype = info["type"]

        if isinstance(val, list):
            val = tuple(val)

        if isinstance(val, tuple):
            if len(val) == 3 and val[0] == "range":
                _, low, high = val
                low = float(low)
                high = float(high)
                where.append(f"{column} BETWEEN %s AND %s")
                params.extend([low, high])
                continue
            if len(val) == 2 and val[0] in COMPARATORS:
                op, num = val
                num = float(num)
                where.append(f"{column} {op} %s")
                params.append(num)
                continue

        if dtype == "bool":
            where.append(f"{column} = %s")
            params.append(as_bool(val))
        elif dtype == "numeric":
            try:
                num = float(val)
            except Exception:
                continue
            where.append(f"{column} = %s")
            params.append(num)
        else:
            where.append(f"{column} ILIKE %s")
            params.append(f"%{val}%")

    return where, params


# ---------------------------------------------------------------------
# Query builder & execution
# ---------------------------------------------------------------------

def build_query(
    select_sql: str,
    where_sql: str,
    order_by: str,
    group_by: str = ""
) -> str:

    if group_by:
        # Aggregated query
        return f"""
WITH base AS (
    SELECT
        dv.dataelementid,
        de.name AS dataelement_name,
        de.code AS dataelement_code,
        de.uid AS dataelement_uid,
        ou.organisationunitid,
        ou.name AS orgunit_name,
        ou.code AS orgunit_code,
        ou.uid AS orgunit_uid,
        ou.hierarchylevel,
        p.periodid,
        p.startdate,
        p.enddate,
        pt.name AS period_type,
        dv.value,
        CASE WHEN dv.value ~ '^[-]?\\d+(\\.\\d+)?$' THEN dv.value::numeric END AS value_num,
        dv.comment,
        dv.storedby,
        dv.lastupdated,
        dv.created,
        COALESCE(dv.followup, FALSE) AS followup
    FROM datavalue dv
    JOIN dataelement de ON dv.dataelementid = de.dataelementid
    JOIN organisationunit ou ON dv.sourceid = ou.organisationunitid
    JOIN period p ON dv.periodid = p.periodid
    LEFT JOIN periodtype pt ON p.periodtypeid = pt.periodtypeid
)
SELECT
    {select_sql}
FROM base
WHERE {where_sql}
GROUP BY {group_by}
ORDER BY {order_by}
LIMIT %s OFFSET %s
"""
    else:
        # Standard row query
        return f"""
WITH base AS (
    SELECT
        dv.dataelementid,
        de.name AS dataelement_name,
        de.code AS dataelement_code,
        de.uid AS dataelement_uid,
        ou.organisationunitid,
        ou.name AS orgunit_name,
        ou.code AS orgunit_code,
        ou.uid AS orgunit_uid,
        ou.hierarchylevel,
        p.periodid,
        p.startdate,
        p.enddate,
        pt.name AS period_type,
        dv.value,
        CASE WHEN dv.value ~ '^[-]?\\d+(\\.\\d+)?$' THEN dv.value::numeric END AS value_num,
        dv.comment,
        dv.storedby,
        dv.lastupdated,
        dv.created,
        COALESCE(dv.followup, FALSE) AS followup
    FROM datavalue dv
    JOIN dataelement de ON dv.dataelementid = de.dataelementid
    JOIN organisationunit ou ON dv.sourceid = ou.organisationunitid
    JOIN period p ON dv.periodid = p.periodid
    LEFT JOIN periodtype pt ON p.periodtypeid = pt.periodtypeid
)
SELECT
    *,
    COUNT(*) OVER() AS total_count
FROM base
WHERE {where_sql}
ORDER BY {order_by}
LIMIT %s OFFSET %s
"""

def is_summary_question(question: str) -> bool:
    q = question.lower().strip()
    return bool(
        re.search(
            r"\b(how many|total|sum|overall|number of|count|counts|counting)\b",
            q,
        )
    )

def build_summary_query(where_sql: str) -> str:
    """
    Summary = one row output.
    We prefer SUM(value_num). If value_num is null, SUM ignores nulls.
    """
    return f"""
    WITH base AS (
        SELECT
            dv.dataelementid,
            de.name AS dataelement_name,
            de.code AS dataelement_code,
            de.uid AS dataelement_uid,
            ou.organisationunitid,
            ou.name AS orgunit_name,
            ou.code AS orgunit_code,
            ou.uid AS orgunit_uid,
            ou.hierarchylevel,
            p.periodid,
            p.startdate,
            p.enddate,
            pt.name AS period_type,
            dv.value,
            CASE WHEN dv.value ~ '^[-]?\\d+(\\.\\d+)?$' THEN dv.value::numeric END AS value_num,
            dv.comment,
            dv.storedby,
            dv.lastupdated,
            dv.created,
            COALESCE(dv.followup, FALSE) AS followup
        FROM datavalue dv
        JOIN dataelement de ON dv.dataelementid = de.dataelementid
        JOIN organisationunit ou ON dv.sourceid = ou.organisationunitid
        JOIN period p ON dv.periodid = p.periodid
        LEFT JOIN periodtype pt ON p.periodtypeid = pt.periodtypeid
    )
    SELECT
        COALESCE(SUM(value_num), 0) AS total_value,
        COUNT(*) AS record_count
    FROM base
    WHERE {where_sql};
    """.strip()

def build_explainable_query(
    select_sql: str,
    where_sql: str,
    order_by: str,
    group_by: str = ""  
) -> str:
    """
    Builds SQL for explainable analytics queries.

    - Aggregated queries (group_by != "") → trend / ranking / totals
    - Non-aggregated queries → raw row inspection
    - No OFFSET (pagination removed)
    - Uses LIMIT %s by default (to match planner contract)
    """
    print("Entered build explainable query")
    base_cte = """
WITH base AS (
    SELECT
        dv.dataelementid,
        de.name AS dataelement_name,
        de.code AS dataelement_code,
        de.uid AS dataelement_uid,
        ou.organisationunitid,
        ou.name AS orgunit_name,
        ou.code AS orgunit_code,
        ou.uid AS orgunit_uid,
        ou.hierarchylevel,
        p.periodid,
        p.startdate,
        p.enddate,
        pt.name AS period_type,
        dv.value,
        CASE 
            WHEN dv.value ~ '^[-]?\\d+(\\.\\d+)?$' 
            THEN dv.value::numeric 
        END AS value_num,
        dv.comment,
        dv.storedby,
        dv.lastupdated,
        dv.created,
        COALESCE(dv.followup, FALSE) AS followup
    FROM datavalue dv
    JOIN dataelement de ON dv.dataelementid = de.dataelementid
    JOIN organisationunit ou ON dv.sourceid = ou.organisationunitid
    JOIN period p ON dv.periodid = p.periodid
    LEFT JOIN periodtype pt ON p.periodtypeid = pt.periodtypeid
)
"""
    # -------------------------
    # Aggregated / Explainable
    # -------------------------
    if group_by:
        return f"""
{base_cte}
SELECT
    {select_sql}
FROM base
WHERE {where_sql}
GROUP BY {group_by}
ORDER BY {order_by}
""".strip()

    # -------------------------
    # Raw Row Inspection
    # -------------------------
    else:
        return f"""
{base_cte}
SELECT
    {select_sql}
FROM base
WHERE {where_sql}
ORDER BY {order_by}
""".strip()

def build_yearly_query(
    dataelement_include: bool,
    where_sql: str,
    order_by: str = "year ASC"
) -> str:
    """
    Builds yearly aggregation query for explainable analytics.

    Expected usage:
    - Used for outbreak / trend explanations
    - Requires startdate filtering in where_sql
    - Returns yearly totals
    """

    print("Entered build yearly query")

    base_cte = """
WITH base AS (
    SELECT
        dv.dataelementid,
        de.name AS dataelement_name,
        de.code AS dataelement_code,
        de.uid AS dataelement_uid,
        ou.organisationunitid,
        ou.name AS orgunit_name,
        ou.code AS orgunit_code,
        ou.uid AS orgunit_uid,
        ou.hierarchylevel,
        p.periodid,
        p.startdate,
        p.enddate,
        pt.name AS period_type,
        dv.value,
        CASE 
            WHEN dv.value ~ '^[-]?\\d+(\\.\\d+)?$' 
            THEN dv.value::numeric 
        END AS value_num,
        dv.comment,
        dv.storedby,
        dv.lastupdated,
        dv.created,
        COALESCE(dv.followup, FALSE) AS followup
    FROM datavalue dv
    JOIN dataelement de ON dv.dataelementid = de.dataelementid
    JOIN organisationunit ou ON dv.sourceid = ou.organisationunitid
    JOIN period p ON dv.periodid = p.periodid
    LEFT JOIN periodtype pt ON p.periodtypeid = pt.periodtypeid
)
"""
    if not dataelement_include:
        return f"""
    {base_cte}
    SELECT
        EXTRACT(YEAR FROM startdate) AS year,
        
        SUM(value_num) AS total_value
    FROM base
    WHERE {where_sql}
    GROUP BY year
    ORDER BY {order_by}
    """.strip()
    else:
        return f"""
    {base_cte}
    SELECT
        EXTRACT(YEAR FROM startdate) AS year,
        dataelement_name,
        SUM(value_num) AS total_value
    FROM base
    WHERE {where_sql}
    GROUP BY year, dataelement_name
    ORDER BY {order_by}
    """.strip()

def build_chart_query(where_sql: str) -> str:
    """
    Returns aggregated chart-ready data with time dimension.
    """

    return f"""
    WITH base AS (
        SELECT
            de.name AS dataelement_name,
            ou.name AS orgunit_name,
            p.startdate,
            CASE
                WHEN dv.value ~ '^[-]?\\d+(\\.\\d+)?$'
                THEN dv.value::numeric
            END AS value_num
        FROM datavalue dv
        JOIN dataelement de ON dv.dataelementid = de.dataelementid
        JOIN organisationunit ou ON dv.sourceid = ou.organisationunitid
        JOIN period p ON dv.periodid = p.periodid
    )
    SELECT
        DATE_TRUNC('month', startdate) AS period,
        orgunit_name,
        dataelement_name,
        SUM(value_num) AS total_value
    FROM base
    WHERE {where_sql}
    GROUP BY 1, 2, 3
    ORDER BY 1;
    """.strip()

def run_query(conn_str: str, sql: str, params: List[Any]) -> pd.DataFrame:
    with psycopg2.connect(conn_str) as conn:
        df = pd.read_sql(sql, conn, params=params)
    return df

def validate_llm_where_sql(where_sql: str, params: List[Any]) -> Tuple[bool, str]:
    if re.search(r";|--|/\\*|\\*/", where_sql):
        return False, "Disallowed SQL tokens in where_sql."
    if re.search(
        r"\\b(insert|update|delete|drop|alter|create|truncate|grant|revoke)\\b",
        where_sql,
        re.IGNORECASE,
    ):
        return False, "Unsafe SQL operation in where_sql."
    if where_sql.count("%s") != len(params):
        return False, "Number of placeholders does not match params."

    scrubbed = re.sub(r"'[^']*'", "''", where_sql)
    scrubbed = scrubbed.replace("%s", "")
    tokens = re.findall(r"\\b[A-Za-z_][A-Za-z0-9_]*\\b", scrubbed)
    allowed = {c.lower() for c in BASE_COLUMNS} | ALLOWED_SQL_KEYWORDS
    for tok in tokens:
        if tok.lower() not in allowed:
            return False, f"Unexpected identifier in where_sql: {tok}"
    return True, ""

def normalize_order_by(order_by: str, default: str, select_sql: str = "") -> str:
    m = re.match(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(ASC|DESC)?\s*$",
        order_by,
        re.IGNORECASE,
    )
    if not m:
        return default

    col = m.group(1).lower()
    direction = (m.group(2) or "DESC").upper()

    base_cols = {c.lower() for c in BASE_COLUMNS}

    # 🔹 Extract aliases from select_sql (e.g., SUM(value_num) AS total_value)
    aliases = set()
    if select_sql:
        for alias_match in re.findall(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", select_sql, re.IGNORECASE):
            aliases.add(alias_match.lower())

    if col not in base_cols and col not in aliases:
        return default

    return f"{col} {direction}"
# ---------------------------------------------------------------------
# Output Sanitizer (User-friendly table)
# ---------------------------------------------------------------------

def make_user_friendly_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats both raw record results and aggregated/grouped results
    into a UI-safe table.
    """
    print("Entered make user friendly")
    if df.empty:
        return df

    technical_cols = {
        "dataelementid",
        "organisationunitid",
        "periodid",
        "dataelement_code",
        "dataelement_uid",
        "orgunit_code",
        "orgunit_uid",
        "hierarchylevel",
        "comment",
        "storedby",
        "lastupdated",
        "created",
        "total_count",
    }

    # Drop technical columns if present
    df = df.drop(columns=[c for c in technical_cols if c in df.columns])
    print(df.columns)
    # -------------------------
    # Detect RAW mode
    # -------------------------
    raw_signature = {"orgunit_name", "dataelement_name", "startdate"}

    if raw_signature.issubset(set(df.columns)):

        clean = df.copy()

        # Prefer numeric value
        if "value_num" in clean.columns and clean["value_num"].notna().any():
            clean["Value"] = clean["value_num"]
        elif "value" in clean.columns:
            clean["Value"] = clean["value"]

        clean = clean.drop(columns=[c for c in ["value_num", "value"] if c in clean.columns])
        rename_map = {
            "orgunit_name": "Organisation Unit",
            "dataelement_name": "Metric",
            "startdate": "Start Date",
            "enddate": "End Date",
            "period_type": "Period Type",
            "followup": "Follow-up",
        }

        clean = clean.rename(columns=rename_map)

        if "Start Date" in clean.columns:
            clean["Start Date"] = pd.to_datetime(clean["Start Date"], errors="coerce").dt.date
        if "End Date" in clean.columns:
            clean["End Date"] = pd.to_datetime(clean["End Date"], errors="coerce").dt.date

        return clean

    # -------------------------
    # Aggregated / Grouped mode
    # -------------------------
    clean = df.copy()

    rename_map = {
        "orgunit_name": "Organisation Unit",
        "dataelement_name": "Metric",
        "total_value": "Total",
        "record_count": "Records",
    }

    clean = clean.rename(columns=rename_map)

    return clean

def mentions_value(question: str) -> bool:
    return bool(
        re.search(
            r"\b(value|values|cases?|counts?|deaths?|tests?|visits?|incidence|coverage)\b",
            question,
            re.IGNORECASE,
        )
    )


def build_conn_str_from_parts() -> str:
    host = os.environ.get("PGHOST", "").strip()
    port = os.environ.get("PGPORT", "5432").strip()
    database = os.environ.get("PGDATABASE", "").strip()
    user = os.environ.get("PGUSER", "").strip()
    password = os.environ.get("PGPASSWORD", "").strip()

    if not host or not database or not user:
        return ""

    dsn = f"host={host} port={port} dbname={database} user={user}"
    if password:
        dsn += f" password={password}"
    return dsn

def make_json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    return obj

def detect_query_shape(chart_df):
    return {
        "has_time": chart_df["period"].nunique() > 1 if "period" in chart_df else False,
        "multi_org": chart_df["orgunit_name"].nunique() > 1,
        "multi_metric": chart_df["dataelement_name"].nunique() > 1,
    }

def build_kpi_cards(df):
    return {
        "total": float(df["total_value"].sum()),
        "orgs": int(df["orgunit_name"].nunique()),
        "metrics": int(df["dataelement_name"].nunique()),
        "records": int(df["record_count"].sum()) if "record_count" in df else None,
    }

def build_metric_breakdown(chart_df, top_n=10):

    metric_totals = (
        chart_df.groupby("dataelement_name")["total_value"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    return {
        "type": "bar_metrics",
        "title": "Top Metrics",
        "data": [
            {"name": r["dataelement_name"], "total": float(r["total_value"])}
            for _, r in metric_totals.iterrows()
        ]
    }

def org_filter_present(where_sql):
    return "orgunit_name" in where_sql

def build_trend(chart_df):

    trend = (
        chart_df.groupby("period")["total_value"]
        .sum()
        .reset_index()
        .sort_values("period")
    )

    return {
        "type": "line_trend",
        "title": "Trend Over Time",
        "data": [
            {"date": str(r["period"].date()), "total": float(r["total_value"])}
            for _, r in trend.iterrows()
        ]
    }

def build_metric_trend(chart_df, top_n=5):

    top_metrics = (
        chart_df.groupby("dataelement_name")["total_value"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    filtered = chart_df[chart_df["dataelement_name"].isin(top_metrics)]

    grouped = (
        filtered.groupby(["period", "dataelement_name"])["total_value"]
        .sum()
        .reset_index()
        .sort_values("period")
    )

    return {
        "type": "multi_line_metric_trend",
        "title": "Metric Trends Over Time",
        "series": [
            {
                "metric": metric,
                "data": [
                    {
                        "date": str(r["period"].date()),
                        "total": float(r["total_value"])
                    }
                    for _, r in grouped[grouped["dataelement_name"] == metric].iterrows()
                ]
            }
            for metric in top_metrics
        ]
    }

def build_org_breakdown(chart_df, top_n=8):
    totals = (
        chart_df.groupby("orgunit_name")["total_value"]
        .sum()
        .sort_values(ascending=False)
    )

    top = totals.head(top_n)
    other_sum = totals.iloc[top_n:].sum()

    data = [
        {"name": org, "total": float(val)}
        for org, val in top.items()
    ]

    if other_sum > 0:
        data.append({"name": "Other", "total": float(other_sum)})

    return {
        "type": "bar_orgs",
        "title": "Top Organisations",
        "data": data
    }

def build_org_trend(chart_df, top_n=5):

    # print("Entered Org Trend Function")

    # --- Clean columns ---
    chart_df["period"] = pd.to_datetime(chart_df["period"], errors="coerce")
    chart_df["total_value"] = pd.to_numeric(chart_df["total_value"], errors="coerce")

    chart_df["orgunit_name"] = (
        chart_df["orgunit_name"]
        .astype(str)
        .str.strip()
    )

    # Drop bad rows
    chart_df = chart_df.dropna(subset=["period", "orgunit_name", "total_value"])

    # print("After cleaning, rows:", len(chart_df))

    # --- Monthly totals per org ---
    monthly_org = (
        chart_df.groupby(["period", "orgunit_name"])["total_value"]
        .sum()
        .reset_index()
    )

    # print("Monthly org rows:", len(monthly_org))
    # print(monthly_org.head())

    # --- Top orgs ---
    top_orgs = (
        monthly_org.groupby("orgunit_name")["total_value"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    # print("Top orgs:", top_orgs)

    if not top_orgs:
        # print("❌ No top orgs found!")
        return None

    # --- Build series ---
    series = []
    for org in top_orgs:

        org_data = monthly_org[monthly_org["orgunit_name"] == org]

        # print(f"Org {org} rows:", len(org_data))

        points = [
            {
                "date": str(r["period"].date()),
                "total": float(r["total_value"])
            }
            for _, r in org_data.iterrows()
        ]

        if points:
            series.append({"metric": org, "data": points})

    # print("Final series length:", len(series))

    return {
        "type": "multi_line_metric_trend",
        "title": "Top Organisation Trends",
        "series": series
    }

def build_summary_insights(where_sql, conn_str, params):

    chart_sql = build_chart_query(where_sql)
    chart_df = run_query(conn_str, chart_sql, params)

    if chart_df.empty:
        return {"mode": "none", "data": []}

    insights = {
        "mode": "dashboard",
        "cards": {
            "total": float(chart_df["total_value"].sum()),
            "metrics": int(chart_df["dataelement_name"].nunique()),
            "orgs": int(chart_df["orgunit_name"].nunique()),
        },
        "charts": []
    }
    # print("ChartDF is", chart_df.head())
    has_time = chart_df["period"].nunique() > 1
    # print("The time period is: ",has_time)
    multi_metric = chart_df["dataelement_name"].nunique() > 1
    multi_org = chart_df["orgunit_name"].nunique() > 1

    if has_time and multi_org and multi_metric:
        # print("Entered time-org-filter-trend")
        insights["charts"].append(build_org_trend(chart_df))
    elif has_time and multi_metric:
        # print("Entered multi metric trend")
        insights["charts"].append(build_metric_trend(chart_df))
    elif multi_org and org_filter_present(where_sql):
        insights["charts"].append(build_org_breakdown(chart_df))
    elif multi_metric:
        insights["charts"].append(build_metric_breakdown(chart_df))
    elif has_time:
        insights["charts"].append(build_trend(chart_df))
    # print("Insights are: ")
    # print(insights)
    return insights

def strip_date_filters(where_sql: str, params):
    if not params:
        params = []
        # print("No params received")

    pieces = re.split(r"\s+AND\s+", where_sql, flags=re.IGNORECASE)
    print(pieces)
    new_sql = []
    new_params = []

    idx = 0
    for p in pieces:
        ph = p.count("%s")

        if re.search(r"\b(startdate|enddate|created|lastupdated)\b", p, re.I):
            idx += ph
            continue

        new_sql.append(p)
        new_params.extend(params[idx: idx + ph])
        idx += ph

    if not new_sql:
        return "TRUE", []

    return " AND ".join(new_sql), new_params
def build_ranking_analysis_payload(df, yearly_df):
    payload = {
        "ranking_summary": {},
        "top_result": None
    }

    if df.empty:
        return payload

    payload["ranking_summary"] = {
        "rows": len(df),
        "max_value": float(df["total_value"].max())
    }

    payload["top_result"] = df.iloc[0].to_dict()
    if yearly_df is not None and not yearly_df.empty:
        ydf = yearly_df.copy()
        ydf["total_value"] = pd.to_numeric(ydf["total_value"], errors="coerce")
        ydf = ydf.sort_values("year")

        ydf["pct_change"] = ydf["total_value"].pct_change()

        payload["yearly_analysis"] = {
            "years_present": ydf["year"].tolist(),
            "yearly_totals": ydf["total_value"].tolist(),
            "year_over_year_change": ydf["pct_change"].tolist()
        }
    return payload

def build_analysis_payload(
    monthly_df: pd.DataFrame,
    yearly_df: pd.DataFrame | None = None
) -> dict:
    """
    Converts SQL result(s) into structured analytical signals
    for the explanation LLM.

    monthly_df must contain:
        - startdate
        - total_value

    yearly_df (optional) must contain:
        - year
        - total_value
    """

    payload = {
        "data_summary": {},
        "monthly_analysis": {},
        "yearly_analysis": {},
        "anomaly_detection": {}
    }

    # -------------------------
    # BASIC CLEANING
    # -------------------------
    df = monthly_df.copy()
    df["startdate"] = pd.to_datetime(df["startdate"])
    df = df.sort_values("startdate")

    # Ensure numeric
    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")

    # -------------------------
    # DATA SUMMARY
    # -------------------------
    payload["data_summary"] = {
        "total_months": int(len(df)),
        "overall_sum": float(df["total_value"].sum()),
        "overall_mean": float(df["total_value"].mean()),
        "max_value": float(df["total_value"].max()),
        "min_value": float(df["total_value"].min())
    }

    # -------------------------
    # MONTHLY TREND METRICS
    # -------------------------
    df["pct_change"] = df["total_value"].pct_change()

    payload["monthly_analysis"] = {
        "average_monthly_change": float(df["pct_change"].mean(skipna=True)),
        "max_month": df.loc[df["total_value"].idxmax(), "startdate"].strftime("%Y-%m"),
        "max_month_value": float(df["total_value"].max())
    }

    # -------------------------
    # SPIKE / ANOMALY DETECTION
    # Rolling z-score (works well for 2-year data)
    # -------------------------
    df["rolling_mean"] = df["total_value"].rolling(3).mean()
    df["rolling_std"] = df["total_value"].rolling(3).std()

    df["z_score"] = (
        (df["total_value"] - df["rolling_mean"]) /
        df["rolling_std"]
    )

    spikes = df[df["z_score"] > 2]

    payload["anomaly_detection"] = {
        "spike_count": int(len(spikes)),
        "spike_months": spikes["startdate"].dt.strftime("%Y-%m").tolist()
    }

    # -------------------------
    # YEARLY COMPARISON (IF AVAILABLE)
    # -------------------------
    if  len(yearly_df)>1:
        ydf = yearly_df.copy()
        ydf["total_value"] = pd.to_numeric(ydf["total_value"], errors="coerce")
        ydf = ydf.sort_values("year")

        ydf["pct_change"] = yearly_df["total_value"].pct_change().iloc[-1]
        print("pct change is ", ydf["pct_change"])
        payload["yearly_analysis"] = {
            "years_present": ydf["year"].tolist(),
            "yearly_totals": ydf["total_value"].tolist(),
            "year_over_year_change": ydf["pct_change"].tolist()
        }

    return payload

def build_indicator_analysis_payload(df, yearly_df):

    summary = (
        df.groupby("dataelement")["total_value"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    total_sum = summary["total_value"].sum()

    indicators = summary.to_dict(orient="records")

    dominant = summary.iloc[0] if not summary.empty else None

    payload = {
        "analysis_type": "indicator_comparison",
        "data_summary": {
            "total_across_indicators": float(total_sum),
            "indicator_count": len(summary),
        },
        "indicator_distribution": indicators,
        "dominant_indicator": {
            "name": dominant["dataelement"],
            "value": float(dominant["total_value"])
        } if dominant is not None else None
    }

    return payload

def build_summary_analysis_payload(df, yearly_df=None):
    """
    Builds analysis payload for summary view:
    dataelement_name | total_value

    Returns structured JSON-ready dictionary.
    """

    required_cols = {"dataelement_name", "total_value"}

    if not required_cols.issubset(df.columns):
        return {"error": "Invalid summary dataframe structure"}

    # Defensive copy
    df = df.copy()

    # Ensure numeric
    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")

    # Drop completely empty totals
    df = df.dropna(subset=["total_value"])

    if df.empty:
        return {"error": "No valid numeric values found"}

    # Sort descending
    df = df.sort_values("total_value", ascending=False).reset_index(drop=True)

    total_sum = df["total_value"].sum()
    mean_value = df["total_value"].mean()
    max_row = df.iloc[0]
    min_row = df.iloc[-1]

    # Contribution percentages
    df["percentage_contribution"] = (
        df["total_value"] / total_sum * 100
    ).round(2)

    payload = {
        "view_type": "summary",
        "total_indicators": int(len(df)),
        "grand_total": float(total_sum),
        "average_value": float(mean_value),
        "top_performer": {
            "dataelement_name": max_row["dataelement_name"],
            "value": float(max_row["total_value"]),
            "contribution_percent": float(df.iloc[0]["percentage_contribution"])
        },
        "lowest_performer": {
            "dataelement_name": min_row["dataelement_name"],
            "value": float(min_row["total_value"]),
            "contribution_percent": float(df.iloc[-1]["percentage_contribution"])
        },
        "ranking": df.to_dict(orient="records")
    }

    return payload

def build_universal_analysis_payload(df, yearly_df):

    df_cols = set(df.columns)
    yearly_cols = set(yearly_df.columns) if yearly_df is not None else set()

    # Records view
    if "startdate" in df_cols:
        return build_analysis_payload(df, yearly_df)

    # Ranking view
    elif "orgunit_name" in df_cols:
        return build_ranking_analysis_payload(df, yearly_df)

    # Summary-only view
    elif {"dataelement_name", "total_value"} <= df_cols:
        return build_summary_analysis_payload(df, yearly_df)

    # Explainable yearly view
    elif "year" in yearly_cols:
        return build_indicator_analysis_payload(df, yearly_df)

    else:
        return {"error": "Unsupported dataframe structure"}

def explain_query(insights: dict, question: str, hf_token: str) -> str:
    print("Entered explain query function")
    final_payload = {
    "insights": insights,
    # "yearly_dataelement_breakdown": yearly_dataelement_df.to_dict(orient="records"),
    "question": question
    }
    print("Final payload for explanation is:")
    explanation_prompt = f"""
You are an epidemiological analytics explanation engine.

You will receive:
1. A structured JSON object named "insights"
2. A yearly breakdown table (optional)
3. The original user question

Your task is to generate a clear, evidence-based explanation of the observed patterns.

STRICT CONSTRAINTS:
- Use ONLY the provided data.
- Do NOT introduce external knowledge.
- Do NOT fabricate causes.
- Do NOT claim causation.
- Use neutral analytical language (e.g., "may indicate", "suggests", "is associated with").
- Do NOT reference SQL, models, prompts, system design, or technical processes.
- Do NOT mention missing data, null values, NaN values, or unavailable fields.
- Do NOT explain what cannot be determined.
- Describe only the patterns that are supported by the provided data.
- Never return JSON.
- Always produce a narrative response.
- Minimum 3 sentences.

REQUIRED OUTPUT FORMAT:
Write 1–3 well-structured paragraphs.

Structure Guidelines:

Paragraph 1:
Describe the overall magnitude and distribution using the values in "insights".
Include grand total, relative contributions, ranking, and differences between indicators where applicable.

Paragraph 2:
If anomaly_detection.spike_count > 0, describe the spikes and months.
If spike_count = 0, simply state that no abnormal temporal spikes were detected.
Do not add any additional commentary.

Paragraph 3:
If yearly totals are provided for multiple years, describe the direction of change (increase, decrease, or relatively stable) based on the totals.
Compare years directly using the numeric values.
Do not discuss computation limitations.
If only one year is present, omit this paragraph.

Write clearly for public health workers.
Be concise but complete.
"""
    print("Sending explanation prompt to LLM...")   
    explanation=run_instruct_llm_prompt( explanation_prompt, hf_token, payload=final_payload,payload_include=True)
    print("Explanation received from LLM:", explanation)
    return explanation

@lru_cache(maxsize=300)

def get_cached_llm_plan(use_model: str,prompt: str, token: str, payload: dict={}, payload_include: bool=False) -> dict:
    """
    Calls the LLM once and caches the generated SQL plan.
    Subsequent identical questions reuse cached result.
    Removes repeated LLM latency during:
      - insights fetch
      - pagination
      - re-renders
    """
    print("Entered get Cached LLm plan")
    if use_model=="SQL":
        decoded = run_sql_llm_prompt(prompt, token).strip()
    elif use_model=="Instruct":
        decoded = run_instruct_llm_prompt(prompt, token).strip()

    instr = {}
    m = re.search(r"\{[\s\S]*\}", decoded)
    if m:
        try:
            instr = json.loads(m.group())
        except Exception:
            instr = {}

    return instr

def detect_intent(question: str):
    q = question.lower()

    if re.search(r"(trend|over time|per month|monthly|weekly|daily|over the last|over past)", q):
        return "trend"

    if re.search(r"(which|top|highest|lowest|most|least|compare|by facility|by organisation)", q):
        return "ranking"

    if re.search(r"(how many|total|sum|overall|count)", q):
        return "summary"

    return "auto"

def build_distinct_dataelement_query(metrics):
    if not metrics:
        return None, None

    cleaned = [m.strip() for m in metrics if m and m.strip()]
    if not cleaned:
        return None, None

    patterns = [f"%{m}%" for m in cleaned]

    sql = """
        SELECT DISTINCT name AS dataelement_name
        FROM dataelement
        WHERE name ILIKE ANY(%s)
        ORDER BY name;
    """

    return sql.strip(), [patterns]

def auto_group_where(where_sql: str) -> str:

    # If already grouped, DO NOTHING
    if "(" in where_sql:
        return where_sql

    clauses = re.split(r"\s+(AND|OR)\s+", where_sql, flags=re.IGNORECASE)

    col_groups = {}

    for clause in clauses:

        clause = clause.strip()

        if clause.upper() in ("AND", "OR"):
            continue

        col = clause.split()[0]

        col_groups.setdefault(col, []).append(clause)

    rebuilt = []

    for col, group in col_groups.items():

        if len(group) > 1:
            rebuilt.append("(" + " OR ".join(group) + ")")
        else:
            rebuilt.append(group[0])

    return " AND ".join(rebuilt)

# ---------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------

def normal_query_answer(question: str, row_limit: int, conn_str: str, hf_token: str, columns_list: list, today_str: date, debug: bool = False, page: int = 1, page_size: int = 100, include_insights: bool = False, include_rows: bool = True)->dict:
    intent_prompt = f"""
You are a strict information extraction engine for DHIS2 health facility data.
CRITICAL RULES:
Return ONLY valid JSON.
Do NOT include explanations.
Do NOT include markdown.
Do NOT include any text outside JSON.
Output must be syntactically valid JSON.
All lists must be arrays.
If a field is missing, return None (not empty string, not empty list).
Required Output Format:
{{
"orgunit": ["string"] | None,
"metric": ["string"] | None
}}
Domain Definitions:
Organisation Unit (orgunit):
A named health facility or reporting unit explicitly written in the question.
Examples: Moyowa MCHP, Bo Government Hospital, Kenema Clinic, Red Cross, Loreto
Metric:
A health-related term explicitly written in the question referring to:
Disease (e.g., malaria, TB, HIV, measles)
Service (e.g., vaccinations, deliveries, OPD visits)
Commodity (e.g., zinc, ORS, bed nets, vaccines)
Indicator word if explicitly written together (e.g., "Malaria positive")
Strict Extraction Rules:
Extract ONLY exact text spans that appear in the question.
Preserve original casing and wording exactly as written.
Do NOT normalize text.
Do NOT infer missing words.
Do NOT add indicator words.
Example: If question says "malaria cases","malaria outbreak", extract "malaria".
Do NOT extract "malaria cases", "malaria outbreaks".
Do NOT invent organisation units.
Do NOT invent metrics.
Do NOT merge separate entities into one.
If multiple orgunits exist, return all as separate list items.
If no orgunit is explicitly mentioned, return:
"orgunit": null
If no metric is explicitly mentioned, return:
"metric": null
Never guess abbreviations.
Ignore time references (years, months, periods).
Ignore aggregation words (most, total, highest, etc.).
Ignore filter intent words (in, for, of, from, related to, etc.).
Examples:
Question:
Which organisation has most measles outbreak in 2015?
Output:
{{
"orgunit": None,
"metric": ["measles"]
}}
Question:
Compare malaria positive cases between Moyowa MCHP and Bo Government Hospital in 2021
Output:
{{
"orgunit": ["Moyowa MCHP","Bo Government Hospital"],
"metric": ["malaria positive"]
}}
Question:
What cases for Red Cross, Loreto and Bumban MCHP related to were malaria positive and Measles in 2016
Output:
{{
"orgunit": ["Red Cross", "Loreto", "Bumban MCHP"],
"metric": ["malaria positive", "Measles"]
}}
Question:
Trend of TB cases in Kenema Clinic from 2018 to 2022
Output:
{{
"orgunit": ["Kenema Clinic"],
"metric": ["TB"]
}}
Now extract from:
Question:
{question}
JSON:
""".strip()
    print(f'the length of intent prompt is {len(intent_prompt)}')
    instr_intent=get_cached_llm_plan("Instruct",intent_prompt,hf_token)
    print(instr_intent)
    decoded_intent = json.dumps(instr_intent)

    intent_json: Dict[str, Any] = {}
    m = re.search(r"\{[\s\S]*\}", decoded_intent)
    if m:
        try:
            intent_json = json.loads(m.group())
        except Exception:
            intent_json = {}
    intent_json= normalize_metrics(intent_json)
    print(intent_json)
    distinct_metric_df = list()
    if intent_json.get("metric") is not None:
        intent_json["metric"] = [m.strip() for m in intent_json["metric"] if m and m.strip()]
        distinct_metric_sql,distinct_metric_params= build_distinct_dataelement_query(intent_json.get("metric"))
        distinct_metric_df = run_query(conn_str, distinct_metric_sql, distinct_metric_params)
    print("Distinct metric df is ", distinct_metric_df)
    prompt = f"""
You are a STRICT Text-to-SQL planner for a PostgreSQL database.
You must return ONLY a valid JSON object.NO markdown.NO explanation.NO SQL outside JSON.
You are querying from a CTE named `base` with ONLY these columns:
{columns_list}

Available valid dataelement_name values (STRICT FILTER LIST):
{distinct_metric_df}

CRITICAL METRIC FILTERING RULE:
- You MUST use ONLY values that exist in the provided valid dataelement_name list above.
- If intent metric is not null, you MUST match it ONLY against items from this list.
- Never generate a new dataelement_name.
- Never use partial words not present in the list.
- If none of the provided values match the intent metric, set where_sql = "FALSE".

Task:
Given the user’s question, produce a JSON plan with exactly these keys:
- `select_sql`: SQL select expression for the outer query (do NOT include the word "SELECT"); use "*" for non-aggregated row queries, use aggregate functions like SUM(value_num) AS total_value when aggregation is required, and use only columns available in base.
- `where_sql`: SQL boolean condition for filtering rows from `base` (do NOT include the word "WHERE").
- `params`: array of values corresponding to the `%s` placeholders in `where_sql`.
- `group_by`: column name for grouping results (do NOT include the words "GROUP BY"); leave as empty string "" if no grouping is required; any non-aggregated column in select_sql MUST appear here.
- `order_by`: one of the base columns + ASC or DESC, or an aggregate alias defined in select_sql (e.g., total_value DESC); use only listed base columns or defined aliases.
- `limit`: integer between 1 and {row_limit}; use 1 for ranking queries like “most” or “highest”, otherwise default to {row_limit}.

Intent for metric and orgunit are preclassified. They are:
orgunits-{intent_json["orgunit"]}
metrics-{intent_json["metric"]}

Follow this reasoning process:

1. Only use dataelement_name in WHERE clause if the intent metric is not "".
   - When filtering by metric, select ONLY matching values from the provided valid dataelement_name list.
   - If multiple valid matches exist, include each as separate:
     (dataelement_name ILIKE %s OR dataelement_name ILIKE %s)
   - ALWAYS wrap metric OR conditions in parentheses.

2. Only use orgunit_name in WHERE clause if the intent orgunit is not "".
   - If multiple orgunits exist:
     (orgunit_name ILIKE %s OR orgunit_name ILIKE %s)
   - NEVER use AND between multiple orgunit_name.
   - ALWAYS wrap orgunit OR conditions in parentheses.

3. Identify any date/time filters (e.g. last 3 months, in 2016).
→ Use startdate >= %s AND startdate < %s for filtering.

4. Identify follow-up status intent phrases meaning completed/maintained/flagged/reviewed → followup IS TRUE
   while phrases meaning not followed up/pending/unresolved/missing → followup IS FALSE
→ if mentioned, include filter: followup = %s

5. Identify sorting direction (e.g. latest → DESC).

Extract the exact date and time from the following phrases and return in ISO format:
- “last X days” → compute today minus X days to today
- “last month” → from 1st of previous month to 1st of current month
- “in 2022” → "2022-01-01" to "2023-01-01"

6. Logical grouping rules (CRITICAL):
- Conditions on the SAME column must be grouped together using parentheses and combined using OR.
- Conditions on DIFFERENT columns must be combined using AND.
- ALWAYS wrap OR groups in parentheses.

Examples:
Multiple facilities:
(orgunit_name ILIKE %s OR orgunit_name ILIKE %s)

Multiple metrics:
(dataelement_name ILIKE %s OR dataelement_name ILIKE %s)

Combined:
(org filters) AND (metric filters) AND date filters

Aggregation Rules:
- If the question asks:
"which organisation", "which facility", "most", "highest",
"lowest", "top", "least", "maximum", "minimum"
THEN aggregation is required.

- For organisation ranking questions:
select_sql = "orgunit_name, SUM(value_num) AS total_value"
group_by = "orgunit_name"
order_by = "total_value DESC"
limit = 1 (unless user specifies otherwise)

- For general totals without ranking:
select_sql = "SUM(value_num) AS total_value"
group_by = ""
order_by = "total_value DESC"

- If no aggregation is required:
select_sql = "*"
group_by = ""

Hard rules:
- If there are multiple metrics in intent then add all matched valid dataelement_name values separately as "dataelement_name ILIKE %s" and join them with OR inside parentheses.
- If there are multiple orgunits in intent then add all of them as separate "orgunit_name ILIKE %s" and join them with OR inside parentheses.
- Use ONLY the listed columns. Do not invent column names.
- Never put values directly into where_sql instead add %s in where_sql and add values in params.
- If group_by is not empty, select_sql MUST include aggregate functions.
- If aggregate alias is used (e.g., total_value), order_by MUST use the alias and NEVER use startdate ordering.
- Never mix aggregated and non-aggregated columns unless included in group_by.
- Do not invent column names. Use value_num for aggregation.
- Do not use semicolons, joins, subqueries, or multiple statements.
- Prefer ILIKE for text matching.
- If user requests a time period, filter using startdate: startdate >= %s AND startdate < %s
- If aggregation is NOT required and user asks “latest/recent”, order by startdate DESC.
- If user asks “oldest/earliest”, order by startdate ASC.
- NEVER output mixed AND/OR without parentheses.
- Every OR group MUST be inside brackets.
- If you are unsure or the user question is vague or chit chat intention use where_sql="FALSE".
- Never return empty JSON always return a valid JSON plan.

So Return EXACTLY this:
{{ "where_sql": "FALSE", "params": [], "order_by": "startdate DESC", "limit": 1}}

Today’s date is: {today_str}

Example 1:
User question:
Show records from Loreto Clinic related to malaria cases between Jan to April 2016
{{
"select_sql": "*",
"where_sql": "orgunit_name ILIKE %s AND dataelement_name ILIKE %s AND startdate >= %s AND startdate < %s",
"params": ["%Loreto Clinic%", "%malaria positive%","2016-01-01", "2016-04-01"],
"group_by": "",
"order_by": "startdate DESC",
"limit": 200
}}

Example 2:
User question:
Show records of malaria with non followed up cases in first 75 days of 2016
{{
"select_sql": "*",
"where_sql": "(dataelement_name ILIKE %s) AND followup = %s AND startdate >= %s AND startdate < %s",
"params": ["%malaria positive%", FALSE,"2016-01-01", "2016-03-16"],
"group_by": "",
"order_by": "startdate DESC",
"limit": 200
}}

Example 3:
User question:
How many red cross clinic patients were affected by malaria negative related results in 2015
{{
"select_sql": "SUM(value_num) AS total_value",
"where_sql": "orgunit_name ILIKE %s AND dataelement_name ILIKE %s AND startdate >= %s AND startdate < %s",
"params": ["%Red Cross Clinic%", "%malaria negative%","2015-01-01", "2016-01-01"],
"group_by": "",
"order_by": "total_value DESC",
"limit": 200
}}

Example 4:
User question:
What are all the cases registered in 2016
{{
"select_sql": "*",
"where_sql": "startdate >= %s AND startdate < %s",
"params": ["2016-01-01", "2017-01-01"],
"group_by": "",
"order_by": "startdate DESC",
"limit": 200
}}

Example 5:
User question:
Which organisation reported most measles cases in 2015
{{
"select_sql": "orgunit_name, SUM(value_num) AS total_value",
"where_sql": "(dataelement_name ILIKE %s) AND startdate >= %s AND startdate < %s",
"params": ["%measles confirmed%","2015-01-01","2016-01-01"],
"group_by": "orgunit_name",
"order_by": "total_value DESC",
"limit": 1
}}

User question:
{question}
""".strip()

    instr = get_cached_llm_plan("SQL",prompt, hf_token)
    print("Instr is ",instr)
    decoded = json.dumps(instr)

    instr: Dict[str, Any] = {}
    m = re.search(r"\{[\s\S]*\}", decoded)
    if m:
        try:
            instr = json.loads(m.group())
        except Exception:
            instr = {}

    use_llm_sql = False
    select_sql="*"
    where_sql = "TRUE"
    params: List[Any] = []
    order_by = ""
    group_by=""
    query_offset = max(0, (page - 1) * page_size)
    query_limit = page_size
    if isinstance(instr, dict) and any(k in instr for k in ("where_sql", "params", "order_by", "limit", "select_sql", "group_by")):
        llm_where = instr.get("where_sql", "")
        llm_params = instr.get("params", [])
        llm_order = instr.get("order_by", order_by)
        llm_limit = instr.get("limit", query_limit)
        llm_select = instr.get("select_sql", select_sql)
        llm_group = instr.get("group_by", group_by)
        if isinstance(llm_where, str):
            where_sql = llm_where.strip() or "TRUE"
        if isinstance(llm_params, list):
            params = llm_params
        if isinstance(llm_order, str):
            order_by = llm_order
        if isinstance(llm_limit, int):
            query_limit = max(1, min(row_limit, llm_limit))
        if isinstance(llm_select, str) and llm_select.strip():
            select_sql = llm_select.strip()
        if isinstance(llm_group, str):
            group_by = llm_group.strip()
        if isinstance(llm_limit, str) and llm_limit.isdigit():
            query_limit = max(1, min(row_limit, int(llm_limit)))

    ok, err = validate_llm_where_sql(where_sql, params)

    if ok and where_sql!="TRUE":
        use_llm_sql = True

        if " OR " in where_sql and " AND " in where_sql:
            where_sql=auto_group_where(where_sql)
        print("After Grouping:",where_sql)
        if re.search(r"(?:last|previous|past|recent|trailing)\s+\d+\s+months?\s+of\s+20\d{2}",question.lower()):
            print("Entered heiristics")
            date_col, start_dt, end_dt, sort_dir = extract_date_range(question)
            print("Dates from function:",start_dt,end_dt)
            if start_dt or end_dt:
                date_params = []

                if start_dt:
                    date_params.append(start_dt.to_pydatetime())

                if end_dt:
                    date_params.append(end_dt.to_pydatetime())

                old_date_count = len(date_params)   # since old dates were same pattern
                params = params[:-old_date_count] if old_date_count else params
                params.extend(date_params)

            print("Updated params are:", params)

    # ---------------------------
    # Fallback: heuristics parsing
    # ---------------------------
    if not use_llm_sql:
        raw_filters = instr.get("filters", {}) if isinstance(instr, dict) else {}
        filters = normalize_filters(raw_filters)

        # Heuristics
        if "followup" not in filters and re.search(r"follow[- ]?up|flagged", question, re.IGNORECASE):
            q = question.lower()

            if re.search(r"(not|no|without|false|pending|missing|unresolved).*follow[- ]?up", q) or \
            re.search(r"follow[- ]?up.*(not|no|without|false|pending|missing|unresolved)", q):
                filters["followup"] = False

            elif re.search(r"(with|has|completed|maintained|true|done).*follow[- ]?up", q) or \
                re.search(r"follow[- ]?up.*(with|has|completed|maintained|true|done)", q):
                filters["followup"] = True

        lvl = re.search(r"\blevel\s*(\d+)\b", question, re.IGNORECASE)
        if lvl and "hierarchylevel" not in filters:
            filters["hierarchylevel"] = int(lvl.group(1))

        if "value_num" not in filters and mentions_value(question):
            rng = detect_numeric_range(question, VALUE_KEYWORDS)
            if rng:
                filters["value_num"] = rng
            else:
                cmp = detect_numeric_comparator(question, VALUE_KEYWORDS)
                if cmp:
                    filters["value_num"] = cmp

        date_col, start_dt, end_dt, sort_dir = extract_date_range(question)
        if date_col not in ORDERABLE_COLUMNS:
            date_col = "startdate"

        where_clauses, params = build_where(filters, start_dt, end_dt, date_col)
        where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"
        order_by = f"{date_col} {sort_dir}"
        query_limit = row_limit
        date_col, start_dt, end_dt, _ = extract_date_range(question)

        if start_dt or end_dt:
            date_clauses = []
            date_params = []

            if start_dt:
                date_clauses.append(f"{date_col} >= %s")
                date_params.append(start_dt.to_pydatetime())

            if end_dt:
                date_clauses.append(f"{date_col} < %s")
                date_params.append(end_dt.to_pydatetime())

            where_sql = f"({where_sql}) AND " + " AND ".join(date_clauses)
            params.extend(date_params)
        if not where_sql:
            return None
    print("Where SQL is:",where_sql)
    print("Params are: ",params)

    summary_mode = is_summary_question(question)

    # -------------------------------
    # CASE 1: SUMMARY MODE
    # -------------------------------
    if summary_mode:

        # Insights-only request (second call)
        if include_insights and not include_rows:
            insights = build_summary_insights(where_sql, conn_str, params)

            return make_json_safe({
                "view": "summary",
                "columns": [],
                "rows": [],
                "row_count": 1,
                "insights": insights,
                "insights_available": True
            })

        # Normal summary table request
        sql = build_summary_query(where_sql)
        df = run_query(conn_str, sql, params)

        clean_df = make_user_friendly_table(df)

        return make_json_safe({
            "view": "summary",
            "columns": list(clean_df.columns),
            "rows": clean_df.values.tolist(),
            "row_count": 1,
            "insights": None,
            "insights_available": True
        })
    # -------------------------------
    # CASE 2: RECORDS MODE
    # -------------------------------
    
    # Rows-only request
    if include_rows:

        sql =  build_query(select_sql=select_sql, where_sql=where_sql, order_by=order_by, group_by=group_by)
        print(sql)
        df = run_query(conn_str, sql, params + [query_limit, query_offset])
        print(df.head())
        if not df.empty and "total_count" in df.columns:
            total_rows = int(df["total_count"].iloc[0])
        elif not df.empty:
            total_rows = len(df)
        else:
            total_rows = 0
        print(total_rows)
        df_clean_input = df.drop(columns=["total_count"], errors="ignore")
        clean_df = make_user_friendly_table(df_clean_input)
        print(clean_df.head())
        # Ensure consistent output even if empty
        if clean_df is None:
            clean_df = pd.DataFrame()
        columns = list(clean_df.columns) if not clean_df.empty else []
        return make_json_safe({
            "view": "records",
            "columns": columns,
            "rows": clean_df.fillna("").values.tolist() if not clean_df.empty else [],
            "row_count": total_rows if total_rows is not None else 0,
            "insights": None,
            "insights_available": True
        })


    # Insights-only request (second call)
    if include_insights and not include_rows:

        insights = build_summary_insights(where_sql, conn_str, params)
        return make_json_safe({
            "view": "records",
            "columns": [],
            "rows": [],
            "row_count": 0,
            "insights": insights,
            "insights_available": True
        })


    # Fallback
    return make_json_safe({
        "view": "records",
        "columns": [],
        "rows": [],
        "row_count": 0,
        "insights": None,
        "insights_available": False
    })

def explainable_query_answer(question: str, row_limit: int, conn_str: str, hf_token: str, columns_list: list, today_str: date, debug: bool = False)->dict:
    print("Entered explainable query answer")
    intent_prompt = f"""
You are a strict information extraction engine for DHIS2 health facility data.
CRITICAL RULES:
Return ONLY valid JSON.
Do NOT include explanations.
Do NOT include markdown.
Do NOT include any text outside JSON.
Output must be syntactically valid JSON.
All lists must be arrays.
If a field is missing, return null (not empty string, not empty list).
Required Output Format:
{{
"orgunit": ["string"] | null,
"metric": ["string"] | null
}}
Domain Definitions:
Organisation Unit (orgunit):
A named health facility or reporting unit explicitly written in the question.
Examples: Moyowa MCHP, Bo Government Hospital, Kenema Clinic, Red Cross, Loreto
Metric:
A health-related term explicitly written in the question referring to:
Disease (e.g., malaria, TB, HIV, measles)
Service (e.g., vaccinations, deliveries, OPD visits)
Commodity (e.g., zinc, ORS, bed nets, vaccines)
Indicator word if explicitly written together (e.g., "Malaria positive")
Strict Extraction Rules:
Extract ONLY exact text spans that appear in the question.
Preserve original casing and wording exactly as written.
Do NOT normalize text.
Do NOT infer missing words.
Do NOT add indicator words.
Example: If question says "malaria cases","malaria outbreak", extract "malaria".
Do NOT extract "malaria cases", "malaria outbreaks".
Do NOT invent organisation units.
Do NOT invent metrics.
Do NOT merge separate entities into one.
If multiple orgunits exist, return all as separate list items.
If no orgunit is explicitly mentioned, return:
"orgunit": null
If no metric is explicitly mentioned, return:
"metric": null
Never guess abbreviations.
Ignore time references (years, months, periods).
Ignore aggregation words (most, total, highest, etc.).
Ignore filter intent words (in, for, of, from, related to, etc.).
Examples:
Question:
Which organisation has most measles outbreak in 2015?
Output:
{{
"orgunit": None,
"metric": ["measles"]
}}
Question:
Compare malaria positive cases between Moyowa MCHP and Bo Government Hospital in 2021
Output:
{{
"orgunit": ["Moyowa MCHP","Bo Government Hospital"],
"metric": ["malaria positive"]
}}
Question:
What cases for Red Cross, Loreto and Bumban MCHP related to were malaria positive and Measles in 2016
Output:
{{
"orgunit": ["Red Cross", "Loreto", "Bumban MCHP"],
"metric": ["malaria positive", "Measles"]
}}
Question:
Trend of TB cases in Kenema Clinic from 2018 to 2022
Output:
{{
"orgunit": ["Kenema Clinic"],
"metric": ["TB"]
}}
Now extract from:
Question:
{question}
JSON:
""".strip()
    print(f'the length of intent prompt is {len(intent_prompt)}')
    instr_intent=get_cached_llm_plan("Instruct",intent_prompt,hf_token)
    print(instr_intent)
    decoded_intent = json.dumps(instr_intent)

    intent_json: Dict[str, Any] = {}
    m = re.search(r"\{[\s\S]*\}", decoded_intent)
    if m:
        try:
            intent_json = json.loads(m.group())
        except Exception:
            intent_json = {}
    intent_json= normalize_metrics(intent_json)
    print(intent_json)
    distinct_metric_df = list()
    if intent_json.get("metric") is not None:
        intent_json["metric"] = [m.strip() for m in intent_json["metric"] if m and m.strip()]
        distinct_metric_sql,distinct_metric_params= build_distinct_dataelement_query(intent_json.get("metric"))
        distinct_metric_df = run_query(conn_str, distinct_metric_sql, distinct_metric_params)
    print("Distinct metric df is ", distinct_metric_df)
    prompt = f"""
You are a STRICT Text-to-SQL planner for a PostgreSQL database.
Return ONLY a valid JSON object.
NO markdown.
NO explanation.
NO additional text.
You query ONLY from a CTE named `base`
with ONLY these columns:
{columns_list}
You are provided VALID metric names from the database:
distinct_metric_list:
{distinct_metric_df}
Intent values extracted from user query (LISTS of raw phrases):
intent_json["metric"] = {intent_json["metric"]}
intent_json["orgunit"] = {intent_json["orgunit"]}

Hard constraints (never violate):

1. NEVER use CASE.
2. NEVER use arithmetic on value_num.
3. NEVER compute differences.
4. NEVER inline literal values in where_sql.
5. ALWAYS use %s placeholders.
6. The number of %s placeholders MUST equal len(params).
7. NEVER mix AND/OR without parentheses.
8. NEVER use AND between multiple dataelement_name filters.
9. NEVER use AND between multiple orgunit_name filters.
10. NEVER include columns not listed in base.
11. If aggregation alias is used (e.g., total_value), order_by MUST use alias.
12. NEVER order by startdate when using aggregate alias.
13. Ignore None, "None", or empty entries in intent lists.

Metric resolution (mandatory first step):
For each phrase in intent_json["metric"]:
- Find semantic matches in distinct_metric_list.
- Select ONLY exact names that exist in distinct_metric_list.
- If multiple match, include ALL.
- If none match, return fail-safe.

Metric filter format:
Single:
dataelement_name ILIKE %s

Multiple:
(dataelement_name ILIKE %s OR dataelement_name ILIKE %s)

Each param must be:
"%<Exact Metric Name From distinct_metric_list>%"

Never partially construct metric names.
Orgunit resolution:
If intent_json["orgunit"] contains valid entries:
Single:
orgunit_name ILIKE %s
Multiple:
(orgunit_name ILIKE %s OR orgunit_name ILIKE %s)
Date rules:
If user specifies time, always use:
startdate >= %s AND startdate < %s
Today:
{today_str}

Interpret:
"in YYYY" → YYYY-01-01 , YYYY+1-01-01
"last X days" → today-X , today+1
"last month" → first day previous month , first day current month

Aggregation rules:

Aggregation REQUIRED if question contains:
"which organisation", "which facility",
"most", "highest", "lowest",
"top", "maximum", "minimum", "least"

Organisation ranking:
select_sql = "orgunit_name, SUM(value_num) AS total_value"
group_by = "orgunit_name"
order_by = "total_value DESC"
limit = 1

General total:
select_sql = "SUM(value_num) AS total_value"
group_by = ""
order_by = "total_value DESC"

Non-aggregation:
select_sql = "*"
group_by = ""
order_by = "startdate DESC"

Causal / outbreak override (highest priority):

If question contains:
"why", "reason", "cause", "outbreak",
"spike", "increase", "rise",
"drop", "decline", "high in", "low in"

AND a time boundary:

Then ALWAYS:
select_sql = "startdate, SUM(value_num) AS total_value"
group_by = "startdate"
order_by = "startdate ASC"
limit = {row_limit}

Outbreak baseline expansion:

If outbreak language AND specific year mentioned:
Expand range by one full year before.

Example:
"in 2015"
→ startdate >= "2014-01-01"
→ startdate < "2016-01-01"

Logical grouping structure:

Final where_sql must follow:
(metric_group)
AND (orgunit_group)
AND date_filter
AND followup_filter

Remove empty sections cleanly.
Every OR group MUST be inside parentheses.

Fail-safe (return exactly this if unsure or no metric match):
{{
  "select_sql": "*",
  "where_sql": "FALSE",
  "params": [],
  "group_by": "",
  "order_by": "startdate DESC",
  "limit": 1
}}

Examples:
Example 1
User question:
Which organisation reported most malaria confirmed cases in 2016?
Output:
{{
  "select_sql": "orgunit_name, SUM(value_num) AS total_value",
  "where_sql": "dataelement_name ILIKE %s AND startdate >= %s AND startdate < %s",
  "params": ["%Malaria Confirmed Cases%","2016-01-01","2017-01-01"],
  "group_by": "orgunit_name",
  "order_by": "total_value DESC",
  "limit": 1
}}

Example 2
User question:
Compare malaria positive cases between Moyowa MCHP and Bo Government Hospital in 2021
Output:
{{
  "select_sql": "orgunit_name, SUM(value_num) AS total_value",
  "where_sql": "(orgunit_name ILIKE %s OR orgunit_name ILIKE %s) AND dataelement_name ILIKE %s AND startdate >= %s AND startdate < %s",
  "params": ["%Moyowa MCHP%","%Bo Government Hospital%","%Malaria Positive Cases%","2021-01-01","2022-01-01"],
  "group_by": "orgunit_name",
  "order_by": "total_value DESC",
  "limit": {row_limit}
}}

Example 3
User question:
in 2015 why was there a high measles outbreak?

Output:
{{
  "select_sql": "startdate, SUM(value_num) AS total_value",
  "where_sql": "dataelement_name ILIKE %s AND startdate >= %s AND startdate < %s",
  "params": ["%Measles Cases%","2014-01-01","2016-01-01"],
  "group_by": "startdate",
  "order_by": "startdate ASC",
  "limit": {row_limit}
}}

User question:
{question}
""".strip()
    instr = get_cached_llm_plan("SQL", prompt, hf_token)
    print("Instr is ", instr)
    decoded = json.dumps(instr)
    instr: Dict[str, Any] = {}
    m = re.search(r"\{[\s\S]*\}", decoded)
    if m:
        try:
            instr = json.loads(m.group())
        except Exception:
            instr = {}
    use_llm_sql = False

    select_sql = "*"
    where_sql = "TRUE"
    params: List[Any] = []
    order_by = ""
    group_by = ""
    query_limit = row_limit  # always use system max limit

    if isinstance(instr, dict) and any(
        k in instr for k in ("where_sql", "params", "order_by", "limit", "select_sql", "group_by")
    ):
        llm_where = instr.get("where_sql", "")
        llm_params = instr.get("params", [])
        llm_order = instr.get("order_by", order_by)
        llm_select = instr.get("select_sql", select_sql)
        llm_group = instr.get("group_by", group_by)

        if isinstance(llm_where, str):
            where_sql = llm_where.strip() or "TRUE"

        if isinstance(llm_params, list):
            params = llm_params

        if isinstance(llm_order, str):
            order_by = llm_order

        if isinstance(llm_select, str) and llm_select.strip():
            select_sql = llm_select.strip()

        if isinstance(llm_group, str):
            group_by = llm_group.strip()

    ok, err = validate_llm_where_sql(where_sql, params)

    if ok and where_sql != "TRUE":
        use_llm_sql = True

        if " OR " in where_sql and " AND " in where_sql:
            where_sql = auto_group_where(where_sql)

        if re.search(
            r"(?:last|previous|past|recent|trailing)\s+\d+\s+months?\s+of\s+20\d{2}",
            question.lower(),
        ):
            date_col, start_dt, end_dt, sort_dir = extract_date_range(question)

            if start_dt or end_dt:
                date_params = []

                if start_dt:
                    date_params.append(start_dt.to_pydatetime())

                if end_dt:
                    date_params.append(end_dt.to_pydatetime())

                old_date_count = len(date_params)
                params = params[:-old_date_count] if old_date_count else params
                params.extend(date_params)

    # ---------------------------
    # Fallback: heuristics parsing
    # ---------------------------
    if not use_llm_sql:
        raw_filters = instr.get("filters", {}) if isinstance(instr, dict) else {}
        filters = normalize_filters(raw_filters)

        if "followup" not in filters and re.search(r"follow[- ]?up|flagged", question, re.IGNORECASE):
            q = question.lower()

            if re.search(
                r"(not|no|without|false|pending|missing|unresolved).*follow[- ]?up",
                q,
            ) or re.search(
                r"follow[- ]?up.*(not|no|without|false|pending|missing|unresolved)",
                q,
            ):
                filters["followup"] = False

            elif re.search(
                r"(with|has|completed|maintained|true|done).*follow[- ]?up",
                q,
            ) or re.search(
                r"follow[- ]?up.*(with|has|completed|maintained|true|done)",
                q,
            ):
                filters["followup"] = True

        lvl = re.search(r"\blevel\s*(\d+)\b", question, re.IGNORECASE)
        if lvl and "hierarchylevel" not in filters:
            filters["hierarchylevel"] = int(lvl.group(1))

        if "value_num" not in filters and mentions_value(question):
            rng = detect_numeric_range(question, VALUE_KEYWORDS)
            if rng:
                filters["value_num"] = rng
            else:
                cmp = detect_numeric_comparator(question, VALUE_KEYWORDS)
                if cmp:
                    filters["value_num"] = cmp

        date_col, start_dt, end_dt, sort_dir = extract_date_range(question)
        if date_col not in ORDERABLE_COLUMNS:
            date_col = "startdate"

        where_clauses, params = build_where(filters, start_dt, end_dt, date_col)
        where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"
        order_by = f"{date_col} {sort_dir}"
        query_limit = row_limit

        if start_dt or end_dt:
            date_clauses = []
            date_params = []

            if start_dt:
                date_clauses.append(f"{date_col} >= %s")
                date_params.append(start_dt.to_pydatetime())

            if end_dt:
                date_clauses.append(f"{date_col} < %s")
                date_params.append(end_dt.to_pydatetime())

            where_sql = f"({where_sql}) AND " + " AND ".join(date_clauses)
            params.extend(date_params)

        if not where_sql:
            return None

    print("Where SQL is:", where_sql)
    print("Params are:", params)
    sql=build_explainable_query(select_sql=select_sql, where_sql=where_sql, order_by=order_by, group_by=group_by)
    print("Final SQL is:", sql)
    df=run_query(conn_str, sql, params)
    print(df.head())
    if "dataelement_name" in df.columns:
        yearly_sql=build_yearly_query(dataelement_include=True,where_sql=where_sql)
    else:
        yearly_sql=build_yearly_query(dataelement_include=False,where_sql=where_sql)
    yearly_df=run_query(conn_str, yearly_sql, params)
    yearly_df["year"] = yearly_df["year"].astype(int)
    print(yearly_df.head())
    # yearly_dataelement_df=build_yearly_query(dataelement_include=True,where_sql=where_sql)
    # yearly_dataelement_df=run_query(conn_str, yearly_dataelement_df, params)
    # print(yearly_dataelement_df.head())
    insights=build_universal_analysis_payload(df, yearly_df)
    print("Insights are:", insights)

    return explain_query(insights,question, hf_token)

def answer_question(question: str, debug: bool = False, page: int = 1, page_size: int = 100, include_insights: bool=False, include_rows:bool=True) -> dict:
    row_limit = int(os.environ.get("ROW_LIMIT", "400"))

    # DB connection from env
    conn_str = os.environ.get("DHIS2_DSN", "").strip() or build_conn_str_from_parts()
    if not conn_str:
        raise ValueError(
            "Missing DB config. Set DHIS2_DSN or PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD."
        )

    hf_token = os.environ.get("HF_TOKEN", "").strip() or HF_TOKEN_FALLBACK
    if not hf_token:
        raise ValueError("Missing HF_TOKEN in environment variables.")

    question = question.strip()
    if not question:
        raise ValueError("No question provided.")

    columns_list = ", ".join(sorted(BASE_COLUMNS))
    today_str = date.today().isoformat()

    classification_prompt = f"""
You are a STRICT query classifier for a DHIS2 health analytics system.
CRITICAL:
Return ONLY valid JSON.
No explanation.
No markdown.
No extra text.
Task:
Classify whether the user question requires:
- "normal" → factual data retrieval (numbers, lists, counts, simple filters, ranking without interpretation)
- "explainable" → analysis, reasoning, trends, causes, interpretation, or analytical discussion

Output format:
{{
"classification": "normal" or "explainable",
"reason": "short reason (max 10 words)"
}}

Classification rules:

NORMAL queries include:
- counts
- lists
- filters
- specific values
- simple aggregation
- ranking questions without explanation intent

Examples:
"How many measles cases in 2015"
"Show malaria cases in Moyowa MCHP"
"List facilities with TB cases"
"Which organisation had the most malaria cases in 2015"
"What is the highest value in 2022"

EXPLAINABLE queries include:
- why
- reason
- explain
- trend
- increase/decrease
- comparison with interpretation
- outbreak analysis
- performance analysis
- requests for interpretation or reasoning

Examples:
"Why was there measles outbreak in 2015"
"Explain malaria increase in Kenema"
"Why are cases higher in 2016"
"Is there a trend in measles cases"
"Why is Red Cross reporting more cases"
"Compare malaria trends between 2015 and 2016"

Decision rules:
- If the user asks ONLY for data → normal
- If the user asks for explanation, reasoning, trend, interpretation, or analytical discussion → explainable
- Ranking questions (most, highest, top, lowest) without explicit explanation words are NORMAL.
- Only classify as explainable if interpretation is explicitly requested.

User question:
{question}

JSON:
    """.strip()
    clssification_instr=get_cached_llm_plan("Instruct",classification_prompt, hf_token)
    print("Classification instruction is ",clssification_instr)
    decoded = json.dumps(clssification_instr)
    clssification_instr: Dict[str, Any] = {}
    m = re.search(r"\{[\s\S]*\}", decoded)
    if m:
        try:
            clssification_instr = json.loads(m.group())
        except Exception:
            clssification_instr = {}
    if clssification_instr.get("classification")=="normal":
        return normal_query_answer(question, row_limit, conn_str, hf_token, columns_list, today_str, debug, page, page_size, include_insights, include_rows)
    elif clssification_instr.get("classification")=="explainable":
        answer= explainable_query_answer(question, row_limit, conn_str, hf_token, columns_list, today_str, debug)
        return {
            "view": "explainable",
            "answer": answer,
            "insights_available": False
        }

def main():
    question = input("\nEnter your question: ").strip()
    if not question:
        raise SystemExit("ERROR: No question provided.")

    try:
        result = answer_question(question, debug=True, page=1, page_size=200)
    except Exception as e:
        raise SystemExit(f"ERROR: {e}")

    print("\n================ API STYLE OUTPUT ================")
    print(json.dumps(result, indent=2, default=str))

    # Optional: pretty print a table view too
    if "columns" in result and "rows" in result:
        df = pd.DataFrame(result["rows"], columns=result["columns"])
        print("\n================ USER FRIENDLY TABLE (first 20) ================")
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
