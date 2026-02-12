import json
import os
import re
from calendar import monthrange
from datetime import date
from typing import Any, Dict, List, Tuple

import pandas as pd
import math
import numpy as np
from datetime import datetime, date
import psycopg2
from dateutil import parser as dateutil_parser
from openai import OpenAI
from duckling_client import parse_dates_with_duckling
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


def run_llm_prompt(prompt: str, token: str) -> str:
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

    LAST_WORDS = r"(last|previous|past|recent|trailing)"
    MONTH_WORDS = r"(month|months|mth|mths)"
    DAY_WORDS = r"(day|days)"


    m = re.search(r"(20\d{2})\s*(?:to|and|-)\s*(20\d{2})", q)
    if m:
        y1, y2 = sorted([int(m.group(1)), int(m.group(2))])
        start = pd.Timestamp(y1, 1, 1)
        end = pd.Timestamp(y2 + 1, 1, 1)
        return date_col, start, end, "DESC"

    m = re.search(
        rf"{LAST_WORDS}\s+(\d+)\s+{MONTH_WORDS}\s+of\s+(20\d{{2}})",
        q,
    )
    if m:
        months = int(m.group(2))
        year = int(m.group(3))

        months = max(1, min(months, 12))

        start_month = 13 - months
        start = pd.Timestamp(year, start_month, 1)
        end = pd.Timestamp(year + 1, 1, 1)

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


# def extract_date_range(question: str):
#     date_col = "startdate"

#     start_dt, end_dt = parse_dates_with_duckling(question)

#     if start_dt and end_dt:
#         return date_col, start_dt, end_dt, "DESC"

#     return date_col, None, None, "DESC"


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

def build_query(where_sql: str, order_by: str) -> str:
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
SELECT *
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

def build_trend_query(where_sql: str) -> str:
    return f"""
    WITH base AS (
        SELECT
            dv.dataelementid,
            de.name AS dataelement_name,
            ou.name AS orgunit_name,
            p.startdate,
            CASE WHEN dv.value ~ '^[-]?\\d+(\\.\\d+)?$'
                 THEN dv.value::numeric
            END AS value_num
        FROM datavalue dv
        JOIN period p ON dv.periodid = p.periodid
        JOIN dataelement de ON dv.dataelementid = de.dataelementid
        JOIN organisationunit ou ON dv.sourceid = ou.organisationunitid
    )
    SELECT
        DATE_TRUNC('month', startdate) AS period,
        SUM(value_num) AS total_value
    FROM base
    WHERE {where_sql}
    GROUP BY 1
    ORDER BY 1;
    """

def build_chart_query(where_sql: str) -> str:
    """
    Returns aggregated data for charts (NO pagination).
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
        orgunit_name,
        dataelement_name,
        SUM(value_num) AS total_value
    FROM base
    WHERE {where_sql}
    GROUP BY orgunit_name, dataelement_name;
    """.strip()

def count_total_rows(where_sql:str,conn_str: str, params: List[Any]):
    count_sql = f"""
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
                SELECT COUNT(*) FROM base WHERE {where_sql};
                """

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


def normalize_order_by(order_by: str, default: str) -> str:
    m = re.match(
        r"^\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*(ASC|DESC)?\\s*$",
        order_by,
        re.IGNORECASE,
    )
    if not m:
        return default
    col = m.group(1).lower()
    if col not in {c.lower() for c in BASE_COLUMNS}:
        return default
    direction = (m.group(2) or "DESC").upper()
    return f"{col} {direction}"


# ---------------------------------------------------------------------
# Output Sanitizer (User-friendly table)
# ---------------------------------------------------------------------

def make_user_friendly_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops technical columns and keeps only columns useful for non-technical users.
    Also renames headers for UI.
    """
    keep_cols = [
        "orgunit_name",
        "dataelement_name",
        "startdate",
        "enddate",
        "period_type",
        "value_num",
        "value",
        "followup",
    ]

    cols = [c for c in keep_cols if c in df.columns]
    clean = df[cols].copy()

    # Create a single "Value" column (prefer numeric)
    if "value_num" in clean.columns and clean["value_num"].notna().any():
        clean["Value"] = clean["value_num"]
    elif "value" in clean.columns:
        clean["Value"] = clean["value"]
    else:
        clean["Value"] = None

    # Drop raw value columns
    clean = clean.drop(columns=[c for c in ["value_num", "value"] if c in clean.columns])

    # Rename columns for UI
    rename_map = {
        "orgunit_name": "Organisation Unit",
        "dataelement_name": "Metric",
        "startdate": "Start Date",
        "enddate": "End Date",
        "period_type": "Period Type",
        "followup": "Follow-up",
    }
    clean = clean.rename(columns=rename_map)

    # Format dates
    if "Start Date" in clean.columns:
        clean["Start Date"] = pd.to_datetime(clean["Start Date"], errors="coerce").dt.date
    if "End Date" in clean.columns:
        clean["End Date"] = pd.to_datetime(clean["End Date"], errors="coerce").dt.date

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

def build_insights(where_sql:str,conn_str: str, params: List[Any]):
    chart_sql = build_chart_query(where_sql)
    chart_df = run_query(conn_str, chart_sql, params)
    insights = {}
    if not chart_df.empty:
        unique_orgs = chart_df["orgunit_name"].nunique()
        unique_metrics = chart_df["dataelement_name"].nunique()

        if unique_orgs > 1:
            top = (
                chart_df.groupby("orgunit_name")["total_value"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                )
            insights["mode"] = "top_orgs"
            insights["data"] = [
                    {"name": r["orgunit_name"], "total": float(r["total_value"])}
                    for _, r in top.iterrows()
                ]

        elif unique_metrics > 1:
            top = (
                chart_df.groupby("dataelement_name")["total_value"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                )
            insights["mode"] = "top_metrics"
            insights["data"] = [
                    {"name": r["dataelement_name"], "total": float(r["total_value"])}
                    for _, r in top.iterrows()
                ]

        else:
            total = chart_df["total_value"].sum()

            insights["mode"] = "single_total"
            insights["data"] = [
                {"name": "Total", "total": float(total)}
            ]
    return insights

def build_trend_insight(where_sql: str, conn_str: str, params: List[Any]) -> dict:
    """
    Insights for summary mode → trend chart.
    """
    trend_sql = build_trend_query(where_sql)
    trend_df = run_query(conn_str, trend_sql, params)
    if trend_df.empty:
        return {
            "mode": "none",
            "data": []
        }

    return {
        "mode": "trend",
        "data": [
            {
                "date": str(r["period"].date()),
                "total": float(r["total_value"])
            }
            for _, r in trend_df.iterrows()
        ]
    }

def build_summary_insights(where_sql, conn_str, params):

    chart_sql = build_chart_query(where_sql)
    chart_df = run_query(conn_str, chart_sql, params)

    if chart_df.empty:
        return {"mode": "none", "data": []}

    unique_orgs = chart_df["orgunit_name"].nunique()
    unique_metrics = chart_df["dataelement_name"].nunique()

    # ---------------------------------------------------
    # 1️⃣ Try trend first (most informative for summaries)
    # ---------------------------------------------------
    trend = build_trend_insight(where_sql, conn_str, params)
    if trend["mode"] != "none" and len(trend["data"]) > 1:
        return trend

    # ---------------------------------------------------
    # 2️⃣ Compare organisations
    # ---------------------------------------------------
    if unique_orgs > 1:
        top = (
            chart_df.groupby("orgunit_name")["total_value"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        return {
            "mode": "top_orgs",
            "data": [
                {"name": r["orgunit_name"], "total": float(r["total_value"])}
                for _, r in top.iterrows()
            ],
        }

    # ---------------------------------------------------
    # 3️⃣ Compare metrics
    # ---------------------------------------------------
    if unique_metrics > 1:
        top = (
            chart_df.groupby("dataelement_name")["total_value"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        return {
            "mode": "top_metrics",
            "data": [
                {"name": r["dataelement_name"], "total": float(r["total_value"])}
                for _, r in top.iterrows()
            ],
        }

    # ---------------------------------------------------
    # 4️⃣ Single value fallback
    # ---------------------------------------------------
    total = chart_df["total_value"].sum()

    return {
        "mode": "single_total",
        "data": [{"name": "Total", "total": float(total)}],
    }

def strip_date_filters(where_sql: str, params: List[Any]):
    pieces = re.split(r"\s+AND\s+", where_sql, flags=re.IGNORECASE)

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


@lru_cache(maxsize=300)
def get_cached_llm_plan(prompt: str, token: str) -> dict:
    """
    Calls the LLM once and caches the generated SQL plan.
    Subsequent identical questions reuse cached result.
    Removes repeated LLM latency during:
      - insights fetch
      - pagination
      - re-renders
    """
    decoded = run_llm_prompt(prompt, token).strip()

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


# ---------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------

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

    prompt = f"""
    You are a Text-to-SQL planner for a PostgreSQL database.

    You must return ONLY a valid JSON object (no markdown, no explanation, no SQL outside JSON).

    You are querying from a CTE named `base` with these columns:
    {columns_list}

    Task:
    Given the user’s question, produce a JSON plan with exactly these keys:

    - `where_sql`: SQL boolean condition for filtering rows from `base` (do NOT include the word "WHERE").
    - `params`: array of values corresponding to the `%s` placeholders in `where_sql`.
    - `order_by`: one of the base columns + ASC or DESC (e.g., `startdate DESC`). Use only listed columns.
    - `limit`: integer between 1 and {row_limit}.

    Follow this reasoning process:
    1. Identify the **metric** or data element (e.g. malaria, malaria negative, deaths, visits). If multiple metrics are mentioned, you MUST include ALL of them. Never drop any metric. Create one ILIKE placeholder per metric and combine using OR.
    → Use `dataelement_name ILIKE %s` to filter.
    2. Identify the **facility** by name (e.g. Red Cross Clinic, Ngelehun CHC). If multiple facilities are mentioned, you MUST include ALL of them. Never drop any facility. Create one ILIKE placeholder per facility and combine using OR.
    → Use `orgunit_name ILIKE %s` to filter.
    3. Identify any **date/time filters** (e.g. last 3 months, in 2016).
    → Use `startdate >= %s AND startdate < %s` for filtering.
    4. Identify follow-up status intent phrases meaning completed/maintained/flagged/reviewed → followup IS TRUE while phrases meaning not followed up/pending/unresolved/missing → followup IS FALSE
    -> if mentioned, include filter: followup = %s
    5. Identify **sorting direction** (e.g. latest → DESC).
    6. 6. Logical grouping rules (CRITICAL):
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

    NEVER mix AND/OR without parentheses.

    Extract the exact date and time from the following phrases and return in ISO format:
    - “last X days” → compute today minus X days to today
    - “last month” → from 1st of previous month to 1st of current month
    - “first 75 days of 2016” → "2016-01-01" to "2016-03-16"
    - “in 2022” → "2022-01-01" to "2023-01-01"
    - “last X months of YYYY”
        Definition:
        Take the FINAL X calendar months of that year.

        Compute:
        startdate = first day of month (13 − X)
        enddate   = first day of next year (YYYY+1-01-01)

        Always output:
        startdate >= %s AND startdate < %s

        Examples:
        • last 3 months of 2016 → 2016-09-01 to 2017-01-01
        • last 6 months of 2016 → 2016-06-01 to 2017-01-01
        • last 4 month of 2016 → 2016-08-01 to 2017-01-01
    - “between Jan 2020 and June 2020” → parse both ends into dates

    Hard rules:
    1) Use ONLY the listed columns. Do not invent column names.
    2) Never put user values directly into SQL. Always use %s placeholders.
    3) Do not use semicolons, joins, subqueries, or multiple statements.
    4) Prefer ILIKE for text matching.
    5) If user mentions an organisation/facility/district name, filter using: orgunit_name ILIKE %s
    6) If user mentions a metric/indicator/disease (e.g., malaria, malaria negative,cholera, deaths, tests), filter using: dataelement_name ILIKE %s
    7) If user requests a time period, filter using startdate:
    startdate >= %s AND startdate < %s
    Use ISO dates (YYYY-MM-DD).
    8)If the question references follow-up status in any way,
    use the column: followup = %s
    (TRUE or FALSE based on intent)
    9) MUST include all filters mentioned in question.
    10) If user asks “latest/recent”, order by startdate DESC.
    If user asks “oldest/earliest”, order by startdate ASC.
    11) NEVER output mixed AND/OR without parentheses.
    12) Every OR group MUST be inside brackets.
    13) If you are unsure or the user question is vague or chit chat intentsion use where_sql="FALSE".
    14) You MUST ALWAYS return all four keys.
    15) Never return empty JSON always return a valid JSON plan. So Return EXACTLY this:
    {{
        "where_sql": "FALSE",
        "params": [],
        "order_by": "startdate DESC",
        "limit": 1
    }}

    Today’s date is: {today_str}

    Example 1:
    User question: "Show records from Loreto Clinic related to malaria cases between Jan to April 2016"
    {{
    "where_sql": "orgunit_name ILIKE %s AND dataelement_name ILIKE %s AND startdate >= %s AND startdate < %s",
    "params": ["%Loreto Clinic%", "%malaria%","2016-01-01", "2016-04-01"],
    "order_by": "startdate DESC",
    "limit": 200
    }}

    Example 2:
    User question: "Show records of malaria with non followed up cases in first 75 days of 2016"
    Output:
    {{
    "where_sql": "dataelement_name ILIKE %s AND followup = %s AND startdate >= %s AND startdate < %s",
    "params": ["%malaria%", FALSE,"2016-01-01", "2016-03-16"],
    "order_by": "startdate DESC",
    "limit": 200
    }}

    Example 3:
    User quesiton: "How many red cross clinic patients were affected by malaria negative related results in last 4 months of 2016"
    {{
    "where_sql": "orgunit_name ILIKE %s AND dataelement_name ILIKE %s AND startdate >= %s AND startdate < %s"
    "params": ['%Red Cross Clinic%', '%malaria negative%', '2016-08-01', '2016-12-31']
    "order_by": "startdate Desc",
    "limit": 200
    }}
    User question:
    {question}
    """.strip()

    instr = get_cached_llm_plan(prompt, hf_token)
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
    where_sql = "TRUE"
    params: List[Any] = []
    order_by = "startdate DESC"
    query_offset = max(0, (page - 1) * page_size)
    query_limit = page_size
    if isinstance(instr, dict) and any(k in instr for k in ("where_sql", "params", "order_by", "limit")):
        llm_where = instr.get("where_sql", "")
        llm_params = instr.get("params", [])
        llm_order = instr.get("order_by", order_by)
        llm_limit = instr.get("limit", query_limit)
        if isinstance(llm_where, str):
            where_sql = llm_where.strip() or "TRUE"
        if isinstance(llm_params, list):
            params = llm_params
        if isinstance(llm_order, str):
            order_by = normalize_order_by(llm_order, order_by)
        if isinstance(llm_limit, int):
            query_limit = max(1, min(row_limit, llm_limit))
        elif isinstance(llm_limit, str) and llm_limit.isdigit():
            query_limit = max(1, min(row_limit, int(llm_limit)))

    ok, err = validate_llm_where_sql(where_sql, params)

    if ok:
        use_llm_sql = True

        if " OR " in where_sql and " AND " in where_sql:
            parts = re.split(r"\s+AND\s+", where_sql, flags=re.I)

            or_parts = [p for p in parts if " OR " in p]
            other_parts = [p for p in parts if " OR " not in p]

            if or_parts:
                grouped = "(" + " OR ".join(or_parts) + ")"
                where_sql = " AND ".join([grouped] + other_parts)

        if re.search(r"(last|previous|past)\s+\d+\s+months?\s+of\s+20\d{2}",question.lower()):
            print("Heuristic date override triggered")

            where_sql, params = strip_date_filters(where_sql, params)
    
            date_col, start_dt, end_dt, sort_dir = extract_date_range(question)

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
                print("Updated params are:",params)

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

        clean_df = df.rename(columns={
            "total_value": "Total",
            "record_count": "Records"
        })

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

        sql = build_query(where_sql, order_by)
        df = run_query(conn_str, sql, params + [query_limit, query_offset])
        clean_df = make_user_friendly_table(df)

        total_rows = count_total_rows(where_sql, conn_str, params)

        return make_json_safe({
            "view": "records",
            "columns": list(clean_df.columns),
            "rows": clean_df.values.tolist(),
            "row_count": total_rows,
            "insights": None,
            "insights_available": True
        })


    # Insights-only request (second call)
    if include_insights and not include_rows:

        insights = build_insights(where_sql, conn_str, params)

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
