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
from dotenv import load_dotenv
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


def extract_date_range(
    question: str,
) -> Tuple[str, pd.Timestamp | None, pd.Timestamp | None, str]:
    date_col = choose_date_column(question)
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    today = date.today()

    if start is None and end is None:
        if re.search(r"\blast year\b", question, re.IGNORECASE):
            year = today.year - 1
            start = pd.Timestamp(year=year, month=1, day=1)
            end = (
                pd.Timestamp(year=year, month=12, day=31)
                + pd.Timedelta(days=1)
                - pd.Timedelta(microseconds=1)
            )
        elif re.search(r"\b(this year|current year)\b", question, re.IGNORECASE):
            year = today.year
            start = pd.Timestamp(year=year, month=1, day=1)
            end = (
                pd.Timestamp(year=year, month=12, day=31)
                + pd.Timedelta(days=1)
                - pd.Timedelta(microseconds=1)
            )

    # between X and Y
    m_between = re.search(
        r"\bbetween\s+([^\s,]+)\s+(?:and|-)\s+([^\s,]+)", question, re.IGNORECASE
    )
    if m_between:
        a, b = m_between.group(1), m_between.group(2)
        if looks_like_date_token(a) or looks_like_date_token(b):
            try:
                d1 = dateutil_parser.parse(a, fuzzy=True)
                d2 = dateutil_parser.parse(b, fuzzy=True)
                if d1 <= d2:
                    start = normalize_date(pd.Timestamp(d1))
                    end = (
                        normalize_date(pd.Timestamp(d2))
                        + pd.Timedelta(days=1)
                        - pd.Timedelta(microseconds=1)
                    )
                else:
                    start = normalize_date(pd.Timestamp(d2))
                    end = (
                        normalize_date(pd.Timestamp(d1))
                        + pd.Timedelta(days=1)
                        - pd.Timedelta(microseconds=1)
                    )
            except Exception:
                pass

    # from X to Y
    if start is None:
        m_from = re.search(
            r"from\s+([^\s,]+)\s+to\s+([^\s,]+)", question, re.IGNORECASE
        )
        if m_from:
            t1, t2 = m_from.group(1), m_from.group(2)
            if looks_like_date_token(t1) or looks_like_date_token(t2):
                try:
                    d1 = dateutil_parser.parse(t1, fuzzy=True)
                    d2 = dateutil_parser.parse(t2, fuzzy=True)
                    if d1 <= d2:
                        start = normalize_date(pd.Timestamp(d1))
                        end = (
                            normalize_date(pd.Timestamp(d2))
                            + pd.Timedelta(days=1)
                            - pd.Timedelta(microseconds=1)
                        )
                    else:
                        start = normalize_date(pd.Timestamp(d2))
                        end = (
                            normalize_date(pd.Timestamp(d1))
                            + pd.Timedelta(days=1)
                            - pd.Timedelta(microseconds=1)
                        )
                except Exception:
                    pass

    # after/since
    if start is None:
        m_after = re.search(
            r"\b(?:after|since)\s+([^\s,]+)", question, re.IGNORECASE
        )
        if m_after and looks_like_date_token(m_after.group(1)):
            try:
                d = dateutil_parser.parse(m_after.group(1), fuzzy=True)
                start = normalize_date(pd.Timestamp(d))
            except Exception:
                pass

    # before
    if end is None:
        m_before = re.search(
            r"\b(?:before)\s+([^\s,]+)", question, re.IGNORECASE
        )
        if m_before and looks_like_date_token(m_before.group(1)):
            try:
                d = dateutil_parser.parse(m_before.group(1), fuzzy=True)
                end = (
                    normalize_date(pd.Timestamp(d))
                    + pd.Timedelta(days=1)
                    - pd.Timedelta(microseconds=1)
                )
            except Exception:
                pass

    # Month YYYY
    if start is None and end is None:
        m_month = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+([0-9]{4})\b",
            question,
            re.IGNORECASE,
        )
        if m_month:
            month_str = m_month.group(1)
            year = int(m_month.group(2))
            try:
                dt_start = pd.Timestamp(f"{month_str} 1 {year}")
                last_day = monthrange(year, dt_start.month)[1]
                start = normalize_date(dt_start)
                end = (
                    pd.Timestamp(year=year, month=dt_start.month, day=last_day)
                    + pd.Timedelta(days=1)
                    - pd.Timedelta(microseconds=1)
                )
            except Exception:
                pass

    # YYYY-MM
    if start is None and end is None:
        ym = re.search(r"\b(20\d{2})[\/\-](0[1-9]|1[0-2])\b", question)
        if ym:
            year = int(ym.group(1))
            month = int(ym.group(2))
            start = pd.Timestamp(year=year, month=month, day=1)
            last_day = monthrange(year, month)[1]
            end = (
                pd.Timestamp(year=year, month=month, day=last_day)
                + pd.Timedelta(days=1)
                - pd.Timedelta(microseconds=1)
            )

    # Year only
    if start is None and end is None:
        y = re.search(r"\b(20\d{2})\b", question)
        if y:
            year = int(y.group(1))
            start = pd.Timestamp(year=year, month=1, day=1)
            end = (
                pd.Timestamp(year=year, month=12, day=31)
                + pd.Timedelta(days=1)
                - pd.Timedelta(microseconds=1)
            )

    sort_dir = "DESC"
    if re.search(r"\b(oldest|ascending|earliest first)\b", question, re.IGNORECASE):
        sort_dir = "ASC"
    elif re.search(r"\b(newest|descending|latest first)\b", question, re.IGNORECASE):
        sort_dir = "DESC"

    return date_col, start, end, sort_dir


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
        dv.followup
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
        dv.followup
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


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

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

    # normal python float NaN/inf
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    return obj

# ---------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------

def answer_question(question: str, debug: bool = False, page: int = 1, page_size: int = 200) -> dict:
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
1. Identify the **metric** or data element (e.g. malaria, deaths, visits).
   → Use `dataelement_name ILIKE %s` to filter.
2. Identify the **location or facility** (e.g. district, hospital).
   → Use `orgunit_name ILIKE %s` to filter.
3. Identify any **date/time filters** (e.g. last 3 months, in 2016).
   → Use `startdate >= %s AND startdate < %s` for filtering.
4. Identify **sorting direction** (e.g. latest → DESC).
5. Identify any **record limit** (e.g. “top 10 outbreaks” → limit = 10).

Handle these phrases using ISO dates:
- “last X days” → compute today minus X days to today
- “last month” → from 1st of previous month to 1st of current month
- “first 75 days of 2016” → "2016-01-01" to "2016-03-16"
- “in 2022” → "2022-01-01" to "2023-01-01"
- “between Jan 2020 and June 2020” → parse both ends into dates

Hard rules:
1) Use ONLY the listed columns. Do not invent column names.
2) Never put user values directly into SQL. Always use %s placeholders.
3) Do not use semicolons, joins, subqueries, or multiple statements.
4) Prefer ILIKE for text matching.
5) If user mentions an organisation/facility/district name, filter using:
   orgunit_name ILIKE %s
6) If user mentions a metric/indicator/disease (e.g., malaria, cholera, deaths, tests), filter using:
   dataelement_name ILIKE %s
7) If user requests a time period, filter using startdate:
   startdate >= %s AND startdate < %s
   Use ISO dates (YYYY-MM-DD).
8) MUST include all filters mentioned in question.
9) If user asks “latest/recent”, order by startdate DESC.
   If user asks “oldest/earliest”, order by startdate ASC.
10) If the user asks for “top N”, return records but set limit=N and order by value_num DESC when value_num is relevant.
11) If you are unsure, use where_sql="TRUE" and choose a safe limit.

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
User question: "Show records of malaria in first 75 days of 2016"
Output:
{{
  "where_sql": "dataelement_name ILIKE %s AND startdate >= %s AND startdate < %s",
  "params": ["%malaria%", "2016-01-01", "2016-03-16"],
  "order_by": "startdate DESC",
  "limit": 200
}}

User question:
{question}
""".strip()

    decoded = run_llm_prompt(prompt, hf_token).strip()

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


    # ---------------------------
    # Try LLM plan
    # ---------------------------
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
    # ---------------------------
    # Fallback: heuristics parsing
    # ---------------------------
    if not use_llm_sql:
        raw_filters = instr.get("filters", {}) if isinstance(instr, dict) else {}
        filters = normalize_filters(raw_filters)

        # Heuristics
        if "followup" not in filters and re.search(r"follow[- ]?up|flagged", question, re.IGNORECASE):
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

    summary_mode = is_summary_question(question)

    if summary_mode:
        sql = build_summary_query(where_sql)
        df = run_query(conn_str, sql, params)
        clean_df = df.copy()

        clean_df = clean_df.rename(
            columns={
                "total_value": "Total",
                "record_count": "Records",
            }
        )

    else:
        sql = build_query(where_sql, order_by)
        df = run_query(conn_str, sql, params + [query_limit, query_offset])

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
        dv.followup
    FROM datavalue dv
    JOIN dataelement de ON dv.dataelementid = de.dataelementid
    JOIN organisationunit ou ON dv.sourceid = ou.organisationunitid
    JOIN period p ON dv.periodid = p.periodid
    LEFT JOIN periodtype pt ON p.periodtypeid = pt.periodtypeid
)
SELECT COUNT(*) FROM base WHERE {where_sql};
"""

        count_df = run_query(conn_str, count_sql, params)
        total_rows = int(count_df.iloc[0, 0])
        clean_df = make_user_friendly_table(df)

    result = {
        "view": "summary" if summary_mode else "records",
        "columns": list(clean_df.columns),
        "rows": clean_df.values.tolist(),
        "row_count": total_rows,
    }

    if debug:
        result["debug"] = {
            "raw_model_output": decoded,
            "where_sql": where_sql,
            "params": params,
            "order_by": order_by,
            "limit": query_limit,
            "sql": sql,
            "raw_columns": list(df.columns),
        }

    return make_json_safe(result)

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
