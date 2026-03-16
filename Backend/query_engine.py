import json
import os
import re
import sqlite3
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

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
    "like",
    "in",
    "is",
    "null",
    "true",
    "false",
    "collate",
    "nocase",
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
    "outbreaks",
}


# ---------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------


def sqlite_regexp(pattern: str, value: Any) -> int:
    if value is None:
        return 0
    try:
        return 1 if re.search(pattern, str(value)) else 0
    except re.error:
        return 0



def get_sqlite_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.create_function("REGEXP", 2, sqlite_regexp)
    return conn



def adapt_param_for_sqlite(param: Any) -> Any:
    if isinstance(param, pd.Timestamp):
        return param.strftime("%Y-%m-%d")
    if isinstance(param, datetime):
        return param.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(param, date):
        return param.isoformat()
    if isinstance(param, bool):
        return 1 if param else 0
    return param



def normalize_llm_sqlite_plan(instr: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(instr, dict):
        return {}

    for key in ("where_sql", "select_sql", "group_by", "order_by"):
        val = instr.get(key)
        if isinstance(val, str):
            val = val.replace("%s", "?")
            val = re.sub(r"\bILIKE\b", "LIKE", val, flags=re.IGNORECASE)
            val = re.sub(r"\bTRUE\b", "1", val, flags=re.IGNORECASE)
            val = re.sub(r"\bFALSE\b", "0", val, flags=re.IGNORECASE)
            instr[key] = val

    return instr


# ---------------------------------------------------------------------
# Shared SQL snippets
# ---------------------------------------------------------------------


def base_cte_sql() -> str:
    return """
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
            WHEN REGEXP('^[-]?\\d+(\\.\\d+)?$', TRIM(CAST(dv.value AS TEXT)))
            THEN CAST(dv.value AS REAL)
        END AS value_num,
        dv.comment,
        dv.storedby,
        dv.lastupdated,
        dv.created,
        COALESCE(dv.followup, 0) AS followup
    FROM datavalue dv
    JOIN dataelement de ON dv.dataelementid = de.dataelementid
    JOIN organisationunit ou ON dv.sourceid = ou.organisationunitid
    JOIN period p ON dv.periodid = p.periodid
    LEFT JOIN periodtype pt ON p.periodtypeid = pt.periodtypeid
)
""".strip()


# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------


def normalize_metrics(intent: dict) -> dict:
    metrics = intent.get("metric") or []
    cleaned_metrics = []

    for metric in metrics:
        words = metric.split()
        filtered = [w for w in words if w.lower() not in INDICATOR_SUFFIXES]
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
    return OpenAI(base_url="https://router.huggingface.co/v1", api_key=token)



def run_sql_llm_prompt(prompt: str, token: str) -> str:
    client = get_llm_client(token)
    if client is None:
        raise ValueError("Missing HF_TOKEN for the Hugging Face router.")
    completion = client.chat.completions.create(
        model="defog/llama-3-sqlcoder-8b:featherless-ai",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return completion.choices[0].message.content or ""



def run_instruct_llm_prompt(prompt: str, token: str, payload: dict | None = None, payload_include: bool = False) -> str:
    client = get_llm_client(token)
    if client is None:
        raise ValueError("Missing HF_TOKEN for the Hugging Face router.")

    if payload_include:
        response = client.chat.completions.create(
            model="defog/llama-3-sqlcoder-8b:featherless-ai",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload or {})},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    completion = client.chat.completions.create(
        model="defog/llama-3-sqlcoder-8b:featherless-ai",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return completion.choices[0].message.content or ""



def preheat_database(db_path: str):
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
        with get_sqlite_connection(db_path) as conn:
            conn.execute(warm_sql).fetchone()
    except Exception as e:
        print("⚠️ Database preheat skipped:", e)


# ---------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------


def looks_like_date_token(tok: str) -> bool:
    if not tok:
        return False
    if re.match(r"^20\d{2}([/\-](0[1-9]|1[0-2])([/\-](0[1-9]|[12]\d|3[01]))?)?$", tok):
        return True
    if re.match(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$", tok):
        return True
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



def extract_date_range(question: str) -> Tuple[str, pd.Timestamp | None, pd.Timestamp | None, str]:
    date_col = "startdate"
    q = question.lower()
    today = pd.Timestamp.today().normalize()

    start = None
    end = None

    last_words = r"(?:last|previous|past|recent|trailing)"
    month_words = r"(?:month|months|mth|mths)"
    day_words = r"(?:day|days)"

    m = re.search(r"(20\d{2})\s*(?:to|and|-)\s*(20\d{2})", q)
    if m:
        y1, y2 = sorted([int(m.group(1)), int(m.group(2))])
        start = pd.Timestamp(y1, 1, 1)
        end = pd.Timestamp(y2 + 1, 1, 1)
        return date_col, start, end, "DESC"

    m = re.search(rf"{last_words}\s+(\d+)\s+{month_words}\s+of\s+(20\d{{2}})", q)
    if m:
        months = int(m.group(1))
        year = int(m.group(2))
        months = max(1, min(months, 12))
        start_month = 13 - months
        start = pd.Timestamp(year, start_month, 1)
        end = pd.Timestamp(year + 1, 1, 1)
        return date_col, start, end, "DESC"

    m = re.search(rf"{last_words}\s+(\d+)\s+{month_words}\b", q)
    if m:
        months = int(m.group(1))
        end = today + pd.Timedelta(days=1)
        start = end - pd.DateOffset(months=months)
        return date_col, start, end, "DESC"

    m = re.search(rf"{last_words}\s+(\d+)\s+{day_words}\b", q)
    if m:
        days = int(m.group(1))
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
        start = pd.Timestamp(int(year), pd.to_datetime(mon1).month, int(d1))
        end = pd.Timestamp(int(year), pd.to_datetime(mon2).month, int(d2)) + pd.Timedelta(days=1)
        return date_col, start, end, "DESC"

    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        year = int(m.group(1))
        start = pd.Timestamp(year, 1, 1)
        end = pd.Timestamp(year + 1, 1, 1)
        return date_col, start, end, "DESC"

    sort_dir = "DESC"
    if re.search(r"\b(oldest|earliest|ascending)\b", q):
        sort_dir = "ASC"

    return date_col, None, None, sort_dir


# ---------------------------------------------------------------------
# Numeric filters
# ---------------------------------------------------------------------


def detect_numeric_range(question: str, keywords: Tuple[str, ...]) -> Tuple[str, float, float] | None:
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



def detect_numeric_comparator(question: str, keywords: Tuple[str, ...]) -> Tuple[str, float] | None:
    for kw in keywords:
        base = re.escape(kw).replace("\\ ", "[\\s_-]?")
        pattern = rf"\b{base}\b\s*(>=|<=|>|<|=)\s*(-?\d+(?:\.\d+)?)"
        m = re.search(pattern, question, re.IGNORECASE)
        if m:
            return m.group(1), float(m.group(2))
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
        display[key] = list(val) if isinstance(val, tuple) else val
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
        where.append(f"{date_col} >= ?")
        params.append(start_dt.date().isoformat())
    if end_dt is not None:
        where.append(f"{date_col} < ?")
        params.append(end_dt.date().isoformat())

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
                where.append(f"{column} BETWEEN ? AND ?")
                params.extend([float(low), float(high)])
                continue
            if len(val) == 2 and val[0] in COMPARATORS:
                op, num = val
                where.append(f"{column} {op} ?")
                params.append(float(num))
                continue

        if dtype == "bool":
            where.append(f"{column} = ?")
            params.append(1 if as_bool(val) else 0)
        elif dtype == "numeric":
            try:
                num = float(val)
            except Exception:
                continue
            where.append(f"{column} = ?")
            params.append(num)
        else:
            where.append(f"{column} LIKE ? COLLATE NOCASE")
            params.append(f"%{val}%")

    return where, params


# ---------------------------------------------------------------------
# Query builder & execution
# ---------------------------------------------------------------------


def build_query(select_sql: str, where_sql: str, order_by: str, group_by: str = "") -> str:
    base_cte = base_cte_sql()

    if group_by:
        return f"""
{base_cte}
SELECT
    {select_sql}
FROM base
WHERE {where_sql}
GROUP BY {group_by}
ORDER BY {order_by}
LIMIT ? OFFSET ?
""".strip()

    return f"""
{base_cte}
SELECT
    *,
    COUNT(*) OVER() AS total_count
FROM base
WHERE {where_sql}
ORDER BY {order_by}
LIMIT ? OFFSET ?
""".strip()



def is_summary_question(question: str) -> bool:
    q = question.lower().strip()
    return bool(re.search(r"\b(how many|total|sum|overall|number of|count|counts|counting)\b", q))



def build_summary_query(where_sql: str) -> str:
    base_cte = base_cte_sql()
    return f"""
{base_cte}
SELECT
    COALESCE(SUM(value_num), 0) AS total_value,
    COUNT(*) AS record_count
FROM base
WHERE {where_sql}
""".strip()



def build_explainable_query(select_sql: str, where_sql: str, order_by: str, group_by: str = "") -> str:
    base_cte = base_cte_sql()

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

    return f"""
{base_cte}
SELECT
    {select_sql}
FROM base
WHERE {where_sql}
ORDER BY {order_by}
""".strip()



def build_yearly_query(dataelement_include: bool, where_sql: str, order_by: str = "year ASC") -> str:
    base_cte = base_cte_sql()

    if not dataelement_include:
        return f"""
{base_cte}
SELECT
    CAST(strftime('%Y', startdate) AS INTEGER) AS year,
    SUM(value_num) AS total_value
FROM base
WHERE {where_sql}
GROUP BY year
ORDER BY {order_by}
""".strip()

    return f"""
{base_cte}
SELECT
    CAST(strftime('%Y', startdate) AS INTEGER) AS year,
    dataelement_name,
    SUM(value_num) AS total_value
FROM base
WHERE {where_sql}
GROUP BY year, dataelement_name
ORDER BY {order_by}
""".strip()



def build_chart_query(where_sql: str) -> str:
    return f"""
WITH base AS (
    SELECT
        de.name AS dataelement_name,
        ou.name AS orgunit_name,
        p.startdate,
        CASE
            WHEN REGEXP('^[-]?\\d+(\\.\\d+)?$', TRIM(CAST(dv.value AS TEXT)))
            THEN CAST(dv.value AS REAL)
        END AS value_num
    FROM datavalue dv
    JOIN dataelement de ON dv.dataelementid = de.dataelementid
    JOIN organisationunit ou ON dv.sourceid = ou.organisationunitid
    JOIN period p ON dv.periodid = p.periodid
)
SELECT
    date(startdate, 'start of month') AS period,
    orgunit_name,
    dataelement_name,
    SUM(value_num) AS total_value
FROM base
WHERE {where_sql}
GROUP BY 1, 2, 3
ORDER BY 1
""".strip()



def run_query(db_path: str, sql: str, params: List[Any]) -> pd.DataFrame:
    fixed_params = [adapt_param_for_sqlite(p) for p in params]
    with get_sqlite_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=fixed_params)



def validate_llm_where_sql(where_sql: str, params: List[Any]) -> Tuple[bool, str]:
    if re.search(r";|--|/\*|\*/", where_sql):
        return False, "Disallowed SQL tokens in where_sql."
    if re.search(r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke)\b", where_sql, re.IGNORECASE):
        return False, "Unsafe SQL operation in where_sql."
    if where_sql.count("?") != len(params):
        return False, "Number of placeholders does not match params."

    scrubbed = re.sub(r"'[^']*'", "''", where_sql)
    scrubbed = scrubbed.replace("?", "")
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", scrubbed)
    allowed = {c.lower() for c in BASE_COLUMNS} | ALLOWED_SQL_KEYWORDS
    for tok in tokens:
        if tok.lower() not in allowed:
            return False, f"Unexpected identifier in where_sql: {tok}"
    return True, ""



def normalize_order_by(order_by: str, default: str, select_sql: str = "") -> str:
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(ASC|DESC)?\s*$", order_by, re.IGNORECASE)
    if not m:
        return default

    col = m.group(1).lower()
    direction = (m.group(2) or "DESC").upper()

    base_cols = {c.lower() for c in BASE_COLUMNS}
    aliases = set(re.findall(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)", select_sql, re.IGNORECASE))
    aliases = {a.lower() for a in aliases}

    if col not in base_cols and col not in aliases:
        return default

    return f"{col} {direction}"


# ---------------------------------------------------------------------
# Output Sanitizer (User-friendly table)
# ---------------------------------------------------------------------


def make_user_friendly_table(df: pd.DataFrame) -> pd.DataFrame:
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

    df = df.drop(columns=[c for c in technical_cols if c in df.columns], errors="ignore")
    raw_signature = {"orgunit_name", "dataelement_name", "startdate"}

    if raw_signature.issubset(set(df.columns)):
        clean = df.copy()
        if "value_num" in clean.columns and clean["value_num"].notna().any():
            clean["Value"] = clean["value_num"]
        elif "value" in clean.columns:
            clean["Value"] = clean["value"]

        clean = clean.drop(columns=[c for c in ["value_num", "value"] if c in clean.columns], errors="ignore")
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

    clean = df.copy()
    rename_map = {
        "orgunit_name": "Organisation Unit",
        "dataelement_name": "Metric",
        "total_value": "Total",
        "record_count": "Records",
    }
    return clean.rename(columns=rename_map)



def mentions_value(question: str) -> bool:
    return bool(re.search(r"\b(value|values|cases?|counts?|deaths?|tests?|visits?|incidence|coverage)\b", question, re.IGNORECASE))



def build_conn_str_from_parts() -> str:
    return os.environ.get("SQLITE_DB_PATH", "dhis2.sqlite").strip() or "dhis2.sqlite"



def make_json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        x = float(obj)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    return obj



def detect_query_shape(chart_df):
    return {
        "has_time": chart_df["period"].nunique() > 1 if "period" in chart_df else False,
        "multi_org": chart_df["orgunit_name"].nunique(),
        "multi_metric": chart_df["dataelement_name"].nunique(),
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
        "data": [{"name": r["dataelement_name"], "total": float(r["total_value"])} for _, r in metric_totals.iterrows()],
    }



def org_filter_present(where_sql):
    return "orgunit_name" in where_sql



def build_trend(chart_df):
    trend = chart_df.groupby("period")["total_value"].sum().reset_index().sort_values("period")
    return {
        "type": "line_trend",
        "title": "Trend Over Time",
        "data": [{"date": str(pd.to_datetime(r["period"]).date()), "total": float(r["total_value"])} for _, r in trend.iterrows()],
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
    grouped = filtered.groupby(["period", "dataelement_name"])["total_value"].sum().reset_index().sort_values("period")

    return {
        "type": "multi_line_metric_trend",
        "title": "Metric Trends Over Time",
        "series": [
            {
                "metric": metric,
                "data": [
                    {"date": str(pd.to_datetime(r["period"]).date()), "total": float(r["total_value"])}
                    for _, r in grouped[grouped["dataelement_name"] == metric].iterrows()
                ],
            }
            for metric in top_metrics
        ],
    }



def build_org_breakdown(chart_df, top_n=8):
    totals = chart_df.groupby("orgunit_name")["total_value"].sum().sort_values(ascending=False)
    top = totals.head(top_n)
    other_sum = totals.iloc[top_n:].sum()
    data = [{"name": org, "total": float(val)} for org, val in top.items()]
    if other_sum > 0:
        data.append({"name": "Other", "total": float(other_sum)})
    return {"type": "bar_orgs", "title": "Top Organisations", "data": data}



def build_org_trend(chart_df, top_n=5):
    chart_df = chart_df.copy()
    chart_df["period"] = pd.to_datetime(chart_df["period"], errors="coerce")
    chart_df["total_value"] = pd.to_numeric(chart_df["total_value"], errors="coerce")
    chart_df["orgunit_name"] = chart_df["orgunit_name"].astype(str).str.strip()
    chart_df = chart_df.dropna(subset=["period", "orgunit_name", "total_value"])

    monthly_org = chart_df.groupby(["period", "orgunit_name"])["total_value"].sum().reset_index()
    top_orgs = (
        monthly_org.groupby("orgunit_name")["total_value"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    if not top_orgs:
        return None

    series = []
    for org in top_orgs:
        org_data = monthly_org[monthly_org["orgunit_name"] == org]
        points = [{"date": str(r["period"].date()), "total": float(r["total_value"])} for _, r in org_data.iterrows()]
        if points:
            series.append({"metric": org, "data": points})

    return {"type": "multi_line_metric_trend", "title": "Top Organisation Trends", "series": series}



def build_summary_insights(where_sql, db_path, params):
    chart_sql = build_chart_query(where_sql)
    chart_df = run_query(db_path, chart_sql, params)
    if chart_df.empty:
        return {"mode": "none", "data": []}

    insights = {
        "mode": "dashboard",
        "cards": {
            "total": float(chart_df["total_value"].sum()),
            "metrics": int(chart_df["dataelement_name"].nunique()),
            "orgs": int(chart_df["orgunit_name"].nunique()),
        },
        "charts": [],
    }

    has_time = chart_df["period"].nunique() > 1
    multi_metric = chart_df["dataelement_name"].nunique() > 1
    multi_org = chart_df["orgunit_name"].nunique() > 1

    if has_time and multi_org and multi_metric:
        chart = build_org_trend(chart_df)
        if chart:
            insights["charts"].append(chart)
    elif has_time and multi_metric:
        insights["charts"].append(build_metric_trend(chart_df))
    elif multi_org and org_filter_present(where_sql):
        insights["charts"].append(build_org_breakdown(chart_df))
    elif multi_metric:
        insights["charts"].append(build_metric_breakdown(chart_df))
    elif has_time:
        insights["charts"].append(build_trend(chart_df))

    return insights



def strip_date_filters(where_sql: str, params):
    if not params:
        params = []

    pieces = re.split(r"\s+AND\s+", where_sql, flags=re.IGNORECASE)
    new_sql = []
    new_params = []

    idx = 0
    for p in pieces:
        ph = p.count("?")
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
    payload = {"ranking_summary": {}, "top_result": None}
    if df.empty:
        return payload

    payload["ranking_summary"] = {"rows": len(df), "max_value": float(df["total_value"].max())}
    payload["top_result"] = df.iloc[0].to_dict()
    if yearly_df is not None and not yearly_df.empty:
        ydf = yearly_df.copy()
        ydf["total_value"] = pd.to_numeric(ydf["total_value"], errors="coerce")
        ydf = ydf.sort_values("year")
        ydf["pct_change"] = ydf["total_value"].pct_change()
        payload["yearly_analysis"] = {
            "years_present": ydf["year"].tolist(),
            "yearly_totals": ydf["total_value"].tolist(),
            "year_over_year_change": ydf["pct_change"].tolist(),
        }
    return payload



def build_analysis_payload(monthly_df: pd.DataFrame, yearly_df: pd.DataFrame | None = None) -> dict:
    payload = {
        "data_summary": {},
        "monthly_analysis": {},
        "yearly_analysis": {},
        "anomaly_detection": {},
    }

    df = monthly_df.copy()
    df["startdate"] = pd.to_datetime(df["startdate"])
    df = df.sort_values("startdate")
    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")

    payload["data_summary"] = {
        "total_months": int(len(df)),
        "overall_sum": float(df["total_value"].sum()),
        "overall_mean": float(df["total_value"].mean()),
        "max_value": float(df["total_value"].max()),
        "min_value": float(df["total_value"].min()),
    }

    df["pct_change"] = df["total_value"].pct_change()
    payload["monthly_analysis"] = {
        "average_monthly_change": float(df["pct_change"].mean(skipna=True)),
        "max_month": df.loc[df["total_value"].idxmax(), "startdate"].strftime("%Y-%m"),
        "max_month_value": float(df["total_value"].max()),
    }

    df["rolling_mean"] = df["total_value"].rolling(3).mean()
    df["rolling_std"] = df["total_value"].rolling(3).std()
    df["z_score"] = (df["total_value"] - df["rolling_mean"]) / df["rolling_std"]
    spikes = df[df["z_score"] > 2]
    payload["anomaly_detection"] = {
        "spike_count": int(len(spikes)),
        "spike_months": spikes["startdate"].dt.strftime("%Y-%m").tolist(),
    }

    if yearly_df is not None and len(yearly_df) > 1:
        ydf = yearly_df.copy()
        ydf["total_value"] = pd.to_numeric(ydf["total_value"], errors="coerce")
        ydf = ydf.sort_values("year")
        ydf["pct_change"] = ydf["total_value"].pct_change()
        payload["yearly_analysis"] = {
            "years_present": ydf["year"].tolist(),
            "yearly_totals": ydf["total_value"].tolist(),
            "year_over_year_change": ydf["pct_change"].tolist(),
        }

    return payload



def build_indicator_analysis_payload(df, yearly_df):
    summary = (
        df.groupby("dataelement_name")["total_value"]
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
            "name": dominant["dataelement_name"],
            "value": float(dominant["total_value"]),
        } if dominant is not None else None,
    }
    return payload



def build_summary_analysis_payload(df, yearly_df=None):
    required_cols = {"dataelement_name", "total_value"}
    if not required_cols.issubset(df.columns):
        return {"error": "Invalid summary dataframe structure"}

    df = df.copy()
    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
    df = df.dropna(subset=["total_value"])
    if df.empty:
        return {"error": "No valid numeric values found"}

    df = df.sort_values("total_value", ascending=False).reset_index(drop=True)
    total_sum = df["total_value"].sum()
    mean_value = df["total_value"].mean()
    max_row = df.iloc[0]
    min_row = df.iloc[-1]
    df["percentage_contribution"] = (df["total_value"] / total_sum * 100).round(2)

    payload = {
        "view_type": "summary",
        "total_indicators": int(len(df)),
        "grand_total": float(total_sum),
        "average_value": float(mean_value),
        "top_performer": {
            "dataelement_name": max_row["dataelement_name"],
            "value": float(max_row["total_value"]),
            "contribution_percent": float(df.iloc[0]["percentage_contribution"]),
        },
        "lowest_performer": {
            "dataelement_name": min_row["dataelement_name"],
            "value": float(min_row["total_value"]),
            "contribution_percent": float(df.iloc[-1]["percentage_contribution"]),
        },
        "ranking": df.to_dict(orient="records"),
    }
    return payload



def build_universal_analysis_payload(df, yearly_df):
    df_cols = set(df.columns)
    yearly_cols = set(yearly_df.columns) if yearly_df is not None else set()

    if "startdate" in df_cols:
        return build_analysis_payload(df, yearly_df)
    if "orgunit_name" in df_cols:
        return build_ranking_analysis_payload(df, yearly_df)
    if {"dataelement_name", "total_value"} <= df_cols:
        return build_summary_analysis_payload(df, yearly_df)
    if "year" in yearly_cols:
        return build_indicator_analysis_payload(df, yearly_df)
    return {"error": "Unsupported dataframe structure"}



def explain_query(insights: dict, question: str, hf_token: str) -> str:
    final_payload = {"insights": insights, "question": question}
    explanation_prompt = """
You are an epidemiological analytics explanation engine.

You will receive:
1. A structured JSON object named "insights"
2. The original user question

Your task is to generate a clear, evidence-based explanation of the observed patterns.

STRICT CONSTRAINTS:
- Use ONLY the provided data.
- Do NOT introduce external knowledge.
- Do NOT fabricate causes.
- Do NOT claim causation.
- Use neutral analytical language (e.g., "may indicate", "suggests", "is associated with").
- Do NOT reference SQL, models, prompts, system design, or technical processes.
- Do NOT mention missing data, null values, NaN values, or unavailable fields.
- Describe only patterns supported by the data.
- Never return JSON.
- Always produce a narrative response.
- Minimum 3 sentences.
""".strip()
    return run_instruct_llm_prompt(explanation_prompt, hf_token, payload=final_payload, payload_include=True)


@lru_cache(maxsize=300)
def get_cached_llm_plan(use_model: str, prompt: str, token: str, payload: dict | None = None, payload_include: bool = False) -> dict:
    if use_model == "SQL":
        decoded = run_sql_llm_prompt(prompt, token).strip()
    else:
        decoded = run_instruct_llm_prompt(prompt, token).strip()

    instr = {}
    m = re.search(r"\{[\s\S]*\}", decoded)
    if m:
        try:
            instr = json.loads(m.group())
        except Exception:
            instr = {}

    return normalize_llm_sqlite_plan(instr)



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

    clauses = ["name LIKE ? COLLATE NOCASE" for _ in cleaned]
    sql = f"""
        SELECT DISTINCT name AS dataelement_name
        FROM dataelement
        WHERE {' OR '.join(clauses)}
        ORDER BY name
    """
    params = [f"%{m}%" for m in cleaned]
    return sql.strip(), params



def auto_group_where(where_sql: str) -> str:
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
    for _, group in col_groups.items():
        if len(group) > 1:
            rebuilt.append("(" + " OR ".join(group) + ")")
        else:
            rebuilt.append(group[0])

    return " AND ".join(rebuilt)


# ---------------------------------------------------------------------
# Main answering functions
# ---------------------------------------------------------------------


def normal_query_answer(
    question: str,
    row_limit: int,
    db_path: str,
    hf_token: str,
    columns_list: str,
    today_str: str,
    debug: bool = False,
    page: int = 1,
    page_size: int = 100,
    include_insights: bool = False,
    include_rows: bool = True,
) -> dict:
    intent_prompt = f"""
You are a strict information extraction engine for DHIS2 health facility data.
Return ONLY valid JSON.
Required output format:
{{
  "orgunit": ["string"] | null,
  "metric": ["string"] | null
}}
Extract only explicit organisation unit names and explicit health metrics from the question.
Question:
{question}
JSON:
""".strip()

    instr_intent = get_cached_llm_plan("Instruct", intent_prompt, hf_token)
    intent_json: Dict[str, Any] = instr_intent if isinstance(instr_intent, dict) else {}
    intent_json = normalize_metrics(intent_json)

    distinct_metric_df = []
    if intent_json.get("metric"):
        intent_json["metric"] = [m.strip() for m in intent_json["metric"] if m and m.strip()]
        distinct_metric_sql, distinct_metric_params = build_distinct_dataelement_query(intent_json.get("metric"))
        if distinct_metric_sql:
            distinct_metric_df = run_query(db_path, distinct_metric_sql, distinct_metric_params)

    prompt = f"""
You are a STRICT Text-to-SQL planner for a SQLite database.
Return ONLY a valid JSON object.
You are querying from a CTE named `base` with ONLY these columns:
{columns_list}

Available valid dataelement_name values:
{distinct_metric_df}

Return exactly these keys:
- select_sql
- where_sql
- params
- group_by
- order_by
- limit

Use SQLite placeholders `?` only.
Use LIKE for text matching.
Use startdate >= ? AND startdate < ? for date windows.
For multiple filters on the same column, use OR inside parentheses.
For aggregation use SUM(value_num) AS total_value.
If unsure, return:
{{"select_sql":"*","where_sql":"FALSE","params":[],"group_by":"","order_by":"startdate DESC","limit":1}}

Today: {today_str}
User question: {question}
""".strip()

    instr = get_cached_llm_plan("SQL", prompt, hf_token)
    use_llm_sql = False
    select_sql = "*"
    where_sql = "TRUE"
    params: List[Any] = []
    order_by = "startdate DESC"
    group_by = ""
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

    ok, _ = validate_llm_where_sql(where_sql, params)
    if ok and where_sql != "TRUE":
        use_llm_sql = True
        if " OR " in where_sql and " AND " in where_sql:
            where_sql = auto_group_where(where_sql)

        if re.search(r"(?:last|previous|past|recent|trailing)\s+\d+\s+months?\s+of\s+20\d{2}", question.lower()):
            date_col, start_dt, end_dt, _ = extract_date_range(question)
            date_params = []
            if start_dt is not None:
                date_params.append(start_dt.date().isoformat())
            if end_dt is not None:
                date_params.append(end_dt.date().isoformat())
            old_date_count = len(date_params)
            params = params[:-old_date_count] if old_date_count else params
            params.extend(date_params)

    if not use_llm_sql:
        raw_filters = instr.get("filters", {}) if isinstance(instr, dict) else {}
        filters = normalize_filters(raw_filters)

        if "followup" not in filters and re.search(r"follow[- ]?up|flagged", question, re.IGNORECASE):
            q = question.lower()
            if re.search(r"(not|no|without|false|pending|missing|unresolved).*follow[- ]?up", q) or re.search(r"follow[- ]?up.*(not|no|without|false|pending|missing|unresolved)", q):
                filters["followup"] = False
            elif re.search(r"(with|has|completed|maintained|true|done).*follow[- ]?up", q) or re.search(r"follow[- ]?up.*(with|has|completed|maintained|true|done)", q):
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

        if not where_sql:
            return None

    summary_mode = is_summary_question(question)

    if summary_mode:
        if include_insights and not include_rows:
            insights = build_summary_insights(where_sql, db_path, params)
            return make_json_safe({
                "view": "summary",
                "columns": [],
                "rows": [],
                "row_count": 1,
                "insights": insights,
                "insights_available": True,
            })

        sql = build_summary_query(where_sql)
        df = run_query(db_path, sql, params)
        clean_df = make_user_friendly_table(df)
        return make_json_safe({
            "view": "summary",
            "columns": list(clean_df.columns),
            "rows": clean_df.values.tolist(),
            "row_count": 1,
            "insights": None,
            "insights_available": True,
        })

    if include_rows:
        sql = build_query(select_sql=select_sql, where_sql=where_sql, order_by=order_by, group_by=group_by)
        df = run_query(db_path, sql, params + [query_limit, query_offset])
        if not df.empty and "total_count" in df.columns:
            total_rows = int(df["total_count"].iloc[0])
        elif not df.empty:
            total_rows = len(df)
        else:
            total_rows = 0

        df_clean_input = df.drop(columns=["total_count"], errors="ignore")
        clean_df = make_user_friendly_table(df_clean_input)
        if clean_df is None:
            clean_df = pd.DataFrame()
        columns = list(clean_df.columns) if not clean_df.empty else []
        return make_json_safe({
            "view": "records",
            "columns": columns,
            "rows": clean_df.fillna("").values.tolist() if not clean_df.empty else [],
            "row_count": total_rows if total_rows is not None else 0,
            "insights": None,
            "insights_available": True,
        })

    if include_insights and not include_rows:
        insights = build_summary_insights(where_sql, db_path, params)
        return make_json_safe({
            "view": "records",
            "columns": [],
            "rows": [],
            "row_count": 0,
            "insights": insights,
            "insights_available": True,
        })

    return make_json_safe({
        "view": "records",
        "columns": [],
        "rows": [],
        "row_count": 0,
        "insights": None,
        "insights_available": False,
    })



def explainable_query_answer(question: str, row_limit: int, db_path: str, hf_token: str, columns_list: str, today_str: str, debug: bool = False) -> dict:
    intent_prompt = f"""
You are a strict information extraction engine for DHIS2 health facility data.
Return ONLY valid JSON.
Required output format:
{{
  "orgunit": ["string"] | null,
  "metric": ["string"] | null
}}
Question:
{question}
JSON:
""".strip()

    instr_intent = get_cached_llm_plan("Instruct", intent_prompt, hf_token)
    intent_json: Dict[str, Any] = instr_intent if isinstance(instr_intent, dict) else {}
    intent_json = normalize_metrics(intent_json)

    distinct_metric_df = []
    if intent_json.get("metric"):
        intent_json["metric"] = [m.strip() for m in intent_json["metric"] if m and m.strip()]
        distinct_metric_sql, distinct_metric_params = build_distinct_dataelement_query(intent_json.get("metric"))
        if distinct_metric_sql:
            distinct_metric_df = run_query(db_path, distinct_metric_sql, distinct_metric_params)

    prompt = f"""
You are a STRICT Text-to-SQL planner for a SQLite database.
Return ONLY valid JSON.
You query ONLY from a CTE named `base` with ONLY these columns:
{columns_list}
You are provided valid metric names from the database:
{distinct_metric_df}

Use SQLite placeholders `?` only.
Use LIKE for text matching.
Use startdate >= ? AND startdate < ? for date filtering.
For ranking questions use SUM(value_num) AS total_value grouped by orgunit_name.
For explainable trend questions use startdate, SUM(value_num) AS total_value grouped by startdate.
If unsure, return:
{{"select_sql":"*","where_sql":"FALSE","params":[],"group_by":"","order_by":"startdate DESC","limit":1}}

Today: {today_str}
User question: {question}
""".strip()

    instr = get_cached_llm_plan("SQL", prompt, hf_token)

    use_llm_sql = False
    select_sql = "*"
    where_sql = "TRUE"
    params: List[Any] = []
    order_by = "startdate DESC"
    group_by = ""
    query_limit = row_limit

    if isinstance(instr, dict) and any(k in instr for k in ("where_sql", "params", "order_by", "limit", "select_sql", "group_by")):
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

    ok, _ = validate_llm_where_sql(where_sql, params)
    if ok and where_sql != "TRUE":
        use_llm_sql = True
        if " OR " in where_sql and " AND " in where_sql:
            where_sql = auto_group_where(where_sql)
        if re.search(r"(?:last|previous|past|recent|trailing)\s+\d+\s+months?\s+of\s+20\d{2}", question.lower()):
            date_col, start_dt, end_dt, _ = extract_date_range(question)
            date_params = []
            if start_dt is not None:
                date_params.append(start_dt.date().isoformat())
            if end_dt is not None:
                date_params.append(end_dt.date().isoformat())
            old_date_count = len(date_params)
            params = params[:-old_date_count] if old_date_count else params
            params.extend(date_params)

    if not use_llm_sql:
        raw_filters = instr.get("filters", {}) if isinstance(instr, dict) else {}
        filters = normalize_filters(raw_filters)

        if "followup" not in filters and re.search(r"follow[- ]?up|flagged", question, re.IGNORECASE):
            q = question.lower()
            if re.search(r"(not|no|without|false|pending|missing|unresolved).*follow[- ]?up", q) or re.search(r"follow[- ]?up.*(not|no|without|false|pending|missing|unresolved)", q):
                filters["followup"] = False
            elif re.search(r"(with|has|completed|maintained|true|done).*follow[- ]?up", q) or re.search(r"follow[- ]?up.*(with|has|completed|maintained|true|done)", q):
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

        if not where_sql:
            return None

    sql = build_explainable_query(select_sql=select_sql, where_sql=where_sql, order_by=order_by, group_by=group_by)
    df = run_query(db_path, sql, params)

    yearly_sql = build_yearly_query(dataelement_include="dataelement_name" in df.columns, where_sql=where_sql)
    yearly_df = run_query(db_path, yearly_sql, params)
    if not yearly_df.empty and "year" in yearly_df.columns:
        yearly_df["year"] = yearly_df["year"].astype(int)

    insights = build_universal_analysis_payload(df, yearly_df)
    return explain_query(insights, question, hf_token)



def answer_question(question: str, debug: bool = False, page: int = 1, page_size: int = 100, include_insights: bool = False, include_rows: bool = True) -> dict:
    row_limit = int(os.environ.get("ROW_LIMIT", "400"))
    db_path = build_conn_str_from_parts()
    if not os.path.exists(db_path):
        raise ValueError(f"SQLite database not found: {db_path}")

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
Return ONLY valid JSON.
Output format:
{{"classification":"normal" or "explainable","reason":"short reason"}}
Question:
{question}
JSON:
""".strip()

    classification_instr = get_cached_llm_plan("Instruct", classification_prompt, hf_token)
    classification = classification_instr.get("classification") if isinstance(classification_instr, dict) else None

    if classification == "explainable":
        answer = explainable_query_answer(question, row_limit, db_path, hf_token, columns_list, today_str, debug)
        return {"view": "explainable", "answer": answer, "insights_available": False}

    return normal_query_answer(
        question,
        row_limit,
        db_path,
        hf_token,
        columns_list,
        today_str,
        debug,
        page,
        page_size,
        include_insights,
        include_rows,
    )



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

    if "columns" in result and "rows" in result:
        df = pd.DataFrame(result["rows"], columns=result["columns"])
        print("\n================ USER FRIENDLY TABLE (first 20) ================")
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()