import json
import os
import re
from calendar import monthrange
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import psycopg2
import streamlit as st
from dateutil import parser as dateutil_parser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------
# Constants & schema mapping
# ---------------------------------------------------------------------

VALUE_KEYWORDS = (
    "value", "values", "case", "cases", "death", "deaths", "count", "counts",
    "test", "tests", "visits", "incidence", "coverage", "total", "sum"
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

# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------


@st.cache_resource
def load_model() -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    model_id = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, use_safetensors=True)
    return tokenizer, model

# ---------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------


def looks_like_date_token(tok: str) -> bool:
    if not tok:
        return False
    if re.match(r"^20\d{2}([\/\-](0[1-9]|1[0-2])([\/\-](0[1-9]|[12]\d|3[01]))?)?$", tok):
        return True
    if re.match(r"^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}$", tok):
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


def extract_date_range(
    question: str,
) -> Tuple[str, pd.Timestamp | None, pd.Timestamp | None, str]:
    date_col = choose_date_column(question)
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

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
# Intent detection (NEW)
# ---------------------------------------------------------------------


def detect_intent(question: str) -> Dict[str, Any]:
    q = question.lower()

    # defaults
    intent = {
        "mode": "rows",  # rows | aggregate | trend | top
        "agg": None,     # sum | count
        "group_by": None,
        "top_n": None,
    }

    # Trend intent
    if re.search(r"\b(trend|over time|monthly|by month|per month|weekly|by week|yearly|by year)\b", q):
        intent["mode"] = "trend"
        intent["agg"] = "sum" if re.search(r"\b(case|cases|death|deaths|tests|value|total|sum)\b", q) else "count"
        intent["group_by"] = "month" if "month" in q or "monthly" in q else "year"
        return intent

    # Top N intent
    m_top = re.search(r"\btop\s+(\d+)\b", q)
    if m_top:
        intent["mode"] = "top"
        intent["top_n"] = int(m_top.group(1))
        intent["agg"] = "sum" if re.search(r"\b(case|cases|death|deaths|tests|value|total|sum)\b", q) else "count"
        intent["group_by"] = "orgunit_name"
        return intent

    # Aggregate intent
    if re.search(r"\b(how many|count|number of)\b", q):
        intent["mode"] = "aggregate"
        intent["agg"] = "count"
        return intent

    if re.search(r"\b(total|sum)\b", q):
        intent["mode"] = "aggregate"
        intent["agg"] = "sum"
        return intent

    return intent

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
        if (
            len(value) == 3
            and isinstance(value[0], str)
            and value[0].lower() == "range"
        ):
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
# Query builder & execution (UPDATED)
# ---------------------------------------------------------------------


def build_base_cte() -> str:
    # Uses your DHIS2 schema (improved base)
    return r"""
WITH base AS (
    SELECT
        dv.dataelementid,
        de.name AS dataelement_name,
        de.code AS dataelement_code,
        de.uid AS dataelement_uid,

        dv.sourceid AS organisationunitid,
        ou.name AS orgunit_name,
        ou.shortname AS orgunit_shortname,
        ou.code AS orgunit_code,
        ou.uid AS orgunit_uid,
        ou.hierarchylevel,
        ou.parentid AS orgunit_parentid,
        ou.path AS orgunit_path,

        dv.periodid,
        p.startdate,
        p.enddate,
        pt.name AS period_type,

        dv.value,
        CASE
            WHEN dv.value ~ '^\s*[-]?\d+(\.\d+)?\s*$' THEN dv.value::numeric
            ELSE NULL
        END AS value_num,

        dv.comment,
        dv.storedby,
        dv.lastupdated,
        dv.created,
        dv.followup,

        dv.categoryoptioncomboid,
        dv.attributeoptioncomboid

    FROM datavalue dv
    JOIN dataelement de ON dv.dataelementid = de.dataelementid
    JOIN organisationunit ou ON dv.sourceid = ou.organisationunitid
    JOIN period p ON dv.periodid = p.periodid
    LEFT JOIN periodtype pt ON p.periodtypeid = pt.periodtypeid
)
"""


def build_query(
    where_sql: str,
    order_by: str,
    limit: int,
    intent: Dict[str, Any],
) -> str:
    base = build_base_cte()

    mode = intent.get("mode", "rows")
    agg = intent.get("agg")
    group_by = intent.get("group_by")
    top_n = intent.get("top_n")

    if mode == "aggregate":
        if agg == "sum":
            select_sql = "SELECT COALESCE(SUM(value_num), 0) AS answer"
        else:
            select_sql = "SELECT COUNT(*) AS answer"
        return f"""
{base}
{select_sql}
FROM base
WHERE {where_sql};
"""

    if mode == "trend":
        if group_by == "month":
            time_expr = "DATE_TRUNC('month', startdate)"
        else:
            time_expr = "DATE_TRUNC('year', startdate)"

        metric_expr = "COALESCE(SUM(value_num), 0)" if agg == "sum" else "COUNT(*)"
        return f"""
{base}
SELECT
    {time_expr} AS period,
    {metric_expr} AS value
FROM base
WHERE {where_sql}
GROUP BY 1
ORDER BY 1 ASC
LIMIT %s;
"""

    if mode == "top":
        metric_expr = "COALESCE(SUM(value_num), 0)" if agg == "sum" else "COUNT(*)"
        n = top_n or 10
        return f"""
{base}
SELECT
    orgunit_name,
    {metric_expr} AS value
FROM base
WHERE {where_sql}
GROUP BY orgunit_name
ORDER BY value DESC
LIMIT %s;
"""

    # default rows mode (original)
    return f"""
{base}
SELECT *
FROM base
WHERE {where_sql}
ORDER BY {order_by}
LIMIT %s;
"""


def run_query(conn_str: str, sql: str, params: List[Any]) -> pd.DataFrame:
    with psycopg2.connect(conn_str) as conn:
        df = pd.read_sql(sql, conn, params=params)
    return df

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def mentions_value(question: str) -> bool:
    return bool(
        re.search(
            r"\b(value|values|cases?|counts?|deaths?|tests?|visits?|incidence|coverage|total|sum)\b",
            question,
            re.IGNORECASE,
        )
    )


def default_conn_uri() -> str:
    return os.environ.get("DHIS2_DSN", "")


def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------


st.set_page_config(layout="wide")
st.title("🩺 DHIS2 Metrics Explorer")

st.markdown(
    """Use natural language to describe the DHIS2 facts you need. The assistant
extracts filters and converts them into a SQL query that runs directly against
your PostgreSQL DHIS2 database."""
)

st.markdown("**Example questions:**")
st.markdown("- `Malaria cases for Bo District between Jan and Mar 2023`")
st.markdown("- `Total malaria cases in 2022`")
st.markdown("- `Top 10 districts by malaria cases in 2023`")
st.markdown("- `Monthly trend of malaria cases in 2024`")
st.markdown("- `Facilities at level 4 with reported deaths >= 10 in 2022`")
st.markdown("- `Show measles vaccine coverage in Kenema updated after 2021-06-01`")
st.markdown("- `Flagged follow-up data for hospitals with > 500 tests`")

# --- Sidebar: connection settings ---

st.sidebar.header("PostgreSQL connection")

dsn_default = get_secret("DHIS2_DSN", default_conn_uri())
uri_input = st.sidebar.text_input(
    "Connection URI",
    value=dsn_default,
    help="Format: postgresql://user:password@host:port/database (takes precedence over fields below).",
)
st.sidebar.caption("Leave blank to build a DSN from the fields underneath.")

host = st.sidebar.text_input(
    "Host", value=get_secret("PGHOST", os.environ.get("PGHOST", ""))
)
port = st.sidebar.text_input(
    "Port", value=get_secret("PGPORT", os.environ.get("PGPORT", ""))
)
database = st.sidebar.text_input(
    "Database", value=get_secret("PGDATABASE", os.environ.get("PGDATABASE", ""))
)
user = st.sidebar.text_input(
    "User", value=get_secret("PGUSER", os.environ.get("PGUSER", ""))
)
password = st.sidebar.text_input(
    "Password",
    type="password",
    value=get_secret("PGPASSWORD", os.environ.get("PGPASSWORD", "")),
)
row_limit = st.sidebar.slider(
    "Max rows", min_value=50, max_value=500, value=200, step=50
)


def build_conn_str(uri: str) -> str:
    if uri:
        return uri.strip()
    if not host or not database or not user:
        return ""
    dsn = f"host={host} port={port or 5432} dbname={database} user={user}"
    if password:
        dsn += f" password={password}"
    return dsn


question = st.text_input("Describe the DHIS2 data slice you need:")
run = st.button("Run")

if run and question:
    conn_str = build_conn_str(uri_input)
    if not conn_str:
        st.error(
            "Provide either a PostgreSQL URI or complete host credentials in the sidebar."
        )
        st.stop()

    # NEW: intent detection
    intent = detect_intent(question)

    tokenizer, model = load_model()

    prompt = f"""
You extract DHIS2 filters from a user question.
Return ONLY valid JSON. No markdown. No explanation.

Allowed keys inside "filters":
orgunit_name, orgunit_code, orgunit_uid, hierarchylevel,
dataelement_name, dataelement_code, dataelement_uid,
period_type, storedby, followup, value_num.

Rules:
- If not mentioned, return null.
- value_num must be either null, ["range", min, max], or [">=", number], ["<=", number], [">", number], ["<", number], ["=", number].
- Do not guess facility names.

Example:
Question: Facilities at level 4 with deaths >= 10
Output:
{{"filters": {{"hierarchylevel": 4, "dataelement_name": "deaths", "value_num": [">=", 10]}}}}

Question: {question}
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=160, do_sample=False, num_beams=1)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    st.subheader("🔍 Raw Model Output")
    st.code(decoded)

    instr: Dict[str, Any] = {}
    m = re.search(r"\{[\s\S]*\}", decoded)
    if m:
        try:
            instr = json.loads(m.group())
        except Exception:
            instr = {}

    raw_filters = instr.get("filters", {}) if isinstance(instr, dict) else {}
    filters = normalize_filters(raw_filters)

    # Heuristics
    if "followup" not in filters and re.search(
        r"follow[- ]?up|flagged", question, re.IGNORECASE
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

    intent_preview = {
        "intent": intent,
        "filters": serialize_filters_for_display(filters),
        "date_range": {
            "column": date_col,
            "start": start_dt.isoformat() if start_dt is not None else None,
            "end": end_dt.isoformat() if end_dt is not None else None,
            "sort_direction": sort_dir,
        },
    }

    st.subheader("✅ Parsed Intent + Filters + Date Range")
    st.json(intent_preview)

    where_clauses, params = build_where(filters, start_dt, end_dt, date_col)
    where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"
    order_by = f"{date_col} {sort_dir}"

    sql = build_query(where_sql, order_by, row_limit, intent)

    st.subheader("💾 Generated SQL")
    st.code(sql.strip())

    # Params handling: aggregate queries don't need limit, trend/top do, rows do
    final_params = list(params)
    if intent["mode"] in {"rows", "trend", "top"}:
        final_params.append(row_limit)

    st.caption("Query parameters:")
    st.write(final_params)

    try:
        df = run_query(conn_str, sql, final_params)
    except Exception as exc:
        st.error(f"Database error: {exc}")
    else:
        # Friendly output for non-technical users
        if intent["mode"] == "aggregate" and "answer" in df.columns:
            st.subheader("📌 Answer")
            st.metric("Result", f"{df.iloc[0]['answer']}")
            st.subheader("📋 Data")
            st.dataframe(df, hide_index=True)
        else:
            st.subheader("📋 Results")
            st.dataframe(df, hide_index=True)
