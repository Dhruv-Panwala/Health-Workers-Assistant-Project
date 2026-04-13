import json
import sqlite3
import time
import re
from difflib import get_close_matches
from functools import lru_cache
from pocketflow import Node
from utils.analytics import build_analytics_payload, build_analytics_context
from utils.call_llm import call_llm, is_model_available, model_unavailable_reason


TABLE_DESCRIPTION_HINTS = {
    "dataelement": "Stores DHIS2 indicator and metric definitions.",
    "datavalue": "Stores reported values for each data element, reporting period, and organisation unit.",
    "organisationunit": "Stores facilities, districts, and the organisation unit hierarchy.",
    "period": "Stores reporting period start and end dates.",
    "periodtype": "Stores reporting frequencies such as monthly, weekly, quarterly, or yearly.",
}

TABLE_ALIAS_HINTS = {
    "dataelement": ["indicator", "indicators", "metric", "metrics", "measure", "measures", "data element", "test", "tests"],
    "datavalue": ["value", "values", "record", "records", "submission", "submissions", "reported value", "result", "results"],
    "organisationunit": ["organisation unit", "org unit", "orgunit", "facility", "facilities", "district", "districts", "location", "locations", "hierarchy", "level"],
    "period": ["period", "periods", "date", "dates", "month", "months", "year", "years", "quarter", "quarters", "reporting period", "time"],
    "periodtype": ["period type", "frequency", "frequencies", "granularity", "monthly", "weekly", "quarterly", "yearly"],
}

COLUMN_DESCRIPTION_HINTS = {
    ("dataelement", "dataelementid"): "Primary key for the data element.",
    ("dataelement", "name"): "Display name of the indicator or metric.",
    ("dataelement", "shortname"): "Short label used in forms or reports.",
    ("dataelement", "code"): "Optional code used by implementations.",
    ("dataelement", "description"): "Optional human-readable description of the metric.",
    ("dataelement", "valuetype"): "Type of stored value, such as NUMBER or TEXT.",
    ("dataelement", "aggregationtype"): "How the metric is aggregated in reports.",
    ("datavalue", "dataelementid"): "Foreign key to dataelement.dataelementid.",
    ("datavalue", "periodid"): "Foreign key to period.periodid.",
    ("datavalue", "sourceid"): "Foreign key to organisationunit.organisationunitid.",
    ("datavalue", "value"): "Reported value stored as TEXT; cast it when numeric aggregation is needed.",
    ("datavalue", "comment"): "Optional comment entered with the reported value.",
    ("datavalue", "followup"): "Follow-up flag on the submission.",
    ("organisationunit", "organisationunitid"): "Primary key for the organisation unit.",
    ("organisationunit", "name"): "Facility or organisation unit name.",
    ("organisationunit", "code"): "Optional implementation code for the organisation unit.",
    ("organisationunit", "path"): "Hierarchy path for the organisation unit.",
    ("organisationunit", "hierarchylevel"): "Depth of the organisation unit in the hierarchy.",
    ("organisationunit", "openingdate"): "Date the organisation unit opened.",
    ("period", "periodid"): "Primary key for the reporting period.",
    ("period", "periodtypeid"): "Foreign key to periodtype.periodtypeid.",
    ("period", "startdate"): "Start date of the reporting period.",
    ("period", "enddate"): "End date of the reporting period.",
    ("periodtype", "periodtypeid"): "Primary key for the period type.",
    ("periodtype", "name"): "Frequency name such as Monthly or Yearly.",
}


def _clean_sql(sql_text):
    return sql_text.strip().strip("`").rstrip(";")


def _extract_sql_from_yaml_like_text(text):
    sql_key_match = re.search(r"sql\s*:\s*\|?\s*\n?(.*)$", text, re.IGNORECASE | re.DOTALL)
    if not sql_key_match:
        return None

    sql_text = sql_key_match.group(1)
    sql_text = sql_text.split("```", 1)[0].rstrip()
    sql_lines = sql_text.splitlines()

    if not sql_lines:
        return None

    indent_levels = [
        len(line) - len(line.lstrip())
        for line in sql_lines
        if line.strip()
    ]
    if indent_levels:
        min_indent = min(indent_levels)
        sql_lines = [line[min_indent:] if len(line) >= min_indent else line for line in sql_lines]

    candidate_sql = "\n".join(sql_lines).strip()
    return _clean_sql(candidate_sql) if candidate_sql else None


def _extract_sql_from_inline_mapping(text):
    if not text:
        return None

    stripped = text.strip()
    try:
        structured_result = json.loads(stripped)
    except Exception:
        structured_result = None

    if isinstance(structured_result, dict) and structured_result.get("sql"):
        return _clean_sql(str(structured_result["sql"]))

    inline_match = re.search(
        r'["\']?sql["\']?\s*:\s*["\'](?P<sql>[\s\S]*?)["\']\s*(?:[,}]\s*)?$',
        stripped,
        re.IGNORECASE,
    )
    if inline_match:
        escaped_sql = inline_match.group("sql")
        try:
            return _clean_sql(bytes(escaped_sql, "utf-8").decode("unicode_escape"))
        except Exception:
            return _clean_sql(escaped_sql)

    return None


def _validate_sql(sql_text):
    normalized = sql_text.upper()

    if "..." in sql_text:
        raise ValueError("Generated SQL contains ellipsis placeholders.")
    if re.search(r"<[^>]+>", sql_text):
        raise ValueError("Generated SQL contains angle-bracket placeholders.")
    if any(token in normalized for token in ("TABLE_NAME", "COLUMN_NAME", "YOUR_SQL_HERE", "CORRECTED QUERY")):
        raise ValueError("Generated SQL contains placeholder tokens.")
    if "DISTINCT ON" in normalized:
        raise ValueError("Generated SQL uses PostgreSQL DISTINCT ON, which is not valid SQLite.")
    if re.search(r"\bILIKE\b", normalized):
        raise ValueError("Generated SQL uses ILIKE, which is not valid SQLite.")
    if normalized.startswith(("SELECT", "WITH")) and not re.search(r"\bFROM\b", normalized):
        raise ValueError("Generated SQL is missing a FROM clause.")


def _extract_sql_from_response(llm_response):
    fenced_yaml = re.search(r"```(?:yaml|yml)\s*(.*?)```", llm_response, re.IGNORECASE | re.DOTALL)
    unclosed_fenced_yaml = re.search(r"```(?:yaml|yml)\s*(.*)$", llm_response, re.IGNORECASE | re.DOTALL)
    yaml_candidates = []

    if fenced_yaml:
        yaml_candidates.append(fenced_yaml.group(1).strip())
    elif unclosed_fenced_yaml:
        yaml_candidates.append(unclosed_fenced_yaml.group(1).strip())
    yaml_candidates.append(llm_response.strip())

    for candidate in yaml_candidates:
        if not candidate:
            continue

        mapped_sql = _extract_sql_from_inline_mapping(candidate)
        if mapped_sql:
            return mapped_sql

        yaml_like_sql = _extract_sql_from_yaml_like_text(candidate)
        if yaml_like_sql:
            return yaml_like_sql

    yaml_like_sql = _extract_sql_from_yaml_like_text(llm_response)
    if yaml_like_sql:
        return yaml_like_sql

    fenced_sql = re.search(r"```sql\s*(.*?)```", llm_response, re.IGNORECASE | re.DOTALL)
    if fenced_sql:
        return _clean_sql(fenced_sql.group(1))

    unclosed_fenced_sql = re.search(r"```sql\s*(.*)$", llm_response, re.IGNORECASE | re.DOTALL)
    if unclosed_fenced_sql:
        return _clean_sql(unclosed_fenced_sql.group(1).split("```", 1)[0].strip())

    stripped = llm_response.strip()
    if stripped.upper().startswith(("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")):
        return _clean_sql(stripped)

    sql_match = re.search(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b[\s\S]*", llm_response, re.IGNORECASE)
    if sql_match:
        return _clean_sql(sql_match.group(0).split("```", 1)[0].strip())

    raise ValueError(f"Could not extract SQL from model response:\n{llm_response}")


def _question_terms(question):
    stop_words = {
        "all", "are", "each", "for", "from", "give", "how", "list", "many",
        "show", "than", "that", "the", "there", "what", "when", "where", "which",
        "our", "your", "their", "does", "do", "were", "was", "report", "reported", "reports",
        "matter", "matters", "about", "case", "cases",
    }
    raw_terms = re.findall(r"[a-z0-9]+", question.lower())
    terms = set()
    filtered_terms = []

    for term in raw_terms:
        if len(term) < 3 or term in stop_words:
            continue
        terms.add(term)
        filtered_terms.append(term)
        if term.endswith("s") and len(term) > 3:
            terms.add(term[:-1])

    for left_term, right_term in zip(filtered_terms, filtered_terms[1:]):
        combined_term = f"{left_term}{right_term}"
        terms.add(combined_term)
        if combined_term.endswith("s") and len(combined_term) > 4:
            terms.add(combined_term[:-1])

    return terms


def _parse_schema_blocks(schema_text):
    table_blocks = {}
    relationships = []
    notes = []
    current_table = None
    current_lines = []
    mode = None

    for line in schema_text.splitlines():
        if line.startswith("Table: "):
            if current_table:
                table_blocks[current_table] = current_lines
            current_table = line.split(": ", 1)[1].strip()
            current_lines = [line]
            mode = "table"
        elif line == "Relationships:":
            if current_table:
                table_blocks[current_table] = current_lines
                current_table = None
            mode = "relationships"
        elif line == "Notes:":
            if current_table:
                table_blocks[current_table] = current_lines
                current_table = None
            mode = "notes"
        else:
            if mode == "table" and current_table:
                current_lines.append(line)
            elif mode == "relationships" and line:
                relationships.append(line)
            elif mode == "notes" and line:
                notes.append(line)

    if current_table:
        table_blocks[current_table] = current_lines

    return table_blocks, relationships, notes


def _schema_identifiers(schema_text):
    table_blocks, _, _ = _parse_schema_blocks(schema_text)
    table_names = {table_name.lower(): table_name for table_name in table_blocks}
    column_names = {}
    table_columns = {}

    for table_name, lines in table_blocks.items():
        current_table_columns = {}
        for line in lines:
            if line.startswith("  - "):
                column_name = line[4:].split(" (", 1)[0].strip()
                column_names[column_name.lower()] = column_name
                current_table_columns[column_name.lower()] = column_name
        table_columns[table_name] = current_table_columns

    return table_names, column_names, table_columns


def _closest_identifier(identifier, known_identifiers, cutoff=0.82):
    matches = get_close_matches(identifier.lower(), list(known_identifiers.keys()), n=1, cutoff=cutoff)
    if matches:
        return known_identifiers[matches[0]]
    return identifier


def _repair_sql_identifiers(sql_text, schema_text):
    table_names, column_names, table_columns = _schema_identifiers(schema_text)
    sql_keywords = {"GROUP", "ORDER", "WHERE", "JOIN", "LEFT", "RIGHT", "FULL", "INNER", "LIMIT", "ON"}

    def replace_table_reference(match):
        keyword = match.group(1)
        table_name = match.group(2)
        fixed_table_name = _closest_identifier(table_name, table_names, cutoff=0.75)
        return f"{keyword} {fixed_table_name}"

    repaired_sql = re.sub(
        r"\b(FROM|JOIN|UPDATE|INTO)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        replace_table_reference,
        sql_text,
        flags=re.IGNORECASE,
    )

    aliases = {}
    for match in re.finditer(
        r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+AS)?\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        repaired_sql,
        flags=re.IGNORECASE,
    ):
        table_name = match.group(1)
        alias = match.group(2)
        if alias.upper() not in sql_keywords:
            aliases[alias] = table_name

    def replace_dotted_identifier(match):
        left_identifier = match.group(1)
        right_identifier = match.group(2)

        fixed_left_identifier = left_identifier
        target_table_name = aliases.get(left_identifier, left_identifier)

        if left_identifier not in aliases:
            fixed_left_identifier = _closest_identifier(left_identifier, table_names, cutoff=0.75)
            target_table_name = fixed_left_identifier

        candidate_columns = column_names
        if target_table_name in table_columns:
            candidate_columns = table_columns[target_table_name]

        fixed_right_identifier = _closest_identifier(right_identifier, candidate_columns)
        return f"{fixed_left_identifier}.{fixed_right_identifier}"

    return re.sub(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
        replace_dotted_identifier,
        repaired_sql,
    )


def _validate_sql_identifiers_exist(sql_text, schema_text):
    table_names, _, table_columns = _schema_identifiers(schema_text)
    sql_keywords = {"GROUP", "ORDER", "WHERE", "JOIN", "LEFT", "RIGHT", "FULL", "INNER", "LIMIT", "ON"}

    alias_to_table = {}
    for match in re.finditer(
        r"\b(?:FROM|JOIN|UPDATE|INTO)\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+AS)?(?:\s+([A-Za-z_][A-Za-z0-9_]*))?\b",
        sql_text,
        flags=re.IGNORECASE,
    ):
        table_name = match.group(1)
        alias = match.group(2)

        if table_name not in table_columns:
            allowed_tables = ", ".join(sorted(table_names.values()))
            raise ValueError(f"SQL references unknown table '{table_name}'. Allowed tables: {allowed_tables}.")

        if alias and alias.upper() not in sql_keywords:
            alias_to_table[alias] = table_name

    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", sql_text):
        left_identifier = match.group(1)
        right_identifier = match.group(2)
        target_table = alias_to_table.get(left_identifier, left_identifier)

        if target_table in table_columns and right_identifier not in table_columns[target_table]:
            allowed_columns = ", ".join(sorted(table_columns[target_table].values()))
            raise ValueError(
                f"Column '{right_identifier}' does not exist on table '{target_table}'. "
                f"Allowed columns: {allowed_columns}."
            )


def _matched_question_columns(question, schema_text):
    _, column_names, _ = _schema_identifiers(schema_text)
    question_terms = _question_terms(question)
    scored_columns = []

    for column_name in column_names.values():
        lower_column_name = column_name.lower()
        if lower_column_name.endswith("id") and "id" not in question_terms:
            continue

        score = sum(1 for term in question_terms if term in lower_column_name)
        if score > 0:
            scored_columns.append((score, column_name))

    scored_columns.sort(key=lambda item: (-item[0], item[1]))
    return [column_name for _, column_name in scored_columns]


def _validate_sql_answers_question(sql_text, question, schema_text):
    normalized_question = question.lower()
    normalized_sql = sql_text.lower()

    if "how many" in normalized_question and "count(" not in normalized_sql:
        raise ValueError("SQL does not calculate a count for a 'how many' question.")

    asks_for_breakdown = any(phrase in normalized_question for phrase in (" each ", " per ", " by ", "grouped by"))
    if not asks_for_breakdown:
        return

    matched_columns = _matched_question_columns(question, schema_text)
    if not matched_columns:
        return

    breakdown_column = matched_columns[0].lower()
    if breakdown_column not in normalized_sql:
        raise ValueError(f"SQL is missing the requested breakdown column '{matched_columns[0]}'.")
    if "group by" not in normalized_sql:
        raise ValueError("SQL is missing GROUP BY for the requested breakdown.")


def _question_keywords(question):
    stop_words = {
        "all", "and", "are", "at", "between", "due", "each", "for", "from", "give",
        "count", "difference", "did", "give", "have", "highest", "how", "in", "list", "many", "max",
        "maximum", "most", "peak", "show", "than", "that", "the", "there", "trend", "trends",
        "was", "were", "what", "when", "where", "which", "year", "our", "your", "their",
        "does", "do", "about", "report", "reported", "reports", "matter", "matters", "case", "cases",
    }
    keywords = []

    for term in re.findall(r"[a-z0-9]+", question.lower()):
        if term in stop_words or len(term) < 3 or term.isdigit():
            continue
        normalized_term = term[:-1] if term.endswith("s") and len(term) > 3 else term
        if normalized_term not in keywords:
            keywords.append(normalized_term)

    if "death" in keywords:
        keywords = [keyword for keyword in keywords if keyword not in {"case", "peak"}]

    return keywords


def _analyze_question(question):
    normalized_question = question.lower().strip()
    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", normalized_question)))
    keywords = _question_keywords(question)
    asks_definition = normalized_question.startswith(("what is ", "who is ", "tell me about ", "explain "))
    asks_record_lookup = any(
        phrase in normalized_question
        for phrase in ("show records", "list records", "show me records", "display records", "show record", "list record")
    ) or (
        normalized_question.startswith(("show ", "list ", "display "))
        and "record" in normalized_question
    )
    asks_difference = any(
        phrase in normalized_question
        for phrase in ("difference between", "difference of", "compare", "comparison", "versus", " vs ")
    )
    asks_when = "when" in normalized_question
    asks_trend = any(
        phrase in normalized_question
        for phrase in ("trend", "trends", "over time", "year by year", "yearly", "monthly", "month by month")
    )
    asks_peak = _question_has_peak_or_max_intent(normalized_question)
    asks_year_ranking = _question_has_year_ranking_intent(normalized_question)
    asks_breakdown = any(phrase in normalized_question for phrase in (" each ", " per ", " by ", "grouped by"))
    asks_count = "how many" in normalized_question or "count" in normalized_question

    if asks_record_lookup and keywords:
        intent = "record_lookup"
    elif asks_definition and keywords:
        intent = "definition_lookup"
    elif asks_difference and keywords:
        intent = "comparison"
    elif asks_year_ranking and keywords:
        intent = "top_year"
    elif (asks_peak or asks_when) and keywords:
        intent = "peak_period"
    elif asks_trend and keywords:
        intent = "trend"
    elif years and keywords:
        intent = "time_filtered_aggregate"
    else:
        intent = "sql_generation"

    prefer_deterministic = intent in {
        "record_lookup",
        "definition_lookup",
        "comparison",
        "top_year",
        "peak_period",
        "trend",
        "time_filtered_aggregate",
    }

    year_range = None
    if years:
        year_range = {"start": years[0], "end": years[-1]}

    return {
        "normalized_question": normalized_question,
        "intent": intent,
        "keywords": keywords,
        "years": years,
        "year_range": year_range,
        "asks_record_lookup": asks_record_lookup,
        "asks_definition": asks_definition,
        "asks_difference": asks_difference,
        "asks_when": asks_when,
        "asks_trend": asks_trend,
        "asks_peak": asks_peak,
        "asks_year_ranking": asks_year_ranking,
        "asks_breakdown": asks_breakdown,
        "asks_count": asks_count,
        "prefer_deterministic": prefer_deterministic,
    }


def _question_has_peak_or_max_intent(question):
    normalized_question = question.lower()
    return any(term in normalized_question for term in ("peak", "highest", "maximum", "max", "most"))


def _question_has_year_ranking_intent(question):
    normalized_question = question.lower()
    return (
        any(phrase in normalized_question for phrase in ("which year", "what year"))
        and _question_has_peak_or_max_intent(normalized_question)
    )


def _should_prefer_fallback(question, analysis=None):
    analysis = analysis or _analyze_question(question)
    normalized_question = analysis["normalized_question"]
    asks_orgunit_dates = (
        any(phrase in normalized_question for phrase in ("organisation unit", "organisation units", "org unit", "org units", "facility", "facilities"))
        and any(term in normalized_question for term in ("opened", "opening date", "openingdate", "closed", "closed date", "closeddate"))
    )
    if asks_orgunit_dates:
        return True
    return analysis["prefer_deterministic"]


def _searchable_keyword_clauses(keywords, alias="de"):
    searchable_fields = [
        f"lower(coalesce({alias}.name, ''))",
        f"lower(coalesce({alias}.shortname, ''))",
        f"lower(coalesce({alias}.code, ''))",
        f"lower(coalesce({alias}.description, ''))",
    ]

    keyword_clauses = []
    for keyword in keywords:
        field_matches = [f"{field} LIKE '%{keyword}%'" for field in searchable_fields]
        keyword_clauses.append(f"({' OR '.join(field_matches)})")

    return keyword_clauses


def _build_year_filter_clause(years, date_expression="substr(p.startdate, 1, 4)"):
    if not years:
        return None

    start_year = min(years)
    end_year = max(years)
    if start_year == end_year:
        return f"{date_expression} = '{start_year}'"
    return f"{date_expression} BETWEEN '{start_year}' AND '{end_year}'"


def _extract_comparison_terms(analysis):
    normalized_question = analysis["normalized_question"]
    detected_terms = []

    for candidate in ("positive", "negative", "male", "female", "urban", "rural"):
        if re.search(rf"\b{candidate}\b", normalized_question):
            detected_terms.append(candidate)

    if len(detected_terms) >= 2:
        return detected_terms[:2]

    between_match = re.search(
        r"difference between\s+([a-z0-9 _-]+?)\s+and\s+([a-z0-9 _-]+?)(?:\s+in\b|\s+for\b|\?|$)",
        normalized_question,
    )
    if between_match:
        left_phrase = between_match.group(1).strip()
        right_phrase = between_match.group(2).strip()
        if left_phrase and right_phrase:
            return [left_phrase, right_phrase]

    return []


def _build_dataelement_evidence_query(keyword_clauses, years=None, limit=None):
    year_filter = _build_year_filter_clause(years)
    numeric_filters = [
        "trim(coalesce(dv.value, '')) <> ''",
        "trim(dv.value) NOT GLOB '*[^0-9]*'",
    ]
    if year_filter:
        numeric_filters.insert(0, year_filter)

    return f"""
WITH matched_elements AS (
  SELECT
    de.dataelementid,
    de.name,
    COALESCE(de.shortname, '') AS shortname,
    COALESCE(de.description, '') AS description,
    de.valuetype,
    de.aggregationtype
  FROM dataelement AS de
  WHERE {" AND ".join(keyword_clauses)}
),
numeric_values AS (
  SELECT
    dv.dataelementid,
    CAST(dv.value AS INTEGER) AS numeric_value,
    p.startdate,
    p.enddate,
    ROW_NUMBER() OVER (
      PARTITION BY dv.dataelementid
      ORDER BY p.startdate DESC, p.enddate DESC, dv.value DESC
    ) AS recency_rank
  FROM datavalue AS dv
  JOIN period AS p ON dv.periodid = p.periodid
  WHERE {" AND ".join(numeric_filters)}
)
SELECT
  me.dataelementid,
  me.name,
  me.shortname,
  me.description,
  me.valuetype,
  me.aggregationtype,
  COUNT(nv.numeric_value) AS data_points,
  COALESCE(SUM(nv.numeric_value), 0) AS total_value,
  MAX(CASE WHEN nv.recency_rank = 1 THEN nv.numeric_value END) AS latest_value,
  MAX(CASE WHEN nv.recency_rank = 1 THEN nv.startdate END) AS latest_period_start,
  MAX(CASE WHEN nv.recency_rank = 1 THEN nv.enddate END) AS latest_period_end,
  MIN(nv.startdate) AS first_period_start,
  MAX(nv.enddate) AS last_period_end
FROM matched_elements AS me
LEFT JOIN numeric_values AS nv ON nv.dataelementid = me.dataelementid
GROUP BY
  me.dataelementid,
  me.name,
  me.shortname,
  me.description,
  me.valuetype,
  me.aggregationtype
ORDER BY total_value DESC, data_points DESC, me.name ASC
""".strip()


def _fallback_sql_for_question(question, schema_text, analysis=None):
    table_names, _, _ = _schema_identifiers(schema_text)
    analysis = analysis or _analyze_question(question)
    years = analysis["years"]
    keywords = analysis["keywords"]
    if not keywords:
        return None

    available_tables = {name.lower() for name in table_names.values()}
    normalized_question = analysis["normalized_question"]

    if "organisationunit" in available_tables and any(
        phrase in normalized_question for phrase in ("organisation unit", "organisation units", "org unit", "org units", "facility", "facilities")
    ):
        asks_opening = any(term in normalized_question for term in ("opened", "opening date", "openingdate"))
        asks_closing = any(term in normalized_question for term in ("closed", "closed date", "closeddate"))
        asks_earliest = any(term in normalized_question for term in ("earliest", "oldest", "first"))
        asks_latest = any(term in normalized_question for term in ("latest", "newest", "most recent", "last"))

        if asks_opening or asks_closing:
            date_column = "openingdate" if asks_opening else "closeddate"
            sort_function = "MIN" if asks_earliest or not asks_latest else "MAX"
            return f"""
SELECT
  name,
  code,
  {date_column}
FROM organisationunit
WHERE trim(coalesce({date_column}, '')) <> ''
  AND {date_column} = (
    SELECT {sort_function}({date_column})
    FROM organisationunit
    WHERE trim(coalesce({date_column}, '')) <> ''
  )
ORDER BY name ASC
""".strip()

    if analysis["intent"] == "record_lookup":
        if "dataelement" not in available_tables:
            return None

        record_keywords = [keyword for keyword in keywords if keyword not in {"record", "records"}]
        record_keyword_clauses = _searchable_keyword_clauses(record_keywords or keywords)
        if not record_keyword_clauses:
            return None

        required_tables = {"dataelement", "datavalue", "period"}
        if required_tables.issubset(available_tables):
            return _build_dataelement_evidence_query(record_keyword_clauses, years=years)

        return f"""
SELECT
  de.dataelementid,
  de.name,
  COALESCE(de.shortname, '') AS shortname,
  COALESCE(de.description, '') AS description,
  de.valuetype,
  de.aggregationtype
FROM dataelement AS de
WHERE {" AND ".join(record_keyword_clauses)}
ORDER BY de.name ASC
""".strip()

    required_tables = {"dataelement", "datavalue", "period"}
    if not required_tables.issubset(available_tables):
        return None

    year_filter = _build_year_filter_clause(years)
    keyword_clauses = _searchable_keyword_clauses(keywords)
    if not keyword_clauses:
        return None

    base_filters = [
        " AND ".join(keyword_clauses),
        "trim(coalesce(dv.value, '')) <> ''",
        "trim(dv.value) NOT GLOB '*[^0-9]*'",
    ]
    if year_filter:
        base_filters.insert(0, year_filter)
    where_clause = "\n  AND ".join(base_filters)

    if analysis["intent"] == "comparison":
        comparison_terms = _extract_comparison_terms(analysis)
        if len(comparison_terms) < 2:
            return None

        comparison_filters = [
            f"(lower(coalesce(de.name, '')) LIKE '%{term}%' OR "
            f"lower(coalesce(de.shortname, '')) LIKE '%{term}%' OR "
            f"lower(coalesce(de.code, '')) LIKE '%{term}%' OR "
            f"lower(coalesce(de.description, '')) LIKE '%{term}%')"
            for term in comparison_terms
        ]
        base_subject_keywords = [
            keyword
            for keyword in keywords
            if keyword not in comparison_terms and keyword not in {"case", "count", "difference"}
        ]
        base_keyword_filters = _searchable_keyword_clauses(base_subject_keywords)
        base_comparison_filters = [
            "trim(coalesce(dv.value, '')) <> ''",
            "trim(dv.value) NOT GLOB '*[^0-9]*'",
        ]
        if year_filter:
            base_comparison_filters.insert(0, year_filter)
        if base_keyword_filters:
            base_comparison_filters.insert(1 if year_filter else 0, " AND ".join(base_keyword_filters))

        comparison_where_clause = "\n  AND ".join(base_comparison_filters)
        left_term, right_term = comparison_terms[:2]

        return f"""
WITH filtered AS (
  SELECT de.name, SUM(CAST(dv.value AS INTEGER)) AS total_value
  FROM datavalue AS dv
  JOIN dataelement AS de ON dv.dataelementid = de.dataelementid
  JOIN period AS p ON dv.periodid = p.periodid
  WHERE {comparison_where_clause}
    AND ({' OR '.join(comparison_filters)})
  GROUP BY de.name
)
SELECT
  '{left_term}' AS left_term,
  SUM(CASE WHEN lower(name) LIKE '%{left_term}%' THEN total_value ELSE 0 END) AS left_total,
  '{right_term}' AS right_term,
  SUM(CASE WHEN lower(name) LIKE '%{right_term}%' THEN total_value ELSE 0 END) AS right_total,
  SUM(CASE WHEN lower(name) LIKE '%{left_term}%' THEN total_value ELSE 0 END) -
  SUM(CASE WHEN lower(name) LIKE '%{right_term}%' THEN total_value ELSE 0 END) AS difference
FROM filtered
""".strip()

    if analysis["intent"] == "top_year":
        return f"""
SELECT substr(p.startdate, 1, 4) AS year, SUM(CAST(dv.value AS INTEGER)) AS total_value
FROM datavalue AS dv
JOIN dataelement AS de ON dv.dataelementid = de.dataelementid
JOIN period AS p ON dv.periodid = p.periodid
WHERE {where_clause}
GROUP BY year
ORDER BY total_value DESC, year ASC
LIMIT 1
""".strip()

    if analysis["intent"] == "peak_period":
        return f"""
SELECT de.name, p.startdate, p.enddate, SUM(CAST(dv.value AS INTEGER)) AS total_value
FROM datavalue AS dv
JOIN dataelement AS de ON dv.dataelementid = de.dataelementid
JOIN period AS p ON dv.periodid = p.periodid
WHERE {where_clause}
GROUP BY de.name, p.startdate, p.enddate
ORDER BY total_value DESC, p.startdate ASC
LIMIT 1
""".strip()

    if analysis["intent"] == "trend":
        return f"""
SELECT substr(p.startdate, 1, 4) AS year, SUM(CAST(dv.value AS INTEGER)) AS total_value
FROM datavalue AS dv
JOIN dataelement AS de ON dv.dataelementid = de.dataelementid
JOIN period AS p ON dv.periodid = p.periodid
WHERE {where_clause}
GROUP BY year
ORDER BY year ASC
""".strip()

    if analysis["intent"] == "definition_lookup":
        return _build_dataelement_evidence_query(keyword_clauses, years=years, limit=10)

    return f"""
SELECT de.name, SUM(CAST(dv.value AS INTEGER)) AS total_value
FROM datavalue AS dv
JOIN dataelement AS de ON dv.dataelementid = de.dataelementid
JOIN period AS p ON dv.periodid = p.periodid
WHERE {where_clause}
GROUP BY de.name
ORDER BY total_value DESC
""".strip()


def _fallback_sql_when_model_unavailable(question, schema_text, analysis=None):
    analysis = analysis or _analyze_question(question)
    preferred_sql = _fallback_sql_for_question(question, schema_text, analysis=analysis)
    if preferred_sql:
        return preferred_sql

    table_names, _, _ = _schema_identifiers(schema_text)
    available_tables = {name.lower() for name in table_names.values()}
    keywords = [keyword for keyword in analysis["keywords"] if keyword not in {"record", "records"}]
    required_tables = {"dataelement", "datavalue", "period"}

    if required_tables.issubset(available_tables) and keywords:
        keyword_clauses = _searchable_keyword_clauses(keywords)
        if keyword_clauses:
            return _build_dataelement_evidence_query(keyword_clauses, years=analysis["years"])

    if "dataelement" in available_tables and keywords:
        keyword_clauses = _searchable_keyword_clauses(keywords)
        if keyword_clauses:
            return f"""
SELECT
  de.dataelementid,
  de.name,
  COALESCE(de.shortname, '') AS shortname,
  COALESCE(de.description, '') AS description,
  de.valuetype,
  de.aggregationtype
FROM dataelement AS de
WHERE {" AND ".join(keyword_clauses)}
ORDER BY de.name ASC
""".strip()

    if "organisationunit" in available_tables and any(
        keyword in analysis["keywords"]
        for keyword in ("organisation", "unit", "hierarchy", "level")
    ):
        return """
SELECT
  organisationunitid,
  name,
  shortname,
  code,
  hierarchylevel,
  uid,
  path,
  openingdate,
  lastupdated
FROM organisationunit
ORDER BY hierarchylevel ASC, name ASC
""".strip()

    return None


def _request_sql_from_llm(
    prompt,
    schema_text,
    question,
    analysis=None,
    max_attempts=3,
    trace=None,
    trace_stage="generate",
    force_model=False,
):
    analysis = analysis or _analyze_question(question)
    preferred_fallback_sql = _fallback_sql_for_question(question, schema_text, analysis=analysis)
    if trace is not None:
        trace.setdefault("entries", [])
        trace.setdefault("raw_responses", [])
        trace.setdefault("model_used", False)
        trace.setdefault("raw_output", None)
        trace.setdefault("final_sql", None)

    if preferred_fallback_sql and _should_prefer_fallback(question, analysis=analysis) and not force_model:
        if trace is not None:
            trace["model_used"] = False
            trace["final_sql"] = preferred_fallback_sql
        return preferred_fallback_sql

    if not is_model_available():
        unavailable_fallback_sql = _fallback_sql_when_model_unavailable(question, schema_text, analysis=analysis)
        if unavailable_fallback_sql:
            if trace is not None:
                trace["model_used"] = False
                trace["final_sql"] = unavailable_fallback_sql
            return unavailable_fallback_sql
        raise ValueError(
            f"Model unavailable and no heuristic fallback matched this question. {model_unavailable_reason()}"
        )

    follow_up = ""
    last_error = None
    last_sql_text = None

    for _ in range(max_attempts):
        llm_response = call_llm(f"{prompt}{follow_up}")
        if trace is not None:
            trace["model_used"] = True
            trace["entries"].append(
                {
                    "stage": trace_stage,
                    "attempt": len(trace["entries"]) + 1,
                    "response": llm_response,
                }
            )
            trace["raw_responses"] = [entry["response"] for entry in trace["entries"]]
            trace["raw_output"] = "\n\n".join(
                f"[{entry['stage']} attempt {entry['attempt']}]\n{entry['response']}"
                for entry in trace["entries"]
            )
        print("\n===== RAW SQL MODEL OUTPUT =====\n")
        print(llm_response)
        print("\n================================\n")
        try:
            sql_text = _extract_sql_from_response(llm_response)
            sql_text = _repair_sql_identifiers(sql_text, schema_text)
            print("\n===== EXTRACTED SQL QUERY =====\n")
            print(sql_text)
            print("\n================================\n")
            last_sql_text = sql_text
            _validate_sql(sql_text)
            _validate_sql_identifiers_exist(sql_text, schema_text)
            _validate_sql_answers_question(sql_text, question, schema_text)
            if trace is not None:
                trace["final_sql"] = sql_text
            return sql_text
        except ValueError as exc:
            last_error = exc
            follow_up = (
                "\n\nThe previous response could not be executed because it was not a complete SQLite query. "
                f"Problem: {exc}. Return one complete SQL query in the YAML block with no placeholders and no comments."
            )

    fallback_sql = preferred_fallback_sql or _fallback_sql_for_question(question, schema_text, analysis=analysis)
    if fallback_sql:
        if trace is not None:
            trace["final_sql"] = fallback_sql
        return fallback_sql

    if last_sql_text:
        if trace is not None:
            trace["final_sql"] = last_sql_text
        return last_sql_text

    raise ValueError(f"Failed to generate executable SQL after {max_attempts} attempts: {last_error}")


def _generic_column_description(column_name):
    normalized = column_name.lower()
    if normalized.endswith("id"):
        return "Identifier column."
    if normalized == "name":
        return "Display name."
    if normalized == "shortname":
        return "Short display label."
    if normalized == "code":
        return "Optional implementation code."
    if normalized == "description":
        return "Optional human-readable description."
    if normalized == "uid":
        return "Stable UID used by DHIS2."
    if normalized == "created":
        return "Record creation timestamp."
    if normalized == "lastupdated":
        return "Last update timestamp."
    if normalized == "startdate":
        return "Start date."
    if normalized == "enddate":
        return "End date."
    return None


def _column_description(table_name, column_name):
    return COLUMN_DESCRIPTION_HINTS.get((table_name.lower(), column_name.lower())) or _generic_column_description(column_name)


def _parse_relationship_tables(line, table_blocks):
    table_matches = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_][A-Za-z0-9_]*", line)
    return [table_name for table_name in table_matches if table_name in table_blocks]


def _schema_summary(schema_text):
    table_blocks, relationships, notes = _parse_schema_blocks(schema_text)
    table_names = list(table_blocks.keys())
    preview = ", ".join(table_names[:12])
    if len(table_names) > 12:
        preview += ", ..."

    lines = [
        f"Tables: {len(table_names)}",
        f"Relationships: {len(relationships)}",
    ]
    if preview:
        lines.append(f"Table preview: {preview}")
    if notes:
        lines.append(f"Schema notes: {len(notes)}")
    return "\n".join(lines)


@lru_cache(maxsize=4)
def _build_schema_chunks(schema_text):
    table_blocks, relationships, notes = _parse_schema_blocks(schema_text)
    relationship_graph = {table_name: set() for table_name in table_blocks}
    relationship_lines_by_table = {table_name: [] for table_name in table_blocks}

    for line in relationships:
        relationship_tables = _parse_relationship_tables(line, table_blocks)
        for table_name in relationship_tables:
            relationship_lines_by_table[table_name].append(line)
        if len(relationship_tables) == 2:
            left_table, right_table = relationship_tables
            relationship_graph[left_table].add(right_table)
            relationship_graph[right_table].add(left_table)

    chunks = {}
    table_order = list(table_blocks.keys())
    for table_name, lines in table_blocks.items():
        normalized_table = table_name.lower()
        aliases = TABLE_ALIAS_HINTS.get(normalized_table, [])
        purpose = TABLE_DESCRIPTION_HINTS.get(normalized_table, f"Stores {table_name.replace('_', ' ')} data.")

        column_lines = []
        table_terms = set(_question_terms(table_name))
        column_terms = set()
        alias_terms = set()

        for alias in aliases:
            alias_terms.update(_question_terms(alias))
            alias_terms.update(re.findall(r"[a-z0-9]+", alias.lower()))

        for line in lines:
            if not line.startswith("  - "):
                continue

            column_name = line[4:].split(" (", 1)[0].strip()
            column_type_match = re.search(r"\(([^)]+)\)", line)
            column_type = column_type_match.group(1).strip() if column_type_match else ""
            column_description = _column_description(table_name, column_name)
            if column_description:
                column_lines.append(f"  - {column_name} ({column_type}): {column_description}")
            else:
                column_lines.append(f"  - {column_name} ({column_type})")

            column_terms.update(_question_terms(column_name))
            column_terms.update(re.findall(r"[a-z0-9]+", column_name.lower()))

        chunk_lines = [
            f"Table: {table_name}",
            f"Purpose: {purpose}",
        ]
        if aliases:
            chunk_lines.append("Aliases: " + ", ".join(aliases))
        if column_lines:
            chunk_lines.append("Columns:")
            chunk_lines.extend(column_lines)

        relationship_lines = relationship_lines_by_table.get(table_name, [])
        if relationship_lines:
            chunk_lines.append("Relationships:")
            chunk_lines.extend(relationship_lines)

        if notes and normalized_table == "datavalue":
            chunk_lines.append("Notes:")
            chunk_lines.extend(notes)

        terms = set()
        terms.update(table_terms)
        terms.update(column_terms)
        terms.update(alias_terms)
        terms.update(_question_terms(purpose))

        chunks[table_name] = {
            "table_name": table_name,
            "text": "\n".join(chunk_lines).strip(),
            "table_terms": table_terms,
            "column_terms": column_terms,
            "alias_terms": alias_terms,
            "terms": terms,
        }

    return table_order, chunks, relationships, notes, relationship_graph


def _schema_table_limit(max_tables, analysis):
    if not analysis:
        return max_tables

    limit = max_tables
    if analysis["intent"] in {"comparison", "trend", "time_filtered_aggregate", "peak_period", "top_year"}:
        limit = max(limit, 3)
    if analysis["asks_breakdown"] or analysis["asks_when"] or analysis["asks_trend"]:
        limit = max(limit, 4)
    return min(limit, 5)


def _schema_chunk_score(chunk, question_terms, analysis):
    score = 0
    for term in question_terms:
        if term in chunk["table_terms"]:
            score += 12
        elif term in chunk["alias_terms"]:
            score += 8
        elif term in chunk["column_terms"]:
            score += 6
        elif term in chunk["terms"]:
            score += 3

    table_name = chunk["table_name"].lower()
    keywords = set((analysis or {}).get("keywords", []))
    if table_name in {"dataelement", "datavalue"} and keywords:
        score += 4
    if table_name == "period" and (analysis or {}).get("years"):
        score += 6
    if table_name == "periodtype" and any(term in question_terms for term in ("monthly", "weekly", "quarterly", "yearly", "frequency", "periodtype")):
        score += 6
    if table_name == "organisationunit" and any(term in question_terms for term in ("facility", "facilities", "organisation", "orgunit", "district", "hierarchy", "level", "path", "opened", "openingdate")):
        score += 8

    if analysis:
        if analysis["intent"] == "definition_lookup" and table_name == "dataelement":
            score += 10
        if analysis["intent"] in {"comparison", "trend", "time_filtered_aggregate", "peak_period", "top_year"} and table_name in {"dataelement", "datavalue", "period"}:
            score += 8
        if analysis["asks_breakdown"] and table_name == "organisationunit":
            score += 5

    return score


def _connect_selected_tables(selected_tables, relationship_graph):
    selected_table_set = set(selected_tables)

    for start_index, start_table in enumerate(selected_tables):
        for end_table in selected_tables[start_index + 1:]:
            queue = [(start_table, [start_table])]
            visited = {start_table}

            while queue:
                current_table, path = queue.pop(0)
                if current_table == end_table:
                    selected_table_set.update(path)
                    break

                for neighbor in relationship_graph.get(current_table, set()):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    return selected_table_set


def _select_relevant_schema(question, schema_text, analysis=None, max_tables=2):
    table_order, chunks, _, _, relationship_graph = _build_schema_chunks(schema_text)
    question_terms = _question_terms(question)

    if not chunks:
        return schema_text

    scored_tables = []
    for table_name in table_order:
        chunk = chunks[table_name]
        score = _schema_chunk_score(chunk, question_terms, analysis)
        if score > 0:
            scored_tables.append((score, table_name))

    table_limit = _schema_table_limit(max_tables, analysis)
    if scored_tables:
        scored_tables.sort(key=lambda item: (-item[0], item[1]))
        selected_tables = [table_name for _, table_name in scored_tables[:table_limit]]
    else:
        preferred_tables = [table_name for table_name in ("dataelement", "datavalue", "organisationunit", "period", "periodtype") if table_name in chunks]
        selected_tables = preferred_tables[:table_limit] if preferred_tables else table_order[:table_limit]

    selected_table_set = _connect_selected_tables(selected_tables, relationship_graph)
    selected_tables_in_order = [table_name for table_name in table_order if table_name in selected_table_set]

    focused_schema = []
    for table_name in selected_tables_in_order:
        focused_schema.append(chunks[table_name]["text"])
        focused_schema.append("")

    return "\n".join(focused_schema).strip()

class AnalyzeQuestion(Node):
    def prep(self, shared):
        return shared["natural_query"]

    def exec(self, natural_query):
        return _analyze_question(natural_query)

    def post(self, shared, prep_res, exec_res):
        shared["query_analysis"] = exec_res
        print("\n===== QUERY ANALYSIS =====\n")
        print(f"Intent: {exec_res['intent']}")
        print(f"Keywords: {', '.join(exec_res['keywords']) or '(none)'}")
        if exec_res["years"]:
            print(f"Years: {', '.join(exec_res['years'])}")
        print("\n==========================\n")

class GetSchema(Node):
    def prep(self, shared):
        return shared["db_path"]

    def exec(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        tables = cursor.fetchall()
        schema = []
        relationships = []
        table_names = {table_name_tuple[0] for table_name_tuple in tables}

        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            schema.append(f"Table: {table_name}")
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            for col in columns:
                schema.append(f"  - {col[1]} ({col[2]})")

            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            for fk in cursor.fetchall():
                relationships.append(f"  - {table_name}.{fk[3]} -> {fk[2]}.{fk[4]}")
            schema.append("")

        if {"datavalue", "dataelement", "organisationunit", "period", "periodtype"}.issubset(table_names):
            relationships.extend(
                [
                    "  - datavalue.dataelementid -> dataelement.dataelementid",
                    "  - datavalue.periodid -> period.periodid",
                    "  - datavalue.sourceid -> organisationunit.organisationunitid",
                    "  - period.periodtypeid -> periodtype.periodtypeid",
                ]
            )

        if relationships:
            deduped_relationships = list(dict.fromkeys(relationships))
            schema.append("Relationships:")
            schema.extend(deduped_relationships)
            schema.append("")

        if "datavalue" in table_names:
            schema.append("Notes:")
            schema.append("  - datavalue.value is stored as TEXT and may need CAST(value AS REAL) for numeric aggregations.")

        conn.close()
        return "\n".join(schema).strip()

    def post(self, shared, prep_res, exec_res):
        shared["schema"] = exec_res
        table_blocks, _, _ = _parse_schema_blocks(exec_res)
        print("\n===== DB SCHEMA =====\n")
        if len(table_blocks) <= 25:
            print(exec_res)
        else:
            print(_schema_summary(exec_res))
            print("\n(Full schema loaded in memory; compact retrieved schema context will be used below.)")
        print("\n=====================\n")
        # return "default"

class GenerateSQL(Node):
    def prep(self, shared):
        focused_schema = _select_relevant_schema(
            shared["natural_query"],
            shared["schema"],
            analysis=shared.get("query_analysis"),
        )
        return (
            shared["natural_query"],
            focused_schema,
            shared["schema"],
            shared["query_analysis"],
            shared.setdefault("sql_trace", {}),
            bool(shared.get("force_sql_model")),
        )

    def exec(self, prep_res):
        natural_query, schema, full_schema, query_analysis, sql_trace, force_sql_model = prep_res
        print("\n===== RETRIEVED SCHEMA CONTEXT =====\n")
        print(schema)
        print("\n====================================\n")
        prompt = f"""
You are an expert SQLite assistant.
Generate a valid SQLite query using only the retrieved schema context shown below.
Return only a YAML block with a single 'sql' key.
Do not use placeholders such as SELECT ..., table_name, column_name, or <query>.
Do not include SQL comments.
Make sure the SQL fully answers the question. If the question asks for a breakdown, include the grouping columns and aggregations needed for that breakdown.

Retrieved schema context:
{schema}

Question: "{natural_query}"

Respond ONLY with a YAML block in this shape:
```yaml
sql: |
  <complete runnable SQLite query>
```"""
        return _request_sql_from_llm(
            prompt,
            full_schema,
            natural_query,
            analysis=query_analysis,
            trace=sql_trace,
            trace_stage="generate",
            force_model=force_sql_model,
        )

    def post(self, shared, prep_res, exec_res):
        # exec_res is now the parsed SQL query string
        shared["generated_sql"] = exec_res
        # Reset debug attempts when *successfully* generating new SQL
        shared["debug_attempts"] = 0
        print(f"\n===== GENERATED SQL (Attempt {shared.get('debug_attempts', 0) + 1}) =====\n")
        print(exec_res)
        print("\n====================================\n")
        # return "default"

class ExecuteSQL(Node):
    def prep(self, shared):
        return shared["db_path"], shared["generated_sql"]

    def exec(self, prep_res):
        db_path, sql_query = prep_res
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            start_time = time.time()
            cursor.execute(sql_query)

            is_select = sql_query.strip().upper().startswith(("SELECT", "WITH"))
            if is_select:
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            else:
                conn.commit()
                results = f"Query OK. Rows affected: {cursor.rowcount}"
                column_names = []
            conn.close()
            duration = time.time() - start_time
            print(f"SQL executed in {duration:.3f} seconds.")
            return (True, results, column_names)
        except sqlite3.Error as e:
            print(f"SQLite Error during execution: {e}")
            if 'conn' in locals() and conn:
                 try:
                     conn.close()
                 except Exception:
                     pass
            return (False, str(e), [])

    def post(self, shared, prep_res, exec_res):
        success, result_or_error, column_names = exec_res

        if success:
            shared["final_result"] = result_or_error
            shared["result_columns"] = column_names
            print("\n===== SQL EXECUTION SUCCESS =====\n")
            # (Same result printing logic as before)
            if isinstance(result_or_error, list):
                 if column_names: print(" | ".join(column_names)); print("-" * (sum(len(str(c)) for c in column_names) + 3 * (len(column_names) -1)))
                 if not result_or_error: print("(No results found)")
                 else:
                     for row in result_or_error: print(" | ".join(map(str, row)))
            else: print(result_or_error)
            print("\n=================================\n")
            return "success"
        else:
            # Execution failed (SQLite error caught in exec)
            shared["execution_error"] = result_or_error # Store the error message
            shared["debug_attempts"] = shared.get("debug_attempts", 0) + 1
            max_attempts = shared.get("max_debug_attempts", 3) # Get max attempts from shared

            print(f"\n===== SQL EXECUTION FAILED (Attempt {shared['debug_attempts']}) =====\n")
            print(f"Error: {shared['execution_error']}")
            print("=========================================\n")

            if shared["debug_attempts"] >= max_attempts:
                print(f"Max debug attempts ({max_attempts}) reached. Stopping.")
                shared["final_error"] = f"Failed to execute SQL after {max_attempts} attempts. Last error: {shared['execution_error']}"
                return "done"
            else:
                print("Attempting to debug the SQL...")
                return "error_retry" # Signal to go to DebugSQL


class BuildAnalytics(Node):
    def prep(self, shared):
        return (
            shared["natural_query"],
            shared["query_analysis"],
            shared["generated_sql"],
            shared.get("result_columns", []),
            shared.get("final_result", []),
        )

    def exec(self, prep_res):
        natural_query, query_analysis, sql_query, columns, results = prep_res
        payload = build_analytics_payload(
            natural_query,
            query_analysis,
            sql_query,
            columns,
            results,
        )
        context = build_analytics_context(payload)
        return payload, context

    def post(self, shared, prep_res, exec_res):
        payload, context = exec_res
        shared["analytics_payload"] = payload
        shared["analytics_context"] = context

        print("\n===== ANALYTICS SUMMARY =====\n")
        for highlight in payload.get("highlights", []):
            print(f"- {highlight}")
        print("\nAnalytics payload ready for a downstream answer model.")
        print("\n=============================\n")

        print("===== FULL ANALYTICS PAYLOAD =====\n")
        print(json.dumps(payload, indent=2))
        print("\n==================================\n")

class DebugSQL(Node):
    def prep(self, shared):
        focused_schema = _select_relevant_schema(
            shared.get("natural_query"),
            shared.get("schema"),
            analysis=shared.get("query_analysis"),
        )
        return (
            shared.get("natural_query"),
            focused_schema,
            shared.get("schema"),
            shared.get("query_analysis"),
            shared.get("generated_sql"),
            shared.get("execution_error"),
            shared.setdefault("sql_trace", {}),
            bool(shared.get("force_sql_model")),
        )

    def exec(self, prep_res):
        (
            natural_query,
            schema,
            full_schema,
            query_analysis,
            failed_sql,
            error_message,
            sql_trace,
            force_sql_model,
        ) = prep_res
        print("\n===== RETRIEVED SCHEMA CONTEXT =====\n")
        print(schema)
        print("\n====================================\n")
        prompt = f"""
You are fixing a SQLite query.
Use only the tables and columns from the retrieved schema context.
Return only a YAML block with a single 'sql' key.
Do not use placeholders such as SELECT ..., table_name, column_name, or <query>.
Do not include SQL comments.
Make sure the corrected SQL fully answers the original question.

The following SQLite SQL query failed:
```sql
{failed_sql}
```
It was generated for: "{natural_query}"
Retrieved schema context:
{schema}
Error: "{error_message}"

Provide a corrected SQLite query.

Respond ONLY with a YAML block in this shape:
```yaml
sql: |
  <complete runnable SQLite query>
```"""
        return _request_sql_from_llm(
            prompt,
            full_schema,
            natural_query,
            analysis=query_analysis,
            trace=sql_trace,
            trace_stage="debug",
            force_model=force_sql_model,
        )

    def post(self, shared, prep_res, exec_res):
        # exec_res is the corrected SQL string
        shared["generated_sql"] = exec_res # Overwrite with the new attempt
        shared.pop("execution_error", None) # Clear the previous error for the next ExecuteSQL attempt

        print(f"\n===== REVISED SQL (Attempt {shared.get('debug_attempts', 0) + 1}) =====\n")
        print(exec_res)
        print("\n====================================\n")


class Finish(Node):
    def exec(self, _):
        return None
