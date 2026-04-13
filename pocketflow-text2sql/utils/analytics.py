import json


def _serialize_value(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _rows_to_dicts(columns, rows, max_rows=200):
    if not columns or not isinstance(rows, list):
        return []

    limited_rows = rows[:max_rows]
    return [
        {column: _serialize_value(value) for column, value in zip(columns, row)}
        for row in limited_rows
    ]


def _is_numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _detect_numeric_columns(row_dicts):
    numeric_columns = []
    if not row_dicts:
        return numeric_columns

    for column in row_dicts[0]:
        values = [row.get(column) for row in row_dicts if row.get(column) is not None]
        if values and all(_is_numeric(value) for value in values):
            numeric_columns.append(column)

    return numeric_columns


def _detect_time_columns(columns):
    time_tokens = ("year", "date", "month", "start", "end")
    return [
        column for column in columns
        if any(token in column.lower() for token in time_tokens)
    ]


def _preferred_analytics_columns(query_analysis, columns):
    intent = (query_analysis or {}).get("intent")
    preferred_by_intent = {
        "record_lookup": [
            "dataelementid",
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
        ],
        "definition_lookup": [
            "name",
            "description",
            "data_points",
            "total_value",
            "latest_value",
            "latest_period_start",
            "latest_period_end",
            "first_period_start",
            "last_period_end",
        ],
        "trend": ["year", "total_value"],
        "top_year": ["year", "total_value"],
        "peak_period": ["name", "startdate", "enddate", "total_value"],
        "comparison": ["left_term", "left_total", "right_term", "right_total", "difference"],
    }

    preferred_columns = preferred_by_intent.get(intent, [])
    selected_columns = [column for column in preferred_columns if column in columns]
    if selected_columns:
        return selected_columns

    priority_tokens = ("name", "description", "value", "count", "total", "year", "date", "period")
    signal_columns = [
        column for column in columns
        if any(token in column.lower() for token in priority_tokens)
    ]
    if signal_columns:
        return signal_columns

    return list(columns)


def _focus_row_dicts(query_analysis, row_dicts):
    if not row_dicts:
        return [], []

    columns = list(row_dicts[0].keys())
    selected_columns = _preferred_analytics_columns(query_analysis, columns)
    focused_rows = [
        {column: row.get(column) for column in selected_columns}
        for row in row_dicts
    ]
    return selected_columns, focused_rows


def _build_highlights(question, query_analysis, row_dicts):
    if not row_dicts:
        return ["No rows were returned from the database for this question."]

    intent = query_analysis.get("intent")
    top_row = row_dicts[0]

    if intent == "top_year" and {"year", "total_value"}.issubset(top_row):
        return [f"Highest year is {top_row['year']} with total value {top_row['total_value']}."]

    if intent == "peak_period" and {"name", "startdate", "enddate", "total_value"}.issubset(top_row):
        return [
            f"Peak period is {top_row['startdate']} to {top_row['enddate']} for {top_row['name']}.",
            f"Peak total value is {top_row['total_value']}.",
        ]

    if intent == "comparison" and {"left_term", "left_total", "right_term", "right_total", "difference"}.issubset(top_row):
        return [
            f"{top_row['left_term']} total is {top_row['left_total']}.",
            f"{top_row['right_term']} total is {top_row['right_total']}.",
            f"Difference ({top_row['left_term']} - {top_row['right_term']}) is {top_row['difference']}.",
        ]

    if intent == "record_lookup":
        names = [row.get("name") for row in row_dicts[:5] if row.get("name")]
        top_value = top_row.get("total_value")
        latest_value = top_row.get("latest_value")
        latest_period = top_row.get("latest_period_start")
        if names:
            highlights = [
                f"Found {len(row_dicts)} matching records with database evidence.",
                f"Top record matches: {', '.join(names[:3])}.",
            ]
            if top_value is not None:
                highlights.append(f"Top match total value is {top_value}.")
            if latest_value is not None:
                if latest_period:
                    highlights.append(f"Latest observed value is {latest_value} for period starting {latest_period}.")
                else:
                    highlights.append(f"Latest observed value is {latest_value}.")
            return highlights

    if intent == "trend":
        time_key = next((key for key in top_row if "year" in key.lower() or "date" in key.lower()), None)
        value_key = next((key for key in top_row if "total" in key.lower() or "count" in key.lower()), None)
        if time_key and value_key:
            return [
                f"Trend contains {len(row_dicts)} time points.",
                f"First point: {row_dicts[0][time_key]} -> {row_dicts[0][value_key]}.",
                f"Last point: {row_dicts[-1][time_key]} -> {row_dicts[-1][value_key]}.",
            ]

    if intent == "definition_lookup":
        names = [row.get("name") for row in row_dicts[:5] if row.get("name")]
        if names:
            highlights = [
                f"Top database matches for '{question}' are: {', '.join(names[:3])}.",
                f"Returned {len(row_dicts)} matched indicators for downstream explanation.",
            ]
            if top_row.get("total_value") is not None:
                highlights.append(f"Top match total value is {top_row['total_value']}.")
            if top_row.get("latest_value") is not None:
                latest_period = top_row.get("latest_period_start")
                if latest_period:
                    highlights.append(
                        f"Top match latest observed value is {top_row['latest_value']} for period starting {latest_period}."
                    )
                else:
                    highlights.append(f"Top match latest observed value is {top_row['latest_value']}.")
            return highlights

    if "name" in top_row and any(key in top_row for key in ("total_value", "count")):
        metric_key = "total_value" if "total_value" in top_row else "count"
        return [f"Top result is {top_row['name']} with {metric_key}={top_row[metric_key]}."]

    return [f"Returned {len(row_dicts)} rows from the database."]


def build_analytics_payload(question, query_analysis, sql_query, columns, results):
    source_columns = columns or []
    source_row_dicts = _rows_to_dicts(source_columns, results)
    focused_columns, row_dicts = _focus_row_dicts(query_analysis, source_row_dicts)
    numeric_columns = _detect_numeric_columns(row_dicts)
    time_columns = _detect_time_columns(focused_columns)

    payload = {
        "question": question,
        "intent": query_analysis.get("intent"),
        "keywords": query_analysis.get("keywords", []),
        "years": query_analysis.get("years", []),
        "sql": sql_query,
        "source_columns": source_columns,
        "columns": focused_columns,
        "row_count": len(results) if isinstance(results, list) else 0,
        "numeric_columns": numeric_columns,
        "time_columns": time_columns,
        "rows": row_dicts,
        "preview_rows": row_dicts[:10],
        "highlights": _build_highlights(question, query_analysis, row_dicts),
        "truncated": isinstance(results, list) and len(results) > len(row_dicts),
    }

    return payload


def build_analytics_context(payload):
    lines = [
        "Question:",
        payload["question"],
        "",
        "Intent:",
        payload.get("intent", "unknown"),
        "",
        "Keywords:",
        ", ".join(payload.get("keywords", [])) or "(none)",
        "",
        "SQL Used:",
        payload.get("sql", ""),
        "",
        "Analytics Columns:",
        ", ".join(payload.get("columns", [])) or "(none)",
        "",
        "Highlights:",
    ]

    if payload.get("source_columns") and payload.get("source_columns") != payload.get("columns"):
        lines.extend(
            [
                "",
                "Source Columns:",
                ", ".join(payload.get("source_columns", [])),
            ]
        )

    for highlight in payload.get("highlights", []):
        lines.append(f"- {highlight}")

    lines.extend(
        [
            "",
            "Preview Rows (JSON):",
            json.dumps(payload.get("preview_rows", []), indent=2),
        ]
    )

    if payload.get("truncated"):
        lines.extend(
            [
                "",
                "Note:",
                "Only the first 200 rows are included in the analytics payload.",
            ]
        )

    return "\n".join(lines).strip()
