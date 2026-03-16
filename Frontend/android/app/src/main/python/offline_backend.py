import json
import os

from java import jclass


def _serialize_query_error(exc: Exception):
    raw_detail = str(exc).strip() or exc.__class__.__name__
    normalized = raw_detail.strip("'\"")

    if isinstance(exc, KeyError) or normalized in {
        "total_value",
        "startdate",
        "period",
        "orgunit_name",
        "dataelement_name",
    }:
        return {
            "detail": "I couldn't summarize that request cleanly. Try rephrasing the question or ask for the raw table.",
            "technical_detail": raw_detail,
            "error_code": "summary_shape_error",
        }

    if "SQLite database not found" in raw_detail:
        return {
            "detail": "The offline database is missing on this device.",
            "technical_detail": raw_detail,
            "error_code": "database_missing",
        }

    if "No question provided" in raw_detail:
        return {
            "detail": "Please enter a question before sending.",
            "technical_detail": raw_detail,
            "error_code": "empty_question",
        }

    return {
        "detail": "I couldn't process that request. Please try again or rephrase the question.",
        "technical_detail": raw_detail,
        "error_code": "query_failed",
    }


def run_query(
    question: str,
    debug: bool = False,
    page: int = 1,
    page_size: int = 200,
    include_insights: bool = False,
    include_rows: bool = True,
):
    try:
        from query_engine import answer_question

        result = answer_question(
            question=question.strip(),
            debug=debug,
            page=page,
            page_size=page_size,
            include_insights=include_insights,
            include_rows=include_rows,
        )
        return json.dumps(result)
    except Exception as e:
        return json.dumps(_serialize_query_error(e))

def run_llm(prompt: str) -> str:
    from com.chaquo.python import Python

    context = Python.getPlatform().getApplication().getApplicationContext()
    LlmService = jclass("com.healthworker.assistant.LlmService")
    service = LlmService(context)
    return service.infer(prompt)


def get_db_path():
    from com.chaquo.python import Python

    context = Python.getPlatform().getApplication().getApplicationContext()
    files_dir = context.getFilesDir().getAbsolutePath()
    return os.path.join(files_dir, "databases", "dhis2.sqlite")
