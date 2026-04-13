import json
import os

from New import answer_question


def configure_runtime(
    db_path,
    sql_model_path,
    chat_model_path,
    use_android_bridge=False,
):
    os.environ["SQLITE_DB_PATH"] = str(db_path)
    os.environ["LLM_MODEL_PATH"] = str(sql_model_path)
    os.environ["CHAT_MODEL_PATH"] = str(chat_model_path)
    os.environ["ANDROID_LLM_BRIDGE"] = "1" if use_android_bridge else "0"


def answer_question_payload(
    question,
    debug=False,
    page=1,
    page_size=100,
    include_insights=False,
    include_rows=True,
    include_debug_trace=False,
    prefer_sqlcoder_first=False,
    resolved_plan=None,
):
    return answer_question(
        question=question,
        debug=bool(debug),
        page=int(page),
        page_size=int(page_size),
        include_insights=bool(include_insights),
        include_rows=bool(include_rows),
        include_debug_trace=bool(include_debug_trace),
        prefer_sqlcoder_first=bool(prefer_sqlcoder_first),
        resolved_plan=resolved_plan,
    )


def answer_question_json(
    question,
    debug=False,
    page=1,
    page_size=100,
    include_insights=False,
    include_rows=True,
    include_debug_trace=False,
    prefer_sqlcoder_first=False,
    resolved_plan=None,
):
    result = answer_question_payload(
        question=question,
        debug=debug,
        page=page,
        page_size=page_size,
        include_insights=include_insights,
        include_rows=include_rows,
        include_debug_trace=include_debug_trace,
        prefer_sqlcoder_first=prefer_sqlcoder_first,
        resolved_plan=resolved_plan,
    )
    return json.dumps(result)
