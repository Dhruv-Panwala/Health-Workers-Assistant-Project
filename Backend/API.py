from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from query_engine import answer_question, build_conn_str_from_parts, preheat_database
from contextlib import asynccontextmanager
# from rag import build_rag_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase
    conn_str = build_conn_str_from_parts()
    if conn_str:
        preheat_database(conn_str)
        # build_rag_cache(conn_str)
    yield  # ⇦ FastAPI will start serving API after this line

    # Shutdown phase (optional cleanup)
    
app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://10.0.2.2:8000",
        "http://127.0.0.1:8000",
        "capacitor://localhost",
        "ionic://localhost",
        "https://health-worker-assistant.netlify.app",
        "https://health-workers-assistant.netlify.app",
    ],
    allow_origin_regex=r"^https?://(10\.0\.2\.2|127\.0\.0\.1|localhost|192\.168\.\d+\.\d+)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def query_endpoint(payload: dict):
    try:
        question = str(payload.get("question", "")).strip()
        result = answer_question(
            question=question,
            debug=bool(payload.get("debug", False)),
            page=int(payload.get("page", 1)),
            page_size=int(payload.get("page_size", 200)),
            include_insights=bool(payload.get("include_insights", False)),
            include_rows=bool(payload.get("include_rows", True)),
        )
        return result
    except Exception as e:
        return {"detail": str(e)}
