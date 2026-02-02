from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from New import answer_question, build_conn_str_from_parts, preheat_database
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase
    conn_str = os.environ.get("DHIS2_DSN", "").strip() or build_conn_str_from_parts()
    if conn_str:
        print("🔥 Preheating PostgreSQL…")
        preheat_database(conn_str)

    yield  # ⇦ FastAPI will start serving API after this line

    # Shutdown phase (optional cleanup)
    print("🛑 API shutting down…")

app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://health-worker-assistant.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    debug: bool = False
    page: int = 1
    page_size: int = 200,
    include_insights: bool = False,
    include_rows: bool=True

@app.post("/query")
async def query_endpoint(payload: QueryRequest):
    try:
        result = answer_question(
            question=payload.question.strip(),
            debug=payload.debug,
            page=payload.page,
            page_size=payload.page_size,
            include_insights=payload.include_insights,
            include_rows=payload.include_rows,
        )
        return result
    except Exception as e:
        return {"detail": str(e)}
