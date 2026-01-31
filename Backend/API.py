from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from New import answer_question

app = FastAPI(title="Text2SQL API")

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
    page_size: int = 200

@app.post("/query")
async def query_endpoint(payload: QueryRequest):
    try:
        result = answer_question(
            question=payload.question.strip(),
            debug=payload.debug,
            page=payload.page,
            page_size=payload.page_size
        )
        return result
    except Exception as e:
        return {"detail": str(e)}
