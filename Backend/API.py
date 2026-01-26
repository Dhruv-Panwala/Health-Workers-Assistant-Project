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

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/query")
async def query(req: QueryRequest):
    try:
        return answer_question(req.question, debug=req.debug)
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))
