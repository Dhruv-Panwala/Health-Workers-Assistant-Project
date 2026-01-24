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
    ],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    debug: bool = False

@app.post("/query")
def query(req: QueryRequest):
    try:
        return answer_question(req.question, debug=req.debug)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
