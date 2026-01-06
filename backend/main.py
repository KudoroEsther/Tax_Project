from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional, List
from tax_reform_using_utils import query_agent


load_dotenv()

app = FastAPI(title="TaxGPT", version="1.0.0")

class ChatRequest(BaseModel):
    req: str

class Sources(BaseModel):
    title: str
    page: int
    excerpt: str

class ChatResponse(BaseModel):
    res: str
    source:Optional[Sources[List]]


@app.post("/ask", response_model=ChatResponse)
def ask_question(payload: ChatRequest):
    try:
        result = query_agent(payload.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate answer")