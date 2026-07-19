# rag1/api.py
from rag import get_answer

from fastapi import FastAPI, Form
api = FastAPI()

@api.post("/ask")
async def ask(question: str = Form(...)) -> dict[str, str]:
    print("/ask")
    answer = get_answer(question)
    return {
        "answer": answer,
    }
