# rag3/api.py
from rag import get_answer_rag

from fastapi import FastAPI, Form
api = FastAPI()

@api.post("/ask")
async def ask(question: str = Form(...)) -> dict[str, str]:
    answer, context = get_answer_rag(question)
    return {
        "answer": answer,
        "context": context,
    }
