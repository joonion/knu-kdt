# api.py
from rag import get_answer_rag, add_pdf_to_vectorstore
from pathlib import Path
import shutil

from fastapi import FastAPI, Form, UploadFile, File
api = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@api.post("/ask")
async def ask(question: str = Form(...)) -> dict[str, str]:
    answer, context = get_answer_rag(question)
    return {
        "answer": answer,
        "context": context,
    }

@api.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, str]:
    pdf_path = UPLOAD_DIR / file.filename
    
    with pdf_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    add_pdf_to_vectorstore(str(pdf_path))

    return {
        "message": "PDF 업로드 및 벡터 DB 생성이 완료되었습니다.",
        "filename": file.filename,
    }