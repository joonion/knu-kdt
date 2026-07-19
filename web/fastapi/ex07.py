# ex07.py
from fastapi import FastAPI
app = FastAPI()

from fastapi import File, UploadFile
from pathlib import Path
import shutil

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)    

    return {"message": f"파일 업로드 완료: {file_path}"}