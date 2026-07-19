# ex03.py
from fastapi import FastAPI
app = FastAPI()

import time
from fastapi.responses import StreamingResponse

def stream_generator():
    for i in range(10):
        yield f"data chunk {i}: [{time.strftime('%H:%M:%S')}]\n"
        time.sleep(0.5)

@app.get("/stream")
def get_answer_stream():
    return StreamingResponse(
        content=stream_generator(),
        media_type="text/plain"
    )
