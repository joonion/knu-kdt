# ex06.py
from fastapi import FastAPI
app = FastAPI()

import time, asyncio

@app.get("/sync")
def read_item_sync():
    start_time = time.time()
    time.sleep(1)
    elapsed_time = time.time() - start_time
    return {"type": "sync/", "elapsed_time": elapsed_time}

@app.get("/async")
async def read_item_async():
    start_time = time.time()
    await asyncio.sleep(1)
    elapsed_time = time.time() - start_time
    return {"type": "async/", "elapsed_time": elapsed_time}

