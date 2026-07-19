# ex05.py
from fastapi import FastAPI
app = FastAPI()

import asyncio

async def fetch_data():
    await asyncio.sleep(2)
    return {"data": "some_data"}

@app.get("/")
async def read_root():
    data = await fetch_data()
    return {"message": "Hello, World!", "data": data}
