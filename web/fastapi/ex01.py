# ex01.py
from fastapi import FastAPI
app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id):
    return {"item_id": item_id}

@app.get("/items/")
def read_items(skip=0, limit=10):
    return {"skip": skip, "limit": limit}
