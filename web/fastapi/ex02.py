# ex02.py
from fastapi import FastAPI
app = FastAPI()

todos = {}
next_id = 1

@app.get("/todos")
def get_todos():
    return todos

@app.post("/todos")
def create_todo(title: str):
    global next_id

    todo = {
        "id": next_id,
        "title": title,
        "completed": False
    }
    todos[next_id] = todo
    next_id += 1

    return todo

@app.put("/todos/{todo_id}")
def update_todo(todo_id: int, completed: bool):
    if todo_id not in todos:
        return {"error": "Todo not found"}

    todos[todo_id]["completed"] = completed

    return todos[todo_id]

@app.delete("/todos/{todo_id}")
def delete_todo(todo_id: int):
    if todo_id not in todos:
        return {"error": "Todo not found"}

    deleted = todos.pop(todo_id)

    return {
        "message": "Todo deleted",
        "todo": deleted
    }