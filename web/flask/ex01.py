# ex01.py
from flask import Flask
app = Flask(__name__)

@app.route("/user/<username>")
def show_username(username):
    return f"User: {username}"