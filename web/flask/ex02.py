# ex02.py
from flask import Flask, request
app = Flask(__name__)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return "GET request"
    else:
        return "POST request"
