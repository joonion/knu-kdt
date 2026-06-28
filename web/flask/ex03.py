# ex03.py
from flask import Flask, request
app = Flask(__name__)

@app.route("/query")
def query_example():
    param = request.args.get("param")
    return f"Requested param: {param}"