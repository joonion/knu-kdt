# ex04.py
from flask import Flask, jsonify
app = Flask(__name__)

@app.route("/json")
def json_example():
    return jsonify({"message": "Hello, Flask!"})