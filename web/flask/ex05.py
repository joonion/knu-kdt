# ex05.py
from flask import Flask, render_template
app = Flask(__name__)

@app.route("/hello/<name>")
def hello_name(name):
    return render_template("hello.html", name=name)