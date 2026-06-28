# ex06.py
from flask import Flask, render_template
app = Flask(__name__)

@app.route("/messages/<name>")
def show_messages(name):
    subjects = ["전이 학습 기반 모델", "웹 기반 AI 서비스", "클라우드 기반 AI 서비스"]
    return render_template("messages.html", name=name, subjects=subjects)