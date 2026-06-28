# ex08.py
from flask import Flask, session
app = Flask(__name__)

app.secret_key = "my secret key"

@app.route("/set_session/<name>")
def set_session(name):
    session["username"] = name
    return "세션에 사용자 이름이 설정되었습니다."

@app.route("/get_session")
def get_session():
    username = session.get("username")
    if username:
        return f"사용자 이름: {username}"
    else:
        return f"현재 세션이 설정되지 않았습니다."
    