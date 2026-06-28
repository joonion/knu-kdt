# qna/app.py
from dotenv import load_dotenv
from huggingface_hub import login, whoami

load_dotenv()
login()
print(f"Hugging Face Login: {whoami()['name']}")

from transformers import pipeline
qna = pipeline(
    task="text-generation", 
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=100,
    do_sample=False,
    return_full_text=False,
)
print(f"Pipeline Model: {qna.model.name_or_path}")

context = """
경북대학교에서 운영하는 KDT14기 교육과정은 경일대학교에서 강의를 진행합니다.
주니온 교수는 전이학습 기반 모델, 웹 기반, 클라우드 기반 AI 서비스 과목을 강의합니다.
"""
print(f"Context: {context}")

from flask import Flask, request, render_template
app = Flask(__name__)
app.secret_key = "my secret key"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")

    if not question:
        return render_template(
            "answer.html",
            question=question,
            answer="질문을 입력하세요."
        )
    
    prompt = [
        {"role": "system", "content": "문맥만 이용하여 한 문장으로 답하세요."},
        {"role": "user", "content": f"문맥: {context}, 질문: {question}"}
    ]
        
    result = qna(prompt)

    return render_template(
        "answer.html", 
        question=question, 
        answer=result[0]["generated_text"].strip()
    )