# rag.py
from transformers import pipeline

chatbot = None

def get_chatbot():
    global chatbot

    if chatbot is None:
        chatbot = pipeline(
            task="text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            return_full_text=False,
        )
    
    return chatbot

def get_answer(question: str) -> str:
    prompt = [
        {
            "role": "system",
            "content": "사용자의 질문에 대해 한국어로 한 문장으로 답변하세요.",
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    chatbot = get_chatbot()
    result = chatbot(prompt, max_new_tokens=100, do_sample=False)

    return str(result[0]['generated_text'])
