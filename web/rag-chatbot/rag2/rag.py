# rag2/rag.py
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "kdt.pdf"

chatbot = None
context = None

def get_chatbot():
    global chatbot

    if chatbot is None:
        chatbot = pipeline(
            task="text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            return_full_text=False,
        )
    
    return chatbot

def get_context():
    global context
    
    if context is None:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        context = "\n\n".join(
            doc.page_content for doc in documents
        )
    
    return context

def get_answer_rag(question: str) -> tuple[str, str]:
    context = get_context()

    prompt = [
        {
            "role": "system",
            "content": """
                사용자의 질문에 대해 한국어로 한 문장으로 답변하세요.
                반드시 제공된 문서 내용만 근거로 답변하세요.
                제공된 문서 내용에서 답을 찾을 수 없으면, '모르겠습니다'라고 답변하세요.
            """
        },
        {
            "role": "user",
            "content": f"[문서 내용] {context} [질문] {question}"
        },
    ]

    chatbot = get_chatbot()
    result = chatbot(prompt, max_new_tokens=100, do_sample=False)

    return str(result[0]['generated_text']), str(context)
    