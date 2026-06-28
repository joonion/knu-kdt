# ex05.py
from pypdf import PdfReader

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

@st.cache_resource
def load_model():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    return tokenizer, model

tokenizer, model = load_model()

def get_ai_answer_stream(question, pdf_text):
    context = pdf_text[:3000]

    prompt = [
        {
            "role": "system",
            "content": (
                "당신은 PDF 문서를 바탕으로 답변하는 AI 상담사입니다."
                "반드시 제공된 문서 내용에 근거하여 한국어로 답변하세요."
                "문서에 없는 내용은 모른다고 답하세요."
            ),
        },
        {
            "role": "user",
            "content": f"""
다음은 PDF 문서에서 추출한 내용입니다.

[문서 내용]
{context}

[질문]
{question}
""",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 200,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }

    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs,
        daemon=True,
    )
    thread.start()

    for text in streamer:
        yield text

st.title("PDF 파일에 대해 답해드립니다.")

uploaded_file = st.file_uploader(
    "PDF 파일을 업로드하세요.",
    type=["pdf"],
)

if uploaded_file is None:
    st.info("먼저 PDF 파일을 업로드하세요.")

else:
    pdf_text = extract_pdf_text(uploaded_file)
    st.success("PDF 파일을 읽었습니다.")

    with st.expander("추출된 PDF 텍스트 확인"):
        st.text(pdf_text[:3000])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("PDF 내용에 대해 질문하세요."):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            answer = st.write_stream(
                get_ai_answer_stream(question, pdf_text)
            )
        st.session_state.messages.append({"role": "assistant", "content": answer})
