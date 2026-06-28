# ex04.py
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

def get_ai_answer_stream(messages):
    prompt = [
        {
            "role": "system",
            "content": "답변은 한국어로 한 문장으로 답하세요.",
        },
        {
            "role": "user",
            "content": messages[-1]["content"],
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
        "max_new_tokens": 100,
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
        
st.title("무엇이든 물어보살!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("질문을 입력하세요."):
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("assistant"):
        answer = st.write_stream(
            get_ai_answer_stream(st.session_state.messages)
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })