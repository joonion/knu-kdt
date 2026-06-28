# ex03.py
import streamlit as st 
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline(
        task="text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens=100,
        do_sample=False,
        return_full_text=False,
    )

chatbot = load_model()

def get_ai_answer(messages):
    prompt = [
        {
            "role": "system",
            "content": "답변은 한국어로 한 문장으로 답하세요.",
        },
        {
            "role": "user",
            "content": messages[-1]['content'],
        },
    ]

    result = chatbot(prompt)
    return result[0]["generated_text"]

st.title("무엇이든 물어보살!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if question := st.chat_input("질문을 입력하세요."):
    
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    
    answer = get_ai_answer(st.session_state.messages)
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})    
