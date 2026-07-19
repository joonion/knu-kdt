# rag1/app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("RAG 챗봇 서비스")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if question := st.chat_input("질문을 입력하세요."):
    with st.chat_message("user"):
        st.markdown(question)
        
    st.session_state.messages.append({"role": "user", "content": question})
    
    response = requests.post(f"{API_URL}/ask", data={"question": question})
    if response.status_code == 200:
        answer = response.json()["answer"]
    else:
        answer = "답변 생성 중 오류가 발생했습니다."
        print(response.status_code)
    
    with st.chat_message("assistant"):
        st.markdown(answer)
        
    st.session_state.messages.append({"role": "assistant", "content": answer})
    