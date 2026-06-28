# app.py
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
        answer = response.json()['answer']
        context = response.json()['context']
    else:
        answer = "답변 생성 중 오류가 발생했습니다."
    
    with st.chat_message("assistant"):
        st.markdown(answer)
        if response.status_code == 200:
            with st.expander("컨텍스트 정보 확인"):
                st.markdown(f"**컨텍스트 정보**")
                st.markdown(context)
                
    st.session_state.messages.append({"role": "assistant", "content": answer})
    