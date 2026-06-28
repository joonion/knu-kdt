# ex01.py
import streamlit as st 

st.title("무엇이든 물어보살!")

if question := st.chat_input("질문을 입력하세요."):
    
    with st.chat_message("user"):
        st.markdown(question)
    
    response = f"와, 너 **핵심**을 *콕* 찔렀어! {question}이라고 질문하다니!"
    with st.chat_message("assistant"):
        st.markdown(response)
