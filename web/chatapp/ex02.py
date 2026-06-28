# ex02.py
import streamlit as st 

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
    
    answer = f"와, 너 **핵심**을 *콕* 찔렀어!"
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})    
