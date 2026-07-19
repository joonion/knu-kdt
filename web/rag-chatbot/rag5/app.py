# rag5/app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("RAG 챗봇 서비스")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("지식 관리")
    
    upload_file = st.file_uploader("PDF 업로드", type=['pdf'])
    if st.button("벡터 DB에 추가"):
        if upload_file is None:
            st.warning("먼저 PDF 파일을 업로드하세요.")
        else:
            with st.spinner("PDF를 벡터 DB에 추가 중입니다..."):
                files = {"file": (upload_file.name, upload_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success("PDF가 벡터 DB에 추가되었습니다.")
                    st.write(f"파일명: {result['filename']}")
                else:
                    st.error("PDF 처리 중 오류가 발생했습니다.")

st.subheader("묻고 답하기")

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
    