# ex09.py
import streamlit as st

st.title("위젯 상태와 세션 상태")

if "name" not in st.session_state:
    st.session_state.name = ""

st.text_input("이름을 입력하세요", key="name")

st.write("입력한 이름:", st.session_state.name)
st.write("현재 Session State:")
st.write(st.session_state)