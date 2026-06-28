# ex06.py
import streamlit as st

st.title("세션 상태가 있는 카운터")

if "count" not in st.session_state:
    st.session_state.count = 0
    print(f"카운터가 초기화되었습니다.")
    
if st.button("Increment"):
    st.session_state.count += 1
    print(st.session_state.count)

st.write("Count =", st.session_state.count)