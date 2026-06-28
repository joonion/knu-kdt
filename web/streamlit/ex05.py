# ex05.py
import streamlit as st

st.title("세션 상태가 없는 카운터")

count = 0
print(f"카운터가 초기화되었습니다.")
    
if st.button("Increment"):
    count += 1
    print(count)

st.write("Count =", count)