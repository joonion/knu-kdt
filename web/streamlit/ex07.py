# ex07.py
import streamlit as st

st.title("콜백을 활용한 카운터")

if "count" not in st.session_state:
    st.session_state.count = 0

def increment_counter():
    st.session_state.count += 1
    
def decrement_counter():
    st.session_state.count -= 1

st.button("Increment", on_click=increment_counter)
st.button("Decrement", on_click=decrement_counter)

st.write("Count =", st.session_state.count)