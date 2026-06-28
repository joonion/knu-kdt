# ex08.py
import streamlit as st

st.title("입력값이 있는 콜백 함수")

if "count" not in st.session_state:
    st.session_state.count = 0

def increment_counter(value):
    st.session_state.count += value
    
def decrement_counter(value):
    st.session_state.count -= value

size = st.number_input("얼마나?", value=1, step=1)

st.button("Increment", on_click=increment_counter, args=(size,))
st.button("Decrement", on_click=decrement_counter, args=(size,))

st.write("Count =", st.session_state.count)