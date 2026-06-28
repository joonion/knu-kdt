# ex02.py
import streamlit as st 
import seaborn as sns 

penguins = sns.load_dataset("penguins")

st.title("팔머 펭귄 데이터셋")

st.header("1. 데이터 프레임")
st.dataframe(penguins)

st.header("2. 부리의 길이-깊이 산점도")
st.write("팔머 펭귄의 부리 길이와 부리 깊이는 상관이 높은가?")
st.scatter_chart(
    data = penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
)
