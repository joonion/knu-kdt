# ex03.py
import streamlit as st
import seaborn as sns
import altair as alt

st.title("Penguins 데이터셋 시각화")

penguins = sns.load_dataset("penguins")

tab1, tab2, tab3 = st.tabs(["데이터 보기", "산점도 보기", "종별 시각화"])

with tab1:
    st.header("Penguins 데이터셋")
    st.dataframe(penguins)

with tab2:
    st.header("부리 길이와 부리 깊이의 관계")

    chart = alt.Chart(penguins).mark_circle(size=60).encode(
        x=alt.X(
            "bill_length_mm:Q",
            scale=alt.Scale(zero=False),
            title="Bill length (mm)"
        ),
        y=alt.Y(
            "bill_depth_mm:Q",
            scale=alt.Scale(zero=False),
            title="Bill depth (mm)"
        ),
    ).interactive()

    st.altair_chart(chart, width='stretch')

with tab3:
    st.header("펭귄의 종별 시각화")

    chart = alt.Chart(penguins).mark_circle(size=60).encode(
        x=alt.X(
            "bill_length_mm:Q",
            scale=alt.Scale(zero=False),
            title="Bill length (mm)"
        ),
        y=alt.Y(
            "bill_depth_mm:Q",
            scale=alt.Scale(zero=False),
            title="Bill depth (mm)"
        ),
        color="species:N",
        tooltip=["species", "bill_length_mm", "bill_depth_mm"]
    ).interactive()

    st.altair_chart(chart, width='stretch')