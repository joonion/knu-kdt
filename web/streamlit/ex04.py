# ex04.py
import streamlit as st
import seaborn as sns
import altair as alt

st.title("입력 위젯을 이용한 Penguins 데이터 탐색")

penguins = sns.load_dataset("penguins")
penguins = penguins.dropna()

st.header("1. 원본 데이터")
st.dataframe(penguins)

st.header("2. 입력 위젯으로 데이터 필터링")

species_list = penguins["species"].unique().tolist()
island_list = penguins["island"].unique().tolist()

selected_species = st.selectbox(
    "펭귄 종을 선택하세요.",
    species_list
)

selected_island = st.multiselect(
    "섬을 선택하세요.",
    island_list,
    default=island_list
)

min_body_mass, max_body_mass = st.slider(
    "몸무게 범위를 선택하세요.",
    min_value=int(penguins["body_mass_g"].min()),
    max_value=int(penguins["body_mass_g"].max()),
    value=(
        int(penguins["body_mass_g"].min()),
        int(penguins["body_mass_g"].max())
    )
)

show_table = st.checkbox("필터링된 데이터 보기", value=True)

filtered = penguins[
    (penguins["species"] == selected_species) &
    (penguins["island"].isin(selected_island)) &
    (penguins["body_mass_g"] >= min_body_mass) &
    (penguins["body_mass_g"] <= max_body_mass)
]

st.header("3. 필터링 결과")

st.write(f"선택된 데이터 개수: **{len(filtered)}개**")

if show_table:
    st.dataframe(filtered)

st.header("4. 산점도")

chart = alt.Chart(filtered).mark_circle(size=70).encode(
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