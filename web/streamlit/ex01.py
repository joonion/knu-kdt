# ex01.py
import streamlit as st

st.title("Streamlit 텍스트 출력 실습")
st.write("이번 실습에서는 Streamlit에서 텍스트를 출력하는 다양한 방법을 배웁니다.")
st.divider()

st.header("1. 일반 텍스트 출력")
st.text("st.text()는 가장 단순한 텍스트 출력 함수입니다.")
st.write("st.write()는 문자열, 숫자, 데이터프레임 등 다양한 객체를 출력할 수 있습니다.")
st.header("2. Markdown 출력")
st.markdown("""
Streamlit에서는 **Markdown 문법**을 사용할 수 있습니다.
예를 들어, 다음과 같은 표현이 가능합니다.
- **굵은 글씨**
- *기울임 글씨*
- `코드 표현`
- [Streamlit 공식 문서 링크](https://docs.streamlit.io/)
""")

st.header("3. 코드와 수식 출력")

code = """
import streamlit as st
st.title("Hello, Streamlit!")
"""
st.code(code, language="python")

st.write("다음은 소프트맥스 함수입니다.")
st.latex(r"""
    p_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
""")

st.divider()
st.success("Streamlit을 이용하면 간단한 웹 페이지를 빠르게 만들 수 있습니다.")