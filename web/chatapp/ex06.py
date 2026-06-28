# ex06.py
from transformers import TextIteratorStreamer
from threading import Thread

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def build_vectorstore(filepath, embeddings):
    loader = PyMuPDFLoader(filepath)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    split_docs = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
    )

    return vectorstore

def search_docs(question, vectorstore, k=3):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever.invoke(question)
    
def get_ai_answer_stream_rag(question, docs, tokenizer, model):
    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    messages = [
        {
            "role": "system",
            "content": (
                "당신은 PDF 문서를 기반으로 답변하는 RAG 챗봇입니다. "
                "반드시 제공된 문서 내용만 근거로 한국어로 답변하세요. "
                "문서에서 답을 찾을 수 없으면 '문서에서 해당 내용을 찾을 수 없습니다.'라고 답하세요."
            ),
        },
        {
            "role": "user",
            "content": f"""
다음은 PDF 문서에서 검색된 내용입니다.

[문서 내용]
{context}

[질문]
{question}
""",
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs,
    )
    thread.start()

    for token in streamer:
        yield token

import streamlit as st 
import tempfile

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def load_llm():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return tokenizer, model, embeddings

tokenizer, model, embeddings = load_llm()

st.title("PDF 기반 RAG 챗봇 서비스")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

pdf_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])

if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    with st.spinner("PDF 문서를 분석 중입니다..."):
        st.session_state.vectorstore = build_vectorstore(tmp_file_path, embeddings)
    st.success("PDF 파일 분석 완료!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if question := st.chat_input("PDF 내용에 대해 질문하세요."):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        docs = search_docs(
            question,
            st.session_state.vectorstore,
            k=3,
        )
        
        with st.chat_message("assistant"):
            answer = st.write_stream(
                get_ai_answer_stream_rag(
                    question, 
                    docs,
                    tokenizer,
                    model
                )
            )
            with st.expander("벡터 스토어에서 검색된 문서 보기"):
                for i, doc in enumerate(docs, start=1):
                    st.markdown(f"### 검색 결과 {i}")
                    st.markdown(doc.page_content)
        st.session_state.messages.append({"role": "assistant", "content": answer})

