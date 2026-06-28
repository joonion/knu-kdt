# rag.py
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

PDF_PATH = "kdt.pdf"
DB_PATH = "chroma_db"

chatbot = None
vectorstore = None

def get_chatbot():
    global chatbot

    if chatbot is None:
        chatbot = pipeline(
            task="text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            return_full_text=False,
        )
    
    return chatbot

def get_vectorstore():
    global vectorstore
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    if vectorstore is None:
        if Path(DB_PATH).exists():
            vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        else:
            loader = PyPDFLoader(PDF_PATH)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            split_docs = splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(
                documents=split_docs, 
                embedding=embeddings,
                persist_directory=DB_PATH
            )

    return vectorstore

def get_answer_rag(question: str) -> tuple[str, str]:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chatbot = get_chatbot()

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )
        
    def generate(inputs):
        prompt = [
            {
                "role": "system",
                "content": """
                    사용자의 질문에 대해 한국어로 한 문장으로 답변하세요.
                    반드시 제공된 문서 내용만 근거로 답변하세요.
                    제공된 문서 내용에서 답을 찾을 수 없으면, '모르겠습니다'라고 답변하세요.
                """
            },
            {
                "role": "user",
                "content": f"[문서 내용] {inputs['context']} [질문] {inputs['question']}"
            },
        ]

        result = chatbot(prompt, max_new_tokens=100, do_sample=False)
        return str(result[0]['generated_text'])
            
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | RunnablePassthrough.assign(answer=RunnableLambda(generate))
    )
    
    result = rag_chain.invoke(question)
    return str(result["answer"]), str(result["context"])
