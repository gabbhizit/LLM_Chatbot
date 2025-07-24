import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def create_vector_db_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query,k=4):
    docs = db.similarity_search(query,k)
    docs_page_content = " ".join([doc.page_content for doc in docs])
    llm = OpenAI(temperature=0.7, max_tokens=64)
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template="Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(context=docs_page_content, query=query)
    response = response.replace("\n", " ").strip()
    return response