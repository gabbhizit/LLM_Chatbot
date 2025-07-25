import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_core.documents import Document
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import Runnable
import streamlit as st

openai_key = st.secrets["OPENAI_API_KEY"]

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
    chain = prompt | llm
    response = chain.invoke({"context": docs_page_content, "query": query})

    response = response.replace("\n", " ").strip()
    return response
#print(create_vector_db_from_pdf("Health_Report.pdf"))  # Example usage, can be removed later
def pdf_agent(pdf_path, q=None):
    db = create_vector_db_from_pdf(pdf_path)
    llm=OpenAI(temperature=0.7, max_tokens=64)

    pdf_tool = Tool(
        name="pdf_tool",
        func=lambda q: get_response_from_query(db, q),
        description="Use this tool to answer questions about the PDF document."
    )
    """"duckduckgo_tool = Tool(
    name="DuckDuckGo Search",
    func=DuckDuckGoSearchRun().run,
    description="Use this tool to search the web using DuckDuckGo"
    )"""
    standard_tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    tools = [pdf_tool]+ standard_tools #+ [duckduckgo_tool]
    agent = initialize_agent(
        tools,
        llm,
        agent_type="zero-shot-react-description",
        verbose=True
    )
    return agent

#print(pdf_agent("Health_Report.pdf").run("What is the patient's biological age?"))  # Example usage, can be removed later