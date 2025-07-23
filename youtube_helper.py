from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings= OpenAI(temperature=0.7, max_tokens=64)
video_url = "https://www.youtube.com/watch?v=vJOGC8QJZJQ"
def create_vector_db_from_youtube(video_url) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    
    vector_db = FAISS.from_documents(texts, embeddings)
    return vector_db

def get_response_from_query(db,query,k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([doc.page_content for doc in docs])
    llm = OpenAI(temperature=0.7, max_tokens=64)
    
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template="Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    # Run the chain with the context and query
    # The response will be a string with the answer
    response = chain.run(context=docs_page_content, query=query)
    response = response.replace("\n", " ").strip()
    
    return response