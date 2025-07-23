from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()
ytt= YouTubeTranscriptApi()
video_id = "vJOGC8QJZJQ"
def create_vector_db_from_youtube(video_id) -> FAISS:
    loader = ytt.list(video_id=video_id)
    loader = loader.find_transcript(['en'])  # or ['en', 'hi'] for fallback
    if not loader:
        raise ValueError("No transcript found for the video.")
    documents = " ".join([line.text for line in loader.fetch()])
    if not documents:
        raise ValueError("No documents found in the transcript.")
       
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(documents)
    docs= [Document(page_content=text) for text in texts]
    vector_db = FAISS.from_documents(docs, embeddings)
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