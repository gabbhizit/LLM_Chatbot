import langchain_helper as lch
import youtube_helper as yth
import pdf_helper as pdfh
import streamlit as st
import tempfile

st.title("PDF Chatbot")

with st.form("pdf_query_form"):
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    query = st.text_input("Ask a question about the PDF:")
    submit = st.form_submit_button("Submit")

if submit and uploaded_file is not None and query:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded successfully!")
    agent = pdfh.pdf_agent(pdf_path, query)  # Initialize agent
    response = agent.run(query)              # Ask your query
    st.markdown("### 🤖 Response")
    st.write(response)


