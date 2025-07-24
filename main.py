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
    db = pdfh.create_vector_db_from_pdf(pdf_path)
    response = pdfh.get_response_from_query(db, query)
    st.write(response)


