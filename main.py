import langchain_helper as lch
import youtube_helper as yth
import textwrap
import streamlit as st

st.title("Youtube Video Q&A")
with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label="Enter YouTube Video URL",
            max_chars=100,
        )
        query = st.sidebar.text_input(
            label="Ask a question about the video",
            max_chars=50,
            key='query'
        )
        submit_button = st.form_submit_button(label='Submit')

if submit_button and youtube_url and query:
    db = yth.create_vector_db_from_youtube(youtube_url)
    response= yth.get_response_from_query(db,query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))




# Uncomment the following lines to run the langchain helper functionality
"""st.title("Cute Pet Name Generator")

animal_type = st.sidebar.selectbox(
    "Select an animal type",("cat", "dog", "cow", "pig", "sheep"))

if animal_type:
    with st.spinner("Generating cute pet names..."):
        names = lch.gen_pet_name(animal_type)
        st.write(f"Here are some cute {animal_type} names:")
        st.write(names)"""