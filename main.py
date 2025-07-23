import langchain_helper as lch
import streamlit as st

st.title("Cute Pet Name Generator")

animal_type = st.sidebar.selectbox(
    "Select an animal type",("cat", "dog", "cow", "pig", "sheep"))

if animal_type:
    with st.spinner("Generating cute pet names..."):
        names = lch.gen_pet_name(animal_type)
        st.write(f"Here are some cute {animal_type} names:")
        st.write(names)