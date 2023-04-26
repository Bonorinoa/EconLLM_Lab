import streamlit as st
import pandas as pd
from utils import chat_with_author, summarize_book
import time
from io import StringIO
    

if "summaries" not in st.session_state:
    st.session_state["summaries"] = []
    
    
st.title("EconGPT")

st.sidebar.title("Inputs")

st.sidebar.subheader("OpenAI API Key")
api_key = st.sidebar.text_input(" ", "Insert API Key")
model = st.sidebar.selectbox(" ", ["davinci", "gpt-4"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5, step=0.05)
max_tokens = st.sidebar.slider("Max tokens", 0, 800, 350, step=50)


# getting the correct files for given author and work should be done automatically. Look into file processing and regex to match with file names.
st.sidebar.subheader("Author")
author_name = st.sidebar.text_input(" ", "Author's name")
author_file = st.sidebar.file_uploader("Upload text file of author's biography", type="txt")

if author_file:
    author_string = StringIO(author_file.getvalue().decode("utf-8"))
    author_bio = author_string.read()

st.sidebar.subheader("Written work")
work_title = st.sidebar.text_input(" ", "Work's title")
work_file = st.sidebar.file_uploader("Upload text file of author's written work summary", type="txt")

if work_file:
    work_string = StringIO(work_file.getvalue().decode("utf-8"))
    work = work_string.read()

st.sidebar.subheader("Ask author a question about their work")
query = st.sidebar.text_input(" ", "Insert question")

if api_key and author_bio and work and query:
    if st.button("Ask author"):
        # summarize the book
        with st.spinner("Give me a minute to read the book... (It may take a few minutes for new books)"):
            
            st.write(f"{author_name}'s response:")
            completion = chat_with_author(api_key, model, 
                                          author_bio, work_title, work, 
                                          query, max_tokens, temperature)
            
            response = completion[0]
            st.write(response)