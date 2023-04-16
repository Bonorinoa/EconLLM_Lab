import streamlit as st
import pandas as pd
from utils import chat_with_author, summarize_book

with open ("AdamSmith_MoralSentiments\\AdamSmith_Bio.txt") as f:
    adam_bio = f.read()
    
with open("AdamSmith_MoralSentiments\\TMS_Summary.txt") as f:
    tms_summary = f.read()
    
st.title("EconGPT")

st.sidebar.title("Inputs")

st.sidebar.subheader("OpenAI API Key")
api_key = st.sidebar.text_input(" ", "Insert API Key")
model = st.sidebar.selectbox(" ", ["davinci", "gpt-4"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5, step=0.05)
max_tokens = st.sidebar.slider("Max tokens", 0, 800, 350, step=50)

st.sidebar.subheader("Author")
author = st.sidebar.selectbox(" ", ["Adam Smith"])

st.sidebar.subheader("Written work")
work = st.sidebar.selectbox(" ", ["The Theory of Moral Sentiments"])

st.sidebar.subheader("Ask author a question about their work")
query = st.sidebar.text_input(" ", "Insert question")

if api_key and author and work and query:
    if st.button("Ask author"):
        # summarize the book
        with st.spinner("Give me a minute to read the book... (It may take a few minutes for new books)"):
            
            #work_summary = summarize_book(tms_summary, api_key)
            
            st.write("Author's response:")
            completion = chat_with_author(api_key, model, adam_bio, tms_summary, query)
            
            response = completion[0]
            st.write(response)