from utils import summarize_book
import streamlit as st
from io import StringIO

st.title("EconGPT Book/Paper Summarizer")

st.subheader("Upload a text file of a book or paper and I will summarize it for you.")

st.sidebar.title("Inputs")
api_key = st.sidebar.text_input(" ", "OpenAI API Key")

work_file = st.file_uploader("Upload text file of book/paper", type="txt")

if work_file:
    work_string = StringIO(work_file.getvalue().decode("utf-8"))
    work = work_string.read()

if st.button("Summarize") and api_key:
    work_summary = summarize_book(work, api_key)

    st.subheader("Summary")
    st.write("Here is the summary generated. Copy the text and save it as a text file. You can use this summary text file as an input for the model in the next page.")
    st.write(work_summary)