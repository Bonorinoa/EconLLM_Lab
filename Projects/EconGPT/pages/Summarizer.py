from utils import summarize_book
import streamlit as st
from io import StringIO

st.title("EconGPT Book/Paper Summarizer")

work_file = st.file_uploader("Upload text file of book/paper", type="txt")
work_string = StringIO(work_file.getvalue().decode("utf-8"))
work = work_string.read()

if st.button("Summarize"):
    work_summary = summarize_book(work)

st.subheader("Summary")
st.write("Here is the summary generated. Copy the text and save it as a text file. You can use this summary text file as an input for the model in the next page.")
st.write(work_summary)