# PDF Summarizer
import streamlit as st
import pdfplumber  # pdfplumber library for PDF text extraction
from openai import OpenAI
import os

# App title 
st.title("PDF Summarizer 📝")   

#Initialize OpenAI API
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None
    st.warning("OpenAI API key not found. Please set it in Streamlit secrets or as an environment variable.")

# User unputs 
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])  # File uploader for PDF files
question = st.text_area("Enter your question ?  about the PDF content:")  # Text area for user question 

# Process Q & A 
if st.button("Ask"):
    if not uploaded_file:
        st.error("Please upload a PDF file.")
    elif not question.strip():
        st.error("Please enter a valid question.")
    elif client is None:
        st.error("OpenAI API key is not configured or missing.")
    else: 
        with pdfplumber.open(uploaded_file) as pdf: # open the uploaded PDF file
            extracted_text = []   # List to hold extracted text
            for page in pdf.pages:  # Iterate through each page
                text = page.extract_text()  # Extract text from the page
                if text:
                    extracted_text.append(text)  # Append extracted text to the list 
        if not extracted_text:
            st.error("No text could be extracted from the PDF.")
        else: 
            content = " ".join (extracted_text)  # Combine all extracted text into a single string

            # Trim Long conent for tp prevent token pverflow 
            max_length = 6000  # Define maximum content length
            if len(content) > max_length:
                content = content[:max_length]  # Trim content to maximum length
            
            prompt = f"Based on the following PDF content, answer the question:\n\n{content}\n\nQuestion: {question}\n\nAnswer:"

            try: # Make openAPi call
                response = client.chat.completions.create(
                    model="gpt-5.1",  # Specify the model
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides concise answers based on the provided PDF content."},
                        {"role": "user", "content": prompt}],
                )
                answer = response.choices[0].message.content  # Extract the answer from the response

                st.subheader("Answer:")  # Display the answer
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
