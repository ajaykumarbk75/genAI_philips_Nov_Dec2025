# text_Summarization / Document summarization

import streamlit as st  # Streamlit library for web app development
from openai import OpenAI  # OpenAI library for interacting with OpenAI API
import os    # OS library for environment variable access

st.set_page_config(page_title="AI Text Summarizer", page_icon="📝")  # Set page configuration
st.title("AI TEXT Summarizer")  # Title of the app

# CUSTOM CSS - (Small Clear button & Clolored summarize button ) ----
st.markdown("""
<style>  
/* Summarize button */
div.stButton > button:first-child {
    background-color: #4CAF50; /* Green */
    color: white;
    padding: 10px 24px;
    border: none;
    padding:0.45em 1.0em;
    font-size: 1rem;
}
div.stButton > button:first-child:hover {
   background-color: #45a049; /* Darker green */
 }
/*Clear Button*/
    background-color: #e53935  ; /* Red */
    color: white;
    border-radius: 4px;
    padding:0.35em 0.9em;
    font-size: 0.9rem;
}
  #clear-btn button:hover {
    background-color: #da190b; /* Darker red */
 </style>       
""", unsafe_allow_html=True)

# ------- Side bar -----

st.sidebar.header("!About")  # Sidebar header
st.sidebar.write("This application uses OpenAI's GPT models to summarize text. Enter your text and click 'Summarize")  # Sidebar description
st.sidebar.info("Developed as part of the GenAI Workshop 2025.")  # Sidebar info

# API Set up 
try:
      api_key = st.secrets.get("OPENAI_API_KEY")  # Try to get API key from Streamlit secrets
except:
       api_key = os.getenv("OPENAI_API_KEY") # Get API key from environment variable

client = OpenAI(api_key=api_key) if api_key else None  # Initialize OpenAI client with API key

text = st.text_area("Enter text to summarize:")   # Text area for user input

#Button Row - Summarize & Clear
col1, col2 = st.columns(2)  # Create two columns for buttons
with col1:  # First column
      summarize_clicked = st.button("Summarize")  # Summarize button
with col2:  # Second column
      clear_clicked = st.button("Clear", key="clear-btn")  # Clear button   

# --- CLEAR BUTTON FUNCTIONALITY ----

if clear_clicked:
      st.session_state["text"] = ""  # Clear the text area
      st.experimental_rerun()  # Rerun the app to reflect changes

# Summarization button
if st.button("Summarize"):
      if client :
            response = client.chat.completions.create( # create chat completion 
                  model="gpt-5.1", #"gpt-4.1-mini",                      #"gpt-5.1",  # specify model
                  messages=[       # messages for chat completion
                        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                        {"role": "user", "content": f"Summarize the following text:\n\n{text}"}]
                        ) 
            st.subheader("Summary:")  # Subheader for summary. 
            st.write(response.choices[0].message.content)  # Display the summary
      else:
            st.error("Missing API Key!") # Error message for missing API key




