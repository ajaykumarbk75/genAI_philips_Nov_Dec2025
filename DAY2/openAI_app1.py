# OpenAI Application - ChatBot - Streamlit App

import streamlit as st # Streamlit library for web app development 

from  openai import OpenAI   # OpenAI library for interacting with OpenAI API 

import os  # OS library for environment variable access

# -------------
# 1. API-Key 
#---------------
api_key = None # Variable to hold API key

try: 
    api_key = st.secrets.get("OPENAI_API_KEY")  # Try to get API key from Streamlit secrets
except Exception:
    pass

#if no api key found in secrets, try environment variable
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")  # Get API key from environment variable

if api_key:
    client = OpenAI(api_key=api_key)  # Initialize OpenAI client with API key
else:
    # if no API key found, show warnings  
    clients = None
    st.warning("OpenAI API key not found. Please set it in Streamlit secrets or environment variable 'OPENAI_API_KEY'.")

#-------------------
#  2. Steamlit UI 
#--------------------
st.title("OpenAI ChatBot App")  # Title of the app
st.write("This app uses OpenAI API to generate text based on your input.")  #
st.write("Enter a prompt and get a response from the OpenAI model.")  # Description

#Maintain chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize chat history #Blank list

# -----------------------
# 3. Input field for the user :
#--------------------------
user_input = st.text_input("You(Your_Name ):", "")  # Text input for user message

#When user submitts the message 
if st.button("Send"):
    # add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input})

    #call the OpenAI API to get response
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=st.session_state["messages"]  # Pass the chat history
    )

    # extract the assistant's reply from the response
    reply = response.choices[0].message.content
    # add assistant's reply to chat history
    st.session_state["messages"].append({"role": "assistant", "content": reply})
# Display the chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.write(f"You: {msg['content']}")  # Display user message
    else:
        st.write(f"Bot: {msg['content']}")  # Display assistant message
# End of OpenAI ChatBot App


