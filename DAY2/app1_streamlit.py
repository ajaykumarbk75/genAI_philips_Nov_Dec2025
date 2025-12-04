# Sample Streamlit application

import streamlit as st # Streamlit librray used for web app development

st.title("Welcome to Streamlit GenAIApp")  # Title of the app
st.write("This is a simple Streamlit application for GenAI workshop.")  # Description
st.write("Explore the power of Generative AI with Streamlit!")  # Additional text

# Input section 
user_input = st.text_input("Enter your name:", "Guest")  # Text input for user name

# Button to submit
if st.button("Submit"): 
    st.write(f"Hello, {user_input}! Welcome to the GenAI Streamlit App.")  # Greeting message

# Footer 
st.markdown("---")  # Horizontal line
st.caption("Thank you for visiting the GenAI Streamlit App!")  # Footer caption
st.write("© 2024 GenAI Workshop")  # Footer text


