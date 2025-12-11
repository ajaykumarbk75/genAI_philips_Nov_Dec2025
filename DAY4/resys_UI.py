# recsys_ui.py # this is the Streamlit UI for the AI-driven Recommendation System
import streamlit as st
import requests

st.title("AI Recommendation System")

user_id = st.number_input("Enter User ID", min_value=1, value=1)
top_k = st.number_input("Number of recommendations", min_value=1, value=3)

if st.button("Get Recommendations"):
    resp = requests.post(
        "http://127.0.0.1:8000/recommend",
        json={"user_id": user_id, "top_k": top_k}
    )
    if resp.status_code == 200:
        st.success(f"Recommended items: {resp.json()['recommendations']}")
    else:
        st.error("Error getting recommendations")
