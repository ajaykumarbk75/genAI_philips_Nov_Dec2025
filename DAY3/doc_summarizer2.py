import streamlit as st
from openai import OpenAI
import os

st.set_page_config(page_title="AI Text Summarizer", page_icon="📝")

# ---------- CUSTOM CSS (small clear button + colored summarize button) ----------
st.markdown("""
<style>
/* Summarize Button */
div.stButton > button:first-child {
    background-color: #4CAF50;
    color:white;
    border-radius: 6px;
    padding: 0.45em 1.0em;
    font-size: 1rem;
}
div.stButton > button:first-child:hover {
    background-color: #43a047;
}

/* Clear Button (small, red) */
#clear-btn button {
    background-color: #e53935;
    color:white;
    border-radius: 6px;
    padding: 0.35em 0.9em;
    font-size: 0.9rem;
}
#clear-btn button:hover {
    background-color: #d32f2f;
}
</style>
""", unsafe_allow_html=True)


# ---------- SIDEBAR ----------
st.sidebar.title("ℹ️ About")
st.sidebar.write("Paste your text, click **Summarize**, and get a clean summary instantly.")
st.sidebar.info("Powered by GPT-5.1")


# ---------- MAIN TITLE ----------
st.title("📝 AI TEXT Summarizer")


# ---------- API KEY ----------
try:
    api_key = st.secrets.get("OPENAI_API_KEY")
except:
    api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key) if api_key else None


# ---------- TEXT INPUT ----------
text = st.text_area("Enter text to summarize:", height=200)


# Buttons Row
col1, col2 = st.columns([3, 1])

with col1:
    summarize_clicked = st.button("Summarize")

with col2:
    clear_clicked = st.button("Clear", key="clear-btn")


# ---------- CLEAR BUTTON ----------
if clear_clicked:
    st.session_state["text"] = ""
    st.rerun()


# ---------- SUMMARIZATION ----------
if summarize_clicked:
    if not client:
        st.error("❌ Missing API Key!")
    elif not text.strip():
        st.warning("⚠️ Please enter text before summarizing.")
    else:
        with st.spinner("Summarizing..."):
            response = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": "You summarize text clearly and concisely."},
                    {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
                ]
            )

        st.subheader("📌 Summary:")
        st.success(response.choices[0].message.content)
