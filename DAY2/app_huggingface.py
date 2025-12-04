# Streamlit application with Hugggingface 

import streamlit as st  # Streamlit library for web app development 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # Huggingface transformers
import torch # PyTorch library

st.title("Huggingface GenAI App")  # Title of the app
st.write("This app uses Huggingface transformers to generate text based on your input.")


# Models
MODEL_OPTIONS = {
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2"
}

model_choice = st.selectbox("Model to load :", list(MODEL_OPTIONS.keys()))
MODEL_ID = model_choice

hf_token = st.text_input("Hugging Face token (optional)", type="password")

# Model Loading functions 
@st.cache_resource # Cache the loaded model to avoid reloading on every interaction
def load_model(model_id: str, token: str | None = None): 
    """ Load the Model and return text-generation pipeline.
    If a token provided it will be passed to the 'from-pretrained' 
    so private/gated repos can be accessed. This function is cached to avoid reloading on every interaction  """ 

    kwargs = {} # kwargs to hold additional parameters
    if token : 
        kwargs["use_auth_token"] = token  # Add token to kwargs if provided 
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)  # Load tokenizer
    # Load the model with low CPU memory usage
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        **(kwargs or {}),
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)  # Return text-generation pipeline

# Model loading with Fallback  - 
try:
    generator = load_model(MODEL_ID, hf_token or None) # Load the selected model
except OSError as e: 
    # generate warning message 
    st.warning(f"Could not load model `{MODEL_ID}`: {e}") 
    st.info("Falling back to tiny public model `distilgpt2` for quick CPU testing.")
    try:
        generator = load_model("distilgpt2")  # Fallback to distilgpt2 # if selected model fails to load
    except Exception as e2:
        st.error("Fallback model also failed to load. See exception details.") # Error message
        st.exception(e2) # Show exception details
        st.stop()
except Exception as e:     # exception handling
    st.error("Unexpected error while loading model — see details.") # Error message
    st.exception(e) # Show exception details
    st.stop()   

# User Input Section :
prompt = st.text_area("Enter prompt:", "Hello, how are you?")  # Text area for user prompt
max_tokens = st.slider("Max tokens:", 50, 256, 128, step=50)  # Slider for max tokens
temperature = st.slider("Temperature:", 0.1, 1.0, 0.7, step=0.1)  #

# Generate text on button click
if st.button("Generate"): 
    with st.spinner("Generating..."):  # Show spinner while generating
        output = generator(
            prompt,                     # User prompt
            max_new_tokens=max_tokens,  # Maximum new tokens to generate
            do_sample=True,  # Enable sampling
            temperature=temperature # Temperature for sampling
        )
    st.write(output[0]["generated_text"])  # Display generated text - It is a list of dictionaries





