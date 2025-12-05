import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

st.title("CPU-Friendly Chat App")

MODEL_OPTIONS = [
    "NousResearch/Nous-Hermes-1b",
    "mosaicml/mpt-7b-instruct",
    "OpenAssistant/oasst-sft-6-llama-2-7b",
    "tiiuae/falcon-7b-instruct",
    "distilgpt2",  # ultra-light fallback (public)
    "gpt2",
]

model_choice = st.selectbox("Model to load (choose public or gated):", MODEL_OPTIONS, index=0)
MODEL_ID = model_choice

# Optional HF token (paste if model is gated/private)
hf_token = st.text_input("Hugging Face token (optional)", type="password")

@st.cache_resource
def load_model(model_id: str, token: str | None = None):
    """Load a model and return a text-generation pipeline.

    If a token is provided it will be passed to `from_pretrained` so private/gated
    repos can be accessed.
    """
    kwargs = {}
    if token:
        kwargs["use_auth_token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        **(kwargs or {}),
    )

    return pipeline("text-generation", model=model, tokenizer=tokenizer)

try:
    generator = load_model(MODEL_ID, hf_token or None)
except OSError as e:
    st.warning(f"Could not load model `{MODEL_ID}`: {e}")
    st.info("Falling back to tiny public model `distilgpt2` for quick CPU testing.")
    try:
        generator = load_model("distilgpt2")
    except Exception as e2:
        st.error("Fallback model also failed to load. See exception details.")
        st.exception(e2)
        st.stop()
except Exception as e:
    st.error("Unexpected error while loading model — see details.")
    st.exception(e)
    st.stop()

prompt = st.text_area("Enter prompt:", "Hello, how are you?")
max_tokens = st.slider("Max tokens:", 50, 256, 128, step=50)
temperature = st.slider("Temperature:", 0.1, 1.0, 0.7, step=0.1)

if st.button("Generate"):
    with st.spinner("Generating..."):
        output = generator(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature
        )
    st.write(output[0]["generated_text"])