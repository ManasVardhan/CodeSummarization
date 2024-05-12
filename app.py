import streamlit as st
from transformers import AutoModel, AutoTokenizer
import torch

st.title("📝 Summarizatize.IT")

# Load pre-trained model and tokenizer
checkpoint = "Salesforce/codet5p-220m-bimodal"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

# Function to summarize code
def summarize_code(code):

    
    input_ids = tokenizer(code, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This web app uses the `Salesforce/codet5p-220m-bimodal` model to summarize code."
)

st.sidebar.header("How it works")
st.sidebar.info("""
1. Paste your code in the textbox.
2. Click on the 'Summarize' button.
3. The app will generate a summary of your code.

**Note**: This model works best with Python code.
""")


# Main content
st.subheader("Enter Your Code Below:")
code_input = st.text_area("Paste your code here:")

if st.button("Summarize"):
    if code_input.strip():
        summary = summarize_code(code_input)
        st.success(f"Summary: {summary}")
    else:
        st.warning("Please enter some code.")

# Emojis
st.sidebar.markdown("""
🚀 Happy summarizing! 📝
""")
