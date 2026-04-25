"""
HR LLM Alignment Pipeline — Streamlit Demo
Compares base Mistral-7B vs SFT vs DPO aligned model on HR queries.
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

st.set_page_config(
    page_title="HR Assistant — LLM Alignment Demo",
    page_icon="🤝",
    layout="wide",
)

st.title("🤝 HR Assistant — LLM Alignment Pipeline")
st.markdown("""
This demo shows the effect of the three-phase alignment pipeline:
**SFT → Reward Model → DPO** on a Mistral-7B base model fine-tuned for HR/workforce queries.
""")

@st.cache_resource
def load_model(adapter_path=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        quantization_config=bnb_config,
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=300):
    formatted = f"\n\nHuman: {prompt}\n\nAssistant:"
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


# sidebar
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose model to query:",
    ["SFT Model", "DPO Aligned Model"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**About this project**")
st.sidebar.markdown("""
- **Phase 1:** SFT on HR Q&A dataset
- **Phase 2:** Reward model on HH-RLHF pairs
- **Phase 3:** DPO alignment
- **Model:** Mistral-7B + QLoRA
- **[GitHub](https://github.com/sanjana231100/hr-llm-alignment)**
""")

# sample questions
st.markdown("### Try an HR question")
sample_questions = [
    "What should I do if I witness workplace harassment?",
    "How do I negotiate a salary increase with my manager?",
    "What are my rights if I am laid off from my job?",
    "How should I prepare for a performance review?",
    "What is the difference between a contractor and a full-time employee?",
]

selected = st.selectbox("Pick a sample question or type your own:", [""] + sample_questions)
user_input = st.text_area("Your HR question:", value=selected, height=100)

if st.button("Generate Response", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        adapter_path = "sanjana231100/hr-sft-mistral-7b" if model_choice == "SFT Model" else "sanjana231100/hr-dpo-mistral-7b"

        with st.spinner(f"Loading {model_choice} and generating response..."):
            model, tokenizer = load_model(adapter_path)
            response = generate_response(model, tokenizer, user_input)

        st.markdown(f"### {model_choice} Response")
        st.markdown(f"> {response}")

        st.markdown("---")
        st.caption("Powered by Mistral-7B + QLoRA | Trained with SFT + Reward Model + DPO")