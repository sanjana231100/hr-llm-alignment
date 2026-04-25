"""
Evaluation script for the alignment pipeline.
Compares SFT model vs DPO model on HR test prompts.
Metrics: perplexity, ROUGE-L, BERTScore
"""

import os
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from eval.metrics import (
    compute_perplexity,
    compute_rouge_l,
    compute_bert_score,
    generate_responses,
)

# HR test prompts — held out, never seen during training
HR_TEST_PROMPTS = [
    "\n\nHuman: What should I do if I witness workplace harassment?\n\nAssistant:",
    "\n\nHuman: How do I negotiate a salary increase with my manager?\n\nAssistant:",
    "\n\nHuman: What are my rights if I am laid off from my job?\n\nAssistant:",
    "\n\nHuman: How should I prepare for a performance review?\n\nAssistant:",
    "\n\nHuman: What is the difference between a contractor and a full-time employee?\n\nAssistant:",
    "\n\nHuman: How do I handle a conflict with a coworker professionally?\n\nAssistant:",
    "\n\nHuman: What does an onboarding process typically include?\n\nAssistant:",
    "\n\nHuman: Can my employer change my job description without notice?\n\nAssistant:",
]

HR_REFERENCE_ANSWERS = [
    "You should document the incident, report it to HR immediately, and ensure confidentiality is maintained throughout the process.",
    "Research market rates for your role, prepare a list of your achievements, and schedule a dedicated meeting with your manager.",
    "You are entitled to any earned wages, and may be eligible for severance pay and unemployment benefits depending on your location.",
    "Review your goals from the previous period, document your achievements with metrics, and prepare questions for your manager.",
    "A contractor is self-employed and works on specific projects, while a full-time employee receives benefits and has ongoing employment.",
    "Address the issue directly and professionally, focus on specific behaviors rather than personal attacks, and involve HR if needed.",
    "Onboarding typically includes paperwork, IT setup, introductions to the team, and training on company policies and tools.",
    "Generally employers can adjust job duties, but significant changes may require notice or renegotiation of your employment terms.",
]


def load_model(model_name: str, adapter_path: str = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if adapter_path:
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def run_evaluation(model, tokenizer, model_name: str):
    print(f"\n=== Evaluating {model_name} ===")

    # generate responses
    print("Generating responses...")
    responses = generate_responses(
        model, tokenizer, HR_TEST_PROMPTS, max_new_tokens=150
    )

    # compute metrics
    rouge = compute_rouge_l(responses, HR_REFERENCE_ANSWERS)
    bert = compute_bert_score(responses, HR_REFERENCE_ANSWERS)
    perplexity = compute_perplexity(
        model, tokenizer,
        [p + r for p, r in zip(HR_TEST_PROMPTS, HR_REFERENCE_ANSWERS)]
    )

    print(f"Perplexity:  {perplexity:.2f}")
    print(f"ROUGE-L:     {rouge:.4f}")
    print(f"BERTScore:   {bert:.4f}")

    print("\nSample responses:")
    for i, (prompt, response) in enumerate(zip(HR_TEST_PROMPTS[:3], responses[:3])):
        print(f"\nQ: {prompt.split('Human:')[1].split('Assistant:')[0].strip()}")
        print(f"A: {response[:200]}")

    return {
        "model": model_name,
        "perplexity": perplexity,
        "rouge_l": rouge,
        "bert_score": bert,
    }


if __name__ == "__main__":
    print("Loading SFT model...")
    sft_model, tokenizer = load_model(
        "mistralai/Mistral-7B-v0.1",
        adapter_path="./outputs/sft"
    )
    sft_results = run_evaluation(sft_model, tokenizer, "SFT")

    # free memory
    del sft_model
    torch.cuda.empty_cache()

    print("\nLoading DPO model...")
    dpo_model, tokenizer = load_model(
        "mistralai/Mistral-7B-v0.1",
        adapter_path="./outputs/dpo"
    )
    dpo_results = run_evaluation(dpo_model, tokenizer, "DPO")

    print("\n=== Final Comparison ===")
    print(f"{'Metric':<15} {'SFT':>10} {'DPO':>10} {'Change':>10}")
    print("-" * 45)
    for metric in ["perplexity", "rouge_l", "bert_score"]:
        sft_val = sft_results[metric]
        dpo_val = dpo_results[metric]
        change = dpo_val - sft_val
        print(f"{metric:<15} {sft_val:>10.4f} {dpo_val:>10.4f} {change:>+10.4f}")