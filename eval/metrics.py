"""
Evaluation metrics for the alignment pipeline.
Computes perplexity, ROUGE-L, BERTScore, and reward model score.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def compute_perplexity(model, tokenizer, texts, max_length=512, device="cuda"):
    """
    Perplexity = exp(average negative log likelihood per token).
    Lower is better — model assigns higher probability to correct tokens.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            num_tokens = inputs["input_ids"].shape[1]

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def compute_rouge_l(predictions, references):
    """
    ROUGE-L measures longest common subsequence overlap.
    Ranges 0-1, higher is better.
    Used to check if model responses overlap with reference answers.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score["rougeL"].fmeasure)
    return np.mean(scores)


def compute_bert_score(predictions, references, lang="en"):
    """
    BERTScore measures semantic similarity using BERT embeddings.
    More robust than ROUGE — captures meaning, not just word overlap.
    Ranges 0-1, higher is better.
    """
    P, R, F1 = bert_score(predictions, references, lang=lang, verbose=False)
    return F1.mean().item()


def generate_responses(model, tokenizer, prompts, max_new_tokens=200, device="cuda"):
    """
    Generate responses from the model for a list of prompts.
    """
    model.eval()
    responses = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            responses.append(response.strip())

    return responses


if __name__ == "__main__":
    print("Metrics module loaded successfully")
    print("Available: compute_perplexity, compute_rouge_l, compute_bert_score")