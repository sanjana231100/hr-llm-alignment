"""
DPO dataset.
Uses the same HH-RLHF preference pairs as the reward model.
DPOTrainer expects 'prompt', 'chosen', 'rejected' columns.
"""

from datasets import load_dataset


def format_for_dpo(example: dict) -> dict:
    """
    HH-RLHF format:
      chosen: '\n\nHuman: <question>\n\nAssistant: <response>'
      rejected: '\n\nHuman: <question>\n\nAssistant: <response>'

    DPO needs the prompt separate from the responses.
    We split on the last 'Assistant:' to get prompt and completion.
    """
    def split_prompt_response(text):
        split = text.rsplit("\n\nAssistant:", 1)
        if len(split) == 2:
            prompt = split[0] + "\n\nAssistant:"
            response = split[1].strip()
        else:
            prompt = text
            response = ""
        return prompt, response

    prompt, chosen = split_prompt_response(example["chosen"])
    _, rejected = split_prompt_response(example["rejected"])

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def load_hr_dpo_dataset(max_samples: int = 2000):
    """
    Load HH-RLHF and format for DPO training.
    """
    print("Loading HH-RLHF for DPO...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    dataset = dataset.select(range(max_samples))

    dataset = dataset.map(
        format_for_dpo,
        remove_columns=dataset.column_names
    )

    dataset = dataset.shuffle(seed=42)
    print(f"DPO dataset ready: {len(dataset)} examples")
    return dataset


if __name__ == "__main__":
    dataset = load_hr_dpo_dataset()
    print("\n=== Sample ===")
    print("PROMPT:", dataset[0]["prompt"][:200])
    print("CHOSEN:", dataset[0]["chosen"][:200])
    print("REJECTED:", dataset[0]["rejected"][:200])