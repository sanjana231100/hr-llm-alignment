"""
Reward Model dataset.
Loads HH-RLHF preference pairs (chosen vs rejected) filtered to HR conversations.
Each example has a chosen response and a rejected response for the same prompt.
"""

from datasets import load_dataset, concatenate_datasets

HR_PHRASES = [
    "job interview", "hiring manager", "job offer", "job description",
    "performance review", "annual review", "salary negotiation",
    "human resources", "hr department", "hr policy",
    "recruitment", "recruiter", "job application",
    "employee benefits", "health insurance", "401k", "paid leave",
    "maternity leave", "paternity leave", "sick leave",
    "wrongful termination", "layoff", "severance",
    "workplace harassment", "sexual harassment", "hostile work environment",
    "onboarding", "offboarding", "background check",
    "labor law", "employment law", "workers compensation",
    "remote work policy", "work from home policy",
    "staffing agency", "contractor vs employee",
    "non-compete", "non disclosure agreement",
    "overtime pay", "minimum wage", "equal pay",
]


def is_hr_relevant(example: dict) -> bool:
    text = example["chosen"].lower()
    return any(phrase in text for phrase in HR_PHRASES)


def format_for_reward(example: dict) -> dict:
    """
    Reward model needs both chosen and rejected text.
    We keep them as separate fields — the trainer handles the pair.
    """
    return {
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def load_hr_reward_dataset(cfg: dict):
    """
    Load HH-RLHF filtered to HR conversations.
    Returns dataset with 'chosen' and 'rejected' columns.
    """
    print("Loading HH-RLHF for reward model training...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    print(f"Full dataset size: {len(dataset)}")

    print("Filtering to HR conversations...")
    hr_dataset = dataset.filter(is_hr_relevant)
    print(f"HR subset size: {len(hr_dataset)}")

    hr_dataset = hr_dataset.map(
        format_for_reward,
        remove_columns=[c for c in hr_dataset.column_names
                       if c not in ["chosen", "rejected"]]
    )

    hr_dataset = hr_dataset.shuffle(seed=42)
    print(f"Reward training examples: {len(hr_dataset)}")
    return hr_dataset


if __name__ == "__main__":
    import yaml
    with open("configs/reward_config.yaml") as f:
        cfg = yaml.safe_load(f)

    dataset = load_hr_reward_dataset(cfg)
    print("\n=== Sample chosen response ===")
    print(dataset[0]["chosen"][:300])
    print("\n=== Sample rejected response ===")
    print(dataset[0]["rejected"][:300])