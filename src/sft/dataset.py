"""
HR dataset filtering and formatting for SFT phase.
Filters Anthropic HH-RLHF to workforce/HR-relevant conversations.
"""

from datasets import load_dataset

HR_KEYWORDS = [
    "hiring", "recruitment", "interview", "candidate", "resume",
    "salary", "compensation", "benefits", "payroll", "bonus",
    "performance review", "termination", "onboarding", "offboarding",
    "manager", "employee", "workplace", "hr policy", "human resources",
    "vendor", "contractor", "workforce", "staffing", "job offer",
    "promotion", "demotion", "harassment", "discrimination", "leave",
    "remote work", "work from home", "overtime", "labor law",
]


def is_hr_relevant(example: dict) -> bool:
    """
    Returns True if the chosen conversation contains
    at least one HR-relevant keyword.
    """
    text = example["chosen"].lower()
    return any(keyword in text for keyword in HR_KEYWORDS)


def format_for_sft(example: dict) -> dict:
    """
    HH-RLHF chosen field format:
      '\n\nHuman: <question>\n\nAssistant: <response>'
    We use it directly — Mistral learns this conversation structure.
    """
    return {"text": example["chosen"]}


def load_hr_sft_dataset(cfg: dict):
    """
    Load HH-RLHF, filter to HR conversations, format for SFT.
    Returns a HuggingFace Dataset with a single 'text' column.
    """
    print("Loading HH-RLHF dataset...")
    dataset = load_dataset(
        cfg["training"]["dataset"],
        split="train"
    )
    print(f"Full dataset size: {len(dataset)}")

    print("Filtering to HR-relevant conversations...")
    hr_dataset = dataset.filter(is_hr_relevant)
    print(f"HR subset size: {len(hr_dataset)}")

    if cfg["training"]["max_samples"]:
        hr_dataset = hr_dataset.select(
            range(min(cfg["training"]["max_samples"], len(hr_dataset)))
        )
        print(f"Using {len(hr_dataset)} samples for training")

    hr_dataset = hr_dataset.map(
        format_for_sft,
        remove_columns=hr_dataset.column_names
    )

    return hr_dataset


if __name__ == "__main__":
    import yaml
    with open("configs/sft_config.yaml") as f:
        cfg = yaml.safe_load(f)

    dataset = load_hr_sft_dataset(cfg)
    print("\n=== Sample HR conversation ===")
    print(dataset[0]["text"][:600])
    print(f"\n=== Total samples: {len(dataset)} ===")