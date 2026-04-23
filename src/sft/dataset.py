"""
HR dataset for SFT phase.
Combines three public HR datasets for a clean, domain-specific training set.
  1. syncora/hr-policies-qa-dataset — HR policy Q&A (644 examples)
  2. strova-ai/hr-policies-qa-dataset — HR compliance Q&A (duplicate source, different curation)
  3. Pradeep016/career-guidance-qa-dataset — career and workforce guidance Q&A
"""

from datasets import load_dataset, concatenate_datasets


def format_syncora(example: dict) -> dict:
    user_msg = example.get("user", "")
    assistant_msg = example.get("assistant", "")
    text = f"\n\nHuman: {user_msg}\n\nAssistant: {assistant_msg}"
    return {"text": text}


def format_career_guidance(example: dict) -> dict:
    question = example.get("question", "")
    answer = example.get("answer", "")
    text = f"\n\nHuman: {question}\n\nAssistant: {answer}"
    return {"text": text}


def load_syncora_dataset():
    print("Loading syncora HR policies dataset...")
    dataset = load_dataset("syncora/hr-policies-qa-dataset", split="train")
    print(f"Syncora size: {len(dataset)}")
    dataset = dataset.map(format_syncora, remove_columns=dataset.column_names)
    return dataset


def load_strova_dataset():
    print("Loading strova HR policies dataset...")
    dataset = load_dataset("strova-ai/hr-policies-qa-dataset", split="train")
    print(f"Strova size: {len(dataset)}")
    dataset = dataset.map(format_syncora, remove_columns=dataset.column_names)
    return dataset


def load_career_dataset():
    print("Loading career guidance dataset...")
    dataset = load_dataset("Pradeep016/career-guidance-qa-dataset", split="train")
    print(f"Career guidance size: {len(dataset)}")
    dataset = dataset.map(format_career_guidance, remove_columns=dataset.column_names)
    return dataset


def load_hr_sft_dataset(cfg: dict):
    """
    Combine all three HR datasets into final SFT training set.
    """
    datasets = []

    try:
        datasets.append(load_syncora_dataset())
    except Exception as e:
        print(f"Syncora dataset failed: {e}")

    try:
        datasets.append(load_strova_dataset())
    except Exception as e:
        print(f"Strova dataset failed: {e}")

    try:
        datasets.append(load_career_dataset())
    except Exception as e:
        print(f"Career guidance dataset failed: {e}")

    combined = concatenate_datasets(datasets)
    combined = combined.shuffle(seed=42)

    print(f"\nCombined dataset size: {len(combined)}")

    if cfg["training"]["max_samples"] and len(combined) > cfg["training"]["max_samples"]:
        combined = combined.select(range(cfg["training"]["max_samples"]))
        print(f"Capped to {len(combined)} samples")

    return combined


if __name__ == "__main__":
    import yaml
    with open("configs/sft_config.yaml") as f:
        cfg = yaml.safe_load(f)

    dataset = load_hr_sft_dataset(cfg)
    print("\n=== Sample HR conversation ===")
    print(dataset[0]["text"][:600])
    print(f"\n=== Total samples: {len(dataset)} ===")