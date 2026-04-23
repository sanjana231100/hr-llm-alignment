"""
Phase 2: Reward Model Training
Replaces Mistral's LM head with a scalar head, trains on chosen vs rejected pairs.
Loss: Bradley-Terry - pushes score(chosen) > score(rejected)
"""

import yaml
import wandb
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, TaskType
from src.reward.dataset import load_hr_reward_dataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def tokenize_reward_dataset(dataset, tokenizer, max_seq_length: int):
    """
    RewardTrainer expects input_ids_chosen, attention_mask_chosen,
    input_ids_rejected, attention_mask_rejected.
    We tokenize both chosen and rejected responses here.
    """
    def tokenize(example):
        tokenized_chosen = tokenizer(
            example["chosen"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        tokenized_rejected = tokenizer(
            example["rejected"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def main():
    cfg = load_config("configs/reward_config.yaml")

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        tags=cfg["wandb"]["tags"],
        config=cfg,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print("Loading HR reward dataset...")
    dataset = load_hr_reward_dataset(cfg)

    print("Tokenizing dataset...")
    tokenized_dataset = tokenize_reward_dataset(
        dataset, tokenizer, cfg["training"]["max_seq_length"]
    )

    print(f"Tokenized dataset size: {len(tokenized_dataset)}")
    print(f"Columns: {tokenized_dataset.column_names}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # LoRA config for reward model
    # We use smaller r=8 since reward model needs less capacity than SFT
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    train_cfg = cfg["training"]
    reward_config = RewardConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        report_to="wandb",
        fp16=True,
        max_length=train_cfg["max_seq_length"],
        remove_unused_columns=False,
    )

    from transformers import AutoModelForSequenceClassification
    from peft import prepare_model_for_kbit_training, get_peft_model

    print("Loading SFT model as reward model base...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        num_labels=1,              # scalar reward output
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
    )

    print("Starting reward model training...")
    trainer.train()

    print("Saving reward model...")
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    print("Reward model saved.")

    wandb.finish()


if __name__ == "__main__":
    main()