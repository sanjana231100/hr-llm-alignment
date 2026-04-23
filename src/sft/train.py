"""
Phase 1: Supervised Fine-Tuning (SFT)
Mistral-7B + QLoRA (4-bit) on HR/workforce conversations.
Commit 1: scaffold — verifies model loads, LoRA applies, dataset filters correctly.
"""

import yaml
import wandb
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from src.sft.dataset import load_hr_sft_dataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    """
    QLoRA: weights stored in 4-bit nf4, dequantized to float16 during forward pass.
    double_quant saves an extra ~0.4 bits per parameter.
    """
    return BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_model_and_tokenizer(cfg: dict):
    model_name = cfg["model"]["name"]
    bnb_config = build_bnb_config(cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    return model, tokenizer


def apply_lora(model, cfg: dict):
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def main():
    cfg = load_config("configs/sft_config.yaml")

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        tags=cfg["wandb"]["tags"],
        config=cfg,
    )

    print(f"Loading model: {cfg['model']['name']}")
    model, tokenizer = build_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    print("Loading HR dataset...")
    dataset = load_hr_sft_dataset(cfg)

    print("\n=== Sample training example ===")
    print(dataset[0]["text"][:500])
    print("=== End sample ===\n")

    train_cfg = cfg["training"]
    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        max_seq_length=train_cfg["max_seq_length"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_strategy="no",
        report_to="wandb",
        fp16=True,
    )
    tokenizer.padding_side = 'right'
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()