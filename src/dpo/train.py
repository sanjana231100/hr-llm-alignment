"""
Phase 3: Direct Preference Optimization (DPO)
Fine-tunes the SFT model to prefer chosen responses over rejected ones.
Uses DPOTrainer from TRL — no RL loop, closed-form objective.
Beta controls KL penalty: higher = stay closer to SFT reference.
"""

import yaml
import os
import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
from trl import DPOTrainer, DPOConfig
from src.dpo.dataset import load_hr_dpo_dataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("configs/dpo_config.yaml")

    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        tags=cfg["wandb"]["tags"],
        config=cfg,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # DPO needs left padding

    print("Loading dataset...")
    dataset = load_hr_dpo_dataset(cfg["training"]["max_samples"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    train_cfg = cfg["training"]
    dpo_config = DPOConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        beta=train_cfg["beta"],
        warmup_steps=train_cfg["warmup_steps"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        report_to="wandb",
        fp16=False,
        bf16=False,
        max_length=train_cfg["max_seq_length"],
        max_prompt_length=train_cfg["max_prompt_length"],
        remove_unused_columns=False,
        run_name=cfg["wandb"]["run_name"],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    print("Saving DPO model...")
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    print("DPO training complete.")

    wandb.finish()


if __name__ == "__main__":
    main()