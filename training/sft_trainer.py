"""
SFT Trainer — Supervised Fine-Tuning with QLoRA + DeepSpeed

Stage 1 of the alignment pipeline:
- Fine-tune a base model on high-quality math reasoning data
- Uses QLoRA (4-bit quantization + LoRA adapters) for memory efficiency
- Supports DeepSpeed ZeRO-2 for optimizer state sharding
- Designed to run on Google Colab T4 (16GB VRAM)
"""

import os
import yaml
import logging
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SFTFineTuner:
    """
    Supervised Fine-Tuning pipeline for math reasoning.

    Usage:
        finetuner = SFTFineTuner("configs/sft_config.yaml")
        finetuner.train()
        finetuner.save()
    """

    def __init__(self, config_path: str = "configs/sft_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.peft_config = None

    # ----------------------------------------------------------
    # Model Setup
    # ----------------------------------------------------------
    def setup_model(self):
        """Load model with QLoRA quantization."""
        model_name = self.config["model"]["name_or_path"]
        logger.info(f"Loading model: {model_name}")

        # QLoRA quantization config
        qlora_cfg = self.config["qlora"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_cfg["load_in_4bit"],
            bnb_4bit_compute_dtype=getattr(torch, qlora_cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=qlora_cfg["bnb_4bit_use_double_quant"],
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization
        model_kwargs = dict(
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=getattr(torch, self.config["model"].get("torch_dtype", "bfloat16")),
        )
        # Only set attn_implementation if explicitly configured (avoids flash_attn errors)
        attn_impl = self.config["model"].get("attn_implementation")
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config["training"].get("gradient_checkpointing", True),
        )

        # LoRA config — SFTTrainer will apply PEFT internally
        self.peft_config = LoraConfig(
            r=qlora_cfg["lora_r"],
            lora_alpha=qlora_cfg["lora_alpha"],
            lora_dropout=qlora_cfg["lora_dropout"],
            target_modules=qlora_cfg["target_modules"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Note: do NOT call get_peft_model() here — SFTTrainer handles PEFT
        # internally when peft_config is passed. Applying it twice causes errors.
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Base model parameters: {total:,}")
        logger.info(f"LoRA rank={qlora_cfg['lora_r']}, alpha={qlora_cfg['lora_alpha']}")

        return self

    # ----------------------------------------------------------
    # Dataset Setup
    # ----------------------------------------------------------
    def setup_dataset(self) -> DatasetDict:
        """Load and prepare the SFT training dataset."""
        ds_cfg = self.config["dataset"]

        # Try to load preprocessed data first
        processed_path = "./data/processed/sft"
        if os.path.exists(processed_path):
            logger.info(f"Loading preprocessed SFT data from: {processed_path}")
            return load_from_disk(processed_path)

        # Load from HuggingFace
        dataset_name = ds_cfg.get("name", "openai/gsm8k")
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, "main", split=ds_cfg.get("split", "train"))

        # Format with chat template
        def format_example(example):
            question = example["question"]
            answer = example["answer"]

            messages = [
                {"role": "system", "content": "You are a helpful math tutor. Solve the problem step by step, showing all your work clearly. End with 'The answer is: [answer]'."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                text = f"<|im_start|>system\nYou are a helpful math tutor.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"

            return {"text": text}

        dataset = dataset.map(format_example, remove_columns=dataset.column_names, desc="Formatting SFT data")

        # Split
        split = dataset.train_test_split(test_size=0.05, seed=42)
        return DatasetDict({"train": split["train"], "validation": split["test"]})

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    def train(self):
        """Run SFT training."""
        if self.model is None:
            self.setup_model()

        dataset = self.setup_dataset()
        train_cfg = self.config["training"]

        # Build base training arguments (standard TrainingArguments params)
        base_kwargs = dict(
            output_dir=train_cfg["output_dir"],
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            warmup_ratio=train_cfg["warmup_ratio"],
            weight_decay=train_cfg["weight_decay"],
            logging_steps=train_cfg["logging_steps"],
            save_steps=train_cfg["save_steps"],
            save_total_limit=train_cfg["save_total_limit"],
            bf16=train_cfg.get("bf16", True),
            gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
            optim=train_cfg.get("optim", "paged_adamw_32bit"),
            report_to=train_cfg.get("report_to", "none"),
            deepspeed=self.config.get("deepspeed_config"),
            eval_strategy="steps",
            eval_steps=train_cfg.get("save_steps", 200),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        # SFT-specific params — may live in SFTConfig or SFTTrainer depending on TRL version
        sft_specific = dict(
            max_seq_length=train_cfg.get("max_seq_length", 1024),
            dataset_text_field="text",
            packing=train_cfg.get("packing", False),
        )

        # Try adding SFT-specific params to SFTConfig; if rejected, pass to SFTTrainer
        trainer_extra_kwargs = {}
        try:
            training_args = SFTConfig(**base_kwargs, **sft_specific)
        except TypeError as e:
            logger.warning(f"SFTConfig rejected some params ({e}), passing to SFTTrainer instead")
            training_args = SFTConfig(**base_kwargs)
            trainer_extra_kwargs = sft_specific

        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            peft_config=self.peft_config,
            **trainer_extra_kwargs,
        )

        logger.info("Starting SFT training...")
        logger.info(f"  Model: {self.config['model']['name_or_path']}")
        logger.info(f"  Effective batch size: {train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']}")
        logger.info(f"  Learning rate: {train_cfg['learning_rate']}")
        logger.info(f"  Epochs: {train_cfg['num_train_epochs']}")

        # Train
        train_result = self.trainer.train()

        # Log metrics
        metrics = train_result.metrics
        logger.info(f"Training complete! Metrics: {metrics}")

        return metrics

    # ----------------------------------------------------------
    # Save & Export
    # ----------------------------------------------------------
    def save(self, output_dir: str = None):
        """Save the fine-tuned model (LoRA adapters only)."""
        output_dir = output_dir or os.path.join(self.config["training"]["output_dir"], "final")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving model to: {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training config for reproducibility
        config_path = os.path.join(output_dir, "training_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"Model saved. To load: AutoPeftModel.from_pretrained('{output_dir}')")

    def merge_and_save(self, output_dir: str = None):
        """Merge LoRA weights into base model and save full model."""
        output_dir = output_dir or os.path.join(self.config["training"]["output_dir"], "merged")
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Merging LoRA weights into base model...")
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Merged model saved to: {output_dir}")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SFT Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA after training")
    args = parser.parse_args()

    finetuner = SFTFineTuner(args.config)
    finetuner.train()
    finetuner.save()

    if args.merge:
        finetuner.merge_and_save()


if __name__ == "__main__":
    main()
