"""
DPO Trainer — Direct Preference Optimization with QLoRA + DeepSpeed

Stage 2a of the alignment pipeline (alternative to PPO):
- Aligns the SFT model using preference data (chosen/rejected pairs)
- DPO eliminates the need for a separate reward model
- Supports beta sweep for Mode Collapse analysis
- More stable and memory-efficient than PPO on single GPU

Key insight: beta (KL penalty) controls the trade-off:
  - Low beta → aggressive learning → risk of mode collapse
  - High beta → conservative → model barely changes from SFT
"""

import os
import yaml
import logging
from typing import Optional

import torch
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, AutoPeftModelForCausalLM
from trl import DPOTrainer, DPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DPOFineTuner:
    """
    DPO alignment pipeline.

    Usage:
        finetuner = DPOFineTuner("configs/dpo_config.yaml")
        finetuner.train()
        finetuner.save()

    For beta ablation:
        finetuner.run_beta_ablation()
    """

    def __init__(self, config_path: str = "configs/dpo_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None
        self.peft_config = None

    # ----------------------------------------------------------
    # Model Setup
    # ----------------------------------------------------------
    def setup_model(self):
        """Load SFT checkpoint with QLoRA for DPO training."""
        # Determine model path: use SFT checkpoint if available
        sft_checkpoint = self.config["model"].get("sft_checkpoint")
        model_name = self.config["model"]["name_or_path"]

        if sft_checkpoint and os.path.exists(sft_checkpoint):
            logger.info(f"Loading SFT checkpoint: {sft_checkpoint}")
            base_model_name = model_name
        else:
            logger.info(f"No SFT checkpoint found, using base model: {model_name}")
            base_model_name = model_name
            sft_checkpoint = None

        # QLoRA config
        qlora_cfg = self.config["qlora"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_cfg["load_in_4bit"],
            bnb_4bit_compute_dtype=getattr(torch, qlora_cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=qlora_cfg["bnb_4bit_use_double_quant"],
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="left",  # left padding for generation
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if sft_checkpoint:
            # Load PEFT model from SFT checkpoint
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                sft_checkpoint,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=getattr(torch, self.config["model"].get("torch_dtype", "bfloat16")),
                is_trainable=True,
            )
        else:
            # Load base model + add new LoRA
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=getattr(torch, self.config["model"].get("torch_dtype", "bfloat16")),
            )
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)

        # LoRA config for DPO
        self.peft_config = LoraConfig(
            r=qlora_cfg["lora_r"],
            lora_alpha=qlora_cfg["lora_alpha"],
            lora_dropout=qlora_cfg["lora_dropout"],
            target_modules=qlora_cfg["target_modules"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Note: DPO uses the model itself as implicit reference
        # (no need for separate ref_model with PEFT)
        self.ref_model = None  # DPOTrainer handles this internally with PEFT

        trainable, total = 0, 0
        for p in self.model.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

        return self

    # ----------------------------------------------------------
    # Dataset Setup
    # ----------------------------------------------------------
    def setup_dataset(self) -> DatasetDict:
        """Load preference dataset for DPO training."""
        ds_cfg = self.config["dataset"]
        data_path = ds_cfg.get("path", "./data/preference_data")

        # Try to load preprocessed preference data
        processed_path = "./data/processed"
        if os.path.exists(processed_path):
            logger.info(f"Loading processed preference data from: {processed_path}")
            dataset = load_from_disk(processed_path)
            if isinstance(dataset, DatasetDict):
                return dataset

        if os.path.exists(data_path):
            logger.info(f"Loading preference data from: {data_path}")
            dataset = load_from_disk(data_path)
        else:
            # Fallback: use HuggingFace preference dataset
            logger.info("No local preference data found. Using HuggingFace preference dataset...")
            from datasets import load_dataset
            dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train[:2000]")

            # Rename columns to match expected format
            def format_row(example):
                return {
                    "prompt": example.get("instruction", example.get("prompt", "")),
                    "chosen": example.get("chosen", ""),
                    "rejected": example.get("rejected", ""),
                }
            dataset = dataset.map(format_row, desc="Formatting DPO data")

        # Ensure correct columns
        required = {"prompt", "chosen", "rejected"}
        if not required.issubset(set(dataset.column_names if not isinstance(dataset, DatasetDict) else dataset["train"].column_names)):
            raise ValueError(f"Dataset must contain columns: {required}")

        # Apply chat template to format prompt/chosen/rejected
        def apply_template(example):
            # Format prompt
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]

            # Wrap in chat format
            example["prompt"] = f"Solve the following math problem step by step:\n\n{prompt}" if not prompt.startswith("Solve") else prompt
            return example

        if isinstance(dataset, DatasetDict):
            dataset = dataset.map(apply_template)
            return dataset

        dataset = dataset.map(apply_template)
        split = dataset.train_test_split(test_size=0.05, seed=42)
        return DatasetDict({"train": split["train"], "validation": split["test"]})

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    def train(self, beta: float = None):
        """Run DPO training."""
        if self.model is None:
            self.setup_model()

        dataset = self.setup_dataset()
        train_cfg = self.config["training"]
        dpo_cfg = self.config["dpo"]

        # Override beta if provided (for ablation study)
        beta_value = beta or dpo_cfg["beta"]

        # Training arguments
        training_args = DPOConfig(
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
            # DeepSpeed
            deepspeed=self.config.get("deepspeed_config"),
            # DPO specific
            beta=beta_value,
            loss_type=dpo_cfg.get("loss_type", "sigmoid"),
            label_smoothing=dpo_cfg.get("label_smoothing", 0.0),
            max_prompt_length=dpo_cfg.get("max_prompt_length", 512),
            max_length=dpo_cfg.get("max_length", 1024),
            # Evaluation
            eval_strategy="steps",
            eval_steps=train_cfg.get("save_steps", 100),
            remove_unused_columns=False,
        )

        # Initialize DPO trainer
        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=self.tokenizer,
            peft_config=self.peft_config if not hasattr(self.model, "peft_config") else None,
        )

        logger.info(f"Starting DPO training with beta={beta_value}...")
        logger.info(f"  Loss type: {dpo_cfg.get('loss_type', 'sigmoid')}")
        logger.info(f"  Training samples: {len(dataset['train'])}")

        # Train
        train_result = self.trainer.train()
        metrics = train_result.metrics

        # Log DPO-specific metrics
        logger.info(f"DPO Training complete!")
        logger.info(f"  Beta: {beta_value}")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v}")

        return metrics

    # ----------------------------------------------------------
    # Beta Ablation Study (Mode Collapse Experiment)
    # ----------------------------------------------------------
    def run_beta_ablation(self) -> dict:
        """
        Run DPO training with different beta values to study mode collapse.

        This is the KEY experiment for your resume:
        - Low beta → model learns aggressively → mode collapse (outputs become uniform)
        - High beta → model is too conservative → barely improves over SFT
        - Sweet spot → best trade-off between alignment and diversity
        """
        ablation_cfg = self.config.get("ablation", {})
        beta_values = ablation_cfg.get("beta_values", [0.05, 0.1, 0.2, 0.5])

        results = {}
        for beta in beta_values:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Running DPO with beta = {beta}")
            logger.info(f"{'='*60}")

            # Reset model for each run
            self.model = None
            self.trainer = None
            self.config["training"]["output_dir"] = f"./outputs/dpo_beta_{beta}"

            try:
                metrics = self.train(beta=beta)
                results[beta] = {
                    "metrics": metrics,
                    "status": "success",
                }
            except Exception as e:
                logger.error(f"Beta={beta} failed: {e}")
                results[beta] = {
                    "metrics": {},
                    "status": f"failed: {str(e)}",
                }

        # Save ablation results
        import json
        os.makedirs("./outputs/ablation", exist_ok=True)
        with open("./outputs/ablation/beta_sweep_results.json", "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2, default=str)

        logger.info(f"\nBeta ablation complete! Results saved to ./outputs/ablation/")
        return results

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------
    def save(self, output_dir: str = None):
        """Save the DPO-aligned model."""
        output_dir = output_dir or os.path.join(self.config["training"]["output_dir"], "final")
        os.makedirs(output_dir, exist_ok=True)

        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "training_config.yaml"), "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"DPO model saved to: {output_dir}")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DPO Alignment Training")
    parser.add_argument("--config", type=str, default="configs/dpo_config.yaml")
    parser.add_argument("--beta", type=float, default=None, help="Override beta value")
    parser.add_argument("--ablation", action="store_true", help="Run beta ablation study")
    args = parser.parse_args()

    finetuner = DPOFineTuner(args.config)

    if args.ablation:
        finetuner.run_beta_ablation()
    else:
        finetuner.train(beta=args.beta)
        finetuner.save()


if __name__ == "__main__":
    main()
