"""
PPO Trainer — Proximal Policy Optimization with QLoRA + Reward Model

Stage 2b of the alignment pipeline (alternative to DPO):
- RLHF-style training with online reward model feedback
- Uses TRL PPOTrainer with adaptive KL penalty
- More compute-intensive than DPO but supports online learning
- Demonstrates understanding of full RLHF pipeline

Architecture:
  Policy Model (QLoRA) ──> Generate response ──> Reward Model ──> Score
       ↑                                                          |
       └─────────── PPO update (maximize reward - KL penalty) ────┘
"""

import os
import yaml
import logging
from typing import Optional

import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPOFineTuner:
    """
    PPO (RLHF) alignment pipeline.

    Architecture:
    1. Load SFT model as policy
    2. Load reward model for scoring
    3. Generate responses → Score with RM → Update policy with PPO

    Usage:
        finetuner = PPOFineTuner("configs/ppo_config.yaml")
        finetuner.train()
        finetuner.save()
    """

    def __init__(self, config_path: str = "configs/ppo_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.ref_model = None
        self.reward_model = None
        self.reward_tokenizer = None
        self.tokenizer = None
        self.trainer = None

    # ----------------------------------------------------------
    # Model Setup
    # ----------------------------------------------------------
    def setup_models(self):
        """Load policy model, reference model, and reward model."""
        # ==================== Policy Model ====================
        sft_checkpoint = self.config["model"].get("sft_checkpoint")
        model_name = self.config["model"]["name_or_path"]

        if sft_checkpoint and os.path.exists(sft_checkpoint):
            logger.info(f"Loading SFT policy model from: {sft_checkpoint}")
            base_model = model_name
        else:
            logger.info(f"Using base model as policy: {model_name}")
            base_model = model_name

        # QLoRA config
        qlora_cfg = self.config["qlora"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_cfg["load_in_4bit"],
            bnb_4bit_compute_dtype=getattr(torch, qlora_cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=qlora_cfg["bnb_4bit_use_double_quant"],
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # LoRA config
        lora_config = LoraConfig(
            r=qlora_cfg["lora_r"],
            lora_alpha=qlora_cfg["lora_alpha"],
            lora_dropout=qlora_cfg["lora_dropout"],
            target_modules=qlora_cfg["target_modules"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Load model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=getattr(torch, self.config["model"].get("torch_dtype", "bfloat16")),
            peft_config=lora_config,
        )

        # ==================== Reward Model ====================
        rm_cfg = self.config.get("reward_model", {})
        rm_name = rm_cfg.get("name_or_path", "weqweasdas/RM-Gemma-2B")
        logger.info(f"Loading reward model: {rm_name}")

        try:
            # Try to load as sequence classification model
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                rm_name,
                torch_dtype=getattr(torch, rm_cfg.get("torch_dtype", "bfloat16")),
                device_map="auto",
                trust_remote_code=True,
                num_labels=1,
            )
            self.reward_tokenizer = AutoTokenizer.from_pretrained(rm_name, trust_remote_code=True)
            if self.reward_tokenizer.pad_token is None:
                self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
            self.reward_pipeline = None
        except Exception as e:
            logger.warning(f"Could not load reward model as classifier: {e}")
            logger.info("Using sentiment pipeline as fallback reward model")
            self.reward_pipeline = pipeline(
                "sentiment-analysis",
                model="lvwerra/distilbert-imdb",
                device_map="auto",
            )
            self.reward_model = None

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Policy model - Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

        return self

    # ----------------------------------------------------------
    # Dataset Setup
    # ----------------------------------------------------------
    def setup_dataset(self):
        """Load prompts/queries for PPO training."""
        ds_cfg = self.config["dataset"]
        dataset_name = ds_cfg.get("name", "openai/gsm8k")
        split = ds_cfg.get("split", "train[:1000]")

        logger.info(f"Loading dataset: {dataset_name} [{split}]")
        dataset = load_dataset(dataset_name, "main", split=split)

        # Format queries
        def format_query(example):
            query = f"Solve the following math problem step by step:\n\n{example['question']}"
            return {"query": query}

        dataset = dataset.map(format_query, remove_columns=dataset.column_names)
        return dataset

    # ----------------------------------------------------------
    # Reward Computation
    # ----------------------------------------------------------
    def compute_reward(self, query: str, response: str) -> float:
        """Compute reward score for a (query, response) pair."""
        combined = f"{query}\n\n{response}"

        if self.reward_model is not None:
            # Use reward model
            inputs = self.reward_tokenizer(
                combined,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.reward_model.device)

            with torch.no_grad():
                outputs = self.reward_model(**inputs)
                # For sequence classification, logits is the reward
                reward = outputs.logits.squeeze().item()
            return reward

        elif self.reward_pipeline is not None:
            # Fallback: use sentiment as proxy reward
            result = self.reward_pipeline(combined[:512], return_all_scores=False)
            score = result[0]["score"] if result[0]["label"] == "POSITIVE" else -result[0]["score"]
            return score

        else:
            # Heuristic reward based on response quality
            return self._heuristic_reward(query, response)

    @staticmethod
    def _heuristic_reward(query: str, response: str) -> float:
        """
        Simple heuristic reward for math responses.
        Useful as a fallback when no reward model is available.
        """
        reward = 0.0

        # Length reward (prefer detailed responses, penalize too short/long)
        length = len(response.split())
        if 50 <= length <= 300:
            reward += 1.0
        elif length < 20:
            reward -= 2.0
        elif length > 500:
            reward -= 0.5

        # Structure reward (has step-by-step reasoning)
        step_keywords = ["step", "first", "then", "next", "therefore", "so", "thus", "hence"]
        step_count = sum(1 for word in step_keywords if word in response.lower())
        reward += min(step_count * 0.3, 1.5)

        # Has final answer
        if any(phrase in response.lower() for phrase in ["the answer is", "answer:", "= "]):
            reward += 1.5

        # Has mathematical notation
        if any(char in response for char in ["×", "÷", "+", "=", "²", "³"]):
            reward += 0.5

        # Penalize repetition
        sentences = response.split(".")
        unique_ratio = len(set(sentences)) / max(len(sentences), 1)
        if unique_ratio < 0.5:
            reward -= 2.0  # heavy penalty for repetitive output (mode collapse signal)

        return reward

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    def train(self):
        """Run PPO training loop."""
        if self.model is None:
            self.setup_models()

        dataset = self.setup_dataset()
        ppo_cfg = self.config["ppo"]
        train_cfg = self.config["training"]

        # PPO Config
        ppo_config = PPOConfig(
            learning_rate=ppo_cfg["learning_rate"],
            batch_size=ppo_cfg["batch_size"],
            mini_batch_size=ppo_cfg["mini_batch_size"],
            gradient_accumulation_steps=ppo_cfg["gradient_accumulation_steps"],
            ppo_epochs=ppo_cfg["ppo_epochs"],
            init_kl_coef=ppo_cfg["init_kl_coef"],
            adap_kl_ctrl=ppo_cfg["adap_kl_ctrl"],
            adap_kl_target=ppo_cfg.get("adap_kl_target", 6.0),
            cliprange=ppo_cfg["cliprange"],
            cliprange_value=ppo_cfg["cliprange_value"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            log_with=train_cfg.get("report_to", "none"),
            project_kwargs={"logging_dir": os.path.join(train_cfg["output_dir"], "logs")},
        )

        # Initialize PPO Trainer
        self.trainer = PPOTrainer(
            model=self.model,
            config=ppo_config,
            dataset=dataset,
            tokenizer=self.tokenizer,
        )

        logger.info("Starting PPO training...")
        logger.info(f"  Batch size: {ppo_cfg['batch_size']}")
        logger.info(f"  Init KL coef: {ppo_cfg['init_kl_coef']}")
        logger.info(f"  PPO epochs: {ppo_cfg['ppo_epochs']}")
        logger.info(f"  Dataset size: {len(dataset)}")

        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": ppo_cfg.get("response_max_length", 512),
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Training loop
        all_stats = []
        for epoch in range(train_cfg.get("num_train_epochs", 1)):
            logger.info(f"\nEpoch {epoch + 1}/{train_cfg.get('num_train_epochs', 1)}")

            for batch_idx, batch in enumerate(tqdm(
                self.trainer.dataloader,
                desc=f"PPO Epoch {epoch+1}",
            )):
                query_tensors = batch["input_ids"]

                # ---- Step 1: Generate responses ----
                response_tensors = []
                for query in query_tensors:
                    response = self.trainer.generate(query, **gen_kwargs)
                    response_tensors.append(response.squeeze())

                # Decode
                batch_queries = [self.tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors]
                batch_responses = [
                    self.tokenizer.decode(r[len(q):], skip_special_tokens=True)
                    for q, r in zip(query_tensors, response_tensors)
                ]

                # ---- Step 2: Compute rewards ----
                rewards = []
                for query_text, response_text in zip(batch_queries, batch_responses):
                    reward = self.compute_reward(query_text, response_text)
                    rewards.append(torch.tensor(reward))

                # ---- Step 3: PPO update ----
                stats = self.trainer.step(query_tensors, response_tensors, rewards)
                all_stats.append(stats)

                # Logging
                if batch_idx % train_cfg.get("logging_steps", 5) == 0:
                    mean_reward = sum(r.item() for r in rewards) / len(rewards)
                    kl = stats.get("objective/kl", 0)
                    logger.info(
                        f"  Batch {batch_idx} | "
                        f"Reward: {mean_reward:.3f} | "
                        f"KL: {kl:.4f} | "
                        f"Entropy: {stats.get('objective/entropy', 0):.4f}"
                    )

                # Save checkpoint
                if batch_idx > 0 and batch_idx % train_cfg.get("save_steps", 100) == 0:
                    save_path = os.path.join(train_cfg["output_dir"], f"checkpoint-{batch_idx}")
                    self.trainer.save_pretrained(save_path)
                    logger.info(f"  Checkpoint saved: {save_path}")

        logger.info("\nPPO training complete!")
        return all_stats

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------
    def save(self, output_dir: str = None):
        """Save the PPO-trained model."""
        output_dir = output_dir or os.path.join(self.config["training"]["output_dir"], "final")
        os.makedirs(output_dir, exist_ok=True)

        self.trainer.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "training_config.yaml"), "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"PPO model saved to: {output_dir}")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PPO (RLHF) Alignment Training")
    parser.add_argument("--config", type=str, default="configs/ppo_config.yaml")
    args = parser.parse_args()

    finetuner = PPOFineTuner(args.config)
    finetuner.train()
    finetuner.save()


if __name__ == "__main__":
    main()
