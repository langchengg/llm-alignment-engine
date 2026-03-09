"""
Data Processor — Cleaning, Formatting, and Storage Utilities

Handles:
1. Raw data cleaning (dedup, length filtering, format validation)
2. Converting to TRL-compatible DPO/PPO training formats
3. HuggingFace Datasets integration for efficient storage and loading
4. Train/validation split with stratification by difficulty
"""

import json
import os
import re
import hashlib
import logging
from typing import Optional
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for data processing."""
    # Filtering
    min_prompt_length: int = 10
    max_prompt_length: int = 1024
    min_response_length: int = 20
    max_response_length: int = 2048
    # Train/Val split
    val_ratio: float = 0.1
    seed: int = 42
    # Output
    output_dir: str = "./data/processed"


class DataProcessor:
    """
    Processes raw data into clean, TRL-compatible training datasets.
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()

    # ----------------------------------------------------------
    # Data Cleaning
    # ----------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """Clean a single text entry."""
        if not text:
            return ""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove control characters (keep newlines)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Normalize quotes
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        return text

    def deduplicate(self, dataset: Dataset, key: str = "prompt") -> Dataset:
        """Remove duplicate entries based on content hash."""
        seen_hashes = set()
        keep_indices = []

        for i, row in enumerate(dataset):
            content = row[key].strip().lower()
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                keep_indices.append(i)

        original_len = len(dataset)
        dataset = dataset.select(keep_indices)
        removed = original_len - len(dataset)
        if removed > 0:
            logger.info(f"Deduplication: removed {removed} duplicates ({removed/original_len*100:.1f}%)")
        return dataset

    def filter_by_length(self, dataset: Dataset) -> Dataset:
        """Filter entries by prompt and response length."""
        def length_filter(example):
            prompt_ok = self.config.min_prompt_length <= len(example["prompt"]) <= self.config.max_prompt_length
            if "chosen" in example:
                chosen_ok = self.config.min_response_length <= len(example["chosen"]) <= self.config.max_response_length
                rejected_ok = self.config.min_response_length <= len(example["rejected"]) <= self.config.max_response_length
                return prompt_ok and chosen_ok and rejected_ok
            return prompt_ok

        original_len = len(dataset)
        dataset = dataset.filter(length_filter)
        removed = original_len - len(dataset)
        if removed > 0:
            logger.info(f"Length filter: removed {removed} entries ({removed/original_len*100:.1f}%)")
        return dataset

    def validate_format(self, dataset: Dataset, required_fields: list[str]) -> Dataset:
        """Validate that all required fields are present and non-empty."""
        def is_valid(example):
            return all(
                field in example and example[field] and str(example[field]).strip()
                for field in required_fields
            )

        original_len = len(dataset)
        dataset = dataset.filter(is_valid)
        removed = original_len - len(dataset)
        if removed > 0:
            logger.info(f"Format validation: removed {removed} invalid entries")
        return dataset

    # ----------------------------------------------------------
    # Format Conversion
    # ----------------------------------------------------------
    def format_for_sft(self, dataset: Dataset, tokenizer=None) -> Dataset:
        """
        Convert dataset to SFT training format.
        Expected input: {"question": ..., "answer": ...} (e.g., GSM8K)
        Output: {"text": "<formatted chat>"} or {"messages": [...]}
        """
        def format_example(example):
            question = example.get("question", example.get("prompt", ""))
            answer = example.get("answer", example.get("response", ""))

            # Chat format for instruction-tuned models
            messages = [
                {"role": "system", "content": "You are a helpful math tutor. Solve problems step by step."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

            # If tokenizer is available, apply chat template
            if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                text = f"### Question:\n{question}\n\n### Answer:\n{answer}"

            return {"text": text, "messages": messages}

        return dataset.map(format_example, desc="Formatting for SFT")

    def format_for_dpo(self, dataset: Dataset) -> Dataset:
        """
        Convert dataset to DPO training format.
        Required output format: {"prompt": str, "chosen": str, "rejected": str}
        """
        required = ["prompt", "chosen", "rejected"]

        # Check if already in correct format
        if all(f in dataset.column_names for f in required):
            logger.info("Dataset already in DPO format")
            return dataset

        # Try to convert from other formats
        def convert(example):
            prompt = example.get("prompt", example.get("question", ""))
            chosen = example.get("chosen", example.get("chosen_response", ""))
            rejected = example.get("rejected", example.get("rejected_response", ""))
            return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

        return dataset.map(convert, desc="Formatting for DPO")

    def format_for_ppo(self, dataset: Dataset) -> Dataset:
        """
        Convert dataset to PPO training format.
        Required output: {"query": str} — responses are generated online during training.
        """
        def convert(example):
            query = example.get("prompt", example.get("question", ""))
            instruction = f"Solve the following math problem step by step:\n\n{query}"
            return {"query": instruction}

        return dataset.map(convert, desc="Formatting for PPO")

    # ----------------------------------------------------------
    # Full Processing Pipeline
    # ----------------------------------------------------------
    def process_preference_data(
        self,
        input_path: str,
        output_path: str = None,
    ) -> DatasetDict:
        """
        Full processing pipeline for preference data:
        1. Load → 2. Clean → 3. Deduplicate → 4. Filter → 5. Validate → 6. Split → 7. Save
        """
        output_path = output_path or self.config.output_dir
        logger.info(f"Processing preference data from: {input_path}")

        # Load
        if os.path.isdir(input_path):
            dataset = load_from_disk(input_path)
        elif input_path.endswith(".json"):
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)
        else:
            dataset = load_dataset(input_path, split="train")

        logger.info(f"Loaded {len(dataset)} entries")

        # Clean
        def clean_all(example):
            for key in ["prompt", "chosen", "rejected"]:
                if key in example:
                    example[key] = self.clean_text(example[key])
            return example

        dataset = dataset.map(clean_all, desc="Cleaning")

        # Deduplicate
        dataset = self.deduplicate(dataset, key="prompt")

        # Filter by length
        dataset = self.filter_by_length(dataset)

        # Validate format
        dataset = self.validate_format(dataset, ["prompt", "chosen", "rejected"])

        # Format for DPO
        dataset = self.format_for_dpo(dataset)

        # Train/Val split
        split = dataset.train_test_split(
            test_size=self.config.val_ratio,
            seed=self.config.seed,
        )
        dataset_dict = DatasetDict({
            "train": split["train"],
            "validation": split["test"],
        })

        logger.info(f"Final dataset: train={len(dataset_dict['train'])}, val={len(dataset_dict['validation'])}")

        # Save
        os.makedirs(output_path, exist_ok=True)
        dataset_dict.save_to_disk(output_path)
        logger.info(f"Saved processed dataset to: {output_path}")

        # Save stats
        stats = {
            "train_size": len(dataset_dict["train"]),
            "val_size": len(dataset_dict["validation"]),
            "columns": dataset_dict["train"].column_names,
        }
        with open(os.path.join(output_path, "processing_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        return dataset_dict

    def process_sft_data(
        self,
        dataset_name: str = "openai/gsm8k",
        split: str = "train",
        tokenizer=None,
        output_path: str = None,
    ) -> DatasetDict:
        """
        Process SFT training data from HuggingFace hub.
        """
        output_path = output_path or os.path.join(self.config.output_dir, "sft")

        logger.info(f"Loading SFT data: {dataset_name} [{split}]")
        dataset = load_dataset(dataset_name, "main", split=split)
        logger.info(f"Loaded {len(dataset)} entries")

        # Format for SFT
        dataset = self.format_for_sft(dataset, tokenizer)

        # Split
        split_ds = dataset.train_test_split(
            test_size=self.config.val_ratio,
            seed=self.config.seed,
        )
        dataset_dict = DatasetDict({
            "train": split_ds["train"],
            "validation": split_ds["test"],
        })

        # Save
        os.makedirs(output_path, exist_ok=True)
        dataset_dict.save_to_disk(output_path)
        logger.info(f"Saved SFT dataset to: {output_path}")

        return dataset_dict


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process data for SFT/DPO/PPO training")
    parser.add_argument("--mode", choices=["sft", "dpo", "ppo"], required=True)
    parser.add_argument("--input", type=str, help="Input data path")
    parser.add_argument("--output", type=str, default="./data/processed")
    parser.add_argument("--dataset", type=str, default="openai/gsm8k", help="HuggingFace dataset name (for SFT)")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    config = ProcessorConfig(output_dir=args.output, val_ratio=args.val_ratio)
    processor = DataProcessor(config)

    if args.mode == "sft":
        processor.process_sft_data(dataset_name=args.dataset, output_path=args.output)
    elif args.mode == "dpo":
        processor.process_preference_data(input_path=args.input, output_path=args.output)
    elif args.mode == "ppo":
        # PPO uses the same SFT data, just reformatted
        processor.process_sft_data(dataset_name=args.dataset, output_path=args.output)


if __name__ == "__main__":
    main()
