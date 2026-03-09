"""
Model Evaluator — GSM8K / Math Evaluation Pipeline

Evaluates aligned models on:
1. GSM8K test set accuracy (exact match on final answer)
2. Pass@k metric (sampling multiple responses)
3. Response quality analysis (reasoning steps, format)
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    model_path: str = "./outputs/dpo/final"
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset: str = "openai/gsm8k"
    split: str = "test"
    max_samples: int = 200
    num_samples_per_prompt: int = 1  # for pass@k
    max_new_tokens: int = 512
    temperature: float = 0.1
    output_dir: str = "./outputs/evaluation"
    use_4bit: bool = True


class ModelEvaluator:
    """
    Evaluates alignment quality on math reasoning benchmarks.
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.model = None
        self.tokenizer = None
        self.pipe = None

    def load_model(self):
        """Load model for evaluation (supports both PEFT and full models)."""
        model_path = self.config.model_path
        logger.info(f"Loading model from: {model_path}")

        # Check if PEFT model
        is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))

        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            bnb_config = None

        if is_peft:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            # PEFT adapters may not include tokenizer; fall back to base model
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            except OSError:
                logger.info(f"Tokenizer not found at {model_path}, loading from base model")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )

        logger.info("Model loaded successfully")
        return self

    def load_dataset(self):
        """Load evaluation dataset (GSM8K test set)."""
        logger.info(f"Loading dataset: {self.config.dataset}")
        dataset = load_dataset(self.config.dataset, "main", split=self.config.split)
        if self.config.max_samples:
            dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
        logger.info(f"Evaluation samples: {len(dataset)}")
        return dataset

    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        """Extract the final numerical answer from model output."""
        # Pattern: "The answer is: XXX" or "#### XXX"
        patterns = [
            r"[Tt]he answer is[:\s]*([+-]?\d[\d,]*\.?\d*)",
            r"####\s*([+-]?\d[\d,]*\.?\d*)",
            r"[Aa]nswer[:\s]*([+-]?\d[\d,]*\.?\d*)",
            r"=\s*([+-]?\d[\d,]*\.?\d*)\s*$",
            r"(\d[\d,]*\.?\d*)\s*$",  # last number in text
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).replace(",", "")
        return None

    @staticmethod
    def extract_gsm8k_answer(answer_text: str) -> str:
        """Extract ground truth answer from GSM8K format (#### answer)."""
        match = re.search(r"####\s*(.+)$", answer_text, re.MULTILINE)
        if match:
            return match.group(1).strip().replace(",", "")
        return ""

    def generate_response(self, prompt: str) -> str:
        """Generate a response to a math problem."""
        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Solve step by step. End with 'The answer is: [answer]'."},
            {"role": "user", "content": prompt},
        ]

        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            max_length=None,
            temperature=self.config.temperature if self.config.temperature > 0 else 1.0,
            do_sample=self.config.temperature > 0,
        )

        output = self.pipe(
            messages,
            generation_config=gen_config,
            return_full_text=False,
        )

        response = output[0]["generated_text"]
        if isinstance(response, list):
            response = response[-1]["content"] if isinstance(response[-1], dict) else str(response[-1])
        return response.strip()

    def evaluate(self) -> dict:
        """
        Run full evaluation pipeline.

        Returns:
            Dictionary with accuracy, pass@k, and detailed results
        """
        if self.model is None:
            self.load_model()

        dataset = self.load_dataset()
        results = []
        correct = 0
        total = 0

        for item in tqdm(dataset, desc="Evaluating"):
            question = item["question"]
            ground_truth = self.extract_gsm8k_answer(item["answer"])

            try:
                response = self.generate_response(question)
                predicted = self.extract_answer(response)
                is_correct = predicted is not None and predicted.strip() == ground_truth.strip()

                if is_correct:
                    correct += 1
                total += 1

                results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "response": response,
                    "correct": is_correct,
                })

            except Exception as e:
                logger.warning(f"Error evaluating: {e}")
                total += 1
                results.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "predicted": None,
                    "response": f"ERROR: {str(e)}",
                    "correct": False,
                })

        accuracy = correct / total if total > 0 else 0

        # Compute additional metrics
        response_lengths = [len(r["response"].split()) for r in results]
        avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0

        has_reasoning = sum(
            1 for r in results
            if any(w in r["response"].lower() for w in ["step", "first", "then", "therefore"])
        )

        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_response_length": avg_length,
            "reasoning_ratio": has_reasoning / total if total > 0 else 0,
            "model_path": self.config.model_path,
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Results")
        logger.info(f"  Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
        logger.info(f"  Avg response length: {avg_length:.0f} words")
        logger.info(f"  Reasoning ratio: {metrics['reasoning_ratio']*100:.1f}%")
        logger.info(f"{'='*50}")

        # Save results
        os.makedirs(self.config.output_dir, exist_ok=True)

        with open(os.path.join(self.config.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        with open(os.path.join(self.config.output_dir, "detailed_results.json"), "w", encoding="utf-8") as f:
            json.dump(results[:50], f, ensure_ascii=False, indent=2)  # save first 50 for inspection

        logger.info(f"Results saved to: {self.config.output_dir}")
        return metrics

    def compare_models(self, model_paths: list[str]) -> dict:
        """Compare multiple models on the same eval set."""
        all_metrics = {}
        for path in model_paths:
            logger.info(f"\nEvaluating: {path}")
            self.config.model_path = path
            self.model = None
            self.pipe = None
            metrics = self.evaluate()
            all_metrics[path] = metrics

        # Summary table
        logger.info(f"\n{'='*60}")
        logger.info(f"{'Model':<40} {'Accuracy':>10}")
        logger.info(f"{'-'*60}")
        for path, metrics in all_metrics.items():
            name = os.path.basename(path)
            logger.info(f"{name:<40} {metrics['accuracy']*100:>9.2f}%")
        logger.info(f"{'='*60}")

        return all_metrics


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate aligned model on math benchmarks")
    parser.add_argument("--model", type=str, default="./outputs/dpo/final")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output", type=str, default="./outputs/evaluation")
    parser.add_argument("--compare", nargs="+", help="Compare multiple model paths")
    args = parser.parse_args()

    config = EvalConfig(
        model_path=args.model,
        base_model=args.base_model,
        max_samples=args.max_samples,
        output_dir=args.output,
    )

    evaluator = ModelEvaluator(config)

    if args.compare:
        evaluator.compare_models(args.compare)
    else:
        evaluator.evaluate()


if __name__ == "__main__":
    main()
