"""
Preference Data Generator — LLM-as-a-Judge + Reject Sampling Pipeline

This module automatically generates high-quality preference data (chosen/rejected pairs)
for DPO/PPO alignment training. It uses an LLM to:
1. Generate multiple candidate responses for each math prompt
2. Score each response using LLM-as-a-Judge
3. Construct preference pairs via reject sampling

Designed to work with free/open-source models via HuggingFace Inference API
or local model inference.
"""

import json
import os
import re
import time
import random
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

@dataclass
class GeneratorConfig:
    """Configuration for the preference data generator."""
    # Model for generating candidate responses
    generator_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # Model for judging response quality (can be same as generator)
    judge_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # Number of candidate responses per prompt
    num_candidates: int = 4
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    # Reject sampling: minimum score gap between chosen and rejected
    min_score_gap: float = 2.0
    # Output settings
    output_dir: str = "./data/preference_data"
    # Device
    device: str = "auto"
    # Batch processing
    batch_size: int = 4
    # Use HuggingFace Inference API instead of local model
    use_api: bool = False
    api_url: str = "https://api-inference.huggingface.co/models/"
    api_token: Optional[str] = None


# ============================================================
# Judge Prompt Template
# ============================================================

JUDGE_SYSTEM_PROMPT = """You are a math reasoning evaluator. Score the given response to a math problem on a scale of 1-10.

Scoring criteria:
- Correctness (0-4 points): Is the final answer correct?
- Reasoning (0-3 points): Are the intermediate steps logical and clear?
- Completeness (0-2 points): Are all steps shown?
- Clarity (0-1 point): Is the explanation easy to follow?

Respond with ONLY a JSON object: {"score": <number>, "reason": "<brief explanation>"}"""

JUDGE_USER_TEMPLATE = """Math Problem: {prompt}

Response to evaluate:
{response}

Evaluate this response. Respond with ONLY a JSON: {{"score": <1-10>, "reason": "<explanation>"}}"""


# ============================================================
# Preference Generator
# ============================================================

class PreferenceGenerator:
    """
    Generates preference data (chosen/rejected pairs) for DPO/PPO training.

    Pipeline:
    1. Load seed prompts
    2. For each prompt, generate N candidate responses
    3. Score each response with LLM-as-a-Judge
    4. Apply reject sampling to build preference pairs
    5. Save as HuggingFace Dataset
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.generator = None
        self.judge = None
        self.tokenizer = None
        self._setup_models()

    def _setup_models(self):
        """Initialize generator and judge models."""
        logger.info(f"Loading generator model: {self.config.generator_model}")

        # Determine device
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device

        if not self.config.use_api:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.generator_model,
                trust_remote_code=True,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model in 4-bit for memory efficiency
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.config.generator_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device_map="auto",
            )

            # If judge model is same as generator, reuse the pipeline
            if self.config.judge_model == self.config.generator_model:
                self.judge = self.generator
                logger.info("Reusing generator model as judge (same model)")
            else:
                logger.info(f"Loading separate judge model: {self.config.judge_model}")
                judge_model = AutoModelForCausalLM.from_pretrained(
                    self.config.judge_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
                judge_tokenizer = AutoTokenizer.from_pretrained(
                    self.config.judge_model,
                    trust_remote_code=True,
                )
                self.judge = pipeline(
                    "text-generation",
                    model=judge_model,
                    tokenizer=judge_tokenizer,
                    device_map="auto",
                )
        else:
            logger.info("Using HuggingFace Inference API mode")

    # ----------------------------------------------------------
    # Step 1: Load Prompts
    # ----------------------------------------------------------
    def load_prompts(self, prompts_path: str = None) -> list[dict]:
        """Load seed prompts from JSON file or HuggingFace dataset."""
        if prompts_path and os.path.exists(prompts_path):
            logger.info(f"Loading prompts from: {prompts_path}")
            with open(prompts_path, "r", encoding="utf-8") as f:
                prompts = json.load(f)
            return prompts

        # Fallback: load GSM8K from HuggingFace
        logger.info("Loading prompts from GSM8K dataset...")
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train[:200]")
        prompts = [
            {"id": f"gsm8k_{i}", "prompt": row["question"], "category": "gsm8k", "difficulty": "medium"}
            for i, row in enumerate(ds)
        ]
        return prompts

    # ----------------------------------------------------------
    # Step 2: Generate Candidate Responses
    # ----------------------------------------------------------
    def generate_candidates(self, prompt: str) -> list[str]:
        """Generate N candidate responses for a given math prompt."""
        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Solve the problem step by step, showing all work clearly. End with 'The answer is: [final answer]'."},
            {"role": "user", "content": prompt},
        ]

        if self.config.use_api:
            return self._generate_via_api(messages)

        candidates = []
        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            max_length=None,  # avoid conflict with max_new_tokens
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            do_sample=True,
        )
        for _ in range(self.config.num_candidates):
            try:
                output = self.generator(
                    messages,
                    generation_config=gen_config,
                    return_full_text=False,
                )
                response = output[0]["generated_text"]
                if isinstance(response, list):
                    # Chat format returns list of dicts
                    response = response[-1]["content"] if isinstance(response[-1], dict) else str(response[-1])
                candidates.append(response.strip())
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                candidates.append("")

        return candidates

    def _generate_via_api(self, messages: list[dict]) -> list[str]:
        """Generate responses via HuggingFace Inference API."""
        import requests

        headers = {}
        if self.config.api_token:
            headers["Authorization"] = f"Bearer {self.config.api_token}"

        url = f"{self.config.api_url}{self.config.generator_model}"
        candidates = []

        for _ in range(self.config.num_candidates):
            payload = {
                "inputs": messages[-1]["content"],  # simplified for API
                "parameters": {
                    "max_new_tokens": self.config.max_new_tokens,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "do_sample": True,
                    "return_full_text": False,
                }
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                result = resp.json()
                text = result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]
                candidates.append(text.strip())
            except Exception as e:
                logger.warning(f"API call failed: {e}")
                candidates.append("")
                time.sleep(1)  # rate limiting

        return candidates

    # ----------------------------------------------------------
    # Step 3: Judge / Score Responses
    # ----------------------------------------------------------
    def judge_response(self, prompt: str, response: str) -> dict:
        """Score a single response using LLM-as-a-Judge."""
        judge_prompt = JUDGE_USER_TEMPLATE.format(prompt=prompt, response=response)

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_prompt},
        ]

        try:
            if self.config.use_api:
                output_text = self._judge_via_api(judge_prompt)
            else:
                judge_gen_config = GenerationConfig(
                    max_new_tokens=100,
                    max_length=None,
                    temperature=0.1,  # low temp for consistent scoring
                    do_sample=True,
                )
                output = self.judge(
                    messages,
                    generation_config=judge_gen_config,
                    return_full_text=False,
                )
                output_text = output[0]["generated_text"]
                if isinstance(output_text, list):
                    output_text = output_text[-1]["content"] if isinstance(output_text[-1], dict) else str(output_text[-1])

            # Parse JSON from output
            score_data = self._parse_judge_output(output_text)
            return score_data

        except Exception as e:
            logger.warning(f"Judging failed: {e}")
            return {"score": 0, "reason": f"Judge error: {str(e)}"}

    def _judge_via_api(self, prompt: str) -> str:
        """Judge via API."""
        import requests
        headers = {}
        if self.config.api_token:
            headers["Authorization"] = f"Bearer {self.config.api_token}"

        url = f"{self.config.api_url}{self.config.judge_model}"
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 100, "temperature": 0.1}
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        return result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]

    @staticmethod
    def _parse_judge_output(text: str) -> dict:
        """Parse the judge's JSON output, with fallback for malformed responses."""
        # Try direct JSON parse
        try:
            # Find JSON in the text
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0))
                reason = data.get("reason", "")
                return {"score": min(max(score, 0), 10), "reason": reason}
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to extract a number
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        if numbers:
            score = float(numbers[0])
            return {"score": min(max(score, 0), 10), "reason": text[:200]}

        return {"score": 0, "reason": "Failed to parse judge output"}

    # ----------------------------------------------------------
    # Step 4: Reject Sampling → Build Preference Pairs
    # ----------------------------------------------------------
    def build_preference_pair(
        self,
        prompt: str,
        candidates: list[str],
        scores: list[dict],
    ) -> Optional[dict]:
        """
        From scored candidates, select (chosen, rejected) pair via reject sampling.

        Strategy:
        - chosen = highest scoring response
        - rejected = lowest scoring response
        - Only create pair if score gap >= min_score_gap
        """
        if not candidates or not scores:
            return None

        # Pair up (candidate, score) and sort by score descending
        scored = [(c, s) for c, s in zip(candidates, scores) if c.strip()]
        if len(scored) < 2:
            return None

        scored.sort(key=lambda x: x[1]["score"], reverse=True)

        best_candidate, best_score = scored[0]
        worst_candidate, worst_score = scored[-1]

        score_gap = best_score["score"] - worst_score["score"]

        if score_gap < self.config.min_score_gap:
            logger.debug(f"Score gap ({score_gap:.1f}) below threshold, skipping")
            return None

        return {
            "prompt": prompt,
            "chosen": best_candidate,
            "rejected": worst_candidate,
            "chosen_score": best_score["score"],
            "rejected_score": worst_score["score"],
            "score_gap": score_gap,
            "chosen_reason": best_score["reason"],
            "rejected_reason": worst_score["reason"],
        }

    # ----------------------------------------------------------
    # Step 5: Run Full Pipeline
    # ----------------------------------------------------------
    def generate_dataset(
        self,
        prompts_path: str = None,
        max_samples: int = None,
        save: bool = True,
    ) -> Dataset:
        """
        Run the complete preference data generation pipeline.

        Args:
            prompts_path: Path to seed prompts JSON
            max_samples: Max number of preference pairs to generate
            save: Whether to save to disk

        Returns:
            HuggingFace Dataset with preference pairs
        """
        # Load prompts
        prompts = self.load_prompts(prompts_path)
        if max_samples:
            prompts = prompts[:max_samples]

        logger.info(f"Processing {len(prompts)} prompts, generating {self.config.num_candidates} candidates each")

        preference_pairs = []
        stats = {"total_prompts": 0, "successful_pairs": 0, "skipped_low_gap": 0, "errors": 0}

        for item in tqdm(prompts, desc="Generating preference data"):
            stats["total_prompts"] += 1
            prompt_text = item["prompt"]

            try:
                # Generate candidates
                candidates = self.generate_candidates(prompt_text)

                # Score each candidate
                scores = []
                for candidate in candidates:
                    if candidate.strip():
                        score = self.judge_response(prompt_text, candidate)
                        scores.append(score)
                    else:
                        scores.append({"score": 0, "reason": "Empty response"})

                # Build preference pair
                pair = self.build_preference_pair(prompt_text, candidates, scores)

                if pair:
                    pair["id"] = item.get("id", f"gen_{stats['total_prompts']}")
                    pair["category"] = item.get("category", "unknown")
                    preference_pairs.append(pair)
                    stats["successful_pairs"] += 1
                else:
                    stats["skipped_low_gap"] += 1

            except Exception as e:
                logger.error(f"Error processing prompt {item.get('id', '?')}: {e}")
                stats["errors"] += 1

        # Log statistics
        logger.info(f"\n{'='*50}")
        logger.info(f"Preference Data Generation Complete")
        logger.info(f"  Total prompts processed: {stats['total_prompts']}")
        logger.info(f"  Successful pairs: {stats['successful_pairs']}")
        logger.info(f"  Skipped (low score gap): {stats['skipped_low_gap']}")
        logger.info(f"  Errors: {stats['errors']}")
        logger.info(f"  Success rate: {stats['successful_pairs']/max(stats['total_prompts'],1)*100:.1f}%")
        logger.info(f"{'='*50}")

        # Convert to HuggingFace Dataset
        if preference_pairs:
            dataset = Dataset.from_list(preference_pairs)
        else:
            logger.warning("No preference pairs generated!")
            dataset = Dataset.from_dict({
                "prompt": [], "chosen": [], "rejected": [],
                "chosen_score": [], "rejected_score": [], "score_gap": [],
            })

        # Save
        if save and preference_pairs:
            os.makedirs(self.config.output_dir, exist_ok=True)
            dataset.save_to_disk(self.config.output_dir)
            logger.info(f"Dataset saved to: {self.config.output_dir}")

            # Also save as JSON for inspection
            json_path = os.path.join(self.config.output_dir, "preference_pairs.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(preference_pairs, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON saved to: {json_path}")

            # Save generation stats
            stats_path = os.path.join(self.config.output_dir, "generation_stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)

        return dataset


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate preference data for DPO/PPO training")
    parser.add_argument("--prompts", type=str, default="data/math_prompts.json", help="Path to seed prompts")
    parser.add_argument("--generator-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--judge-model", type=str, default=None, help="Judge model (default: same as generator)")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--min-score-gap", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./data/preference_data")
    parser.add_argument("--use-api", action="store_true", help="Use HuggingFace Inference API")
    parser.add_argument("--api-token", type=str, default=None)
    args = parser.parse_args()

    config = GeneratorConfig(
        generator_model=args.generator_model,
        judge_model=args.judge_model or args.generator_model,
        num_candidates=args.num_candidates,
        min_score_gap=args.min_score_gap,
        output_dir=args.output_dir,
        use_api=args.use_api,
        api_token=args.api_token or os.environ.get("HF_TOKEN"),
    )

    generator = PreferenceGenerator(config)
    dataset = generator.generate_dataset(
        prompts_path=args.prompts,
        max_samples=args.max_samples,
    )

    print(f"\nGenerated dataset: {dataset}")
    print(f"Columns: {dataset.column_names}")
    if len(dataset) > 0:
        print(f"\nSample entry:")
        print(f"  Prompt: {dataset[0]['prompt'][:100]}...")
        print(f"  Chosen score: {dataset[0]['chosen_score']}")
        print(f"  Rejected score: {dataset[0]['rejected_score']}")


if __name__ == "__main__":
    main()
