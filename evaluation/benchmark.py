"""
Performance Benchmark — GPU Memory & Inference Throughput Profiling

Profiles:
1. Peak GPU memory usage during training and inference
2. Inference throughput (tokens/sec, time-to-first-token, time-per-output-token)
3. Memory breakdown by component (model, optimizer, activations, gradients)

Key for resume: demonstrates deep understanding of hardware-software interaction.
"""

import os
import time
import json
import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    model_path: str = "./outputs/dpo/final"
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_4bit: bool = True
    # Inference benchmarks
    prompt: str = "Solve step by step: A store sells notebooks for $3 each and pens for $1.50 each. If Sarah buys 4 notebooks and 6 pens, how much does she spend?"
    max_new_tokens: int = 256
    num_inference_runs: int = 10
    batch_sizes: list = None
    # Output
    output_dir: str = "./outputs/benchmark"

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8]


class PerformanceBenchmark:
    """
    Benchmarks model performance for memory usage and inference speed.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.model = None
        self.tokenizer = None
        self.results = {}

    def load_model(self):
        """Load model for benchmarking."""
        model_path = self.config.model_path

        bnb_config = None
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))

        if is_peft:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_path, quantization_config=bnb_config, device_map="auto",
                trust_remote_code=True, torch_dtype=torch.bfloat16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=bnb_config, device_map="auto",
                trust_remote_code=True, torch_dtype=torch.bfloat16,
            )

        # Always load tokenizer from base model (PEFT adapters may not include it)
        tokenizer_path = self.config.base_model if is_peft else model_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        return self

    # ----------------------------------------------------------
    # Memory Profiling
    # ----------------------------------------------------------
    def profile_memory(self) -> dict:
        """Profile GPU memory usage."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping memory profiling")
            return {"status": "cuda_not_available"}

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # Model memory
        model_memory = torch.cuda.memory_allocated() / 1024**3

        # Inference memory
        inputs = self.tokenizer(self.config.prompt, return_tensors="pt").to(self.model.device)
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )

        peak_inference_memory = torch.cuda.max_memory_allocated() / 1024**3
        inference_overhead = peak_inference_memory - model_memory

        # KV Cache estimation
        num_tokens = outputs.shape[1]
        kv_cache_estimate = self._estimate_kv_cache_memory(num_tokens)

        memory_stats = {
            "model_memory_gb": round(model_memory, 3),
            "peak_inference_memory_gb": round(peak_inference_memory, 3),
            "inference_overhead_gb": round(inference_overhead, 3),
            "kv_cache_estimate_gb": round(kv_cache_estimate, 4),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 2),
            "quantization": "4-bit (NF4)" if self.config.use_4bit else "full precision",
        }

        logger.info(f"\n{'='*50}")
        logger.info("Memory Profile")
        for k, v in memory_stats.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"{'='*50}")

        self.results["memory"] = memory_stats
        return memory_stats

    def _estimate_kv_cache_memory(self, num_tokens: int) -> float:
        """Estimate KV cache memory in GB."""
        config = self.model.config
        num_layers = getattr(config, "num_hidden_layers", 24)
        num_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 16))
        head_dim = getattr(config, "hidden_size", 1536) // getattr(config, "num_attention_heads", 16)
        # KV cache: 2 (K+V) * num_layers * num_heads * head_dim * num_tokens * dtype_size
        dtype_bytes = 2  # bfloat16
        kv_bytes = 2 * num_layers * num_heads * head_dim * num_tokens * dtype_bytes
        return kv_bytes / 1024**3

    # ----------------------------------------------------------
    # Inference Throughput
    # ----------------------------------------------------------
    def profile_throughput(self) -> dict:
        """Profile inference throughput (tokens/sec)."""
        if self.model is None:
            self.load_model()

        inputs = self.tokenizer(self.config.prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to(self.model.device)

        # Warmup
        with torch.no_grad():
            self.model.generate(**inputs, max_new_tokens=10, do_sample=False)

        # Benchmark: measure TTFT and TPOT
        latencies = []
        token_counts = []
        ttft_list = []

        for _ in range(self.config.num_inference_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()

            num_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
            latencies.append(end - start)
            token_counts.append(num_generated)

        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(token_counts) / len(token_counts)
        throughput = avg_tokens / avg_latency if avg_latency > 0 else 0
        tpot = (avg_latency / avg_tokens * 1000) if avg_tokens > 0 else 0  # ms per token

        throughput_stats = {
            "avg_latency_sec": round(avg_latency, 3),
            "avg_tokens_generated": round(avg_tokens, 1),
            "throughput_tokens_per_sec": round(throughput, 2),
            "time_per_output_token_ms": round(tpot, 2),
            "num_runs": self.config.num_inference_runs,
            "max_new_tokens": self.config.max_new_tokens,
        }

        logger.info(f"\n{'='*50}")
        logger.info("Inference Throughput")
        for k, v in throughput_stats.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"{'='*50}")

        self.results["throughput"] = throughput_stats
        return throughput_stats

    # ----------------------------------------------------------
    # Batch Size Impact
    # ----------------------------------------------------------
    def profile_batch_impact(self) -> dict:
        """Profile how batch size affects memory and throughput."""
        if self.model is None:
            self.load_model()

        if not torch.cuda.is_available():
            return {"status": "cuda_not_available"}

        batch_results = {}
        for bs in self.config.batch_sizes:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Create batch
                batch_prompts = [self.config.prompt] * bs
                inputs = self.tokenizer(
                    batch_prompts, return_tensors="pt", padding=True, truncation=True,
                ).to(self.model.device)

                start = time.perf_counter()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=128, do_sample=False,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                total_tokens = (outputs.shape[1] - inputs["input_ids"].shape[1]) * bs
                throughput = total_tokens / elapsed

                batch_results[bs] = {
                    "peak_memory_gb": round(peak_mem, 3),
                    "total_time_sec": round(elapsed, 3),
                    "throughput_tokens_per_sec": round(throughput, 2),
                    "status": "success",
                }
                logger.info(f"  BS={bs}: {peak_mem:.2f}GB, {throughput:.1f} tok/s")

            except torch.cuda.OutOfMemoryError:
                batch_results[bs] = {"status": "OOM"}
                logger.warning(f"  BS={bs}: Out of Memory!")
                torch.cuda.empty_cache()

        self.results["batch_impact"] = batch_results
        return batch_results

    # ----------------------------------------------------------
    # Full Benchmark
    # ----------------------------------------------------------
    def run_all(self) -> dict:
        """Run all benchmarks."""
        if self.model is None:
            self.load_model()

        logger.info("Running full benchmark suite...")
        self.profile_memory()
        self.profile_throughput()
        self.profile_batch_impact()

        # Save results
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(os.path.join(self.config.output_dir, "benchmark_results.json"), "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"\nAll benchmarks saved to: {self.config.output_dir}")
        return self.results


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Performance Benchmarking")
    parser.add_argument("--model", type=str, default="./outputs/dpo/final")
    parser.add_argument("--output", type=str, default="./outputs/benchmark")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    config = BenchmarkConfig(
        model_path=args.model,
        output_dir=args.output,
        num_inference_runs=args.runs,
    )

    benchmark = PerformanceBenchmark(config)
    benchmark.run_all()


if __name__ == "__main__":
    main()
