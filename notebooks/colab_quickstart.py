#!/usr/bin/env python3
"""
=================================================================
LLM Alignment Engine — Google Colab Quick Start
=================================================================

One-click notebook to run the full alignment pipeline on Colab.
Copy this file to a Colab notebook cell-by-cell, or upload as .py
and convert to notebook.

Runtime: GPU → T4 (free tier) or L4/A100 (Pro)
Estimated time: ~2-3 hours for full pipeline on T4
=================================================================
"""

# ============================================================
# Cell 1: Environment Setup
# ============================================================
# !pip install -q torch transformers datasets accelerate peft trl
# !pip install -q bitsandbytes deepspeed evaluate
# !pip install -q matplotlib seaborn pandas wandb
# !pip install -q sentencepiece protobuf

# Clone the project
# !git clone https://github.com/yourusername/llm-alignment-engine.git
# %cd llm-alignment-engine
# !pip install -e .

import os
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# ============================================================
# Cell 2: Configuration
# ============================================================
# Choose your model (adjust based on GPU memory)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B — fits T4 easily
# MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # 3B — needs more memory

# Training parameters
SFT_EPOCHS = 2
DPO_EPOCHS = 1
MAX_SEQ_LENGTH = 512  # reduce for T4

# Set WandB (optional)
os.environ["WANDB_PROJECT"] = "llm-alignment-engine"
# os.environ["WANDB_API_KEY"] = "your_key_here"
os.environ["WANDB_DISABLED"] = "true"  # disable by default

print(f"Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  SFT Epochs: {SFT_EPOCHS}")
print(f"  DPO Epochs: {DPO_EPOCHS}")

# ============================================================
# Cell 3: Stage 1 — Data Preparation
# ============================================================
print("\n" + "="*60)
print("Stage 1: Preparing Training Data")
print("="*60)

from data.data_processor import DataProcessor, ProcessorConfig

# Process SFT data (GSM8K)
processor = DataProcessor(ProcessorConfig(output_dir="./data/processed/sft"))
sft_dataset = processor.process_sft_data(
    dataset_name="openai/gsm8k",
    output_path="./data/processed/sft",
)

print(f"\nSFT Dataset:")
print(f"  Train: {len(sft_dataset['train'])} samples")
print(f"  Validation: {len(sft_dataset['validation'])} samples")

# ============================================================
# Cell 4: Stage 2 — Generate Preference Data
# ============================================================
print("\n" + "="*60)
print("Stage 2: Generating Preference Data (LLM-as-a-Judge)")
print("="*60)

from data.preference_generator import PreferenceGenerator, GeneratorConfig

gen_config = GeneratorConfig(
    generator_model=MODEL_NAME,
    judge_model=MODEL_NAME,
    num_candidates=3,        # reduce for speed
    min_score_gap=1.5,       # lower threshold for more data
    output_dir="./data/preference_data",
)

generator = PreferenceGenerator(gen_config)
preference_dataset = generator.generate_dataset(
    prompts_path="data/math_prompts.json",
    max_samples=20,          # small set for demo; increase for real training
)

print(f"\nPreference Dataset: {len(preference_dataset)} pairs")
if len(preference_dataset) > 0:
    print(f"Sample: {preference_dataset[0]['prompt'][:80]}...")

# Process for DPO training
dpo_processor = DataProcessor(ProcessorConfig(output_dir="./data/processed/dpo"))
dpo_dataset = dpo_processor.process_preference_data(
    input_path="./data/preference_data",
    output_path="./data/processed/dpo",
)

# ============================================================
# Cell 5: Stage 3 — SFT Training
# ============================================================
print("\n" + "="*60)
print("Stage 3: Supervised Fine-Tuning (SFT)")
print("="*60)

from training.sft_trainer import SFTFineTuner
import yaml

# Override config for Colab
sft_config = {
    "model": {"name_or_path": MODEL_NAME, "torch_dtype": "bfloat16"},
    "qlora": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "training": {
        "output_dir": "./outputs/sft",
        "num_train_epochs": SFT_EPOCHS,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_seq_length": MAX_SEQ_LENGTH,
        "logging_steps": 20,
        "save_steps": 500,
        "save_total_limit": 2,
        "bf16": True,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_32bit",
        "report_to": "none",
    },
    "dataset": {"name": "gsm8k", "split": "train"},
}

# Save config and run SFT
os.makedirs("configs", exist_ok=True)
with open("configs/sft_config_colab.yaml", "w") as f:
    yaml.dump(sft_config, f)

sft_finetuner = SFTFineTuner("configs/sft_config_colab.yaml")
sft_metrics = sft_finetuner.train()
sft_finetuner.save("./outputs/sft/final")

print(f"\nSFT Training Complete!")
print(f"  Final loss: {sft_metrics.get('train_loss', 'N/A')}")

# Free memory
del sft_finetuner
torch.cuda.empty_cache()

# ============================================================
# Cell 6: Stage 4 — DPO Alignment
# ============================================================
print("\n" + "="*60)
print("Stage 4: DPO Alignment Training")
print("="*60)

from training.dpo_trainer import DPOFineTuner

dpo_config = {
    "model": {
        "name_or_path": MODEL_NAME,
        "sft_checkpoint": "./outputs/sft/final",
        "torch_dtype": "bfloat16",
    },
    "qlora": sft_config["qlora"],
    "dpo": {
        "beta": 0.1,
        "loss_type": "sigmoid",
        "label_smoothing": 0.0,
        "max_prompt_length": 256,
        "max_length": 512,
    },
    "training": {
        "output_dir": "./outputs/dpo",
        "num_train_epochs": DPO_EPOCHS,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_steps": 200,
        "save_total_limit": 2,
        "bf16": True,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_32bit",
        "report_to": "none",
    },
    "dataset": {"path": "./data/processed/dpo"},
    "ablation": {"beta_values": [0.05, 0.1, 0.2, 0.5]},
}

with open("configs/dpo_config_colab.yaml", "w") as f:
    yaml.dump(dpo_config, f)

dpo_finetuner = DPOFineTuner("configs/dpo_config_colab.yaml")
dpo_metrics = dpo_finetuner.train(beta=0.1)
dpo_finetuner.save("./outputs/dpo/final")

print(f"\nDPO Training Complete!")

del dpo_finetuner
torch.cuda.empty_cache()

# ============================================================
# Cell 7: Stage 5 — Evaluation
# ============================================================
print("\n" + "="*60)
print("Stage 5: Model Evaluation (GSM8K)")
print("="*60)

from evaluation.evaluate import ModelEvaluator, EvalConfig

# Evaluate DPO model
eval_config = EvalConfig(
    model_path="./outputs/dpo/final",
    base_model=MODEL_NAME,
    max_samples=100,
    output_dir="./outputs/evaluation/dpo",
)

evaluator = ModelEvaluator(eval_config)
dpo_results = evaluator.evaluate()

print(f"\nDPO Model Results:")
print(f"  Accuracy: {dpo_results['accuracy']*100:.2f}%")

del evaluator
torch.cuda.empty_cache()

# ============================================================
# Cell 8: Stage 6 — Benchmarking
# ============================================================
print("\n" + "="*60)
print("Stage 6: Performance Benchmarking")
print("="*60)

from evaluation.benchmark import PerformanceBenchmark, BenchmarkConfig

bench_config = BenchmarkConfig(
    model_path="./outputs/dpo/final",
    base_model=MODEL_NAME,
    num_inference_runs=5,
    output_dir="./outputs/benchmark",
)

benchmark = PerformanceBenchmark(bench_config)
bench_results = benchmark.run_all()

del benchmark
torch.cuda.empty_cache()

# ============================================================
# Cell 9: Stage 7 — Generate Comparison Visualizations
# ============================================================
print("\n" + "="*60)
print("Stage 7: Generating Analysis Visualizations")
print("="*60)

from analysis.compare_methods import MethodComparator, ComparatorConfig

comparator = MethodComparator(ComparatorConfig(
    output_dir="./outputs/analysis",
    use_simulated=True,  # set to False after running full pipeline
))
summary = comparator.generate_all()

# Display images
from IPython.display import display, Image as IPImage
for img_file in ["accuracy_progression.png", "beta_ablation_study.png", "resource_efficiency.png"]:
    img_path = f"./outputs/analysis/{img_file}"
    if os.path.exists(img_path):
        print(f"\n{img_file}:")
        display(IPImage(filename=img_path, width=800))

# ============================================================
# Cell 10: Final Summary
# ============================================================
print("\n" + "="*60)
print("🎉 Pipeline Complete!")
print("="*60)
print(f"""
Results Summary:
  ✅ SFT Training: {sft_metrics.get('train_loss', 'N/A')} final loss
  ✅ DPO Alignment: beta=0.1
  ✅ GSM8K Accuracy: {dpo_results.get('accuracy', 0)*100:.1f}%
  ✅ Visualizations: saved to ./outputs/analysis/

Key Artifacts:
  📁 SFT Model:     ./outputs/sft/final/
  📁 DPO Model:     ./outputs/dpo/final/
  📁 Evaluation:    ./outputs/evaluation/
  📁 Benchmarks:    ./outputs/benchmark/
  📁 Visualizations: ./outputs/analysis/

Next Steps:
  1. Run beta ablation: dpo_finetuner.run_beta_ablation()
  2. Try PPO: python training/ppo_trainer.py
  3. Push to Hub: model.push_to_hub("your-username/math-aligned-qwen")
""")
