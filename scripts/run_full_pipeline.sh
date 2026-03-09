#!/bin/bash
# ================================================================
# LLM Alignment Engine — Full Pipeline Runner
# ================================================================
# Runs the complete alignment pipeline:
#   1. Data preparation (SFT + Preference generation)
#   2. SFT training
#   3. DPO alignment
#   4. Evaluation
#   5. Benchmarking
#   6. Analysis visualization
# ================================================================

set -e  # Exit on error

echo "============================================"
echo "  LLM Alignment Engine — Full Pipeline"
echo "============================================"
echo ""

# ------ Configuration ------
MODEL=${MODEL:-"Qwen/Qwen2.5-1.5B-Instruct"}
MAX_PREF_SAMPLES=${MAX_PREF_SAMPLES:-50}
EVAL_SAMPLES=${EVAL_SAMPLES:-100}

echo "Model: $MODEL"
echo "Preference samples: $MAX_PREF_SAMPLES"
echo "Eval samples: $EVAL_SAMPLES"
echo ""

# ------ Step 1: Data Preparation ------
echo "[1/6] Preparing SFT data..."
python -m data.data_processor --mode sft --dataset openai/gsm8k --output ./data/processed/sft

echo "[1/6] Generating preference data..."
python -m data.preference_generator \
    --prompts data/math_prompts.json \
    --generator-model $MODEL \
    --num-candidates 3 \
    --min-score-gap 1.5 \
    --max-samples $MAX_PREF_SAMPLES \
    --output-dir ./data/preference_data

echo "[1/6] Processing preference data for DPO..."
python -m data.data_processor --mode dpo --input ./data/preference_data --output ./data/processed/dpo

# ------ Step 2: SFT Training ------
echo ""
echo "[2/6] Starting SFT training..."
python -m training.sft_trainer --config configs/sft_config.yaml

# ------ Step 3: DPO Alignment ------
echo ""
echo "[3/6] Starting DPO alignment..."
python -m training.dpo_trainer --config configs/dpo_config.yaml --beta 0.1

# ------ Step 4: Evaluation ------
echo ""
echo "[4/6] Evaluating model..."
python -m evaluation.evaluate \
    --model ./outputs/dpo/final \
    --base-model $MODEL \
    --max-samples $EVAL_SAMPLES \
    --output ./outputs/evaluation

# ------ Step 5: Benchmarking ------
echo ""
echo "[5/6] Running performance benchmarks..."
python -m evaluation.benchmark \
    --model ./outputs/dpo/final \
    --output ./outputs/benchmark \
    --runs 5

# ------ Step 6: Analysis ------
echo ""
echo "[6/6] Generating comparison visualizations..."
python -m analysis.compare_methods \
    --results-dir ./outputs \
    --output ./outputs/analysis \
    --simulated

echo ""
echo "============================================"
echo "  ✅ Pipeline Complete!"
echo "============================================"
echo "  Results in: ./outputs/"
echo "  Model:      ./outputs/dpo/final/"
echo "  Eval:       ./outputs/evaluation/"
echo "  Graphs:     ./outputs/analysis/"
echo "============================================"
