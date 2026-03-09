from setuptools import setup, find_packages

setup(
    name="llm-alignment-engine",
    version="1.0.0",
    description="End-to-End LLM Alignment Engine for Math Reasoning (DPO/PPO + QLoRA + DeepSpeed)",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/llm-alignment-engine",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "datasets>=2.16.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "trl>=0.7.6",
        "bitsandbytes>=0.41.0",
        "deepspeed>=0.12.0",
        "evaluate>=0.4.0",
        "pyyaml>=6.0",
        "wandb>=0.15.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
    ],
)
