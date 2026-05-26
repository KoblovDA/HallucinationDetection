# Hallucination Detection in Tool Calling

This repository contains the codebase, dataset generation pipelines, and evaluation notebooks for detecting hallucinations at the **character-span level** in tool-augmented LLM dialogues. 

The project introduces a custom dataset built on top of the ToolACE corpus and evaluates five different detection strategies, ranging from off-the-shelf token classifiers and internal attention analysis (LookBackLens) to massive LLM-as-a-judge pipelines and domain-specific fine-tuning.

## Key Results

We evaluated all methods on a combined test set (N=599) containing 25% clean interactions and 75% hallucinated samples across three distinct hallucination types. The primary metric is **Span Micro F1** (RAGTruth-style character intersection).

| Method | Combined F1 | Type 1 (Contradiction) | Type 2 (Overgeneration) | Type 3 (Missing Tool) |
|:---|:---:|:---:|:---:|:---:|
| **Baseline 1:** LettuceDetect (off-the-shelf) | 0.660 | 0.137 | 0.726 | 0.597 |
| **Baseline 2:** LookBackLens (Qwen2.5-3B + LogReg) | 0.771 | 0.293 | 0.805 | 0.766 |
| **Method 1:** LLM + Rule-Based Hybrid (7B) | 0.191 | 0.310 | 0.161 | 0.113 |
| **Method 2:** LLM-as-a-judge (Qwen3-235B) | 0.859 | 0.315 | 0.844 | 0.792 |
| **Method 3: Fine-tuned ModernBERT** 🏆 | **0.979** | **0.762** | **0.985** | **0.981** |

> **Key Finding:** Supervised fine-tuning of a token classifier (`ModernBERT`) on our domain-specific dataset overwhelmingly outperforms all zero-shot and few-shot methods. Meanwhile, `LookBackLens` proves to be a highly effective and memory-efficient baseline when applied to a 3B backbone using our custom hook-based attention extraction.

## Taxonomy and Dataset

The dataset (`dakoblov/Hallucinations` on HuggingFace) is derived from ToolACE and structured in the RAGTruth format. It features three types of injected hallucinations:

1. **Type 1 (Contradiction):** The answer contradicts a specific value in the tool output JSON. Generated via a priority-based deterministic substring substitution pipeline (in-sample pools, type-aware numeric/date/URL shifts) with an LLM fallback.
2. **Type 2 (Overgeneration):** The answer adds facts, statistics, or historical context not present in the tool output. Generated using Qwen3-235B.
3. **Type 3 (Missing Tool):** The answer proposes an action requiring an API capability that is not present in the system's `tools_available` list. Generated using Qwen3-235B.

## Evaluated Methods

1. **LettuceDetect (Baseline):** Off-the-shelf inference using `KRLabsOrg/lettucedect-large-modernbert-en-v1`. Struggles with fine-grained Type 1 point substitutions due to its RAGTruth training distribution.
2. **LookBackLens (Baseline):** An attention-based "white-box" approach. We extract self-attention lookback ratios from `Qwen/Qwen2.5-3B-Instruct` and train a logistic regression classifier over sliding windows. We implemented a custom forward-hook extraction mechanism that drops peak attention memory from 18 GB to <1 GB, allowing processing on a single 16GB T4 GPU.
3. **LLM + Rule-Based Hybrid:** A zero-shot pipeline using `Qwen2.5-7B-Instruct` to propose hallucinated spans, strictly gated by a regular-expression parser that protects verified grounded values from the JSON context.
4. **LLM-as-a-judge:** An API-based approach using `Qwen3-235B` via OpenRouter. Utilizes a detailed taxonomy prompt, 4 few-shot examples, and a strict `TIGHT SPANS` instruction to extract character-accurate value-level contradictions.
5. **Fine-tuned ModernBERT:** The LettuceDetect backbone explicitly fine-tuned on our `combined_train` dataset. It successfully learns the structural boundaries of tool-calling JSON schemas and achieves state-of-the-art results on this task.

## Repository Structure

To keep the final report clean and academic, all execution code is factored out into standalone, reproducible Jupyter notebooks.

* `final_notebook.ipynb` — The main academic report discussing methodology, architecture choices, and aggregated results.
* `notebooks/` — Executable pipelines for each method:
  * `dataset_construction.ipynb` — Rule-based and LLM-based hallucination injection.
  * `1_lettuce_baseline.ipynb` — Off-the-shelf LettuceDetect evaluation.
  * `2_lookback_baseline.ipynb` — LookBackLens hook-based extraction and classifier training.
  * `3_finetune_modernbert.ipynb` — Supervised token classification fine-tuning.
  * `4_llm_rulebased.ipynb` — Hybrid generation + regex guardrails.
  * `5_llm_as_a_judge.ipynb` — OpenRouter 235B API evaluation.
* `src/` & `scripts/` — Underlying Python modules for dataset building and baselines.
* `results/` — Raw JSON metric outputs from all notebooks.
* `checkpoints/` — Local storage for trained weights (ignored in Git).

## How to Run (Docker Environment)

We recommend a Linux environment with at least one 16GB GPU (e.g., NVIDIA T4, V100, RTX 3090/4090) and Docker with the NVIDIA Container Toolkit installed.

### 1. Setup Environment Credentials
Before building, create a file named `credentials` in the root directory of the project to store your API keys and environment variables (used for the API-based judge):
```bash
echo "export OPENROUTER_API_KEY='your_api_key_here'" > credentials
```

### 2. Build the Docker Image
Run the provided build script to compile the Docker image containing all dependencies (CUDA, PyTorch, Transformers, and Lettucedetect):
```bash
chmod +x build launch_container
./build
```

### 3. Launch the Container
Run the launch script. This will start the container with GPU access, mount your local project directory to `/app`, and automatically start the Jupyter Notebook server:
```bash
./launch_container
```

### 4. Connect to the Jupyter Notebook
Once the container starts, open your web browser and connect to the Jupyter server running on port **8881**:
```
http://localhost:8881
```

### 5. Execute the Pipelines
Once inside the Jupyter interface:
1. Open the main report notebook: **`final_notebook.ipynb`**.
2. Follow the inline instructions and run the cells sequentially to aggregate the results or navigate to the individual notebooks in the `notebooks/` folder to run specific experiments.
```
