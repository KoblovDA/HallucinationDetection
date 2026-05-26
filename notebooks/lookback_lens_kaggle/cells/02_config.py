from pathlib import Path

CANDIDATE_DIRS = [
    Path("/content"),
    Path("/kaggle/input/halluc-toolace"),
    Path("data"),
    Path("."),
]
REQUIRED = ["combined_train.jsonl", "combined_val.jsonl", "combined_test.jsonl"]
DATA_DIR = next((d for d in CANDIDATE_DIRS if all((d / f).exists() for f in REQUIRED)), None)

if DATA_DIR is None:
    print("Data not found. Upload via the next cell or set DATA_DIR manually.")
else:
    print(f"Using DATA_DIR = {DATA_DIR}")

# Backbone LM that exposes per-layer attention. Qwen2.5-3B is fully open (no HF auth).
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_LENGTH = 4096
WINDOW = 8                  # chunk size from the paper (Table 3)
STRIDE = 8                  # non-overlapping chunks (paper's sliding-window setup)
THRESHOLD = 0.5
SEED = 42

import random, numpy as np, torch
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"CUDA: {torch.cuda.is_available()}",
      f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")
