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
    print("Files not found in standard locations. Upload via the cell below.")
else:
    print(f"Using DATA_DIR = {DATA_DIR}")

BASE_MODEL = "KRLabsOrg/lettucedect-large-modernbert-en-v1"
OUTPUT_DIR = "/content/checkpoints"
# T4 (16 GB) fits: ModernBERT-large + max_len=2048 + bs=2 + gradient_checkpointing.
# Effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS.
MAX_LENGTH = 2048
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
GRADIENT_CHECKPOINTING = True
LR = 1e-5
WEIGHT_DECAY = 0.01
EPOCHS = 3
WARMUP_RATIO = 0.05
EVAL_EVERY_STEPS = 200
SEED = 42
CLEAN_OVERSAMPLE = 4   # replicate clean samples K times to reach ~25% clean in train

import random, numpy as np, torch
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}, mem {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
