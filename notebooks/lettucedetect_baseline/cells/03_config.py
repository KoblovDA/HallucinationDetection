from pathlib import Path

CANDIDATE_DIRS = [
    Path("/content"),
    Path("/kaggle/input/halluc-toolace"),
    Path("data"),
    Path("."),
]
REQUIRED = ["combined_test.jsonl"]
DATA_DIR = next((d for d in CANDIDATE_DIRS if all((d / f).exists() for f in REQUIRED)), None)

if DATA_DIR is None:
    print("combined_test.jsonl not found. Upload it via the cell below.")
else:
    print(f"Using DATA_DIR = {DATA_DIR}")

MODEL_PATH = "KRLabsOrg/lettucedect-large-modernbert-en-v1"
