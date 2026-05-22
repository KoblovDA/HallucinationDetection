if DATA_DIR is None:
    try:
        from google.colab import files
        uploaded = files.upload()
        DATA_DIR = Path("/content")
        print(f"Uploaded: {list(uploaded.keys())}")
        print(f"DATA_DIR = {DATA_DIR}")
    except ImportError:
        print("Not in Colab. Set DATA_DIR manually or place combined_test.jsonl in one of the candidate dirs.")

target = DATA_DIR / "combined_test.jsonl"
print(f"combined_test.jsonl: {target}  exists={target.exists()}")
