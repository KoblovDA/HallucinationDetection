if DATA_DIR is None:
    try:
        from google.colab import files
        uploaded = files.upload()
        DATA_DIR = Path("/content")
    except ImportError:
        raise RuntimeError("Set DATA_DIR manually if not in Colab.")
for f in REQUIRED:
    print(f"  {f}: {(DATA_DIR / f).exists()}")
