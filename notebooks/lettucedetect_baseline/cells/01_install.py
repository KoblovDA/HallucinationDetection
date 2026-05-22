import subprocess, sys, os, importlib.util

NEEDED = ("lettucedetect",)
missing = [p for p in NEEDED if importlib.util.find_spec(p) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    # Colab pre-loads older numpy/transformers; restart the kernel so new versions are picked up.
    if "google.colab" in sys.modules:
        print("Restarting kernel to pick up new packages — re-run the notebook from the top after restart.")
        os.kill(os.getpid(), 9)
else:
    print("lettucedetect already installed.")
