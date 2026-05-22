import subprocess, sys, os, importlib.util

NEEDED = ("transformers", "accelerate", "torch")
missing = [p for p in NEEDED if importlib.util.find_spec(p) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "transformers>=4.48", "accelerate", "datasets"])
    if "google.colab" in sys.modules:
        print("Restarting kernel…")
        os.kill(os.getpid(), 9)
else:
    print("Already installed.")
