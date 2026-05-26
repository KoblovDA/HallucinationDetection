import subprocess, sys, os, importlib.util

NEEDED = ("transformers", "accelerate", "sklearn")
missing = [p for p in NEEDED if importlib.util.find_spec(p) is None]
if missing:
    pkgs = []
    for p in missing:
        pkgs.append("scikit-learn" if p == "sklearn" else p)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "transformers>=4.48", "accelerate", "scikit-learn"])
    if "google.colab" in sys.modules:
        print("Restarting kernel to load new packages…")
        os.kill(os.getpid(), 9)
else:
    print("All packages already installed.")
