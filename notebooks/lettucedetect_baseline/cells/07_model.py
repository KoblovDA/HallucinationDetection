from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(method="transformer", model_path=MODEL_PATH)
print(f"Loaded {MODEL_PATH}")
