import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, num_labels=2,
                                                        ignore_mismatched_sizes=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Loaded {BASE_MODEL} on {device}, params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")
