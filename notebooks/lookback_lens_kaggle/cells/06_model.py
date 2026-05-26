import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Pick up HF token from Colab/Kaggle secrets if present
try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
    if hf_token: os.environ["HF_TOKEN"] = hf_token
except Exception:
    pass
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    if hf_token: os.environ["HF_TOKEN"] = hf_token
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    attn_implementation="eager",   # eager attention so output_attentions=True works
    device_map=device,
)
model.eval()
print(f"Loaded {BASE_MODEL} on {device}; params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")
