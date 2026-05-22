FINAL_DIR = "/content/checkpoints/best"
trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"Saved to {FINAL_DIR}")

# Optional: zip for download
import shutil
shutil.make_archive("/content/best_model", "zip", FINAL_DIR)
print("Zipped to /content/best_model.zip — download via the file panel.")
