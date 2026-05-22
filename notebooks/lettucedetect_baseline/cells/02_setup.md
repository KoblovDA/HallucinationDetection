## Setup

This notebook runs in **Google Colab** or **Kaggle**. Steps below:

1. **Enable GPU**: `Runtime → Change runtime type → T4 GPU` (Colab) or `Settings → Accelerator → GPU` (Kaggle).
2. **Upload the three test files** (`test.jsonl`, `type2_test.jsonl`, `type3_test.jsonl`):
   - **Colab**: click the folder icon in the left sidebar → upload button → drop the three files into `/content/`. (Or run the upload cell below.)
   - **Kaggle**: upload them as a Kaggle dataset, then point `DATA_DIR` at `/kaggle/input/<your-dataset>/`.
3. **Run all cells**.

The expected filenames are `test.jsonl` (Type 1), `type2_test.jsonl`, `type3_test.jsonl`.