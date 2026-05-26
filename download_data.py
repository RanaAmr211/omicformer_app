import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "rana234/omicformer-data"

REQUIRED_FILES = [
    "best_model_OmicFormer_149drugs_seed456.pth",
    "preds_OmicFormer_149drugs_seed456.npy",
    "true_OmicFormer_149drugs_seed456.npy",
    "attn_weights_OmicFormer_149drugs_seed456.npy",
    "pw_importance_overall_OmicFormer_149drugs_seed456.npy",
    "h_mut_OmicFormer_149drugs_seed456.npy",
    "h_cnv_OmicFormer_149drugs_seed456.npy",
    "h_exp_OmicFormer_149drugs_seed456.npy",
    "train_embeddings_part0.npy",
    "train_embeddings_part1.npy",
    "train_embeddings_part2.npy",
    "train_embeddings_part3.npy",
    "y_train.npy",
    "drug_names_train.npy",
    "drug_names_test.npy",
    "cell_ids_test.npy",
    "dims.pkl",
    "test_lookup.json",
    "pathway_names_ordered.json",
    "drug_mechanisms.json",
    "ensembl_to_symbol.json",
    "unique_cell_lines.json",
    "unique_drugs.json",
    "training_log_OmicFormer_149drugs_seed456.csv",
    "per_drug_metrics_OmicFormer_149drugs_seed456.csv",
    "exp_profiles_149drugs.parquet",
    "mut_profiles_149drugs.parquet",
    "cnv_profiles_149drugs.parquet",
]

def download_all_files(base_dir):
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(base_dir, f))]
    if not missing:
        print("All files already present.")
        return

    print(f"Downloading {len(missing)} files from Hugging Face...")
    for filename in missing:
        dest = os.path.join(base_dir, filename)
        print(f"Downloading {filename}...")
        try:
            # Download to HF cache then copy to base_dir
            cached = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
            )
            shutil.copy2(cached, dest)
            print(f"Done: {filename} ({os.path.getsize(dest):,} bytes)")
        except Exception as e:
            print(f"Failed: {filename} — {e}")

    print("All downloads complete.")
