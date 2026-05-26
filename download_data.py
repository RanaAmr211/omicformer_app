import os
import gdown

# Google Drive folder ID
FOLDER_ID = "1ExeBFG53Q9zVdn35snJZrWHwaUS8Q0Xp"

REQUIRED_FILES = [
    "best_model_OmicFormer_149drugs_seed456.pth",
    "preds_OmicFormer_149drugs_seed456.npy",
    "true_OmicFormer_149drugs_seed456.npy",
    "attn_weights_OmicFormer_149drugs_seed456.npy",
    "pw_importance_overall_OmicFormer_149drugs_seed456.npy",
    "h_mut_OmicFormer_149drugs_seed456.npy",
    "h_cnv_OmicFormer_149drugs_seed456.npy",
    "h_exp_OmicFormer_149drugs_seed456.npy",
    "train_embeddings.npy",
    "y_train.npy",
    "drug_names_train.npy",
    "drug_names_test.npy",
    "cell_ids_test.npy",
    "dims.pkl",
    "test_lookup.json",
    "exp_profiles_149drugs.parquet",
    "mut_profiles_149drugs.parquet",
    "cnv_profiles_149drugs.parquet",
    "training_log_OmicFormer_149drugs_seed456.csv",
    "per_drug_metrics_OmicFormer_149drugs_seed456.csv",
]

def download_all_files(base_dir):
    all_present = all(
        os.path.exists(os.path.join(base_dir, f)) for f in REQUIRED_FILES
    )
    if all_present:
        print("All files already present.")
        return

    print("Downloading files from Google Drive...")
    gdown.download_folder(
        id=FOLDER_ID,
        output=base_dir,
        quiet=False,
        use_cookies=False,
        remaining_ok=True
    )
    print("Download complete.")
