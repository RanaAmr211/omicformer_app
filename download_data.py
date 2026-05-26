import os
import gdown

FILE_IDS = {
    "best_model_OmicFormer_149drugs_seed456.pth":            "1HRygM8y4q3JFGvwtyEGfqoMYhExIdezu",
    "preds_OmicFormer_149drugs_seed456.npy":                 "18tFxrDuvnCazrVrhyOWad-dHdermW1Sb",
    "true_OmicFormer_149drugs_seed456.npy":                  "1uB86KwaEZxc7b1y-Q5goEOYYiIOCv_1L",
    "attn_weights_OmicFormer_149drugs_seed456.npy":          "1JrwswhETo6YytTiSahq4LkaIjERK3Zsp",
    "pw_importance_overall_OmicFormer_149drugs_seed456.npy": "1GH4mSddZX0foH3IHL2KgT_Eq937LS6_3",
    "h_mut_OmicFormer_149drugs_seed456.npy":                 "1NkZ3nIU9nDoL9I00vqMpkuO2NDgem-Zl",
    "h_cnv_OmicFormer_149drugs_seed456.npy":                 "1MjIJOgpy2_Suo8Lkyf5NdY3YrlJBdUdy",
    "h_exp_OmicFormer_149drugs_seed456.npy":                 "19qh4Vnpr3zziPDaj4RifEmZgFdd5BTP1",
    "train_embeddings.npy":                                  "1BFf6kvWlzq2Qs5h5OtUj7I9loir0UPG2",
    "y_train.npy":                                           "1cghRFCyQmuJfg4bjZ2GeCFVllMSfk1Vm",
    "drug_names_train.npy":                                  "12uRxpPfVuI-Hn3sQiVuZecOuG_xGYfak",
    "drug_names_test.npy":                                   "1IHoyXUcayJyjOZDrhoBMWtsMT649pnvO",
    "cell_ids_test.npy":                                     "1tGvPmq2n4ppIvN6VKcNJ4o1ihO8-1nIB",
    "dims.pkl":                                              "1DdmlXyJs-lOwNeGjKlRo6mh8TDyCUIYB",
    "test_lookup.json":                                      "1AlqUGspLM4DxA13PEAdsTY8nzJwoSAYm",
    "pathway_names_ordered.json":                            "1Gycr4Ha8hqbahWSJGg_tFsL2Q6xQ6H2n",
    "drug_mechanisms.json":                                  "1t8dpOE455NohTSDFpsBrDzToKUnLSKW1",
    "ensembl_to_symbol.json":                                "1Tl3h3RTxwVvWSn7g0e7Ym_f8gtNCebwr",
    "unique_cell_lines.json":                                "1o8M5QIkO7Sn5uiTA4kuQQzJK69rHh2jc",
    "unique_drugs.json":                                     "1yeI3adgYNHcMnaB_yD2G40QiAEM0zUU7",
    "training_log_OmicFormer_149drugs_seed456.csv":          "1ibfw_53e1YJt42lR2y229Qe-F0qIuf8h",
    "per_drug_metrics_OmicFormer_149drugs_seed456.csv":      "1F9lMK7gQDU5xLT5R4f7bIFUG5QiXBE9C",
    "exp_profiles_149drugs.parquet":                         "1gpJjlFdmScmFBGDlRZfIod8sjWSFhEIg",
    "mut_profiles_149drugs.parquet":                         "1obqi9_iJNlYNdPz2FzLnNuq8h6kSyBd4",
    "cnv_profiles_149drugs.parquet":                         "1HR4agIxo0sccTKF1PSxrHBEv5D6I8e--",
}

def download_all_files(base_dir):
    missing = [f for f in FILE_IDS if not os.path.exists(os.path.join(base_dir, f))]
    if not missing:
        print("All files already present.")
        return

    print(f"Downloading {len(missing)} missing files...")
    for filename in missing:
        file_id = FILE_IDS[filename]
        output = os.path.join(base_dir, filename)
        print(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False, fuzzy=True)

    print("All downloads complete.")
