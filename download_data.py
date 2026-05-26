import os
import requests
import re

FILE_IDS = {
    "best_model_OmicFormer_149drugs_seed456.pth":            "1HRygM8y4q3JFGvwtyEGfqoMYhExIdezu",
    "preds_OmicFormer_149drugs_seed456.npy":                 "18tFxrDuvnCazrVrhyOWad-dHdermW1Sb",
    "true_OmicFormer_149drugs_seed456.npy":                  "1uB86KwaEZxc7b1y-Q5goEOYYiIOCv_1L",
    "attn_weights_OmicFormer_149drugs_seed456.npy":          "1JrwswhETo6YytTiSahq4LkaIjERK3Zsp",
    "pw_importance_overall_OmicFormer_149drugs_seed456.npy": "1GH4mSddZX0foH3IHL2KgT_Eq937LS6_3",
    "h_mut_OmicFormer_149drugs_seed456.npy":                 "1NkZ3nIU9nDoL9I00vqMpkuO2NDgem-Zl",
    "h_cnv_OmicFormer_149drugs_seed456.npy":                 "1MjIJOgpy2_Suo8Lkyf5NdY3YrlJBdUdy",
    "h_exp_OmicFormer_149drugs_seed456.npy":                 "19qh4Vnpr3zziPDaj4RifEmZgFdd5BTP1",
    "train_embeddings_part0.npy":                            "18Lsl1apqZQhYVlZh0P4L5iD28mtBplLC",
    "train_embeddings_part1.npy":                            "1MXr4_sJu1pLlFaBhLknxqwcNBkX22IO3",
    "train_embeddings_part2.npy":                            "1uT0v5yCGoup54JnxSy9hB5BfBWaKPwhI",
    "train_embeddings_part3.npy":                            "1IzJxWSGVXbLmmo-nEWSlomrYPJ8s5jUE",
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

MIN_SIZES = {
    "best_model_OmicFormer_149drugs_seed456.pth": 100_000_000,
    "train_embeddings_part0.npy":                  50_000_000,
    "train_embeddings_part1.npy":                  50_000_000,
    "train_embeddings_part2.npy":                  50_000_000,
    "train_embeddings_part3.npy":                  50_000_000,
    "exp_profiles_149drugs.parquet":               10_000_000,
    "mut_profiles_149drugs.parquet":               10_000_000,
    "cnv_profiles_149drugs.parquet":               10_000_000,
}

def download_file(file_id, output_path):
    session = requests.Session()
    url = "https://drive.google.com/uc"
    params = {"id": file_id, "export": "download"}
    response = session.get(url, params=params, stream=True)

    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break
    if token:
        params["confirm"] = token
        response = session.get(url, params=params, stream=True)

    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type:
        match = re.search(r'confirm=([0-9A-Za-z_-]+)', response.text)
        if match:
            params["confirm"] = match.group(1)
            response = session.get(url, params=params, stream=True)

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

def file_is_valid(filepath, filename):
    if not os.path.exists(filepath):
        return False
    size = os.path.getsize(filepath)
    return size >= MIN_SIZES.get(filename, 100)

def download_all_files(base_dir):
    to_download = [f for f in FILE_IDS if not file_is_valid(os.path.join(base_dir, f), f)]
    if not to_download:
        print("All files present and valid.")
        return

    print(f"Downloading {len(to_download)} files...")
    for filename in to_download:
        output = os.path.join(base_dir, filename)
        print(f"Downloading {filename}...")
        try:
            download_file(FILE_IDS[filename], output)
            size = os.path.getsize(output)
            print(f"Done: {filename} ({size:,} bytes)")
        except Exception as e:
            print(f"Failed: {filename} — {e}")
    print("All downloads complete.")
