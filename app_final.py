import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import torch
import joblib
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error
import sys
import warnings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Auto-download data if needed ──────────────────────
from download_data import download_all_files
download_all_files(BASE_DIR)
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="OmicFormer — Precision Oncology",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] {
    min-width: 320px !important;
    max-width: 320px !important;
    transform: none !important;
    visibility: visible !important;
    margin-left: 0 !important;
    overflow: visible !important;       
}
header[data-testid="stHeader"] { display: none !important; }
.block-container { padding-top: 1rem !important; overflow: visible !important; }
.main { overflow: visible !important; }
section.main > div { overflow: visible !important; }
.main .block-container { margin-left: 0 !important; overflow: visible !important; }
</style>
""", unsafe_allow_html=True)

GROQ_API_KEY = st.secrets["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"]

# ── Constants ──────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
GROQ_API_KEY = st.secrets["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"]
RUN_NAME       = "OmicFormer_149drugs_seed456"
COLORS = {
    "primary":    "#0A2342",
    "accent":     "#00B4D8",
    "accent2":    "#90E0EF",
    "sensitive":  "#06D6A0",
    "moderate":   "#FFB703",
    "resistant":  "#EF476F",
    "bg":         "#F8FAFC",
    "card":       "#FFFFFF",
    "text":       "#1A1A2E",
    "muted":      "#6B7280",
}

# ── CSS ────────────────────────────────────────────────
P = "#0A2342"; A = "#00B4D8"; A2 = "#90E0EF"
SEN = "#06D6A0"; MOD = "#FFB703"; RES = "#EF476F"
BG = "#F8FAFC"; TX = "#1A1A2E"; MU = "#6B7280"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background: {BG};
    color: {TX};
    font-size: 14px;
}}
.stApp {{ background: {BG}; }}

/* ── Hide branding ── */
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 100% !important; padding-left: 2rem !important; padding-right: 2rem !important; overflow: visible !important; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {P} !important;
    min-width: 320px !important;
    max-width: 320px !important;
}}
section[data-testid="stSidebar"] * {{
    color: white !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.85rem !important;
}}

/* ── Sidebar logo ── */
.logo-wrap {{
    padding: 1.2rem 1rem 1rem;
    border-bottom: 1px solid rgba(0,180,216,0.25);
    margin-bottom: 1rem;
    text-align: center;
}}
.logo-name {{
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: white !important;
    white-space: nowrap;
    margin-top: 0.5rem;
}}
.logo-name span {{ color: {A} !important; }}
.logo-tag {{
    font-size: 0.65rem;
    color: {A2} !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}}

/* ── Sidebar stat pills ── */
.spill {{
    background: rgba(0,180,216,0.12);
    border: 1px solid rgba(0,180,216,0.25);
    border-radius: 6px;
    padding: 0.35rem 0.6rem;
    margin: 0.2rem 0;
    font-size: 0.75rem;
    color: white !important;
    display: flex;
    justify-content: space-between;
}}
.spill b {{ color: {A} !important; font-weight: 600; }}

/* ── Page header ── */
.pg-title {{
    font-family: 'Inter', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: {P};
    margin-bottom: 0.4rem;
    line-height: 1.6;
    display: block;
}}
.pg-sub {{
    font-size: 0.82rem;
    color: {MU};
    margin-bottom: 1rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

/* ── Hero ── */
.hero {{
    background: linear-gradient(135deg, {P} 0%, #1a3a6b 100%);
    border-radius: 14px;
    padding: 2rem 2rem;
    color: white;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}}
.hero::after {{
    content: '';
    position: absolute;
    top: -40%;
    right: -5%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,180,216,0.12) 0%, transparent 70%);
    border-radius: 50%;
}}
.hero-title {{
    font-family: 'Inter', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1.4;
    margin-bottom: 0.6rem;
}}
.hero-title span {{ color: {A}; }}
.hero-sub {{
    font-size: 0.88rem;
    color: rgba(255,255,255,0.78);
    line-height: 1.6;
    max-width: 580px;
}}

/* ── Stat cards (home row) ── */
.sc {{
    background: white;
    border-radius: 10px;
    padding: 1rem 0.75rem;
    border: 1px solid rgba(0,0,0,0.06);
    text-align: center;
}}
.sc-val {{
    font-family: 'Inter', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: {P};
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}}
.sc-lbl {{
    font-size: 0.68rem;
    color: {MU};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.15rem;
}}
.sc-del {{
    font-size: 0.75rem;
    color: {SEN};
    font-weight: 600;
    margin-top: 0.1rem;
}}

/* ── Cards ── */
.card {{
    background: white;
    border-radius: 12px;
    padding: 1.2rem;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 1px 8px rgba(0,0,0,0.04);
    margin-bottom: 0.8rem;
}}
.card-hdr {{
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {MU};
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}
.card-hdr::before {{
    content: '';
    display: inline-block;
    width: 3px; height: 12px;
    background: {A};
    border-radius: 2px;
    flex-shrink: 0;
}}

/* ── Prediction value ── */
.pred-val {{
    font-family: 'Inter', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: {P};
    line-height: 1;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}}
.pred-unit {{
    font-size: 0.85rem;
    color: {MU};
    font-weight: 400;
    margin-left: 0.25rem;
    vertical-align: middle;
}}

/* ── Sensitivity badges ── */
.badge {{
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
}}
.badge-hs {{ background: rgba(6,214,160,0.12); color: #047857; border: 1px solid rgba(6,214,160,0.35); }}
.badge-ms {{ background: rgba(255,183,3,0.12);  color: #92400E; border: 1px solid rgba(255,183,3,0.35); }}
.badge-mr {{ background: rgba(255,183,3,0.12);  color: #92400E; border: 1px solid rgba(255,183,3,0.35); }}
.badge-hr {{ background: rgba(239,71,111,0.12); color: #9B1C1C; border: 1px solid rgba(239,71,111,0.35); }}

/* ── Mechanism box ── */
.mech {{
    background: rgba(10,35,66,0.04);
    border: 1px solid rgba(10,35,66,0.1);
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    font-size: 0.78rem;
    color: {P};
    line-height: 1.5;
    margin-top: 0.75rem;
}}

/* ── Narrative ── */
.narrative {{
    background: linear-gradient(135deg, rgba(10,35,66,0.03), rgba(0,180,216,0.04));
    border-left: 3px solid {A};
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    font-size: 0.86rem;
    line-height: 1.75;
    color: {TX};
    font-style: italic;
}}

/* ── Similar cases ── */
.case-row {{
    display: flex;
    align-items: center;
    padding: 0.5rem 0.65rem;
    border-radius: 7px;
    margin: 0.25rem 0;
    background: {BG};
    border: 1px solid rgba(0,0,0,0.05);
    font-size: 0.8rem;
    gap: 0.75rem;
}}
.case-id {{ font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 600; color: {P}; flex: 2; }}
.case-sim {{ color: {MU}; flex: 1; font-size: 0.75rem; }}
.case-ic50 {{ font-weight: 700; font-family: 'JetBrains Mono', monospace; flex: 1; font-size: 0.78rem; }}

/* ── Divider ── */
.div {{
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,180,216,0.2), transparent);
    margin: 1rem 0;
}}

/* ── Warn box ── */
.warn {{
    background: rgba(255,183,3,0.08);
    border: 1px solid rgba(255,183,3,0.3);
    border-radius: 7px;
    padding: 0.6rem 0.9rem;
    font-size: 0.8rem;
    color: #78350F;
}}

/* ── Buttons ── */
.stButton > button {{
    background: {P} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    width: 100% !important;
    transition: all 0.15s !important;
}}
.stButton > button:hover {{
    background: #0d2f5e !important;
    box-shadow: 0 3px 12px rgba(10,35,66,0.25) !important;
}}


#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 100% !important; padding-left: 2rem !important; padding-right: 2rem !important; overflow: visible !important; }}
</style>
""", unsafe_allow_html=True)


# ── Load all data ──────────────────────────────────────
@st.cache_resource
def load_all_data():
    data = {}

    # JSON files
    with open(f"{BASE_DIR}/test_lookup.json") as f:
        data["lookup"] = json.load(f)
    with open(f"{BASE_DIR}/unique_cell_lines.json") as f:
        data["cell_lines"] = json.load(f)
    with open(f"{BASE_DIR}/unique_drugs.json") as f:
        data["drugs"] = json.load(f)
    with open(f"{BASE_DIR}/pathway_names_ordered.json") as f:
        data["pathway_names"] = json.load(f)
    with open(f"{BASE_DIR}/drug_mechanisms.json") as f:
        data["mechanisms"] = json.load(f)
    with open(f"{BASE_DIR}/ensembl_to_symbol.json") as f:
        data["gene_symbols"] = json.load(f)

    # Numpy arrays
    data["preds"]        = np.load(f"{BASE_DIR}/preds_{RUN_NAME}.npy")
    data["true"]         = np.load(f"{BASE_DIR}/true_{RUN_NAME}.npy")
    data["attn"]         = np.load(f"{BASE_DIR}/attn_weights_{RUN_NAME}.npy")
    data["pw_importance"]= np.load(f"{BASE_DIR}/pw_importance_overall_{RUN_NAME}.npy")
    data["h_mut"]        = np.load(f"{BASE_DIR}/h_mut_{RUN_NAME}.npy")
    data["h_cnv"]        = np.load(f"{BASE_DIR}/h_cnv_{RUN_NAME}.npy")
    data["h_exp"]        = np.load(f"{BASE_DIR}/h_exp_{RUN_NAME}.npy")
    # Load train embeddings from chunks
    parts = [np.load(f"{BASE_DIR}/train_embeddings_part{i}.npy", allow_pickle=True) for i in range(4)]
    data["train_emb"] = np.concatenate(parts, axis=0)
    data["y_train"]      = np.load(f"{BASE_DIR}/y_train.npy", allow_pickle=True)
    data["drug_names_train"] = np.load(f"{BASE_DIR}/drug_names_train.npy", allow_pickle=True)
    data["drug_names_test"]  = np.load(f"{BASE_DIR}/drug_names_test.npy",  allow_pickle=True)
    data["cell_ids_test"]    = np.load(f"{BASE_DIR}/cell_ids_test.npy",    allow_pickle=True)
    data["dims"]         = joblib.load(f"{BASE_DIR}/dims.pkl")

    # Build lookup index: (cell_line, drug) -> sample index
    lookup_index = {}
    for rec in data["lookup"]:
        key = (rec["cell_line"], rec["drug"])
        lookup_index[key] = rec["idx"]
    data["lookup_index"] = lookup_index

    # Build cell line → available drugs mapping
    cell_drug_map = {}
    for rec in data["lookup"]:
        cl = rec["cell_line"]
        if cl not in cell_drug_map:
            cell_drug_map[cl] = []
        cell_drug_map[cl].append(rec["drug"])
    data["cell_drug_map"] = cell_drug_map

    # Training log
    try:
        data["train_log"] = pd.read_csv(f"{BASE_DIR}/training_log_{RUN_NAME}.csv")
    except:
        data["train_log"] = None

    # Per drug metrics
    try:
        data["per_drug"] = pd.read_csv(f"{BASE_DIR}/per_drug_metrics_{RUN_NAME}.csv")
    except:
        data["per_drug"] = None

    # Gene names from omic parquets
    try:
        exp_df = pd.read_parquet(f"{BASE_DIR}/exp_profiles_149drugs.parquet").set_index("CellLine_ID")
        mut_df = pd.read_parquet(f"{BASE_DIR}/mut_profiles_149drugs.parquet").set_index("CellLine_ID")
        cnv_df = pd.read_parquet(f"{BASE_DIR}/cnv_profiles_149drugs.parquet").set_index("CellLine_ID")
        data["exp_genes"] = exp_df.columns.tolist()
        data["mut_genes"] = mut_df.columns.tolist()
        data["cnv_genes"] = cnv_df.columns.tolist()
        data["exp_df"]    = exp_df
        data["mut_df"]    = mut_df
        data["cnv_df"]    = cnv_df
    except Exception as e:
        data["exp_genes"] = []
        data["mut_genes"] = []
        data["cnv_genes"] = []
        data["exp_df"] = None
        data["mut_df"] = None
        data["cnv_df"] = None

    return data

@st.cache_resource
def load_model(dims):
    sys.path.insert(0, BASE_DIR)
    from omicformer import OmicFormer
    device = torch.device("cpu")
    model = OmicFormer(
        mut_dim     = dims["mut_dim"],
        cnv_dim     = dims["cnv_dim"],
        exp_dim     = dims["exp_dim"],
        drug_dim    = dims["drug_dim"],
        context_dim = dims["context_dim"],
        n_pathways  = dims["n_pathways"],
        pw_stats    = 3,
        token_dim   = 64,
        bottleneck  = 512,
        n_pw_heads  = 8,
        n_co_heads  = 8,
        n_co_layers = 2,
        dropout     = 0.15)
    model.load_state_dict(torch.load(
        f"{BASE_DIR}/best_model_{RUN_NAME}.pth",
        map_location=device))
    model.eval()
    return model, device

# ── Helper functions ───────────────────────────────────
def sensitivity_label(pred_ic50, drug, data):
    # Use training set IC50 values for this drug as threshold
    # This matches what similar cases use, ensuring consistency
    train_mask = data["drug_names_train"] == drug
    if train_mask.sum() == 0:
        return "UNKNOWN", "badge badge-ms"
    train_ic50s = data["y_train"][train_mask].flatten()
    median = np.median(train_ic50s)
    p25    = np.percentile(train_ic50s, 25)
    p75    = np.percentile(train_ic50s, 75)
    if pred_ic50 <= p25:
        return "HIGH SENSITIVITY", "badge badge-hs"
    elif pred_ic50 <= median:
        return "MODERATE SENSITIVITY", "badge badge-ms"
    elif pred_ic50 <= p75:
        return "MODERATE RESISTANCE", "badge badge-mr"
    else:
        return "HIGH RESISTANCE", "badge badge-hr"

def sensitivity_color(label):
    if "HIGH SENSITIVITY" in label:
        return COLORS["sensitive"]
    elif "MODERATE" in label:
        return COLORS["moderate"]
    else:
        return COLORS["resistant"]

def get_similar_cases(idx, drug, data, k=5):
    query = np.concatenate([
        data["h_mut"][idx],
        data["h_cnv"][idx],
        data["h_exp"][idx]
    ]).reshape(1, -1)
    mask = data["drug_names_train"] == drug
    if mask.sum() < k:
        return []
    drug_emb  = data["train_emb"][mask]
    drug_y    = data["y_train"][mask].flatten()
    drug_cells= data["drug_names_train"][mask]
    sims      = cosine_similarity(query, drug_emb)[0]
    top_k     = sims.argsort()[-k:][::-1]
    results   = []
    # Use median of training IC50 for this drug as threshold
    drug_median = np.median(drug_y)
    for i in top_k:
        results.append({
            "cell_id":    drug_cells[i],
            "similarity": round(float(sims[i]), 3),
            "ic50":       round(float(drug_y[i]), 3),
            "status":     "Sensitive" if drug_y[i] <= drug_median else "Resistant"
        })
    return results

def generate_narrative(drug, mechanism, pred_ic50, sens_label,
                       omic_w, top_pathways, top_mut, top_exp, cases):
    try:
        case_lines = "\n".join([f"- {c['cell_id']}: observed IC50={c['ic50']:.2f} ({c['status']})" for c in cases[:3]]) if cases else "- No similar cases available"
        prompt = f"""You are a clinical oncology assistant explaining a drug response prediction made by OmicFormer, a validated deep learning model (Pearson r=0.9467±0.0012, beats state-of-the-art PASO model).

Drug: {drug}
Mechanism: {mechanism}
Predicted LN_IC50: {pred_ic50:.3f} — {sens_label}

Omic importance driving this prediction:
- Mutation profile: {omic_w[0]*100:.0f}%
- Copy number variation: {omic_w[1]*100:.0f}%
- Gene expression: {omic_w[2]*100:.0f}%

Top activated biological pathways:
{chr(10).join([f"- {p}" for p in top_pathways[:5]])}

Top mutation genes:
{chr(10).join([f"- {g}" for g in top_mut[:5]])}

Top expression genes:
{chr(10).join([f"- {g}" for g in top_exp[:5]])}

Similar historical cases (same drug, most similar genomic profiles):
{case_lines}

Write a concise clinical interpretation for an oncologist in exactly 3 sentences. Do not number the sentences or add any introduction like "Here are three sentences". Just write the 3 sentences directly, one after another. Cover: (1) the prediction and primary omic driver, (2) the most relevant pathways and genes and their connection to the drug mechanism, (3) support from historical similar cases. Be precise, clinically relevant, and grounded in the evidence above only."""

        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 200, "temperature": 0.3},
            timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return f"API error {r.status_code}: {r.text[:100]}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)[:150]}"

def get_top_genes(idx, omic, data, model, device, n=10):
    if data["exp_df"] is None:
        return [], []
    try:
        cell_id = data["cell_ids_test"][idx]
        if omic == "exp":
            genes  = data["exp_genes"]
            raw_df = data["exp_df"]
        elif omic == "mut":
            genes  = data["mut_genes"]
            raw_df = data["mut_df"]
        else:
            genes  = data["cnv_genes"]
            raw_df = data["cnv_df"]

        if cell_id not in raw_df.index:
            return [], []

        vals = torch.tensor(
            raw_df.loc[cell_id].values.astype(np.float32)).unsqueeze(0)
        vals.requires_grad_(True)

        dummy_mut = torch.zeros(1, data["dims"]["mut_dim"])
        dummy_cnv = torch.zeros(1, data["dims"]["cnv_dim"])
        dummy_exp = torch.zeros(1, data["dims"]["exp_dim"])
        dummy_drug= torch.zeros(1, data["dims"]["drug_dim"])
        dummy_ctx = torch.zeros(1, data["dims"]["context_dim"])
        n_pw      = data["dims"]["n_pathways"]
        dummy_mp  = torch.zeros(1, n_pw * 3)
        dummy_cp  = torch.zeros(1, n_pw * 3)
        dummy_ep  = torch.zeros(1, n_pw * 3)

        if omic == "exp":
            inp = (dummy_mut, dummy_cnv, vals, dummy_drug,
                   dummy_ctx, dummy_mp, dummy_cp, dummy_ep)
        elif omic == "mut":
            inp = (vals, dummy_cnv, dummy_exp, dummy_drug,
                   dummy_ctx, dummy_mp, dummy_cp, dummy_ep)
        else:
            inp = (dummy_mut, vals, dummy_exp, dummy_drug,
                   dummy_ctx, dummy_mp, dummy_cp, dummy_ep)

        pred, *_ = model(*inp)
        model.zero_grad()
        pred.sum().backward()

        grad = vals.grad
        if omic == "cnv":
            attr = grad.abs().squeeze().detach().numpy()
        else:
            attr = (grad * vals).abs().squeeze().detach().numpy()

        top_idx   = attr.argsort()[::-1][:n]
        top_scores= attr[top_idx]
        symbols   = []
        for i in top_idx:
            eid    = genes[i].replace("exp_", "").replace("mut_", "").replace("cnv_", "")
            symbol = data["gene_symbols"].get(eid, eid)
            symbols.append(symbol)

        return symbols, top_scores.tolist()
    except Exception as e:
        return [], []

# ── Plotly chart helpers ───────────────────────────────
def omic_bar_chart(omic_weights):
    labels = ["Mutation", "CNV", "Expression"]
    values = [w * 100 for w in omic_weights]
    colors = [COLORS["accent"], "#6366F1", "#F59E0B"]
    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=13, family="Space Grotesk", color="white")))
    fig.update_layout(
        height=160,
        margin=dict(l=0, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 110]),
        yaxis=dict(showgrid=False, tickfont=dict(size=13, family="Space Grotesk")),
        showlegend=False)
    return fig

def pathway_bar_chart(top10_pw_idx, pathway_names, pw_scores):
    names  = [pathway_names[i] if i < len(pathway_names) else f"Pathway {i}"
              for i in top10_pw_idx]
    scores = [pw_scores[i] for i in top10_pw_idx]
    norm   = [s / max(scores) * 100 for s in scores]
    colors = [f"rgba(0,180,216,{0.4 + 0.6*(s/100)})" for s in norm]
    fig = go.Figure(go.Bar(
        x=norm[::-1], y=names[::-1],
        orientation="h",
        marker=dict(color=colors[::-1], line=dict(width=0)),
        text=[f"{s:.0f}%" for s in norm[::-1]],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=11, family="Space Grotesk", color="white")))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 110]),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, family="Space Grotesk")),
        showlegend=False)
    return fig

def gene_bar_chart(genes, scores, color):
    if not genes:
        return None
    norm = [s / max(scores) * 100 if max(scores) > 0 else 0 for s in scores]
    fig = go.Figure(go.Bar(
        x=norm[::-1], y=genes[::-1],
        orientation="h",
        marker=dict(color=color, opacity=0.85, line=dict(width=0)),
        text=[f"{s:.0f}%" for s in norm[::-1]],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=11, family="Space Grotesk", color="white")))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 110]),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, family="Space Grotesk")),
        showlegend=False)
    return fig

def true_vs_pred_plot(true_vals, pred_vals, drug_names, title="All Drugs"):
    pcc  = pearsonr(true_vals, pred_vals)[0]
    r2   = r2_score(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=true_vals, y=pred_vals,
        mode="markers",
        marker=dict(
            size=3,
            color=COLORS["accent"],
            opacity=0.4,
            line=dict(width=0)),
        name="Predictions",
        hovertemplate="True: %{x:.2f}<br>Pred: %{y:.2f}<extra></extra>"))
    mn = min(true_vals.min(), pred_vals.min())
    mx = max(true_vals.max(), pred_vals.max())
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        line=dict(color=COLORS["resistant"], width=2, dash="dash"),
        name="Perfect fit"))
    fig.update_layout(
        title=dict(
            text=f"True vs Predicted LN_IC50  |  PCC={pcc:.4f}  R²={r2:.4f}  RMSE={rmse:.4f}",
            font=dict(size=13, family="Space Grotesk")),
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
        xaxis=dict(title="True LN_IC50", gridcolor="rgba(0,0,0,0.05)",
                   tickfont=dict(size=11)),
        yaxis=dict(title="Predicted LN_IC50", gridcolor="rgba(0,0,0,0.05)",
                   tickfont=dict(size=11)),
        legend=dict(font=dict(size=11)))
    return fig

def training_curve_plot(train_log):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Training & Validation Loss",
                                        "Validation PCC"))
    fig.add_trace(go.Scatter(
        x=train_log["epoch"], y=train_log["train_loss"],
        name="Train Loss", line=dict(color=COLORS["accent"], width=2)),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=train_log["epoch"], y=train_log["val_loss"],
        name="Val Loss", line=dict(color=COLORS["resistant"], width=2)),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=train_log["epoch"], y=train_log["val_pcc"],
        name="Val PCC", line=dict(color=COLORS["sensitive"], width=2),
        fill="tozeroy", fillcolor="rgba(6,214,160,0.08)"),
        row=1, col=2)
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
        legend=dict(font=dict(size=11)),
        font=dict(family="Space Grotesk"))
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=10))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=10))
    return fig

def per_drug_chart(per_drug_df):
    df = per_drug_df.sort_values("pearson", ascending=True).tail(30)
    colors = [COLORS["sensitive"] if p >= 0.9 else
              COLORS["moderate"]  if p >= 0.8 else
              COLORS["resistant"] for p in df["pearson"]]
    fig = go.Figure(go.Bar(
        x=df["pearson"], y=df["drug"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.3f}" for p in df["pearson"]],
        textposition="outside",
        textfont=dict(size=10, family="Space Grotesk")))
    fig.update_layout(
        title="Per-Drug Pearson Correlation (Top 30)",
        height=600,
        margin=dict(l=0, r=60, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
        xaxis=dict(title="Pearson r", range=[0, 1.1],
                   gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, tickfont=dict(size=10, family="Space Grotesk")))
    return fig

def ranking_scatter(records):
    drugs  = [r["drug"]      for r in records]
    preds  = [r["pred_ic50"] for r in records]
    labels = [r["sensitivity"] for r in records]

    # Color by rank position: top = green, bottom = red, middle = amber
    n = len(records)
    bar_colors = []
    for i in range(n):
        if i < n * 0.33:
            bar_colors.append(COLORS["sensitive"])
        elif i < n * 0.66:
            bar_colors.append(COLORS["moderate"])
        else:
            bar_colors.append(COLORS["resistant"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=drugs,
        x=preds,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{p:.2f}" for p in preds],
        textposition="outside",
        textfont=dict(size=10, family="Space Grotesk", color=COLORS["text"]),
        hovertemplate="<b>%{y}</b><br>Predicted LN_IC50: %{x:.3f}<extra></extra>"))

    # Add legend annotations
    fig.add_annotation(x=0.02, y=1.05, xref="paper", yref="paper",
        text="Green = Most Sensitive    Amber = Moderate    Red = Most Resistant",
        showarrow=False, font=dict(size=10, color=COLORS["muted"]),
        align="left")

    mn = min(preds) - 0.3
    mx = max(preds) + 0.8
    fig.update_layout(
        height=max(300, n * 28),
        margin=dict(l=10, r=80, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
        xaxis=dict(title="Predicted LN_IC50", gridcolor="rgba(0,0,0,0.06)",
                   tickfont=dict(size=10), range=[mn, mx]),
        yaxis=dict(showgrid=False, tickfont=dict(size=10, family="Space Grotesk"),
                   autorange="reversed"))
    return fig

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    # Logo image
    logo_path = os.path.join(BASE_DIR, "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
        st.markdown("""
        <style>
        section[data-testid="stSidebar"] img {
        transform: scale(1.6);
        transform-origin: center center;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="logo-wrap">
            <svg width="56" height="56" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="28" cy="28" r="26" fill="rgba(0,180,216,0.12)" stroke="#00B4D8" stroke-width="1.5"/>
              <path d="M20 10 C20 10 24 17 28 17 C32 17 36 10 36 10" stroke="#00B4D8" stroke-width="2" stroke-linecap="round" fill="none"/>
              <path d="M20 46 C20 46 24 39 28 39 C32 39 36 46 36 46" stroke="#00B4D8" stroke-width="2" stroke-linecap="round" fill="none"/>
              <path d="M20 10 C17 15 17 23 20 28 C17 33 17 41 20 46" stroke="#90E0EF" stroke-width="1.5" stroke-linecap="round" fill="none"/>
              <path d="M36 10 C39 15 39 23 36 28 C39 33 39 41 36 46" stroke="#90E0EF" stroke-width="1.5" stroke-linecap="round" fill="none"/>
              <line x1="17" y1="19" x2="39" y2="19" stroke="#00B4D8" stroke-width="1.5" stroke-linecap="round"/>
              <line x1="17" y1="28" x2="39" y2="28" stroke="#00B4D8" stroke-width="1.5" stroke-linecap="round"/>
              <line x1="17" y1="37" x2="39" y2="37" stroke="#00B4D8" stroke-width="1.5" stroke-linecap="round"/>
            </svg>
            <div class="logo-name">Omic<span>Former</span></div>
            <div class="logo-tag">Biomolecular AI for Drug Response</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # Navigation
    page = st.radio(
        "Navigation",
        ["Home", "Single Drug Analysis", "Drug Ranking", "Model Performance"],
        label_visibility="collapsed")

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # Model stats
    st.markdown("<div style='font-size:0.7rem;font-weight:700;color:rgba(255,255,255,0.5);letter-spacing:0.1em;text-transform:uppercase;padding:0 0.25rem;margin-bottom:0.4rem'>Model Statistics</div>", unsafe_allow_html=True)
    for label, val in [
        ("Model", "OmicFormer"),
        ("Seed", "456 (best)"),
        ("Test PCC", "0.9479"),
        ("Test R²", "0.8981"),
        ("vs PASO", "+0.0042 ↑"),
        ("Drugs", "149"),
        ("Cell Lines", "937"),
        ("Parameters", "80.9M"),
    ]:
        st.markdown(
            f"<div class='spill'><span>{label}</span><b>{val}</b></div>",
            unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.68rem;color:rgba(255,255,255,0.35);padding:0 0.25rem;line-height:1.6'>"
        "Rana Amr &nbsp;|&nbsp; BUE CS &amp; AI 2026<br>"
        "Genomic-AI Precision Oncology"
        "</div>",
        unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────
with st.spinner("Loading OmicFormer data..."):
    D = load_all_data()

# ══════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            Genomic-AI <span>Precision Oncology</span><br>
            Drug Response Prediction
        </div>
        <div class="hero-sub">
            OmicFormer is a drug-conditioned multi-omic transformer that predicts cancer cell line
            sensitivity to 149 anti-cancer drugs using mutation, copy number variation, and gene
            expression profiles — with full explainability at the omic, pathway, and gene level.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label, delta in [
        (c1, "0.9467", "Pearson r",   "Mean ± 0.0012"),
        (c2, "0.8956", "R² Score",    "Mean ± 0.0024"),
        (c3, "149",    "Drugs",       "GDSC2 dataset"),
        (c4, "937",    "Cell Lines",  "19,459 pairs"),
        (c5, "+0.45%", "Beats PASO",  "PCC improvement"),
    ]:
        with col:
            st.markdown(f"""
            <div class="sc">
                <div class="sc-val">{val}</div>
                <div class="sc-lbl">{label}</div>
                <div class="sc-del">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-hdr">Architecture — 5 Novel Contributions</div>
        """, unsafe_allow_html=True)
        contribs = [
            "FiLM Drug Conditioning",
            "Pathway Token Attention (370 KEGG)",
            "Drug-Conditioned Pathway Attention",
            "Dual-Stream Per-Omic Encoder",
            "Cross-Omic Transformer (4-token)",
        ]
        for i, c in enumerate(contribs):
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.75rem;
                        padding:0.6rem 0;border-bottom:1px solid rgba(0,0,0,0.05)">
                <div style="background:{COLORS['accent']};color:white;border-radius:50%;
                            width:24px;height:24px;display:flex;align-items:center;
                            justify-content:center;font-size:0.75rem;font-weight:700;
                            flex-shrink:0">{i+1}</div>
                <div style="font-size:0.88rem;color:{COLORS['text']}">{c}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-hdr">Results vs State-of-the-Art</div>
        """, unsafe_allow_html=True)
        results_data = {
            "Model":   ["SVM", "RF", "XGBoost", "LightGBM", "PASO (2025)", "OmicFormer"],
            "PCC":     ["0.8735", "0.9048", "0.9288", "0.9305", "0.9425", "0.9467"],
            "Status":  ["Baseline","Ours ✓","Ours ✓","Ours ✓","SOTA","Ours ✓"],
        }
        df_res = pd.DataFrame(results_data)
        st.dataframe(
            df_res,
            hide_index=True,
            use_container_width=True,
            column_config={
                "PCC": st.column_config.TextColumn("PCC ± Std"),
                "Status": st.column_config.TextColumn(""),
            })
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:1rem">
        <div class="card-hdr">Explainability Framework — 3 Layers</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:0.5rem">
            <div style="text-align:center;padding:1rem;background:#F8FAFC;border-radius:12px">
                <div style="font-size:1.8rem;margin-bottom:0.5rem">
                    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                        <circle cx="16" cy="16" r="14" fill="rgba(0,180,216,0.1)" stroke="#00B4D8" stroke-width="1.5"/>
                        <text x="16" y="21" text-anchor="middle" font-size="12" fill="#00B4D8" font-weight="700">①</text>
                    </svg>
                </div>
                <div style="font-weight:600;font-size:0.85rem;color:{COLORS['primary']}">Omic Driver</div>
                <div style="font-size:0.78rem;color:{COLORS['muted']};margin-top:0.3rem">Which omic type (mutation, CNV, expression) dominates this prediction</div>
            </div>
            <div style="text-align:center;padding:1rem;background:#F8FAFC;border-radius:12px">
                <div style="font-size:1.8rem;margin-bottom:0.5rem">
                    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                        <circle cx="16" cy="16" r="14" fill="rgba(99,102,241,0.1)" stroke="#6366F1" stroke-width="1.5"/>
                        <text x="16" y="21" text-anchor="middle" font-size="12" fill="#6366F1" font-weight="700">②</text>
                    </svg>
                </div>
                <div style="font-weight:600;font-size:0.85rem;color:{COLORS['primary']}">Pathway Activation</div>
                <div style="font-size:0.78rem;color:{COLORS['muted']};margin-top:0.3rem">Which KEGG biological pathways the model attended to most strongly</div>
            </div>
            <div style="text-align:center;padding:1rem;background:#F8FAFC;border-radius:12px">
                <div style="font-size:1.8rem;margin-bottom:0.5rem">
                    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                        <circle cx="16" cy="16" r="14" fill="rgba(245,158,11,0.1)" stroke="#F59E0B" stroke-width="1.5"/>
                        <text x="16" y="21" text-anchor="middle" font-size="12" fill="#F59E0B" font-weight="700">③</text>
                    </svg>
                </div>
                <div style="font-weight:600;font-size:0.85rem;color:{COLORS['primary']}">Gene Attribution</div>
                <div style="font-size:0.78rem;color:{COLORS['muted']};margin-top:0.3rem">Specific genes driving the prediction via gradient×input attribution</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE: SINGLE DRUG ANALYSIS
# ══════════════════════════════════════════════════════
elif page == "Single Drug Analysis":
    st.markdown('<div class="pg-title">Single Drug Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Select a cell line and drug to receive a full explainable prediction with clinical narrative and similar cases.</div>', unsafe_allow_html=True)

    col_sel1, col_sel2, col_sel3 = st.columns([2, 2, 1])
    with col_sel1:
        cell_line = st.selectbox("Cell Line", D["cell_lines"], key="single_cell")
    with col_sel2:
        available_drugs = sorted(D["cell_drug_map"].get(cell_line, D["drugs"]))
        drug = st.selectbox("Drug", available_drugs, key="single_drug")
    with col_sel3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("Analyse", key="run_single")

    if run_btn:
        key = (cell_line, drug)
        if key not in D["lookup_index"]:
            st.warning(f"No test data found for {cell_line} + {drug}. Try a different combination.")
        else:
            idx = D["lookup_index"][key]
            rec = D["lookup"][idx]

            pred_ic50  = rec["pred_ic50"]
            true_ic50  = rec["true_ic50"]
            omic_w     = rec["omic_weights"]
            top10_pw   = rec["top10_pathways"]
            sens_label, badge_class = sensitivity_label(pred_ic50, drug, D)
            mechanism  = D["mechanisms"].get(drug, "Mechanism not available")
            pw_scores  = D["pw_importance"][idx]
            pw_names   = [D["pathway_names"][i] if i < len(D["pathway_names"]) else f"Pathway {i}"
                          for i in top10_pw]

            # ── Row 1: Prediction + Omic + Pathway ────
            r1c1, r1c2, r1c3 = st.columns([1, 1, 1.2])

            with r1c1:
                st.markdown(f"""
                <div class="card">
                    <div class="card-hdr">Prediction</div>
                    <div class="pred-val">{pred_ic50:.3f}<span class="pred-unit">LN_IC50</span></div>
                    <div class="sensitivity-badge {badge_class}">{sens_label}</div>
                    <div style="margin-top:1rem;font-size:0.82rem;color:{COLORS['muted']}">
                        <div style="display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid rgba(0,0,0,0.05)">
                            <span>Observed IC50</span>
                            <strong style="color:{COLORS['text']}">{true_ic50:.3f}</strong>
                        </div>
                        <div style="display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid rgba(0,0,0,0.05)">
                            <span>Absolute Error</span>
                            <strong style="color:{COLORS['text']}">{abs(pred_ic50-true_ic50):.3f}</strong>
                        </div>
                        <div style="display:flex;justify-content:space-between;padding:0.3rem 0">
                            <span>Cell Line</span>
                            <strong style="color:{COLORS['text']};font-family:'JetBrains Mono',monospace;font-size:0.78rem">{cell_line}</strong>
                        </div>
                    </div>
                    <div class="mech" style="margin-top:1rem">{mechanism}</div>
                </div>""", unsafe_allow_html=True)

            with r1c2:
                st.markdown('<div class="card"><div class="card-hdr">Layer 1 — Omic Driver</div>', unsafe_allow_html=True)
                dominant_omic = ["Mutation", "CNV", "Expression"][np.argmax(omic_w)]
                st.markdown(f'<div style="font-size:0.82rem;color:{COLORS["muted"]};margin-bottom:0.5rem">Dominant omic: <strong style="color:{COLORS["primary"]}">{dominant_omic}</strong></div>', unsafe_allow_html=True)
                st.plotly_chart(omic_bar_chart(omic_w), use_container_width=True, config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

            with r1c3:
                st.markdown('<div class="card"><div class="card-hdr">Layer 2 — Top Pathways</div>', unsafe_allow_html=True)
                st.plotly_chart(pathway_bar_chart(top10_pw, D["pathway_names"], pw_scores),
                                use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Row 2: Gene Attribution ────────────────
            st.markdown('<div class="card"><div class="card-hdr">Layer 3 — Gene Attribution (Gradient × Input)</div>', unsafe_allow_html=True)

            try:
                model, device = load_model(D["dims"])
                g1, g2, g3 = st.columns(3)

                with g1:
                    st.markdown(f'<div style="font-size:0.82rem;font-weight:600;color:{COLORS["primary"]};margin-bottom:0.5rem;text-align:center">Mutation Genes</div>', unsafe_allow_html=True)
                    with st.spinner("Computing..."):
                        mut_genes, mut_scores = get_top_genes(idx, "mut", D, model, device)
                    if mut_genes:
                        fig = gene_bar_chart(mut_genes, mut_scores, COLORS["accent"])
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.markdown('<div class="warn">Gene names not available</div>', unsafe_allow_html=True)

                with g2:
                    st.markdown(f'<div style="font-size:0.82rem;font-weight:600;color:{COLORS["primary"]};margin-bottom:0.5rem;text-align:center">CNV Genes</div>', unsafe_allow_html=True)
                    with st.spinner("Computing..."):
                        cnv_genes, cnv_scores = get_top_genes(idx, "cnv", D, model, device)
                    if cnv_genes:
                        fig = gene_bar_chart(cnv_genes, cnv_scores, "#6366F1")
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.markdown('<div class="warn">Gene names not available</div>', unsafe_allow_html=True)

                with g3:
                    st.markdown(f'<div style="font-size:0.82rem;font-weight:600;color:{COLORS["primary"]};margin-bottom:0.5rem;text-align:center">Expression Genes</div>', unsafe_allow_html=True)
                    with st.spinner("Computing..."):
                        exp_genes, exp_scores = get_top_genes(idx, "exp", D, model, device)
                    if exp_genes:
                        fig = gene_bar_chart(exp_genes, exp_scores, "#F59E0B")
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.markdown('<div class="warn">Gene names not available</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="warn">Gene attribution requires model files. Error: {str(e)[:80]}</div>', unsafe_allow_html=True)
                mut_genes, mut_scores = [], []
                exp_genes, exp_scores = [], []

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Row 3: Narrative + Similar Cases ──────
            nc1, nc2 = st.columns([1.3, 1])

            with nc1:
                st.markdown('<div class="card"><div class="card-hdr">Clinical Narrative — AI Generated</div>', unsafe_allow_html=True)
                similar = get_similar_cases(idx, drug, D)
                with st.spinner("Generating clinical narrative..."):
                    narrative = generate_narrative(
                        drug, mechanism, pred_ic50, sens_label,
                        omic_w, pw_names,
                        mut_genes[:5] if mut_genes else ["N/A"],
                        exp_genes[:5] if exp_genes else ["N/A"],
                        similar)
                st.markdown(f'<div class="narrative">{narrative}</div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.72rem;color:#9CA3AF;margin-top:0.5rem">Generated by Llama 3.3 70B from model evidence only. Not a clinical recommendation.</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with nc2:
                st.markdown('<div class="card"><div class="card-hdr">Similar Historical Cases</div>', unsafe_allow_html=True)
                if similar:
                    for c in similar:
                        ic50_color = COLORS["sensitive"] if c["status"] == "Sensitive" else COLORS["resistant"]
                        st.markdown(f"""
                        <div class="case-row">
                            <div class="case-id">{c['cell_id']}</div>
                            <div class="case-sim">sim: {c['similarity']:.2f}</div>
                            <div class="case-ic50" style="color:{ic50_color}">{c['ic50']:.3f}</div>
                            <div style="font-size:0.75rem;color:{ic50_color};font-weight:600">{c['status']}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warn">No similar cases found for this drug.</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE: DRUG RANKING
# ══════════════════════════════════════════════════════
elif page == "Drug Ranking":
    st.markdown('<div class="pg-title">Drug Ranking</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Rank all drugs for a cell line by predicted sensitivity.</div>', unsafe_allow_html=True)

    rc1, rc2, rc3, rc4 = st.columns([2, 1, 1, 1])
    with rc1:
        rank_cell = st.selectbox("Cell Line", D["cell_lines"], key="rank_cell")
    with rc2:
        top_n = st.selectbox("Show Top", [10, 20, 50, 149], key="rank_n")
    with rc3:
        sort_by = st.selectbox("Sort", ["Most Sensitive", "Most Resistant"], key="rank_sort")
    with rc4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        rank_btn = st.button("Rank Drugs", key="run_rank")

    if rank_btn:
        available = D["cell_drug_map"].get(rank_cell, [])
        if not available:
            st.warning("No data available for this cell line.")
        else:
            ranking_records = []
            for drug in available:
                key = (rank_cell, drug)
                if key in D["lookup_index"]:
                    idx = D["lookup_index"][key]
                    rec = D["lookup"][idx]
                    sens, _ = sensitivity_label(rec["pred_ic50"], drug, D)
                    ranking_records.append({
                        "idx": idx,
                        "drug": drug,
                        "pred_ic50": rec["pred_ic50"],
                        "true_ic50": rec["true_ic50"],
                        "sensitivity": sens,
                        "omic_weights": rec["omic_weights"],
                        "top10_pathways": rec["top10_pathways"],
                    })

            ascending = sort_by == "Most Sensitive"
            ranking_records.sort(key=lambda x: x["pred_ic50"], reverse=not ascending)
            ranking_records = ranking_records[:top_n]

            # Bar chart overview
            st.markdown('<div class="card"><div class="card-hdr">Sensitivity Profile Overview</div>', unsafe_allow_html=True)
            st.plotly_chart(ranking_scatter(ranking_records), use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

            # Ranked list
            st.markdown('<div class="card"><div class="card-hdr">Ranked Drug List — Click to Expand</div>', unsafe_allow_html=True)

            for rank, rec in enumerate(ranking_records):
                sens_color = sensitivity_color(rec["sensitivity"])
                label_short = rec["sensitivity"].replace(" SENSITIVITY","").replace(" RESISTANCE","")

                with st.expander(
                    f"#{rank+1}  {rec['drug']}  —  IC50: {rec['pred_ic50']:.3f}  |  {rec['sensitivity']}",
                    expanded=False):

                    ec1, ec2, ec3 = st.columns([1, 1, 1.2])
                    omic_w   = rec["omic_weights"]
                    top10_pw = rec["top10_pathways"]
                    pw_scores= D["pw_importance"][rec["idx"]]
                    mechanism= D["mechanisms"].get(rec["drug"], "")

                    with ec1:
                        st.markdown(f"""
                        <div style="background:{COLORS['bg']};border-radius:12px;padding:1rem">
                            <div style="font-size:0.72rem;color:{COLORS['muted']};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">Prediction</div>
                            <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:700;color:{COLORS['primary']}">{rec['pred_ic50']:.3f}</div>
                            <div class="sensitivity-badge" style="background:{sens_color}22;color:{sens_color};border:1px solid {sens_color}44;margin-top:0.3rem">{rec['sensitivity']}</div>
                            <div style="margin-top:0.75rem;font-size:0.8rem;color:{COLORS['muted']}">Observed: {rec['true_ic50']:.3f}</div>
                            <div class="mech" style="margin-top:0.75rem;font-size:0.78rem">{mechanism[:120]}...</div>
                        </div>""", unsafe_allow_html=True)

                    with ec2:
                        st.markdown(f'<div style="font-size:0.72rem;color:{COLORS["muted"]};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem">Omic Driver</div>', unsafe_allow_html=True)
                        st.plotly_chart(omic_bar_chart(omic_w), use_container_width=True,
                                       config={"displayModeBar": False})

                    with ec3:
                        st.markdown(f'<div style="font-size:0.72rem;color:{COLORS["muted"]};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.3rem">Top Pathways</div>', unsafe_allow_html=True)
                        st.plotly_chart(
                            pathway_bar_chart(top10_pw[:6], D["pathway_names"], pw_scores),
                            use_container_width=True,
                            config={"displayModeBar": False})

                    # Narrative
                    pw_names = [D["pathway_names"][i] if i < len(D["pathway_names"]) else f"Pathway {i}"
                                for i in top10_pw]
                    similar  = get_similar_cases(rec["idx"], rec["drug"], D, k=3)
                    with st.spinner("Generating narrative..."):
                        narrative = generate_narrative(
                            rec["drug"], mechanism,
                            rec["pred_ic50"], rec["sensitivity"],
                            omic_w, pw_names, [], [], similar)
                    st.markdown(f'<div class="narrative" style="margin-top:0.75rem">{narrative}</div>', unsafe_allow_html=True)
                    st.markdown('<div style="font-size:0.7rem;color:#9CA3AF;margin-top:0.3rem">Not a clinical recommendation.</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown('<div class="pg-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Comprehensive evaluation of OmicFormer across 3 seeds, comparison with baselines, and analysis of predictions.</div>', unsafe_allow_html=True)

    # Results comparison table
    st.markdown('<div class="card"><div class="card-hdr">Full Results — All Models vs PASO (PLOS Computational Biology 2025)</div>', unsafe_allow_html=True)
    comp_df = pd.DataFrame({
        "Model":    ["SVM (PASO)", "RF (PASO)", "XGBoost (PASO)", "LightGBM (PASO)",
                     "RF (Ours)", "XGBoost (Ours)", "LightGBM (Ours)",
                     "OmicFusionNet", "PASO (best)", "OmicFormer (Ours)"],
        "PCC":      ["0.8735±0.0021", "0.9006±0.0025", "0.9100±0.0016", "0.9054±0.0026",
                     "0.9048±0.0016", "0.9288±0.0013", "0.9305±0.0013",
                     "0.9474±0.0015", "0.9425", "0.9467±0.0013"],
        "RMSE":     ["1.3632±0.0088", "—", "1.1611±0.0066", "1.1953±0.0079",
                     "~1.277", "~1.085", "~1.077",
                     "~0.956", "—", "0.9444±0.0100"],
        "R²":       ["—", "0.8110±0.0044", "—", "—",
                     "~0.809", "~0.862", "~0.865",
                     "~0.895", "—", "0.8956±0.0023"],
        "Beats PASO": ["—","—","—","—","✓","+1.9%","+2.5%","✓","baseline","✓ +0.42%"],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Per-seed results
    st.markdown('<div class="card"><div class="card-hdr">OmicFormer — Per-Seed Results</div>', unsafe_allow_html=True)
    seed_df = pd.DataFrame({
        "Seed": ["42", "123", "456 (best)", "Mean ± Std"],
        "PCC":  ["0.945078", "0.947287", "0.947859", "0.9467 ± 0.0013"],
        "SCC":  ["0.927155", "0.928538", "0.930124", "0.9286 ± 0.0012"],
        "RMSE": ["0.956944", "0.943482", "0.932614", "0.9444 ± 0.0100"],
        "R²":   ["0.892432", "0.896319", "0.898063", "0.8956 ± 0.0023"],
        "MAE":  ["0.692018", "0.681515", "0.676254", "0.6833 ± 0.0064"],
    })
    st.dataframe(seed_df, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # True vs Predicted plot
    st.markdown('<div class="card"><div class="card-hdr">True vs Predicted LN_IC50 — Test Set (19,459 samples)</div>', unsafe_allow_html=True)
    true_flat = D["true"].flatten()
    pred_flat = D["preds"].flatten()
    drug_flat = D["drug_names_test"]

    tab_all, tab_drug = st.tabs(["All Drugs", "By Drug"])
    with tab_all:
        st.plotly_chart(true_vs_pred_plot(true_flat, pred_flat, drug_flat),
                       use_container_width=True, config={"displayModeBar": False})
    with tab_drug:
        sel_drug = st.selectbox("Select Drug", D["drugs"], key="perf_drug")
        mask = drug_flat == sel_drug
        if mask.sum() > 0:
            st.plotly_chart(
                true_vs_pred_plot(true_flat[mask], pred_flat[mask],
                                  drug_flat[mask], sel_drug),
                use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # Training curve
    if D["train_log"] is not None:
        st.markdown('<div class="card"><div class="card-hdr">Training Curve — Seed 456</div>', unsafe_allow_html=True)
        st.plotly_chart(training_curve_plot(D["train_log"]),
                       use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div style="font-size:0.8rem;color:{COLORS["muted"]};margin-top:0.3rem">Early stopping at epoch 115 | Best val PCC: 0.94724</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Per-drug performance
    if D["per_drug"] is not None:
        st.markdown('<div class="card"><div class="card-hdr">Per-Drug Performance — Top 30 Drugs by Pearson r</div>', unsafe_allow_html=True)
        st.plotly_chart(per_drug_chart(D["per_drug"]),
                       use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Summary stats
        df_pd = D["per_drug"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, val, label in [
            (mc1, f"{df_pd['pearson'].mean():.4f}", "Mean Drug PCC"),
            (mc2, f"{(df_pd['pearson']>=0.9).sum()}", "Drugs PCC ≥ 0.9"),
            (mc3, f"{(df_pd['pearson']>=0.8).sum()}", "Drugs PCC ≥ 0.8"),
            (mc4, f"{df_pd['pearson'].max():.4f}", "Best Drug PCC"),
        ]:
            with col:
                st.markdown(f"""
                <div class="sc">
                    <div class="sc-val" style="font-size:1.6rem">{val}</div>
                    <div class="sc-lbl">{label}</div>
                </div>""", unsafe_allow_html=True)
