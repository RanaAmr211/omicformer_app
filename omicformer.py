import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, drug_dim):
        super().__init__()
        self.bn    = nn.BatchNorm1d(feature_dim)
        self.gamma = nn.Linear(drug_dim, feature_dim)
        self.beta  = nn.Linear(drug_dim, feature_dim)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, drug_emb):
        γ = self.gamma(drug_emb)
        β = self.beta(drug_emb)
        # ← ONLY FIX: clamp to prevent NaN explosion
        γ = torch.clamp(γ, -10, 10)
        β = torch.clamp(β, -10, 10)
        return γ * self.bn(x) + β


class FiLMBottleneck(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 output_dim, drug_dim, dropout=0.2):
        super().__init__()
        self.fc1   = nn.Linear(input_dim,  hidden_dim)
        self.film1 = FiLMLayer(hidden_dim, drug_dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(hidden_dim, output_dim)
        self.film2 = FiLMLayer(output_dim, drug_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, drug_emb):
        h = F.relu(self.film1(self.fc1(x), drug_emb))
        h = self.drop1(h)
        h = F.relu(self.film2(self.fc2(h), drug_emb))
        h = self.drop2(h)
        return h


class PathwayTokenAttention(nn.Module):
    def __init__(self, token_dim=64, n_heads=8,
                 ff_dim=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim   = token_dim,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True)
        self.ff = nn.Sequential(
            nn.Linear(token_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, token_dim),
            nn.Dropout(dropout))
        self.ln1 = nn.LayerNorm(token_dim)
        self.ln2 = nn.LayerNorm(token_dim)

    def forward(self, x):
        attn_out, attn_weights = self.attn(
            x, x, x,
            need_weights         = True,
            average_attn_weights = True)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x, attn_weights


class DrugConditionedPathwayAttention(nn.Module):
    def __init__(self, token_dim=64, n_heads=8,
                 drug_dim=512, dropout=0.1):
        super().__init__()
        self.film = FiLMLayer(token_dim, drug_dim)
        self.attn = PathwayTokenAttention(
            token_dim, n_heads,
            ff_dim=token_dim*4, dropout=dropout)

    def forward(self, tokens, drug_emb):
        B, N, D     = tokens.shape
        tokens_flat = tokens.view(B*N, D)
        drug_rep    = drug_emb.unsqueeze(1)\
                              .expand(-1,N,-1)\
                              .reshape(B*N, -1)
        tokens_flat = self.film(tokens_flat, drug_rep)
        tokens      = tokens_flat.view(B, N, D)
        tokens, attn_weights = self.attn(tokens)
        return tokens, attn_weights


class OmicFormer(nn.Module):
    """
    OmicFormer: Drug-Conditioned Multi-Omic Transformer
    ORIGINAL architecture — identical to best run.
    Only change: FiLM clamp in FiLMLayer (above).

    forward() now returns 9 values:
        pred, attn, mut_pw_attn, cnv_pw_attn, exp_pw_attn,
        h_mut, h_cnv, h_exp, h_drug
    The last 4 are the post-transformer embeddings used
    for explainability (case-based retrieval, attribution).
    """

    def __init__(self,
                 mut_dim, cnv_dim, exp_dim,
                 drug_dim    = 2048,
                 context_dim = 108,
                 n_pathways  = 370,
                 pw_stats    = 3,
                 token_dim   = 64,
                 bottleneck  = 512,
                 n_pw_heads  = 8,
                 n_co_heads  = 8,
                 n_co_layers = 2,
                 dropout     = 0.15,
                 use_gene_attention = False):
        super().__init__()

        self.drug_enc = nn.Sequential(
            nn.Linear(drug_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout))

        self.ctx_enc = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, token_dim),
            nn.BatchNorm1d(token_dim),
            nn.ReLU(),
            nn.Dropout(dropout))

        self.drug_fusion = nn.Sequential(
            nn.Linear(bottleneck + token_dim, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout))

        self.mut_film_enc = FiLMBottleneck(
            mut_dim, 1024, bottleneck,
            bottleneck, dropout)
        self.cnv_film_enc = FiLMBottleneck(
            cnv_dim, 1024, bottleneck,
            bottleneck, dropout)
        self.exp_film_enc = FiLMBottleneck(
            exp_dim, 2048, bottleneck,
            bottleneck, dropout)

        self.pw_embed    = nn.Linear(pw_stats, token_dim)
        self.mut_pw_attn = DrugConditionedPathwayAttention(
            token_dim, n_pw_heads, bottleneck, dropout)
        self.cnv_pw_attn = DrugConditionedPathwayAttention(
            token_dim, n_pw_heads, bottleneck, dropout)
        self.exp_pw_attn = DrugConditionedPathwayAttention(
            token_dim, n_pw_heads, bottleneck, dropout)

        self.mut_stream = nn.Sequential(
            nn.Linear(bottleneck + token_dim, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout))
        self.cnv_stream = nn.Sequential(
            nn.Linear(bottleneck + token_dim, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout))
        self.exp_stream = nn.Sequential(
            nn.Linear(bottleneck + token_dim, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout))

        self.drug_to_mut = nn.Linear(bottleneck, bottleneck)
        self.drug_to_cnv = nn.Linear(bottleneck, bottleneck)
        self.drug_to_exp = nn.Linear(bottleneck, bottleneck)
        self.mut_ln      = nn.LayerNorm(bottleneck)
        self.cnv_ln      = nn.LayerNorm(bottleneck)
        self.exp_ln      = nn.LayerNorm(bottleneck)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = bottleneck,
            nhead           = n_co_heads,
            dim_feedforward = bottleneck * 2,
            dropout         = dropout,
            batch_first     = True,
            activation      = "gelu")
        self.cross_omic_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_co_layers)

        self.attn_net = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1))

        fusion_dim = bottleneck * 4
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1))

        self._init_weights()
        self.n_pathways = n_pathways
        self.token_dim  = token_dim

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reshape_pathway_tokens(self, pw_flat):
        B = pw_flat.shape[0]
        return pw_flat.view(B, self.n_pathways, 3)

    def forward(self, X_mut, X_cnv, X_exp,
                X_drug, X_ctx,
                X_mut_pw, X_cnv_pw, X_exp_pw):

        h_drug_raw = self.drug_enc(X_drug)
        h_ctx      = self.ctx_enc(X_ctx)
        h_drug     = self.drug_fusion(
            torch.cat([h_drug_raw, h_ctx], dim=1))

        h_mut_raw = self.mut_film_enc(X_mut, h_drug)
        h_cnv_raw = self.cnv_film_enc(X_cnv, h_drug)
        h_exp_raw = self.exp_film_enc(X_exp, h_drug)

        mut_tokens = self.pw_embed(
            self.reshape_pathway_tokens(X_mut_pw))
        cnv_tokens = self.pw_embed(
            self.reshape_pathway_tokens(X_cnv_pw))
        exp_tokens = self.pw_embed(
            self.reshape_pathway_tokens(X_exp_pw))

        mut_tokens, mut_pw_attn = \
            self.mut_pw_attn(mut_tokens, h_drug)
        cnv_tokens, cnv_pw_attn = \
            self.cnv_pw_attn(cnv_tokens, h_drug)
        exp_tokens, exp_pw_attn = \
            self.exp_pw_attn(exp_tokens, h_drug)

        h_mut = self.mut_stream(torch.cat(
            [h_mut_raw, mut_tokens.mean(dim=1)], dim=1))
        h_cnv = self.cnv_stream(torch.cat(
            [h_cnv_raw, cnv_tokens.mean(dim=1)], dim=1))
        h_exp = self.exp_stream(torch.cat(
            [h_exp_raw, exp_tokens.mean(dim=1)], dim=1))

        h_mut = self.mut_ln(
            h_mut + self.drug_to_mut(h_drug))
        h_cnv = self.cnv_ln(
            h_cnv + self.drug_to_cnv(h_drug))
        h_exp = self.exp_ln(
            h_exp + self.drug_to_exp(h_drug))

        tokens = torch.stack(
            [h_mut, h_cnv, h_exp, h_drug], dim=1)
        tokens = self.cross_omic_transformer(tokens)
        h_mut  = tokens[:, 0, :]
        h_cnv  = tokens[:, 1, :]
        h_exp  = tokens[:, 2, :]
        h_drug = tokens[:, 3, :]

        attn  = self.attn_net(h_drug)
        fused = torch.cat(
            [h_mut, h_cnv, h_exp, h_drug], dim=1)
        pred  = self.fusion_mlp(fused)

        # ── CHANGED: now returns post-transformer embeddings
        # for explainability (pathway attribution, case retrieval)
        # h_mut, h_cnv, h_exp, h_drug → each shape [B, 512]
        return pred, attn, mut_pw_attn, \
               cnv_pw_attn, exp_pw_attn, \
               h_mut, h_cnv, h_exp, h_drug