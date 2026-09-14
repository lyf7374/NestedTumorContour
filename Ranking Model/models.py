"""Pairwise contour-ranking models and checkpoint loading.

Input columns: crop-voxel X, Y, Z, FLAIR, T1, T1ce, T2. Forward returns
(sigmoid probability, logit); probability > 0.5 favors the first contour.
The learned nonlinear head does not enforce exact antisymmetry.
"""
import torch
import torch.nn as nn

PAPER_MODEL_KWARGS = dict(input_dim=7, d_model=128, nhead=8, num_layers=6,
                          dim_feedforward=256, dropout=0.2)

class SimpleComparatorMLP(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, g_a, g_b):
        mu_a = g_a.mean(dim=1)  # [B,7]
        mu_b = g_b.mean(dim=1)  # [B,7]
        h_a = self.mlp(mu_a)
        h_b = self.mlp(mu_b)
        h_diff = h_a - h_b
        logits = self.classifier(h_diff).squeeze(1)
        p = torch.sigmoid(logits)
        return p, logits

class PointTransformerComparator(nn.Module):
    def __init__(self, input_dim=7, d_model=64, nhead=4,
                 num_layers=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim-3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, g_a, g_b):
        def encode(g):
            coords = g[:, :, :3]
            feats  = g[:, :, 3:]
            f_embed = self.input_mlp(feats)
            p_embed = self.pos_mlp(coords)
            x = (f_embed + p_embed).permute(1, 0, 2)  # [S,B,E]
            x = self.transformer(x).permute(1, 0, 2)
            return x.mean(dim=1)
        h_a = encode(g_a)
        h_b = encode(g_b)
        h_diff = h_a - h_b
        logits = self.classifier(h_diff).squeeze(1)
        return torch.sigmoid(logits), logits

# =========================
# Load checkpoint helpers
# =========================

def build_model(model_type, m_kwargs):
    if model_type == "mlp":
        return SimpleComparatorMLP(**m_kwargs)
    if model_type == "transformer":
        return PointTransformerComparator(**m_kwargs)
    raise ValueError("Unknown model_type: %s" % model_type)

def load_pretrained(model, ckpt_path, map_location="cpu", strict=True):
    sd = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        state_dict = sd["state_dict"]
    elif isinstance(sd, dict) and "model_state_dict" in sd:
        state_dict = sd["model_state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        state_dict = sd["model"]
    else:
        state_dict = sd
    if isinstance(state_dict, dict):
        state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=strict)
    return model
