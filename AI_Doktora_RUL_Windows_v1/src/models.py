"""Un encodeur CNN commun, quatre variantes comparables et deux sorties.

Axes : B=batch, W=cycles observés, P=128 instants, C=3 signaux.
Les masques/conv causaux interdisent les observations des cycles suivants.
"""
import torch
from torch import nn
from torch.nn import functional as F

MODELS = ("cnn", "cnn_lstm", "cnn_tcn", "hybrid")


class CycleEncoder(nn.Module):
    """Deux convolutions 1D sur le signal d'un cycle de décharge.

    Les petits noyaux capturent la forme locale V/I/T; 2 couches limitent
    les paramètres sur seulement quelques batteries NASA.
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4))
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(128, hidden), nn.ReLU())
        self.fuse_aux = nn.Linear(hidden + 2, hidden)

    def forward(self, x, aux):
        b, w, p, c = x.shape
        cycle = x.reshape(b * w, p, c).transpose(1, 2)
        features = self.projection(self.features(cycle)).reshape(b, w, -1)
        return F.relu(self.fuse_aux(torch.cat([features, aux], dim=-1)))


class CausalBlock(nn.Module):
    """Convolution résiduelle sur les cycles; padding seulement à gauche."""
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.kernel = 3
        self.dilation = dilation
        self.conv = nn.Conv1d(width, width, kernel_size=3, dilation=dilation)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        left = self.dilation * (self.kernel - 1)
        y = self.conv(F.pad(x, (left, 0)))
        return x + self.dropout(self.activation(y))


class BatteryPredictor(nn.Module):
    """Variants : CNN; CNN-LSTM; CNN-TCN; CNN-MHA-TCN (hybrid).

    Attention par 4 têtes causales. TCN de 2 blocs de dilatations 1 et 2.
    Les deux têtes sont linéaires : les cibles z-score peuvent être négatives.
    """
    def __init__(self, variant="hybrid", hidden=32, heads=4, dropout=0.10):
        super().__init__()
        if variant not in MODELS:
            raise ValueError("Modele inconnu : " + variant)
        if hidden % heads:
            raise ValueError("hidden doit etre divisible par heads")
        self.variant = variant
        self.encoder = CycleEncoder(hidden)
        if variant == "cnn_lstm":
            self.lstm = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        if variant == "hybrid":
            self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout,
                                                    batch_first=True)
            self.attention_norm = nn.LayerNorm(hidden)
        if variant in ("cnn_tcn", "hybrid"):
            self.temporal = nn.Sequential(CausalBlock(hidden, 1, dropout),
                                          CausalBlock(hidden, 2, dropout))
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(),
                                  nn.Dropout(dropout))
        self.soh = nn.Linear(16, 1)
        self.rul = nn.Linear(16, 1)

    def forward(self, x, aux):
        if self.variant == "cnn":
            # Référence sans mémoire : encode seulement la dernière décharge.
            x, aux = x[:, -1:, :, :], aux[:, -1:, :]
        h = self.encoder(x, aux)  # [B, W, hidden]
        if self.variant == "cnn_lstm":
            h, _ = self.lstm(h)
        if self.variant == "hybrid":
            w = h.size(1)
            # True = position future inaccessible pour toute requête ancienne.
            mask = torch.ones(w, w, dtype=torch.bool, device=h.device).triu(1)
            attended, _ = self.attention(h, h, h, attn_mask=mask,
                                         need_weights=False)
            h = self.attention_norm(h + attended)
        if self.variant in ("cnn_tcn", "hybrid"):
            h = self.temporal(h.transpose(1, 2)).transpose(1, 2)
        last = self.head(h[:, -1, :])
        return torch.cat([self.soh(last), self.rul(last)], dim=1)
