import math
import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_feats=128, row_num_embed=50, col_num_embed=50, **kwargs):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats

    def forward(self, mask):
        h, w = mask.shape[-2:]
        i = torch.arange(w, device=mask.device)
        j = torch.arange(h, device=mask.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1)
        return pos.permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)


class SinePositionalEncoding(nn.Module):
    def __init__(self, num_feats=128, normalize=False, offset=0.0, temperature=10000, **kwargs):
        super().__init__()
        self.num_feats = num_feats
        self.normalize = normalize
        self.offset = offset
        self.temperature = temperature

    def forward(self, mask):
        not_mask = ~mask.to(torch.bool)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + eps) * 2 * math.pi
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + eps) * 2 * math.pi
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


def build_positional_encoding(cfg):
    cfg = dict(cfg)
    typ = cfg.pop("type")
    if typ == "LearnedPositionalEncoding":
        return LearnedPositionalEncoding(**cfg)
    if typ == "SinePositionalEncoding":
        return SinePositionalEncoding(**cfg)
    raise KeyError(f"Unsupported positional encoding: {typ}")
