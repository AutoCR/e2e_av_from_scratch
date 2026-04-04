import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import DropPath
from timm.layers import Mlp

class DiffusionPlanner(nn.Module):
    def __init__(
            self,
            cfg: dict,
    ):
        super().__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs
    
class DiffusionPlannerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = None
        
## Encoder

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = cfg['hidden_dim']
        self.token_num = cfg['agent_num'] + cfg['static_objects_num'] + cfg['lane_num']
        
        self.neighbor_encoder = AgentFusionEncoder(
            cfg['time_len'],
            drop_path_rate=cfg['encoder_drop_path_rate'],
            hidden_dim = cfg['hidden_dim'],
            depth=cfg['encoder_depth'],
        )
        self.static_encoder = StaticFusionEncoder(
            cfg['static_objects_state_dim'],
            drop_path_rate=cfg['encoder_drop_path_rate'],
            hidden_dim=cfg['hidden_dim'],
        )
        self.lane_encoder = LaneFusionEncoder(
            cfg['lane_len'],
            drop_path_rate=cfg['encoder_drop_path_rate'],
            hidden_dim=cfg['hidden_dim'],
            depth=cfg['encoder_depth'],
        )
        self.fusion = FusionEncoder(
            hidden_dim=cfg['hidden_dim'],
            num_heads=cfg['num_heads'],
            drop_path_rate=cfg['encoder_drop_path_rate'],
            depth=cfg['encoder_depth'],
        )
        self.pos_emb = nn.Linear(7, cfg['hidden_dim'])

    def forward(self, inputs):

        # - neighbor_agents_past: (B, Pn, T, 11), T 21, [x, y, cos, sin, vx, vy, width, len, 8:11(type)]
        # - static_objects: (B, Ps, 10), [x, y, cos, sin, width, len, 6:10(type)]
        # - lanes: (B, Pl, L, 12), L = lane_len, Pl = lane_num, [x, y, x'-x, y'-y, x_l-x, y_l-x, x_r-x, y_r-y, 8:12(type)]
        # - lanes_speed_limit: (B, Pl, 1)
        # - lanes_has_speed_limit: (B, Pl, 1)

        neighbors = inputs['neighbour_agents_past']
        static = inputs['static_objects']
        lanes = inputs['lanes']
        # lanes_speed_limit = inputs['lanes_speed_limit']
        # lanes_has_speed_limit = inputs['lanes_has_speed_limit']

        B = neighbors.shape[0]

        encoding_neighbors, neighbors_mask, neighbors_pos = self.neighbor_encoder(neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes)

        encoding_input = torch.cat([encoding_neighbors, encoding_static, encoding_lanes], dim = 1)

        encoding_pos = torch.cat([neighbors_pos, static_pos, lane_pos], dim=1).view(B * self.token_num, -1)
        encoding_mask = torch.cat([neighbors_mask, static_mask, lanes_mask], dim=1).view(-1)
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros((B * self.token_num, self.hidden_dim), device=encoding_pos.device)
        encoding_pos_result[~encoding_mask] = encoding_pos # ??

        encoding_input = encoding_input + encoding_pos_result.view(B, self.token_num, -1)
        encoder_outputs = self.fusion(encoding_input, encoding_mask.view(B, self.token_num))

        return {"encoding": encoder_outputs}

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=dropout,
        )

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0]) # why only normalize q?
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop_path_rate):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate
        )
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(
            in_features=tokens_mlp_dim,
            hidden_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        y = self.norm1(x)
        y = y.permute(0, 2, 1)
        y = self.tokens_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.norm2(x)
        return x + self.channels_mlp(y)

class AgentFusionEncoder(nn.Module):
    def __init__(
            self,
            time_len,
            drop_path_rate=0.3,
            hidden_dim=192,
            depth=3,
            tokens_mlp_dim=64,
            channels_mlp_dim=128,
    ):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Linear(3, channels_mlp_dim)

        self.channel_pre_project = Mlp(
            in_features = 8 + 1,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.token_pre_project = Mlp(
            in_features=time_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0
        )

        self.blocks = nn.ModuleList(
            [MixerBlock(tokens_mlp_dim,
                        channels_mlp_dim,
                        drop_path_rate)
                        for i in range(depth)]
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: B, P, V, D (x, y, cos, sin, vx, vy, w, l, type(3))
        """

        neighbor_type = x[:, :, -1, 8:]
        x = x[..., :8]

        pos = x[:, :, -1, :7].clone()
        pos[..., -3:] = 0.0
        pos[..., -3] = 1.0

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        neighbor_type = neighbor_type.view(B * P, -1)
        neighbor_type = neighbor_type[valid_indices]
        type_embedding = self.type_emb(neighbor_type)
        x = x + type_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x
        
        return x_result.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)

class StaticFusionEncoder(nn.Module):
    def __init__(self, dim, drop_path_rate=0.3, hidden_dim=192):
        super().__init__()

        self._hidden_dim = hidden_dim

        self.projection = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: B, P, D (x, y, cos, sin, w, l, type(4))
        """       

        B, P, _ = x.shape

        pos = x[:, :, :7].clone()
        pos[..., -3:] = 0.0
        pos[..., -2] = 1.0

        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)

        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0

        valid_indices = ~mask_p.view(-1)

        if valid_indices.sum() > 0:
            x = x.view(B * P, -1)
            x = x[valid_indices]
            x = self.projection(x)
            x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)
    
class LaneFusionEncoder(nn.Module):
    def __init__(
            self,
            lane_len,
            drop_path_rate=0.3,
            hidden_dim=192,
            depth=3,
            tokens_mlp_dim=64,
            channels_mlp_dim=128,
    ):
        super().__init__()

        self._lane_len = lane_len
        self._channel = channels_mlp_dim

        self.traffic_emb = nn.Linear(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(
            in_features=8,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=lane_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [MixerBlock(
                tokens_mlp_dim,
                channels_mlp_dim,
                drop_path_rate
            )
            for i in range(depth)]
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """
        x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4))
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        """

        traffic = x[:, :, 0, 8:]      
        x = x[..., :8]

        pos = x[:, :, int(self._lane_len / 2), :7].clone()
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        pos[..., -3:] = 0.0
        pos[..., -1] = 1.0

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)
        traffic = traffic.view(B * P, -1)
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(traffic)

        x = x + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x
        
        return x_result.view(B, P, -1), mask_p.reshape(B, -1), pos.view(B, P, -1)
    
class FusionEncoder(nn.Module):
    def __init__(self,
                 hidden_dim=192,
                 num_heads=6,
                 drop_path_rate=0.3,
                 depth=3):
        super().__init__()

        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(
                hidden_dim,
                num_heads,
                dropout=drop_path_rate
            )
            for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        mask[:, 0] = False

        for b in self.blocks:
            x = b(x, mask)

        return self.norm(x)


cfg = {
    'hidden_dim': 192,
    'agent_num': 32,
    'static_objects_state_dim': 10,
    'static_objects_num': 5,
    'lane_num': 70,
    'lane_len': 20,
    'time_len': 21,
    'encoder_drop_path_rate': 0.1,
    'encoder_depth': 3,
    'num_heads': 6,
}

inputs = {
    'neighbour_agents_past': torch.randn(8, cfg['agent_num'], cfg['time_len'], 11),
    'static_objects': torch.randn(8, cfg['static_objects_num'], 10),
    'lanes': torch.randn(8, cfg['lane_num'], cfg['lane_len'], 12),
}

encoder = Encoder(cfg)
op = encoder(inputs)
print(op['encoding'].shape)
