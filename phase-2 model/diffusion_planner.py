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
        self.encoder = DiffusionPlannerEncoder(cfg)
        self.decoder = DiffusionPlannerDecoder(cfg)

    @property
    def sde(self):
        return self.decoder.decoder.sde

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs
    
class DiffusionPlannerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder =Encoder(cfg)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        
        self.apply(_basic_init)
        
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.neighbor_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)

    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)

        return encoder_outputs
    
class DiffusionPlannerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.decoder = Decoder(cfg)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)

        for block in self.decoder.dit.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)

    def forward(self, encoder_outputs, inputs):

        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return decoder_outputs
        
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

        neighbors = inputs['neighbor_agents_past']
        static = inputs['static_objects']
        lanes = inputs['lanes']
        lanes_speed_limit = inputs['lanes_speed_limit']
        lanes_has_speed_limit = inputs['lanes_has_speed_limit']

        B = neighbors.shape[0]

        encoding_neighbors, neighbors_mask, neighbors_pos = self.neighbor_encoder(neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes, lanes_speed_limit, lanes_has_speed_limit)

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

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
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

    def forward(self, x, speed_limit, has_speed_limit):
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

        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.view(B * P, -1)

        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        speed_limit_embedding = torch.zeros((speed_limit.shape[0], self._channel), device=x.device)

        if has_speed_limit.sum() > 0:
            speed_limit_with_limit = self.speed_limit_emb(speed_limit[has_speed_limit].unsqueeze(-1))
            speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        if (~has_speed_limit.sum()) > 0:
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                (~has_speed_limit).sum().item(), -1
            )
            speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit

        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(traffic)

        x = x + speed_limit_embedding + traffic_light_embedding
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

## Decoder

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dpr = cfg['decoder_drop_path_rate']
        self._predicted_neihbor_num = cfg['predicted_neighbor_num']
        self._future_len = cfg['future_len']
        self._sde = VPSDE_linear()
        
        self.dit = DiT(
            sde=self._sde,
            route_encoder=RouteEncoder(
                cfg['route_num'],
                cfg['lane_len'],
                drop_path_rate=cfg['encoder_drop_path_rate'],
                hidden_dim=cfg['hidden_dim'],
            ),
            depth=cfg['decoder_depth'],
            output_dim=(cfg['future_len'] + 1) * 4,
            hidden_dim=cfg['hidden_dim'],
            heads=cfg['num_heads'],
            dropout=dpr,
            model_type=cfg['diffusion_model_type'],
        )
        
        self._state_normalizer = StateNormalizer(**cfg['state_normalizer'])
        self._observation_normalizer = ObservationNormalizer(cfg['observation_normalizer'])

        self._guidance_fn = None

    @property
    def sde(self):
        return self._sde
    
    def forward(self, encoder_outputs, inputs):
        """
        Diffusion decoder process.

        Args:
            encoder_outputs: Dict
                {
                    ...
                    "encoding": agents, static objects and lanes context encoding
                    ...
                }
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,
                    "neighbor_agent_past": past and current neighbor states,

                    [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + V_future, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, V_future, 4]
                    ...
                }

        """   
        ego_current = inputs['ego_current_state'][:, None, :4]
        neighbors_current = inputs['neighbor_agents_past'][:, :self._predicted_neihbor_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs['neighbor_current_mask'] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1)

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neihbor_num)

        ego_neighbor_encoding = encoder_outputs['encoding']
        route_lanes = inputs['route_lanes']

        if self.training:
            sampled_trajectories = inputs['sampled_trajectories'].reshape(
                B, P, -1
            )
            diffusion_time = inputs['diffusion_time']

            return {
                'score': self.dit(
                    sampled_trajectories,
                    diffusion_time,
                    ego_neighbor_encoding,
                    route_lanes,
                    neighbor_current_mask,
                ).reshape(B, P, -1, 4)
            }
        else:
            xT = torch.cat(
                [
                    current_states[:, :, None],
                    torch.randn(B, P, self._future_len, 4).to(current_states.device) * 0.5,
                ],
                dim=2,
            ).reshape(B, P, -1)

            def initial_state_constraint(xt, t, step):
                xt = xt.reshape(B, P, -1, 4)
                xt[:, :, 0, :] = current_states
                return xt.reshape(B, P, -1)
            
            x0 = dpm_sampler(
                self.dit,
                xT,
                other_model_params={
                    'cross_c': ego_neighbor_encoding,
                    'route_lanes': route_lanes,
                    'neighbor_current_mask': neighbor_current_mask,
                },
                dpm_solver_params={
                    'correcting_xt_fn': initial_state_constraint,
                },
                model_wraper_params={
                    'classifier_fn': self._guidance_fn,
                    'classifier_kwargs': {
                        'model': self.dit,
                        'model_condition': {
                            'cross_c': ego_neighbor_encoding,
                            'route_lanes': route_lanes,
                            'neighbor_current_mask': neighbor_current_mask,
                        },
                        'inputs': inputs,
                        'observeation_normalizer': self._observation_normalizer,
                        'state_normalizer': self._state_normalizer,
                    },
                    'guidance_scale': 0.5,
                    'guidance_type': 'classifier' if self._guidance_fn is not None else 'uncond',
                },
            )
            x0 = self._state_normalizer.inverse(x0.reshape(B, P, -1, 4))[:, :, 1:]

            return {'prediction': x0}

class DiT(nn.Module):
    def __init__(
            self,
            sde: 'SDE',
            route_encoder: nn.Module,
            depth,
            output_dim,
            hidden_dim=192,
            heads=6,
            dropout=0.1,
            mlp_ratio=4.0,
            model_type='x_start',
    ):
        super().__init__()

        assert model_type in ['score', 'x_start'], f'Unkonwn model type: {model_type}'

        self._model_type = model_type
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(
            in_features=output_dim,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std

    @property
    def model_type(self):
        return self._model_type
    
    def forward(self, x, t, cross_c, route_lanes, neighbor_current_mask):
        """
        Forward pass of DiT.
        x: (B, P, output_dim)   -> Embedded out of DiT
        t: (B,)
        cross_c: (B, N, D)      -> Cross-Attention context
        """
        B, P, _ = x.shape
        x = self.preproj(x)

        x_embedding = torch.cat(
            [
                self.agent_embedding.weight[0][None, :],
                self.agent_embedding.weight[1][None, :].expand(P - 1, -1),
            ],
            dim=0,
        ) # (P, D)
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1) # (B, P, D)
        x = x + x_embedding

        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding
        y = y + self.t_embedder(t)

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask
        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)

        x = self.final_layer(x, y)
        if self._model_type == 'score':
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == 'x_start':
            return x
        else:
            raise ValueError(f'Unknown model type: {self._mdoel_type}')

from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

def dpm_sampler(
        model: nn.Module,
        x_T,
        other_model_params: dict = {},
        diffusion_steps=10,
        noise_schedule_params: dict = {},
        model_wraper_params: dict = {},
        dpm_solver_params: dict = {},
        sample_params: dict = {},
):
    with torch.no_grad():
        noise_schedule = NoiseScheduleVP(schedule='linear', **noise_schedule_params)

        model_fn = model_wrapper(
            model,
            noise_schedule=noise_schedule,
            model_type=model.model_type,
            model_kwargs=other_model_params,
            **model_wraper_params,
        )

        dpm_solver = DPM_Solver(
            model_fn, noise_schedule, algorithm_type='dpmsolver++', **dpm_solver_params
        )

        sample_dpm = dpm_solver.sample(
            x_T,
            steps=diffusion_steps,
            order=2,
            skip_type='logSNR',
            method='multistep',
            denoise_to_zero=True,
            **sample_params,
        )

    return sample_dpm

class RouteEncoder(nn.Module):
    def __init__(
            self,
            route_num,
            lane_len,
            drop_path_rate=0.3,
            hidden_dim=192,
            tokens_mlp_dim=32,
            channels_mlp_dim=64,
    ):
        super().__init__()

        self.channel = channels_mlp_dim
        self.channel_pre_project = Mlp(
            in_features=4,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=route_num * lane_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

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
        x: B, P, V, D
        """
        x = x[..., :4]
        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0

        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)

        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x
        return x_result.view(B, -1)

import json

class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
    
    @classmethod
    def from_json(cls, args):
        data = json.load(args['normalization_file_path'])
        mean = [[data['ego']['mean']]] + [[data['neighbor']['mean']]] * args['predicted_neighbor_num']
        std = [[data['ego']['std']]] + [[data['neighbor']['std']]] * args['predicted_neigbhor_num']
        return cls(mean, std)
    
    def __call__(self, data):
        return (data - self.mean.to(data.device)) / self.std.to(data.device)
    
    def inverse(self, data):
        return data * self.std.to(data.device) + self.mean.to(data.device)
    
    def to_dict(self):
        return {
            'mean': self.mean.detach().cpu().numpy().tolist(),
            'std': self.std.detach().cpu().numpy().tolist(),
        }

from copy import copy

class ObservationNormalizer:
    def __init__(self, normalization_dict):
        self._normalization_dict = normalization_dict

    @classmethod
    def from_json(cls, args):
        if isinstance(args, str):
            path = args
        else:
            path = args['normalization_file_path']

        data = json.load(path)
        ndt = {}
        for k, v in data.items():
            if k not in ['ego', 'neighbor']:
                ndt[k] = {
                    'mean': torch.tensor(v['mean'], dtype=torch.float32),
                    'std': torch.tensor(v['std'], dtype=torch.float32),
                }
        return cls(ndt)
    
    def __call__(self, data):
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = (data[k] - v['mean']).to(data[k].device) / v['std'].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data
    
    def inverse(self, data):
        norm_data = copy(data)
        for k, v in self._normalization_dict.items():
            if k not in data:
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = data[k] * v['std'].to(data[k].device) + v['mean'].to(data[k].device)
            norm_data[k][mask] = 0
        return norm_data
    
    def to_dict(self):
        return {
            k: {kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()}
            for k, v in self._normalization_dict.items()
        }

def scale(x, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))
    return x

def modulate(x, shift, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return x

class DiTBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, cross_c, y, attn_mask):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, ) = self.adaLN_modulation(
            y
        ).chunk(6, dim=1)

        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulated_x, modulated_x, modulated_x, key_padding_mask=attn_mask)[0]

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_msa.unsqueeze(1) * self.mlp1(modulated_x)

        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        x = self.mlp2(self.norm4(x))

        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate='tanh'),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True),
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        B, P, _ = x.shape

        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequncy_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequncy_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb        

## SDE

from abc import ABC, abstractmethod
class SDE(ABC):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def T(self):
        pass

    @abstractmethod
    def sde(self, x, t):
        pass

    @abstractmethod
    def marginal_prob(self, x, t):
        pass

    @abstractmethod
    def diffusion_coeff(self, t):
        pass

    @abstractmethod
    def marginal_prob_std(self, t):
        pass

class VPSDE_linear(SDE):
    def __init__(self, beta_max=20.0, beta_min=0.1):
        """
        VP SDE

        SDE:
        $ \mathrm{d}x = -\frac{\beta(t)}{2} x \mathrm{d}t + \sqrt{\beta(t)} \mathrm{d}W_t $
        """
        super().__init__()
        self._beta_max = beta_max
        self._beta_min = beta_min     

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        """
        SDE of diffusion process

        drift = $-\frac{\beta(t)}{2} x$
        diffusion = $\sqrt{\beta(t)}$
        """

        shape = x.shape
        reshape = [-1] + [1] * (len(shape) - 1)
        t = t.reshape(reshape)

        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)

        return drift, diffusion

    def marginal_prob(self, x, t):   
        """
        Parameters to determine the marginal distribution of the SDE, $p_t(x)$.
        """
        shape = x.shape
        reshape = [-1] + [1] * (len(shape) - 1)
        t = t.reshape(reshape)
        mean_log_coeff = -0.25 * t**2 * (self._beta_max - self._beta_min) - 0.5 * self._beta_min * t
        mean = torch.exp(mean_log_coeff) * x
        std = torch.sqrt(1 - torch.exp(2.0 * mean_log_coeff))
        return mean, std

    def diffusion_coeff(self, t):
        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        diffusion = torch.sqrt(beta_t)
        return diffusion

    def marginal_prob_std(self, t):
        discount = torch.exp(-0.5 * t**2 * (self._beta_max - self._beta_min) - self._beta_min * t)
        std = torch.sqrt(1 - discount)
        return std                
    

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
    
    'decoder_drop_path_rate': 0.1,
    'predicted_neighbor_num': 10,
    'route_num': 25,
    'decoder_depth': 3,
    'future_len': 80,
    'diffusion_model_type': 'x_start',
    "state_normalizer": {
    "mean": [
      [
        [
          10,
          0,
          0,
          0
        ]
      ],
      [
        [
          10,
          0,
          0,
          0
        ]
      ],
      [
        [
          10,
          0,
          0,
          0
        ]
      ],
      [
        [
          10,
          0,
          0,
          0
        ]
      ],
      [
        [
          10,
          0,
          0,
          0
        ]
      ],
      [
        [
          10,
          0,
          0,
          0
        ]
      ],
      [
        [
          10,
          0,
          0,
          0
        ]
      ],
      [
        [
          10,
          0,
          0,
          0
        ]
      ],
      [
        [
          10,
          0,
          0,
          0
        ]
      ],
      [
        [
          10,
          0,
          0,
          0
        ]
      ],
      [
        [
          10,
          0,
          0,
          0
        ]
      ]
    ],
    "std": [
      [
        [
          20,
          20,
          1,
          1
        ]
      ],
      [
        [
          20,
          20,
          1,
          1
        ]
      ],
      [
        [
          20,
          20,
          1,
          1
        ]
      ],
      [
        [
          20,
          20,
          1,
          1
        ]
      ],
      [
        [
          20,
          20,
          1,
          1
        ]
      ],
      [
        [
          20,
          20,
          1,
          1
        ]
      ],
      [
        [
          20,
          20,
          1,
          1
        ]
      ],
      [
        [
          20,
          20,
          1,
          1
        ]
      ],
      [
        [
          20,
          20,
          1,
          1
        ]
      ],
      [
        [
          20,
          20,
          1,
          1
        ]
      ],
      [
        [
          20,
          20,
          1,
          1
        ]
      ]
    ]
  },
  "observation_normalizer": {
    "ego_current_state": {
      "mean": [
        10.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "std": [
        20.0,
        20.0,
        1.0,
        1.0,
        20.0,
        20.0,
        20.0,
        20.0,
        1.0,
        1.0
      ]
    },
    "neighbor_agents_past": {
      "mean": [
        10.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "std": [
        20.0,
        20.0,
        1.0,
        1.0,
        20.0,
        20.0,
        20.0,
        20.0,
        1.0,
        1.0,
        1.0
      ]
    },
    "static_objects": {
      "mean": [
        10.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "std": [
        20.0,
        20.0,
        1.0,
        1.0,
        20.0,
        20.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    },
    "lanes": {
      "mean": [
        10.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "std": [
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    },
    "lanes_speed_limit": {
      "mean": [
        0.0
      ],
      "std": [
        20.0
      ]
    },
    "route_lanes": {
      "mean": [
        10.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "std": [
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    },
    "route_lanes_speed_limit": {
      "mean": [
        0.0
      ],
      "std": [
        20.0
      ]
    }
  }
}

if __name__ == '__main__':
    inputs = {
        'neighbor_agents_past': torch.randn(8, cfg['agent_num'], cfg['time_len'], 11),
        'static_objects': torch.randn(8, cfg['static_objects_num'], 10),
        'lanes': torch.randn(8, cfg['lane_num'], cfg['lane_len'], 12),
        'ego_current_state':torch.randn(8, 10),
        'route_lanes': torch.randn(8, cfg['route_num'], cfg['lane_len'], 12),
        'sampled_trajectories': torch.randn(8, (1 + cfg['predicted_neighbor_num']), 1 + cfg['future_len'], 4),
        'diffusion_time': torch.randn(8),
    }

    model = DiffusionPlanner(cfg)

    op = model(inputs)
    print(op)
