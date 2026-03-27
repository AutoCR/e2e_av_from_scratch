import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
import numpy as np

class GPT(nn.Module):
    def __init__(self, n_embd, cfg, lidar_time_frames):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = 1
        self.lidar_seq_len = cfg['lidar_seq_len']
        self.cfg = cfg
        self.lidar_time_frames = lidar_time_frames

        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len * cfg['img_vert_anchors'] * cfg['img_horz_anchors'] +
                lidar_time_frames * cfg['lidar_vert_anchors'] * cfg['lidar_horz_anchors'],
                self.n_embd
            )
        )

        self.drop = nn.Dropout(cfg['embd_pdrop'])

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    cfg['n_head'],
                    cfg['block_exp'],
                    cfg['attn_pdrop'],
                    cfg['resid_pdrop'],
                )
                for layer in range(cfg['n_layer'])
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean = self.cfg['gpt_linear_layer_init_mean'],
                std = self.cfg['gpt_linear_layer_init_std'],
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.cfg['gpt_layer_norm_init_weight'])
    
    def forward(self, image_tensor, lidar_tensor):
        ## image_tensor [B * 4 * seq_len, C, H, W]
        ## lidar_tensor [B * seq_len, C, H, W]
        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        assert self.seq_len == 1
        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        lidar_tensor = lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        image_tensor_out = (
            x[
                :,
                : self.seq_len * self.cfg['img_vert_anchors'] * self.cfg['img_horz_anchors'],
                :,
            ]
            .view(bz * self.seq_len, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        lidar_tensor_out = (
            x[
                :,
                self.seq_len * self.cfg['img_vert_anchors'] * self.cfg['img_horz_anchors'] :,
                :,
            ]
            .view(bz, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return image_tensor_out, lidar_tensor_out

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        b, t, c = x.size()

        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TransfuserBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.image_encoder = timm.create_model("resnet34", pretrained=True, features_only=True)
        self.lidar_encoder = timm.create_model("resnet34", pretrained=True, in_chans=cfg['lidar_seq_len'], features_only=True)
        self.avgpool_img = nn.AdaptiveAvgPool2d((cfg['img_vert_anchors'], cfg['img_horz_anchors']))
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((cfg['lidar_vert_anchors'], cfg['lidar_horz_anchors']))
        
        self.global_pool_lidar = nn.AdaptiveAvgPool2d(output_size=1)
        self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)
        start_index = 0
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1
        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    self.lidar_encoder.feature_info.info[start_index + i]['num_chs'],
                    self.image_encoder.feature_info.info[start_index + i]['num_chs'],
                    kernel_size = 1,
                )
                for i in range(4)
            ]
        )

        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    self.image_encoder.feature_info.info[start_index + i]['num_chs'],
                    self.lidar_encoder.feature_info.info[start_index + i]['num_chs'],
                    kernel_size = 1,
                )
                for i in range(4)
            ]
        )

        lidar_time_frames = [1, 1, 1, 1]

        self.transformers = nn.ModuleList(
            [
                GPT(
                    n_embd=self.image_encoder.feature_info.info[start_index + i]['num_chs'],
                    cfg=cfg,
                    lidar_time_frames=lidar_time_frames[i],
                )
                for i in range(4)
            ]
        )

        self.relu = nn.ReLU(inplace=True)
        # top down
        channel = cfg['bev_features_channels']
        self.upsample = nn.Upsample(
            scale_factor=cfg['bev_upsample_factor'],
            mode='bilinear',
            align_corners=False,
        )
        self.upsample2 = nn.Upsample(
            size = (
                cfg['lidar_resolution_height'] // cfg['bev_down_sample_factor'],
                cfg['lidar_resolution_width'] // cfg['bev_down_sample_factor'],
            ),
            mode = 'bilinear',
            align_corners=False,
        )
        self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
        self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)
        self.c5_conv = nn.Conv2d(
            self.lidar_encoder.feature_info.info[start_index + 3]['num_chs'],
            channel,
            (1, 1),
        )

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))
        return p3
    
    def forward(self, image, lidar):
        image_features = image
        lidar_features = lidar
        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)         
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)
        
        for i in range(4):
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
            lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)

            image_features, lidar_features = self.fuse_feature(image_features, lidar_features, i)
        
        x4 = lidar_features
        fused_features = lidar_features
        features = self.top_down(x4)
        return features, fused_features
    
    def forward_layer_block(self, layers, return_layers, features):
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features
    
    def fuse_feature(self, image_features, lidar_features, layer_idx):
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)
        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        image_features_layer, lidar_features_layer = self.transformers[layer_idx](image_embd_layer, lidar_embd_layer)
        lidar_features_layer = self.img_channel_to_lidar[layer_idx](lidar_features_layer)

        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode='bilinear',
            align_corners=False,
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        image_features = image_features + image_features_layer
        lidar_features = lidar_features + lidar_features_layer
        return image_features, lidar_features
    
class TransfuserModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._backbone = TransfuserBackbone(cfg['backbone'])
        self._bev_downscale = nn.Conv2d(512, cfg['tf_d_model'], kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, cfg['tf_d_model'])

        self._query_splits = [
            1,
            cfg['num_bounding_boxes'],
        ]
        self._keyval_embedding = nn.Embedding(8**2 + 1, cfg['tf_d_model'])
        self._query_embedding = nn.Embedding(sum(self._query_splits), cfg['tf_d_model'])

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                cfg['backbone']['bev_features_channels'],
                cfg['backbone']['bev_features_channels'],
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                cfg['backbone']['bev_features_channels'],
                cfg['num_bev_classes'],
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    cfg['backbone']['lidar_resolution_height'] // 2,
                    cfg['backbone']['lidar_resolution_width'],
                ),
                mode='bilinear',
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg['tf_d_model'],
            nhead=cfg['tf_num_head'],
            dim_feedforward=cfg['tf_d_ffn'],
            dropout=cfg['tf_dropout'],
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, cfg['tf_num_layers'])
        self._agent_head = AgentHead(
            num_agents=cfg['num_bounding_boxes'],
            d_ffn=cfg['tf_d_ffn'],
            d_model=cfg['tf_d_model'],
        )
        self._trajectory_head = TrajectoryHead(
            num_poses=cfg['trajectory_num_poses'],
            d_ffn=cfg['tf_d_ffn'],
            d_model=cfg['tf_d_model'],
        )

    def forward(self, features):
        camera_features = features['camera_feature']
        lidar_features = features['lidar_feature']
        status_features = features['status_feature']
        batch_size = status_features.shape[0]
        bev_feature_upscale, bev_feature = self._backbone(camera_features, lidar_features)

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_features)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        # - 0: background
        # - 1: road / intersection
        # - 2: walkways
        # - 3: lane centerline
        # - 4: static objects
        # - 5: vehicles
        # - 6: pedestrians
        output = {'bev_semantic_map': bev_semantic_map}
        trajectory = self._trajectory_head(trajectory_query)
        output.update(trajectory)
        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

class AgentHead(nn.Module):

    def __init__(
            self,
            num_agents: int,
            d_ffn: int,
            d_model: int,
    ):
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries):
        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {'agent_states': agent_states, 'agent_labels': agent_labels}
    
class TrajectoryHead(nn.Module):
    
    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, num_poses * StateSE2Index.size()),
        )

    def forward(self, object_queries):
        poses = self._mlp(object_queries)
        poses = poses.reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return {'trajectory': poses}
    
TRANSFUSER_BACKBONE_CONFIG = {
    "camera_width": 1024,
    "camera_height": 256,

    "lidar_seq_len": 1,
    "img_vert_anchors": 256 // 32,
    "img_horz_anchors": 1024 // 32,
    "lidar_vert_anchors": 256 // 32,
    "lidar_horz_anchors": 256 // 32,
    "bev_features_channels": 64,
    "bev_upsample_factor": 2,
    "bev_down_sample_factor": 4,
    "lidar_resolution_width": 256,
    "lidar_resolution_height": 256,

    "lidar_seq_len": 1,
    "embd_pdrop": 0.1,
    "n_head": 4,
    "block_exp": 4,
    "attn_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "n_layer": 2,
    "gpt_linear_layer_init_mean": 0.0,
    "gpt_linear_layer_init_std": 0.02,
    "gpt_layer_norm_init_weight": 1.0,
}

TRANSFUSER_CONFIG = {
    "backbone": TRANSFUSER_BACKBONE_CONFIG,
    "tf_d_model": 256,
    "tf_num_head": 8,
    "tf_d_ffn": 1024,
    "tf_num_layers": 3,
    "tf_dropout": 0.0,
    "num_bev_classes": 7,
    "num_bounding_boxes": 30,
    "trajectory_num_poses": 8,
}