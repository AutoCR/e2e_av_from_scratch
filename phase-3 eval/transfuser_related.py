import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import logging
from dataclasses import fields
from typing import Dict, List, Optional, Tuple, Union

from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import PDMResults, SensorConfig, Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents

## model
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
        self.image_encoder = timm.create_model("resnet34", pretrained=False, features_only=True)
        self.lidar_encoder = timm.create_model(
            "resnet34",
            pretrained=False,
            in_chans=cfg['lidar_seq_len'],
            features_only=True,
        )
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

## dataset

import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import LineString, Polygon

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.enums import BoundingBoxIndex, SceneFrameType
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scene_loader: SceneLoader,
        max_len=None,
        random_sample=False,
        trajectory_num_poses=None,
    ):
        self.scene_loader = scene_loader
        self.trajectory_num_poses = trajectory_num_poses

        if random_sample:
            if max_len is None:
                max_len = len(scene_loader.tokens)
            max_len = min(max_len, len(scene_loader.tokens))
            self.tokens = np.random.choice(scene_loader.tokens, size=max_len, replace=False).tolist()
        else:
            max_len = min(max_len, len(scene_loader.tokens)) if max_len is not None else len(scene_loader.tokens)
            self.tokens = scene_loader.tokens[:max_len]

        self.max_len = len(self.tokens)
        self.current_index = self.scene_loader._scene_filter.num_history_frames - 1

        self.camera_size = (1024, 256)
        self.lidar_range = (-32.0, 32.0, -32.0, 32.0)
        self.pixels_per_meter = 4.0
        self.hist_max_per_pixel = 5
        self.max_height_lidar = 100.0
        self.lidar_split_height = 0.2
        self.num_bounding_boxes = 30

        self.bev_pixel_width = 256
        self.bev_pixel_height = 128
        self.bev_pixel_size = 0.25
        self.num_bev_classes = 7
        self.bev_radius = 32.0

        self.tracked_object_types = {
            "vehicle": TrackedObjectType.VEHICLE,
            "pedestrian": TrackedObjectType.PEDESTRIAN,
            "bicycle": TrackedObjectType.BICYCLE,
            "traffic_cone": TrackedObjectType.TRAFFIC_CONE,
            "barrier": TrackedObjectType.BARRIER,
            "czone_sign": TrackedObjectType.CZONE_SIGN,
            "generic_object": TrackedObjectType.GENERIC_OBJECT,
            "ego": TrackedObjectType.EGO,
        }
        self.bev_semantic_classes = {
            1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),
            2: ("polygon", [SemanticMapLayer.WALKWAYS]),
            3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),
            4: (
                "box",
                [
                    TrackedObjectType.CZONE_SIGN,
                    TrackedObjectType.BARRIER,
                    TrackedObjectType.TRAFFIC_CONE,
                    TrackedObjectType.GENERIC_OBJECT,
                ],
            ),
            5: ("box", [TrackedObjectType.VEHICLE]),
            6: ("box", [TrackedObjectType.PEDESTRIAN]),
        }

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        token = self.tokens[idx]
        frame_list = self.scene_loader.scene_frames_dicts[token]
        current_frame = frame_list[self.current_index]

        features = {
            "camera_feature": self._build_camera_feature(current_frame),
            "lidar_feature": self._build_lidar_feature(current_frame),
            "status_feature": self._build_status_feature(current_frame),
        }
        targets = {
            "trajectory": self._build_future_trajectory(frame_list),
            "agent_states": self._build_agent_states(current_frame),
            "agent_labels": self._build_agent_labels(current_frame),
            "bev_semantic_map": self._build_bev_semantic_map(current_frame),
        }
        return token, features, targets

    def _sensor_root(self) -> Path:
        return Path(self.scene_loader._original_sensor_path)

    def _maps_root(self) -> str:
        maps_root = os.getenv("NUPLAN_MAPS_ROOT")
        if maps_root is None:
            raise RuntimeError("NUPLAN_MAPS_ROOT is not set.")
        return maps_root

    def _map_api(self, map_name: str):
        return get_maps_api(self._maps_root(), "nuplan-maps-v1.0", map_name)

    def _load_image(self, relative_path: str) -> np.ndarray:
        image_path = self._sensor_root() / relative_path
        return np.array(Image.open(image_path))

    def _build_camera_feature(self, frame_dict) -> torch.Tensor:
        cameras = frame_dict["cams"]

        left = self._load_image(cameras["CAM_L0"]["data_path"])[28:-28, 416:-416]
        front = self._load_image(cameras["CAM_F0"]["data_path"])[28:-28]
        right = self._load_image(cameras["CAM_R0"]["data_path"])[28:-28, 416:-416]

        stitched = np.concatenate([left, front, right], axis=1)
        resized = cv2.resize(stitched, self.camera_size)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).to(torch.float32) / 255.0
        return tensor

    def _load_lidar_points(self, relative_path: str) -> np.ndarray:
        lidar_path = self._sensor_root() / relative_path
        with open(lidar_path, "rb") as fp:
            lidar_pc = LidarPointCloud.from_buffer(BytesIO(fp.read()), "pcd").points
        return lidar_pc[:3].T

    def _splat_points(self, point_cloud: np.ndarray) -> np.ndarray:
        min_x, max_x, min_y, max_y = self.lidar_range
        xbins = np.linspace(min_x, max_x, int((max_x - min_x) * self.pixels_per_meter) + 1)
        ybins = np.linspace(min_y, max_y, int((max_y - min_y) * self.pixels_per_meter) + 1)
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist = np.clip(hist, 0, self.hist_max_per_pixel)
        return hist / self.hist_max_per_pixel

    def _build_lidar_feature(self, frame_dict) -> torch.Tensor:
        lidar_pc = self._load_lidar_points(frame_dict["lidar_path"])
        lidar_pc = lidar_pc[lidar_pc[:, 2] < self.max_height_lidar]
        above = lidar_pc[lidar_pc[:, 2] > self.lidar_split_height]
        overhead = self._splat_points(above)
        feature = overhead[None].astype(np.float32)
        return torch.from_numpy(feature)

    def _build_status_feature(self, frame_dict) -> torch.Tensor:
        driving_command = np.asarray(frame_dict["driving_command"], dtype=np.float32)
        ego_dynamic_state = np.asarray(frame_dict["ego_dynamic_state"], dtype=np.float32)
        ego_velocity = ego_dynamic_state[:2]
        ego_acceleration = ego_dynamic_state[2:]
        status = np.concatenate([driving_command, ego_velocity, ego_acceleration], axis=0)
        return torch.from_numpy(status)

    def _frame_pose(self, frame_dict) -> np.ndarray:
        translation = np.asarray(frame_dict["ego2global_translation"][:2], dtype=np.float64)
        yaw = Quaternion(*frame_dict["ego2global_rotation"]).yaw_pitch_roll[0]
        return np.array([translation[0], translation[1], yaw], dtype=np.float64)

    def _to_relative_poses(self, origin_pose: np.ndarray, global_poses: np.ndarray) -> np.ndarray:
        theta = -origin_pose[2]
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.float64,
        )
        relative = global_poses - origin_pose[None]
        relative[:, :2] = relative[:, :2] @ rotation.T
        relative[:, 2] = np.arctan2(np.sin(relative[:, 2]), np.cos(relative[:, 2]))
        return relative

    def _build_future_trajectory(self, frame_list) -> torch.Tensor:
        current_pose = self._frame_pose(frame_list[self.current_index])
        future_frames = frame_list[self.current_index + 1 :]

        if self.trajectory_num_poses is not None:
            future_frames = future_frames[: self.trajectory_num_poses]

        future_global_poses = np.stack([self._frame_pose(frame_dict) for frame_dict in future_frames], axis=0)
        future_relative_poses = self._to_relative_poses(current_pose, future_global_poses).astype(np.float32)
        return torch.from_numpy(future_relative_poses)

    def _xy_in_lidar(self, x: float, y: float) -> bool:
        min_x, max_x, min_y, max_y = self.lidar_range
        return min_x <= x <= max_x and min_y <= y <= max_y

    def _build_agent_targets(self, frame_dict):
        boxes = np.asarray(frame_dict["anns"]["gt_boxes"], dtype=np.float32)
        names = np.asarray(frame_dict["anns"]["gt_names"])

        selected = []
        for box, name in zip(boxes, names):
            x = float(box[BoundingBoxIndex.X])
            y = float(box[BoundingBoxIndex.Y])
            if name == "vehicle" and self._xy_in_lidar(x, y):
                selected.append(
                    np.array(
                        [
                            box[BoundingBoxIndex.X],
                            box[BoundingBoxIndex.Y],
                            box[BoundingBoxIndex.HEADING],
                            box[BoundingBoxIndex.LENGTH],
                            box[BoundingBoxIndex.WIDTH],
                        ],
                        dtype=np.float32,
                    )
                )

        if selected:
            selected = np.stack(selected, axis=0)
            distances = np.linalg.norm(selected[:, :2], axis=-1)
            selected = selected[np.argsort(distances)[: self.num_bounding_boxes]]
        else:
            selected = np.zeros((0, 5), dtype=np.float32)

        agent_states = np.zeros((self.num_bounding_boxes, 5), dtype=np.float32)
        agent_labels = np.zeros((self.num_bounding_boxes,), dtype=bool)

        num_valid = len(selected)
        if num_valid > 0:
            agent_states[:num_valid] = selected
            agent_labels[:num_valid] = True

        return torch.from_numpy(agent_states), torch.from_numpy(agent_labels)

    def _build_agent_states(self, frame_dict) -> torch.Tensor:
        agent_states, _ = self._build_agent_targets(frame_dict)
        return agent_states

    def _build_agent_labels(self, frame_dict) -> torch.Tensor:
        _, agent_labels = self._build_agent_targets(frame_dict)
        return agent_labels

    def _geometry_local_coords(self, geometry, origin: StateSE2):
        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        return affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

    def _coords_to_pixel(self, coords: np.ndarray) -> np.ndarray:
        pixel_center = np.array([[0.0, self.bev_pixel_width / 2.0]])
        coords_idx = (coords / self.bev_pixel_size) + pixel_center
        return coords_idx.astype(np.int32)

    def _compute_map_polygon_mask(self, map_api, ego_pose: StateSE2, layers) -> np.ndarray:
        map_object_dict = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self.bev_radius, layers=layers)
        mask = np.zeros((self.bev_pixel_width, self.bev_pixel_height), dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(mask, [exterior], color=255)
        return np.rot90(mask)[::-1] > 0

    def _compute_map_linestring_mask(self, map_api, ego_pose: StateSE2, layers) -> np.ndarray:
        map_object_dict = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self.bev_radius, layers=layers)
        mask = np.zeros((self.bev_pixel_width, self.bev_pixel_height), dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, ego_pose)
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(mask, [points], isClosed=False, color=255, thickness=2)
        return np.rot90(mask)[::-1] > 0

    def _compute_box_mask(self, frame_dict, tracked_types) -> np.ndarray:
        mask = np.zeros((self.bev_pixel_width, self.bev_pixel_height), dtype=np.uint8)
        boxes = np.asarray(frame_dict["anns"]["gt_boxes"], dtype=np.float32)
        names = np.asarray(frame_dict["anns"]["gt_names"])

        for name, box in zip(names, boxes):
            tracked_type = self.tracked_object_types.get(str(name))
            if tracked_type not in tracked_types:
                continue

            x = float(box[BoundingBoxIndex.X])
            y = float(box[BoundingBoxIndex.Y])
            heading = float(box[BoundingBoxIndex.HEADING])
            length = float(box[BoundingBoxIndex.LENGTH])
            width = float(box[BoundingBoxIndex.WIDTH])
            height = float(box[BoundingBoxIndex.HEIGHT])

            oriented_box = OrientedBox(StateSE2(x, y, heading), length, width, height)
            exterior = np.array(oriented_box.geometry.exterior.coords).reshape((-1, 1, 2))
            exterior = self._coords_to_pixel(exterior)
            cv2.fillPoly(mask, [exterior], color=255)

        return np.rot90(mask)[::-1] > 0

    def _build_bev_semantic_map(self, frame_dict) -> torch.Tensor:
        bev_map = np.zeros((self.bev_pixel_height, self.bev_pixel_width), dtype=np.int64)
        ego_pose_arr = self._frame_pose(frame_dict)
        ego_pose = StateSE2(*ego_pose_arr)
        map_api = self._map_api(frame_dict["map_location"])

        for label, (entity_type, layers) in self.bev_semantic_classes.items():
            if entity_type == "polygon":
                mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                mask = self._compute_box_mask(frame_dict, layers)
            bev_map[mask] = label

        return torch.from_numpy(bev_map)
    

## visualization

def show_result(output, targets, features):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    pred_bev = output['bev_semantic_map'][0].detach().cpu().numpy().argmax(axis=0)
    pred_traj = output['trajectory'][0].detach().cpu().numpy()
    pred_boxes = output['agent_states'][0].detach().cpu().numpy()
    pred_scores = output['agent_labels'][0].detach().cpu().numpy()

    gt_bev = targets['bev_semantic_map'][0].detach().cpu().numpy()
    gt_traj = targets['trajectory'][0].detach().cpu().numpy()
    gt_boxes = targets['agent_states'][0].detach().cpu().numpy()
    gt_scores = targets['agent_labels'][0].detach().cpu().numpy()
    camera_image = features['camera_feature'][0].detach().cpu().permute(1, 2, 0).numpy()
    lidar_feature = features['lidar_feature'][0, 0].detach().cpu().numpy()

    PIXELS_PER_METER = 4.0
    Y_MIN = -32.0
    X_MAX = 32.0
    BEV_WIDTH = pred_bev.shape[1]
    BEV_HEIGHT = pred_bev.shape[0]
    Y_MAX = Y_MIN + BEV_WIDTH / PIXELS_PER_METER
    X_MIN = X_MAX - BEV_HEIGHT / PIXELS_PER_METER

    def ego_to_pixel(x, y):
        px = (y - Y_MIN) * PIXELS_PER_METER
        py = (X_MAX - x) * PIXELS_PER_METER
        return px, py

    def plot_box(ax, box, color, linewidth=1.5):
        corners = box_corners_xy(box)
        ax.plot(corners[:, 1], corners[:, 0], color=color, linewidth=linewidth)

    def box_corners_xy(box):
        x, y, heading, length, width = box
        local = np.array([
            [ length / 2,  width / 2],
            [ length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2,  width / 2],
            [ length / 2,  width / 2],
        ])
        rot = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading),  np.cos(heading)],
        ])
        return local @ rot.T + np.array([x, y])

    def plot_traj(ax, traj, color, label, linestyle='-', marker='o'):
        ax.plot(traj[:, 1], traj[:, 0], color=color, linewidth=2.2, linestyle=linestyle, marker=marker, markersize=4, label=label)

    def draw_panel(ax, bev_map, boxes, scores, traj, title, box_color, traj_color):
        ax.imshow(
            bev_map,
            cmap=ListedColormap(class_colors),
            interpolation='nearest',
            origin='lower',
            extent=[Y_MIN, Y_MAX, X_MIN, X_MAX],
            vmin=0,
            vmax=len(class_colors) - 1,
        )
        valid_mask = scores > 0
        for box in boxes[valid_mask]:
            plot_box(ax, box, color=box_color, linewidth=1.2)
        plot_traj(ax, traj, color=traj_color, label='trajectory')
        ax.scatter(0.0, 0.0, color='red', s=60, marker='*', label='ego')
        ax.set_title(title)
        ax.set_xlim(Y_MIN, Y_MAX)
        ax.set_ylim(X_MIN, X_MAX)
        ax.invert_xaxis()
        ax.set_aspect('equal')
        ax.legend(loc='upper right')

    class_names = ['background', 'road', 'walkway', 'centerline', 'static', 'vehicle', 'pedestrian']
    class_colors = ['#FFFFFF', '#D3D3D3', '#d4d19e', '#666666', '#edc948', '#699CDB', '#b07aa1']
    class_handles = [Patch(facecolor=color, edgecolor='none', label=name) for name, color in zip(class_names, class_colors)]
    overlay_handles = [
        Patch(facecolor='deepskyblue', edgecolor='none', label='pred boxes (BEV)'),
        Patch(facecolor='lime', edgecolor='none', label='gt boxes (BEV)'),
        Patch(facecolor='green', edgecolor='none', label='pred boxes (lidar_feature)'),
        Patch(facecolor='red', edgecolor='none', label='gt boxes (lidar_feature)'),
    ]

    fig = plt.figure(figsize=(16, 24), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 2, 8])
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_cam = fig.add_subplot(gs[1, :])
    ax_lidar = fig.add_subplot(gs[2, :])

    draw_panel(ax_pred, pred_bev, pred_boxes, pred_scores, pred_traj, 'Prediction', 'deepskyblue', 'white')
    draw_panel(ax_gt, gt_bev, gt_boxes, gt_scores, gt_traj, 'Ground Truth', 'lime', 'magenta')

    ax_cam.imshow(camera_image)
    ax_cam.set_title('Stitched Front Camera Image')
    ax_cam.axis('off')

    ax_lidar.imshow(lidar_feature, cmap='gray', origin='lower', extent=[-32, 32, -32, 32])
    for box in gt_boxes[gt_scores > 0]:
        corners = box_corners_xy(box)
        ax_lidar.plot(corners[:, 1], corners[:, 0], color='red', linewidth=1.8)
    for box in pred_boxes[pred_scores > 0]:
        corners = box_corners_xy(box)
        ax_lidar.plot(corners[:, 1], corners[:, 0], color='green', linewidth=1.4)
    ax_lidar.scatter(0.0, 0.0, color='red', s=80, marker='*')
    ax_lidar.set_title('Model Input lidar_feature with Bounding Boxes')
    ax_lidar.set_xlabel('y (m)')
    ax_lidar.set_ylabel('x (m)')
    ax_lidar.set_xlim(-32, 32)
    ax_lidar.set_ylim(-32, 32)
    ax_lidar.invert_xaxis()
    ax_lidar.set_box_aspect(lidar_feature.shape[0] / lidar_feature.shape[1])
    ax_lidar.grid(True, alpha=0.2)
    ax_lidar.set_facecolor('#111111')

    fig.legend(handles=class_handles + overlay_handles, loc='center right', bbox_to_anchor=(1.02, 0.5), frameon=True, title='Semantic Classes')
    plt.show()


def load_transfuser_model():
    import torch
    ckpt_path = Path(__file__).resolve().parents[1] / "model_weights" / "transfuser_seed_0.ckpt"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    state_dict = {
        k.replace("agent._transfuser_model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("agent._transfuser_model.")
    }

    model = TransfuserModel(TRANSFUSER_CONFIG)
    model.load_state_dict(state_dict, strict=True)
    return model


logger = logging.getLogger(__name__)

TRANSFUSER_TRAJECTORY_SAMPLING = TrajectorySampling(num_poses=8, interval_length=0.5)
PDM_PROPOSAL_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)


def compute_final_scores(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    """Inject two-frame comfort into weighted metrics and recompute the final score."""
    df = pdm_score_df.reset_index()

    assert (
        not df["two_frame_extended_comfort"].isna().any()
    ), "Found NaN in 'two_frame_extended_comfort'. Please check aggregator completeness."

    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())

    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT
    weighted_metrics[:, two_frame_idx] = two_frame_scores

    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)
    assert np.all(total_weight > 0), "Found total_weight == 0 during score computation."

    weighted_metric_scores = weighted_sum / total_weight
    df["score"] = df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores
    df.drop(columns=["weighted_metrics", "weighted_metrics_array", "multiplicative_metrics_prod"], inplace=True)
    return df


def calculate_weighted_average_score(df: pd.DataFrame) -> pd.Series:
    """Calculate a weighted average over all metric columns."""
    score_cols = [column for column in df.columns if column not in {"weight", "token"}]

    if df.empty:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weights = df["weight"].to_numpy()
    scores = df[score_cols].to_numpy()
    total_weight = weights.sum()

    if total_weight == 0:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weighted_avg = (scores * weights[:, None]).sum(axis=0) / total_weight
    return pd.Series(weighted_avg, index=score_cols)


def calculate_individual_mapping_scores(
    pdm_score_df: pd.DataFrame,
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Aggregate per-mapping two-stage scores the same way as the PDM script."""
    all_group_scores = []
    stage1_group_scores = []
    stage2_group_scores = []

    for (orig_token, prev_token), second_stage_pairs in all_mappings.items():
        first_tokens = [pair[0] for pair in second_stage_pairs if len(pair) > 0]
        second_tokens = [pair[1] for pair in second_stage_pairs if len(pair) > 1]

        group1_stage1_df = pdm_score_df[pdm_score_df["token"] == orig_token]
        group1_stage2_df = pdm_score_df[pdm_score_df["token"].isin(first_tokens)]
        group2_stage1_df = pdm_score_df[pdm_score_df["token"] == prev_token]
        group2_stage2_df = pdm_score_df[pdm_score_df["token"].isin(second_tokens)]

        group1_stage1_scores = calculate_weighted_average_score(group1_stage1_df)
        group1_stage2_scores = calculate_weighted_average_score(group1_stage2_df)
        group2_stage1_scores = calculate_weighted_average_score(group2_stage1_df)
        group2_stage2_scores = calculate_weighted_average_score(group2_stage2_df)

        stage1_group_scores.append(group1_stage1_scores)
        stage1_group_scores.append(group2_stage1_scores)
        stage2_group_scores.append(group1_stage2_scores)
        stage2_group_scores.append(group2_stage2_scores)

        group1_scores = group1_stage1_scores * group1_stage2_scores
        group2_scores = group2_stage1_scores * group2_stage2_scores
        avg_scores = (group1_scores + group2_scores) / 2
        all_group_scores.append(avg_scores)

    return (
        pd.DataFrame(all_group_scores).mean(),
        pd.DataFrame(stage1_group_scores).mean(),
        pd.DataFrame(stage2_group_scores).mean(),
    )


def create_scene_aggregators(
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
    full_score_df: pd.DataFrame,
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:
    """Populate pseudo closed-loop weights and two-frame comfort using SceneAggregator."""
    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df["weight"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []
    for (now_frame, previous_frame), second_stage in all_mappings.items():
        if now_frame not in full_score_df.index or previous_frame not in full_score_df.index:
            continue
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            second_stage=second_stage,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        all_updates.append(aggregator.aggregate_scores())

    if not all_updates:
        full_score_df.reset_index(inplace=True)
        return full_score_df

    all_updates_df = pd.concat(all_updates, ignore_index=True).set_index("token")
    full_score_df.update(all_updates_df)
    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"], errors="ignore")
    return full_score_df


class TransfuserPDMFeatureBuilder:
    """Builds Transfuser features from the current history frame."""

    def __init__(self):
        self.camera_size = (1024, 256)
        self.lidar_range = (-32.0, 32.0, -32.0, 32.0)
        self.pixels_per_meter = 4.0
        self.hist_max_per_pixel = 5
        self.max_height_lidar = 100.0
        self.lidar_split_height = 0.2

    def _splat_points(self, point_cloud: np.ndarray) -> np.ndarray:
        min_x, max_x, min_y, max_y = self.lidar_range
        xbins = np.linspace(min_x, max_x, int((max_x - min_x) * self.pixels_per_meter) + 1)
        ybins = np.linspace(min_y, max_y, int((max_y - min_y) * self.pixels_per_meter) + 1)
        hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
        hist = np.clip(hist, 0, self.hist_max_per_pixel)
        return hist / self.hist_max_per_pixel

    def _build_camera_feature(self, cameras) -> torch.Tensor:
        left = cameras.cam_l0.image[28:-28, 416:-416]
        front = cameras.cam_f0.image[28:-28]
        right = cameras.cam_r0.image[28:-28, 416:-416]

        stitched = np.concatenate([left, front, right], axis=1)
        resized = cv2.resize(stitched, self.camera_size)
        return torch.from_numpy(resized).permute(2, 0, 1).to(torch.float32) / 255.0

    def _build_lidar_feature(self, lidar) -> torch.Tensor:
        lidar_pc = lidar.lidar_pc[:3].T
        lidar_pc = lidar_pc[lidar_pc[:, 2] < self.max_height_lidar]
        above = lidar_pc[lidar_pc[:, 2] > self.lidar_split_height]
        overhead = self._splat_points(above)
        return torch.from_numpy(overhead[None].astype(np.float32))

    def _build_status_feature(self, ego_status) -> torch.Tensor:
        status = np.concatenate(
            [
                np.asarray(ego_status.driving_command, dtype=np.float32),
                np.asarray(ego_status.ego_velocity, dtype=np.float32),
                np.asarray(ego_status.ego_acceleration, dtype=np.float32),
            ],
            axis=0,
        )
        return torch.from_numpy(status)

    def build(self, agent_input) -> Dict[str, torch.Tensor]:
        current_cameras = agent_input.cameras[-1]
        current_lidar = agent_input.lidars[-1]
        current_ego_status = agent_input.ego_statuses[-1]
        return {
            "camera_feature": self._build_camera_feature(current_cameras),
            "lidar_feature": self._build_lidar_feature(current_lidar),
            "status_feature": self._build_status_feature(current_ego_status),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_train_test_split_config(train_test_split_name: str = "navhard_two_stage"):
    config_path = _repo_root() / "navsim/planning/script/config/common/train_test_split" / f"{train_test_split_name}.yaml"
    return OmegaConf.load(config_path)


def load_scene_filter(train_test_split_name: str = "navhard_two_stage"):
    split_cfg = load_train_test_split_config(train_test_split_name)
    scene_filter_name = split_cfg.defaults[0]["scene_filter"]
    scene_filter_path = (
        _repo_root() / "navsim/planning/script/config/common/train_test_split/scene_filter" / f"{scene_filter_name}.yaml"
    )
    return instantiate(OmegaConf.load(scene_filter_path)), split_cfg


def build_transfuser_two_stage_scene_loader(
    openscene_data_root: Union[str, Path],
    train_test_split_name: str = "navhard_two_stage",
    synthetic_dataset_dir: str = "navhard_two_stage",
    sensor_config: Optional[SensorConfig] = None,
    max_scenes: Optional[int] = None,
):
    scene_filter, split_cfg = load_scene_filter(train_test_split_name)
    if max_scenes is not None:
        scene_filter.max_scenes = max_scenes
    if sensor_config is None:
        current_idx = scene_filter.num_history_frames - 1
        sensor_config = SensorConfig(
            cam_f0=[current_idx],
            cam_l0=[current_idx],
            cam_l1=False,
            cam_l2=False,
            cam_r0=[current_idx],
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=[current_idx],
        )

    openscene_data_root = Path(openscene_data_root)
    scene_loader = SceneLoader(
        data_path=openscene_data_root / f"navsim_logs/{split_cfg.data_split}",
        original_sensor_path=openscene_data_root / f"sensor_blobs/{split_cfg.data_split}",
        scene_filter=scene_filter,
        synthetic_sensor_path=openscene_data_root / f"sensor_blobs/{split_cfg.data_split}",
        synthetic_scenes_path=openscene_data_root / synthetic_dataset_dir / "synthetic_scene_pickles",
        sensor_config=sensor_config,
    )
    return scene_loader, split_cfg


def repair_metric_cache_paths(metric_cache_loader: MetricCacheLoader, metric_cache_root: Union[str, Path]) -> MetricCacheLoader:
    """Rewrite stale absolute cache paths from metadata to the provided local cache root."""
    metric_cache_root = Path(metric_cache_root)
    repaired_paths = {}

    for token, cache_file in metric_cache_loader.metric_cache_paths.items():
        cache_file = Path(cache_file)
        if cache_file.exists():
            repaired_paths[token] = cache_file
            continue

        repaired_candidate = metric_cache_root.joinpath(*cache_file.parts[-4:])
        repaired_paths[token] = repaired_candidate

    metric_cache_loader.metric_cache_paths = repaired_paths
    return metric_cache_loader


def build_pdm_components():
    simulator = PDMSimulator(proposal_sampling=PDM_PROPOSAL_SAMPLING)
    scorer = PDMScorer(
        proposal_sampling=PDM_PROPOSAL_SAMPLING,
        config=PDMScorerConfig(
            progress_weight=5.0,
            ttc_weight=5.0,
            lane_keeping_weight=2.0,
            history_comfort_weight=2.0,
            two_frame_extended_comfort_weight=2.0,
            driving_direction_horizon=1.0,
            driving_direction_compliance_threshold=2.0,
            driving_direction_violation_threshold=6.0,
            stopped_speed_threshold=5e-03,
            future_collision_horizon_window=1.0,
            progress_distance_threshold=5.0,
            lane_keeping_deviation_limit=0.5,
            lane_keeping_horizon_window=2.0,
            human_penalty_filter=True,
        ),
    )

    def _build_policy() -> NavsimIDMTrafficAgents:
        idm_agents = NavsimIDMAgents(
            target_velocity=10.0,
            min_gap_to_lead_agent=1.0,
            headway_time=1.5,
            accel_max=1.0,
            decel_max=2.0,
            open_loop_detections_types=[],
            minimum_path_length=20.0,
            planned_trajectory_samples=None,
            planned_trajectory_sample_interval=None,
            radius=100.0,
            add_open_loop_parked_vehicles=True,
            idm_snap_threshold=3.0,
        )
        return NavsimIDMTrafficAgents(
            future_trajectory_sampling=PDM_PROPOSAL_SAMPLING,
            idm_agents_observation=idm_agents,
        )

    return simulator, scorer, _build_policy(), _build_policy()


def predict_transfuser_trajectory(
    model: nn.Module,
    agent_input,
    feature_builder: Optional[TransfuserPDMFeatureBuilder] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Trajectory:
    if feature_builder is None:
        feature_builder = TransfuserPDMFeatureBuilder()
    if device is None:
        device = next(model.parameters()).device

    features = feature_builder.build(agent_input)
    batched_features = {key: value.unsqueeze(0).to(device) for key, value in features.items()}

    model.eval()
    with torch.inference_mode():
        output = model(batched_features)

    trajectory = output["trajectory"][0].detach().cpu().numpy().astype(np.float32)
    return Trajectory(poses=trajectory, trajectory_sampling=TRANSFUSER_TRAJECTORY_SAMPLING)


def _score_token_list(
    model: nn.Module,
    tokens: List[str],
    scene_loader: SceneLoader,
    metric_cache_loader: MetricCacheLoader,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy,
    feature_builder: TransfuserPDMFeatureBuilder,
    device: Optional[Union[str, torch.device]] = None,
) -> List[pd.DataFrame]:
    pdm_results: List[pd.DataFrame] = []

    for token in tokens:
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            agent_input = scene_loader.get_agent_input_from_token(token)
            trajectory = predict_transfuser_trajectory(
                model=model,
                agent_input=agent_input,
                feature_builder=feature_builder,
                device=device,
            )

            score_row, ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
            )
            score_row["valid"] = True
            score_row["log_name"] = metric_cache.log_name
            score_row["frame_type"] = metric_cache.scene_type
            score_row["start_time"] = metric_cache.timepoint.time_s
            end_pose = StateSE2(
                x=trajectory.poses[-1, 0],
                y=trajectory.poses[-1, 1],
                heading=trajectory.poses[-1, 2],
            )
            absolute_endpoint = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
            score_row["endpoint_x"] = absolute_endpoint.x
            score_row["endpoint_y"] = absolute_endpoint.y
            score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
            score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
            score_row["ego_simulated_states"] = [ego_simulated_states]
        except Exception:
            logger.exception("Transfuser PDM scoring failed for token %s", token)
            score_row = pd.DataFrame([PDMResults.get_empty_results()])
            score_row["valid"] = False
            score_row["log_name"] = np.nan
            score_row["frame_type"] = np.nan
            score_row["start_time"] = np.nan
            score_row["endpoint_x"] = np.nan
            score_row["endpoint_y"] = np.nan
            score_row["start_point_x"] = np.nan
            score_row["start_point_y"] = np.nan
            score_row["ego_simulated_states"] = np.nan

        score_row["token"] = token
        pdm_results.append(score_row)

    return pdm_results


def evaluate_transfuser_two_stage_pdm(
    model: nn.Module,
    openscene_data_root: Union[str, Path],
    metric_cache_path: Union[str, Path],
    train_test_split_name: str = "navhard_two_stage",
    synthetic_dataset_dir: str = "navhard_two_stage",
    device: Optional[Union[str, torch.device]] = None,
    max_scenes: Optional[int] = None,
) -> pd.DataFrame:
    scene_loader, split_cfg = build_transfuser_two_stage_scene_loader(
        openscene_data_root=openscene_data_root,
        train_test_split_name=train_test_split_name,
        synthetic_dataset_dir=synthetic_dataset_dir,
        max_scenes=max_scenes,
    )
    metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))
    metric_cache_loader = repair_metric_cache_paths(metric_cache_loader, metric_cache_path)
    metric_cache_tokens = set(metric_cache_loader.tokens)

    stage_one_tokens = [token for token in scene_loader.tokens_stage_one if token in metric_cache_tokens]
    reactive_stage_two_tokens = scene_loader.reactive_tokens_stage_two or []
    stage_two_tokens = [token for token in reactive_stage_two_tokens if token in metric_cache_tokens]

    simulator, scorer, stage_one_policy, stage_two_policy = build_pdm_components()
    feature_builder = TransfuserPDMFeatureBuilder()

    score_rows: List[pd.DataFrame] = []
    score_rows.extend(
        _score_token_list(
            model=model,
            tokens=stage_one_tokens,
            scene_loader=scene_loader,
            metric_cache_loader=metric_cache_loader,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=stage_one_policy,
            feature_builder=feature_builder,
            device=device,
        )
    )
    score_rows.extend(
        _score_token_list(
            model=model,
            tokens=stage_two_tokens,
            scene_loader=scene_loader,
            metric_cache_loader=metric_cache_loader,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=stage_two_policy,
            feature_builder=feature_builder,
            device=device,
        )
    )

    if not score_rows:
        raise ValueError("No scenes were scored. Check scene filtering and metric cache coverage.")

    pdm_score_df = pd.concat(score_rows, ignore_index=True)

    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    valid_rows = pdm_score_df[
        pdm_score_df["valid"].fillna(False)
        & pdm_score_df["start_time"].notna()
        & pdm_score_df["ego_simulated_states"].notna()
    ]
    scored_tokens = set(valid_rows["token"].tolist())
    for orig_token, prev_token, two_stage_pairs in split_cfg.reactive_all_mapping:
        filtered_pairs = [
            tuple(pair)
            for pair in two_stage_pairs
            if len(pair) == 2 and pair[0] in scored_tokens and pair[1] in scored_tokens
        ]
        if orig_token in scored_tokens and prev_token in scored_tokens and filtered_pairs:
            all_mappings[(orig_token, prev_token)] = filtered_pairs

    pdm_score_df = create_scene_aggregators(
        all_mappings=all_mappings,
        full_score_df=pdm_score_df,
        proposal_sampling=simulator.proposal_sampling,
    )
    if all_mappings:
        pdm_score_df = compute_final_scores(pdm_score_df)
    else:
        pdm_score_df["score"] = pdm_score_df["pdm_score"]
        pdm_score_df["weight"] = 1.0

    score_cols = [
        column
        for column in pdm_score_df.columns
        if (
            any(score.name in column for score in fields(PDMResults))
            or column == "two_frame_extended_comfort"
            or column == "score"
        )
        and column != "pdm_score"
    ]

    pcl_group_score, pcl_stage1_score, pcl_stage2_score = calculate_individual_mapping_scores(
        pdm_score_df[score_cols + ["token", "weight"]],
        all_mappings,
    )

    for column in score_cols:
        stage_one_mask = pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL
        stage_two_mask = pdm_score_df["frame_type"] == SceneFrameType.SYNTHETIC
        pdm_score_df.loc[stage_one_mask, f"{column}_stage_one"] = pdm_score_df.loc[stage_one_mask, column]
        pdm_score_df.loc[stage_two_mask, f"{column}_stage_two"] = pdm_score_df.loc[stage_two_mask, column]

    pdm_score_df.drop(columns=score_cols, inplace=True)
    pdm_score_df["score"] = pdm_score_df["score_stage_one"].combine_first(pdm_score_df["score_stage_two"])
    pdm_score_df.drop(columns=["score_stage_one", "score_stage_two"], inplace=True)

    stage1_cols = [f"{column}_stage_one" for column in score_cols if column != "score"]
    stage2_cols = [f"{column}_stage_two" for column in score_cols if column != "score"]
    ordered_score_cols = stage1_cols + stage2_cols + ["score"]
    pdm_score_df = pdm_score_df[["token", "valid"] + ordered_score_cols]

    summary_rows = []

    stage1_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage1_row["token"] = "extended_pdm_score_stage_one"
    stage1_row["valid"] = True
    stage1_row["score"] = pcl_stage1_score.get("score", np.nan)
    for column in pcl_stage1_score.index:
        if column not in ["token", "valid", "score"]:
            stage1_row[f"{column}_stage_one"] = pcl_stage1_score[column]
    summary_rows.append(stage1_row)

    stage2_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    stage2_row["token"] = "extended_pdm_score_stage_two"
    stage2_row["valid"] = True
    stage2_row["score"] = pcl_stage2_score.get("score", np.nan)
    for column in pcl_stage2_score.index:
        if column not in ["token", "valid", "score"]:
            stage2_row[f"{column}_stage_two"] = pcl_stage2_score[column]
    summary_rows.append(stage2_row)

    combined_row = pd.Series(index=pdm_score_df.columns, dtype=object)
    combined_row["token"] = "extended_pdm_score_combined"
    combined_row["valid"] = True
    combined_row["score"] = pcl_group_score.get("score", np.nan)
    for column in pcl_stage1_score.index:
        if column not in ["token", "valid", "score"]:
            combined_row[f"{column}_stage_one"] = pcl_stage1_score[column]
    for column in pcl_stage2_score.index:
        if column not in ["token", "valid", "score"]:
            combined_row[f"{column}_stage_two"] = pcl_stage2_score[column]
    summary_rows.append(combined_row)

    return pd.concat([pdm_score_df, pd.DataFrame(summary_rows)], ignore_index=True)
