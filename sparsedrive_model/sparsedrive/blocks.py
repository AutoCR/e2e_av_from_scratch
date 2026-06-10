from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .nn_utils import build_activation_layer, build_norm_layer, build_dropout
from .nn_utils import constant_init, xavier_init

Linear = nn.Linear
Sequential = nn.Sequential

try:
    from .ops import deformable_aggregation_function as DAF
except:
    DAF = None

__all__ = [
    "DeformableFeatureAggregation",
    "DenseDepthNet",
    "AsymmetricFFN",
]

# Physical minimum camera-frame depth (metres) for keypoint feature sampling.
# Points at/behind this plane are masked off-image in project_points (zero
# forward contribution, zero gradient). See the comment in project_points for
# the full empirical justification; raising/lowering this trades how close to
# a lens a keypoint may sample against the 1/z^2 backward amplification bound
# (max |du/dx| = 1/PROJECT_Z_MIN per decoder layer).
PROJECT_Z_MIN = 0.5


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class FFN(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.0,
        act="ReLU",
        add_identity=True,
    ):
        super().__init__()
        if num_fcs < 2:
            raise ValueError("num_fcs must be at least 2")
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.extend(
                [
                    nn.Linear(in_channels, feedforward_channels),
                    build_activation_layer(act),
                    nn.Dropout(ffn_drop),
                ]
            )
            in_channels = feedforward_channels
        layers.extend([nn.Linear(feedforward_channels, embed_dims), nn.Dropout(ffn_drop)])
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return out
        return out + (x if identity is None else identity)


class DeformableFeatureAggregation(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: nn.Module = None,
        temporal_fusion_module: nn.Module = None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super().__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        if kps_generator is None:
            raise ValueError("kps_generator must be an instantiated module")
        self.kps_generator = kps_generator
        self.num_pts = self.kps_generator.num_pts
        self.temp_module = temporal_fusion_module
        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        weights = self._get_weights(instance_feature, anchor_embed, metas)

        if self.use_deformable_func:
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)  # fuse multi-point features
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None],
            pts_extend[:, None, ..., None],
        ).squeeze(-1)
        # Forward probe (no-op unless SD_DEBUG>=1): log the camera-frame depth
        # distribution before the divide.
        from .debug_probe import probe_projection_depth

        probe_projection_depth(points_2d[..., 2:3], floor=PROJECT_Z_MIN)
        # Bounded-Jacobian masked perspective division (deliberate deviation from
        # upstream's bare `xy / clamp(z, min=1e-5)`), because upstream's exact
        # numerics are empirically UNSTABLE in the NAVSIM regime. Evidence from
        # three instrumented 6-GPU runs (2026-06-09/10): training explodes at
        # iter ~7.3k-9.3k under floor=1e-1+skip, floor=1e-5+skip, AND
        # floor=1e-5+clip-25-step-through (upstream's own policy) -- grad norms
        # escalate 1e5 -> 1e34 within ~15 iters while the LOSS STAYS ~22 and the
        # forward depth stats are static. Mechanism: NAVSIM has ONE vehicle
        # (constant camera extrinsics) and the kmeans det anchors are frozen, so
        # ~400-600 of 421k keypoints sit within 10 cm of a camera on EVERY
        # forward pass; the backward of x/z scales as 1/z^2 there, and the
        # 6-layer decoder chains six such divisions (each refined anchor feeds
        # the next layer's projection, un-detached -- upstream-canonical), giving
        # up to (1/z)^6 ~ 1e30 amplification into the refinement MLP. No
        # clip/skip policy can manage that; the cliff itself must go.
        #
        # Fix: nothing visible can be feature-sampled closer than PROJECT_Z_MIN
        # to a lens, so points at/behind it are pushed far off-image through a
        # GRADIENT-LESS branch (torch.where on a binary depth test; the constant
        # branch carries no graph). DAF / grid_sample then give them zero
        # contribution AND zero gradient -- which upstream's behind-camera points
        # got anyway via z=1e-5 -> coords ~1e6 off-image; the only behavioral
        # change is removing the rare ON-image sample with z < 0.5 m, which is
        # precisely the pathological population. For surviving points the
        # Jacobian is bounded: |du/dx| = 1/z <= 2, and on-image |u|<=1 implies
        # |du/dz| = |u|/z <= 2, so the 6-layer cascade is <= ~2^6 = 64 --
        # trivially clippable.
        depth = points_2d[..., 2:3]
        points_2d = points_2d[..., :2] / torch.clamp(
            depth, min=PROJECT_Z_MIN
        )
        points_2d = torch.where(
            depth > PROJECT_Z_MIN,
            points_2d,
            torch.full_like(points_2d, -1e4),
        )
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            fm_flat = fm.flatten(end_dim=1)
            features.append(
                torch.nn.functional.grid_sample(fm_flat, points_2d)
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features


class DenseDepthNet(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1)).clamp(max=88.0).exp()
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1)
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            error = torch.abs(pred - gt).sum()
            _loss = (
                error
                / max(1.0, len(gt) * len(depth_preds))
                * self.loss_weight
            )
            loss = loss + _loss
        return loss


class AsymmetricFFN(nn.Module):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg="ReLU",
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__()
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)
