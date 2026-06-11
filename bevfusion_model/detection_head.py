import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle both relative imports (when used as module) and direct execution
try:
    from .detection_losses_base import (FocalLoss, GaussianFocalLoss, L1Loss,
        gaussian_radius, draw_heatmap_gaussian, clip_sigmoid, normalize_bbox)
    from .detection_assigner import build_assigner
except (ImportError, ValueError):
    # For direct script execution, add parent to path
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from detection_losses_base import (FocalLoss, GaussianFocalLoss, L1Loss,
        gaussian_radius, draw_heatmap_gaussian, clip_sigmoid, normalize_bbox)
    from detection_assigner import build_assigner


def _circle_nms(boxes, min_radius, post_max_size=83):
    """Simple greedy circle NMS. boxes: (N, 3) [x, y, score]."""
    if len(boxes) == 0:
        return []
    order = np.argsort(-boxes[:, 2])
    keep = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        if len(keep) >= post_max_size:
            break
        for j in order:
            if suppressed[j] or i == j:
                continue
            dist = np.sqrt(((boxes[i, :2] - boxes[j, :2]) ** 2).sum())
            if dist < min_radius * 2:
                suppressed[j] = True
    return keep


class TransFusionBBoxCoder(nn.Module):
    """Bbox coder for TransFusion head."""
    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 post_center_range=None,
                 score_threshold=None,
                 code_size=8):
        super().__init__()
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold
        self.code_size = code_size

    def encode(self, gt_bboxes):
        """Encode ground-truth boxes to feature-space targets.

        Args:
            gt_bboxes: [G, 9] or [G, 10] world/lidar [cx,cy,cz,w,l,h,yaw,(vx,vy)]

        Returns:
            [G, code_size] feature-map-space targets [cx_f, cy_f, height, log_w, log_l, log_h, sin, cos, (vx, vy)]
        """
        targets = gt_bboxes.new_zeros((gt_bboxes.shape[0], self.code_size))
        # Center in feature coordinates
        targets[:, 0] = (gt_bboxes[:, 0] - self.pc_range[0]) / (self.out_size_factor * self.voxel_size[0])
        targets[:, 1] = (gt_bboxes[:, 1] - self.pc_range[1]) / (self.out_size_factor * self.voxel_size[1])
        # Log dimensions
        targets[:, 3] = gt_bboxes[:, 3].log()  # log_w
        targets[:, 4] = gt_bboxes[:, 4].log()  # log_l
        targets[:, 5] = gt_bboxes[:, 5].log()  # log_h
        # Height (center z + half-height, to invert decode's subtraction)
        targets[:, 2] = gt_bboxes[:, 2] + gt_bboxes[:, 5] * 0.5
        # Rotation as sin/cos
        targets[:, 6] = torch.sin(gt_bboxes[:, 6])
        targets[:, 7] = torch.cos(gt_bboxes[:, 6])
        # Velocity (if present)
        if self.code_size > 8 and gt_bboxes.shape[1] > 7:
            targets[:, 8] = gt_bboxes[:, 7]
            targets[:, 9] = gt_bboxes[:, 8]
        return targets

    def decode(self, heatmap, rot, dim, center, height, vel, filter=False):
        """Decode bboxes.
        Args:
            heatmap (torch.Tensor): Heatmap with shape [B, num_cls, num_proposals].
            rot (torch.Tensor): Rotation with shape [B, 2, num_proposals].
            dim (torch.Tensor): Dimensions with shape [B, 3, num_proposals].
            center (torch.Tensor): BEV center with shape [B, 2, num_proposals] (feature map coords).
            height (torch.Tensor): Height with shape [B, 1, num_proposals].
            vel (torch.Tensor): Velocity with shape [B, 2, num_proposals] or None.
            filter (bool): Whether to filter by score threshold and center range.
        Returns:
            list[dict]: Decoded boxes.
        """
        final_preds = heatmap.max(1, keepdim=False).indices
        final_scores = heatmap.max(1, keepdim=False).values

        center[:, 0, :] = center[:, 0, :] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        center[:, 1, :] = center[:, 1, :] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]

        dim = dim.exp()

        height = height - dim[:, 2:3, :] * 0.5

        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot_angle = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot_angle], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot_angle, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            predictions_dicts.append({
                "bboxes": final_box_preds[i],
                "scores": final_scores[i],
                "labels": final_preds[i],
            })

        if not filter:
            return predictions_dicts

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            post_center_range = torch.tensor(self.post_center_range, device=heatmap.device)
            mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(heatmap.shape[0]):
                cmask = mask[i]
                if self.score_threshold is not None:
                    cmask = cmask & thresh_mask[i]
                predictions_dicts.append({
                    "bboxes": final_box_preds[i, cmask],
                    "scores": final_scores[i, cmask],
                    "labels": final_preds[i, cmask],
                })
        else:
            raise NotImplementedError("post_center_range must be set")

        return predictions_dicts


class _Conv1dBnRelu(nn.Module):
    """1D Conv with BatchNorm and ReLU activation."""
    def __init__(self, in_ch, out_ch, kernel=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class _Conv2dBnRelu(nn.Module):
    """2D Conv with BatchNorm and ReLU activation."""
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class SeparateHead(nn.Module):
    """Separate prediction heads for each output type (FFN equivalent)."""
    def __init__(self, in_channels, heads):
        super().__init__()
        self._head_names = list(heads.keys())
        for head_name, (out_ch, num_conv) in heads.items():
            layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                layers.append(_Conv1dBnRelu(c_in, 64, kernel=1))
                c_in = 64
            layers.append(nn.Conv1d(c_in, out_ch, 1))
            setattr(self, head_name, nn.Sequential(*layers))

    def forward(self, x):
        ret_dict = {}
        for head_name in self._head_names:
            ret_dict[head_name] = getattr(self, head_name)(x)
        return ret_dict


class PositionEmbeddingLearned(nn.Module):
    """Learned positional embeddings."""
    def __init__(self, input_channel, num_pos_feats=128):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, 1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, 1),
        )

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention, cross-attention, and FFN."""
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError(f"activation should be relu/gelu, not {activation}")

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, attn_mask=None):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 2
        :param key_pos: B Pk 2
        :param attn_mask: optional attention mask
        :return: query (B C Pq)
        """
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        q = k = v = self.with_pos_embed(query, query_pos_embed)
        query2 = self.self_attn(q, k, value=v)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.multihead_attn(
            query=self.with_pos_embed(query, query_pos_embed),
            key=self.with_pos_embed(key, key_pos_embed),
            value=self.with_pos_embed(key, key_pos_embed),
            attn_mask=attn_mask
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        query = query.permute(1, 2, 0)
        return query


class TransFusionHead(nn.Module):
    """BEVFusion TransFusion detection head."""
    def __init__(self,
                 num_proposals=200,
                 in_channels=512,
                 hidden_channel=128,
                 num_classes=10,
                 num_decoder_layers=1,
                 num_heads=8,
                 nms_kernel_size=1,
                 ffn_channel=256,
                 dropout=0.1,
                 bn_momentum=0.1,
                 activation="relu",
                 common_heads=None,
                 auxiliary=True,
                 test_cfg=None,
                 train_cfg=None):
        super().__init__()

        self.num_proposals = num_proposals
        self.num_classes = num_classes
        self.auxiliary = auxiliary
        self.num_decoder_layers = num_decoder_layers
        self.nms_kernel_size = nms_kernel_size
        self.test_cfg = test_cfg or {}
        self.train_cfg = train_cfg
        self.bn_momentum = bn_momentum

        self.shared_conv = nn.Conv2d(in_channels, hidden_channel, 3, padding=1)

        layers = []
        layers.append(_Conv2dBnRelu(hidden_channel, hidden_channel, kernel=3, padding=1))
        layers.append(nn.Conv2d(hidden_channel, num_classes, 3, padding=1))
        self.heatmap_head = nn.Sequential(*layers)

        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        self.decoder = nn.ModuleList()
        for i in range(num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    d_model=hidden_channel,
                    nhead=num_heads,
                    dim_feedforward=ffn_channel,
                    dropout=dropout,
                    activation=activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                )
            )

        if common_heads is None:
            common_heads = {"center": [2, 2], "height": [1, 2], "dim": [3, 2], "rot": [2, 2], "vel": [2, 2]}

        self.prediction_heads = nn.ModuleList()
        for i in range(num_decoder_layers):
            heads = dict(common_heads)
            heads["heatmap"] = [num_classes, 2]
            self.prediction_heads.append(SeparateHead(hidden_channel, heads))

        self.init_bn_momentum()

        x_size = self.test_cfg.get("grid_size", [1440, 1440])[0] // self.test_cfg.get("out_size_factor", 8)
        y_size = self.test_cfg.get("grid_size", [1440, 1440])[1] // self.test_cfg.get("out_size_factor", 8)
        bev_pos = self.create_2D_grid(x_size, y_size)
        self.register_buffer("bev_pos", bev_pos)

        self.bbox_coder = TransFusionBBoxCoder(
            pc_range=self.test_cfg.get("pc_range", [-54.0, -54.0, -5.0]),
            out_size_factor=self.test_cfg.get("out_size_factor", 8),
            voxel_size=self.test_cfg.get("voxel_size", [0.075, 0.075, 0.2]),
            post_center_range=self.test_cfg.get("post_center_limit_range",
                                                [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]),
            score_threshold=self.test_cfg.get("score_threshold", 0.0),
            code_size=10,
        )

        self.query_labels = None

        # Initialize training components if train_cfg is provided with loss config
        has_loss_cfg = train_cfg is not None and "loss_cls" in train_cfg and "assigner" in train_cfg
        if has_loss_cfg:
            lc = train_cfg.get("loss_cls", {})
            lh = train_cfg.get("loss_heatmap", {})
            lb = train_cfg.get("loss_bbox", {})
            self.loss_cls = FocalLoss(use_sigmoid=True, gamma=lc.get("gamma", 2.0),
                                     alpha=lc.get("alpha", 0.25), loss_weight=lc.get("loss_weight", 1.0))
            self.loss_heatmap = GaussianFocalLoss(loss_weight=lh.get("loss_weight", 1.0))
            self.loss_bbox = L1Loss(loss_weight=lb.get("loss_weight", 0.25))
            self.bbox_assigner = build_assigner(train_cfg["assigner"])
            self.train_pc_range = train_cfg["point_cloud_range"]
            self.train_voxel_size = train_cfg["voxel_size"]
            self.train_grid_size = train_cfg["grid_size"]
            self.train_out_size_factor = train_cfg["out_size_factor"]
            self.gaussian_overlap = train_cfg.get("gaussian_overlap", 0.1)
            self.min_radius = train_cfg.get("min_radius", 2)
            self.code_weights = train_cfg.get("code_weights", [1.0]*8 + [0.2]*2)
            self.pos_weight = train_cfg.get("pos_weight", -1)
        else:
            self.loss_cls = None
            self.loss_heatmap = None
            self.loss_bbox = None
            self.bbox_assigner = None
            self.train_pc_range = None
            self.train_voxel_size = None
            self.train_grid_size = None
            self.train_out_size_factor = None
            self.gaussian_overlap = None
            self.min_radius = None
            self.code_weights = None
            self.pos_weight = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward_single(self, inputs, metas):
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        dense_heatmap = self.heatmap_head(lidar_feat)
        heatmap = dense_heatmap.detach().sigmoid()

        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding) if padding > 0 else None, padding:(-padding) if padding > 0 else None] = local_max_inner

        if self.test_cfg.get("dataset") == "nuScenes" and self.num_classes >= 10:
            local_max[:, 8] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg.get("dataset") == "Waymo" and self.num_classes >= 3:
            local_max[:, 1] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)

        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]

        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
            dim=-1,
        )

        self.query_labels = top_proposals_class

        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1).float()
        query_cat_encoding = self.class_encoding(one_hot)
        query_feat = query_feat + query_cat_encoding

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        ret_dicts = []
        for i in range(self.num_decoder_layers):
            query_feat = self.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos)
            res_layer = self.prediction_heads[i](query_feat)
            res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
            ret_dicts.append(res_layer)
            query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

        ret_dicts[0]["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
            dim=-1,
        )
        ret_dicts[0]["dense_heatmap"] = dense_heatmap

        if not self.auxiliary:
            return [ret_dicts[-1]]

        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ["dense_heatmap", "query_heatmap_score"]:
                new_res[key] = torch.cat([d[key] for d in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, metas):
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        results = [self.forward_single(f, metas) for f in feats]
        return tuple(results)

    def get_targets_single(self, gt_bboxes, gt_labels, pred_dict, meta=None):
        """Compute targets for a single sample.

        Args:
            gt_bboxes: [G, 9] ground-truth boxes in world coords.
            gt_labels: [G] long labels in [0, num_classes).
            pred_dict: Dict with one batch-element sliced predictions (each shape [1, C, P]).

        Returns:
            Tuple of:
            - labels: [P] long tensor, num_classes for bg
            - label_weights: [P] float tensor, all 1.0 (pos_weight=-1)
            - bbox_targets: [P, code_size]
            - bbox_weights: [P, code_size]
            - num_pos: int, number of positive assignments
            - heatmap_t: [num_classes, Hf, Wf] dense heatmap target
        """
        if gt_labels.numel() > 0 and not ((gt_labels >= 0) & (gt_labels < self.num_classes)).all():
            bad = gt_labels[(gt_labels < 0) | (gt_labels >= self.num_classes)].detach().cpu().tolist()
            token = meta.get("token") if isinstance(meta, dict) else None
            where = f" for token {token}" if token else ""
            raise ValueError(
                f"gt_labels must be in [0, {self.num_classes - 1}] for "
                f"TransFusionHead(num_classes={self.num_classes}){where}; "
                f"found invalid labels {bad[:10]}"
            )

        num_proposals = pred_dict["center"].shape[-1]

        # Decode proposals to world coordinates (clone to avoid mutating input)
        score = pred_dict["heatmap"].detach().sigmoid().clone()
        center = pred_dict["center"].detach().clone()
        height = pred_dict["height"].detach().clone()
        dim = pred_dict["dim"].detach().clone()
        rot = pred_dict["rot"].detach().clone()
        vel = pred_dict["vel"].detach().clone()
        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)
        bbox_pred_world = boxes_dict[0]["bboxes"]  # [num_proposals, 9]

        # Class logits per proposal
        cls_pred = pred_dict["heatmap"][0].permute(1, 0)  # [num_proposals, num_classes]

        # Run Hungarian assignment
        assigned_gt_inds, assigned_labels, max_overlaps = self.bbox_assigner.assign(
            bbox_pred_world, cls_pred, gt_bboxes, gt_labels)

        # Extract positive indices
        pos_inds = (assigned_gt_inds > 0).nonzero(as_tuple=False).squeeze(-1)
        pos_gt_idx = assigned_gt_inds[pos_inds] - 1

        # Initialize target tensors
        labels = bbox_pred_world.new_full((num_proposals,), self.num_classes, dtype=torch.long)
        label_weights = bbox_pred_world.new_ones(num_proposals)
        bbox_targets = bbox_pred_world.new_zeros((num_proposals, self.bbox_coder.code_size))
        bbox_weights = bbox_pred_world.new_zeros((num_proposals, self.bbox_coder.code_size))

        # Fill in positive examples
        if pos_inds.numel() > 0:
            labels[pos_inds] = gt_labels[pos_gt_idx]
            bbox_targets[pos_inds] = self.bbox_coder.encode(gt_bboxes[pos_gt_idx])
            bbox_weights[pos_inds] = 1.0

        # Dense heatmap target
        feat_h = self.train_grid_size[1] // self.train_out_size_factor
        feat_w = self.train_grid_size[0] // self.train_out_size_factor
        heatmap_t = gt_bboxes.new_zeros((self.num_classes, feat_h, feat_w))

        for k in range(gt_bboxes.shape[0]):
            w = gt_bboxes[k, 3]
            l = gt_bboxes[k, 4]
            w_f = w / self.train_voxel_size[0] / self.train_out_size_factor
            l_f = l / self.train_voxel_size[1] / self.train_out_size_factor

            if w_f <= 0 or l_f <= 0:
                continue

            radius = gaussian_radius((l_f, w_f), min_overlap=self.gaussian_overlap)
            radius = max(self.min_radius, int(radius))

            cx = gt_bboxes[k, 0]
            cy = gt_bboxes[k, 1]
            coor_x = (cx - self.train_pc_range[0]) / self.train_voxel_size[0] / self.train_out_size_factor
            coor_y = (cy - self.train_pc_range[1]) / self.train_voxel_size[1] / self.train_out_size_factor
            ctr = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=gt_bboxes.device)
            ctr_int = ctr.to(torch.int32)

            if not (0 <= ctr_int[0] < feat_w and 0 <= ctr_int[1] < feat_h):
                continue

            draw_heatmap_gaussian(heatmap_t[gt_labels[k]], ctr_int, radius)

        return labels, label_weights, bbox_targets, bbox_weights, int(pos_inds.numel()), heatmap_t

    def get_targets(self, gt_bboxes_list, gt_labels_list, preds_dict, metas=None):
        """Compute targets for all samples in a batch.

        Args:
            gt_bboxes_list: List[Tensor], each [G_i, 9] ground-truth boxes.
            gt_labels_list: List[Tensor], each [G_i] long labels.
            preds_dict: Dict with batched predictions [B, C, P].

        Returns:
            Tuple of:
            - labels: [B, P] long
            - label_weights: [B, P] float
            - bbox_targets: [B, P, code_size]
            - bbox_weights: [B, P, code_size]
            - heatmap_t: [B, num_classes, Hf, Wf]
            - num_pos: int, total positive count
        """
        B = preds_dict["center"].shape[0]
        labels_l, lw_l, bt_l, bw_l, hm_l = [], [], [], [], []
        num_pos = 0

        for b in range(B):
            single = {k: (v[b:b+1] if torch.is_tensor(v) else v) for k, v in preds_dict.items()}
            meta = metas[b] if metas is not None and b < len(metas) else None
            labels, lw, bt, bw, npos, hm = self.get_targets_single(
                gt_bboxes_list[b], gt_labels_list[b], single, meta=meta)
            labels_l.append(labels)
            lw_l.append(lw)
            bt_l.append(bt)
            bw_l.append(bw)
            hm_l.append(hm)
            num_pos += npos

        return (torch.stack(labels_l), torch.stack(lw_l), torch.stack(bt_l), torch.stack(bw_l),
                torch.stack(hm_l), max(num_pos, 1))

    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, metas=None, **kwargs):
        """Compute training losses.

        Args:
            gt_bboxes_3d: List[Tensor], each [G_i, 9] ground-truth boxes.
            gt_labels_3d: List[Tensor], each [G_i] long labels.
            preds_dicts: Output from forward(), tuple([merged_dict]).

        Returns:
            Dict of loss scalars.
        """
        # fp32 fence mirroring upstream @force_fp32 on TransFusionHead.loss:
        # under fp16 autocast the raw L1 sum over [B, P, code_size] overflows
        # fp16's max (65504) at init, making the total loss inf on every batch.
        with torch.cuda.amp.autocast(enabled=False):
            preds_dict = {
                k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in preds_dicts[0][0].items()
            }
            gt_bboxes_3d = [b.float() for b in gt_bboxes_3d]
            return self._loss_fp32(gt_bboxes_3d, gt_labels_3d, preds_dict, metas=metas)

    def _loss_fp32(self, gt_bboxes_3d, gt_labels_3d, preds_dict, metas=None):
        labels, label_weights, bbox_targets, bbox_weights, heatmap_t, num_pos = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dict, metas=metas)

        loss_dict = {}

        # Heatmap loss (once)
        loss_dict["loss_heatmap"] = self.loss_heatmap(
            clip_sigmoid(preds_dict["dense_heatmap"]), heatmap_t,
            avg_factor=max(heatmap_t.eq(1).float().sum().item(), 1))

        # Classification loss
        P = preds_dict["heatmap"].shape[-1]
        cls_score = preds_dict["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)
        loss_dict["loss_cls"] = self.loss_cls(
            cls_score, labels.reshape(-1), label_weights.reshape(-1), avg_factor=num_pos)

        # Regression loss (concatenate preds in order: center, height, dim, rot, vel)
        preds = torch.cat([preds_dict["center"], preds_dict["height"], preds_dict["dim"],
                           preds_dict["rot"], preds_dict["vel"]], dim=1).permute(0, 2, 1)
        code_weights = preds.new_tensor(self.code_weights)
        reg_weights = bbox_weights * code_weights.view(1, 1, -1)
        loss_dict["loss_bbox"] = self.loss_bbox(preds, bbox_targets, reg_weights, avg_factor=num_pos)

        return loss_dict

    def get_bboxes(self, preds_dicts, metas):
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]["heatmap"].shape[0]

            batch_score = preds_dict[0]["heatmap"][..., -self.num_proposals:].sigmoid()

            one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1).float()
            batch_score = batch_score * preds_dict[0]["query_heatmap_score"] * one_hot

            batch_center = preds_dict[0]["center"][..., -self.num_proposals:]
            batch_height = preds_dict[0]["height"][..., -self.num_proposals:]
            batch_dim = preds_dict[0]["dim"][..., -self.num_proposals:]
            batch_rot = preds_dict[0]["rot"][..., -self.num_proposals:]
            batch_vel = preds_dict[0].get("vel", None)
            if batch_vel is not None:
                batch_vel = batch_vel[..., -self.num_proposals:]

            temp = self.bbox_coder.decode(
                batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel,
                filter=True,
            )

            if self.test_cfg.get("dataset") == "nuScenes":
                tasks = [
                    {"num_class": 8, "class_names": [], "indices": list(range(8)), "radius": -1},
                    {"num_class": 1, "class_names": ["pedestrian"], "indices": [8], "radius": 0.175},
                    {"num_class": 1, "class_names": ["traffic_cone"], "indices": [9], "radius": 0.175},
                ]
            elif self.test_cfg.get("dataset") == "Waymo":
                tasks = [
                    {"num_class": 1, "class_names": ["Car"], "indices": [0], "radius": 0.7},
                    {"num_class": 1, "class_names": ["Pedestrian"], "indices": [1], "radius": 0.7},
                    {"num_class": 1, "class_names": ["Cyclist"], "indices": [2], "radius": 0.7},
                ]
            else:
                tasks = None

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]["bboxes"]
                scores = temp[i]["scores"]
                labels = temp[i]["labels"]

                if tasks is not None and self.test_cfg.get("nms_type") == "circle":
                    keep_mask = torch.zeros_like(scores)
                    for task in tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task["indices"]:
                            task_mask += (labels == cls_idx).float()
                        task_mask = task_mask.bool()
                        if task["radius"] > 0 and task_mask.sum() > 0:
                            boxes_for_nms = torch.cat([
                                boxes3d[task_mask][:, :2],
                                scores[task_mask][:, None],
                            ], dim=1)
                            keep_indices = _circle_nms(
                                boxes_for_nms.detach().cpu().numpy(),
                                task["radius"],
                            )
                            keep_indices = torch.tensor(keep_indices, device=scores.device)
                            if keep_indices.shape[0] > 0:
                                keep_orig = torch.where(task_mask)[0][keep_indices]
                                keep_mask[keep_orig] = 1
                        else:
                            keep_mask[task_mask] = 1
                    keep_mask = keep_mask.bool()
                    ret_layer.append({
                        "boxes_3d": boxes3d[keep_mask],
                        "scores_3d": scores[keep_mask],
                        "labels_3d": labels[keep_mask],
                    })
                else:
                    ret_layer.append({
                        "boxes_3d": boxes3d,
                        "scores_3d": scores,
                        "labels_3d": labels,
                    })

            rets.append(ret_layer)

        return [rets[-1][i] for i in range(batch_size)]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from bevfusion_model.configs.bevfusion_hyperparams import get_training_hyperparams

    print("="*70)
    print("TransFusionHead Loss Path Self-Test")
    print("="*70)

    # Get training hyperparams
    hp = get_training_hyperparams()["detection_head"]

    # Construct head with train_cfg
    print("\n[1] Constructing TransFusionHead with train_cfg...")
    head = TransFusionHead(
        num_proposals=hp["num_proposals"],
        in_channels=hp["in_channels"],
        hidden_channel=hp["hidden_channel"],
        num_classes=hp["num_classes"],
        num_decoder_layers=hp["num_decoder_layers"],
        num_heads=hp["num_heads"],
        nms_kernel_size=hp["nms_kernel_size"],
        ffn_channel=hp["ffn_channel"],
        dropout=hp["dropout"],
        bn_momentum=hp["bn_momentum"],
        activation=hp["activation"],
        common_heads=hp["common_heads"],
        test_cfg=hp["test_cfg"],
        train_cfg=hp["train_cfg"])
    head.train()
    print(f"    Created head with num_classes={hp['num_classes']}, train_cfg={'present' if hp['train_cfg'] else 'None'}")

    # Verify inference construction still works (test_cfg only)
    print("\n[2] Verifying inference construction (train_cfg=None)...")
    head_inference = TransFusionHead(
        num_proposals=hp["num_proposals"],
        in_channels=hp["in_channels"],
        hidden_channel=hp["hidden_channel"],
        num_classes=hp["num_classes"],
        num_decoder_layers=hp["num_decoder_layers"],
        num_heads=hp["num_heads"],
        nms_kernel_size=hp["nms_kernel_size"],
        ffn_channel=hp["ffn_channel"],
        dropout=hp["dropout"],
        bn_momentum=hp["bn_momentum"],
        activation=hp["activation"],
        common_heads=hp["common_heads"],
        test_cfg=hp["test_cfg"])
    head_inference.eval()
    assert head_inference.loss_cls is None, "Inference head should have loss_cls=None"
    print("    Inference construction OK (loss components=None)")

    # Forward pass
    print("\n[3] Running forward pass...")
    B = 2
    bev = torch.randn(B, hp["in_channels"], 180, 180)
    preds = head(bev, metas=[{} for _ in range(B)])
    print(f"    Forward output shape: {type(preds)}, len={len(preds)}")
    print(f"    preds[0] type: {type(preds[0])}, len={len(preds[0])}")
    assert len(preds) == 1, "Should have 1 scale"
    assert len(preds[0]) == 1, "Should have 1 auxiliary level"
    merged_dict = preds[0][0]
    print(f"    Merged dict keys: {list(merged_dict.keys())}")
    assert "center" in merged_dict and "heatmap" in merged_dict, "Missing key predictions"
    print(f"    center shape: {merged_dict['center'].shape}, heatmap shape: {merged_dict['heatmap'].shape}")

    # Construct GT
    print("\n[4] Constructing ground-truth boxes...")
    gt_b = [
        torch.tensor([[0., 0., -1., 4., 2., 1.5, 0.3, 0., 0.],
                      [10., 5., -1., 2., 1., 1., 1.0, 0., 0.]], dtype=torch.float32),
        torch.zeros(0, 9, dtype=torch.float32)
    ]
    gt_l = [
        torch.tensor([0, 3], dtype=torch.long),
        torch.zeros(0, dtype=torch.long)
    ]
    print(f"    Batch 0: {gt_b[0].shape[0]} boxes, labels {gt_l[0].tolist()}")
    print(f"    Batch 1: {gt_b[1].shape[0]} boxes")

    # Encode/decode round-trip test
    print("\n[5] Testing encode/decode round-trip...")
    world_box = torch.tensor([[0., 0., -1., 4., 2., 1.5, 0.3, 0., 0.]], dtype=torch.float32)
    encoded = head.bbox_coder.encode(world_box)
    print(f"    Encoded shape: {encoded.shape}, values: {encoded[0].tolist()}")

    # Reconstruct via manual decode math
    coder = head.bbox_coder
    cx_f = encoded[0, 0].item()
    cy_f = encoded[0, 1].item()
    height_t = encoded[0, 2].item()
    log_w = encoded[0, 3].item()
    log_l = encoded[0, 4].item()
    log_h = encoded[0, 5].item()
    sin_yaw = encoded[0, 6].item()
    cos_yaw = encoded[0, 7].item()

    cx_world = cx_f * coder.out_size_factor * coder.voxel_size[0] + coder.pc_range[0]
    cy_world = cy_f * coder.out_size_factor * coder.voxel_size[1] + coder.pc_range[1]
    w_world = torch.tensor(log_w).exp().item()
    l_world = torch.tensor(log_l).exp().item()
    h_world = torch.tensor(log_h).exp().item()
    cz_world = height_t - h_world * 0.5
    yaw_world = torch.atan2(torch.tensor(sin_yaw), torch.tensor(cos_yaw)).item()

    recovered = torch.tensor([[cx_world, cy_world, cz_world, w_world, l_world, h_world, yaw_world, 0., 0.]])
    error = (recovered - world_box).abs().max().item()
    print(f"    Original: {world_box[0].tolist()}")
    print(f"    Recovered: {recovered[0].tolist()}")
    print(f"    Max error: {error:.6e}")
    assert error < 1e-4, f"Round-trip error too large: {error}"
    print("    Encode/decode round-trip OK")

    # Loss computation
    print("\n[6] Computing losses...")
    ld = head.loss(gt_b, gt_l, preds)
    print(f"    Loss dict keys: {list(ld.keys())}")
    loss_values = {k: float(v) for k, v in ld.items()}
    print(f"    Loss values: {loss_values}")
    for k, v in ld.items():
        assert torch.isfinite(v), f"{k} is not finite: {v}"
    print("    All losses are finite")

    # Backward pass
    print("\n[7] Testing backward pass...")
    total = sum(ld.values())
    total.backward()
    assert head.shared_conv.weight.grad is not None, "shared_conv.weight.grad is None"
    assert torch.isfinite(head.shared_conv.weight.grad).all(), "shared_conv.weight.grad contains NaN/Inf"
    print(f"    Gradient on shared_conv.weight: max={head.shared_conv.weight.grad.abs().max().item():.6e}")
    print("    Backward pass OK")

    print("\n" + "="*70)
    print("All self-tests PASSED!")
    print("="*70)
