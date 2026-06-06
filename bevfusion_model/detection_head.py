import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 test_cfg=None):
        super().__init__()

        self.num_proposals = num_proposals
        self.num_classes = num_classes
        self.auxiliary = auxiliary
        self.num_decoder_layers = num_decoder_layers
        self.nms_kernel_size = nms_kernel_size
        self.test_cfg = test_cfg or {}
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

        if self.test_cfg.get("dataset") == "nuScenes":
            local_max[:, 8] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg.get("dataset") == "Waymo":
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
