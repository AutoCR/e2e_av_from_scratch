import torch
import torch.nn.functional as F


def multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights):
    """Small PyTorch fallback matching MMCV's function signature.

    This is intended for CPU/import smoke tests. Production inference should
    use a compiled multi-scale deformable attention op for performance/parity.
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([int(h * w) for h, w in spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(spatial_shapes):
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, int(h), int(w))
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l = F.grid_sample(value_l, grid_l, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampling_value_l)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1)
    return output.view(bs, num_heads * embed_dims, num_queries).transpose(1, 2).contiguous()


def nms_bev(boxes, scores, thresh=0.5):
    return torch.argsort(scores, descending=True)
