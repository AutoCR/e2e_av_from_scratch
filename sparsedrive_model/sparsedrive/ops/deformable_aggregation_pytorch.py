import torch
import torch.nn.functional as F


def deformable_aggregation_pytorch(
    mc_ms_feat,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    bs, _, num_embeds = mc_ms_feat.shape
    num_cams, num_scale = spatial_shape.shape[:2]
    num_anchors, num_pts = sampling_location.shape[1:3]
    num_groups = weights.shape[5]
    cpg = num_embeds // num_groups  # channels per group

    output = torch.zeros(bs, num_anchors, num_embeds, dtype=mc_ms_feat.dtype, device=mc_ms_feat.device)

    for cam in range(num_cams):
        for scale in range(num_scale):
            h = int(spatial_shape[cam, scale, 0])
            w = int(spatial_shape[cam, scale, 1])
            start = int(scale_start_index[cam, scale])

            # [bs, num_embeds, h, w]
            feat = mc_ms_feat[:, start:start + h * w, :].permute(0, 2, 1).reshape(bs, num_embeds, h, w)

            # loc: [bs, num_anchors, num_pts, 2] — (w, h) normalized in (0, 1)
            loc = sampling_location[:, :, :, cam, :]
            # replicate CUDA boundary check: skip if loc <= 0 or loc >= 1
            valid = ((loc > 0) & (loc < 1)).all(-1, keepdim=True).to(mc_ms_feat.dtype)

            # grid_sample expects grid in [-1, 1]; align_corners=False maps
            # grid value g to pixel (g+1)/2 * size - 0.5, matching the CUDA
            # kernel's h_im = loc_h * h - 0.5
            grid = (2.0 * loc - 1.0).reshape(bs, num_anchors * num_pts, 1, 2)
            sampled = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            # [bs, num_embeds, num_anchors*num_pts, 1] -> [bs, num_anchors, num_pts, num_embeds]
            sampled = sampled.squeeze(-1).reshape(bs, num_embeds, num_anchors, num_pts).permute(0, 2, 3, 1)
            sampled = sampled * valid

            # expand group weights to per-channel: [bs, num_anchors, num_pts, num_embeds]
            w_cs = weights[:, :, :, cam, scale, :].repeat_interleave(cpg, dim=-1)
            output = output + (sampled * w_cs).sum(dim=2)

    return output
