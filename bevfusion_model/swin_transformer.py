"""SwinTransformer backbone for BEVFusion camera encoder.

Pure PyTorch implementation matching the checkpoint structure from original
mmdet/timm implementations. No mmdet/mmcv imports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def window_partition(x, window_size):
    """Partition input into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size: int or tuple of int

    Returns:
        windows: (B*num_windows, window_size, window_size, C)
        (Hp, Wp): padded height and width
    """
    if isinstance(window_size, int):
        window_size = (window_size, window_size)

    B, H, W, C = x.shape
    pad_h = (window_size[0] - H % window_size[0]) % window_size[0]
    pad_w = (window_size[1] - W % window_size[1]) % window_size[1]

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(B, Hp // window_size[0], window_size[0], Wp // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = x.reshape(-1, window_size[0], window_size[1], C)

    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """Reverse operation of window_partition.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size: int or tuple
        pad_hw: (Hp, Wp)
        hw: (H, W)

    Returns:
        x: (B, H, W, C)
    """
    if isinstance(window_size, int):
        window_size = (window_size, window_size)

    H, W = hw
    Hp, Wp = pad_hw
    B = windows.shape[0] // ((Hp // window_size[0]) * (Wp // window_size[1]))
    C = windows.shape[-1]

    x = windows.reshape(
        B,
        Hp // window_size[0],
        Wp // window_size[1],
        window_size[0],
        window_size[1],
        C,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.reshape(B, Hp, Wp, C)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]

    return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, in_channels=3, embed_dim=256, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, H, W)

        Returns:
            x: (B, H*W, embed_dim)
            H, W: spatial dimensions of embedded features
        """
        x = self.projection(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class WindowAttention(nn.Module):
    """Window-based Multi-head Self Attention (W-MSA) module."""

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """Forward pass.

        Args:
            x: (num_windows*B, N, C) where N = window_size[0] * window_size[1]
            mask: attention mask

        Returns:
            x: (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias

        if mask is not None:
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FFN(nn.Module):
    """Feed Forward Network module with structure matching checkpoint."""

    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, hidden_dim)),
            nn.Linear(hidden_dim, embed_dim),
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0:
                x = F.gelu(x)
                x = self.drop(x)
        x = self.drop(x)
        return x


class ShiftWindowAttention(nn.Module):
    """Shifted Window Attention with W-MSA."""

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        shift_size=0,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
        self.shift_size = shift_size
        self.w_msa = WindowAttention(
            dim,
            window_size,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x, H, W):
        """Forward pass with window shifting.

        Args:
            x: (B, H*W, C)
            H, W: spatial dimensions

        Returns:
            x: (B, H*W, C)
        """
        B, L, C = x.shape
        x = x.reshape(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows, pad_hw = window_partition(x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size[0] * self.window_size[1], C)
        x_windows = self.w_msa(x_windows)
        x_windows = x_windows.reshape(-1, self.window_size[0], self.window_size[1], C)

        x = window_unpartition(x_windows, self.window_size, pad_hw, (H, W))

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.reshape(B, H * W, C)
        return x


class SwinBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = ShiftWindowAttention(
            dim,
            window_size,
            num_heads,
            shift_size=shift_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = FFN(dim, hidden_dim, dropout=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path1(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer (downsampling)."""

    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_channels)
        self.reduction = nn.Linear(4 * in_channels, 2 * in_channels, bias=False)

    def forward(self, x, H, W):
        """Forward pass.

        Args:
            x: (B, H*W, C)
            H, W: spatial dimensions

        Returns:
            x: (B, (H/2)*(W/2), 2*C)
            H//2, W//2: new spatial dimensions
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x, H // 2, W // 2


class SwinStage(nn.Module):
    """Swin Transformer Stage with multiple blocks and optional downsampling."""

    def __init__(
        self,
        embed_dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rates=None,
        downsample=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth

        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_rates[i],
                )
                for i in range(depth)
            ]
        )

        self.downsample = PatchMerging(embed_dim) if downsample else None

    def forward(self, x, H, W):
        """Forward pass.

        Args:
            x: (B, H*W, C)
            H, W: spatial dimensions

        Returns:
            x: (B, H'*W', C') where H', W' depend on downsampling
            H', W': new spatial dimensions
        """
        for block in self.blocks:
            x = block(x, H, W)

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)

        return x, H, W


class SwinTransformer(nn.Module):
    """Swin Transformer backbone."""

    def __init__(
        self,
        in_channels=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        patch_size=4,
        out_indices=(0, 1, 2),
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.out_indices = out_indices
        self.window_size = window_size

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur_channels = embed_dim
        block_idx = 0
        for i in range(len(depths)):
            depth = depths[i]
            num_head = num_heads[i]
            stage_drop_path_rates = dpr[block_idx : block_idx + depth]

            stage = SwinStage(
                embed_dim=cur_channels,
                depth=depth,
                num_heads=num_head,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rates=stage_drop_path_rates,
                downsample=(i < len(depths) - 1),
            )
            self.stages.append(stage)
            block_idx += depth
            if i < len(depths) - 1:
                cur_channels *= 2

        self.num_features = [embed_dim * 2**i for i in range(len(depths))]

        # Stage-level LayerNorms, one per out_index (named norm1, norm2, norm3...)
        # Applied after each stage that appears in out_indices.
        # Channel dim of stage i output = embed_dim * 2^i (post-downsample for stages < last)
        for j, i in enumerate(out_indices):
            norm = nn.LayerNorm(self.num_features[i + 1] if i < len(depths) - 1 else self.num_features[i])
            setattr(self, f"norm{j + 1}", norm)

    def init_weights(self):
        """Initialize weights. Typically called for random init; checkpoint loading overrides."""
        pass

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, H, W) batched images

        Returns:
            outs: list of feature maps at specified out_indices
        """
        x, H, W = self.patch_embed(x)
        x = self.drop(x)

        outs = []
        out_idx_order = 0
        for i, stage in enumerate(self.stages):
            x, H, W = stage(x, H, W)
            if i in self.out_indices:
                norm = getattr(self, f"norm{out_idx_order + 1}")
                x_normed = norm(x)
                B, _, C = x_normed.shape
                x_out = x_normed.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                outs.append(x_out)
                out_idx_order += 1

        return outs


class DropPath(nn.Module):
    """Stochastic Depth (DropPath) layer."""

    def __init__(self, drop_prob=0.0, training=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.training = training

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.bernoulli(
            torch.full(shape, keep_prob, device=x.device, dtype=x.dtype)
        )
        return x * random_tensor / keep_prob
