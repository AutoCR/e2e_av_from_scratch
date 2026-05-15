# SparseDrive Standalone Model

This directory contains the SparseDrive model code extracted from the original
MMDetection3D plugin into a plain PyTorch package.

Runtime dependencies are `torch`, `timm`, `einops`, `numpy`, and `scipy`.
`flash-attn` is used when available; the attention wrapper falls back to
PyTorch scaled-dot-product attention when the extension is unavailable.

## Build

```bash
cd sparsedrive_model
python setup.py develop
```

The setup script builds `sparsedrive.ops.deformable_aggregation_ext` for the
optional deformable aggregation CUDA path.

## Usage

```python
from configs.sparsedrive_small_stage1 import build

model = build()
model.init_weights()
```

The model expects the same tensor/data contract as the original SparseDrive
heads: `img` is shaped `[B, N_cam, 3, H, W]`, and metadata such as
`projection_mat`, `image_wh`, `timestamp`, `img_metas`, labels, boxes, and map
points are passed as keyword arguments to `model(img, **data)`.

Mixed precision is controlled by the caller with `torch.amp.autocast`; the
old MMCV `auto_fp16`/`force_fp32` decorators are not used.

## Checkpoints

The top-level attribute names are preserved:

- `img_backbone`
- `img_neck`
- `depth_branch`
- `head.det_head`
- `head.map_head`
- `head.motion_plan_head`

The FPN implementation keeps MMDetection-style `lateral_convs.*.conv` and
`fpn_convs.*.conv` child names so released SparseDrive checkpoints can be
loaded without changing neck keys. The ResNet trunk exposes `conv1`, `bn1`,
and `layer1` through `layer4` directly under `img_backbone`.
