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

Stage configs now only select hyperparameters. The hyperparameter dictionaries
live in `configs/sparsedrive_hyperparams.py`, while `SparseDrive` and its heads
construct their own backbone, neck, task heads, losses, samplers, and decoders.

The model expects the same tensor/data contract as the original SparseDrive
heads: `img` is shaped `[B, N_cam, 3, H, W]`, and metadata such as
`projection_mat`, `image_wh`, `timestamp`, `img_metas`, labels, boxes, and map
points are passed as keyword arguments to `model(img, **data)`.

Mixed precision is controlled by the caller with `torch.amp.autocast`; the
old MMCV `auto_fp16`/`force_fp32` decorators are not used.

## nuScenes K-means anchors

From the `sparsedrive_model` directory:

```bash
bash scripts/kmeans_nuscenes.sh
```

By default this reads `DATA_PATH=/Users/chenran/Code/nuscenes/nuscenes`,
uses `VERSION=v1.0-mini`, and writes SparseDrive-compatible K-means files to
`OUT_DIR=data/kmeans_nuscenes`. These files use the expected SparseDrive
K-means filenames and can be copied into an existing anchor directory or used by
pointing config `kmeans_dir` at the output directory.

Map anchors are saved in raw `LIDAR_TOP` `[x_right, y_forward]` order so
SparseDrive's camera projection and temporal map transforms use the same frame.

The mini split can have fewer detection/map samples than the full default
cluster counts, so full defaults may fail clearly when `K` exceeds available
samples. For a quick mini smoke validation, override the counts:

```bash
DET_K=4 MAP_K=4 MOTION_K=2 PLAN_K=2 MAP_NUM_SAMPLE=4 \
  OUT_DIR=outputs/kmeans_nuscenes_validation bash scripts/kmeans_nuscenes.sh
```

This smoke path was validated with output shapes: detection `(4, 11)`, map
`(4, 4, 2)`, motion `(10, 2, 12, 2)`, and plan `(3, 2, 6, 2)`. The script also
supports env overrides for future steps, thresholds, map sampling, and related
K-means parameters.

## nuScenes mini smoke test

From the repository root:

```bash
uv run --no-sync python sparsedrive_model/test_nuscenes_mini.py
```

The script intentionally has no CLI parser. Edit the constants at the top of
`test_nuscenes_mini.py` for dataset root, version, checkpoint, limit, device,
and output directory. By default it reads nuScenes from
`/Users/chenran/Code/nuscenes/nuscenes` and writes to
`sparsedrive_model/outputs/nuscenes_mini_inference`.
When `VISUALIZE=True`, visualization PNGs are written under
`sparsedrive_model/outputs/nuscenes_mini_inference/visualizations/`.
The visualizer overlays static GT map elements from dependency-free nuScenes
expansion JSON (`maps/expansion/{map_name}.json`) when available, so no extra
map-rendering dependencies are required. PNGs use conventional ego BEV display
coordinates (`x` forward, `y` left), with raw `LIDAR_TOP` data converted for
display; the validation sample rendered 76 polygons, 15 lines, and final future
path `[x_forward, y_left]=[24.744, 0.383]`.

## Checkpoints

The top-level attribute names are preserved:

- `img_backbone`
- `img_neck`
- `depth_branch`
- `head.det_head`
- `head.map_head`
- `head.motion_plan_head`

`head.det_head` is implemented by `Sparse4DDetHead`, and `head.map_head` is
implemented by `Sparse4DMap`. The old public `Sparse4DHead` API has been
removed; use the task-specific classes directly.

The FPN implementation keeps MMDetection-style `lateral_convs.*.conv` and
`fpn_convs.*.conv` child names so released SparseDrive checkpoints can be
loaded without changing neck keys. The ResNet trunk exposes `conv1`, `bn1`,
and `layer1` through `layer4` directly under `img_backbone`.

Raw SparseDrive checkpoints that store det/map decoder blocks as flat
`head.det_head.layers.*` and `head.map_head.layers.*` keys are converted by
`SparseDrive.load_state_dict()` to the split-head ModuleList keys. Already
converted checkpoints remain load-compatible.
