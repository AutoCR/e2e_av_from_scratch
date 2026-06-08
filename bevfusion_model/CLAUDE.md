# bevfusion_model

## Origin

Ported from `/Users/chenran/Code/bevfusion` (the official BEVFusion repo by MIT HAN Lab).
All mmdet/mmcv/mmdet3d dependencies and the registry mechanism have been removed.
The model runs on pure PyTorch and loads the official pretrained checkpoint unchanged.

Original paper: "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation" (MIT HAN Lab, 2022).

## What this implements

Full camera+LiDAR BEVFusion detection pipeline for nuScenes (10 classes):

```
Camera images (6×) → SwinTransformer → GeneralizedLSSFPN → DepthLSSTransform → camera BEV (80ch)
LiDAR points       → HardVoxelization → SparseEncoder/spconv_mac fallback → lidar BEV (256ch)
                        ↓
                   ConvFuser (cat → 256ch)
                        ↓
              SECONDBackbone + SECONDNeck → BEV features (512ch)
                        ↓
               TransFusionHead → 3D boxes (10 classes)
```

## File map

| File | Contents | Source in raw repo |
|------|----------|--------------------|
| `bevfusion.py` | Top-level `BEVFusion` module | `mmdet3d/models/detectors/bevfusion.py` |
| `swin_transformer.py` | `SwinTransformer` backbone (embed_dim=96, depths=[2,2,6,2]) | `mmdet3d/models/backbones/swin.py` |
| `camera_encoder.py` | `GeneralizedLSSFPN` neck (in=[192,384,768], out=256) | `mmdet3d/models/necks/generalized_lssfpn.py` |
| `view_transform.py` | `DepthLSSTransform` (D=118 depth bins, BEV 80ch, 128×128) | `mmdet3d/models/vtransforms/depth_lss.py` |
| `lidar_encoder.py` | `HardVoxelization` + `SparseEncoder` (spconv v2 path) | `mmdet3d/models/backbones/sparse_encoder.py` |
| `spconv_mac/` | Pure-PyTorch sparse convolution fallback used when `spconv` is unavailable | custom |
| `conv_fuser.py` | `ConvFuser` (cat camera+lidar BEV → 256ch) | `mmdet3d/models/fusers/conv_fuser.py` |
| `decoder.py` | `SECONDBackbone` + `SECONDNeck` | `mmdet3d/models/backbones/second.py` + neck |
| `detection_head.py` | `TransFusionHead`, `TransFusionBBoxCoder`, transformer layers | `mmdet3d/models/heads/bbox/transfusion.py` |
| `nuscenes_adapter.py` | Pure-JSON nuScenes loader (no nuscenes-devkit) | custom |
| `bev_pool/` | CUDA-accelerated BEV pooling op (JIT-compiled when CUDA available) | `mmdet3d/ops/bev_pool/` |
| `configs/bevfusion_hyperparams.py` | All hyperparameters in one dict file | extracted from yaml configs |
| `test_nuscenes_mini.py` | End-to-end inference + visualization test | custom |

## Checkpoint

`model_weights/bevfusion/bevfusion-det.pth` — official BEVFusion detection checkpoint.
Config it corresponds to: `swint_v0p075/convfuser.yaml` (SwinT-Tiny, voxel_size=0.075, TransFusionHead).

Load with `strict=False`. On platforms without `spconv`, the LiDAR backbone keys load into the pure-PyTorch `spconv_mac` fallback.

## Key architecture details

### SwinTransformer
- `embed_dim=96`, `depths=[2,2,6,2]`, `num_heads=[3,6,12,24]`, `out_indices=(0,1,2)`
- Stage-level LayerNorms named `norm1`, `norm2`, `norm3` (indexed by enumeration order over `out_indices`, not stage index)
- Output feature channels: `[192, 384, 768]`

### GeneralizedLSSFPN
- `lateral_convs[0]`: input = 192 + 256 = 448 (finest backbone level + top-down)
- `lateral_convs[1]`: input = 384 + 768 = 1152 (two backbone levels merged)
- Output: 256ch at the finest scale

### DepthLSSTransform
- D=118 depth bins (dbound=[2.0, 58.0, 0.5])
- Camera BEV: 80ch, 128×128
- Downsample=2 applied after scatter
- **BEV pooling**: `bev_pool_pure` scatters frustum features into the BEV grid.
  When a CUDA device is available it uses the JIT-compiled `bev_pool/` CUDA
  kernel (fused sort + interval-sum, the official `bev_pool_v2` path); otherwise
  it falls back to the pure-PyTorch `index_put_(accumulate=True)` scatter. Both
  paths are numerically equivalent (verified to ~1e-6). The fast path activates
  only when `BEV_POOL_CUDA_AVAILABLE and x.is_cuda`.

### SparseEncoder (LiDAR)
- Uses spconv v2 (`spconv.pytorch`)
- **Not available on macOS ARM64** — install `spconv-cu120` on Linux/CUDA
- macOS/no-spconv fallback: `bevfusion_model/spconv_mac/` implements a pure-PyTorch sparse convolution shim that loads the official LiDAR backbone checkpoint keys.
- The fallback is slower than CUDA `spconv`, but it produces nonzero LiDAR BEV features instead of the old zero tensor.

### ConvFuser
- Input: cat([camera_bev=80ch, lidar_bev=256ch]) → 336ch total
- Output: 256ch via Conv2d(336→256, k=3, pad=1) + BN + ReLU

### SECONDNeck
- `deblocks.0`: stride=1 → `nn.Conv2d` (NOT ConvTranspose2d), shape [256,128,1,1]
- `deblocks.1`: stride=2 → `nn.ConvTranspose2d`
- Output: cat of two deblocks = 512ch

### TransFusionHead
- `bev_pos`: pre-computed 2D grid (1, 180×180, 2) registered as a buffer (not in checkpoint — expected)
- `auxiliary=True`: forward concatenates all decoder-layer outputs along last dim; `get_bboxes` takes `[..., -num_proposals:]`
- Cross-attention VALUE = key + key_pos_embed (NOT just key — critical for correct predictions)
- Score = `sigmoid(decoder_heatmap) × query_heatmap_score × one_hot_class_mask` (triple product)
- BBoxCoder: center→world via `* out_size_factor * voxel_size + pc_range`; dim via `exp()`; height is bottom center (`height - dim_z/2`)
- Circle NMS for pedestrians (radius=0.175) and traffic cones (radius=0.175)

## nn.ModuleDict key structure (must match checkpoint)

```
encoders["camera"]["backbone"]  →  SwinTransformer
encoders["camera"]["neck"]      →  GeneralizedLSSFPN
encoders["camera"]["vtransform"]→  DepthLSSTransform
encoders["lidar"]["voxelize"]   →  HardVoxelization
encoders["lidar"]["backbone"]   →  SparseEncoder if spconv is available, TorchSparseEncoder otherwise
fuser                           →  ConvFuser
decoder["backbone"]             →  SECONDBackbone
decoder["neck"]                 →  SECONDNeck
heads["object"]                 →  TransFusionHead
```

## Running the test

```bash
uv run python bevfusion_model/test_nuscenes_mini.py
```

Required env / paths (hardcoded in test script):
- nuScenes mini: `/Users/chenran/Code/nuscenes/nuscenes`
- Checkpoint: `model_weights/bevfusion/bevfusion-det.pth`
- Output: `bevfusion_model/outputs/nuscenes_mini_inference/`

Expected output: 200 detections per sample, with high-confidence scores (>0.3, typically up to ~0.8 for clear objects) on both macOS (pure-PyTorch `spconv_mac`) and Linux (CUDA spconv). The `spconv_mac` fallback is numerically verified against the official spconv convention, so results match the CUDA path.

> Historical note: an earlier version produced max score ~0.12 and wrong boxes. The root cause was **LiDAR input loading**, not the sparse-conv fallback: (1) the 5th point channel was fed the raw nuScenes *ring* index (0–31) instead of the relative-timestamp slot the checkpoint expects (must be 0 for the keyframe); (2) only a single sweep was loaded instead of the 10-sweep motion-compensated aggregation (`LoadPointsFromMultiSweeps`, `sweeps_num=9`). Both are fixed in `nuscenes_adapter.py`.

## Known limitations

- **macOS ARM64**: spconv wheels don't exist for this platform. The pure-PyTorch `spconv_mac` fallback runs the LiDAR backbone with official weights, but it is slower than CUDA `spconv` and is intended for local debugging/inference rather than production performance.
- **No training code**: only the inference path is ported. Loss functions and Hungarian matching are not implemented.
- **No mmdet3d**: all geometry ops, NMS, and voxelization are re-implemented in pure PyTorch/numpy.
