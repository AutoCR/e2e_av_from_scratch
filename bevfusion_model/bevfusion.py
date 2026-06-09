"""BEVFusion camera+LiDAR detection model.

Main module orchestrating:
- Camera encoder (SwinTransformer + LSSFPN + view transform)
- LiDAR encoder (voxelization + sparse convolution)
- Fuser (multi-modal feature fusion)
- Decoder (SECOND backbone + SECONDFPN neck)
- Detection head (TransFusion for 3D object detection)

All components use pure PyTorch (no mmdet/mmcv).
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .swin_transformer import SwinTransformer
from .camera_encoder import GeneralizedLSSFPN
from .view_transform import DepthLSSTransform
from .lidar_encoder import HardVoxelization, SparseEncoder, SPCONV_AVAILABLE
from .spconv_mac import TorchSparseEncoder
from .conv_fuser import ConvFuser
from .decoder import SECONDBackbone, SECONDNeck
from .detection_head import TransFusionHead


class BEVFusion(nn.Module):
    """BEVFusion camera+LiDAR 3D detection model.

    Combines camera and LiDAR modalities via BEV-space fusion for object detection.

    Args:
        hyperparams (dict): Configuration dict with keys:
            - camera_encoder: camera backbone/neck/vtransform configs
            - lidar_encoder: lidar voxelization and backbone configs
            - fuser: fusion layer config
            - decoder: BEV decoder config
            - detection_head: detection head config
    """

    def __init__(self, hyperparams: dict):
        super().__init__()
        hp = hyperparams

        # ── Camera encoder ──────────────────────────────────────────────
        cam_hp = hp["camera_encoder"]
        backbone = SwinTransformer(
            embed_dim=cam_hp["backbone"]["embed_dim"],
            depths=cam_hp["backbone"]["depths"],
            num_heads=cam_hp["backbone"]["num_heads"],
            window_size=cam_hp["backbone"]["window_size"],
            out_indices=cam_hp["backbone"]["out_indices"],
        )
        neck = GeneralizedLSSFPN(
            in_channels=cam_hp["neck"]["in_channels"],
            out_channels=cam_hp["neck"]["out_channels"],
            num_outs=cam_hp["neck"]["num_outs"],
            start_level=cam_hp["neck"].get("start_level", 0),
        )
        vtransform = DepthLSSTransform(
            in_channels=cam_hp["vtransform"]["in_channels"],
            out_channels=cam_hp["vtransform"]["out_channels"],
            image_size=cam_hp["vtransform"]["image_size"],
            feature_size=cam_hp["vtransform"]["feature_size"],
            xbound=cam_hp["vtransform"]["xbound"],
            ybound=cam_hp["vtransform"]["ybound"],
            zbound=cam_hp["vtransform"]["zbound"],
            dbound=cam_hp["vtransform"]["dbound"],
            downsample=cam_hp["vtransform"].get("downsample", 2),
        )

        # ── LiDAR encoder ───────────────────────────────────────────────
        lidar_hp = hp["lidar_encoder"]
        voxelizer = HardVoxelization(
            voxel_size=lidar_hp["voxelize"]["voxel_size"],
            point_cloud_range=lidar_hp["voxelize"]["point_cloud_range"],
            max_num_points=lidar_hp["voxelize"]["max_num_points"],
            max_voxels=lidar_hp["voxelize"]["max_voxels"],
        )
        if SPCONV_AVAILABLE:
            lidar_backbone = SparseEncoder(
                in_channels=lidar_hp["backbone"]["in_channels"],
                sparse_shape=lidar_hp["backbone"]["sparse_shape"],
                output_channels=lidar_hp["backbone"]["output_channels"],
                encoder_channels=lidar_hp["backbone"]["encoder_channels"],
                encoder_paddings=lidar_hp["backbone"]["encoder_paddings"],
                block_type=lidar_hp["backbone"].get("block_type", "conv_module"),
            )
        else:
            lidar_backbone = TorchSparseEncoder(
                in_channels=lidar_hp["backbone"]["in_channels"],
                sparse_shape=lidar_hp["backbone"]["sparse_shape"],
                output_channels=lidar_hp["backbone"]["output_channels"],
                encoder_channels=lidar_hp["backbone"]["encoder_channels"],
                encoder_paddings=lidar_hp["backbone"]["encoder_paddings"],
                block_type=lidar_hp["backbone"].get("block_type", "conv_module"),
            )

        self._lidar_bev_channels = 256
        self._lidar_bev_h = 256
        self._lidar_bev_w = 256

        # ── Build encoders as nn.ModuleDict (MUST match checkpoint keys) ─
        self.encoders = nn.ModuleDict()
        self.encoders["camera"] = nn.ModuleDict({
            "backbone": backbone,
            "neck": neck,
            "vtransform": vtransform,
        })
        lidar_modules = {"voxelize": voxelizer}
        if lidar_backbone is not None:
            lidar_modules["backbone"] = lidar_backbone
        self.encoders["lidar"] = nn.ModuleDict(lidar_modules)
        self.voxelize_reduce = lidar_hp.get("voxelize_reduce", True)

        # ── Fuser ────────────────────────────────────────────────────────
        fuser_hp = hp["fuser"]
        self.fuser = ConvFuser(
            in_channels=fuser_hp["in_channels"],
            out_channels=fuser_hp["out_channels"],
        )

        # ── Decoder ──────────────────────────────────────────────────────
        dec_hp = hp["decoder"]
        self.decoder = nn.ModuleDict({
            "backbone": SECONDBackbone(
                in_channels=dec_hp["in_channels"],
                out_channels=dec_hp["backbone_out_channels"],
                layer_nums=dec_hp["backbone_layer_nums"],
                layer_strides=dec_hp["backbone_layer_strides"],
            ),
            "neck": SECONDNeck(
                in_channels=dec_hp["backbone_out_channels"],
                out_channels=dec_hp["neck_out_channels"],
                upsample_strides=dec_hp["neck_upsample_strides"],
            ),
        })

        # ── Detection head ───────────────────────────────────────────────
        head_hp = hp["detection_head"]
        self.heads = nn.ModuleDict()
        self.heads["object"] = TransFusionHead(
            num_proposals=head_hp.get("num_proposals", 200),
            in_channels=head_hp.get("in_channels", 512),
            hidden_channel=head_hp.get("hidden_channel", 128),
            num_classes=head_hp.get("num_classes", 10),
            num_decoder_layers=head_hp.get("num_decoder_layers", 1),
            num_heads=head_hp.get("num_heads", 8),
            nms_kernel_size=head_hp.get("nms_kernel_size", 1),
            ffn_channel=head_hp.get("ffn_channel", 256),
            dropout=head_hp.get("dropout", 0.1),
            bn_momentum=head_hp.get("bn_momentum", 0.1),
            activation=head_hp.get("activation", "relu"),
            common_heads=head_hp.get("common_heads", None),
            test_cfg=head_hp.get("test_cfg", None),
            train_cfg=head_hp.get("train_cfg", None),
        )

    def extract_camera_features(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas=None,
    ):
        """Extract camera BEV features via view transform.

        Args:
            img: (B, N, C, H, W) multi-camera images
            points: list of (P_i, C) lidar point clouds
            camera2ego: (B, N, 4, 4) camera to ego extrinsics
            lidar2ego: (B, 4, 4) lidar to ego transform
            lidar2camera: (B, N, 4, 4) lidar to camera transform
            lidar2image: (B, N, 4, 4) lidar to image projection
            camera_intrinsics: (B, N, 3, 4) or (B, N, 4, 4) camera intrinsics
            camera2lidar: (B, N, 4, 4) camera to lidar transform
            img_aug_matrix: (B, N, 4, 4) image augmentation
            lidar_aug_matrix: (B, 4, 4) lidar augmentation
            metas: Optional metadata

        Returns:
            torch.Tensor: (B, C, H, W) camera BEV features
        """
        B, N, C, H, W = img.shape
        x = img.view(B * N, C, H, W)

        # Backbone
        x = self.encoders["camera"]["backbone"](x)

        # Neck
        x = self.encoders["camera"]["neck"](x)
        if isinstance(x, (list, tuple)):
            x = x[0]

        BN, C_out, fH, fW = x.shape
        x = x.view(B, N, C_out, fH, fW)

        # View transform: camera BEV features
        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
        )
        return x

    def voxelize(self, points):
        """Voxelize a batch of point clouds.

        Args:
            points: List of (P_i, C) point cloud tensors

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - feats: (num_voxels, C) voxel features
                - coords: (num_voxels, 4) voxel coordinates (batch_idx, x, y, z)
                - sizes: (num_voxels,) number of points per voxel
        """
        feats_list, coords_list, sizes_list = [], [], []
        for k, pts in enumerate(points):
            f, c, n = self.encoders["lidar"]["voxelize"](pts)
            c = c.clone()
            c[:, 0] = k
            feats_list.append(f)
            coords_list.append(c)
            sizes_list.append(n)

        feats = torch.cat(feats_list, dim=0)
        coords = torch.cat(coords_list, dim=0)
        sizes = torch.cat(sizes_list, dim=0)

        if self.voxelize_reduce:
            feats = feats.sum(dim=1) / sizes.float().view(-1, 1).clamp(min=1e-8)
            feats = feats.contiguous()

        return feats, coords, sizes

    def extract_lidar_features(self, points, camera_bev_shape=None):
        """Extract LiDAR BEV features via sparse convolution.

        Uses the pure-PyTorch fallback backbone when spconv is unavailable.
        """
        feats, coords, sizes = self.voxelize(points)
        batch_size = len(points)
        return self.encoders["lidar"]["backbone"](feats, coords, batch_size)

    def _extract_bev(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas=None,
    ):
        """Shared encoder→fuser→decoder path producing the BEV feature map.

        Used by both the training (grad-enabled) and inference (no_grad) forward
        branches so the two paths cannot drift.

        Returns:
            torch.Tensor: (B, C, H, W) decoded BEV features fed to the head.
        """
        # Camera features → BEV
        camera_bev = self.extract_camera_features(
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
        )

        # LiDAR features → BEV (pass camera_bev shape for fallback sizing)
        lidar_bev = self.extract_lidar_features(points, camera_bev_shape=camera_bev.shape)

        # Fuse camera and LiDAR BEV features
        bev = self.fuser([camera_bev, lidar_bev])

        # Decode BEV features via backbone and neck
        bev = self.decoder["backbone"](bev)
        bev = self.decoder["neck"](bev)
        if isinstance(bev, (list, tuple)):
            bev = bev[0]
        return bev

    def forward(self, *args, **kwargs):
        """Dispatch to the training-loss path or the inference path.

        DDP only synchronizes gradients through ``forward``, so the training
        loss must be reached via ``forward`` (not a separate method). Inference
        stays under ``torch.no_grad`` in ``_forward_test``.
        """
        if self.training:
            return self._forward_train(*args, **kwargs)
        return self._forward_test(*args, **kwargs)

    def _forward_train(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        gt_bboxes_3d,
        gt_labels_3d,
        metas=None,
        **kwargs,
    ):
        """Grad-enabled training forward returning the detection loss dict.

        Args:
            gt_bboxes_3d: list of (G_i, 9) [cx,cy,cz,w,l,h,yaw,vx,vy] lidar-frame GT boxes.
            gt_labels_3d: list of (G_i,) long class labels in the head's class space.
            (other args identical to the inference forward.)

        Returns:
            dict[str, torch.Tensor]: {loss_cls, loss_bbox, loss_heatmap}.
        """
        bev = self._extract_bev(
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
        )
        pred_dicts = self.heads["object"](bev, metas)
        return self.heads["object"].loss(gt_bboxes_3d, gt_labels_3d, pred_dicts)

    @torch.no_grad()
    def _forward_test(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas=None,
        **kwargs,
    ):
        """Inference forward for 3D object detection.

        Args:
            img: (B, N, 3, H, W) multi-camera images
            points: list of (P_i, 5) lidar point clouds per sample
            camera2ego: (B, N, 4, 4) camera to ego extrinsics
            lidar2ego: (B, 4, 4) lidar to ego transform
            lidar2camera: (B, N, 4, 4) lidar to camera transform
            lidar2image: (B, N, 4, 4) lidar to image projection
            camera_intrinsics: (B, N, 3, 4) or (B, N, 4, 4) camera intrinsics
            camera2lidar: (B, N, 4, 4) camera to lidar transform
            img_aug_matrix: (B, N, 4, 4) image augmentation (identity at test)
            lidar_aug_matrix: (B, 4, 4) lidar augmentation (identity at test)
            metas: list of metadata dicts

        Returns:
            list of dicts, one per sample with keys:
                - "boxes_3d": (N, 10) 3D bounding boxes
                - "scores_3d": (N,) detection scores
                - "labels_3d": (N,) class labels
        """
        bev = self._extract_bev(
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
        )

        batch_size = img.shape[0]
        pred_dicts = self.heads["object"](bev, metas)
        outputs_raw = self.heads["object"].get_bboxes(pred_dicts, metas)

        outputs = [{} for _ in range(batch_size)]
        for k, result in enumerate(outputs_raw):
            outputs[k] = {
                "boxes_3d": result["boxes_3d"].cpu(),
                "scores_3d": result["scores_3d"].cpu(),
                "labels_3d": result["labels_3d"].cpu(),
            }
        return outputs

    def init_weights(self):
        """Initialize weights (no-op when loading from checkpoint)."""
        pass
