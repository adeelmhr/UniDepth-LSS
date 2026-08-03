from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF

try:
    from torch_scatter import scatter_add
except ImportError:
    def scatter_add(src, index, *, dim=0, out=None, dim_size=None):
        if out is None:
            size = list(src.shape)
            size[dim] = dim_size or int(index.max()) + 1
            out = src.new_zeros(size)
        return out.index_add_(dim, index, src)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _position_encoding(height: int, width: int, channels: int, device):
    half = channels // 2
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).float()
    divisor = torch.exp(
        torch.arange(0, half, 2, device=device)
        * (-math.log(10000.0) / half)
    )
    encoding = []
    for axis in range(2):
        position = grid[..., axis].reshape(1, -1, 1)
        sin_cos = torch.stack(
            (torch.sin(position * divisor), torch.cos(position * divisor)),
            dim=3,
        ).flatten(2)
        encoding.append(sin_cos)
    encoding = torch.cat(encoding, dim=2)
    if encoding.shape[-1] < channels:
        encoding = F.pad(encoding, (0, channels - encoding.shape[-1]))
    return encoding[0]


class _LiftSplatProjector(nn.Module):
    def __init__(
        self,
        *,
        img_height: int = 294,
        img_width: int = 518,
        bev_height: int = 128,
        bev_width: int = 128,
        voxel_height: int = 8,
        voxel_size: Tuple[float, float, float] = (0.5, 0.5, 1.0),
        voxel_origin: Tuple[float, float, float] = (-32.0, -32.0, -4.0),
        feature_channels: int = 128,
        use_transformer: bool = True,
    ):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.voxel_height = voxel_height
        self.feature_channels = feature_channels
        self.register_buffer("dx", torch.tensor(voxel_size).view(1, 1, 3))
        self.register_buffer("bx", torch.tensor(voxel_origin).view(1, 1, 3))
        self.depth_min = 1.0
        self.depth_max = 50.0

        from unidepth.models.unidepthv2 import UniDepthV2

        self.backbone = UniDepthV2.from_pretrained(
            "lpiccinelli/unidepth-v2-vitl14"
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.feat_adapter = nn.Sequential(
            nn.Linear(1024, feature_channels),
            nn.GELU(),
            nn.Linear(feature_channels, feature_channels),
        )

        self.use_transformer = use_transformer
        if use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=feature_channels,
                nhead=4,
                dropout=0.1,
                batch_first=True,
            )
            self.bev_trans = nn.TransformerEncoder(layer, num_layers=2)

        self._frustum_cache = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _make_frustum(self, device, feature_height: int, feature_width: int):
        key = (device, feature_height, feature_width)
        if key not in self._frustum_cache:
            scale_x = self.img_width / feature_width
            scale_y = self.img_height / feature_height
            x = (torch.arange(feature_width, device=device) + 0.5) * scale_x
            y = (torch.arange(feature_height, device=device) + 0.5) * scale_y
            x, y = torch.meshgrid(x, y, indexing="xy")
            self._frustum_cache[key] = torch.stack(
                (x, y, torch.ones_like(x), torch.ones_like(x)), dim=-1
            ).reshape(-1, 4)
        return self._frustum_cache[key]

    def _voxel_pool(self, xyz, features, batch_size: int):
        cameras_per_batch, channels, point_count = features.shape
        dx = self.dx.to(xyz.device, xyz.dtype)
        bx = self.bx.to(xyz.device, xyz.dtype)

        half_x = self.bev_height * dx[0, 0, 0] / 2
        half_y = self.bev_width * dx[0, 0, 1] / 2
        row = ((half_x - xyz[..., 0]) / dx[0, 0, 0]).round().long()
        col = ((half_y - xyz[..., 1]) / dx[0, 0, 1]).round().long()
        height = ((xyz[..., 2] - bx[0, 0, 2]) / dx[0, 0, 2]).round().long()

        valid = (
            (row >= 0)
            & (row < self.bev_height)
            & (col >= 0)
            & (col < self.bev_width)
            & (height >= 0)
            & (height < self.voxel_height)
        )
        if not valid.any():
            return features.new_zeros(
                batch_size,
                channels,
                self.voxel_height,
                self.bev_height,
                self.bev_width,
            )

        pooled_features = features.permute(0, 2, 1)[valid]
        row, col, height = row[valid], col[valid], height[valid]
        batch = (
            torch.arange(cameras_per_batch, device=features.device)
            .view(-1, 1)
            .expand(-1, point_count)[valid]
            // (cameras_per_batch // batch_size)
        )
        linear_index = (
            ((batch * self.voxel_height + height) * self.bev_height + row)
            * self.bev_width
            + col
        )
        output_size = (
            batch_size * self.voxel_height * self.bev_height * self.bev_width
        )
        volume = scatter_add(
            pooled_features,
            linear_index,
            dim=0,
            dim_size=output_size,
        )
        counts = scatter_add(
            torch.ones_like(pooled_features[:, 0]),
            linear_index,
            dim=0,
            dim_size=output_size,
        ).clamp_min(1).unsqueeze(-1)
        volume = (volume / counts).view(
            batch_size,
            self.voxel_height,
            self.bev_height,
            self.bev_width,
            channels,
        )
        return volume.permute(0, 4, 1, 2, 3)

    def forward(self, images, intrinsics, extrinsics):
        batch_size, camera_count, _, height, width = images.shape
        flat_batch = batch_size * camera_count
        rgb = images.reshape(flat_batch, 3, height, width)
        rgb = TF.normalize(rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        with torch.no_grad():
            encoder_inputs, encoder_outputs = self.backbone.encode_decode(
                {"image": rgb, "camera": None}, image_metas=[]
            )

        depth_full = encoder_outputs["depth"]
        encoder_features = encoder_inputs["features"][-1]
        feature_height, feature_width = encoder_features.shape[1:3]

        features = self.feat_adapter(encoder_features).permute(0, 3, 1, 2)
        depth = F.interpolate(
            depth_full,
            size=(feature_height, feature_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).clamp(self.depth_min, self.depth_max)

        point_count = feature_height * feature_width
        frustum = self._make_frustum(
            images.device, feature_height, feature_width
        )
        flat_intrinsics = intrinsics.reshape(flat_batch, 3, 3)
        focal_x = flat_intrinsics[:, 0, 0]
        focal_y = flat_intrinsics[:, 1, 1]
        center_x = flat_intrinsics[:, 0, 2]
        center_y = flat_intrinsics[:, 1, 2]
        pixel_x = frustum[:, 0].expand(flat_batch, point_count)
        pixel_y = frustum[:, 1].expand(flat_batch, point_count)
        depth = depth.reshape(flat_batch, point_count)

        camera_xyz = torch.stack(
            (
                (pixel_x - center_x[:, None]) / focal_x[:, None] * depth,
                (pixel_y - center_y[:, None]) / focal_y[:, None] * depth,
                depth,
            ),
            dim=-1,
        )
        flat_extrinsics = extrinsics.reshape(flat_batch, 4, 4)
        ego_xyz = (
            torch.bmm(camera_xyz, flat_extrinsics[:, :3, :3].transpose(1, 2))
            + flat_extrinsics[:, :3, 3].unsqueeze(1)
        )

        bev = self._voxel_pool(
            ego_xyz,
            features.reshape(flat_batch, self.feature_channels, point_count),
            batch_size,
        ).max(dim=2).values

        if self.use_transformer:
            bev_height, bev_width = bev.shape[2:]
            position = _position_encoding(
                bev_height,
                bev_width,
                self.feature_channels,
                bev.device,
            )
            tokens = bev.flatten(2).transpose(1, 2) + position
            bev = self.bev_trans(tokens).transpose(1, 2).reshape(
                batch_size,
                self.feature_channels,
                bev_height,
                bev_width,
            )
        return bev


class _SegmentationHead(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, 1),
        )

    def forward(self, features):
        return self.net(features)


class UniDepthLSS(nn.Module):
    """UniDepth V2 features and metric depth projected into a BEV grid."""

    def __init__(
        self,
        *,
        img_height: int = 294,
        img_width: int = 518,
        num_classes: int = 1,
        feature_channels: int = 128,
        **projector_kwargs,
    ):
        super().__init__()
        self.projector = _LiftSplatProjector(
            img_height=img_height,
            img_width=img_width,
            feature_channels=feature_channels,
            **projector_kwargs,
        )
        self.seg = _SegmentationHead(feature_channels, num_classes)

    def initialize_head_bias(self, bias: float = -2.19):
        nn.init.constant_(self.seg.net[-1].bias, bias)

    def forward(self, images, intrinsics, extrinsics):
        return self.seg(self.projector(images, intrinsics, extrinsics))
