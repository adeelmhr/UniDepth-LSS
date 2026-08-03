from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from PIL import Image
from pyquaternion import Quaternion


MAP_HASHES = {
    "singapore-onenorth": "53992ee3023e5494b90c316c183be829",
    "singapore-hollandvillage": "37819e65e09e5547b8a3ceaefba56bb2",
    "singapore-queenstown": "93406b464a165eaba6d9de76ca09f5da",
    "boston-seaport": "36092f0b03a857c6a3403e25b4b7aab3",
}

COLOR_BACKGROUND = np.array([247, 217, 166], dtype=np.float32) / 255
COLOR_ROAD = np.array([145, 145, 145], dtype=np.float32) / 255
COLOR_GT = np.array([43, 108, 246], dtype=np.float32) / 255
COLOR_PREDICTION = np.array([220, 53, 69], dtype=np.float32) / 255


def _sample_context(dataset, index):
    sample = dataset.nusc.get("sample", dataset.sample_tokens[index])
    scene = dataset.nusc.get("scene", sample["scene_token"])
    log = dataset.nusc.get("log", scene["log_token"])
    lidar = dataset.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = dataset.nusc.get("ego_pose", lidar["ego_pose_token"])
    return scene, log, ego_pose


def _extract_road_patch(data_root, log, ego_pose, bev_size, bev_resolution):
    map_path = Path(data_root) / "maps" / f"{MAP_HASHES[log['location']]}.png"
    semantic_map = np.asarray(Image.open(map_path))
    if semantic_map.ndim == 3:
        semantic_map = semantic_map[..., 0]
    semantic_map = semantic_map.max() - semantic_map

    pixels_per_meter = 10
    patch_pixels = int(bev_size * bev_resolution * pixels_per_meter)
    work_pixels = int(patch_pixels * 1.6)
    half = work_pixels // 2
    ego_x, ego_y = ego_pose["translation"][:2]
    center_col = int(round(ego_x * pixels_per_meter))
    center_row = int(round(semantic_map.shape[0] - ego_y * pixels_per_meter))

    padded = np.pad(semantic_map, half + 4, mode="constant")
    center_col += half + 4
    center_row += half + 4
    crop = padded[
        center_row - half:center_row + half,
        center_col - half:center_col + half,
    ]

    yaw = Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0]
    rotation = cv2.getRotationMatrix2D(
        (work_pixels / 2, work_pixels / 2),
        90 - np.degrees(yaw),
        1.0,
    )
    rotated = cv2.warpAffine(crop, rotation, (work_pixels, work_pixels))
    start = (work_pixels - patch_pixels) // 2
    patch = rotated[
        start:start + patch_pixels,
        start:start + patch_pixels,
    ]
    patch = cv2.resize(patch, (bev_size, bev_size)).astype(np.float32)
    if patch.max() > patch.min():
        patch = (patch - patch.min()) / (patch.max() - patch.min())
    return patch


def _make_canvas(road, bev_size):
    canvas = np.ones((bev_size, bev_size, 3), dtype=np.float32)
    canvas *= COLOR_BACKGROUND
    alpha = np.clip(road, 0, 1)[..., None] * 0.95
    return canvas * (1 - alpha) + COLOR_ROAD * alpha


def _color_mask(canvas, mask, color, alpha=1.0):
    alpha_map = mask.astype(np.float32)[..., None] * alpha
    return canvas * (1 - alpha_map) + color * alpha_map


def _draw_ego_vehicle(axis):
    axis.add_patch(
        Rectangle(
            (-1.0, -2.4),
            2.0,
            4.8,
            facecolor="white",
            edgecolor="#222222",
            linewidth=1.8,
            zorder=10,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (0, 0.2),
            (0, 4.2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.5,
            color="#222222",
            zorder=11,
        )
    )


def _style_bev_axis(axis, title, half_range):
    axis.set_title(title)
    axis.set(xlabel="lateral (m)", ylabel="forward (m)")
    axis.set_xlim(half_range, -half_range)
    axis.set_ylim(-half_range, half_range)
    axis.set_aspect("equal")
    axis.grid(color="white", alpha=0.25)
    _draw_ego_vehicle(axis)


@torch.inference_mode()
def visualize_random_sample(
    dataset,
    model,
    device,
    data_root,
    *,
    threshold=0.5,
    bev_resolution=0.5,
):
    sample_index = random.randrange(len(dataset))
    images, intrinsics, extrinsics, target = dataset[sample_index]
    logits = model(
        images.unsqueeze(0).to(device),
        intrinsics.unsqueeze(0).to(device),
        extrinsics.unsqueeze(0).to(device),
    )
    probability = logits.sigmoid()[0, 0].cpu().numpy()
    prediction = probability >= threshold
    ground_truth = target[0].numpy().astype(bool)

    bev_size = ground_truth.shape[0]
    half_range = bev_size * bev_resolution / 2
    bev_extent = [half_range, -half_range, -half_range, half_range]
    scene, log, ego_pose = _sample_context(dataset, sample_index)
    road = _extract_road_patch(
        data_root, log, ego_pose, bev_size, bev_resolution
    )

    base = _make_canvas(road, bev_size)
    ground_truth_panel = _color_mask(base, ground_truth, COLOR_GT)
    prediction_panel = _color_mask(base, prediction, COLOR_PREDICTION)
    overlay_panel = _color_mask(base, ground_truth, COLOR_GT, alpha=0.78)
    overlay_panel = _color_mask(
        overlay_panel, prediction, COLOR_PREDICTION, alpha=0.78
    )

    camera_order = [5, 0, 1, 4, 3, 2]
    camera_titles = [
        "Front Left",
        "Front",
        "Front Right",
        "Back Left",
        "Back",
        "Back Right",
    ]
    figure, axes = plt.subplots(
        3, 3, figsize=(16, 14), constrained_layout=True
    )
    for axis, camera_index, title in zip(
        axes[:2].flat, camera_order, camera_titles
    ):
        axis.imshow(images[camera_index].permute(1, 2, 0).numpy())
        axis.set_title(title)
        axis.axis("off")

    panels = [
        (ground_truth_panel, "Ground Truth"),
        (prediction_panel, f"Raw Prediction @ {threshold:.2f}"),
        (overlay_panel, "Ground Truth + Raw Prediction"),
    ]
    for axis, (panel, title) in zip(axes[2], panels):
        axis.imshow(
            panel,
            origin="upper",
            extent=bev_extent,
            interpolation="nearest",
        )
        _style_bev_axis(axis, title, half_range)

    axes[2, 2].legend(
        handles=[
            Patch(color=COLOR_ROAD, label="Road"),
            Patch(color=COLOR_GT, label="Ground truth"),
            Patch(color=COLOR_PREDICTION, label="Prediction"),
            Patch(
                facecolor="white",
                edgecolor="#222222",
                label="Ego vehicle",
            ),
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )
    figure.suptitle(
        f"UniDepthLSS — {scene['name']} / sample {sample_index}",
        fontsize=16,
    )
    plt.show()
    return figure, sample_index
