from __future__ import annotations

import math
import os
import random
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from torch.utils.data import Dataset


CAMERA_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)
CLASS_NAMES = ("vehicle",)


def _draw_rotated_box(mask, center_x, center_y, width, length, yaw, value):
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    half_width, half_length = width / 2, length / 2
    corners = []
    for x, y in (
        (-half_width, -half_length),
        (half_width, -half_length),
        (half_width, half_length),
        (-half_width, half_length),
    ):
        corners.append(
            [
                center_x + x * cos_yaw - y * sin_yaw,
                center_y + x * sin_yaw + y * cos_yaw,
            ]
        )
    points = np.round(np.asarray(corners)).astype(np.int32)
    cv2.fillPoly(mask, [points], value)


class NuScenesBEVDataset(Dataset):
    def __init__(
        self,
        *,
        dataroot: str,
        version: str = "v1.0-trainval",
        split: str = "train",
        img_size: Tuple[int, int] = (294, 518),
        bev_size: int = 128,
        bev_res: float = 0.5,
        augment: bool | None = None,
        return_visibility: bool = False,
    ):
        self.img_height, self.img_width = img_size
        self.bev_size = bev_size
        self.bev_res = bev_res
        self.augment = split == "train" if augment is None else augment
        self.return_visibility = return_visibility
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        split_name = f"mini_{split}" if "mini" in version else split
        scene_names = set(create_splits_scenes()[split_name])
        scenes = [scene for scene in self.nusc.scene if scene["name"] in scene_names]
        if not scenes and version == "v1.0-test" and split == "test":
            scenes = self.nusc.scene

        self.sample_tokens = []
        for scene in scenes:
            token = scene["first_sample_token"]
            while token:
                self.sample_tokens.append(token)
                token = self.nusc.get("sample", token)["next"]

    def __len__(self):
        return len(self.sample_tokens)

    def _load_cameras(self, sample):
        images, intrinsics, extrinsics = [], [], []
        for camera_name in CAMERA_NAMES:
            sample_data = self.nusc.get(
                "sample_data", sample["data"][camera_name]
            )
            image_path = os.path.join(self.nusc.dataroot, sample_data["filename"])
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(image_path)
            image = image[:, :, ::-1]
            original_height, original_width = image.shape[:2]
            image = cv2.resize(image, (self.img_width, self.img_height))
            image = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255

            if self.augment:
                if random.random() < 0.8:
                    brightness, contrast, saturation = [
                        1 + random.uniform(-0.3, 0.3) for _ in range(3)
                    ]
                    hue = random.uniform(-0.1, 0.1)
                    image = TF.adjust_brightness(image, brightness)
                    image = TF.adjust_contrast(image, contrast)
                    image = TF.adjust_saturation(image, saturation)
                    image = TF.adjust_hue(image, hue)
                if random.random() < 0.5:
                    image = (image + torch.randn_like(image) * 0.02).clamp(0, 1)

            calibration = self.nusc.get(
                "calibrated_sensor", sample_data["calibrated_sensor_token"]
            )
            camera_intrinsics = np.asarray(
                calibration["camera_intrinsic"], dtype=np.float32
            ).copy()
            scale_x = self.img_width / original_width
            scale_y = self.img_height / original_height
            camera_intrinsics[0, 0] *= scale_x
            camera_intrinsics[0, 2] *= scale_x
            camera_intrinsics[1, 1] *= scale_y
            camera_intrinsics[1, 2] *= scale_y

            transform = torch.eye(4, dtype=torch.float32)
            transform[:3, :3] = torch.from_numpy(
                Quaternion(calibration["rotation"]).rotation_matrix
            ).float()
            transform[:3, 3] = torch.tensor(
                calibration["translation"], dtype=torch.float32
            )

            images.append(image)
            intrinsics.append(torch.from_numpy(camera_intrinsics))
            extrinsics.append(transform)

        return (
            torch.stack(images),
            torch.stack(intrinsics),
            torch.stack(extrinsics),
        )

    def _build_targets(self, sample):
        segmentation = np.zeros(
            (self.bev_size, self.bev_size), dtype=np.uint8
        )
        visibility = np.full(
            (self.bev_size, self.bev_size), 255, dtype=np.uint8
        )

        lidar_data = self.nusc.get(
            "sample_data", sample["data"]["LIDAR_TOP"]
        )
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        ego_yaw = Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0]
        ego_rotation = Quaternion(
            scalar=np.cos(ego_yaw / 2),
            vector=[0, 0, np.sin(ego_yaw / 2)],
        ).rotation_matrix
        ego_translation = np.asarray(ego_pose["translation"])
        half_range = self.bev_size * self.bev_res / 2

        for annotation_token in sample["anns"]:
            annotation = self.nusc.get(
                "sample_annotation", annotation_token
            )
            if annotation["category_name"].split(".")[0] != "vehicle":
                continue

            position = ego_rotation.T @ (
                np.asarray(annotation["translation"]) - ego_translation
            )
            col = (half_range - position[1]) / self.bev_res
            row = (half_range - position[0]) / self.bev_res
            if not (0 <= col < self.bev_size and 0 <= row < self.bev_size):
                continue

            width, length, _ = annotation["size"]
            global_yaw = Quaternion(annotation["rotation"]).yaw_pitch_roll[0]
            box_yaw = math.pi - (global_yaw - ego_yaw)
            box_args = (
                col,
                row,
                width / self.bev_res,
                length / self.bev_res,
                box_yaw,
            )
            _draw_rotated_box(segmentation, *box_args, value=1)

            visibility_token = annotation.get("visibility_token", "0")
            try:
                visibility_level = int(visibility_token)
            except (TypeError, ValueError):
                visibility_level = 0
            _draw_rotated_box(
                visibility,
                *box_args,
                value=max(0, min(4, visibility_level)),
            )

        segmentation = torch.from_numpy(segmentation).float().unsqueeze(0)
        visibility = torch.from_numpy(visibility).long()
        return segmentation, visibility

    def __getitem__(self, index):
        sample = self.nusc.get("sample", self.sample_tokens[index])
        images, intrinsics, extrinsics = self._load_cameras(sample)
        segmentation, visibility = self._build_targets(sample)
        output = (images, intrinsics, extrinsics, segmentation)
        if self.return_visibility:
            output += (visibility,)
        return output
