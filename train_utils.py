from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def segmentation_loss(logits, target, bce_weight=1.0, dice_weight=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    probability = logits.sigmoid()
    intersection = (probability * target).sum()
    union = probability.sum() + target.sum()
    dice = 1 - (2 * intersection + 1) / (union + 1)
    return bce_weight * bce + dice_weight * dice


class BinaryIoU:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    @torch.no_grad()
    def update(self, logits, target, valid_mask=None):
        prediction = logits.sigmoid() >= self.threshold
        target = target.bool()
        if valid_mask is not None:
            if valid_mask.ndim == target.ndim - 1:
                valid_mask = valid_mask.unsqueeze(1)
            valid_mask = valid_mask.expand_as(target)
            prediction = prediction[valid_mask]
            target = target[valid_mask]
        self.true_positive += int((prediction & target).sum().item())
        self.false_positive += int((prediction & ~target).sum().item())
        self.false_negative += int((~prediction & target).sum().item())

    def compute(self):
        union = self.true_positive + self.false_positive + self.false_negative
        return self.true_positive / union if union else 1.0


def run_epoch(
    model,
    loader,
    device,
    *,
    optimizer=None,
    scaler=None,
    grad_clip=1.0,
    threshold=0.5,
):
    training = optimizer is not None
    model.train(training)
    metric = BinaryIoU(threshold)
    loss_sum = 0.0
    sample_count = 0
    amp_enabled = scaler is not None and device.type == "cuda"

    for batch in tqdm(loader, desc="train" if training else "validation"):
        images, intrinsics, extrinsics, target = [
            value.to(device, non_blocking=True) for value in batch[:4]
        ]
        if training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if training else torch.no_grad()
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with grad_context, amp_context:
            logits = model(images, intrinsics, extrinsics)
            loss = segmentation_loss(logits, target)

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        metric.update(logits.detach(), target)

    return {
        "loss": loss_sum / max(sample_count, 1),
        "iou": metric.compute(),
    }
