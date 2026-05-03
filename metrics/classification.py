from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def _resolve_eval_options(device: str, use_amp: bool, channels_last: bool, non_blocking: bool):
    device_obj = torch.device(device)
    resolved_amp = bool(use_amp and device_obj.type == "cuda" and torch.cuda.is_available())
    resolved_channels_last = bool(channels_last and device_obj.type == "cuda")
    resolved_non_blocking = bool(non_blocking and device_obj.type == "cuda")
    return device_obj, resolved_amp, resolved_channels_last, resolved_non_blocking


def _move_images_to_device(images, device_obj, non_blocking: bool, channels_last: bool):
    if channels_last and images.dim() == 4:
        return images.to(
            device=device_obj,
            non_blocking=non_blocking,
            memory_format=torch.channels_last,
        )
    return images.to(device=device_obj, non_blocking=non_blocking)


@torch.no_grad()
def evaluate_accuracy(
    model,
    data_loader,
    device: str,
    use_amp: bool = False,
    channels_last: bool = False,
    non_blocking: bool = False,
) -> float:
    device_obj, use_amp, channels_last, non_blocking = _resolve_eval_options(
        device=device,
        use_amp=use_amp,
        channels_last=channels_last,
        non_blocking=non_blocking,
    )
    model.eval()
    model.to(device_obj)
    if channels_last:
        # 评估阶段保持与训练相同的内存格式，避免每个 batch 触发额外格式转换。
        model.to(memory_format=torch.channels_last)
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = _move_images_to_device(images, device_obj, non_blocking, channels_last)
        labels = labels.to(device=device_obj, non_blocking=non_blocking)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    return float(correct / total) if total > 0 else float("nan")


@torch.no_grad()
def evaluate_asr(
    model,
    data_loader,
    device: str,
    use_amp: bool = False,
    channels_last: bool = False,
    non_blocking: bool = False,
) -> float:
    if data_loader is None:
        return float("nan")
    device_obj, use_amp, channels_last, non_blocking = _resolve_eval_options(
        device=device,
        use_amp=use_amp,
        channels_last=channels_last,
        non_blocking=non_blocking,
    )
    model.eval()
    model.to(device_obj)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = _move_images_to_device(images, device_obj, non_blocking, channels_last)
        labels = labels.to(device=device_obj, non_blocking=non_blocking)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    return float(correct / total) if total > 0 else float("nan")
