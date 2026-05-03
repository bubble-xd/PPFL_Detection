from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


def configure_torch_runtime(config) -> None:
    """
    配置 PyTorch 运行时的硬件相关开关。

    这些开关只影响执行效率，不改变联邦学习算法本身；CUDA 不可用时会自动退化为 CPU 路径。
    """
    num_threads = getattr(config, "TORCH_NUM_THREADS", None)
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))

    interop_threads = getattr(config, "TORCH_NUM_INTEROP_THREADS", None)
    if interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(interop_threads))
        except RuntimeError:
            # 该接口只能在并行运行时初始化前设置；重复跑实验时保持已有设置即可。
            pass

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(getattr(config, "CUDNN_BENCHMARK", True))
        if bool(getattr(config, "ENABLE_TF32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")


def _resolve_pin_memory(device: str, pin_memory: Optional[bool]) -> bool:
    if pin_memory is not None:
        return bool(pin_memory)
    return str(device).startswith("cuda") and torch.cuda.is_available()


def build_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    effective_num_workers = max(0, int(num_workers))
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": effective_num_workers,
        "pin_memory": bool(pin_memory),
        "generator": generator,
    }
    if effective_num_workers > 0:
        # worker 常驻和预取只在多进程 DataLoader 下有效；
        # 对内存中的 TensorDataset 可以按机器情况调小或关掉。
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def _move_images_to_device(
    images: torch.Tensor,
    device: torch.device,
    non_blocking: bool,
    channels_last: bool,
) -> torch.Tensor:
    if channels_last and images.dim() == 4:
        return images.to(
            device=device,
            non_blocking=non_blocking,
            memory_format=torch.channels_last,
        )
    return images.to(device=device, non_blocking=non_blocking)


def train_local_model(
    model: nn.Module,
    dataset,
    device: str,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    seed: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    use_amp: bool = False,
    channels_last: bool = False,
    max_batches: Optional[int] = None,
) -> nn.Module:
    device_obj = torch.device(device)
    effective_pin_memory = _resolve_pin_memory(device, pin_memory)
    use_amp = bool(use_amp and device_obj.type == "cuda" and torch.cuda.is_available())
    channels_last = bool(channels_last and device_obj.type == "cuda")
    model.train()
    model.to(device_obj)
    if channels_last:
        # ResNet/VGG 这类卷积模型在 CUDA 上通常能从 channels_last 获益。
        model.to(memory_format=torch.channels_last)
    loader = build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    batches_processed = 0
    for _ in range(local_epochs):
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and int(max_batches) > 0 and batches_processed >= int(max_batches):
                break
            images = _move_images_to_device(
                images,
                device=device_obj,
                non_blocking=effective_pin_memory,
                channels_last=channels_last,
            )
            labels = labels.to(device=device_obj, non_blocking=effective_pin_memory)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batches_processed += 1
        if max_batches is not None and int(max_batches) > 0 and batches_processed >= int(max_batches):
            break

    return model
