from __future__ import annotations

from zlib import crc32

import torch


def build_orthogonal_projection(
    input_dim: int,
    output_dim: int,
    seed: int,
) -> torch.Tensor:
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    effective_dim = min(int(output_dim), int(input_dim))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_matrix = torch.randn(input_dim, effective_dim, generator=generator, dtype=torch.float32)
    q_matrix, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return q_matrix[:, :effective_dim].contiguous()


def stable_string_seed(value: str, seed: int) -> int:
    """为投影流式路径生成稳定子种子，避免依赖 Python 进程内置 hash。"""
    return int(crc32(str(value).encode("utf-8"), int(seed) & 0xFFFFFFFF))


def build_hash_projection_plan(
    chunk_start: int,
    chunk_size: int,
    output_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    为一段连续特征生成 CountSketch 风格的桶映射与符号。

    这里不再显式构造 `(input_dim, output_dim)` 稠密投影矩阵，
    而是按块生成“每个输入维度映射到哪个输出桶、带什么符号”，
    这样大模型也能流式做随机投影。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")

    positions = torch.arange(
        int(chunk_start),
        int(chunk_start) + int(chunk_size),
        dtype=torch.int64,
    )
    bucket_seed = int(seed) + 17
    sign_seed = int(seed) + 53
    bucket_indices = ((positions * 1103515245 + bucket_seed) % int(output_dim)).to(torch.int64)
    sign_bits = ((positions * 214013 + sign_seed) % 2).to(torch.float32)
    signs = sign_bits.mul(2.0).sub_(1.0)
    return bucket_indices, signs
