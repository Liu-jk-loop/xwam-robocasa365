"""Distributed sampler that drops only one global-batch tail per epoch."""

from __future__ import annotations

from collections.abc import Iterator, Sized

import torch
from torch.utils.data import Sampler


class EpochAlignedDistributedSampler(Sampler[int]):
    """Shuffle globally, truncate to complete optimizer batches, then shard by rank."""

    def __init__(
        self,
        dataset: Sized,
        *,
        num_replicas: int,
        rank: int,
        samples_per_rank: int,
        seed: int,
    ):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.samples_per_rank = int(samples_per_rank)
        self.seed = int(seed)
        self.epoch = 0
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"分布式采样 rank 非法：rank={self.rank}, replicas={self.num_replicas}"
            )
        if self.samples_per_rank <= 0:
            raise ValueError("samples_per_rank 必须为正整数")
        self.total_size = self.samples_per_rank * self.num_replicas
        if self.total_size > len(self.dataset):
            raise ValueError(
                "对齐后的全局采样数不能超过 Dataset："
                f"sampled={self.total_size}, dataset={len(self.dataset)}"
            )

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        global_indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        global_indices = global_indices[: self.total_size]
        rank_indices = global_indices[self.rank : self.total_size : self.num_replicas]
        if len(rank_indices) != self.samples_per_rank:
            raise RuntimeError(
                "分布式采样器内部长度不一致："
                f"rank={self.rank}, actual={len(rank_indices)}, "
                f"expected={self.samples_per_rank}"
            )
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.samples_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def provenance(self) -> dict[str, int | str]:
        return {
            "type": "epoch_aligned_distributed",
            "rank": self.rank,
            "num_replicas": self.num_replicas,
            "samples_per_rank": self.samples_per_rank,
            "total_size": self.total_size,
            "dropped_per_epoch": len(self.dataset) - self.total_size,
            "seed": self.seed,
        }
