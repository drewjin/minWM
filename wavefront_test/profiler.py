from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


class CudaTimer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and torch.cuda.is_available()
        self._active: dict[str, torch.cuda.Event] = {}
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._active[name] = event

    def stop(self, name: str) -> None:
        if not self.enabled:
            return
        start = self._active.pop(name)
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._events[name].append((start, end))

    def summary_seconds(self) -> dict[str, float]:
        # 小段计时用 CUDA event，统一在汇总时 synchronize，避免每个阶段都全局同步。
        if not self.enabled:
            return {}
        torch.cuda.synchronize()
        return {
            name: sum(start.elapsed_time(end) for start, end in pairs) / 1000.0
            for name, pairs in self._events.items()
        }


@dataclass
class BenchMetrics:
    rank: int
    mode: str
    cuda: CudaTimer = field(default_factory=CudaTimer)
    values: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, name: str, value: float) -> None:
        self.values[name] += float(value)

    def inc(self, name: str, value: int = 1) -> None:
        self.values[name] += float(value)

    def start_cuda(self, name: str) -> None:
        self.cuda.start(name)

    def stop_cuda(self, name: str) -> None:
        self.cuda.stop(name)

    def summary(self) -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {
            "rank": self.rank,
            "mode": self.mode,
            **dict(self.values),
        }
        for name, seconds in self.cuda.summary_seconds().items():
            out[f"{name}_sec"] = seconds
        return out


def gather_summaries(local: dict[str, float | int | str]) -> list[dict[str, float | int | str]]:
    gathered: list[dict[str, float | int | str] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    return [x for x in gathered if x is not None]
