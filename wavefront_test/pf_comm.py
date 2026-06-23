from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class RecvSlot:
    src: int
    k: torch.Tensor
    v: torch.Tensor
    k_work: dist.Work
    v_work: dist.Work


class FrontierKVComm:
    def __init__(
        self,
        rank: int,
        world_size: int,
        n_tokens: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        direct_async: bool = False,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.shape = (n_tokens, n_heads, head_dim)
        self.dtype = dtype
        self.device = device
        self.direct_async = direct_async
        self._recv_slots: dict[int, list[RecvSlot]] = {}
        self._sent_layers: set[int] = set()
        self._send_work: list[tuple[dist.Work, torch.Tensor]] = []

    def older_sources(self) -> list[int]:
        # rank 越大表示越 old。当前 rank 只能 attend 自己和更 old 的 chunks。
        return list(range(self.rank + 1, self.world_size))

    def newer_dests(self) -> list[int]:
        # 本 rank 的 K/V 要发给所有更 new 的 ranks，供它们拼成长上下文。
        return list(range(0, self.rank))

    def warmup_p2p(self) -> None:
        # NCCL P2P lazy 初始化时，不同 peer 顺序容易导致首轮卡住。
        # 先用 1 元素 tensor 按统一 batch 建好通信器，后面的 async recv/send 更稳定。
        send = torch.zeros((1,), dtype=self.dtype, device=self.device)
        recvs = []
        ops = []
        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            recv = torch.empty_like(send)
            recvs.append(recv)
            ops.append(dist.P2POp(dist.irecv, recv, peer))
            ops.append(dist.P2POp(dist.isend, send, peer))
        works = dist.batch_isend_irecv(ops) if ops else []
        for work in works:
            work.wait()

    def post_recvs(self, layer_id: int) -> None:
        if layer_id in self._recv_slots:
            return
        slots = []
        if self.direct_async:
            # prefetch 模式：提前贴 recv，等待时只测已经在路上的 remote KV。
            # src 用 old -> new 的顺序，后面 assemble 时也按这个方向拼接。
            for src in range(self.world_size - 1, self.rank, -1):
                k = torch.empty(self.shape, dtype=self.dtype, device=self.device)
                v = torch.empty(self.shape, dtype=self.dtype, device=self.device)
                slots.append(
                    RecvSlot(
                        src=src,
                        k=k,
                        v=v,
                        k_work=dist.irecv(k, src=src),
                        v_work=dist.irecv(v, src=src),
                    )
                )
            self._recv_slots[layer_id] = slots
            return

        ops = []
        tensors = []
        for src in self.older_sources():
            k = torch.empty(self.shape, dtype=self.dtype, device=self.device)
            v = torch.empty(self.shape, dtype=self.dtype, device=self.device)
            ops.append(dist.P2POp(dist.irecv, k, src))
            ops.append(dist.P2POp(dist.irecv, v, src))
            tensors.append((src, k, v))
        works = dist.batch_isend_irecv(ops) if ops else []
        for idx, (src, k, v) in enumerate(tensors):
            slots.append(RecvSlot(src=src, k=k, v=v, k_work=works[2 * idx], v_work=works[2 * idx + 1]))
        self._recv_slots[layer_id] = slots

    def send_kv(self, layer_id: int, k: torch.Tensor, v: torch.Tensor) -> int:
        if layer_id in self._sent_layers:
            return 0
        self._sent_layers.add(layer_id)
        bytes_sent = 0
        k = k.contiguous()
        v = v.contiguous()
        if self.direct_async:
            # 保留 send tensor 引用直到 work 完成，避免底层还在读 buffer 时被释放。
            for dst in self.newer_dests():
                self._send_work.append((dist.isend(k, dst=dst), k))
                self._send_work.append((dist.isend(v, dst=dst), v))
                bytes_sent += k.numel() * k.element_size() + v.numel() * v.element_size()
            self._drain_completed_sends()
            return bytes_sent

        ops = []
        tensors = []
        for dst in self.newer_dests():
            ops.append(dist.P2POp(dist.isend, k, dst))
            ops.append(dist.P2POp(dist.isend, v, dst))
            tensors.extend([k, v])
            bytes_sent += k.numel() * k.element_size() + v.numel() * v.element_size()
        if ops:
            works = dist.batch_isend_irecv(ops)
            self._send_work.extend(zip(works, tensors))
        self._drain_completed_sends()
        return bytes_sent

    def exchange_blocking(
        self,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], int, int, float]:
        # blocking baseline：本层 recv 和 send 一起 batch 提交，语义清楚，也避免逐 peer 顺序死锁。
        if layer_id in self._sent_layers:
            raise RuntimeError(f"layer {layer_id} KV was already sent")
        self._sent_layers.add(layer_id)
        k = k.contiguous()
        v = v.contiguous()

        ops = []
        recv_tensors = []
        bytes_recv = 0
        for src in self.older_sources():
            k_buf = torch.empty(self.shape, dtype=self.dtype, device=self.device)
            v_buf = torch.empty(self.shape, dtype=self.dtype, device=self.device)
            ops.append(dist.P2POp(dist.irecv, k_buf, src))
            ops.append(dist.P2POp(dist.irecv, v_buf, src))
            recv_tensors.append((src, k_buf, v_buf))
            bytes_recv += k_buf.numel() * k_buf.element_size() + v_buf.numel() * v_buf.element_size()

        send_tensors = []
        bytes_sent = 0
        for dst in self.newer_dests():
            ops.append(dist.P2POp(dist.isend, k, dst))
            ops.append(dist.P2POp(dist.isend, v, dst))
            send_tensors.extend([k, v])
            bytes_sent += k.numel() * k.element_size() + v.numel() * v.element_size()

        start = time.perf_counter()
        works = dist.batch_isend_irecv(ops) if ops else []
        for work in works:
            work.wait()
        wait_sec = time.perf_counter() - start

        remote = {src: (k_buf, v_buf) for src, k_buf, v_buf in recv_tensors}
        return remote, bytes_sent, bytes_recv, wait_sec

    def wait_recvs(self, layer_id: int) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], int, float]:
        slots = self._recv_slots.pop(layer_id)
        start = time.perf_counter()
        bytes_recv = 0
        out = {}
        for slot in slots:
            slot.k_work.wait()
            slot.v_work.wait()
            out[slot.src] = (slot.k, slot.v)
            bytes_recv += slot.k.numel() * slot.k.element_size() + slot.v.numel() * slot.v.element_size()
        return out, bytes_recv, time.perf_counter() - start

    def assemble_chronological(
        self,
        remote: dict[int, tuple[torch.Tensor, torch.Tensor]],
        k_local: torch.Tensor,
        v_local: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # KV 拼接必须 old -> ... -> current，例如 rank0 是 [rank3, rank2, rank1, rank0]。
        k_list = []
        v_list = []
        for src in range(self.world_size - 1, self.rank, -1):
            k_src, v_src = remote[src]
            k_list.append(k_src)
            v_list.append(v_src)
        k_list.append(k_local)
        v_list.append(v_local)
        return k_list, v_list

    def wait_sends(self) -> None:
        for work, _tensor in self._send_work:
            work.wait()
        self._send_work.clear()

    def _drain_completed_sends(self) -> None:
        pending = []
        for work, tensor in self._send_work:
            if work.is_completed():
                work.wait()
            else:
                pending.append((work, tensor))
        self._send_work = pending
