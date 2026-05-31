"""自博弈样本的 Replay Buffer。

约定的样本结构是 :class:`selfplay.self_play.SelfPlaySample`：

- ``state``  : ``np.ndarray`` 形如 ``(4, 15, 15)``，dtype ``np.float32``
- ``pi``     : ``np.ndarray`` 形如 ``(225,)``，   dtype ``np.float32``
- ``z``      : ``float``

Buffer 行为：
- 容量满后丢最旧(``deque(maxlen=capacity)``)。
- ``sample(batch_size)`` 不放回随机抽样，返回堆叠后的
  ``(states, policies, values)``，shape 分别是 ``(B, 4, 15, 15)``、
  ``(B, 225)``、``(B, 1)``，dtype 全部 ``float32``。
- ``save`` / ``load`` 用 ``np.savez_compressed`` 持久化。
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .self_play import SelfPlaySample


_STATE_SHAPE = (4, 15, 15)
_POLICY_SHAPE = (225,)


class ReplayBuffer:
    """有限容量的样本池，支持随机抽样与磁盘持久化。"""

    def __init__(self, capacity: int = 50_000, seed: Optional[int] = None) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity 必须为正，得到 {capacity}")
        self.capacity = int(capacity)
        self._buffer: deque = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)

    # ---- 基本操作 ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, sample: SelfPlaySample) -> None:
        self._buffer.append(sample)

    def extend(self, samples: Iterable[SelfPlaySample]) -> None:
        for s in samples:
            self._buffer.append(s)

    def clear(self) -> None:
        self._buffer.clear()

    # ---- 采样 -------------------------------------------------------------
    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """无放回随机抽样，返回 ``(states, policies, values)``。"""
        n = len(self._buffer)
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正，得到 {batch_size}")
        if batch_size > n:
            raise ValueError(
                f"batch_size={batch_size} 大于 buffer 大小 {n}"
            )

        population = list(self._buffer)
        chosen: List[SelfPlaySample] = self._rng.sample(population, batch_size)
        return self._stack(chosen)

    # ---- 整体导入 / 导出 --------------------------------------------------
    def to_arrays(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """把当前 buffer 全部转成 ndarray 三元组。"""
        return self._stack(list(self._buffer))

    def from_arrays(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """把 ``(states, policies, values)`` 数组转回样本写入 buffer(会先 clear)。"""
        states = np.asarray(states, dtype=np.float32)
        policies = np.asarray(policies, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)

        if states.ndim != 4 or states.shape[1:] != _STATE_SHAPE:
            raise ValueError(f"states shape 应为 (N, 4, 15, 15)，实际 {states.shape}")
        if policies.ndim != 2 or policies.shape[1:] != _POLICY_SHAPE:
            raise ValueError(f"policies shape 应为 (N, 225)，实际 {policies.shape}")
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError(f"values shape 应为 (N,) 或 (N, 1)，实际 {values.shape}")
        if not (states.shape[0] == policies.shape[0] == values.shape[0]):
            raise ValueError(
                f"三种数组的样本数不一致: "
                f"states={states.shape[0]}, policies={policies.shape[0]}, values={values.shape[0]}"
            )

        self._buffer.clear()
        for i in range(states.shape[0]):
            sample = SelfPlaySample(
                state=np.array(states[i], dtype=np.float32, copy=True),
                pi=np.array(policies[i], dtype=np.float32, copy=True),
                z=float(values[i, 0]),
            )
            self._buffer.append(sample)

    # ---- 持久化 -----------------------------------------------------------
    def save(self, path: str) -> None:
        """保存到 ``.npz``(自动建父目录)。"""
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        states, policies, values = self.to_arrays()
        np.savez_compressed(path, states=states, policies=policies, values=values)

    def load(self, path: str) -> None:
        """从 ``.npz`` 还原(会先清空当前 buffer)。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"replay buffer 文件不存在: {path}")
        with np.load(path, allow_pickle=False) as data:
            self.from_arrays(
                data["states"], data["policies"], data["values"]
            )

    # ---- 内部工具 ---------------------------------------------------------
    @staticmethod
    def _stack(
        samples: List[SelfPlaySample],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(samples)
        if n == 0:
            return (
                np.zeros((0, *_STATE_SHAPE), dtype=np.float32),
                np.zeros((0, *_POLICY_SHAPE), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
            )
        states = np.stack([np.asarray(s.state, dtype=np.float32) for s in samples])
        policies = np.stack([np.asarray(s.pi, dtype=np.float32) for s in samples])
        values = np.array([[float(s.z)] for s in samples], dtype=np.float32)
        return states, policies, values


__all__ = ["ReplayBuffer"]
