"""Квантование и зашумление эмбеддингов (DESIGN.md §9, защиты 2–3).

Квантование симметричное, с масштабом на вектор (float16): int8 (уровни ±127), int4 (±7, две компоненты
в байте), binary (знак компоненты, 1 бит; дек­вантование ±1/√d). Память на вектор относительно float16
(2·d байт): ≈ 0.5, 0.25, 0.0625. Шум Гаусса: e + σ·n, n ~ N(0, I), затем L2-нормировка;
режим «relative» (по умолчанию) масштабирует σ на ‖e‖/√d, так что σ = ожидаемое отношение нормы шума
к норме вектора (cos ≈ 1/√(1+σ²)); режим «absolute» — σ на компоненту.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

BITS: tuple[int, ...] = (8, 4, 1)
_LEVELS = {8: 127, 4: 7}
SCALE_BYTES = 2  # масштаб на вектор хранится как float16


def _as_2d(emb: np.ndarray) -> np.ndarray:
    x = np.asarray(emb, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"emb must be 2-D, got shape {x.shape}")
    return x


def as_generator(rng: Any) -> np.random.Generator:
    """np.random.Generator из Generator | random.Random | int | None (детерминированно для Random/int)."""
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, random.Random):
        return np.random.default_rng(rng.getrandbits(64))
    if rng is None:
        return np.random.default_rng()
    return np.random.default_rng(int(rng))


# ----------------------------------------------------------------------------- упаковка int4


def pack_int4(q: np.ndarray) -> np.ndarray:
    """Коды int8 в [−8, 7] → uint8 (N, ⌈d/2⌉): старший полубайт — чётная компонента, младший — нечётная."""
    q = np.asarray(q, dtype=np.int8)
    n, d = q.shape
    if d % 2:
        q = np.concatenate([q, np.zeros((n, 1), dtype=np.int8)], axis=1)
    u = (q.astype(np.int16) + 8).astype(np.uint8)
    return ((u[:, 0::2] << 4) | u[:, 1::2]).astype(np.uint8)


def unpack_int4(packed: np.ndarray, d: int) -> np.ndarray:
    """Обратно к int8 (N, d)."""
    p = np.asarray(packed, dtype=np.uint8)
    hi = (p >> 4).astype(np.int16) - 8
    lo = (p & 0x0F).astype(np.int16) - 8
    q = np.empty((p.shape[0], p.shape[1] * 2), dtype=np.int8)
    q[:, 0::2] = hi
    q[:, 1::2] = lo
    return q[:, :d]


# ----------------------------------------------------------------------------- квантование


@dataclass
class Quantized:
    """Квантованные эмбеддинги: коды + масштаб на вектор + метаданные; dequantize() → float32 (N, d)."""

    codes: np.ndarray  # int8 (N, d) для 8 бит; uint8 (N, ⌈d/2⌉) для 4 бит; uint8 (N, ⌈d/8⌉) для 1 бита
    scale: np.ndarray  # float16 (N,) — для binary единицы
    bits: int
    dim: int

    @property
    def n(self) -> int:
        return int(self.codes.shape[0])

    @property
    def nbytes(self) -> int:
        """Байты хранения: коды + масштабы (для binary масштаб не хранится)."""
        return int(self.codes.nbytes) + (0 if self.bits == 1 else int(self.scale.nbytes))

    def dequantize(self) -> np.ndarray:
        return dequantize(self)


def bytes_per_vector(d: int, bits: int) -> int:
    """Байт на вектор при хранении: ⌈d·bits/8⌉ (+2 байта масштаба для 8/4 бит)."""
    if bits not in BITS:
        raise ValueError(f"bits must be one of {BITS}, got {bits}")
    body = -(-d * bits // 8)  # ceil
    return body + (0 if bits == 1 else SCALE_BYTES)


def memory_ratio(d: int, bits: int | None) -> float:
    """Память на вектор относительно float16 (2·d байт); None → 1.0 (без квантования)."""
    if bits is None:
        return 1.0
    return bytes_per_vector(d, bits) / float(2 * d)


def quantize_full(emb: np.ndarray, bits: int) -> Quantized:
    """Симметричное квантование с масштабом max|x|/levels на вектор (8/4 бит) или знаковое (1 бит)."""
    x = _as_2d(emb)
    n, d = x.shape
    if bits not in BITS:
        raise ValueError(f"bits must be one of {BITS}, got {bits}")
    if bits == 1:
        codes = np.packbits(x >= 0.0, axis=1)
        return Quantized(codes=codes, scale=np.ones(n, dtype=np.float16), bits=1, dim=d)
    levels = _LEVELS[bits]
    amax = np.max(np.abs(x), axis=1)
    amax = np.where(amax > 0, amax, 1.0)
    scale = (amax / levels).astype(np.float16)
    q = np.clip(np.rint(x / scale.astype(np.float32)[:, None]), -levels, levels).astype(np.int8)
    codes = pack_int4(q) if bits == 4 else q
    return Quantized(codes=codes, scale=scale, bits=bits, dim=d)


def dequantize(q: Quantized) -> np.ndarray:
    """float32 (N, d): коды·масштаб; для binary ±1/√d (единичная норма)."""
    if q.bits == 1:
        bits = np.unpackbits(q.codes, axis=1)[:, : q.dim].astype(np.float32)
        return ((bits * 2.0 - 1.0) / np.sqrt(float(q.dim))).astype(np.float32)
    codes = unpack_int4(q.codes, q.dim) if q.bits == 4 else q.codes
    return (codes.astype(np.float32) * q.scale.astype(np.float32)[:, None]).astype(np.float32)


def quantize(emb: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """(коды, деквантованные float32 (N, d)). Коды: int8 (N, d) для 8 бит, упакованные uint8 для 4 и 1 бита;
    масштабы см. quantize_full()."""
    q = quantize_full(emb, bits)
    return q.codes, q.dequantize()


# ----------------------------------------------------------------------------- шум


def expected_cosine(sigma: float, mode: str = "relative", d: int | None = None) -> float:
    """Ожидаемый косинус между чистым и зашумлённым вектором: 1/√(1+σ²) (relative) или 1/√(1+σ²d) (absolute)."""
    if mode == "relative":
        return 1.0 / float(np.sqrt(1.0 + sigma * sigma))
    if d is None:
        raise ValueError("d is required for absolute mode")
    return 1.0 / float(np.sqrt(1.0 + sigma * sigma * d))


def add_noise(emb: np.ndarray, sigma: float, rng: Any, mode: str = "relative") -> np.ndarray:
    """e + σ·n (n ~ N(0, I)) с последующей L2-нормировкой строк; σ ≤ 0 → копия нормированного входа.

    mode="relative": σ_коорд = σ·‖e‖/√d (σ — отношение норм шума и вектора); mode="absolute": σ_коорд = σ.
    rng — np.random.Generator | random.Random | int | None.
    """
    x = _as_2d(emb)
    n, d = x.shape
    if mode not in ("relative", "absolute"):
        raise ValueError(f"unknown noise mode {mode!r}")
    if sigma <= 0.0:
        out = x.copy()
    else:
        g = as_generator(rng)
        noise = g.standard_normal((n, d), dtype=np.float32)
        if mode == "relative":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            noise = noise * (float(sigma) * norms / np.sqrt(float(d)))
        else:
            noise = noise * float(sigma)
        out = x + noise
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.ascontiguousarray(out / np.maximum(norms, 1e-12), dtype=np.float32)
