"""Ключевая ортогональная проекция эмбеддингов (DESIGN.md §9, защита 1).

e' = Q e, где Q — случайная ортогональная матрица d×d, распределённая по Хаару (QR-разложение
гауссовой матрицы с коррекцией знаков), seed которой выводится из секретного ключа через SHA-256.
Косинусы между векторами сохраняются точно, поэтому полезность индекса не меняется; без ключа
Q неизвестна и векторы индекса бесполезны для инверсии (пока атакующий не оценит Q по утёкшим парам,
см. attack_align.py). В строковой (row-major) форме: E' = E Qᵀ, обратно E = E' Q.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

import numpy as np

from smcode.config import hmac_key

log = logging.getLogger(__name__)

PROJECTION_KEY_ENV_DEFAULT = "SMCODE_PROJECTION_KEY"


def key_seed(key: bytes, salt: str = "") -> int:
    """64-битный seed из ключа: первые 8 байт SHA-256(key || 0x00 || salt)."""
    h = hashlib.sha256(bytes(key) + b"\x00" + salt.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def keyed_orthogonal(d: int, key: bytes) -> np.ndarray:
    """Ортогональная матрица Q (d×d, float32) из ключа: QR гауссовой матрицы с seed = sha256(key).

    Знаки столбцов фиксируются по диагонали R (diag(R) > 0), что делает Q однозначной и равномерной
    по мере Хаара. Детерминизм: один ключ и одна размерность → одна и та же Q.
    """
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    rng = np.random.default_rng(key_seed(key, "keyed_orthogonal"))
    g = rng.standard_normal((d, d), dtype=np.float64)
    q, r = np.linalg.qr(g)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs[None, :]
    return np.ascontiguousarray(q, dtype=np.float32)


def apply(emb: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """E' = E Qᵀ (для каждой строки e' = Q e). Возвращает float32 той же формы; 1-D вход → (1, d)."""
    x = np.asarray(emb, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[1] != Q.shape[0]:
        raise ValueError(f"dim mismatch: emb {x.shape[1]} vs Q {Q.shape}")
    return np.ascontiguousarray(x @ Q.T.astype(np.float32, copy=False), dtype=np.float32)


def invert(emb_def: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Обратная проекция E = E' Q (Q ортогональна, Q⁻¹ = Qᵀ)."""
    x = np.asarray(emb_def, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return np.ascontiguousarray(x @ Q.astype(np.float32, copy=False), dtype=np.float32)


def is_orthogonal(Q: np.ndarray, tol: float = 1e-3) -> bool:
    """True, если ‖Q Qᵀ − I‖_max ≤ tol."""
    q = np.asarray(Q, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        return False
    return bool(np.max(np.abs(q @ q.T - np.eye(q.shape[0]))) <= tol)


def projection_key(cfg: dict[str, Any]) -> bytes:
    """Ключ проекции: переменная окружения cfg.privacy.projection_key_env (по умолчанию SMCODE_PROJECTION_KEY),
    иначе ключ HMAC отпечатков (smcode.config.hmac_key), иначе детерминированный ключ из seed (с предупреждением:
    только для экспериментов/тестов)."""
    env = (cfg.get("privacy", {}) or {}).get("projection_key_env", PROJECTION_KEY_ENV_DEFAULT)
    val = os.environ.get(env)
    if val:
        return val.encode("utf-8")
    k = hmac_key(cfg)
    if k:
        return k
    seed = int(cfg.get("seed", 0))
    log.warning("projection key: переменная %s не задана, используется ключ из seed=%d (только для экспериментов)", env, seed)
    return f"smcode-projection-{seed}".encode("utf-8")
