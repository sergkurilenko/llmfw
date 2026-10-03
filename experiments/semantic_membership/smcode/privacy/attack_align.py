"""Атака A2 — выравнивание (ALGEN-подобная, DESIGN.md §9).

Атакующему известны n пар (код → защищённый вектор). Он вычисляет plain-эмбеддинги кода публичным
энкодером и оценивает проекцию Q̂ по парам (plain, defended): ортогональный Прокруст через SVD
(по умолчанию) или МНК (np.linalg.lstsq). Затем инвертирует защищённые векторы индекса
(E ≈ E' Q̂ для ортогональной Q̂, иначе через псевдообратную) и применяет обученную модель A1.
Кривая утечки — по n ∈ cfg.privacy.attack_leaked_pairs; n = 0 ⇒ Q̂ = I (прямое применение A1).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from smcode.privacy.attack_bow import BowInverter
from smcode.privacy.projection import is_orthogonal
from smcode.privacy.quantize import as_generator

log = logging.getLogger(__name__)

ALIGN_METHODS: tuple[str, ...] = ("procrustes", "lstsq")


def estimate_projection(plain: np.ndarray, defended: np.ndarray, method: str = "procrustes") -> np.ndarray:
    """Оценка Q̂ (d×d, float32) по парам строк plain ↔ defended (defended ≈ plain Q̂ᵀ).

    procrustes: W = U Vᵀ, где U S Vᵀ = SVD(plainᵀ defended) — ближайшая ортогональная матрица (устойчива к шуму
    и квантованию, работает и при n < d); lstsq: W = argmin ‖plain W − defended‖ (минимальная норма при n < d).
    Пустой вход → единичная матрица.
    """
    if method not in ALIGN_METHODS:
        raise ValueError(f"unknown align method {method!r}; known: {ALIGN_METHODS}")
    A = np.asarray(plain, dtype=np.float64)
    B = np.asarray(defended, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape != B.shape:
        raise ValueError(f"plain {A.shape} and defended {B.shape} must be equal 2-D shapes")
    d = A.shape[1]
    if A.shape[0] == 0:
        return np.eye(d, dtype=np.float32)
    if method == "procrustes":
        u, _, vt = np.linalg.svd(A.T @ B, full_matrices=False)
        W = u @ vt
    else:
        W = np.linalg.lstsq(A, B, rcond=None)[0]
    return np.ascontiguousarray(W.T, dtype=np.float32)


def recover(defended: np.ndarray, Q_hat: np.ndarray) -> np.ndarray:
    """Оценка plain-эмбеддингов: E' Q̂ при ортогональной Q̂, иначе E' pinv(Q̂ᵀ)."""
    x = np.asarray(defended, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if is_orthogonal(Q_hat, tol=1e-3):
        return np.ascontiguousarray(x @ Q_hat.astype(np.float32, copy=False), dtype=np.float32)
    inv = np.linalg.pinv(np.asarray(Q_hat, dtype=np.float64).T)
    return np.ascontiguousarray(x @ inv.astype(np.float32), dtype=np.float32)


def relative_error(Q_hat: np.ndarray, Q: np.ndarray) -> float:
    """‖Q̂ − Q‖_F / ‖Q‖_F (для ортогональной Q знаменатель = √d; ≈ √2 для несвязанной оценки)."""
    q = np.asarray(Q, dtype=np.float64)
    return float(np.linalg.norm(np.asarray(Q_hat, dtype=np.float64) - q) / max(np.linalg.norm(q), 1e-12))


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Средний косинус между соответствующими строками a и b."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.maximum(np.linalg.norm(a, axis=1), 1e-12)
    nb = np.maximum(np.linalg.norm(b, axis=1), 1e-12)
    return float(np.mean(np.sum(a * b, axis=1) / (na * nb))) if a.shape[0] else 0.0


def sample_leaked_pairs(pool_plain: np.ndarray, pool_defended: np.ndarray, n: int, rng: Any) -> tuple[np.ndarray, np.ndarray]:
    """n случайных пар из пула (без возвращения; n обрезается до размера пула)."""
    m = pool_plain.shape[0]
    if pool_defended.shape[0] != m:
        raise ValueError("pool_plain and pool_defended must have the same number of rows")
    k = int(min(max(0, n), m))
    if k == 0:
        return pool_plain[:0], pool_defended[:0]
    idx = np.sort(as_generator(rng).choice(m, size=k, replace=False))
    return pool_plain[idx], pool_defended[idx]


def align_once(
    model: BowInverter,
    pool_plain: np.ndarray,
    pool_defended: np.ndarray,
    target_defended: np.ndarray,
    target_Y: sparse.csr_matrix,
    n: int,
    rng: Any,
    method: str = "procrustes",
    Q_true: np.ndarray | None = None,
    target_plain: np.ndarray | None = None,
) -> dict[str, Any]:
    """Одна точка кривой: n утёкших пар → Q̂ → инверсия target_defended → метрики A1.
    Дополнительно: q_rel_error (если известна Q_true) и recovered_cos (если известны plain-цели)."""
    p, dfd = sample_leaked_pairs(pool_plain, pool_defended, n, rng)
    Q_hat = estimate_projection(p, dfd, method=method) if p.shape[0] else np.eye(target_defended.shape[1], dtype=np.float32)
    rec = recover(target_defended, Q_hat)
    out = model.evaluate(rec, target_Y)
    out.update({"n": int(n), "n_effective": int(p.shape[0]), "method": method})
    if Q_true is not None:
        out["q_rel_error"] = relative_error(Q_hat, Q_true)
    if target_plain is not None:
        out["recovered_cos"] = mean_cosine(rec, target_plain)
    return out


def run_attack_align(
    model: BowInverter,
    pool_plain: np.ndarray,
    pool_defended: np.ndarray,
    target_defended: np.ndarray,
    target_Y: sparse.csr_matrix,
    leaked_pairs: Sequence[int],
    rng: random.Random | np.random.Generator | int | None,
    method: str = "procrustes",
    Q_true: np.ndarray | None = None,
    target_plain: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """Кривая утечки {str(n): метрики} по n ∈ leaked_pairs (одна и та же выборка пар — вложенная по n)."""
    g = as_generator(rng)
    seeds = {int(n): int(g.integers(0, 2**63 - 1)) for n in leaked_pairs}
    out: dict[str, dict[str, Any]] = {}
    for n in leaked_pairs:
        res = align_once(model, pool_plain, pool_defended, target_defended, target_Y, int(n), np.random.default_rng(seeds[int(n)]),
                         method=method, Q_true=Q_true, target_plain=target_plain)
        log.info("A2 n=%d (eff %d): f1=%.3f rare_recall=%s q_err=%s", n, res["n_effective"], res["f1"], res.get("rare_id_recall"),
                 None if "q_rel_error" not in res else round(res["q_rel_error"], 4))
        out[str(int(n))] = res
    return out
