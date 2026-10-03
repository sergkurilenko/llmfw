"""Поиск ближайших соседей по косинусу (DESIGN.md §6, M3).

Бэкенд: FAISS ``IndexFlatIP`` (если установлен), иначе numpy — блочное матричное умножение
``q @ X.T`` с частичной сортировкой (``argpartition``) по блокам строк индекса, что
укладывается в память при N ~ 10^6 и d ~ 768. Все векторы считаются L2-нормированными,
поэтому скалярное произведение = косинус. torch/faiss импортируются лениво.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_CHUNK_ROWS = 65536  # строк индекса на блок (65536 × 1024 запросов × 4 байта = 256 МБ)
DEFAULT_CHUNK_QUERIES = 1024


def _try_faiss() -> Any | None:
    try:
        import faiss  # type: ignore

        return faiss
    except Exception:  # noqa: BLE001 — ImportError или битая сборка
        return None


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-нормировка строк (float32, C-contiguous); одномерный вход → (1, d)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x / np.maximum(norms, eps), dtype=np.float32)


def _sort_rows_desc(sims: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-sims, axis=1, kind="stable")
    return np.take_along_axis(sims, order, axis=1), np.take_along_axis(idx, order, axis=1)


def topk_numpy(
    q: np.ndarray,
    X: np.ndarray,
    k: int,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    chunk_queries: int = DEFAULT_CHUNK_QUERIES,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k по скалярному произведению для каждой строки q среди строк X.

    Возвращает (sims (M, k) float32, idx (M, k) int64), отсортированные по убыванию.
    k обрезается до N; при N = 0 — массивы формы (M, 0).
    """
    q = np.asarray(q, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    if q.ndim == 1:
        q = q[None, :]
    M, N = q.shape[0], X.shape[0]
    k = int(min(k, N))
    if k <= 0 or M == 0:
        return np.zeros((M, 0), dtype=np.float32), np.zeros((M, 0), dtype=np.int64)
    sims_out = np.empty((M, k), dtype=np.float32)
    idx_out = np.empty((M, k), dtype=np.int64)
    for qs in range(0, M, chunk_queries):
        qb = q[qs : qs + chunk_queries]
        m = qb.shape[0]
        cur_s = np.full((m, 0), -np.inf, dtype=np.float32)
        cur_i = np.zeros((m, 0), dtype=np.int64)
        for r0 in range(0, N, chunk_rows):
            r1 = min(N, r0 + chunk_rows)
            s = qb @ X[r0:r1].T  # (m, c)
            c = r1 - r0
            if c > k:
                part = np.argpartition(-s, k - 1, axis=1)[:, :k]
                s_top = np.take_along_axis(s, part, axis=1)
                i_top = part.astype(np.int64) + r0
            else:
                s_top = s
                i_top = np.broadcast_to(np.arange(r0, r1, dtype=np.int64), (m, c))
            cat_s = np.concatenate([cur_s, s_top], axis=1)
            cat_i = np.concatenate([cur_i, i_top], axis=1)
            if cat_s.shape[1] > k:
                part = np.argpartition(-cat_s, k - 1, axis=1)[:, :k]
                cur_s = np.take_along_axis(cat_s, part, axis=1)
                cur_i = np.take_along_axis(cat_i, part, axis=1)
            else:
                cur_s, cur_i = cat_s, cat_i
        cur_s, cur_i = _sort_rows_desc(cur_s, cur_i)
        sims_out[qs : qs + m] = cur_s
        idx_out[qs : qs + m] = cur_i
    return sims_out, idx_out


def cosine_topk(q: np.ndarray, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k косинусов: нормирует q и X и вызывает topk_numpy."""
    return topk_numpy(l2_normalize(q), l2_normalize(X), k)


class ANN:
    """Индекс точного поиска по скалярному произведению: FAISS IndexFlatIP или numpy."""

    def __init__(
        self,
        matrix: np.ndarray | None = None,
        use_faiss: bool | None = None,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
    ) -> None:
        self.use_faiss = use_faiss
        self.chunk_rows = int(chunk_rows)
        self.backend = "numpy"
        self.X: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._faiss_index: Any | None = None
        if matrix is not None:
            self.add(matrix)

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def dim(self) -> int:
        return int(self.X.shape[1]) if self.X.ndim == 2 else 0

    def add(self, matrix: np.ndarray) -> None:
        """Заменяет содержимое индекса матрицей (N, d) float32 (нормировка — забота вызывающего)."""
        X = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
        if X.ndim != 2:
            raise ValueError(f"matrix must be 2-D, got shape {X.shape}")
        self.X = X
        self._faiss_index = None
        self.backend = "numpy"
        if self.use_faiss is False or X.shape[0] == 0:
            return
        faiss = _try_faiss()
        if faiss is None:
            if self.use_faiss:
                log.warning("faiss requested but not importable; using numpy backend")
            return
        try:
            index = faiss.IndexFlatIP(int(X.shape[1]))
            index.add(X)
            self._faiss_index = index
            self.backend = "faiss"
        except Exception as exc:  # noqa: BLE001
            log.warning("faiss index failed (%s); using numpy backend", exc)
            self._faiss_index = None
            self.backend = "numpy"

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(sims (M, k) float32, idx (M, k) int64) по убыванию; k обрезается до N."""
        q = np.ascontiguousarray(np.asarray(q, dtype=np.float32))
        if q.ndim == 1:
            q = q[None, :]
        k = int(min(k, self.n))
        if k <= 0 or q.shape[0] == 0:
            return np.zeros((q.shape[0], 0), dtype=np.float32), np.zeros((q.shape[0], 0), dtype=np.int64)
        if self._faiss_index is not None:
            sims, idx = self._faiss_index.search(q, k)
            return np.asarray(sims, dtype=np.float32), np.asarray(idx, dtype=np.int64)
        return topk_numpy(q, self.X, k, chunk_rows=self.chunk_rows)

    def memory_bytes(self) -> int:
        """Память под матрицу (FAISS хранит свою копию — учитываем её отдельно)."""
        extra = int(self.X.nbytes) if self._faiss_index is not None else 0
        return int(self.X.nbytes) + extra
