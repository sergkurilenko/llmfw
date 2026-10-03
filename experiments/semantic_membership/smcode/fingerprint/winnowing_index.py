"""M1 winnowing: индекс отпечатков k-грамм abstract(full)-токенов (DESIGN.md §6).

Отпечатки — smcode.fingerprint.winnowing.winnow_fingerprints (HMAC-SHA256 с ключом
config.hmac_key(cfg), усечение до 64 бит). Хранение: InvertedIndex (hash64 → int32
индексы записей) + CSR-последовательности отпечатков каждой записи (для
longest_common_run и overlap). Фильтр общности: отпечатки с df > common_df среди
функций public_train (fit_common_filter) или встречающиеся более чем в
common_file_frac (5 %) файлов protected (и не менее common_min_files файлов) исключаются
из индекса и из запроса. score = доля (не общих) отпечатков запроса, найденных в индексе;
details: best record по числу общих отпечатков, longest_common_run, n_fps.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Sequence

import numpy as np

from smcode.config import hmac_key
from smcode.fingerprint.index import (
    BaseIndex,
    InvertedIndex,
    chunked,
    iter_records,
    pool_context,
    pool_map,
    score_hashes,
    tokens_for,
)
from smcode.fingerprint.winnowing import Fingerprint, longest_common_run, winnow_fingerprints
from smcode.types import FunctionRecord, QueryResult

log = logging.getLogger(__name__)

DEFAULT_K = 25
DEFAULT_W = 25
DEFAULT_COMMON_DF = 50
DEFAULT_COMMON_FILE_FRAC = 0.05
DEFAULT_COMMON_MIN_FILES = 3


def _fp_task(args: tuple[str, str, int, int, bytes | None]) -> np.ndarray:
    """Хэши отпечатков записи по порядку (uint64); пустой массив, если токенов < k."""
    code, lang, k, w, key = args
    toks = tokens_for(code, lang, mode="full")
    fps = winnow_fingerprints(toks, k, w, key) if len(toks) >= k else []
    return np.fromiter((h for h, _ in fps), dtype=np.uint64, count=len(fps))


class WinnowingIndex(BaseIndex):
    """M1: индекс winnowing-отпечатков с фильтром общности."""

    name = "winnowing"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.k = DEFAULT_K
        self.w = DEFAULT_W
        self.common_df = DEFAULT_COMMON_DF
        self.common_file_frac = DEFAULT_COMMON_FILE_FRAC
        self.common_min_files = DEFAULT_COMMON_MIN_FILES
        self.workers = 1
        self.key: bytes | None = None
        self.inv = InvertedIndex()
        self.rec_offsets = np.zeros(1, dtype=np.int64)  # CSR-границы отпечатков записей
        self.rec_hashes = np.zeros(0, dtype=np.uint64)  # отпечатки всех записей по порядку
        self.public_common = np.zeros(0, dtype=np.uint64)  # общие по public_train
        self.protected_common = np.zeros(0, dtype=np.uint64)  # общие по файлам protected
        self.common = np.zeros(0, dtype=np.uint64)  # объединение (отсортировано)
        self.n_public_fitted = 0
        self._configure(self.cfg)

    def _configure(self, cfg: dict[str, Any] | None) -> None:
        if not cfg:
            return
        self.cfg = cfg
        fp = cfg.get("fingerprint", {}) or {}
        self.k = int(fp.get("k", DEFAULT_K))
        self.w = int(fp.get("w", DEFAULT_W))
        self.common_df = int(fp.get("common_df", DEFAULT_COMMON_DF))
        self.common_file_frac = float(fp.get("common_file_frac", DEFAULT_COMMON_FILE_FRAC))
        self.common_min_files = int(fp.get("common_min_files", DEFAULT_COMMON_MIN_FILES))
        self.workers = int(fp.get("workers", 1) or 1)
        self.key = hmac_key(cfg)

    # --- отпечатки

    def fingerprints(self, code: str, lang: str) -> list[Fingerprint]:
        """Полная последовательность отпечатков (hash, pos) без фильтра общности."""
        toks = tokens_for(code, lang, mode="full")
        if len(toks) < self.k:
            return []
        return winnow_fingerprints(toks, self.k, self.w, self.key)

    def filter_common(self, hs: np.ndarray) -> np.ndarray:
        """Убирает общие отпечатки из массива хэшей."""
        hs = np.asarray(hs, dtype=np.uint64)
        if self.common.size == 0 or hs.size == 0:
            return hs
        return hs[~np.isin(hs, self.common)]

    def query_hashes(self, code: str, lang: str) -> np.ndarray:
        """Уникальные отпечатки запроса после фильтра общности (uint64) — для гибрида."""
        fps = self.fingerprints(code, lang)
        if not fps:
            return np.zeros(0, dtype=np.uint64)
        return self.filter_common(np.unique(np.fromiter((h for h, _ in fps), dtype=np.uint64, count=len(fps))))

    def record_hashes(self, idx: int) -> np.ndarray:
        """Отпечатки записи idx по порядку (uint64, без фильтра)."""
        return self.rec_hashes[self.rec_offsets[idx] : self.rec_offsets[idx + 1]]

    def overlap(self, query_hashes: np.ndarray, idx: int) -> float:
        """Доля (отфильтрованных) уникальных отпечатков запроса, присутствующих у записи idx."""
        q = np.asarray(query_hashes, dtype=np.uint64)
        if q.size == 0:
            return 0.0
        return float(np.isin(q, self.record_hashes(idx)).sum() / q.size)

    # --- фильтр общности

    def fit_common_filter(self, public_records: Iterable[FunctionRecord | dict[str, Any]], common_df: int | None = None) -> int:
        """Отпечатки с df > common_df (по функциям public_train) помечаются общими. Возвращает их число.
        Можно вызывать до или после build (индекс перестраивается)."""
        df_thr = int(common_df if common_df is not None else self.common_df)
        parts: list[np.ndarray] = []
        n = 0
        with pool_context(self.workers) as pool:
            for chunk in chunked(iter_records(public_records)):
                tasks = [(r.code, r.lang, self.k, self.w, self.key) for r in chunk]
                for hs in pool_map(pool, _fp_task, tasks):
                    n += 1
                    if hs.size:
                        parts.append(np.unique(hs))
        if parts:
            keys, counts = np.unique(np.concatenate(parts), return_counts=True)
            self.public_common = keys[counts > df_thr]
        else:
            self.public_common = np.zeros(0, dtype=np.uint64)
        self.n_public_fitted = n
        self._apply_common()
        log.info("winnowing: common filter from %d public records: %d fingerprints with df > %d", n, self.public_common.size, df_thr)
        return int(self.public_common.size)

    def _apply_common(self) -> None:
        self.common = np.union1d(self.public_common, self.protected_common).astype(np.uint64)
        if self.n_records:
            self._rebuild_inverted()

    def _rebuild_inverted(self) -> None:
        counts = np.diff(self.rec_offsets)
        rec_idx = np.repeat(np.arange(self.n_records, dtype=np.int32), counts)
        self.inv = InvertedIndex.from_pairs(self.rec_hashes, rec_idx).without(self.common)
        self.meta["n_keys"] = len(self.inv)
        self.meta["n_common"] = int(self.common.size)

    # --- построение

    def build(self, records: Iterable[FunctionRecord], cfg: dict[str, Any]) -> None:
        self._configure(cfg)
        self._reset()
        t0 = time.perf_counter()
        ids: list[str] = []
        file_of: list[int] = []
        files: dict[tuple[str, str], int] = {}
        parts: list[np.ndarray] = []
        counts: list[int] = []
        n_short = 0
        with pool_context(self.workers) as pool:
            for chunk in chunked(iter_records(records)):
                tasks = [(r.code, r.lang, self.k, self.w, self.key) for r in chunk]
                for r, hs in zip(chunk, pool_map(pool, _fp_task, tasks)):
                    ids.append(r.id)
                    file_of.append(files.setdefault((r.repo, r.path), len(files)))
                    counts.append(int(hs.size))
                    if hs.size:
                        parts.append(hs)
                    else:
                        n_short += 1
        self.ids = ids
        self.rec_hashes = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint64)
        self.rec_offsets = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(np.asarray(counts, dtype=np.int64), out=self.rec_offsets[1:])
        # общие по файлам protected: число различных файлов на отпечаток
        n_files = len(files)
        if self.rec_hashes.size and n_files:
            file_idx = np.asarray(file_of, dtype=np.int32)[np.repeat(np.arange(len(ids), dtype=np.int32), np.asarray(counts))]
            per_file = InvertedIndex.from_pairs(self.rec_hashes, file_idx)
            df_files = per_file.df()
            mask = (df_files > self.common_file_frac * n_files) & (df_files >= self.common_min_files)
            self.protected_common = per_file.keys[mask]
        else:
            self.protected_common = np.zeros(0, dtype=np.uint64)
        self.meta = {
            "k": self.k,
            "w": self.w,
            "keyed": self.key is not None,
            "n_files": n_files,
            "n_fps_total": int(self.rec_hashes.size),
            "n_short_records": n_short,
            "n_public_fitted": self.n_public_fitted,
            "n_common_public": int(self.public_common.size),
            "n_common_protected": int(self.protected_common.size),
        }
        self._apply_common()
        self.meta["build_seconds"] = round(time.perf_counter() - t0, 3)
        log.info(
            "winnowing: %d records (%d files), %d fps, %d keys after common filter (%d common) in %.1fs",
            len(ids), n_files, self.rec_hashes.size, len(self.inv), self.common.size, self.meta["build_seconds"],
        )

    # --- запрос

    def query(self, code: str, lang: str) -> QueryResult:
        toks = tokens_for(code, lang, mode="full")
        n_tok = len(toks)
        if n_tok < self.k:
            return QueryResult(0.0, None, {"reason": "too_short", "n_tokens": n_tok, "n_fps": 0})
        fps = winnow_fingerprints(toks, self.k, self.w, self.key)
        seq = np.fromiter((h for h, _ in fps), dtype=np.uint64, count=len(fps))
        hs = self.filter_common(np.unique(seq))
        if hs.size == 0:
            return QueryResult(0.0, None, {"reason": "all_common", "n_tokens": n_tok, "n_fps": len(fps), "n_fps_filtered": 0})
        score, best, details = score_hashes(self.inv, hs)
        details.update({"n_tokens": n_tok, "n_fps": len(fps), "n_fps_filtered": int(hs.size), "longest_common_run": 0})
        best_id = None
        if best is not None:
            best_id = self.ids[best]
            q_seq = self.filter_common(seq)
            c_seq = self.filter_common(self.record_hashes(best))
            details["longest_common_run"] = longest_common_run([(int(h), i) for i, h in enumerate(q_seq)], [(int(h), i) for i, h in enumerate(c_seq)])
            details["best_overlap"] = details["best_shared"] / hs.size
        return QueryResult(float(score), best_id, details)

    # --- сериализация

    def _state(self) -> dict[str, Any]:
        return {
            **self.inv.arrays(),
            "rec_offsets": self.rec_offsets,
            "rec_hashes": self.rec_hashes,
            "public_common": self.public_common,
            "protected_common": self.protected_common,
            "common": self.common,
            "params": {
                "k": self.k, "w": self.w, "common_df": self.common_df, "common_file_frac": self.common_file_frac,
                "common_min_files": self.common_min_files, "keyed": self.key is not None, "n_public_fitted": self.n_public_fitted,
            },
        }

    def _restore(self, state: dict[str, Any]) -> None:
        self.inv = InvertedIndex.from_arrays(state)
        self.rec_offsets = np.asarray(state["rec_offsets"], dtype=np.int64)
        self.rec_hashes = np.asarray(state["rec_hashes"], dtype=np.uint64)
        self.public_common = np.asarray(state["public_common"], dtype=np.uint64)
        self.protected_common = np.asarray(state["protected_common"], dtype=np.uint64)
        self.common = np.asarray(state["common"], dtype=np.uint64)
        p = state.get("params", {})
        self.k = int(p.get("k", self.k))
        self.w = int(p.get("w", self.w))
        self.common_df = int(p.get("common_df", self.common_df))
        self.n_public_fitted = int(p.get("n_public_fitted", 0))
        if p.get("keyed") and self.key is None:
            log.warning("winnowing index was built with an HMAC key, but none is set now: queries will not match")
