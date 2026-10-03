"""M1 winnowing: индекс отпечатков k-грамм abstract-токенов (DESIGN.md §6).

Режим токенов — cfg.fingerprint.token_mode: full (по умолчанию; ID/NUM/STR без комментариев — детектор клонов
типа 2, по построению инвариантный к reformat/strip_comments/rename_ids/change_literals/combo) или lexical
(классические Moss-отпечатки по лексическим токенам, которые эти преобразования ломают; вариант winnowing_lexical).
Отпечатки — smcode.fingerprint.winnowing.winnow_fingerprints (HMAC-SHA256 с ключом
config.hmac_key(cfg), усечение до 64 бит). Хранение: InvertedIndex (hash64 → int32
индексы записей) + CSR-последовательности отпечатков каждой записи (для
longest_common_run и overlap). Фильтр общности: отпечатки с df > common_df среди
функций public_train (fit_common_filter) или встречающиеся более чем в
common_file_frac (5 %) файлов protected (и не менее common_min_files файлов) исключаются
из индекса и из запроса; проверка общности — searchsorted по отсортированному массиву
common (стоимость запроса не зависит от |common|). score = доля (не общих) отпечатков
запроса, найденных в индексе; details: best record по числу общих отпечатков,
longest_common_run, n_fps.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable

import numpy as np

from smcode.config import hmac_key
from smcode.fingerprint.index import (
    REASON_TOO_SHORT,
    BaseIndex,
    InvertedIndex,
    chunked,
    iter_records,
    pool_context,
    pool_map,
    score_hashes,
    sorted_contains,
    tokens_or_reason,
)
from smcode.fingerprint.winnowing import Fingerprint, longest_common_run, winnow_fingerprints
from smcode.types import FunctionRecord, QueryResult

log = logging.getLogger(__name__)

DEFAULT_K = 25
DEFAULT_W = 25
DEFAULT_COMMON_DF = 50
DEFAULT_COMMON_FILE_FRAC = 0.05
DEFAULT_COMMON_MIN_FILES = 3
DEFAULT_TOKEN_MODE = "full"
TOKEN_MODES: tuple[str, ...] = ("full", "lexical", "indexed")
REASON_ALL_COMMON = "all_common"


def token_mode_of(cfg: dict[str, Any] | None) -> str:
    """cfg.fingerprint.token_mode (full | lexical | indexed), по умолчанию full; ValueError для неизвестного режима."""
    mode = str(((cfg or {}).get("fingerprint", {}) or {}).get("token_mode", DEFAULT_TOKEN_MODE) or DEFAULT_TOKEN_MODE)
    if mode not in TOKEN_MODES:
        raise ValueError(f"fingerprint.token_mode={mode!r}; known: {TOKEN_MODES}")
    return mode


def _fp_task(args: tuple[str, str, int, int, bytes | None, str]) -> tuple[np.ndarray, str | None]:
    """(хэши отпечатков записи по порядку (uint64), причина отказа токенизации); пустой массив, если токенов < k.
    args = (code, lang, k, w, key, token_mode)."""
    code, lang, k, w, key, mode = args
    toks, reason = tokens_or_reason(code, lang, mode=mode)
    fps = winnow_fingerprints(toks, k, w, key) if len(toks) >= k else []
    return np.fromiter((h for h, _ in fps), dtype=np.uint64, count=len(fps)), reason


def _lcr(q_seq: np.ndarray, c_seq: np.ndarray) -> int:
    """longest_common_run по двум последовательностям хэшей (позиции — порядковые номера)."""
    return longest_common_run([(int(h), i) for i, h in enumerate(q_seq)], [(int(h), i) for i, h in enumerate(c_seq)])


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
        self.token_mode = DEFAULT_TOKEN_MODE
        self.workers = 1
        self.key: bytes | None = None
        self.inv = InvertedIndex()
        self.rec_offsets = np.zeros(1, dtype=np.int64)  # CSR-границы отпечатков записей
        self.rec_hashes = np.zeros(0, dtype=np.uint64)  # отпечатки всех записей по порядку
        self.public_common = np.zeros(0, dtype=np.uint64)  # общие по public_train
        self.protected_common = np.zeros(0, dtype=np.uint64)  # общие по файлам protected
        self.common = np.zeros(0, dtype=np.uint64)  # объединение (отсортировано, уникально)
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
        self.token_mode = token_mode_of(cfg)
        self.workers = int(fp.get("workers", 1) or 1)
        self.key = hmac_key(cfg)

    # --- отпечатки

    def fingerprints(self, code: str, lang: str) -> list[Fingerprint]:
        """Полная последовательность отпечатков (hash, pos) без фильтра общности."""
        toks, _ = tokens_or_reason(code, lang, mode=self.token_mode)
        if len(toks) < self.k:
            return []
        return winnow_fingerprints(toks, self.k, self.w, self.key)

    def common_mask(self, hs: np.ndarray) -> np.ndarray:
        """Булева маска общих отпечатков (searchsorted по self.common; O(|hs| log |common|))."""
        return sorted_contains(self.common, np.asarray(hs, dtype=np.uint64))

    def filter_common(self, hs: np.ndarray) -> np.ndarray:
        """Убирает общие отпечатки из массива хэшей (порядок сохраняется)."""
        hs = np.asarray(hs, dtype=np.uint64)
        if self.common.size == 0 or hs.size == 0:
            return hs
        return hs[~self.common_mask(hs)]

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
        return float(sorted_contains(np.unique(self.record_hashes(idx)), q).sum() / q.size)

    # --- фильтр общности

    def fit_common_filter(self, public_records: Iterable[FunctionRecord | dict[str, Any]], common_df: int | None = None) -> int:
        """Отпечатки с df > common_df (по функциям public_train) помечаются общими. Возвращает их число.
        Можно вызывать до или после build (индекс и meta перестраиваются)."""
        df_thr = int(common_df if common_df is not None else self.common_df)
        parts: list[np.ndarray] = []
        n = 0
        with pool_context(self.workers) as pool:
            for chunk in chunked(iter_records(public_records)):
                tasks = [(r.code, r.lang, self.k, self.w, self.key, self.token_mode) for r in chunk]
                for hs, _ in pool_map(pool, _fp_task, tasks):
                    n += 1
                    if hs.size:
                        parts.append(np.unique(hs))
        if parts:
            keys, counts = np.unique(np.concatenate(parts), return_counts=True)
            self.public_common = keys[counts > df_thr]
        else:
            self.public_common = np.zeros(0, dtype=np.uint64)
        self.n_public_fitted = n
        self.common_df = df_thr
        self._apply_common()
        log.info("winnowing: common filter from %d public records: %d fingerprints with df > %d", n, self.public_common.size, df_thr)
        return int(self.public_common.size)

    def _apply_common(self) -> None:
        """Пересчитывает common = public ∪ protected, инвертированный индекс и поля meta фильтра."""
        self.common = np.union1d(self.public_common, self.protected_common).astype(np.uint64)
        self.meta.update({
            "n_public_fitted": int(self.n_public_fitted),
            "common_df": self.common_df,
            "n_common_public": int(self.public_common.size),
            "n_common_protected": int(self.protected_common.size),
            "n_common": int(self.common.size),
        })
        if self.n_records:
            self._rebuild_inverted()

    def _rebuild_inverted(self) -> None:
        counts = np.diff(self.rec_offsets)
        rec_idx = np.repeat(np.arange(self.n_records, dtype=np.int32), counts)
        self.inv = InvertedIndex.from_pairs(self.rec_hashes, rec_idx).without(self.common)
        self.meta["n_keys"] = len(self.inv)

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
        n_short = n_unsupported = n_errors = 0
        with pool_context(self.workers) as pool:
            for chunk in chunked(iter_records(records)):
                tasks = [(r.code, r.lang, self.k, self.w, self.key, self.token_mode) for r in chunk]
                for r, (hs, reason) in zip(chunk, pool_map(pool, _fp_task, tasks)):
                    ids.append(r.id)
                    file_of.append(files.setdefault((r.repo, r.path), len(files)))
                    counts.append(int(hs.size))
                    if hs.size:
                        parts.append(hs)
                    elif reason == "unsupported_lang":
                        n_unsupported += 1
                    elif reason == "tokenize_error":
                        n_errors += 1
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
            "token_mode": self.token_mode,
            **self.key_params(),
            "common_file_frac": self.common_file_frac,
            "common_min_files": self.common_min_files,
            "n_files": n_files,
            "n_fps_total": int(self.rec_hashes.size),
            "n_short_records": n_short,
            "n_unsupported": n_unsupported,
            "n_tokenize_errors": n_errors,
        }
        self._apply_common()
        self.meta["build_seconds"] = round(time.perf_counter() - t0, 3)
        log.info(
            "winnowing: %d records (%d files), %d fps, %d keys after common filter (%d common) in %.1fs",
            len(ids), n_files, self.rec_hashes.size, len(self.inv), self.common.size, self.meta["build_seconds"],
        )

    # --- запрос

    def query(self, code: str, lang: str) -> QueryResult:
        toks, reason = tokens_or_reason(code, lang, mode=self.token_mode)
        n_tok = len(toks)
        if n_tok < self.k:
            return QueryResult(0.0, None, {"reason": reason or REASON_TOO_SHORT, "n_tokens": n_tok, "n_fps": 0})
        fps = winnow_fingerprints(toks, self.k, self.w, self.key)
        seq = np.fromiter((h for h, _ in fps), dtype=np.uint64, count=len(fps))
        # маска общности считается один раз по последовательности; hs — уникальные неотфильтрованные
        q_seq = seq[~self.common_mask(seq)] if self.common.size else seq
        hs = np.unique(q_seq)
        if hs.size == 0:
            return QueryResult(0.0, None, {"reason": REASON_ALL_COMMON, "n_tokens": n_tok, "n_fps": len(fps), "n_fps_filtered": 0})
        score, best, details = score_hashes(self.inv, hs)
        details.update({"n_tokens": n_tok, "n_fps": len(fps), "n_fps_filtered": int(hs.size), "longest_common_run": 0})
        best_id = None
        if best is not None:
            best_id = self.ids[best]
            c_seq = self.filter_common(self.record_hashes(best))
            details["longest_common_run"] = _lcr(q_seq, c_seq)
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
                "k": self.k, "w": self.w, "token_mode": self.token_mode, "common_df": self.common_df,
                "common_file_frac": self.common_file_frac, "common_min_files": self.common_min_files,
                "n_public_fitted": self.n_public_fitted, **self.key_params(),
            },
        }

    def _restore(self, state: dict[str, Any]) -> None:
        p = state.get("params", {}) or {}
        self._check_key(p)
        self.token_mode = str(p.get("token_mode", DEFAULT_TOKEN_MODE))  # режим индекса важнее конфига (иначе запросы не совпадут)
        self.inv = InvertedIndex.from_arrays(state)
        self.rec_offsets = np.asarray(state["rec_offsets"], dtype=np.int64)
        self.rec_hashes = np.asarray(state["rec_hashes"], dtype=np.uint64)
        self.public_common = np.asarray(state["public_common"], dtype=np.uint64)
        self.protected_common = np.asarray(state["protected_common"], dtype=np.uint64)
        self.common = np.asarray(state["common"], dtype=np.uint64)
        self.k = int(p.get("k", self.k))
        self.w = int(p.get("w", self.w))
        self.common_df = int(p.get("common_df", self.common_df))
        self.common_file_frac = float(p.get("common_file_frac", self.common_file_frac))
        self.common_min_files = int(p.get("common_min_files", self.common_min_files))
        self.n_public_fitted = int(p.get("n_public_fitted", 0))
        self.meta.setdefault("token_mode", self.token_mode)
