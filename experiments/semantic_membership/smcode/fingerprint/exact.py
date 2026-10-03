"""M0 exact: хэши нормализованных окон из `window` строк (DESIGN.md §6).

Строка нормализуется как последовательность лексических токенов (комментарии и
пустые строки удалены, пробелы схлопнуты); окно = `window` подряд идущих строк,
хэш = blake2b-64 (с ключом HMAC из конфига, если задан). Записи короче окна дают
одно окно из всех своих строк. score = доля окон запроса, найденных в индексе;
best_id — запись с наибольшим числом общих окон.
"""

from __future__ import annotations

import hashlib
import logging
import time
from bisect import bisect_right
from typing import Any, Iterable

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
)
from smcode.normalize import tokenize
from smcode.types import FunctionRecord, QueryResult

log = logging.getLogger(__name__)

DEFAULT_WINDOW = 5


def hash64(text: str, key: bytes | None = None) -> int:
    """64-битный blake2b текста; ключ (если задан) делает хэши неузнаваемыми без ключа."""
    k = key or b""
    if len(k) > 64:
        k = hashlib.sha256(k).digest()
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8, key=k).digest(), "big")


def normalized_lines(code: str, lang: str) -> list[str]:
    """Строки кода как «токен токен ...» (лексические токены без комментариев); пустые строки опущены."""
    try:
        toks = tokenize(code, lang)
    except ValueError:
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning("tokenize crashed (%s): %s", lang, exc)
        return []
    src = code.encode("utf-8", "replace")
    newlines = np.flatnonzero(np.frombuffer(src, dtype=np.uint8) == 0x0A).tolist()
    lines: dict[int, list[str]] = {}
    for t in toks:
        if t.kind == "comment":
            continue
        ln = bisect_right(newlines, t.start)
        text = " ".join(t.text.split()) if ("\n" in t.text or "\t" in t.text) else t.text
        lines.setdefault(ln, []).append(text)
    return [" ".join(parts) for _, parts in sorted(lines.items())]


def window_hashes(code: str, lang: str, window: int = DEFAULT_WINDOW, key: bytes | None = None) -> list[int]:
    """Хэши всех окон из `window` строк; для кода короче окна — одно окно из всех строк; [] без токенов."""
    lines = normalized_lines(code, lang)
    if not lines:
        return []
    if len(lines) <= window:
        return [hash64("\n".join(lines), key)]
    return [hash64("\n".join(lines[i : i + window]), key) for i in range(len(lines) - window + 1)]


def _win_task(args: tuple[str, str, int, bytes | None]) -> list[int]:
    code, lang, window, key = args
    return window_hashes(code, lang, window, key)


class ExactIndex(BaseIndex):
    """M0: индекс окон строк; хранение — InvertedIndex (hash64 → int32 индексы записей)."""

    name = "exact"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.window = DEFAULT_WINDOW
        self.workers = 1
        self.key: bytes | None = None
        self.inv = InvertedIndex()
        self.n_windows = np.zeros(0, dtype=np.int32)
        self._configure(self.cfg)

    def _configure(self, cfg: dict[str, Any] | None) -> None:
        if not cfg:
            return
        self.cfg = cfg
        fp = cfg.get("fingerprint", {}) or {}
        self.window = int(fp.get("exact_window", DEFAULT_WINDOW))
        self.workers = int(fp.get("workers", 1) or 1)
        self.key = hmac_key(cfg)

    # --- построение

    def build(self, records: Iterable[FunctionRecord], cfg: dict[str, Any]) -> None:
        self._configure(cfg)
        self._reset()
        t0 = time.perf_counter()
        ids: list[str] = []
        n_win: list[int] = []
        hashes: list[np.ndarray] = []
        with pool_context(self.workers) as pool:
            for chunk in chunked(iter_records(records)):
                tasks = [(r.code, r.lang, self.window, self.key) for r in chunk]
                for r, ws in zip(chunk, pool_map(pool, _win_task, tasks)):
                    ids.append(r.id)
                    n_win.append(len(ws))
                    if ws:
                        hashes.append(np.array(ws, dtype=np.uint64))
        self.ids = ids
        self.n_windows = np.asarray(n_win, dtype=np.int32)
        all_h = np.concatenate(hashes) if hashes else np.zeros(0, dtype=np.uint64)
        rec_idx = np.repeat(np.arange(len(ids), dtype=np.int32), self.n_windows)
        self.inv = InvertedIndex.from_pairs(all_h, rec_idx)
        self.meta = {
            "window": self.window,
            "keyed": self.key is not None,
            "n_windows_total": int(all_h.size),
            "n_keys": len(self.inv),
            "build_seconds": round(time.perf_counter() - t0, 3),
        }
        log.info("exact: %d records, %d windows, %d unique keys in %.1fs", len(ids), all_h.size, len(self.inv), self.meta["build_seconds"])

    # --- запрос

    def query_hashes(self, code: str, lang: str) -> np.ndarray:
        """Уникальные хэши окон запроса (uint64)."""
        ws = window_hashes(code, lang, self.window, self.key)
        return np.unique(np.array(ws, dtype=np.uint64)) if ws else np.zeros(0, dtype=np.uint64)

    def query(self, code: str, lang: str) -> QueryResult:
        ws = window_hashes(code, lang, self.window, self.key)
        if not ws:
            return QueryResult(0.0, None, {"reason": "too_short", "n_windows": 0})
        hs = np.unique(np.array(ws, dtype=np.uint64))
        score, best, details = score_hashes(self.inv, hs)
        details.update({"n_windows": len(ws), "n_unique": int(hs.size)})
        return QueryResult(float(score), self.ids[best] if best is not None else None, details)

    # --- сериализация

    def _state(self) -> dict[str, Any]:
        return {**self.inv.arrays(), "n_windows": self.n_windows, "params": {"window": self.window, "keyed": self.key is not None}}

    def _restore(self, state: dict[str, Any]) -> None:
        self.inv = InvertedIndex.from_arrays(state)
        self.n_windows = np.asarray(state["n_windows"], dtype=np.int32)
        self.window = int(state.get("params", {}).get("window", self.window))
        if state.get("params", {}).get("keyed") and self.key is None:
            log.warning("exact index was built with an HMAC key, but none is set now: queries will not match")
