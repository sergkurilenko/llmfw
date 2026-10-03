"""M2 minhash: MinHash + LSH (datasketch) по shingle'ам abstract(full)-токенов (DESIGN.md §6).

Shingle = `minhash_shingle` подряд идущих абстрактных токенов → blake2b-64 (ключ HMAC
из конфига, если задан). Подпись: datasketch.MinHash(num_perm=minhash_perm) над
младшими 32 битами хэшей shingle'ов; кандидаты: MinHashLSH(threshold=0.3);
score = max точного Жаккара по множествам shingle'ов (хранятся как CSR отсортированных
uint64). На диск пишутся матрица подписей и множества shingle'ов; LSH пересобирается при
загрузке (≈60 мкс на запись).
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

from smcode.config import hmac_key
from smcode.fingerprint.index import (
    BaseIndex,
    chunked,
    deep_sizeof,
    iter_records,
    pool_context,
    pool_map,
    tokens_for,
)
from smcode.types import FunctionRecord, QueryResult

log = logging.getLogger(__name__)

DEFAULT_NUM_PERM = 128
DEFAULT_SHINGLE = 5
DEFAULT_LSH_THRESHOLD = 0.3
_MASK32 = np.uint64(0xFFFFFFFF)


def _h32(b: bytes) -> int:
    """hashfunc для datasketch: shingle уже захэширован, берём 4 байта как есть."""
    return int.from_bytes(b, "big")


def shingle_hash(tokens: Sequence[str], key: bytes | None = None) -> int:
    k = key or b""
    if len(k) > 64:
        k = hashlib.sha256(k).digest()
    data = "\x1f".join(tokens).encode("utf-8", "replace")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8, key=k).digest(), "big")


def shingle_set(tokens: Sequence[str], s: int, key: bytes | None = None) -> np.ndarray:
    """Уникальные отсортированные хэши shingle'ов (uint64); пусто, если токенов < s."""
    n = len(tokens)
    if n < s:
        return np.zeros(0, dtype=np.uint64)
    hs = np.fromiter((shingle_hash(tokens[i : i + s], key) for i in range(n - s + 1)), dtype=np.uint64, count=n - s + 1)
    return np.unique(hs)


def jaccard(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Точный Жаккар двух отсортированных уникальных массивов; возвращает (J, |A∩B|)."""
    if a.size == 0 or b.size == 0:
        return 0.0, 0
    inter = int(np.intersect1d(a, b, assume_unique=True).size)
    return inter / (a.size + b.size - inter), inter


_PERM_CACHE: dict[tuple[int, int], Any] = {}


def _base_minhash(num_perm: int, seed: int):
    """Эталонный MinHash (перестановки кэшируются, чтобы не генерировать их на каждую запись)."""
    from datasketch import MinHash

    key = (num_perm, seed)
    if key not in _PERM_CACHE:
        _PERM_CACHE[key] = MinHash(num_perm=num_perm, seed=seed, hashfunc=_h32)
    return _PERM_CACHE[key]


def minhash_from_shingles(shingles: np.ndarray, num_perm: int, seed: int):
    """datasketch.MinHash по множеству shingle'ов (младшие 32 бита каждого хэша)."""
    from datasketch import MinHash

    base = _base_minhash(num_perm, seed)
    m = MinHash(num_perm=num_perm, seed=seed, hashfunc=_h32, permutations=base.permutations, scheme=base.scheme)
    if shingles.size:
        raw = (shingles & _MASK32).astype(">u4").tobytes()
        m.update_batch([raw[i : i + 4] for i in range(0, len(raw), 4)])
    return m


def minhash_from_values(hashvalues: np.ndarray, num_perm: int, seed: int):
    from datasketch import MinHash

    base = _base_minhash(num_perm, seed)
    return MinHash(num_perm=num_perm, seed=seed, hashfunc=_h32, hashvalues=hashvalues, permutations=base.permutations, scheme=base.scheme)


def _mh_task(args: tuple[str, str, int, bytes | None, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(shingle-множество, подпись MinHash) записи."""
    code, lang, s, key, num_perm, seed = args
    sh = shingle_set(tokens_for(code, lang, mode="full"), s, key)
    return sh, minhash_from_shingles(sh, num_perm, seed).hashvalues.copy()


class MinHashIndex(BaseIndex):
    """M2: MinHash-LSH кандидаты → точный Жаккар по shingle-множествам."""

    name = "minhash"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.num_perm = DEFAULT_NUM_PERM
        self.shingle = DEFAULT_SHINGLE
        self.threshold = DEFAULT_LSH_THRESHOLD
        self.seed = 1
        self.workers = 1
        self.key: bytes | None = None
        self.hashvalues = np.zeros((0, DEFAULT_NUM_PERM), dtype=np.uint32)
        self.sh_offsets = np.zeros(1, dtype=np.int64)
        self.sh_values = np.zeros(0, dtype=np.uint64)
        self.lsh = None
        self._mem_cache: int | None = None
        self._configure(self.cfg)

    def _configure(self, cfg: dict[str, Any] | None) -> None:
        if not cfg:
            return
        self.cfg = cfg
        fp = cfg.get("fingerprint", {}) or {}
        self.num_perm = int(fp.get("minhash_perm", DEFAULT_NUM_PERM))
        self.shingle = int(fp.get("minhash_shingle", DEFAULT_SHINGLE))
        self.threshold = float(fp.get("minhash_lsh_threshold", DEFAULT_LSH_THRESHOLD))
        self.seed = int(cfg.get("seed", 1)) & 0x7FFFFFFF
        self.workers = int(fp.get("workers", 1) or 1)
        self.key = hmac_key(cfg)

    # --- shingle'ы

    def shingles(self, code: str, lang: str) -> np.ndarray:
        return shingle_set(tokens_for(code, lang, mode="full"), self.shingle, self.key)

    def record_shingles(self, idx: int) -> np.ndarray:
        return self.sh_values[self.sh_offsets[idx] : self.sh_offsets[idx + 1]]

    def _build_lsh(self) -> None:
        from datasketch import MinHashLSH

        t0 = time.perf_counter()
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        for i in range(self.n_records):
            if self.sh_offsets[i + 1] > self.sh_offsets[i]:  # пустые записи не индексируются
                lsh.insert(i, minhash_from_values(self.hashvalues[i], self.num_perm, self.seed), check_duplication=False)
        self.lsh = lsh
        self._mem_cache = None
        log.info("minhash: LSH (b=%d, r=%d) built for %d records in %.1fs", lsh.b, lsh.r, self.n_records, time.perf_counter() - t0)

    # --- построение

    def build(self, records: Iterable[FunctionRecord], cfg: dict[str, Any]) -> None:
        self._configure(cfg)
        self._reset()
        t0 = time.perf_counter()
        ids: list[str] = []
        sh_parts: list[np.ndarray] = []
        counts: list[int] = []
        hv_rows: list[np.ndarray] = []
        n_empty = 0
        with pool_context(self.workers) as pool:
            for chunk in chunked(iter_records(records)):
                tasks = [(r.code, r.lang, self.shingle, self.key, self.num_perm, self.seed) for r in chunk]
                for r, (sh, hv) in zip(chunk, pool_map(pool, _mh_task, tasks)):
                    ids.append(r.id)
                    counts.append(int(sh.size))
                    hv_rows.append(hv)
                    if sh.size:
                        sh_parts.append(sh)
                    else:
                        n_empty += 1
        self.ids = ids
        self.sh_values = np.concatenate(sh_parts) if sh_parts else np.zeros(0, dtype=np.uint64)
        self.sh_offsets = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(np.asarray(counts, dtype=np.int64), out=self.sh_offsets[1:])
        self.hashvalues = np.vstack(hv_rows) if hv_rows else np.zeros((0, self.num_perm), dtype=np.uint32)
        self._build_lsh()
        self.meta = {
            "num_perm": self.num_perm,
            "shingle": self.shingle,
            "threshold": self.threshold,
            "keyed": self.key is not None,
            "n_shingles_total": int(self.sh_values.size),
            "n_empty_records": n_empty,
            "build_seconds": round(time.perf_counter() - t0, 3),
        }
        log.info("minhash: %d records, %d shingles in %.1fs", len(ids), self.sh_values.size, self.meta["build_seconds"])

    # --- запрос

    def query(self, code: str, lang: str) -> QueryResult:
        toks = tokens_for(code, lang, mode="full")
        n_tok = len(toks)
        if n_tok < self.shingle:
            return QueryResult(0.0, None, {"reason": "too_short", "n_tokens": n_tok, "n_shingles": 0})
        sh = shingle_set(toks, self.shingle, self.key)
        details: dict[str, Any] = {"n_tokens": n_tok, "n_shingles": int(sh.size), "n_candidates": 0, "best_shared": 0}
        if self.lsh is None or sh.size == 0:
            return QueryResult(0.0, None, details)
        m = minhash_from_shingles(sh, self.num_perm, self.seed)
        cands = sorted(int(c) for c in self.lsh.query(m))
        details["n_candidates"] = len(cands)
        best, best_j, best_inter = None, -1.0, 0
        for c in cands:
            j, inter = jaccard(sh, self.record_shingles(c))
            if j > best_j:
                best, best_j, best_inter = c, j, inter
        if best is None:
            return QueryResult(0.0, None, details)
        details["best_shared"] = best_inter
        details["containment"] = best_inter / sh.size
        return QueryResult(float(best_j), self.ids[best], details)

    # --- память и сериализация

    def memory_bytes(self) -> int:
        if self._mem_cache is None:
            total = deep_sizeof(self.ids) + int(self.hashvalues.nbytes + self.sh_offsets.nbytes + self.sh_values.nbytes)
            total += self._lsh_bytes()
            self._mem_cache = total
        return self._mem_cache

    def _lsh_bytes(self) -> int:
        """Оценка памяти структур datasketch.MinHashLSH (hashtables + keys)."""
        if self.lsh is None:
            return 0
        total = 0
        for ht in self.lsh.hashtables:
            d = getattr(ht, "_dict", None)
            if d is None:
                continue
            total += sys.getsizeof(d)
            for k, v in d.items():
                total += sys.getsizeof(k) + sys.getsizeof(v) + 28 * len(v)
        kd = getattr(self.lsh.keys, "_dict", None)
        if kd is not None:
            total += sys.getsizeof(kd)
            for k, v in kd.items():
                total += sys.getsizeof(k) + sys.getsizeof(v) + sum(sys.getsizeof(x) for x in v)
        return int(total)

    def _state(self) -> dict[str, Any]:
        return {
            "hashvalues": self.hashvalues,
            "sh_offsets": self.sh_offsets,
            "sh_values": self.sh_values,
            "params": {"num_perm": self.num_perm, "shingle": self.shingle, "threshold": self.threshold, "seed": self.seed, "keyed": self.key is not None},
        }

    def _restore(self, state: dict[str, Any]) -> None:
        p = state.get("params", {})
        self.num_perm = int(p.get("num_perm", self.num_perm))
        self.shingle = int(p.get("shingle", self.shingle))
        self.threshold = float(p.get("threshold", self.threshold))
        self.seed = int(p.get("seed", self.seed))
        self.hashvalues = np.asarray(state["hashvalues"])
        self.sh_offsets = np.asarray(state["sh_offsets"], dtype=np.int64)
        self.sh_values = np.asarray(state["sh_values"], dtype=np.uint64)
        if p.get("keyed") and self.key is None:
            log.warning("minhash index was built with an HMAC key, but none is set now: queries will not match")
        self._build_lsh()
