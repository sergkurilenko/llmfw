"""M2 minhash: MinHash + LSH-banding по shingle'ам abstract-токенов (DESIGN.md §6).

Режим токенов — cfg.fingerprint.token_mode (full по умолчанию; lexical — см. winnowing_index).
Shingle = `minhash_shingle` подряд идущих абстрактных токенов → blake2b-64 (ключ HMAC
из конфига, если задан). Подпись: `minhash_perm` аффинных перестановок
h_i(x) = (a_i·x + b_i) mod (2^31 − 1) над младшими 31 битами хэша shingle'а, минимум по
множеству (numpy, детерминированно по cfg.seed; не зависит от версии datasketch).
Кандидаты: LSH-banding (b полос по r строк; (b, r) подбираются как в datasketch.MinHashLSH —
минимум взвешенной суммы вероятностей ложных срабатываний/пропусков при пороге
`minhash_lsh_threshold`); ключи полос хранятся как отсортированные массивы uint64 с
массивом индексов записей (≈ 12 байт на запись на полосу), запрос — b двоичных поисков.
score = max точного Жаккара по множествам shingle'ов (CSR отсортированных uint64) среди
кандидатов, предварительно отранжированных оценкой Жаккара по подписям (не более
`minhash_max_candidates`). Все структуры — numpy-массивы: arrays.npz без пересборки при загрузке.
"""

from __future__ import annotations

import hashlib
import logging
import time
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np

from smcode.config import hmac_key
from smcode.fingerprint.index import (
    REASON_TOO_SHORT,
    BaseIndex,
    chunked,
    deep_sizeof,
    iter_records,
    pool_context,
    pool_map,
    tokens_or_reason,
)
from smcode.types import FunctionRecord, QueryResult

log = logging.getLogger(__name__)

DEFAULT_NUM_PERM = 128
DEFAULT_SHINGLE = 5
DEFAULT_LSH_THRESHOLD = 0.3
DEFAULT_MAX_CANDIDATES = 200
MERSENNE_31 = (1 << 31) - 1
_MASK31 = np.uint64(MERSENNE_31)
_EMPTY_HASH = np.uint32(MERSENNE_31)  # значение подписи пустого множества
SCHEME = "numpy-affine31-v1"  # формат подписей/ключей полос; индекс другой схемы не загружается
_SIG_CHUNK = 4096  # shingle'ов за один проход (ограничивает (S, P) буфер ~4 МБ)
_BAND_SALT = np.uint64(0x9E3779B97F4A7C15)
_BAND_MULT = np.uint64(0xBF58476D1CE4E5B9)
_BAND_SHIFT = np.uint64(29)


# ----------------------------------------------------------------------------- shingle'ы


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


# ----------------------------------------------------------------------------- подписи MinHash


@lru_cache(maxsize=8)
def minhash_permutations(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Коэффициенты (a, b) аффинных перестановок над Z_p, p = 2^31 − 1 (uint64, детерминированно по seed)."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    a = rng.integers(1, MERSENNE_31, size=num_perm, dtype=np.int64).astype(np.uint64)
    b = rng.integers(0, MERSENNE_31, size=num_perm, dtype=np.int64).astype(np.uint64)
    return a, b


def minhash_signature(shingles: np.ndarray, num_perm: int, seed: int) -> np.ndarray:
    """Подпись MinHash (uint32, num_perm) множества shingle'ов; для пустого множества — все значения = 2^31 − 1."""
    out = np.full(num_perm, MERSENNE_31, dtype=np.uint64)
    sh = np.asarray(shingles, dtype=np.uint64)
    if sh.size:
        a, b = minhash_permutations(num_perm, seed)
        x = sh & _MASK31
        for start in range(0, x.size, _SIG_CHUNK):
            hv = (x[start : start + _SIG_CHUNK, None] * a[None, :] + b[None, :]) % _MASK31  # < 2^63: без переполнения
            np.minimum(out, hv.min(axis=0), out=out)
    return out.astype(np.uint32)


def estimate_jaccard(sig: np.ndarray, sigs: np.ndarray) -> np.ndarray:
    """Оценка Жаккара по подписям: доля совпадающих позиций (для матрицы (M, P) — вектор (M,))."""
    sigs = np.asarray(sigs)
    if sigs.ndim == 1:
        return np.asarray(np.mean(sigs == sig), dtype=np.float64)
    if sigs.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return (sigs == sig[None, :]).mean(axis=1)


# ----------------------------------------------------------------------------- LSH-banding


def _integration_grid(a: float, b: float, p: float = 0.001) -> np.ndarray:
    """Середины шагов численного интегрирования на [a, b) (та же сетка, что в datasketch.lsh._integration)."""
    xs = []
    x = a
    while x < b:
        xs.append(x + 0.5 * p)
        x += p
    return np.asarray(xs, dtype=np.float64)


def optimal_bands(threshold: float, num_perm: int, fp_weight: float = 0.5, fn_weight: float = 0.5) -> tuple[int, int]:
    """(b, r): минимум fp_weight·P(ложное срабатывание) + fn_weight·P(пропуск) при пороге Жаккара threshold
    (та же целевая функция и сетка интегрирования, что у datasketch.MinHashLSH)."""
    p = 0.001
    s_fp = _integration_grid(0.0, threshold, p)
    s_fn = _integration_grid(threshold, 1.0, p)
    best, best_err = (0, 0), float("inf")
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            prob_fp = 1.0 - (1.0 - s_fp**r) ** b
            prob_fn = (1.0 - s_fn**r) ** b
            err = float(prob_fp.sum() * p) * fp_weight + float(prob_fn.sum() * p) * fn_weight
            if err < best_err:
                best, best_err = (b, r), err
    return best


def band_keys(hashvalues: np.ndarray, b: int, r: int) -> np.ndarray:
    """Ключи полос (N, b) uint64: перемешивание r значений подписи каждой полосы с солью номера полосы
    (ключи разных полос различимы, поэтому все полосы хранятся в одном отсортированном массиве)."""
    hv = np.asarray(hashvalues)
    if hv.ndim == 1:
        hv = hv[None, :]
    n = hv.shape[0]
    block = hv[:, : b * r].astype(np.uint64).reshape(n, b, r)
    k = np.broadcast_to(_BAND_SALT ^ np.arange(1, b + 1, dtype=np.uint64), (n, b)).copy()
    for j in range(r):
        k = (k ^ block[:, :, j]) * _BAND_MULT  # uint64-массивы: переполнение — по модулю 2^64
        k ^= k >> _BAND_SHIFT
    return k


class BandIndex:
    """LSH-полосы: все ключи (запись × полоса) в одном отсортированном массиве + индексы записей."""

    __slots__ = ("sorted_keys", "order", "b")

    def __init__(self, sorted_keys: np.ndarray | None = None, order: np.ndarray | None = None, b: int = 0):
        self.sorted_keys = sorted_keys if sorted_keys is not None else np.zeros(0, dtype=np.uint64)
        self.order = order if order is not None else np.zeros(0, dtype=np.int32)
        self.b = int(b)

    @classmethod
    def build(cls, hashvalues: np.ndarray, rec_idx: np.ndarray, b: int, r: int) -> "BandIndex":
        """Полосы для подписей hashvalues (M, P) записей rec_idx (M,)."""
        rec_idx = np.asarray(rec_idx, dtype=np.int32)
        if rec_idx.size == 0:
            return cls(b=b)
        keys = band_keys(hashvalues, b, r).ravel()  # (M*b,): запись-мажорный порядок
        order = np.argsort(keys, kind="stable")
        return cls(keys[order], np.repeat(rec_idx, b)[order], b)

    def candidates(self, sig: np.ndarray, r: int) -> np.ndarray:
        """Индексы записей, совпавших с подписью хотя бы в одной полосе (уникальные, по возрастанию)."""
        if self.b == 0 or self.sorted_keys.size == 0:
            return np.zeros(0, dtype=np.int32)
        qk = band_keys(sig, self.b, r)[0]
        lo = np.searchsorted(self.sorted_keys, qk, side="left")
        hi = np.searchsorted(self.sorted_keys, qk, side="right")
        hit = np.flatnonzero(hi > lo)
        if hit.size == 0:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate([self.order[lo[i] : hi[i]] for i in hit]))

    def nbytes(self) -> int:
        return int(self.sorted_keys.nbytes + self.order.nbytes)

    def arrays(self, prefix: str = "band_") -> dict[str, np.ndarray]:
        return {f"{prefix}keys": self.sorted_keys, f"{prefix}order": self.order}

    @classmethod
    def from_arrays(cls, d: dict[str, Any], b: int, prefix: str = "band_") -> "BandIndex":
        return cls(np.asarray(d[f"{prefix}keys"], dtype=np.uint64), np.asarray(d[f"{prefix}order"], dtype=np.int32), b)


# ----------------------------------------------------------------------------- индекс


def _mh_task(args: tuple[str, str, int, bytes | None, int, int, str]) -> tuple[np.ndarray, np.ndarray, str | None]:
    """(shingle-множество, подпись MinHash, причина отказа токенизации) записи; args = (code, lang, s, key, num_perm, seed, token_mode)."""
    code, lang, s, key, num_perm, seed, mode = args
    toks, reason = tokens_or_reason(code, lang, mode=mode)
    sh = shingle_set(toks, s, key)
    return sh, minhash_signature(sh, num_perm, seed), reason


class MinHashIndex(BaseIndex):
    """M2: LSH-banding кандидаты → точный Жаккар по shingle-множествам."""

    name = "minhash"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        self.num_perm = DEFAULT_NUM_PERM
        self.shingle = DEFAULT_SHINGLE
        self.threshold = DEFAULT_LSH_THRESHOLD
        self.max_candidates = DEFAULT_MAX_CANDIDATES
        self.seed = 1
        self.workers = 1
        self.token_mode = "full"
        self.key: bytes | None = None
        self.b, self.r = 0, 0
        self.hashvalues = np.zeros((0, DEFAULT_NUM_PERM), dtype=np.uint32)
        self.sh_offsets = np.zeros(1, dtype=np.int64)
        self.sh_values = np.zeros(0, dtype=np.uint64)
        self.bands = BandIndex()
        self._configure(self.cfg)

    def _configure(self, cfg: dict[str, Any] | None) -> None:
        if not cfg:
            return
        self.cfg = cfg
        fp = cfg.get("fingerprint", {}) or {}
        self.num_perm = int(fp.get("minhash_perm", DEFAULT_NUM_PERM))
        self.shingle = int(fp.get("minhash_shingle", DEFAULT_SHINGLE))
        self.threshold = float(fp.get("minhash_lsh_threshold", DEFAULT_LSH_THRESHOLD))
        self.max_candidates = int(fp.get("minhash_max_candidates", DEFAULT_MAX_CANDIDATES) or DEFAULT_MAX_CANDIDATES)
        self.seed = int(cfg.get("seed", 1)) & 0x7FFFFFFF
        self.workers = int(fp.get("workers", 1) or 1)
        from smcode.fingerprint.winnowing_index import token_mode_of  # общий разбор fingerprint.token_mode

        self.token_mode = token_mode_of(cfg)
        self.key = hmac_key(cfg)

    # --- shingle'ы и подписи

    def shingles(self, code: str, lang: str) -> np.ndarray:
        return shingle_set(tokens_or_reason(code, lang, mode=self.token_mode)[0], self.shingle, self.key)

    def record_shingles(self, idx: int) -> np.ndarray:
        return self.sh_values[self.sh_offsets[idx] : self.sh_offsets[idx + 1]]

    def signature(self, shingles: np.ndarray) -> np.ndarray:
        return minhash_signature(shingles, self.num_perm, self.seed)

    def _build_bands(self) -> None:
        t0 = time.perf_counter()
        self.b, self.r = optimal_bands(self.threshold, self.num_perm)
        nonempty = np.flatnonzero(np.diff(self.sh_offsets) > 0).astype(np.int32)
        self.bands = BandIndex.build(self.hashvalues[nonempty], nonempty, self.b, self.r)
        log.info("minhash: LSH bands (b=%d, r=%d) built for %d records in %.2fs", self.b, self.r, nonempty.size, time.perf_counter() - t0)

    # --- построение

    def build(self, records: Iterable[FunctionRecord], cfg: dict[str, Any]) -> None:
        self._configure(cfg)
        self._reset()
        t0 = time.perf_counter()
        ids: list[str] = []
        sh_parts: list[np.ndarray] = []
        counts: list[int] = []
        hv_rows: list[np.ndarray] = []
        n_empty = n_unsupported = n_errors = 0
        with pool_context(self.workers) as pool:
            for chunk in chunked(iter_records(records)):
                tasks = [(r.code, r.lang, self.shingle, self.key, self.num_perm, self.seed, self.token_mode) for r in chunk]
                for r, (sh, hv, reason) in zip(chunk, pool_map(pool, _mh_task, tasks)):
                    ids.append(r.id)
                    counts.append(int(sh.size))
                    hv_rows.append(hv)
                    if sh.size:
                        sh_parts.append(sh)
                    elif reason == "unsupported_lang":
                        n_unsupported += 1
                    elif reason == "tokenize_error":
                        n_errors += 1
                    else:
                        n_empty += 1
        self.ids = ids
        self.sh_values = np.concatenate(sh_parts) if sh_parts else np.zeros(0, dtype=np.uint64)
        self.sh_offsets = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(np.asarray(counts, dtype=np.int64), out=self.sh_offsets[1:])
        self.hashvalues = np.vstack(hv_rows).astype(np.uint32) if hv_rows else np.zeros((0, self.num_perm), dtype=np.uint32)
        self._build_bands()
        self.meta = {
            "num_perm": self.num_perm,
            "shingle": self.shingle,
            "token_mode": self.token_mode,
            "threshold": self.threshold,
            "max_candidates": self.max_candidates,
            "scheme": SCHEME,
            "lsh_bands": self.b,
            "lsh_rows": self.r,
            **self.key_params(),
            "n_shingles_total": int(self.sh_values.size),
            "n_empty_records": n_empty,
            "n_unsupported": n_unsupported,
            "n_tokenize_errors": n_errors,
            "build_seconds": round(time.perf_counter() - t0, 3),
        }
        log.info("minhash: %d records, %d shingles in %.1fs", len(ids), self.sh_values.size, self.meta["build_seconds"])

    # --- запрос

    def candidates(self, sig: np.ndarray) -> np.ndarray:
        """Индексы записей-кандидатов LSH для подписи (uint32, num_perm)."""
        return self.bands.candidates(np.asarray(sig, dtype=np.uint32), self.r)

    def query(self, code: str, lang: str) -> QueryResult:
        toks, reason = tokens_or_reason(code, lang, mode=self.token_mode)
        n_tok = len(toks)
        if n_tok < self.shingle:
            return QueryResult(0.0, None, {"reason": reason or REASON_TOO_SHORT, "n_tokens": n_tok, "n_shingles": 0})
        sh = shingle_set(toks, self.shingle, self.key)
        details: dict[str, Any] = {"n_tokens": n_tok, "n_shingles": int(sh.size), "n_candidates": 0, "n_scored": 0, "best_shared": 0}
        if sh.size == 0 or self.n_records == 0:
            return QueryResult(0.0, None, details)
        sig = self.signature(sh)
        cands = self.candidates(sig)
        details["n_candidates"] = int(cands.size)
        if cands.size == 0:
            return QueryResult(0.0, None, details)
        if cands.size > self.max_candidates:  # ранжирование оценкой по подписям, точный Жаккар — для лучших
            est = estimate_jaccard(sig, self.hashvalues[cands])
            top = np.lexsort((cands, -est))[: self.max_candidates]
            cands = np.sort(cands[top])
        details["n_scored"] = int(cands.size)
        best, best_j, best_inter = None, -1.0, 0
        for c in cands.tolist():  # по возрастанию индекса: при равном J побеждает более ранняя запись
            j, inter = jaccard(sh, self.record_shingles(c))
            if j > best_j:
                best, best_j, best_inter = c, j, inter
        if best is None:
            return QueryResult(0.0, None, details)
        details["best_shared"] = best_inter
        details["containment"] = best_inter / sh.size
        details["best_est_jaccard"] = float(estimate_jaccard(sig, self.hashvalues[best]))
        return QueryResult(float(best_j), self.ids[best], details)

    # --- память и сериализация

    def memory_bytes(self) -> int:
        return int(deep_sizeof(self.ids) + self.hashvalues.nbytes + self.sh_offsets.nbytes + self.sh_values.nbytes + self.bands.nbytes())

    def _state(self) -> dict[str, Any]:
        return {
            "hashvalues": self.hashvalues,
            "sh_offsets": self.sh_offsets,
            "sh_values": self.sh_values,
            **self.bands.arrays(),
            "params": {
                "num_perm": self.num_perm, "shingle": self.shingle, "token_mode": self.token_mode, "threshold": self.threshold,
                "seed": self.seed, "max_candidates": self.max_candidates, "b": self.b, "r": self.r, "scheme": SCHEME,
                **self.key_params(),
            },
        }

    def _restore(self, state: dict[str, Any]) -> None:
        p = state.get("params", {}) or {}
        self._check_key(p)
        scheme = p.get("scheme")
        if scheme != SCHEME:
            raise ValueError(f"minhash index scheme {scheme!r} is not supported by this code ({SCHEME!r}); rebuild the index")
        self.num_perm = int(p.get("num_perm", self.num_perm))
        self.shingle = int(p.get("shingle", self.shingle))
        self.token_mode = str(p.get("token_mode", "full"))
        self.threshold = float(p.get("threshold", self.threshold))
        self.seed = int(p.get("seed", self.seed))
        self.max_candidates = int(p.get("max_candidates", self.max_candidates))
        self.b, self.r = int(p.get("b", 0)), int(p.get("r", 0))
        self.hashvalues = np.asarray(state["hashvalues"], dtype=np.uint32)
        self.sh_offsets = np.asarray(state["sh_offsets"], dtype=np.int64)
        self.sh_values = np.asarray(state["sh_values"], dtype=np.uint64)
        if "band_keys" in state and self.b > 0:
            self.bands = BandIndex.from_arrays(state, self.b)
        else:
            self._build_bands()
