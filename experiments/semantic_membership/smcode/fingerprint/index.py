"""Реестр индексов членства, базовый класс и общие структуры хранения (DESIGN.md §6).

REGISTRY отображает имя метода в "module:Class"; make_index лениво импортирует класс
(semantic/hybrid тянут torch только при обращении к ним). BaseIndex реализует
query_batch с замером латентности, save/load (pickle + npz), оценку памяти и проверку
HMAC-ключа (key_id в state.pkl/meta.json: несовпадение ключа при загрузке — ValueError).
InvertedIndex — компактный CSR-словарь hash64 → массив int32 индексов записей,
общий для exact (M0) и winnowing (M1). Артефакты индекса: data/indexes/<method>/
(meta.json, state.pkl, arrays.npz).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import pickle
import sys
import time
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from smcode.config import hmac_key, resolve_path
from smcode.normalize import abstract_tokens, tokenize
from smcode.types import FunctionRecord, MembershipIndex, QueryResult, read_functions

log = logging.getLogger(__name__)

REGISTRY: dict[str, str] = {
    "exact": "smcode.fingerprint.exact:ExactIndex",
    "winnowing": "smcode.fingerprint.winnowing_index:WinnowingIndex",
    "minhash": "smcode.fingerprint.minhash:MinHashIndex",
    "semantic": "smcode.semantic.semantic_index:SemanticIndex",
    "hybrid": "smcode.semantic.hybrid:HybridIndex",
}
FINGERPRINT_METHODS: tuple[str, ...] = ("exact", "winnowing", "minhash")
GPU_METHODS: tuple[str, ...] = ("semantic", "hybrid")

META_FILE = "meta.json"
STATE_FILE = "state.pkl"
ARRAYS_FILE = "arrays.npz"
PROTECTED_FILE = "protected.jsonl"
PROTECTED_WINDOWS_FILE = "protected_windows.jsonl"  # = smcode.data.extract.WINDOWS_FILE
PUBLIC_TRAIN_FILE = "public_train.jsonl"
INDEX_STATS_FILE = "index_stats.json"
CHUNK = 2000
# поля meta.json, которые save() вычисляет заново и которые load() не копирует в self.meta
# (n_records тоже пересчитывается при save(), но остаётся в meta: semantic/hybrid держат его в своём meta)
RESERVED_META = ("name", "class", "saved_at", "memory_bytes")

# причины нулевого скора в QueryResult.details["reason"]
REASON_UNSUPPORTED = "unsupported_lang"
REASON_TOKENIZE_ERROR = "tokenize_error"
REASON_TOO_SHORT = "too_short"


# ----------------------------------------------------------------------------- реестр


def resolve_class(name: str) -> type:
    """Класс индекса по имени из REGISTRY (ленивый импорт 'module:Class')."""
    if name not in REGISTRY:
        raise KeyError(f"unknown index {name!r}; known: {sorted(REGISTRY)}")
    mod_name, _, cls_name = REGISTRY[name].partition(":")
    module = importlib.import_module(mod_name)
    return getattr(module, cls_name)


def make_index(name: str, cfg: dict[str, Any]) -> MembershipIndex:
    """Создаёт индекс по имени метода: Class(cfg). ImportError для GPU-методов без torch."""
    return resolve_class(name)(cfg)


def index_dir(cfg: dict[str, Any], name: str) -> Path:
    """Каталог артефактов индекса: paths.indexes/<name>."""
    return resolve_path(cfg, "indexes") / name


# ----------------------------------------------------------------------------- ключ HMAC


def key_id(key: bytes | None) -> str | None:
    """Публичный идентификатор ключа (sha256[:16]); None — индекс без ключа. Сам ключ не хранится."""
    return hashlib.sha256(key).hexdigest()[:16] if key else None


def key_params(key: bytes | None) -> dict[str, Any]:
    """Поля ключа для params/meta: keyed и key_id."""
    return {"keyed": key is not None, "key_id": key_id(key)}


def check_key(stored: dict[str, Any], key: bytes | None, name: str = "index") -> None:
    """Сверяет ключ, с которым индекс был построен, с текущим. Несовпадение (без ключа → с ключом,
    с ключом → без, другой ключ) — ValueError: иначе все запросы молча получали бы score 0."""
    cur = key_id(key)
    if "key_id" in stored:
        old = stored.get("key_id")
        if old != cur:
            raise ValueError(
                f"{name}: HMAC key mismatch: index built with key_id={old}, current key_id={cur} "
                f"(set the same $SMCODE_INDEX_KEY as at build time, or rebuild the index)"
            )
        return
    # старый формат без key_id: доступен только флаг keyed
    if bool(stored.get("keyed")) != (key is not None):
        raise ValueError(f"{name}: HMAC key mismatch: index keyed={bool(stored.get('keyed'))}, current key set={key is not None}")
    if stored.get("keyed"):
        log.warning("%s: index has no key_id (old format); cannot verify that the current HMAC key matches", name)


def warn_if_unkeyed(cfg: dict[str, Any], what: str = "index") -> bool:
    """Предупреждение, если ключ HMAC не задан (DESIGN §1(d)/§6 требуют ключ в развёртывании). True — ключ есть."""
    if hmac_key(cfg) is None:
        env = cfg.get("fingerprint", {}).get("hmac_key_env", "SMCODE_INDEX_KEY")
        log.warning("building unkeyed %s: set $%s for the real experiment (fingerprints are then HMAC-keyed)", what, env)
        return False
    return True


# ----------------------------------------------------------------------------- утилиты


def iter_records(records: Iterable[FunctionRecord | dict[str, Any]]) -> Iterator[FunctionRecord]:
    """Приводит поток записей (FunctionRecord или dict) к FunctionRecord."""
    for r in records:
        yield r if isinstance(r, FunctionRecord) else FunctionRecord.from_dict(r)


def tokens_or_reason(code: str, lang: str, mode: str = "full") -> tuple[list[str], str | None]:
    """abstract_tokens(tokenize(code, lang), mode) и причина отказа: None (успех),
    'unsupported_lang' (язык вне поддерживаемых) или 'tokenize_error' (падение парсера)."""
    try:
        return abstract_tokens(tokenize(code, lang), mode=mode), None
    except ValueError as exc:  # неподдерживаемый язык
        log.debug("tokenize failed (%s): %s", lang, exc)
        return [], REASON_UNSUPPORTED
    except Exception as exc:  # noqa: BLE001 — не ронять сборку индекса из-за одной записи
        log.warning("tokenize crashed (%s): %s", lang, exc)
        return [], REASON_TOKENIZE_ERROR


def tokens_for(code: str, lang: str, mode: str = "full") -> list[str]:
    """abstract_tokens(tokenize(code, lang), mode); пустой список при неподдерживаемом языке/ошибке парсера."""
    return tokens_or_reason(code, lang, mode)[0]


def chunked(it: Iterable[Any], n: int = CHUNK) -> Iterator[list[Any]]:
    buf: list[Any] = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


@contextmanager
def pool_context(workers: int) -> Iterator[Pool | None]:
    """Пул процессов при workers > 1, иначе None (последовательный режим)."""
    if workers and workers > 1:
        with Pool(workers) as pool:
            yield pool
    else:
        yield None


def pool_map(pool: Pool | None, fn: Callable[[Any], Any], tasks: list[Any]) -> list[Any]:
    """map с сохранением порядка; маленькие пачки считаются в текущем процессе."""
    if pool is None or len(tasks) < 64:
        return [fn(t) for t in tasks]
    return pool.map(fn, tasks, chunksize=max(1, len(tasks) // 32))


def deep_sizeof(obj: Any, _seen: set[int] | None = None) -> int:
    """Приблизительный объём объекта в байтах (контейнеры рекурсивно, numpy — nbytes)."""
    seen = _seen if _seen is not None else set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes) + 112
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_sizeof(k, seen) + deep_sizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_sizeof(x, seen) for x in obj)
    return size


def disk_bytes(path: str | Path) -> int:
    """Суммарный размер файлов каталога."""
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def sorted_contains(sorted_keys: np.ndarray, hs: np.ndarray) -> np.ndarray:
    """Булева маска: hs[i] ∈ sorted_keys (отсортированный uint64) — searchsorted, O(|hs| log |keys|)."""
    hs = np.asarray(hs, dtype=np.uint64)
    if sorted_keys.size == 0 or hs.size == 0:
        return np.zeros(hs.size, dtype=bool)
    pos = np.searchsorted(sorted_keys, hs)
    pos = np.minimum(pos, sorted_keys.size - 1)
    return sorted_keys[pos] == hs


# ----------------------------------------------------------------------------- инвертированный индекс


class InvertedIndex:
    """CSR-словарь hash64 → отсортированный массив int32 индексов записей.

    keys — уникальные хэши по возрастанию (uint64); postings[offsets[i]:offsets[i+1]] —
    записи, содержащие keys[i]. Эквивалент dict[int, np.ndarray] без накладных расходов Python.
    """

    __slots__ = ("keys", "offsets", "postings")

    def __init__(self, keys: np.ndarray | None = None, offsets: np.ndarray | None = None, postings: np.ndarray | None = None):
        self.keys = keys if keys is not None else np.zeros(0, np.uint64)
        self.offsets = offsets if offsets is not None else np.zeros(1, np.int64)
        self.postings = postings if postings is not None else np.zeros(0, np.int32)

    @classmethod
    def from_pairs(cls, hashes: Any, ids: Any) -> "InvertedIndex":
        """Строит индекс из параллельных массивов (hash, record_idx); дубликаты пар схлопываются."""
        h = np.asarray(hashes, dtype=np.uint64)
        p = np.asarray(ids, dtype=np.int32)
        if h.size == 0:
            return cls()
        order = np.lexsort((p, h))
        h, p = h[order], p[order]
        keep = np.ones(h.size, dtype=bool)
        keep[1:] = (h[1:] != h[:-1]) | (p[1:] != p[:-1])
        h, p = h[keep], p[keep]
        keys, counts = np.unique(h, return_counts=True)
        offsets = np.zeros(keys.size + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return cls(keys, offsets, p)

    def __len__(self) -> int:
        return int(self.keys.size)

    def find(self, hs: np.ndarray) -> np.ndarray:
        """Позиции ключей для массива хэшей; −1 для отсутствующих."""
        hs = np.asarray(hs, dtype=np.uint64)
        if self.keys.size == 0 or hs.size == 0:
            return np.full(hs.size, -1, dtype=np.int64)
        pos = np.searchsorted(self.keys, hs)
        pos_c = np.minimum(pos, self.keys.size - 1)
        return np.where(self.keys[pos_c] == hs, pos_c, -1)

    def contains(self, hs: np.ndarray) -> np.ndarray:
        return self.find(hs) >= 0

    def postings_for(self, key_idx: int) -> np.ndarray:
        return self.postings[self.offsets[key_idx] : self.offsets[key_idx + 1]]

    def gather(self, hs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(маска найденных хэшей, конкатенация постингов найденных хэшей)."""
        idx = self.find(hs)
        found = idx >= 0
        parts = [self.postings_for(int(i)) for i in idx[found]]
        post = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
        return found, post

    def df(self) -> np.ndarray:
        """Число записей на ключ."""
        return np.diff(self.offsets)

    def without(self, drop: np.ndarray) -> "InvertedIndex":
        """Копия без указанных ключей (drop — массив uint64, порядок не важен)."""
        drop = np.asarray(drop, dtype=np.uint64)
        if drop.size == 0 or self.keys.size == 0:
            return InvertedIndex(self.keys.copy(), self.offsets.copy(), self.postings.copy())
        keep = ~sorted_contains(np.unique(drop), self.keys)
        counts = self.df()
        sel = np.repeat(keep, counts)
        offsets = np.zeros(int(keep.sum()) + 1, dtype=np.int64)
        np.cumsum(counts[keep], out=offsets[1:])
        return InvertedIndex(self.keys[keep], offsets, self.postings[sel])

    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.offsets.nbytes + self.postings.nbytes)

    def arrays(self, prefix: str = "inv_") -> dict[str, np.ndarray]:
        return {f"{prefix}keys": self.keys, f"{prefix}offsets": self.offsets, f"{prefix}postings": self.postings}

    @classmethod
    def from_arrays(cls, d: dict[str, Any], prefix: str = "inv_") -> "InvertedIndex":
        return cls(
            np.asarray(d[f"{prefix}keys"], dtype=np.uint64),
            np.asarray(d[f"{prefix}offsets"], dtype=np.int64),
            np.asarray(d[f"{prefix}postings"], dtype=np.int32),
        )


def score_hashes(inv: InvertedIndex, hs: np.ndarray) -> tuple[float, int | None, dict[str, Any]]:
    """Доля уникальных хэшей запроса, найденных в индексе; лучшая запись — по числу общих хэшей.

    При равенстве выбирается запись с меньшим индексом, т. е. добавленная в индекс раньше:
    в build_from_corpus функции (protected.jsonl) идут перед файловыми окнами
    (protected_windows.jsonl), поэтому при равном числе общих хэшей best_id — функция.
    Возвращает (score, best_idx | None, details)."""
    hs = np.asarray(hs, dtype=np.uint64)
    details: dict[str, Any] = {"n_found": 0, "n_candidates": 0, "best_shared": 0}
    if hs.size == 0:
        return 0.0, None, details
    found, post = inv.gather(hs)
    n_found = int(found.sum())
    details["n_found"] = n_found
    if post.size == 0:
        return 0.0, None, details
    cand, counts = np.unique(post, return_counts=True)
    j = int(np.argmax(counts))
    details["n_candidates"] = int(cand.size)
    details["best_shared"] = int(counts[j])
    return n_found / hs.size, int(cand[j]), details


# ----------------------------------------------------------------------------- базовый класс


class BaseIndex:
    """Базовая реализация MembershipIndex: латентность в query_batch, save/load, память, ключ HMAC.

    Подклассы реализуют build/query и сериализуемое состояние через _state()/_restore():
    значения np.ndarray сохраняются в arrays.npz, остальное — в state.pkl. Подклассы с ключом
    HMAC держат его в self.key, пишут key_params(self.key) в params и вызывают
    self._check_key(params) в _restore().
    """

    name: str = "base"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg: dict[str, Any] = cfg or {}
        self.ids: list[str] = []
        self.meta: dict[str, Any] = {}
        self.key: bytes | None = None
        self._id_to_idx: dict[str, int] | None = None

    # --- контракт

    @property
    def n_records(self) -> int:
        return len(self.ids)

    def build(self, records: Iterable[FunctionRecord], cfg: dict[str, Any]) -> None:
        raise NotImplementedError

    def query(self, code: str, lang: str) -> QueryResult:
        raise NotImplementedError

    def query_batch(self, items: list[tuple[str, str]]) -> list[QueryResult]:
        """Последовательные запросы; latency_ms каждого заполняется по time.perf_counter."""
        out: list[QueryResult] = []
        for code, lang in items:
            t0 = time.perf_counter()
            res = self.query(code, lang)
            res.latency_ms = (time.perf_counter() - t0) * 1000.0
            out.append(res)
        return out

    def memory_bytes(self) -> int:
        return deep_sizeof(self.ids) + sum(deep_sizeof(v) for v in self._state().values())

    # --- идентификаторы

    def idx_of(self, rec_id: str) -> int | None:
        """Индекс записи по её id (словарь строится лениво)."""
        if self._id_to_idx is None or len(self._id_to_idx) != len(self.ids):
            self._id_to_idx = {rid: i for i, rid in enumerate(self.ids)}
        return self._id_to_idx.get(rec_id)

    def _reset(self) -> None:
        self.ids = []
        self.meta = {}
        self._id_to_idx = None

    # --- ключ

    def key_params(self) -> dict[str, Any]:
        """{'keyed', 'key_id'} текущего ключа (для params и meta)."""
        return key_params(getattr(self, "key", None))

    def _check_key(self, params: dict[str, Any]) -> None:
        """ValueError при несовпадении ключа индекса и текущего ключа (см. check_key)."""
        check_key(params or {}, getattr(self, "key", None), f"{self.name} index")

    # --- сериализация

    def _state(self) -> dict[str, Any]:
        """Сериализуемое состояние (без ids): np.ndarray → npz, остальное → pickle."""
        return {}

    def _restore(self, state: dict[str, Any]) -> None:
        """Обратная операция к _state()."""

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = dict(self._state())
        arrays = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
        rest = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
        rest["ids"] = list(self.ids)
        np.savez(path / ARRAYS_FILE, **arrays)
        with open(path / STATE_FILE, "wb") as f:
            pickle.dump(rest, f, protocol=pickle.HIGHEST_PROTOCOL)
        # зарезервированные поля — после self.meta: свежие значения всегда побеждают устаревшие
        meta = {
            **self.meta,
            **self.key_params(),
            "name": self.name,
            "class": f"{type(self).__module__}:{type(self).__qualname__}",
            "n_records": len(self.ids),
            "memory_bytes": self.memory_bytes(),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(path / META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1, default=str)
        log.info("index %s saved to %s (%d records, %.1f MB on disk)", self.name, path, len(self.ids), disk_bytes(path) / 1e6)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        state_file = path / STATE_FILE
        if not state_file.exists():
            raise FileNotFoundError(f"index state not found: {state_file}")
        with open(state_file, "rb") as f:
            state: dict[str, Any] = pickle.load(f)
        arrays_file = path / ARRAYS_FILE
        if arrays_file.exists():
            with np.load(arrays_file, allow_pickle=False) as z:
                state.update({k: z[k] for k in z.files})
        self._reset()
        self.ids = list(state.pop("ids", []))
        meta_file = path / META_FILE
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                self.meta = {k: v for k, v in json.load(f).items() if k not in RESERVED_META}
        self._restore(state)
        log.info("index %s loaded from %s (%d records)", self.name, path, len(self.ids))


# ----------------------------------------------------------------------------- сборка из корпуса


def load_index(name: str, cfg: dict[str, Any], path: str | Path | None = None) -> MembershipIndex:
    """make_index + load из data/indexes/<name> (или указанного каталога). ValueError при несовпадении ключа."""
    idx = make_index(name, cfg)
    idx.load(path or index_dir(cfg, name))
    return idx


def build_index(
    name: str,
    cfg: dict[str, Any],
    records: Iterable[FunctionRecord | dict[str, Any]],
    out_dir: str | Path | None = None,
    public_records: Iterable[FunctionRecord | dict[str, Any]] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[MembershipIndex, dict[str, Any]]:
    """Создаёт индекс, (для winnowing) подгоняет фильтр общности, строит, сохраняет; возвращает (индекс, статистика).

    meta['build_seconds'] — полное время (подгонка фильтра + сборка), meta['fit_seconds'] — подгонка,
    meta['index_build_seconds'] — собственная сборка индекса; extra_meta (например with_windows) пишется в meta.json."""
    idx = make_index(name, cfg)
    t0 = time.perf_counter()
    fit_seconds = 0.0
    n_public = None
    if public_records is not None and hasattr(idx, "fit_common_filter"):
        n_common = idx.fit_common_filter(public_records)  # type: ignore[attr-defined]
        fit_seconds = time.perf_counter() - t0
        n_public = getattr(idx, "n_public_fitted", None)
        log.info("%s: common filter fitted on public_train (%d common fingerprints, %.1fs)", name, n_common, fit_seconds)
    idx.build(iter_records(records), cfg)
    build_seconds = time.perf_counter() - t0
    meta = getattr(idx, "meta", None)
    if isinstance(meta, dict):
        if "build_seconds" in meta:
            meta["index_build_seconds"] = meta["build_seconds"]
        meta["build_seconds"] = round(build_seconds, 3)
        meta["fit_seconds"] = round(fit_seconds, 3)
        meta["public_filter_fitted"] = bool(public_records is not None and hasattr(idx, "fit_common_filter"))
        if n_public is not None:
            meta["n_public_fitted"] = int(n_public)
        if extra_meta:
            meta.update(extra_meta)
    out = Path(out_dir) if out_dir is not None else index_dir(cfg, name)
    idx.save(out)
    # build_seconds и disk_bytes — в meta.json индекса (T5/summary берут их оттуда, если index_stats.json не содержит варианта)
    on_disk = disk_bytes(out)
    update_meta_file(out, {"build_seconds": round(build_seconds, 3), "disk_bytes": on_disk, "inputs_mtime": time.time()})
    stats = {
        "method": name,
        "n_records": idx.n_records if hasattr(idx, "n_records") else None,
        "build_seconds": round(build_seconds, 3),
        "memory_bytes": int(idx.memory_bytes()),
        "disk_bytes": on_disk,
        "path": str(out),
        "meta": dict(meta or {}),
    }
    return idx, stats


def update_meta_file(out: str | Path, fields: dict[str, Any]) -> None:
    """Дописывает поля в data/indexes/<name>/meta.json (если файл есть)."""
    path = Path(out) / META_FILE
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta.update(fields)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)


def index_inputs(name: str, cfg: dict[str, Any], with_windows: bool = False) -> list[Path]:
    """Существующие входные файлы индекса: protected.jsonl, protected_windows.jsonl (при with_windows), public_train.jsonl
    (фильтр общности winnowing) и готовые эмбеддинги protected[_windows].npz (semantic/hybrid)."""
    fdir = resolve_path(cfg, "functions")
    paths = [fdir / PROTECTED_FILE]
    if with_windows:
        paths.append(fdir / PROTECTED_WINDOWS_FILE)
    if name == "winnowing":
        paths.append(fdir / PUBLIC_TRAIN_FILE)
    if name in GPU_METHODS and "embeddings" in (cfg.get("paths") or {}):
        edir = resolve_path(cfg, "embeddings")
        paths.append(edir / "protected.npz")
        if with_windows:
            paths.append(edir / "protected_windows.npz")
    return [p for p in paths if p.exists()]


def stale_input(meta_path: Path, inputs: Iterable[Path]) -> str | None:
    """Имя входного файла, который новее meta.json индекса (None — индекс актуален)."""
    ref = meta_path.stat().st_mtime_ns
    for p in inputs:
        if p.stat().st_mtime_ns > ref:
            return p.name
    return None


def corpus_records(cfg: dict[str, Any], with_windows: bool = False) -> Iterator[FunctionRecord]:
    """Записи protected.jsonl (+ protected_windows.jsonl при with_windows, если файл есть).
    Функции идут первыми: при равных скорах best_id достаётся функции, а не окну (см. score_hashes)."""
    fdir = resolve_path(cfg, "functions")
    main = fdir / PROTECTED_FILE
    if not main.exists():
        raise FileNotFoundError(f"protected corpus not found: {main} (run scripts/01_extract.py, 02_dedup_split.py)")
    yield from read_functions(main)
    if with_windows:
        win = fdir / PROTECTED_WINDOWS_FILE
        if win.exists():
            yield from read_functions(win)
        else:
            log.warning("windows file not found, building without it: %s", win)


def _skip_stats(name: str, out: Path, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": name,
        "n_records": meta.get("n_records"),
        "build_seconds": meta.get("build_seconds"),
        "memory_bytes": meta.get("memory_bytes"),
        "disk_bytes": meta.get("disk_bytes") if meta.get("disk_bytes") is not None else disk_bytes(out),
        "path": str(out),
        "with_windows": meta.get("with_windows"),
        "skipped": True,
        "meta": {k: v for k, v in meta.items() if k not in RESERVED_META and k != "n_records"},
    }


def build_from_corpus(name: str, cfg: dict[str, Any], with_windows: bool = False, force: bool = False) -> dict[str, Any]:
    """Шаг 04 для одного метода: data/functions → data/indexes/<name>/. Идемпотентно (meta.json = готово и не старше входов).

    --with-windows применяется ко всем методам одинаково (protected_windows.jsonl добавляется после функций)
    и записывается в meta.json; индекс без окон пересобирается, если окна запрошены; индекс с окнами
    не пересобирается при запросе без них (надмножество, предупреждение в лог). Индекс пересобирается, если любой
    входной файл (index_inputs: protected[_windows].jsonl, public_train.jsonl, protected[_windows].npz) новее meta.json —
    как stats.json шага 02 (dedup_split.stats_fresh)."""
    out = index_dir(cfg, name)
    if (out / META_FILE).exists() and not force:
        with open(out / META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        have = meta.get("with_windows")
        win_file = resolve_path(cfg, "functions") / PROTECTED_WINDOWS_FILE
        newer = stale_input(out / META_FILE, index_inputs(name, cfg, with_windows=bool(with_windows or have)))
        if newer is not None:
            log.warning("%s: input %s is newer than %s — rebuilding the index", name, newer, out / META_FILE)
        elif with_windows and have is False and win_file.exists():
            log.warning("%s: index at %s was built without windows but --with-windows requested: rebuilding", name, out)
        else:
            if with_windows and have is False:
                log.warning("%s: windows requested but %s is missing; keeping the index without windows", name, win_file)
            if with_windows and have is None:
                log.warning("%s: index at %s does not record with_windows (old format); skipping, use --force to rebuild", name, out)
            elif have and not with_windows:
                log.info("%s: index at %s contains windows (superset of the request), skipping", name, out)
            else:
                log.info("%s: index exists at %s, skipping (use --force)", name, out)
            return _skip_stats(name, out, meta)
    public: Iterable[FunctionRecord] | None = None
    if name == "winnowing":
        pub_path = resolve_path(cfg, "functions") / PUBLIC_TRAIN_FILE
        if pub_path.exists():
            public = read_functions(pub_path)
        else:
            log.warning("winnowing: %s not found, common filter uses protected files only", pub_path)
    windows_present = with_windows and (resolve_path(cfg, "functions") / PROTECTED_WINDOWS_FILE).exists()
    extra = {"with_windows": bool(windows_present), "windows_requested": bool(with_windows)}
    _, stats = build_index(name, cfg, corpus_records(cfg, with_windows=with_windows), out, public_records=public, extra_meta=extra)
    stats["with_windows"] = bool(windows_present)
    stats["skipped"] = False
    return stats


def write_index_stats(cfg: dict[str, Any], stats: dict[str, dict[str, Any]]) -> Path:
    """Объединяет статистику с results/index_stats.json (по методам, поля сливаются) и записывает его."""
    path = resolve_path(cfg, "results") / INDEX_STATS_FILE
    data: dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            log.warning("corrupt %s, overwriting", path)
    for m, s in stats.items():
        prev = data.get(m) if isinstance(data.get(m), dict) else {}
        merged = {**prev, **{k: v for k, v in s.items() if v is not None or k not in prev}}
        if isinstance(prev.get("meta"), dict) and isinstance(s.get("meta"), dict):
            merged["meta"] = {**prev["meta"], **s["meta"]}
        data[m] = merged
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)
    return path
