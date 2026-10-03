"""Реестр индексов членства, базовый класс и общие структуры хранения (DESIGN.md §6).

REGISTRY отображает имя метода в "module:Class"; make_index лениво импортирует класс
(semantic/hybrid тянут torch только при обращении к ним). BaseIndex реализует
query_batch с замером латентности, save/load (pickle + npz) и оценку памяти.
InvertedIndex — компактный CSR-словарь hash64 → массив int32 индексов записей,
общий для exact (M0) и winnowing (M1). Артефакты индекса: data/indexes/<method>/
(meta.json, state.pkl, arrays.npz).
"""

from __future__ import annotations

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

from smcode.config import resolve_path
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


# ----------------------------------------------------------------------------- утилиты


def iter_records(records: Iterable[FunctionRecord | dict[str, Any]]) -> Iterator[FunctionRecord]:
    """Приводит поток записей (FunctionRecord или dict) к FunctionRecord."""
    for r in records:
        yield r if isinstance(r, FunctionRecord) else FunctionRecord.from_dict(r)


def tokens_for(code: str, lang: str, mode: str = "full") -> list[str]:
    """abstract_tokens(tokenize(code, lang), mode); пустой список при неподдерживаемом языке/ошибке парсера."""
    try:
        return abstract_tokens(tokenize(code, lang), mode=mode)
    except ValueError as exc:  # неподдерживаемый язык
        log.debug("tokenize failed (%s): %s", lang, exc)
        return []
    except Exception as exc:  # noqa: BLE001 — не ронять сборку индекса из-за одной записи
        log.warning("tokenize crashed (%s): %s", lang, exc)
        return []


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
        """Копия без указанных ключей."""
        drop = np.asarray(drop, dtype=np.uint64)
        if drop.size == 0 or self.keys.size == 0:
            return InvertedIndex(self.keys.copy(), self.offsets.copy(), self.postings.copy())
        keep = ~np.isin(self.keys, drop)
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
    """Доля уникальных хэшей запроса, найденных в индексе; лучшая запись — по числу общих хэшей
    (при равенстве — с меньшим индексом). Возвращает (score, best_idx | None, details)."""
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
    """Базовая реализация MembershipIndex: латентность в query_batch, save/load, память.

    Подклассы реализуют build/query и сериализуемое состояние через _state()/_restore():
    значения np.ndarray сохраняются в arrays.npz, остальное — в state.pkl.
    """

    name: str = "base"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg: dict[str, Any] = cfg or {}
        self.ids: list[str] = []
        self.meta: dict[str, Any] = {}
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
        meta = {
            "name": self.name,
            "class": f"{type(self).__module__}:{type(self).__qualname__}",
            "n_records": len(self.ids),
            "memory_bytes": self.memory_bytes(),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **self.meta,
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
                self.meta = json.load(f)
        self._restore(state)
        log.info("index %s loaded from %s (%d records)", self.name, path, len(self.ids))


# ----------------------------------------------------------------------------- сборка из корпуса


def load_index(name: str, cfg: dict[str, Any], path: str | Path | None = None) -> MembershipIndex:
    """make_index + load из data/indexes/<name> (или указанного каталога)."""
    idx = make_index(name, cfg)
    idx.load(path or index_dir(cfg, name))
    return idx


def build_index(
    name: str,
    cfg: dict[str, Any],
    records: Iterable[FunctionRecord | dict[str, Any]],
    out_dir: str | Path | None = None,
    public_records: Iterable[FunctionRecord | dict[str, Any]] | None = None,
) -> tuple[MembershipIndex, dict[str, Any]]:
    """Создаёт индекс, (для winnowing) подгоняет фильтр общности, строит, сохраняет; возвращает (индекс, статистика)."""
    idx = make_index(name, cfg)
    t0 = time.perf_counter()
    if public_records is not None and hasattr(idx, "fit_common_filter"):
        n_common = idx.fit_common_filter(public_records)  # type: ignore[attr-defined]
        log.info("%s: common filter fitted on public_train (%d common fingerprints)", name, n_common)
    idx.build(iter_records(records), cfg)
    build_seconds = time.perf_counter() - t0
    out = Path(out_dir) if out_dir is not None else index_dir(cfg, name)
    idx.save(out)
    stats = {
        "method": name,
        "n_records": idx.n_records if hasattr(idx, "n_records") else None,
        "build_seconds": round(build_seconds, 3),
        "memory_bytes": int(idx.memory_bytes()),
        "disk_bytes": disk_bytes(out),
        "path": str(out),
        "meta": dict(getattr(idx, "meta", {}) or {}),
    }
    return idx, stats


def corpus_records(cfg: dict[str, Any], with_windows: bool = False) -> Iterator[FunctionRecord]:
    """Записи protected.jsonl (+ protected_windows.jsonl при with_windows, если файл есть)."""
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


def build_from_corpus(name: str, cfg: dict[str, Any], with_windows: bool = False, force: bool = False) -> dict[str, Any]:
    """Шаг 04 для одного метода: data/functions → data/indexes/<name>/. Идемпотентно (meta.json = готово)."""
    out = index_dir(cfg, name)
    if (out / META_FILE).exists() and not force:
        with open(out / META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        log.info("%s: index exists at %s, skipping (use --force)", name, out)
        return {
            "method": name,
            "n_records": meta.get("n_records"),
            "build_seconds": meta.get("build_seconds"),
            "memory_bytes": meta.get("memory_bytes"),
            "disk_bytes": disk_bytes(out),
            "path": str(out),
            "skipped": True,
        }
    public: Iterable[FunctionRecord] | None = None
    use_windows = with_windows and name == "winnowing"
    if name == "winnowing":
        pub_path = resolve_path(cfg, "functions") / PUBLIC_TRAIN_FILE
        if pub_path.exists():
            public = read_functions(pub_path)
        else:
            log.warning("winnowing: %s not found, common filter uses protected files only", pub_path)
    _, stats = build_index(name, cfg, corpus_records(cfg, with_windows=use_windows), out, public_records=public)
    stats["with_windows"] = use_windows
    stats["skipped"] = False
    return stats


def write_index_stats(cfg: dict[str, Any], stats: dict[str, dict[str, Any]]) -> Path:
    """Объединяет статистику с results/index_stats.json и записывает его."""
    path = resolve_path(cfg, "results") / INDEX_STATS_FILE
    data: dict[str, Any] = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            log.warning("corrupt %s, overwriting", path)
    data.update(stats)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)
    return path
