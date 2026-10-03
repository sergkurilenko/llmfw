"""Прогон методов по наборам запросов (DESIGN.md §10): results/scores/<method>/<set>.jsonl.

Для каждого метода индекс загружается из data/indexes/<method>/ (make_index + load). Методы с numpy-API
(``score_embeddings(q, top_k)`` у semantic, ``score_embeddings_with_codes(q, codes, langs, top_k)`` у hybrid;
результат — кортеж (scores, best_ids, ...) или список QueryResult) при наличии файла
data/embeddings/queries/<set>.npz (массивы qids, emb) оцениваются без torch; иначе — ``query_batch``.
Строка результата: {"qid", "score", "best_id", "latency_ms"}; рядом пишется <set>.meta.json с памятью
индекса, числом записей, режимом замера латентности и временем прогона.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from smcode.config import ROOT, resolve_path
from smcode.eval.build_queries import QUERY_SETS, WINDOW_SET
from smcode.fingerprint.index import REGISTRY, index_dir, make_index
from smcode.types import QueryResult, read_jsonl, write_jsonl

log = logging.getLogger(__name__)

SCORES_DIR = "scores"
META_SUFFIX = ".meta.json"
DEFAULT_BATCH = 256
QUERY_CHUNK = 500


# ----------------------------------------------------------------------------- пути


def scores_dir(cfg: dict[str, Any], method: str | None = None) -> Path:
    """results/scores[/<method>]."""
    d = resolve_path(cfg, "results") / SCORES_DIR
    if method:
        d = d / method
    d.mkdir(parents=True, exist_ok=True)
    return d


def scores_path(cfg: dict[str, Any], method: str, set_name: str) -> Path:
    return scores_dir(cfg, method) / f"{set_name}.jsonl"


def embeddings_dir(cfg: dict[str, Any]) -> Path:
    """data/embeddings (cfg.paths.embeddings, если задан)."""
    if "embeddings" in cfg.get("paths", {}):
        return resolve_path(cfg, "embeddings")
    return ROOT / "data" / "embeddings"


def query_embeddings_path(cfg: dict[str, Any], set_name: str) -> Path:
    return embeddings_dir(cfg) / "queries" / f"{set_name}.npz"


def queries_path(cfg: dict[str, Any], set_name: str) -> Path:
    return resolve_path(cfg, "queries") / f"{set_name}.jsonl"


def available_sets(cfg: dict[str, Any]) -> list[str]:
    """Наборы запросов, файлы которых существуют (QUERY_SETS + protected_windows)."""
    return [s for s in (*QUERY_SETS, WINDOW_SET) if queries_path(cfg, s).exists()]


def available_methods(cfg: dict[str, Any]) -> list[str]:
    """Методы из REGISTRY, у которых есть каталог индекса с meta.json."""
    return [m for m in REGISTRY if (index_dir(cfg, m) / "meta.json").exists()]


# ----------------------------------------------------------------------------- загрузка индекса


def load_method_index(method: str, cfg: dict[str, Any]) -> Any:
    """make_index + load из data/indexes/<method>/."""
    idx = make_index(method, cfg)
    idx.load(index_dir(cfg, method))
    return idx


def _n_records(idx: Any) -> int | None:
    n = getattr(idx, "n_records", None)
    if n is None and hasattr(idx, "ids"):
        n = len(idx.ids)
    return int(n) if n is not None else None


def _memory_bytes(idx: Any) -> int | None:
    try:
        return int(idx.memory_bytes())
    except Exception as exc:  # noqa: BLE001
        log.warning("memory_bytes() failed: %s", exc)
        return None


# ----------------------------------------------------------------------------- numpy-путь


def load_query_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    """(qids, emb float32 (M, d)) из npz; массивы qids (или ids) и emb."""
    with np.load(path, allow_pickle=False) as z:
        key = "qids" if "qids" in z.files else "ids"
        qids = [str(x) for x in z[key].tolist()]
        emb = np.asarray(z["emb"], dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != len(qids):
        raise ValueError(f"bad embeddings file {path}: emb {emb.shape}, qids {len(qids)}")
    return qids, emb


def _unpack_scores(res: Any, n: int) -> tuple[list[float], list[str | None]]:
    """Приводит результат score_embeddings* к (scores, best_ids): кортеж (scores, best_ids, ...) или список QueryResult."""
    if isinstance(res, (list, tuple)) and res and isinstance(res[0], QueryResult):
        return [float(r.score) for r in res], [r.best_id for r in res]
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        scores = np.asarray(res[0], dtype=np.float64).ravel().tolist()
        best = [None if b is None else str(b) for b in list(res[1])]
        if len(scores) != n or len(best) != n:
            raise ValueError(f"score_embeddings returned {len(scores)} scores / {len(best)} ids for {n} queries")
        return scores, best
    raise TypeError(f"unexpected score_embeddings result: {type(res)}")


def numpy_scorer(idx: Any, cfg: dict[str, Any]) -> Callable[[np.ndarray, list[tuple[str, str]]], tuple[list[float], list[str | None]]] | None:
    """Функция (emb, items) → (scores, best_ids) через numpy-API индекса; None, если API нет."""
    top_k = int(cfg.get("semantic", {}).get("ann_top_k", 20))
    if hasattr(idx, "score_embeddings_with_codes"):
        def _hybrid(emb: np.ndarray, items: list[tuple[str, str]]):
            codes, langs = [c for c, _ in items], [l for _, l in items]
            try:
                res = idx.score_embeddings_with_codes(emb, codes, langs, top_k=top_k)
            except TypeError:  # реализация без top_k
                res = idx.score_embeddings_with_codes(emb, codes, langs)
            return _unpack_scores(res, emb.shape[0])
        return _hybrid
    if hasattr(idx, "score_embeddings"):
        def _semantic(emb: np.ndarray, items: list[tuple[str, str]]):
            try:
                res = idx.score_embeddings(emb, top_k=top_k)
            except TypeError:  # реализация без top_k
                res = idx.score_embeddings(emb)
            return _unpack_scores(res, emb.shape[0])
        return _semantic
    return None


# ----------------------------------------------------------------------------- оценка одного набора


def _progress(it: Iterable[Any], total: int, desc: str) -> Iterable[Any]:
    try:
        from tqdm import tqdm

        return tqdm(it, total=total, desc=desc, unit="q", disable=total < 200)
    except Exception:  # pragma: no cover
        return it


def score_set(
    idx: Any,
    queries: Sequence[dict[str, Any]],
    emb: tuple[list[str], np.ndarray] | None = None,
    cfg: dict[str, Any] | None = None,
    batch: int = DEFAULT_BATCH,
    desc: str = "eval",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Скоры для списка запросов. С emb (qids, matrix) и numpy-API индекса — батчевый путь без torch
    (latency_ms = время батча / размер батча); иначе query_batch (latency_ms на запрос). Возвращает (строки, мета)."""
    cfg = cfg or {}
    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    scorer = numpy_scorer(idx, cfg) if emb is not None else None
    if emb is not None and scorer is not None:
        qids, matrix = emb
        pos = {q: i for i, q in enumerate(qids)}
        missing = [q["qid"] for q in queries if q["qid"] not in pos]
        if missing:
            log.warning("%s: %d запросов без эмбеддинга пропущено (напр. %s)", desc, len(missing), missing[0])
        sel = [q for q in queries if q["qid"] in pos]
        n_chunks = (len(sel) + batch - 1) // batch
        for c in _progress(range(n_chunks), n_chunks, desc):
            chunk = sel[c * batch:(c + 1) * batch]
            e = matrix[[pos[q["qid"]] for q in chunk]]
            items = [(q["code"], q["lang"]) for q in chunk]
            t0 = time.perf_counter()
            scores, best = scorer(e, items)
            per = (time.perf_counter() - t0) * 1000.0 / max(1, len(chunk))
            for q, s, b in zip(chunk, scores, best):
                rows.append({"qid": q["qid"], "score": float(s), "best_id": b, "latency_ms": per})
        mode = f"batch_amortized(batch={batch})"
        n_missing = len(missing)
    else:
        if emb is not None:
            log.info("%s: у индекса нет numpy-API, используется query_batch", desc)
        n_chunks = (len(queries) + QUERY_CHUNK - 1) // QUERY_CHUNK
        for c in _progress(range(n_chunks), n_chunks, desc):
            chunk = queries[c * QUERY_CHUNK:(c + 1) * QUERY_CHUNK]
            res = idx.query_batch([(q["code"], q["lang"]) for q in chunk])
            for q, r in zip(chunk, res):
                rows.append({"qid": q["qid"], "score": float(r.score), "best_id": r.best_id,
                             "latency_ms": None if r.latency_ms is None else float(r.latency_ms)})
        mode = "per_query"
        n_missing = 0
    meta = {
        "n_queries": len(queries), "n_scored": len(rows), "n_missing_embeddings": n_missing,
        "latency_mode": mode, "wall_seconds": round(time.perf_counter() - t_start, 3),
    }
    return rows, meta


def benchmark_latency(idx: Any, items: Sequence[tuple[str, str]], repeats: int, emb: np.ndarray | None = None,
                      cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Повторный замер латентности одиночных запросов (батч 1): p50/p95 по всем (запрос × повтор)."""
    scorer = numpy_scorer(idx, cfg or {}) if emb is not None else None
    lat: list[float] = []
    for i, (code, lang) in enumerate(items):
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            if scorer is not None and emb is not None:
                scorer(emb[i:i + 1], [(code, lang)])
            else:
                idx.query(code, lang)
            lat.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(lat, dtype=np.float64)
    return {"n_queries": len(items), "repeats": repeats, "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean()),
            "mode": "numpy_batch1" if scorer is not None else "query"}


# ----------------------------------------------------------------------------- оценка метода


def eval_method(cfg: dict[str, Any], method: str, sets: Iterable[str] | None = None, force: bool = False,
                batch: int = DEFAULT_BATCH, limit: int | None = None, bench: int = 0) -> dict[str, Any]:
    """Шаг 07 для одного метода: все наборы → results/scores/<method>/<set>.jsonl (+ .meta.json).
    Идемпотентно: готовые файлы пропускаются без force. Возвращает {set: meta}."""
    sets = list(sets) if sets else available_sets(cfg)
    todo = [s for s in sets if force or not scores_path(cfg, method, s).exists()]
    out: dict[str, Any] = {}
    for s in sets:
        if s not in todo:
            log.info("%s/%s: scores exist, skipping (use --force)", method, s)
            out[s] = {"skipped": True}
    if not todo:
        return out
    idx = load_method_index(method, cfg)
    n_rec, mem = _n_records(idx), _memory_bytes(idx)
    log.info("%s: index loaded (%s records, %.1f MB)", method, n_rec, (mem or 0) / 1e6)
    repeats = int(cfg.get("eval", {}).get("latency_repeats", 200))
    for s in todo:
        qpath = queries_path(cfg, s)
        if not qpath.exists():
            log.warning("%s: queries file not found: %s", s, qpath)
            continue
        queries = list(read_jsonl(qpath))
        if limit:
            queries = queries[:limit]
        emb = None
        epath = query_embeddings_path(cfg, s)
        if epath.exists() and numpy_scorer(idx, cfg) is not None:
            emb = load_query_embeddings(epath)
            log.info("%s/%s: numpy path with %s (%d embeddings)", method, s, epath.name, len(emb[0]))
        rows, meta = score_set(idx, queries, emb=emb, cfg=cfg, batch=batch, desc=f"{method}/{s}")
        meta.update({"method": method, "set": s, "n_records": n_rec, "memory_bytes": mem,
                     "index_dir": str(index_dir(cfg, method)), "limit": limit})
        if bench > 0 and rows:
            sample = queries[:bench]
            e_s = None
            if emb is not None:
                pos = {q: i for i, q in enumerate(emb[0])}
                sample = [q for q in sample if q["qid"] in pos]
                e_s = emb[1][[pos[q["qid"]] for q in sample]] if sample else None
            if sample:
                meta["latency_benchmark"] = benchmark_latency(idx, [(q["code"], q["lang"]) for q in sample], repeats, e_s, cfg)
        n = write_jsonl(scores_path(cfg, method, s), rows)
        with open(scores_dir(cfg, method) / f"{s}{META_SUFFIX}", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        log.info("%s/%s: %d scores → %s (%.1fs)", method, s, n, scores_path(cfg, method, s), meta["wall_seconds"])
        out[s] = meta
    return out


def run_eval(cfg: dict[str, Any], methods: Iterable[str] | None = None, sets: Iterable[str] | None = None,
             force: bool = False, batch: int = DEFAULT_BATCH, limit: int | None = None, bench: int = 0) -> dict[str, Any]:
    """Шаг 07 для списка методов (по умолчанию — все с готовым индексом). ImportError GPU-методов → пропуск."""
    methods = list(methods) if methods else available_methods(cfg)
    result: dict[str, Any] = {}
    for m in methods:
        try:
            result[m] = eval_method(cfg, m, sets=sets, force=force, batch=batch, limit=limit, bench=bench)
        except ImportError as exc:
            log.error("%s: пропуск — отсутствуют зависимости (torch/faiss?): %s", m, exc)
            result[m] = {"error": f"ImportError: {exc}"}
        except FileNotFoundError as exc:
            log.error("%s: %s", m, exc)
            result[m] = {"error": str(exc)}
    return result


def read_scores(cfg: dict[str, Any], method: str, set_name: str) -> list[dict[str, Any]]:
    """Строки results/scores/<method>/<set>.jsonl (пустой список, если файла нет)."""
    p = scores_path(cfg, method, set_name)
    return list(read_jsonl(p)) if p.exists() else []


def read_scores_meta(cfg: dict[str, Any], method: str, set_name: str) -> dict[str, Any]:
    p = scores_dir(cfg, method) / f"{set_name}{META_SUFFIX}"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
