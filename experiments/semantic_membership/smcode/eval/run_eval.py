"""Прогон методов по наборам запросов (DESIGN.md §10): results/scores/<method>/<set>.jsonl.

Для каждого метода индекс загружается из data/indexes/<method>/ (make_index + load). Методы с numpy-API
(``score_embeddings(q, top_k)`` у semantic, ``score_embeddings_with_codes(q, codes, langs, top_k)`` у hybrid;
результат — кортеж (scores, best_ids, ...) или список QueryResult) при наличии файла
data/embeddings/queries/<set>.npz (массивы qids, emb) оцениваются без torch; иначе — ``query_batch``.
Строка результата: {"qid", "score", "best_id", "latency_ms"} (+ "reason" и для M1 "n_fps"/"n_fps_filtered" из
QueryResult.details, если индекс их вернул: too_short / all_common / unsupported_lang / tokenize_error — разбор
пропусков в metrics.misses_by_reason и T2); рядом пишется <set>.meta.json с памятью индекса, числом записей,
режимом замера латентности и временем прогона.

Идемпотентность по содержимому: файл скоров считается готовым, только если он не старше data/queries/<set>.jsonl
и meta.json индекса (иначе пересчёт с сообщением — как stats.json шага 02). Запросы без эмбеддинга на numpy-пути —
ошибка (ValueError) при полном прогоне: укажите scripts/06_embed.py --sets <set> (дозапись недостающих строк).

Режимы латентности (meta["latency_mode"]): ``per_query`` — точное время query() на запрос;
``amortized_batch(batch=N)`` — query_batch индекса кодирует N запросов разом (semantic/hybrid, torch-путь);
``batch_amortized(batch=N)`` — numpy-путь без энкодера (только ANN/переранжирование, время батча / N).
Для сквозной латентности одного запроса используйте ``--bench N`` (батч 1) — см. smcode.eval.metrics.end_to_end_latency.

Варианты (абляции, DESIGN §6 «правило двух порогов», §7 zero-shot и т. п.): ``eval_method(..., out_name="hybrid_rule",
index_path=..., cfg_overrides={...})`` пишет скоры в results/scores/<out_name>/ и фиксирует в <set>.meta.json базовый
метод, каталог индекса и переопределения конфига; CLI: ``07_eval.py --methods hybrid --variant hybrid_rule
--set semantic.hybrid_rule=two_threshold [--index-dir PATH]``.

``--limit N`` (отладка) пишет <set>.limit<N>.jsonl / .limit<N>.meta.json и никогда не трогает полный файл.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import yaml

from smcode.config import ROOT, resolve_path
from smcode.eval.build_queries import QUERY_SETS, WINDOW_SET, all_sets  # noqa: F401 — QUERY_SETS/WINDOW_SET реэкспорт
from smcode.fingerprint.index import REGISTRY, index_dir, make_index
from smcode.types import QueryResult, read_jsonl, write_jsonl

log = logging.getLogger(__name__)

SCORES_DIR = "scores"
META_SUFFIX = ".meta.json"
LIMIT_TAG = ".limit"
DEFAULT_BATCH = 256
QUERY_CHUNK = 500
AMORTIZED_MODE = "amortized_batch"   # details["latency_mode"] у SemanticIndex/HybridIndex.query_batch
PER_QUERY_MODE = "per_query"
DETAIL_FIELDS: tuple[str, ...] = ("reason", "n_fps", "n_fps_filtered")  # поля details, сохраняемые в строке скоров


# ----------------------------------------------------------------------------- пути


def scores_dir(cfg: dict[str, Any], method: str | None = None) -> Path:
    """results/scores[/<method>]."""
    d = resolve_path(cfg, "results") / SCORES_DIR
    if method:
        d = d / method
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stem(set_name: str, limit: int | None = None) -> str:
    return f"{set_name}{LIMIT_TAG}{int(limit)}" if limit else set_name


def scores_path(cfg: dict[str, Any], method: str, set_name: str, limit: int | None = None) -> Path:
    """results/scores/<method>/<set>.jsonl (с limit — <set>.limit<N>.jsonl)."""
    return scores_dir(cfg, method) / f"{_stem(set_name, limit)}.jsonl"


def meta_path(cfg: dict[str, Any], method: str, set_name: str, limit: int | None = None) -> Path:
    return scores_dir(cfg, method) / f"{_stem(set_name, limit)}{META_SUFFIX}"


def is_limit_file(path: str | Path) -> bool:
    """True для усечённых файлов <set>.limit<N>.jsonl (отладочные прогоны --limit)."""
    return LIMIT_TAG in Path(path).name


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
    """Наборы запросов, файлы которых существуют (QUERY_SETS + окна: protected_windows и <split>_windows из cfg.windows.splits)."""
    return [s for s in all_sets(cfg) if queries_path(cfg, s).exists()]


def available_methods(cfg: dict[str, Any]) -> list[str]:
    """Методы из REGISTRY, у которых есть каталог индекса с meta.json."""
    return [m for m in REGISTRY if (index_dir(cfg, m) / "meta.json").exists()]


# ----------------------------------------------------------------------------- переопределения конфига


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def parse_overrides(items: Iterable[str] | None) -> dict[str, Any]:
    """['semantic.hybrid_rule=two_threshold', 'semantic.ann_top_k=50'] → вложенный словарь; значения — YAML-литералы."""
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        value = yaml.safe_load(raw) if raw.strip() != "" else None
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {item!r} conflicts with a scalar at {p!r}")
        node[parts[-1]] = value
    return out


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Копия cfg с глубоким слиянием overrides (исходный cfg не меняется)."""
    out = copy.deepcopy(cfg)
    if overrides:
        _deep_update(out, overrides)
    return out


# ----------------------------------------------------------------------------- загрузка индекса


def load_method_index(method: str, cfg: dict[str, Any], path: str | Path | None = None) -> Any:
    """make_index + load из data/indexes/<method>/ (или указанного каталога)."""
    idx = make_index(method, cfg)
    idx.load(Path(path) if path is not None else index_dir(cfg, method))
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


def _query_batch_size(idx: Any) -> int | None:
    """Размер батча query_batch индекса (semantic: query_batch_size; hybrid: sem.query_batch_size)."""
    for obj in (idx, getattr(idx, "sem", None)):
        bs = getattr(obj, "query_batch_size", None)
        if bs is not None:
            return int(bs)
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


def index_model(idx: Any) -> str | None:
    """model_key модели индекса semantic/hybrid (атрибут model_name, у hybrid — sem.model_name); None у отпечатков."""
    for obj in (idx, getattr(idx, "sem", None)):
        name = getattr(obj, "model_name", None)
        if name:
            return str(name)
    return None


def check_embeddings_model(idx: Any, path: Path) -> str | None:
    """Сверяет модель эмбеддингов запросов (meta.model в npz, пишет scripts/06_embed.py) с моделью индекса.

    Несовпадение — ValueError: иначе скоры считались бы между векторами разных энкодеров и молча были бы мусором
    (например, индекс дообученной модели и zero-shot эмбеддинги запросов). Возвращает модель файла (None, если
    в файле нет meta — старый формат, проверка невозможна, предупреждение)."""
    want = index_model(idx)
    if want is None:
        return None
    try:
        from smcode.semantic.embed import read_embeddings_meta
        from smcode.semantic.model import model_key
    except ImportError:  # pragma: no cover — модуль semantic отсутствует
        return None
    have = read_embeddings_meta(path).get("model")
    if not have:
        log.warning("%s: в файле эмбеддингов нет meta.model — соответствие модели индекса (%s) не проверено", path.name, want)
        return None
    if model_key(have) != model_key(want):
        raise ValueError(
            f"query embeddings {path} were produced by model {have!r}, but the index was built with {want!r}: "
            "re-run scripts/06_embed.py with the index's checkpoint (or point paths.embeddings / --set semantic.checkpoint "
            "at the matching data/embeddings/by_model/<tag>/ cache)"
        )
    return str(have)


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
    (latency_ms = время батча / размер батча, режим batch_amortized); иначе query_batch: режим per_query, либо
    amortized_batch(batch=N), если индекс сам амортизирует батч (details["latency_mode"]). Возвращает (строки, мета)."""
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
        modes_seen: set[str] = set()
        for c in _progress(range(n_chunks), n_chunks, desc):
            chunk = queries[c * QUERY_CHUNK:(c + 1) * QUERY_CHUNK]
            res = idx.query_batch([(q["code"], q["lang"]) for q in chunk])
            for q, r in zip(chunk, res):
                d = r.details if isinstance(r.details, dict) else {}
                if d.get("latency_mode"):
                    modes_seen.add(str(d["latency_mode"]))
                row = {"qid": q["qid"], "score": float(r.score), "best_id": r.best_id,
                       "latency_ms": None if r.latency_ms is None else float(r.latency_ms)}
                row.update({k: d[k] for k in DETAIL_FIELDS if d.get(k) is not None})
                rows.append(row)
        if AMORTIZED_MODE in modes_seen:
            bs = _query_batch_size(idx)
            mode = f"{AMORTIZED_MODE}(batch={bs})" if bs else AMORTIZED_MODE
            log.info("%s: query_batch амортизирует батч (%s) — latency_ms не является временем одного запроса, "
                     "используйте --bench N", desc, mode)
        else:
            mode = PER_QUERY_MODE
        n_missing = 0
    meta = {
        "n_queries": len(queries), "n_scored": len(rows), "n_missing_embeddings": n_missing,
        "latency_mode": mode, "wall_seconds": round(time.perf_counter() - t_start, 3),
    }
    return rows, meta


def benchmark_latency(idx: Any, items: Sequence[tuple[str, str]], repeats: int, emb: np.ndarray | None = None,
                      cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Повторный замер латентности одиночных запросов (батч 1): p50/p95 по всем (запрос × повтор).

    mode = "numpy_batch1" — numpy-API по готовому эмбеддингу (без энкодера); "query" — idx.query(code, lang)
    (у semantic/hybrid включает энкодер)."""
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


def _is_done(cfg: dict[str, Any], out_name: str, set_name: str, limit: int | None, index_path: str | Path | None = None) -> bool:
    """Файл скоров готов (идемпотентность). Полный файл, записанный старым --limit (meta.limit), готовым не считается;
    файл старше data/queries/<set>.jsonl или meta.json индекса (index_path) — тоже (пересчёт с сообщением)."""
    spath = scores_path(cfg, out_name, set_name, limit)
    if not spath.exists():
        return False
    if limit is None:
        old = read_scores_meta(cfg, out_name, set_name).get("limit")
        if old:
            log.warning("%s/%s: полный файл скоров записан отладочным прогоном --limit %s — пересчёт", out_name, set_name, old)
            return False
    smt = spath.stat().st_mtime_ns
    qpath = queries_path(cfg, set_name)
    if qpath.exists() and qpath.stat().st_mtime_ns > smt:
        log.warning("%s/%s: запросы %s новее файла скоров — пересчёт", out_name, set_name, qpath.name)
        return False
    if index_path is not None:
        mpath = Path(index_path) / "meta.json"
        if mpath.exists() and mpath.stat().st_mtime_ns > smt:
            log.warning("%s/%s: индекс %s новее файла скоров — пересчёт", out_name, set_name, mpath)
            return False
    return True


def eval_method(cfg: dict[str, Any], method: str, sets: Iterable[str] | None = None, force: bool = False,
                batch: int = DEFAULT_BATCH, limit: int | None = None, bench: int = 0,
                out_name: str | None = None, index_path: str | Path | None = None,
                cfg_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Шаг 07 для одного метода: все наборы → results/scores/<out_name>/<set>.jsonl (+ .meta.json).

    out_name — имя каталога скоров (по умолчанию = method; абляции: hybrid_rule, semantic_zeroshot, ...);
    index_path — каталог индекса (по умолчанию data/indexes/<method>); cfg_overrides — переопределения конфига
    варианта (например {"semantic": {"hybrid_rule": "two_threshold"}}), фиксируются в meta.json.
    Идемпотентно: готовые файлы пропускаются без force. limit → отдельные файлы <set>.limit<N>.*. Возвращает {set: meta}."""
    if method not in REGISTRY:
        raise KeyError(f"unknown method {method!r}; known: {sorted(REGISTRY)}")
    cfg = apply_overrides(cfg, cfg_overrides) if cfg_overrides else cfg
    out_name = out_name or method
    index_path = Path(index_path) if index_path is not None else index_dir(cfg, method)
    sets = list(sets) if sets else available_sets(cfg)
    todo = [s for s in sets if force or not _is_done(cfg, out_name, s, limit, index_path)]
    out: dict[str, Any] = {}
    for s in sets:
        if s not in todo:
            log.info("%s/%s: scores exist, skipping (use --force)", out_name, s)
            out[s] = {"skipped": True}
    if not todo:
        return out
    idx = load_method_index(method, cfg, index_path)
    n_rec, mem = _n_records(idx), _memory_bytes(idx)
    log.info("%s (%s @ %s): index loaded (%s records, %.1f MB)", out_name, method, index_path, n_rec, (mem or 0) / 1e6)
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
            check_embeddings_model(idx, epath)
            emb = load_query_embeddings(epath)
            log.info("%s/%s: numpy path with %s (%d embeddings)", out_name, s, epath.name, len(emb[0]))
        rows, meta = score_set(idx, queries, emb=emb, cfg=cfg, batch=batch, desc=f"{out_name}/{s}")
        if meta.get("n_missing_embeddings") and limit is None:
            raise ValueError(
                f"{out_name}/{s}: {meta['n_missing_embeddings']} of {len(queries)} queries have no embedding in {epath} "
                f"(queries added after step 06?): run scripts/06_embed.py --config <cfg> --sets {s} (the missing rows are appended)"
            )
        spath = scores_path(cfg, out_name, s, limit)
        meta.update({"method": method, "variant": out_name, "set": s, "n_records": n_rec, "memory_bytes": mem,
                     "index_dir": str(index_path), "cfg_overrides": cfg_overrides or {}, "limit": limit,
                     "scores_path": str(spath)})
        if bench > 0 and rows:
            sample = queries[:bench]
            e_s = None
            if emb is not None:
                pos = {q: i for i, q in enumerate(emb[0])}
                sample = [q for q in sample if q["qid"] in pos]
                e_s = emb[1][[pos[q["qid"]] for q in sample]] if sample else None
            if sample:
                meta["latency_benchmark"] = benchmark_latency(idx, [(q["code"], q["lang"]) for q in sample], repeats, e_s, cfg)
        n = write_jsonl(spath, rows)
        with open(meta_path(cfg, out_name, s, limit), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        log.info("%s/%s: %d scores → %s (%.1fs)", out_name, s, n, spath, meta["wall_seconds"])
        out[s] = meta
    return out


def run_eval(cfg: dict[str, Any], methods: Iterable[str] | None = None, sets: Iterable[str] | None = None,
             force: bool = False, batch: int = DEFAULT_BATCH, limit: int | None = None, bench: int = 0,
             out_name: str | None = None, index_path: str | Path | None = None,
             cfg_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Шаг 07 для списка методов (по умолчанию — все с готовым индексом). ImportError GPU-методов → пропуск.
    out_name / index_path задают вариант и допустимы только для одного метода."""
    methods = list(methods) if methods else available_methods(cfg)
    if (out_name or index_path is not None) and len(methods) != 1:
        raise ValueError(f"variant (out_name/index_path) requires exactly one method, got {methods}")
    result: dict[str, Any] = {}
    for m in methods:
        try:
            result[m] = eval_method(cfg, m, sets=sets, force=force, batch=batch, limit=limit, bench=bench,
                                    out_name=out_name, index_path=index_path, cfg_overrides=cfg_overrides)
        except ImportError as exc:
            log.error("%s: пропуск — отсутствуют зависимости (torch/faiss?): %s", m, exc)
            result[m] = {"error": f"ImportError: {exc}"}
        except FileNotFoundError as exc:
            log.error("%s: %s", m, exc)
            result[m] = {"error": str(exc)}
    return result


def read_scores(cfg: dict[str, Any], method: str, set_name: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Строки results/scores/<method>/<set>.jsonl (пустой список, если файла нет)."""
    p = scores_path(cfg, method, set_name, limit)
    return list(read_jsonl(p)) if p.exists() else []


def read_scores_meta(cfg: dict[str, Any], method: str, set_name: str, limit: int | None = None) -> dict[str, Any]:
    p = meta_path(cfg, method, set_name, limit)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
