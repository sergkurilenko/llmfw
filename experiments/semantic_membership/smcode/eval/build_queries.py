"""Построение наборов запросов (DESIGN.md §5): к каждой отобранной функции применяется каждое
программное преобразование и partial для каждого L; результат — data/queries/{set}.jsonl.

Наборы: protected (label=1), hard_neg, public_calib, public_test (label=0) и protected_windows
(файловые окна protected, label=1, преобразования identity + partial).
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from smcode.config import get_rng, resolve_path
from smcode.normalize import canon_lang, count_tokens
from smcode.transforms.programmatic import count_lines
from smcode.transforms.registry import apply_transform
from smcode.types import FunctionRecord, QueryRecord, TransformResult, read_functions, write_jsonl

log = logging.getLogger(__name__)

QUERY_SETS: tuple[str, ...] = ("protected", "hard_neg", "public_calib", "public_test")
WINDOW_SET = "protected_windows"
ALL_SETS: tuple[str, ...] = QUERY_SETS + (WINDOW_SET,)
WINDOW_TRANSFORMS: tuple[str, ...] = ("identity",)
DEFAULT_MAX_PER_SPLIT = 5000
_DROP_PARAMS = ("mapping",)  # объёмные параметры, не нужные в файле запросов


def label_for_set(set_name: str) -> int:
    """label = 1 только для защищённого кода (protected, protected_windows)."""
    return 1 if set_name.startswith("protected") else 0


def source_split(set_name: str) -> str:
    """Файл функций для набора запросов (protected_windows берётся из protected.jsonl)."""
    return "protected" if set_name == WINDOW_SET else set_name


def length_bin(n_tokens: int, bins: Sequence[Sequence[int]]) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= n_tokens < hi:
            return i
    return len(bins) - 1


def sample_records(records: Sequence[FunctionRecord], max_n: int, rng: random.Random,
                   bins: Sequence[Sequence[int]]) -> list[FunctionRecord]:
    """Стратифицированная выборка по (язык, бин длины): пропорциональные квоты (метод наибольших остатков)."""
    if max_n <= 0 or len(records) <= max_n:
        return sorted(records, key=lambda r: r.id)
    groups: dict[tuple[str, int], list[FunctionRecord]] = defaultdict(list)
    for r in records:
        groups[(r.lang, length_bin(r.n_tokens, bins))].append(r)
    keys = sorted(groups)
    total = len(records)
    raw = {k: max_n * len(groups[k]) / total for k in keys}
    quota = {k: int(raw[k]) for k in keys}
    rest = max_n - sum(quota.values())
    for k in sorted(keys, key=lambda k: (-(raw[k] - quota[k]), k))[:rest]:
        quota[k] += 1
    out: list[FunctionRecord] = []
    for k in keys:
        grp = sorted(groups[k], key=lambda r: r.id)
        rng.shuffle(grp)
        out.extend(grp[: quota[k]])
    return sorted(out, key=lambda r: r.id)


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k not in _DROP_PARAMS}


def _make_query(rec: FunctionRecord, set_name: str, res: TransformResult, idx: int) -> QueryRecord:
    code = res.code or ""
    lang = canon_lang(rec.lang)
    return QueryRecord(
        qid=f"{rec.id}#{res.name}#{idx}", source_id=rec.id, label=label_for_set(set_name), set=set_name,
        transform=res.name, params=_clean_params(res.params), lang=lang, repo=rec.repo, code=code,
        n_lines=count_lines(code), n_tokens=count_tokens(code, lang),
    )


def queries_for_record(rec: FunctionRecord, set_name: str, cfg: dict[str, Any],
                       transforms: Iterable[str] | None = None,
                       partial_lines: Iterable[int] | None = None,
                       stats: dict[str, Counter] | None = None) -> list[QueryRecord]:
    """Все запросы для одной функции: каждое программное преобразование + partial для каждого L.
    ГПСЧ детерминирован по (seed, set, id, transform)."""
    tcfg = cfg.get("transforms", {})
    names = list(tcfg.get("programmatic", ["identity"]) if transforms is None else transforms)
    Ls = list(tcfg.get("partial_lines", []) if partial_lines is None else partial_lines)
    every = int(tcfg.get("deadcode_every", 3))
    out: list[QueryRecord] = []
    lang = canon_lang(rec.lang)
    for name in names:
        rng = get_rng(cfg, f"queries:{set_name}:{rec.id}:{name}")
        params = {"every": every} if name == "insert_deadcode" else {}
        res = apply_transform(name, rec.code, lang, rng, **params)
        if stats is not None:
            stats[name]["ok" if res.ok else "failed"] += 1
        if res.ok:
            out.append(_make_query(rec, set_name, res, 0))
    for i, L in enumerate(Ls):
        rng = get_rng(cfg, f"queries:{set_name}:{rec.id}:partial:{L}")
        res = apply_transform("partial", rec.code, lang, rng, L=int(L))
        if stats is not None:
            stats[f"partial@{L}"]["ok" if res.ok else "failed"] += 1
        if res.ok:
            out.append(_make_query(rec, set_name, res, i))
    return out


def _load_records(cfg: dict[str, Any], set_name: str) -> list[FunctionRecord]:
    path = resolve_path(cfg, "functions") / f"{source_split(set_name)}.jsonl"
    if not path.exists():
        log.warning("functions file not found for set %s: %s", set_name, path)
        return []
    want_kind = "window" if set_name == WINDOW_SET else "function"
    recs = [r for r in read_functions(path) if (r.kind or "function") == want_kind]
    langs = set(cfg.get("languages", []))
    if langs:
        recs = [r for r in recs if canon_lang(r.lang) in langs]
    return recs


def build_set(cfg: dict[str, Any], set_name: str, max_per_split: int | None = None,
              force: bool = False) -> int:
    """Строит один набор запросов; возвращает число записанных строк (−1, если пропущен как готовый)."""
    out_dir = resolve_path(cfg, "queries")
    out_path = out_dir / f"{set_name}.jsonl"
    if out_path.exists() and not force:
        log.info("skip %s: exists (%s)", set_name, out_path)
        return -1
    recs = _load_records(cfg, set_name)
    if not recs:
        log.warning("no source records for set %s; nothing written", set_name)
        return 0
    tcfg = cfg.get("transforms", {})
    max_n = int(max_per_split if max_per_split is not None else tcfg.get("max_per_split", DEFAULT_MAX_PER_SPLIT))
    bins = cfg.get("eval", {}).get("length_bins_tokens", [[0, 10 ** 9]])
    sampled = sample_records(recs, max_n, get_rng(cfg, f"sample:{set_name}"), bins)
    log.info("set %s: %d records, sampled %d", set_name, len(recs), len(sampled))
    transforms = WINDOW_TRANSFORMS if set_name == WINDOW_SET else None
    stats: dict[str, Counter] = defaultdict(Counter)
    per_lang: Counter = Counter()

    def gen():
        try:
            from tqdm import tqdm  # noqa: WPS433
            it = tqdm(sampled, desc=f"queries:{set_name}", unit="fn", disable=len(sampled) < 50)
        except Exception:  # pragma: no cover
            it = sampled
        for rec in it:
            for q in queries_for_record(rec, set_name, cfg, transforms=transforms, stats=stats):
                per_lang[q.lang] += 1
                yield q

    n = write_jsonl(out_path, gen())
    summary = {
        "set": set_name, "n_source": len(recs), "n_sampled": len(sampled), "n_queries": n,
        "label": label_for_set(set_name), "per_transform": {k: dict(v) for k, v in sorted(stats.items())},
        "per_lang": dict(per_lang),
    }
    with open(out_dir / f"{set_name}.stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("set %s: wrote %d queries → %s", set_name, n, out_path)
    return n


def build_queries(cfg: dict[str, Any], splits: Iterable[str] | None = None, force: bool = False,
                  max_per_split: int | None = None) -> dict[str, int]:
    """Строит наборы запросов для указанных наборов (по умолчанию все из ALL_SETS).
    Возвращает {set: число строк}; −1 — набор уже существовал и был пропущен."""
    sets = list(splits) if splits else list(ALL_SETS)
    result: dict[str, int] = {}
    for s in sets:
        result[s] = build_set(cfg, s, max_per_split=max_per_split, force=force)
    return result


def queries_path(cfg: dict[str, Any], set_name: str) -> Path:
    return resolve_path(cfg, "queries") / f"{set_name}.jsonl"
