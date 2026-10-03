"""Построение наборов запросов (DESIGN.md §5): к каждой отобранной функции применяется каждое
программное преобразование и partial для каждого L; результат — data/queries/{set}.jsonl.

Наборы: protected (label=1), hard_neg, public_calib, public_test (label=0) и наборы окон <split>_windows
(файловые окна из data/functions/<split>_windows.jsonl, cfg.windows.splits; protected_windows — label=1,
public_calib_windows / public_test_windows — негативы формы окна, label=0; преобразования identity + partial).
insert_deadcode в наборах оценки использует набор шаблонов cfg.transforms.deadcode_templates (по умолчанию eval —
не тот, что при обучении энкодера/комбинатора, evasion.TEMPLATE_SETS).
Файлы пишутся атомарно (tmp + os.replace); пустой файл считается отсутствующим.
"""

from __future__ import annotations

import json
import logging
import os
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
WINDOW_SUFFIX = "_windows"
ALL_SETS: tuple[str, ...] = QUERY_SETS + (WINDOW_SET,)
WINDOW_TRANSFORMS: tuple[str, ...] = ("identity",)
WINDOWS_FILE = "protected_windows.jsonl"  # = smcode.data.extract.WINDOWS_FILE (окна protected, kind == window)
DEFAULT_MAX_PER_SPLIT = 5000
DEFAULT_DEADCODE_TEMPLATES = "eval"
_DROP_PARAMS = ("mapping",)  # объёмные параметры, не нужные в файле запросов


def is_window_set(set_name: str) -> bool:
    """Набор окон (<split>_windows)."""
    return set_name.endswith(WINDOW_SUFFIX)


def window_sets(cfg: dict[str, Any]) -> list[str]:
    """Наборы окон по cfg.windows.splits: protected_windows (члены) и <split>_windows негативов (public_calib, public_test, ...)."""
    raw = (cfg.get("windows") or {}).get("splits") or ["protected"]
    return [f"{s}{WINDOW_SUFFIX}" for s in raw]


def all_sets(cfg: dict[str, Any]) -> list[str]:
    """Все наборы запросов конфига: QUERY_SETS + наборы окон (protected_windows всегда первый среди окон)."""
    wins = window_sets(cfg)
    if WINDOW_SET not in wins:
        wins = [WINDOW_SET, *wins]
    return [*QUERY_SETS, *wins]


def label_for_set(set_name: str) -> int:
    """label = 1 только для защищённого кода (protected, protected_windows)."""
    return 1 if set_name.startswith("protected") else 0


def source_split(set_name: str) -> str:
    """Сплит функций для набора запросов (окна <split>_windows — сплит <split>; см. functions_path)."""
    return set_name[: -len(WINDOW_SUFFIX)] if is_window_set(set_name) else set_name


def functions_path(cfg: dict[str, Any], set_name: str) -> Path:
    """Файл функций-источников: data/functions/<split>.jsonl; для наборов окон — <split>_windows.jsonl
    (как пишет модуль данных), с откатом на <split>.jsonl (строки kind == window), если его нет."""
    fdir = resolve_path(cfg, "functions")
    if is_window_set(set_name):
        win = fdir / f"{set_name}.jsonl"
        if _is_complete(win):
            return win
    return fdir / f"{source_split(set_name)}.jsonl"


def _is_complete(path: Path) -> bool:
    """Файл существует и непуст (прерванная запись оставляет пустой/отсутствующий файл)."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:  # pragma: no cover
        return False


def write_jsonl_atomic(path: str | Path, rows: Iterable[Any]) -> int:
    """JSONL через временный файл + os.replace: прерванный запуск не оставляет усечённого файла."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        n = write_jsonl(tmp, rows)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return n


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


def deadcode_template_set(cfg: dict[str, Any]) -> str:
    """Набор шаблонов insert_deadcode для наборов запросов (cfg.transforms.deadcode_templates: eval | train)."""
    return str((cfg.get("transforms", {}) or {}).get("deadcode_templates") or DEFAULT_DEADCODE_TEMPLATES)


def queries_for_record(rec: FunctionRecord, set_name: str, cfg: dict[str, Any],
                       transforms: Iterable[str] | None = None,
                       partial_lines: Iterable[int] | None = None,
                       stats: dict[str, Counter] | None = None,
                       template_set: str | None = None,
                       rename_styles: Iterable[str] | None = None) -> list[QueryRecord]:
    """Все запросы для одной функции: каждое программное преобразование + partial для каждого L.
    ГПСЧ детерминирован по (seed, set, id, transform). template_set — шаблоны insert_deadcode
    (по умолчанию cfg.transforms.deadcode_templates); rename_styles — ограничение стилей rename_ids (обучение)."""
    tcfg = cfg.get("transforms", {})
    names = list(tcfg.get("programmatic", ["identity"]) if transforms is None else transforms)
    Ls = list(tcfg.get("partial_lines", []) if partial_lines is None else partial_lines)
    every = int(tcfg.get("deadcode_every", 3))
    tset = template_set or deadcode_template_set(cfg)
    styles = [str(s) for s in rename_styles] if rename_styles else None
    out: list[QueryRecord] = []
    lang = canon_lang(rec.lang)
    for name in names:
        rng = get_rng(cfg, f"queries:{set_name}:{rec.id}:{name}")
        params: dict[str, Any] = {"every": every, "template_set": tset} if name == "insert_deadcode" else {}
        if name == "rename_ids" and styles:
            params["style"] = rng.choice(styles)
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
            if res.ok and res.params.get("parses") is False:
                stats[f"partial@{L}"]["unparsed"] += 1  # окно принято, но не разбирается (IDE-фрагмент)
        if res.ok:
            out.append(_make_query(rec, set_name, res, i))
    return out


def _load_records(cfg: dict[str, Any], set_name: str) -> list[FunctionRecord]:
    """Функции-источники набора (kind == window для наборов окон, иначе function), только языки из cfg."""
    path = functions_path(cfg, set_name)
    if not _is_complete(path):
        log.warning("functions file not found for set %s: %s", set_name, path)
        return []
    want_kind = "window" if is_window_set(set_name) else "function"
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
    if _is_complete(out_path) and not force:
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
    transforms = WINDOW_TRANSFORMS if is_window_set(set_name) else None
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

    n = write_jsonl_atomic(out_path, gen())
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
    """Строит наборы запросов для указанных наборов (по умолчанию все из all_sets(cfg): QUERY_SETS + окна).
    Возвращает {set: число строк}; −1 — набор уже существовал и был пропущен."""
    sets = list(splits) if splits else all_sets(cfg)
    result: dict[str, int] = {}
    for s in sets:
        result[s] = build_set(cfg, s, max_per_split=max_per_split, force=force)
    return result


def queries_path(cfg: dict[str, Any], set_name: str) -> Path:
    return resolve_path(cfg, "queries") / f"{set_name}.jsonl"
