"""Дедупликация корпуса функций (DESIGN.md §2.3).

1. Внутри каждого сплита схлопываются дубликаты по sha (sha1 нормализованного кода, DESIGN §2.4:
   abstract-токены ID/NUM/STR, т.е. тип-2 клоны); первая запись в порядке файла сохраняется.
   То же для файловых окон protected. В stats.json это поле removed_exact (ключ exact_dedup_key).
2. По protected (функции + окна) строится множество winnowing-отпечатков (k, w из cfg.dedup,
   без HMAC-ключа). Из hard_neg и public_* удаляется каждая функция, доля отпечатков которой,
   общих с protected, >= cfg.dedup.overlap_threshold (near_dup), а также функции с sha,
   совпадающим с protected, при n_tokens >= dedup.sha_protected_min_tokens (sha_protected; по умолчанию 0 —
   все тип-2 клоны, иначе короткие структурные клоны — геттеры, `return NUM;` — остаются негативами с меткой 0
   и считаются в kept_short_sha_protected). Окна негативов (<split>_windows.jsonl, cfg.windows.splits)
   фильтруются так же. Число удалённых логируется.
3. data/functions/stats.json — счётчики по сплитам (по языку, по бинам длины, удалённые),
   data/functions/dedup_removed.jsonl — список удалённых записей (id, причина, overlap).

Файлы data/functions/<split>.jsonl перезаписываются на месте (через временный файл); маркер
завершения этапа — stats.json; он считается действительным, только если новее всех входных
<split>.jsonl / protected_windows.jsonl (шаг 01 к тому же удаляет его при перезаписи сплита).
Публичные функции: run_dedup(cfg, ...), stats_fresh(out_dir), length_bin_label(...).
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm

from smcode.config import resolve_path
from smcode.data.extract import DEDUP_REMOVED_FILE, DEDUP_STATS_FILE, WINDOWS_FILE, windows_file
from smcode.fingerprint.winnowing import fingerprint_set
from smcode.normalize import abstract_tokens, tokenize
from smcode.types import SPLITS, read_jsonl

log = logging.getLogger(__name__)

STATS_FILE = DEDUP_STATS_FILE
REMOVED_FILE = DEDUP_REMOVED_FILE
NEGATIVE_SPLITS = ("hard_neg", "public_train", "public_calib", "public_test")
DEFAULT_LENGTH_BINS = [[0, 32], [32, 64], [64, 128], [128, 256], [256, 100000]]
CHUNK = 2000


# ----------------------------------------------------------------------------- utils


def length_bins(cfg: dict[str, Any]) -> list[list[int]]:
    return [list(map(int, b)) for b in (cfg.get("eval", {}).get("length_bins_tokens") or DEFAULT_LENGTH_BINS)]


def length_bin_label(n_tokens: int, bins: list[list[int]]) -> str:
    """Метка бина длины 'lo-hi' (lo <= n_tokens < hi; последний бин включает всё сверху)."""
    for lo, hi in bins:
        if lo <= n_tokens < hi:
            return f"{lo}-{hi}"
    lo, hi = bins[-1]
    return f"{lo}-{hi}"


def input_files(out_dir: Path) -> list[Path]:
    """Существующие входы дедупликации: <split>.jsonl и <split>_windows.jsonl (protected_windows.jsonl и окна негативов)."""
    out_dir = Path(out_dir)
    cands = [out_dir / f"{s}.jsonl" for s in SPLITS] + [out_dir / windows_file(s) for s in SPLITS]
    return [p for p in cands if p.exists()]


def stats_fresh(out_dir: Path) -> tuple[bool, str]:
    """(True, '') если stats.json существует и не старше всех входных файлов; иначе (False, причина)."""
    stats_path = Path(out_dir) / STATS_FILE
    if not stats_path.exists():
        return False, f"{STATS_FILE} отсутствует"
    smt = stats_path.stat().st_mtime_ns
    for p in input_files(out_dir):
        if p.stat().st_mtime_ns > smt:
            return False, f"{p.name} новее {STATS_FILE}"
    return True, ""


def _fps_task(args: tuple[str, str, int, int]) -> frozenset[int]:
    """Множество winnowing-отпечатков функции (abstract full-токены), пустое при ошибке."""
    code, lang, k, w = args
    try:
        toks = abstract_tokens(tokenize(code, lang), mode="full")
        return frozenset(fingerprint_set(toks, k, w, key=None))
    except Exception:  # noqa: BLE001
        return frozenset()


def _chunks(it: Iterable[Any], n: int) -> Iterator[list[Any]]:
    buf: list[Any] = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _map(pool: Pool | None, fn, items: list[Any]) -> list[Any]:
    if pool is None or len(items) < 64:
        return [fn(x) for x in items]
    return pool.map(fn, items, chunksize=max(1, len(items) // 32))


class _SplitStats:
    def __init__(self, bins: list[list[int]]):
        self.bins = bins
        self.n_raw = 0
        self.n_final = 0
        self.removed_exact = 0
        self.removed_near_dup = 0
        self.removed_sha_protected = 0
        self.kept_short_sha_protected = 0
        self.by_lang: Counter[str] = Counter()
        self.by_lang_raw: Counter[str] = Counter()
        self.by_length_bin: Counter[str] = Counter()
        self.by_kind: Counter[str] = Counter()
        self.repos: set[str] = set()
        self.removed_by_repo: Counter[str] = Counter()
        self.by_repo: Counter[str] = Counter()

    def see_raw(self, r: dict[str, Any]) -> None:
        self.n_raw += 1
        self.by_lang_raw[r["lang"]] += 1

    def keep(self, r: dict[str, Any]) -> None:
        self.n_final += 1
        self.by_lang[r["lang"]] += 1
        self.by_length_bin[length_bin_label(int(r["n_tokens"]), self.bins)] += 1
        self.by_kind[r.get("kind", "function")] += 1
        self.repos.add(r["repo"])
        self.by_repo[r["repo"]] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_raw": self.n_raw,
            "n_final": self.n_final,
            "removed_exact": self.removed_exact,
            "removed_near_dup": self.removed_near_dup,
            "removed_sha_protected": self.removed_sha_protected,
            "kept_short_sha_protected": self.kept_short_sha_protected,
            "n_repos": len(self.repos),
            "by_lang": dict(sorted(self.by_lang.items())),
            "by_lang_raw": dict(sorted(self.by_lang_raw.items())),
            "by_length_bin": {f"{lo}-{hi}": self.by_length_bin.get(f"{lo}-{hi}", 0) for lo, hi in self.bins},
            "by_kind": dict(sorted(self.by_kind.items())),
            "by_repo": dict(sorted(self.by_repo.items())),
            "removed_by_repo": dict(sorted(self.removed_by_repo.items())),
        }


# ----------------------------------------------------------------------------- core steps


def dedup_exact(records: Iterable[dict[str, Any]], seen: set[str] | None = None) -> Iterator[tuple[dict[str, Any], bool]]:
    """Генератор (запись, is_duplicate) — дубликаты по sha (нормализованный код) внутри потока."""
    seen = seen if seen is not None else set()
    for r in records:
        sha = r["sha"]
        dup = sha in seen
        seen.add(sha)
        yield r, dup


def _process_split(
    split: str,
    path: Path,
    k: int,
    w: int,
    threshold: float,
    protected_fps: set[int] | None,
    protected_shas: set[str] | None,
    pool: Pool | None,
    stats: _SplitStats,
    removed_out,
    collect_fps: set[int] | None = None,
    collect_shas: set[str] | None = None,
    sha_min_tokens: int = 0,
) -> None:
    """Проход по одному файлу: exact-dedup, (опц.) near-dup фильтр против protected, запись на место.
    collect_fps/collect_shas — для protected: накопление отпечатков и sha. sha_min_tokens — минимальная
    длина (в токенах) для удаления по совпадению sha с protected."""
    total = _count_lines(path)
    tmp = path.with_suffix(".jsonl.tmp")
    seen_sha: set[str] = set()
    filter_mode = protected_fps is not None
    with open(tmp, "w", encoding="utf-8") as f_out, tqdm(total=total, desc=f"dedup {split}", unit="rec", disable=total == 0) as bar:
        for chunk in _chunks(read_jsonl(path), CHUNK):
            kept: list[dict[str, Any]] = []
            for r, dup in dedup_exact(chunk, seen_sha):
                stats.see_raw(r)
                if dup:
                    stats.removed_exact += 1
                    stats.removed_by_repo[r["repo"]] += 1
                    removed_out.write(json.dumps({"id": r["id"], "split": split, "repo": r["repo"], "reason": "exact", "overlap": 1.0}) + "\n")
                    continue
                kept.append(r)
            need_fps = filter_mode or collect_fps is not None
            fps_list = _map(pool, _fps_task, [(r["code"], r["lang"], k, w) for r in kept]) if need_fps else [None] * len(kept)
            for r, fps in zip(kept, fps_list):
                if collect_fps is not None and fps:
                    collect_fps.update(fps)
                if collect_shas is not None:
                    collect_shas.add(r["sha"])
                if filter_mode:
                    assert protected_fps is not None and protected_shas is not None
                    sha_hit = r["sha"] in protected_shas
                    if sha_hit and int(r.get("n_tokens", 0)) >= sha_min_tokens:
                        ov, reason = 1.0, "sha_protected"
                        stats.removed_sha_protected += 1
                    else:
                        if sha_hit:
                            stats.kept_short_sha_protected += 1
                        ov = (len(fps & protected_fps) / len(fps)) if fps else 0.0
                        reason = "near_dup"
                    if ov >= threshold:
                        stats.removed_near_dup += 1
                        stats.removed_by_repo[r["repo"]] += 1
                        removed_out.write(json.dumps({"id": r["id"], "split": split, "repo": r["repo"], "reason": reason, "overlap": round(ov, 4)}) + "\n")
                        continue
                stats.keep(r)
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            bar.update(len(chunk))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- run


def run_dedup(cfg: dict[str, Any], workers: int = 4, force: bool = False) -> dict[str, Any]:
    """Дедупликация data/functions/*.jsonl на месте + stats.json. Идемпотентно: если stats.json
    существует, новее всех входных файлов и не force — возвращает его содержимое."""
    out_dir = resolve_path(cfg, "functions")
    stats_path = out_dir / STATS_FILE
    if not force:
        fresh, why = stats_fresh(out_dir)
        if fresh:
            log.info("stats.json найден (%s) и новее входных файлов — дедупликация уже выполнена, пропуск", stats_path)
            with open(stats_path, "r", encoding="utf-8") as f:
                return json.load(f)
        if stats_path.exists():
            log.warning("stats.json устарел (%s) — дедупликация выполняется заново", why)

    dcfg = cfg.get("dedup") or {}
    k = int(dcfg.get("k", 25))
    w = int(dcfg.get("w", 25))
    threshold = float(dcfg.get("overlap_threshold", 0.3))
    sha_min_tokens = int(dcfg.get("sha_protected_min_tokens", 0))
    bins = length_bins(cfg)
    pool = Pool(processes=workers) if workers > 1 else None
    split_stats: dict[str, _SplitStats] = {}
    protected_fps: set[int] = set()
    protected_shas: set[str] = set()
    removed_path = out_dir / REMOVED_FILE
    inputs = {p.name: {"size": p.stat().st_size, "lines": _count_lines(p)} for p in input_files(out_dir)}
    try:
        with open(removed_path, "w", encoding="utf-8") as removed_out:
            # --- protected: exact dedup + накопление отпечатков
            p_path = out_dir / "protected.jsonl"
            st = _SplitStats(bins)
            split_stats["protected"] = st
            if p_path.exists():
                _process_split("protected", p_path, k, w, threshold, None, None, pool, st, removed_out,
                               collect_fps=protected_fps, collect_shas=protected_shas)
                log.info("[protected] %d -> %d (дубликатов по sha %d); отпечатков: %d",
                         st.n_raw, st.n_final, st.removed_exact, len(protected_fps))
            else:
                log.warning("protected.jsonl не найден — near-dup фильтр будет пустым")
            # --- окна protected: exact dedup + отпечатки
            w_path = out_dir / WINDOWS_FILE
            wst = _SplitStats(bins)
            if w_path.exists():
                _process_split("protected_windows", w_path, k, w, threshold, None, None, pool, wst, removed_out,
                               collect_fps=protected_fps, collect_shas=None)
                log.info("[protected_windows] %d -> %d (дубликатов по sha %d); отпечатков всего: %d",
                         wst.n_raw, wst.n_final, wst.removed_exact, len(protected_fps))
            # --- негативы: exact + near-dup против protected (функции и, если есть, окна той же формы)
            neg_window_stats: dict[str, _SplitStats] = {}
            for split in NEGATIVE_SPLITS:
                path = out_dir / f"{split}.jsonl"
                st = _SplitStats(bins)
                split_stats[split] = st
                if not path.exists():
                    log.warning("[%s] файл %s не найден — пропуск", split, path)
                    continue
                _process_split(split, path, k, w, threshold, protected_fps, protected_shas, pool, st, removed_out,
                               sha_min_tokens=sha_min_tokens)
                log.info("[%s] %d -> %d: дубликатов по sha %d, near-dup с protected %d (из них по sha %d; "
                         "коротких sha-совпадений оставлено %d)",
                         split, st.n_raw, st.n_final, st.removed_exact, st.removed_near_dup, st.removed_sha_protected,
                         st.kept_short_sha_protected)
                for repo, n in sorted(st.removed_by_repo.items(), key=lambda x: -x[1])[:20]:
                    log.info("    удалено %5d  %s", n, repo)
                nw_path = out_dir / windows_file(split)
                if nw_path.exists():
                    nst = _SplitStats(bins)
                    neg_window_stats[f"{split}_windows"] = nst
                    _process_split(f"{split}_windows", nw_path, k, w, threshold, protected_fps, protected_shas, pool, nst, removed_out,
                                   sha_min_tokens=sha_min_tokens)
                    log.info("[%s_windows] %d -> %d: дубликатов по sha %d, near-dup с protected %d", split, nst.n_raw, nst.n_final,
                             nst.removed_exact, nst.removed_near_dup)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    stats: dict[str, Any] = {
        "config": {
            "k": k, "w": w, "overlap_threshold": threshold,
            "exact_dedup_key": "normalized_sha",  # sha1 abstract-токенов (тип-2 клоны), DESIGN §2.4
            "sha_protected_min_tokens": sha_min_tokens,
        },
        "inputs": inputs,
        "length_bins_tokens": bins,
        "protected_fingerprints": len(protected_fps),
        "splits": {s: split_stats[s].to_dict() for s in SPLITS if s in split_stats},
        "windows": wst.to_dict(),
        "windows_by_split": {"protected_windows": wst.to_dict(), **{k: v.to_dict() for k, v in neg_window_stats.items()}},
        "totals": {
            "n_raw": sum(s.n_raw for s in split_stats.values()),
            "n_final": sum(s.n_final for s in split_stats.values()),
            "removed_exact": sum(s.removed_exact for s in split_stats.values()),
            "removed_near_dup": sum(s.removed_near_dup for s in split_stats.values()),
        },
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    log.info("stats.json записан: %s", stats_path)
    return stats
