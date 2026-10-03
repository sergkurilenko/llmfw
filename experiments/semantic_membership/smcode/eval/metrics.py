"""Метрики оценки (DESIGN.md §10) и схема results/summary.json.

Скоры (results/scores/<method>/<set>.jsonl) соединяются с запросами (data/queries/<set>.jsonl) по qid;
по каждому методу считаются пороги, TPR/FPR с интервалами, AUROC, латентность и память. Правило решения:
score > τ (см. smcode.calibration). Бутстрэп TPR/AUROC кластеризован по source_id (исходной функции);
FPR — интервалы Клоппера–Пирсона.

# Схема results/summary.json

```
{
  "generated_at": "YYYY-MM-DDTHH:MM:SS",
  "config_path": str, "seed": int,
  "alphas": [0.01, 0.001],                      # уровни α из cfg.calibration.alphas
  "alpha_keys": ["0.01", "0.001"],             # ключи словарей ниже (str(alpha))
  "bootstrap": int,                             # число повторов бутстрэпа
  "length_bins": [[lo, hi], ...],               # бины длины (токены запроса), ключ "lo-hi"
  "decision_rule": "score > tau",
  "domain_calibration": {"fraction": f, "calib_repos": [...], "test_repos": [...]},   # общее для всех методов
  "data_stats": {                               # data/functions/stats.json + статистика запросов
      "functions": {... содержимое stats.json ...} | null,
      "queries": {set: {"n": int, "label": 0|1, "by_transform": {t: n}, "by_lang": {lang: n},
                        "stats": {... <set>.stats.json ...}, "llm_stats": {... <set>.llm_stats.json ...}}}
  },
  "methods": {
    "<method>": {                               # имя = каталог results/scores/<method>
      "n_records": int|null, "memory_bytes": int|null, "bytes_per_record": float|null,
      "build_seconds": float|null, "disk_bytes": int|null,
      "n_scores": {set: int},                   # число соединённых (qid найден в запросах) скоров
      "thresholds": {alpha_key: {"conformal": τ|null, "oracle": τ|null, "domain": τ|null,
                                 "n_calib": int, "n_calib_domain": int, "min_n_calib": int}},
      "tpr": {alpha_key: {kind: GROUPS}},       # kind ∈ {conformal, oracle, domain}; GROUPS =
            #   {"overall": RATE, "by_transform": {t: RATE}, "by_lang": {lang: RATE},
            #    "by_length_bin": {"lo-hi": RATE}, "by_partial_L": {"L": RATE}, "by_translate_dst": {lang: RATE}}
            #   RATE = {"value": float|null, "ci": [lo, hi]|null, "n": int, "k": int}  (ДИ — бутстрэп по кластерам)
      "tpr_windows": {alpha_key: GROUPS}|null,  # набор protected_windows при конформном пороге
      "fpr": {alpha_key: {kind: {neg_set: FPR}}},   # neg_set ∈ {public_calib, public_test, hard_neg,
            #   hard_neg_test (только domain)}; FPR = RATE (ДИ Клоппера–Пирсона) +
            #   {"by_lang": {lang: RATE}, "by_transform": {t: RATE}, "max_subgroup": {"group": str, **RATE}}
      "confusion": {alpha_key: {kind: {set: {"tp","fn"} | {"fp","tn"}}}},   # полная матрица ошибок
      "auroc": {neg_set: {"value": float|null, "ci": [lo, hi]|null, "n_pos": int, "n_neg": int}},
      "roc": {neg_set: {"fpr": [...], "tpr": [...]}},                      # прореженные кривые для F3
      "latency_ms": {"p50","p95","mean","n", "mode": str, "by_set": {set: {"p50","p95","n"}},
                     "benchmark": {...<set>.meta.json["latency_benchmark"]...}|null},
      "sets": {set: {"n": int, "n_queries": int, "latency_mode": str}}
    }
  }
}
```
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from smcode.calibration import (
    INF,
    conformal_threshold,
    decide,
    domain_calibration,
    min_calibration_size,
    rate_with_ci,
    split_repos,
    threshold_at_fpr,
)
from smcode.config import get_rng, resolve_path
from smcode.eval.build_queries import QUERY_SETS, WINDOW_SET
from smcode.eval.run_eval import read_scores, read_scores_meta, scores_dir
from smcode.fingerprint.index import INDEX_STATS_FILE, REGISTRY, index_dir
from smcode.types import read_jsonl

log = logging.getLogger(__name__)

SUMMARY_FILE = "summary.json"
SCHEMA_FILE = "summary_schema.md"
NEG_SETS: tuple[str, ...] = ("public_calib", "public_test", "hard_neg")
KINDS: tuple[str, ...] = ("conformal", "oracle", "domain")
MIN_SUBGROUP = 50          # минимальный размер подгруппы для "max_subgroup"
ROC_POINTS = 300
DEFAULT_DOMAIN_FRACTION = 0.5


def alpha_key(alpha: float) -> str:
    return f"{alpha:g}"


def length_bin_label(n_tokens: int, bins: Sequence[Sequence[int]]) -> str:
    """Метка бина 'lo-hi' (lo ≤ n < hi; последний бин включает всё сверху) — как в data/functions/stats.json."""
    for lo, hi in bins:
        if lo <= n_tokens < hi:
            return f"{lo}-{hi}"
    lo, hi = bins[-1]
    return f"{lo}-{hi}"


# ----------------------------------------------------------------------------- данные


@dataclass
class ScoredSet:
    """Скоры одного метода на одном наборе запросов, соединённые с метаданными запросов."""

    name: str
    label: int
    qid: np.ndarray
    score: np.ndarray
    latency: np.ndarray
    transform: np.ndarray
    lang: np.ndarray
    n_tokens: np.ndarray
    L: np.ndarray
    repo: np.ndarray
    source_id: np.ndarray
    dst_lang: np.ndarray
    best_id: np.ndarray
    n_queries: int = 0
    latency_mode: str = ""

    @property
    def n(self) -> int:
        return int(self.score.size)

    def subset(self, mask: np.ndarray) -> "ScoredSet":
        return ScoredSet(self.name, self.label, self.qid[mask], self.score[mask], self.latency[mask], self.transform[mask],
                         self.lang[mask], self.n_tokens[mask], self.L[mask], self.repo[mask], self.source_id[mask],
                         self.dst_lang[mask], self.best_id[mask], int(mask.sum()), self.latency_mode)


def load_query_meta(path: str | Path) -> dict[str, dict[str, Any]]:
    """qid → метаданные запроса (без кода): label, set, transform, L, dst_lang, lang, n_tokens, repo, source_id."""
    out: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(path):
        p = r.get("params") or {}
        out[r["qid"]] = {
            "label": int(r.get("label", 0)), "set": r.get("set"), "transform": r.get("transform", "identity"),
            "L": int(p.get("L", -1) or -1), "dst_lang": p.get("dst_lang") or "",
            "lang": r.get("lang", ""), "n_tokens": int(r.get("n_tokens", 0) or 0), "repo": r.get("repo", ""),
            "source_id": r.get("source_id") or r["qid"],
        }
    return out


def join_scores(rows: Iterable[dict[str, Any]], qmeta: dict[str, dict[str, Any]], set_name: str,
                latency_mode: str = "") -> ScoredSet:
    """Соединяет строки скоров с метаданными запросов; строки с неизвестным qid или NaN-скором отбрасываются."""
    cols: dict[str, list[Any]] = {k: [] for k in ("qid", "score", "latency", "transform", "lang", "n_tokens", "L",
                                                  "repo", "source_id", "dst_lang", "best_id")}
    n_unknown = n_nan = 0
    label = 1 if set_name.startswith("protected") else 0
    for r in rows:
        m = qmeta.get(r["qid"])
        if m is None:
            n_unknown += 1
            continue
        s = r.get("score")
        if s is None or (isinstance(s, float) and np.isnan(s)):
            n_nan += 1
            continue
        label = m["label"]
        cols["qid"].append(r["qid"]); cols["score"].append(float(s))
        lat = r.get("latency_ms")
        cols["latency"].append(np.nan if lat is None else float(lat))
        cols["transform"].append(m["transform"]); cols["lang"].append(m["lang"]); cols["n_tokens"].append(m["n_tokens"])
        cols["L"].append(m["L"]); cols["repo"].append(m["repo"]); cols["source_id"].append(m["source_id"])
        cols["dst_lang"].append(m["dst_lang"]); cols["best_id"].append(r.get("best_id") or "")
    if n_unknown or n_nan:
        log.warning("%s: отброшено строк — неизвестный qid: %d, NaN-скор: %d", set_name, n_unknown, n_nan)
    n = len(cols["qid"])
    if n < len(qmeta):
        log.warning("%s: скоры есть для %d из %d запросов", set_name, n, len(qmeta))
    obj = lambda k: np.asarray(cols[k], dtype=object)  # noqa: E731
    return ScoredSet(
        name=set_name, label=label, qid=obj("qid"), score=np.asarray(cols["score"], dtype=np.float64),
        latency=np.asarray(cols["latency"], dtype=np.float64), transform=obj("transform"), lang=obj("lang"),
        n_tokens=np.asarray(cols["n_tokens"], dtype=np.int64), L=np.asarray(cols["L"], dtype=np.int64),
        repo=obj("repo"), source_id=obj("source_id"), dst_lang=obj("dst_lang"), best_id=obj("best_id"),
        n_queries=len(qmeta), latency_mode=latency_mode,
    )


def load_scored_set(cfg: dict[str, Any], method: str, set_name: str, qmeta: dict[str, dict[str, Any]]) -> ScoredSet | None:
    rows = read_scores(cfg, method, set_name)
    if not rows:
        return None
    meta = read_scores_meta(cfg, method, set_name)
    return join_scores(rows, qmeta, set_name, latency_mode=str(meta.get("latency_mode", "")))


def available_methods(cfg: dict[str, Any]) -> list[str]:
    """Каталоги results/scores/<method>, где есть хотя бы один файл скоров; известные методы — первыми."""
    base = scores_dir(cfg)
    found = [d.name for d in base.iterdir() if d.is_dir() and any(d.glob("*.jsonl"))]
    known = [m for m in REGISTRY if m in found]
    return known + sorted(m for m in found if m not in REGISTRY)


# ----------------------------------------------------------------------------- бутстрэп


class ClusterBootstrap:
    """Бутстрэп с ресемплированием кластеров (source_id): одна матрица счётчиков (B × C) на набор,
    общая для всех подгрупп и метрик → согласованные ДИ."""

    def __init__(self, clusters: np.ndarray, n_boot: int, rng: np.random.Generator) -> None:
        self.uniq, self.cidx = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
        self.C = int(self.uniq.size)
        self.B = int(n_boot)
        if self.B > 0 and self.C > 0:
            self.counts = rng.multinomial(self.C, np.full(self.C, 1.0 / self.C), size=self.B).astype(np.float64)
        else:
            self.counts = None

    def rate_ci(self, hits: np.ndarray, mask: np.ndarray | None = None, conf: float = 0.95) -> list[float] | None:
        """Персентильный ДИ доли hits (bool) в подмножестве mask."""
        if self.counts is None:
            return None
        m = np.ones(hits.shape, dtype=bool) if mask is None else mask
        if not m.any():
            return None
        h = np.bincount(self.cidx[m], weights=hits[m].astype(np.float64), minlength=self.C)
        t = np.bincount(self.cidx[m], minlength=self.C).astype(np.float64)
        num, den = self.counts @ h, self.counts @ t
        ok = den > 0
        if not ok.any():
            return None
        vals = num[ok] / den[ok]
        q = (1.0 - conf) / 2.0
        return [float(np.percentile(vals, 100 * q)), float(np.percentile(vals, 100 * (1 - q)))]

    def weights(self, b: int) -> np.ndarray:
        """Веса запросов в b-м повторе (число вхождений их кластера)."""
        assert self.counts is not None
        return self.counts[b][self.cidx]


class AurocEstimator:
    """AUROC = P(s_pos > s_neg) + ½·P(s_pos = s_neg) с весами наблюдений (для кластерного бутстрэпа).
    Предрасчёт сортировки и групп равных скоров делает один повтор O(N)."""

    def __init__(self, y: np.ndarray, s: np.ndarray) -> None:
        self.order = np.argsort(s, kind="stable")
        s_sorted = s[self.order]
        self.y_sorted = np.asarray(y, dtype=bool)[self.order]
        self.starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]]) if s.size else np.zeros(0, np.int64)

    def auc(self, w: np.ndarray | None = None) -> float:
        if self.y_sorted.size == 0 or self.starts.size == 0:
            return float("nan")
        ws = np.ones(self.y_sorted.size) if w is None else np.asarray(w, dtype=np.float64)[self.order]
        pos_w, neg_w = ws * self.y_sorted, ws * ~self.y_sorted
        P, N = pos_w.sum(), neg_w.sum()
        if P <= 0 or N <= 0:
            return float("nan")
        pg, ng = np.add.reduceat(pos_w, self.starts), np.add.reduceat(neg_w, self.starts)
        cum_before = np.cumsum(ng) - ng
        return float((pg * (cum_before + 0.5 * ng)).sum() / (P * N))


def auroc_with_ci(pos: ScoredSet, neg: ScoredSet, n_boot: int, rng: np.random.Generator,
                  conf: float = 0.95) -> dict[str, Any]:
    """AUROC «члены vs негативы» и персентильный ДИ кластерного бутстрэпа (кластеры — source_id обоих наборов)."""
    y = np.concatenate([np.ones(pos.n, bool), np.zeros(neg.n, bool)])
    s = np.concatenate([pos.score, neg.score])
    est = AurocEstimator(y, s)
    out: dict[str, Any] = {"value": None, "ci": None, "n_pos": pos.n, "n_neg": neg.n}
    if pos.n == 0 or neg.n == 0:
        return out
    out["value"] = est.auc()
    boot = ClusterBootstrap(np.concatenate([pos.source_id, neg.source_id]), n_boot, rng)
    if boot.counts is not None:
        vals = np.array([est.auc(boot.weights(b)) for b in range(boot.B)])
        vals = vals[~np.isnan(vals)]
        if vals.size:
            q = (1.0 - conf) / 2.0
            out["ci"] = [float(np.percentile(vals, 100 * q)), float(np.percentile(vals, 100 * (1 - q)))]
    return out


def roc_points(pos: ScoredSet, neg: ScoredSet, n_points: int = ROC_POINTS) -> dict[str, list[float]]:
    """Прореженная ROC-кривая (решение score > τ): точки на лог-сетке по FPR + равномерная сетка."""
    from sklearn.metrics import roc_curve

    if pos.n == 0 or neg.n == 0:
        return {"fpr": [], "tpr": []}
    y = np.concatenate([np.ones(pos.n, int), np.zeros(neg.n, int)])
    s = np.concatenate([pos.score, neg.score])
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=True)
    if fpr.size <= n_points:
        return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    grid = np.concatenate([np.logspace(-6, 0, n_points * 2 // 3), np.linspace(0, 1, n_points // 3)])
    idx = np.unique(np.concatenate([[0, fpr.size - 1], np.clip(np.searchsorted(fpr, grid), 0, fpr.size - 1)]))
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}


# ----------------------------------------------------------------------------- группы


def _rate_block(hits: np.ndarray, mask: np.ndarray, boot: ClusterBootstrap | None) -> dict[str, Any]:
    n = int(mask.sum())
    k = int(hits[mask].sum()) if n else 0
    return {"value": (k / n) if n else None, "ci": boot.rate_ci(hits, mask) if (boot and n) else None, "n": n, "k": k}


def tpr_groups(ss: ScoredSet, tau: float, boot: ClusterBootstrap | None, bins: Sequence[Sequence[int]]) -> dict[str, Any]:
    """TPR overall / по преобразованию / языку / бину длины / L (partial) / целевому языку (translate)."""
    hits = decide(ss.score, tau)
    all_mask = np.ones(ss.n, dtype=bool)
    out: dict[str, Any] = {"overall": _rate_block(hits, all_mask, boot)}
    out["by_transform"] = {str(t): _rate_block(hits, ss.transform == t, boot) for t in sorted(set(ss.transform.tolist()))}
    out["by_lang"] = {str(l): _rate_block(hits, ss.lang == l, boot) for l in sorted(set(ss.lang.tolist()))}
    labels = np.asarray([length_bin_label(int(n), bins) for n in ss.n_tokens], dtype=object)
    out["by_length_bin"] = {f"{lo}-{hi}": _rate_block(hits, labels == f"{lo}-{hi}", boot) for lo, hi in bins}
    part = ss.transform == "partial"
    out["by_partial_L"] = {str(int(L)): _rate_block(hits, part & (ss.L == L), boot) for L in sorted(set(ss.L[part].tolist()))}
    tr = ss.transform == "translate"
    out["by_translate_dst"] = {str(d): _rate_block(hits, tr & (ss.dst_lang == d), boot)
                               for d in sorted(set(ss.dst_lang[tr].tolist())) if d}
    return out


def fpr_block(ss: ScoredSet, tau: float, conf: float = 0.95) -> dict[str, Any]:
    """FPR с ДИ Клоппера–Пирсона: overall, по языку, по преобразованию, худшая подгруппа (n ≥ MIN_SUBGROUP)."""
    out = rate_with_ci(ss.score, tau, conf)
    out["by_lang"] = {str(l): rate_with_ci(ss.score[ss.lang == l], tau, conf) for l in sorted(set(ss.lang.tolist()))}
    out["by_transform"] = {str(t): rate_with_ci(ss.score[ss.transform == t], tau, conf) for t in sorted(set(ss.transform.tolist()))}
    cands = [("lang:" + k, v) for k, v in out["by_lang"].items()] + [("transform:" + k, v) for k, v in out["by_transform"].items()]
    big = [c for c in cands if c[1]["n"] >= MIN_SUBGROUP] or cands
    if big:
        g, v = max(big, key=lambda c: (c[1]["value"] or 0.0, c[1]["n"]))
        out["max_subgroup"] = {"group": g, **v}
    else:
        out["max_subgroup"] = None
    return out


def _confusion(ss: ScoredSet, tau: float) -> dict[str, int]:
    k = int(decide(ss.score, tau).sum())
    return {"tp": k, "fn": ss.n - k} if ss.label == 1 else {"fp": k, "tn": ss.n - k}


def _percentiles(a: np.ndarray) -> dict[str, Any]:
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"p50": None, "p95": None, "mean": None, "n": 0}
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean()), "n": int(a.size)}


# ----------------------------------------------------------------------------- метод


def _index_info(cfg: dict[str, Any], method: str, sets: dict[str, ScoredSet]) -> dict[str, Any]:
    """n_records / memory_bytes / build_seconds / disk_bytes из meta скоров, results/index_stats.json, meta.json индекса."""
    info: dict[str, Any] = {"n_records": None, "memory_bytes": None, "build_seconds": None, "disk_bytes": None}
    for s in sets:
        m = read_scores_meta(cfg, method, s)
        if m.get("memory_bytes") is not None:
            info["memory_bytes"], info["n_records"] = m.get("memory_bytes"), m.get("n_records")
            break
    stats_path = resolve_path(cfg, "results") / INDEX_STATS_FILE
    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                st = json.load(f).get(method) or {}
            for k in ("n_records", "memory_bytes", "build_seconds", "disk_bytes"):
                if info.get(k) is None and st.get(k) is not None:
                    info[k] = st[k]
        except json.JSONDecodeError:
            log.warning("corrupt %s", stats_path)
    meta_path = index_dir(cfg, method) / "meta.json" if method in REGISTRY else None
    if meta_path and meta_path.exists() and (info["n_records"] is None or info["memory_bytes"] is None):
        with open(meta_path, "r", encoding="utf-8") as f:
            im = json.load(f)
        info["n_records"] = info["n_records"] if info["n_records"] is not None else im.get("n_records")
        info["memory_bytes"] = info["memory_bytes"] if info["memory_bytes"] is not None else im.get("memory_bytes")
    n, mem = info.get("n_records"), info.get("memory_bytes")
    info["bytes_per_record"] = (float(mem) / n) if (n and mem is not None) else None
    return info


def method_metrics(cfg: dict[str, Any], method: str, sets: dict[str, ScoredSet], alphas: Sequence[float],
                   n_boot: int, domain_split: tuple[list[str], list[str]] | None = None) -> dict[str, Any]:
    """Все метрики одного метода по загруженным наборам (ключи — имена наборов)."""
    bins = [list(map(int, b)) for b in cfg.get("eval", {}).get("length_bins_tokens", [[0, 10 ** 9]])]
    conf = 0.95
    seed = get_rng(cfg, "eval:bootstrap").getrandbits(32)
    prot, calib, test, hard, win = (sets.get(k) for k in ("protected", "public_calib", "public_test", "hard_neg", WINDOW_SET))
    out: dict[str, Any] = {**_index_info(cfg, method, sets), "n_scores": {k: v.n for k, v in sets.items()},
                           "sets": {k: {"n": v.n, "n_queries": v.n_queries, "latency_mode": v.latency_mode} for k, v in sets.items()}}
    boot_prot = ClusterBootstrap(prot.source_id, n_boot, np.random.default_rng(seed)) if prot is not None and prot.n else None
    boot_win = ClusterBootstrap(win.source_id, n_boot, np.random.default_rng(seed + 1)) if win is not None and win.n else None

    out["thresholds"], out["tpr"], out["fpr"], out["confusion"], out["tpr_windows"] = {}, {}, {}, {}, ({} if win is not None else None)
    for a in alphas:
        ak = alpha_key(a)
        taus: dict[str, float | None] = {"conformal": None, "oracle": None, "domain": None}
        n_calib = calib.n if calib is not None else 0
        n_dom = 0
        hard_test_mask: np.ndarray | None = None
        if calib is not None and calib.n:
            taus["conformal"] = conformal_threshold(calib.score, a)
        if test is not None and test.n:
            taus["oracle"] = threshold_at_fpr(test.score, a)
        if calib is not None and hard is not None and hard.n and domain_split is not None:
            dc = domain_calibration(calib.score, hard.score, hard.repo.tolist(), a, calib_repos=domain_split[0])
            taus["domain"], n_dom, hard_test_mask = dc.tau, dc.n_calib, dc.test_mask
        out["thresholds"][ak] = {**{k: (None if v is None else (v if np.isfinite(v) else "inf")) for k, v in taus.items()},
                                 "n_calib": n_calib, "n_calib_domain": n_dom, "min_n_calib": min_calibration_size(a)}
        if taus["conformal"] is not None and n_calib < min_calibration_size(a):
            log.warning("%s α=%s: калибровочных негативов %d < %d → порог +inf", method, ak, n_calib, min_calibration_size(a))
        out["tpr"][ak], out["fpr"][ak], out["confusion"][ak] = {}, {}, {}
        for kind in KINDS:
            tau = taus[kind]
            if tau is None:
                continue
            if prot is not None and prot.n:
                out["tpr"][ak][kind] = tpr_groups(prot, tau, boot_prot, bins)
            negs: dict[str, ScoredSet] = {}
            if kind == "domain":
                if test is not None:
                    negs["public_test"] = test
                if hard is not None and hard_test_mask is not None:
                    negs["hard_neg_test"] = hard.subset(hard_test_mask)
            else:
                negs = {k: v for k, v in ((n, sets.get(n)) for n in NEG_SETS) if v is not None}
            out["fpr"][ak][kind] = {n: fpr_block(ss, tau, conf) for n, ss in negs.items() if ss.n}
            conf_m: dict[str, Any] = {}
            if prot is not None and prot.n:
                conf_m["protected"] = _confusion(prot, tau)
            if win is not None and win.n:
                conf_m[WINDOW_SET] = _confusion(win, tau)
            conf_m.update({n: _confusion(ss, tau) for n, ss in negs.items() if ss.n})
            out["confusion"][ak][kind] = conf_m
        if win is not None and win.n and taus["conformal"] is not None:
            g = tpr_groups(win, taus["conformal"], boot_win, bins)
            out["tpr_windows"][ak] = {k: g[k] for k in ("overall", "by_transform", "by_partial_L", "by_length_bin")}

    out["auroc"], out["roc"] = {}, {}
    if prot is not None and prot.n:
        for n, neg in (("public_test", test), ("hard_neg", hard)):
            if neg is not None and neg.n:
                out["auroc"][n] = auroc_with_ci(prot, neg, n_boot, np.random.default_rng(seed + 7))
                out["roc"][n] = roc_points(prot, neg)

    lat_all = np.concatenate([v.latency for v in sets.values()]) if sets else np.zeros(0)
    modes = sorted({v.latency_mode for v in sets.values() if v.latency_mode})
    bench = None
    for s in sets:
        b = read_scores_meta(cfg, method, s).get("latency_benchmark")
        if b:
            bench = {"set": s, **b}
            break
    out["latency_ms"] = {**_percentiles(lat_all), "mode": ",".join(modes), "by_set": {k: _percentiles(v.latency) for k, v in sets.items()},
                         "benchmark": bench}
    return out


# ----------------------------------------------------------------------------- данные/статистика


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        log.warning("corrupt json: %s", path)
        return None


def data_stats(cfg: dict[str, Any], qmeta: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """data/functions/stats.json + статистика наборов запросов (+ *.stats.json, *.llm_stats.json)."""
    qdir = resolve_path(cfg, "queries")
    queries: dict[str, Any] = {}
    for s, meta in qmeta.items():
        by_t: dict[str, int] = {}
        by_l: dict[str, int] = {}
        label = 0
        for m in meta.values():
            by_t[m["transform"]] = by_t.get(m["transform"], 0) + 1
            by_l[m["lang"]] = by_l.get(m["lang"], 0) + 1
            label = m["label"]
        queries[s] = {"n": len(meta), "label": label, "by_transform": dict(sorted(by_t.items())), "by_lang": dict(sorted(by_l.items())),
                      "stats": _read_json(qdir / f"{s}.stats.json"), "llm_stats": _read_json(qdir / f"{s}.llm_stats.json")}
    return {"functions": _read_json(resolve_path(cfg, "functions") / "stats.json"), "queries": queries}


# ----------------------------------------------------------------------------- точка входа


def compute_summary(cfg: dict[str, Any], methods: Iterable[str] | None = None, bootstrap: int | None = None,
                    sets: Iterable[str] | None = None) -> dict[str, Any]:
    """Считает метрики всех методов (по умолчанию — все каталоги results/scores/*) и возвращает summary (см. схему)."""
    t0 = time.perf_counter()
    methods = list(methods) if methods else available_methods(cfg)
    alphas = [float(a) for a in cfg.get("calibration", {}).get("alphas", [0.01])]
    n_boot = int(cfg.get("eval", {}).get("bootstrap", 1000) if bootstrap is None else bootstrap)
    set_names = list(sets) if sets else [*QUERY_SETS, WINDOW_SET]
    qdir = resolve_path(cfg, "queries")
    qmeta: dict[str, dict[str, dict[str, Any]]] = {}
    for s in set_names:
        p = qdir / f"{s}.jsonl"
        if p.exists():
            qmeta[s] = load_query_meta(p)
            log.info("queries %s: %d", s, len(qmeta[s]))
        else:
            log.warning("queries file missing: %s", p)
    frac = float(cfg.get("calibration", {}).get("domain_calib_fraction", DEFAULT_DOMAIN_FRACTION))
    domain_split: tuple[list[str], list[str]] | None = None
    if "hard_neg" in qmeta:
        repos = sorted({m["repo"] for m in qmeta["hard_neg"].values()})
        domain_split = split_repos(repos, frac, get_rng(cfg, "domain_calibration"))
        log.info("domain calibration: %d calib repos, %d test repos", len(domain_split[0]), len(domain_split[1]))
    summary: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "config_path": cfg.get("_config_path"), "seed": cfg.get("seed"),
        "alphas": alphas, "alpha_keys": [alpha_key(a) for a in alphas], "bootstrap": n_boot,
        "length_bins": [list(map(int, b)) for b in cfg.get("eval", {}).get("length_bins_tokens", [[0, 10 ** 9]])],
        "decision_rule": "score > tau",
        "domain_calibration": {"fraction": frac, "calib_repos": domain_split[0] if domain_split else [],
                               "test_repos": domain_split[1] if domain_split else []},
        "data_stats": data_stats(cfg, qmeta), "methods": {},
    }
    for m in methods:
        loaded = {s: ss for s, ss in ((s, load_scored_set(cfg, m, s, qm)) for s, qm in qmeta.items()) if ss is not None}
        if not loaded:
            log.warning("%s: нет скоров ни для одного набора — пропуск", m)
            continue
        tm = time.perf_counter()
        summary["methods"][m] = method_metrics(cfg, m, loaded, alphas, n_boot, domain_split)
        log.info("%s: metrics done in %.1fs (sets: %s)", m, time.perf_counter() - tm, ",".join(loaded))
    summary["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
    return summary


def write_summary(cfg: dict[str, Any], summary: dict[str, Any]) -> Path:
    """results/summary.json + results/summary_schema.md (схема из докстринга модуля)."""
    rdir = resolve_path(cfg, "results")
    path = rdir / SUMMARY_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=_json_default)
    write_schema(cfg)
    log.info("summary → %s", path)
    return path


def write_schema(cfg: dict[str, Any]) -> Path:
    path = resolve_path(cfg, "results") / SCHEMA_FILE
    doc = __doc__ or ""
    start = doc.find("# Схема")
    with open(path, "w", encoding="utf-8") as f:
        f.write((doc[start:] if start >= 0 else doc).strip() + "\n")
    return path


def load_summary(cfg: dict[str, Any]) -> dict[str, Any] | None:
    return _read_json(resolve_path(cfg, "results") / SUMMARY_FILE)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else ("inf" if v > 0 else "-inf")
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return "inf" if o > 0 else "-inf"
    return str(o)
