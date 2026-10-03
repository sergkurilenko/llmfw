"""Метрики оценки (DESIGN.md §10) и схема results/summary.json.

Скоры (results/scores/<method>/<set>.jsonl) соединяются с запросами (data/queries/<set>.jsonl) по qid;
по каждому методу считаются пороги, TPR/FPR с интервалами, AUROC, латентность и память. Правило решения:
score > τ (см. smcode.calibration). Бутстрэп TPR/AUROC/FPR кластеризован по source_id (исходной функции):
все ~13 запросов одной функции (преобразования + окна partial) не независимы, поэтому для FPR рядом с
интервалом Клоппера–Пирсона (DESIGN §8, i.i.d.-предположение) даётся кластерный ДИ ``ci_cluster`` — именно он
используется в T3/F4 для вывода о нарушении гарантии. По той же причине калибровочные скоры не обменяемы с
новым негативом строго (обменяемые единицы — функции): объединённая калибровка — приближение второго порядка
(~1/n_функций), а порог ``conformal_single`` считается по одному случайному запросу на функцию (строгий вариант).

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
  "notes": {"exchangeability": str, "fpr_ci": str},   # оговорки для статьи (T3)
  "domain_calibration": {"fraction": f, "calib_repos": [...], "test_repos": [...]},   # общее для всех методов
  "warnings": [str],                            # например, неполные прогоны (--limit)
  "data_stats": {                               # data/functions/stats.json + статистика запросов
      "functions": {... содержимое stats.json ...} | null,
      "queries": {set: {"n": int, "label": 0|1, "by_transform": {t: n}, "by_lang": {lang: n},
                        "stats": {... <set>.stats.json ...}, "llm_stats": {... <set>.llm_stats.json ...}}}
  },
  "methods": {
    "<method>": {                               # имя = каталог results/scores/<method> (варианты: hybrid_rule, ...)
      "variant": {"base_method": str, "index_dir": str, "cfg_overrides": {...}} | null,   # из <set>.meta.json
      "incomplete": {set: {"n": int, "n_queries": int}} | null,   # скоров < 95 % запросов (прогон --limit?)
      "n_records": int|null, "memory_bytes": int|null, "bytes_per_record": float|null,
      "build_seconds": float|null, "disk_bytes": int|null,
      "n_scores": {set: int},                   # число соединённых (qid найден в запросах) скоров
      "thresholds": {alpha_key: {"conformal": τ|null, "conformal_single": τ|null, "mondrian": τ|null, "oracle": τ|null,
                                 "domain": τ|null, "windows": τ|null,
                                 "mondrian_cells": {cell: {"tau": τ|"inf", "n": int, "sufficient": bool}},  # cell = transform | partial@L
                                 "n_calib": int, "n_calib_single": int, "n_calib_domain": int, "min_n_calib": int,
                                 "tie_mass_at_tau": float|null}},   # доля калибровочных скоров, равных τ (конформный)
      "tpr": {alpha_key: {kind: GROUPS}},       # kind ∈ {conformal, conformal_single, mondrian, oracle, domain}; GROUPS =
            #   {"overall": RATE, "by_transform": {t: RATE}, "by_lang": {lang: RATE},
            #    "by_length_bin": {"lo-hi": RATE}, "by_partial_L": {"L": RATE}, "by_translate_dst": {lang: RATE},
            #    "misses_by_reason": {"n_misses": int, reason: int}}   # too_short / all_common / unsupported_lang /
            #                                                            # tokenize_error / no_hit (score 0) / below_tau
            #   RATE = {"value": float|null, "ci": [lo, hi]|null, "n": int, "k": int}  (ДИ — бутстрэп по кластерам)
      "tpr_windows": {alpha_key: {**GROUPS (overall, by_transform, by_partial_L, by_length_bin, misses_by_reason),
            #   "threshold": τ_w, "calibration": "public_calib_windows" | "public_calib (functions)", "n_calib": int,
            #   "fpr": {"public_test_windows": FPR}, "fpr_at_function_tau": {"public_test_windows": FPR},
            #   "confusion": {set: {...}}}} | null,   # сценарий IDE: порог по негативам формы окна (если набор есть);
            #   TPR identity-окон — проверка выравнивания индекса (окна индекса = окна запросов), не результат детекции
      "fpr": {alpha_key: {kind: {neg_set: FPR}}},   # neg_set ∈ {public_calib, public_test, hard_neg,
            #   hard_neg_test (только domain)}; FPR = RATE (ci — Клоппер–Пирсон) + "ci_cluster": [lo, hi]|null
            #   (кластерный бутстрэп по source_id) + {"by_lang": {lang: FPR}, "by_transform": {t: FPR},
            #   "max_subgroup": {"group": str, **FPR},
            #   "test_above_alpha": {"alpha", "z", "p_value", "se_cluster", "n_eff", "design_effect", "mde_80"}}
            #   — односторонний кластерно-робастный z-тест H0: FPR ≤ α (сэндвич-дисперсия по source_id) и минимально
            #   обнаружимый FPR при мощности 80 % (при α = 0.001 и ~5000 функций тест не отличает 0.001 от 0.003)
      "confusion": {alpha_key: {kind: {set: {"tp","fn"} | {"fp","tn"}}}},   # полная матрица ошибок
      "auroc": {neg_set: {"value": float|null, "ci": [lo, hi]|null, "n_pos": int, "n_neg": int}},
      "roc": {neg_set: {"fpr": [...], "tpr": [...]}},                      # прореженные кривые для F3
      "latency_ms": {"p50","p95","mean","n", "mode": str, "by_set": {set: {"p50","p95","n"}},
                     "benchmark": {...<set>.meta.json["latency_benchmark"]...}|null,
                     "end_to_end": {"p50": float|null, "p95": float|null, "source": str|null,   # сквозная латентность
                                    "components": {...}, "reason": str|null}},  # одного запроса (CPU, 1 поток) для T5/F7:
            #   per_query — p50/p95 прогона; batch_amortized (numpy, без энкодера) — энкодер CPU-1 батч 1
            #   (results/latency_encoder.json) + бенчмарк батча 1 (--bench, numpy_batch1); amortized_batch (torch) —
            #   бенчмарк батча 1 через query() (--bench, mode query). Без входов — null + reason.
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
    clopper_pearson,
    conformal_threshold,
    decide,
    domain_calibration,
    min_calibration_size,
    split_repos,
    threshold_at_fpr,
)
from smcode.config import get_rng, resolve_path
from smcode.eval.build_queries import QUERY_SETS, WINDOW_SET, all_sets, is_window_set
from smcode.eval.run_eval import is_limit_file, read_scores, read_scores_meta, scores_dir
from smcode.eval.tables import LATENCY_ENCODER_FILE, encoder_latency
from smcode.fingerprint.index import INDEX_STATS_FILE, REGISTRY, index_dir
from smcode.types import read_jsonl

log = logging.getLogger(__name__)

SUMMARY_FILE = "summary.json"
SCHEMA_FILE = "summary_schema.md"
NEG_SETS: tuple[str, ...] = ("public_calib", "public_test", "hard_neg")
KINDS: tuple[str, ...] = ("conformal", "conformal_single", "mondrian", "oracle", "domain")
WINDOW_CALIB_SET = "public_calib_windows"
WINDOW_TEST_SET = "public_test_windows"
MIN_SUBGROUP = 50          # минимальный размер подгруппы для "max_subgroup"
ROC_POINTS = 300
DEFAULT_DOMAIN_FRACTION = 0.5
INCOMPLETE_FRACTION = 0.95   # скоров < 95 % запросов набора → метод помечается как неполный
AMORTIZED_PREFIXES: tuple[str, ...] = ("batch_amortized", "amortized_batch")
MDE_POWER_Z = 0.8416  # z_{0.80}: минимально обнаружимый FPR при мощности 80 % одностороннего теста уровня 5 %
_SET_SEED_OFFSET = {"protected": 0, WINDOW_SET: 1, "public_calib": 2, "public_test": 3, "hard_neg": 4, "hard_neg_test": 5,
                    WINDOW_CALIB_SET: 7, WINDOW_TEST_SET: 8}
NOTES = {
    "exchangeability": ("Обменяемые единицы — функции-источники (все преобразования одной функции коррелированы); "
                        "объединённая калибровка по всем запросам — приближение (погрешность ~1/n_функций), строгий вариант — "
                        "порог conformal_single по одному случайному запросу на функцию (маргинальная гарантия) и "
                        "мондриановский порог mondrian = max по ячейкам (преобразование, L) (гарантия по каждому классу)."),
    "fpr_ci": ("ДИ Клоппера–Пирсона предполагает независимость запросов и занижает ширину при кластеризации по функции; "
               "вывод о превышении α делается по кластерно-робастному одностороннему z-тесту (test_above_alpha: p_value) "
               "и кластерному бутстрэпу (ci_cluster); mde_80 — минимально обнаружимый FPR, при α = 0.001 и ~5000 функций "
               "тест не отличает FPR 0.001 от 0.003."),
    "windows": ("Сценарий IDE: порог окон калибруется по public_calib_windows (негативы формы окна), FPR — на "
                "public_test_windows; TPR identity-окон protected_windows — проверка выравнивания индекса (окна индекса = "
                "окна запросов, M0 = 1 по построению), информативны только частичные окна partial@L."),
    "fingerprint_invariance": ("M1/M2 на abstract(full)-токенах — детекторы клонов типа 2, по построению инвариантные к "
                               "reformat/strip_comments/rename_ids/change_literals/combo; информативные классы RQ1 — "
                               "insert_deadcode, reorder_stmts, partial, paraphrase, translate; вариант winnowing_lexical "
                               "(fingerprint.token_mode=lexical) — классический Moss, который эти классы ломают."),
}


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
    reason: np.ndarray | None = None  # details["reason"] из строки скоров ("" — нет)

    @property
    def n(self) -> int:
        return int(self.score.size)

    def subset(self, mask: np.ndarray) -> "ScoredSet":
        return ScoredSet(self.name, self.label, self.qid[mask], self.score[mask], self.latency[mask], self.transform[mask],
                         self.lang[mask], self.n_tokens[mask], self.L[mask], self.repo[mask], self.source_id[mask],
                         self.dst_lang[mask], self.best_id[mask], int(mask.sum()), self.latency_mode,
                         reason=self.reason[mask] if self.reason is not None else None)


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
                                                  "repo", "source_id", "dst_lang", "best_id", "reason")}
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
        cols["dst_lang"].append(m["dst_lang"]); cols["best_id"].append(r.get("best_id") or ""); cols["reason"].append(r.get("reason") or "")
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
        n_queries=len(qmeta), latency_mode=latency_mode, reason=obj("reason"),
    )


def load_scored_set(cfg: dict[str, Any], method: str, set_name: str, qmeta: dict[str, dict[str, Any]]) -> ScoredSet | None:
    rows = read_scores(cfg, method, set_name)
    if not rows:
        return None
    meta = read_scores_meta(cfg, method, set_name)
    return join_scores(rows, qmeta, set_name, latency_mode=str(meta.get("latency_mode", "")))


def available_methods(cfg: dict[str, Any]) -> list[str]:
    """Каталоги results/scores/<method>, где есть хотя бы один полный файл скоров (не *.limitN.jsonl); известные — первыми."""
    base = scores_dir(cfg)
    found = [d.name for d in base.iterdir() if d.is_dir() and any(not is_limit_file(p) for p in d.glob("*.jsonl"))]
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


def one_per_cluster(clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Маска «один случайный элемент на кластер» (калибровка conformal_single: один запрос на функцию)."""
    c = np.asarray(clusters, dtype=object).astype(str)
    n = int(c.size)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    perm = rng.permutation(n)
    _, first = np.unique(c[perm], return_index=True)
    mask[perm[first]] = True
    return mask


def tie_mass(scores: np.ndarray, tau: float) -> float | None:
    """Доля скоров, равных τ (масса связей в пороге): > 0 означает, что строгое правило score > τ консервативно."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0 or tau is None or not np.isfinite(tau):
        return None
    return float(np.mean(s == tau))


def mondrian_cells(ss: ScoredSet) -> np.ndarray:
    """Ячейка мондриановской калибровки каждого запроса: transform, для partial — 'partial@L' (DESIGN §8)."""
    cells = ss.transform.astype(str).copy()
    part = ss.transform == "partial"
    cells[part] = np.array([f"partial@{int(L)}" for L in ss.L[part]], dtype=object)
    return cells


def mondrian_threshold(calib: ScoredSet, alpha: float) -> tuple[float, dict[str, dict[str, Any]]]:
    """Мондриановский порог: τ_t по каждой ячейке (преобразование, L) калибровочных негативов; порог развёртывания —
    max_t τ_t по ячейкам с достаточным числом функций (n ≥ ⌈1/α⌉ − 1, DESIGN §8). Ячейки с недостатком данных получают
    τ = +inf, помечаются sufficient=false и в max не входят (гарантия по классу относится к покрытым ячейкам).
    Возвращает (τ, {cell: {"tau", "n", "sufficient"}})."""
    cells = mondrian_cells(calib)
    need = min_calibration_size(alpha)
    out: dict[str, dict[str, Any]] = {}
    finite: list[float] = []
    for c in sorted(set(cells.tolist())):
        m = cells == c
        n_fn = int(len(set(calib.source_id[m].tolist())))
        ok = n_fn >= need
        tau = conformal_threshold(calib.score[m], alpha) if ok else float("inf")
        out[str(c)] = {"tau": tau if np.isfinite(tau) else "inf", "n": n_fn, "sufficient": bool(ok and np.isfinite(tau))}
        if ok and np.isfinite(tau):
            finite.append(float(tau))
    return (max(finite) if finite else float("inf")), out


def cluster_robust_test(hits: np.ndarray, clusters: np.ndarray, alpha: float) -> dict[str, Any]:
    """Односторонний кластерно-робастный z-тест H0: FPR ≤ α (сэндвич-оценка дисперсии доли по кластерам source_id)
    и минимально обнаружимый FPR mde_80 (мощность 80 %, уровень 5 %) при эффективном объёме n_eff = n / design_effect."""
    from scipy.stats import norm

    h = np.asarray(hits, dtype=np.float64)
    c = np.asarray(clusters)
    n = int(h.size)
    out: dict[str, Any] = {"alpha": float(alpha), "n": n, "z": None, "p_value": None, "se_cluster": None, "n_eff": None,
                           "design_effect": None, "mde_80": None}
    if n == 0:
        return out
    _, cidx = np.unique(c.astype(str), return_inverse=True)
    C = int(cidx.max()) + 1
    hc = np.bincount(cidx, weights=h, minlength=C)
    tc = np.bincount(cidx, minlength=C).astype(np.float64)
    p = float(h.sum() / n)
    resid = hc - p * tc
    var = (C / (C - 1) if C > 1 else 1.0) * float((resid ** 2).sum()) / float(n) ** 2
    se = float(np.sqrt(max(var, 0.0)))
    out["se_cluster"] = se
    if se > 0:
        z = (p - alpha) / se
        out["z"], out["p_value"] = float(z), float(1.0 - norm.cdf(z))
    else:
        out["z"], out["p_value"] = None, (1.0 if p <= alpha else 0.0)
    binom_var = p * (1 - p) / n
    deff = (var / binom_var) if binom_var > 0 else None
    n_eff = (n / deff) if deff and deff > 0 else float(n)
    out["design_effect"], out["n_eff"] = deff, n_eff
    out["mde_80"] = float(alpha + (norm.ppf(0.95) + MDE_POWER_Z) * np.sqrt(alpha * (1 - alpha) / max(n_eff, 1.0)))
    return out


MISS_REASONS: tuple[str, ...] = ("too_short", "all_common", "unsupported_lang", "tokenize_error", "no_hit", "below_tau")


def misses_by_reason(ss: ScoredSet, tau: float) -> dict[str, int]:
    """Пропуски членов (score ≤ τ) по причинам: reason из строки скоров (too_short / all_common / unsupported_lang /
    tokenize_error, M0–M2), иначе no_hit (score = 0 без причины) или below_tau (0 < score ≤ τ)."""
    hits = decide(ss.score, tau)
    miss = ~hits
    out: dict[str, int] = {"n_misses": int(miss.sum())}
    if not miss.any():
        return out
    reasons = ss.reason if ss.reason is not None else np.full(ss.n, "", dtype=object)
    for i in np.flatnonzero(miss):
        r = str(reasons[i]) if reasons[i] else ("no_hit" if ss.score[i] <= 0 else "below_tau")
        out[r] = out.get(r, 0) + 1
    return out


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
    out["misses_by_reason"] = misses_by_reason(ss, tau)
    return out


def _fpr_rate(hits: np.ndarray, mask: np.ndarray, boot: ClusterBootstrap | None, conf: float,
              alpha: float | None = None, clusters: np.ndarray | None = None) -> dict[str, Any]:
    """FPR-блок подмножества: ДИ Клоппера–Пирсона (ci) + кластерный бутстрэп (ci_cluster) + (при alpha и clusters)
    кластерно-робастный тест превышения α (test_above_alpha)."""
    n = int(mask.sum())
    k = int(hits[mask].sum()) if n else 0
    lo, hi = clopper_pearson(k, n, conf) if n else (0.0, 1.0)
    out = {"value": (k / n) if n else None, "ci": [lo, hi], "n": n, "k": k,
           "ci_cluster": boot.rate_ci(hits, mask, conf) if (boot is not None and n) else None}
    if alpha is not None and clusters is not None and n:
        out["test_above_alpha"] = cluster_robust_test(hits[mask], clusters[mask], alpha)
    return out


def fpr_block(ss: ScoredSet, tau: float, conf: float = 0.95, boot: ClusterBootstrap | None = None,
              alpha: float | None = None) -> dict[str, Any]:
    """FPR overall, по языку, по преобразованию, худшая подгруппа (n ≥ MIN_SUBGROUP).

    ci — Клоппер–Пирсон (i.i.d.-предположение); ci_cluster — персентильный ДИ кластерного бутстрэпа по source_id
    (учитывает, что все запросы одной функции коррелированы); при alpha — test_above_alpha (cluster_robust_test)
    для общего FPR — основа вывода о превышении α в T3/F4."""
    hits = decide(ss.score, tau)
    out = _fpr_rate(hits, np.ones(ss.n, dtype=bool), boot, conf, alpha=alpha, clusters=ss.source_id)
    out["by_lang"] = {str(l): _fpr_rate(hits, ss.lang == l, boot, conf) for l in sorted(set(ss.lang.tolist()))}
    out["by_transform"] = {str(t): _fpr_rate(hits, ss.transform == t, boot, conf) for t in sorted(set(ss.transform.tolist()))}
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


# ----------------------------------------------------------------------------- латентность


def end_to_end_latency(lat: dict[str, Any], bench: dict[str, Any] | None, modes: Iterable[str],
                       enc: dict[str, Any] | None) -> dict[str, Any]:
    """Оценка сквозной латентности одного запроса (CPU, 1 поток) для T5/F7 — {p50, p95, source, components, reason}.

    per_query → p50/p95 прогона eval. numpy-путь (batch_amortized: только ANN/переранжирование, без энкодера) →
    энкодер CPU-1 батч 1 из results/latency_encoder.json (enc["cpu1"]) + бенчмарк батча 1 (--bench, mode numpy_batch1);
    сумма p95 — верхняя оценка. torch-путь (amortized_batch: query_batch кодирует батч) → бенчмарк батча 1 через
    query() (mode query, с энкодером). Без нужных входов p50/p95 = None и причина в reason."""
    modes = [m for m in modes if m]
    amort = [m for m in modes if m.startswith(AMORTIZED_PREFIXES)]
    out: dict[str, Any] = {"p50": None, "p95": None, "source": None, "components": {}, "reason": None}
    if not amort:
        if lat.get("p95") is None:
            out["reason"] = "нет данных о латентности"
            return out
        out.update(p50=lat.get("p50"), p95=lat.get("p95"), source="per_query (прогон eval)")
        return out
    bmode = (bench or {}).get("mode")
    if any(m.startswith("batch_amortized") for m in amort):
        enc1 = (enc or {}).get("cpu1") or {}
        if enc1.get("p95_ms") is None:
            out["reason"] = "нет results/latency_encoder.json (энкодер, CPU 1 поток, батч 1; scripts/06_embed.py)"
        elif not bench or bmode != "numpy_batch1":
            out["reason"] = "нет бенчмарка батча 1 без энкодера (scripts/07_eval.py --bench N)"
        else:
            out.update(p50=float(enc1["p50_ms"]) + float(bench["p50_ms"]), p95=float(enc1["p95_ms"]) + float(bench["p95_ms"]),
                       source="энкодер CPU-1 (latency_encoder.json) + ANN/переранжирование батч 1 (--bench); сумма p95 — верхняя оценка",
                       components={"encoder": {"p50": float(enc1["p50_ms"]), "p95": float(enc1["p95_ms"])},
                                   "search": {"p50": float(bench["p50_ms"]), "p95": float(bench["p95_ms"])}})
        return out
    if bench and bmode == "query":
        out.update(p50=float(bench["p50_ms"]), p95=float(bench["p95_ms"]), source="бенчмарк батча 1 через query() (--bench, с энкодером)")
        return out
    out["reason"] = "нет бенчмарка батча 1 (scripts/07_eval.py --bench N)"
    return out


def load_encoder_latency(cfg: dict[str, Any]) -> dict[str, Any]:
    """results/latency_encoder.json → encoder_latency(...) ({} если файла нет)."""
    return encoder_latency(_read_json(resolve_path(cfg, "results") / LATENCY_ENCODER_FILE))


# ----------------------------------------------------------------------------- метод


def _variant_info(cfg: dict[str, Any], method: str, sets: dict[str, ScoredSet]) -> dict[str, Any] | None:
    """Базовый метод, каталог индекса и переопределения конфига варианта (из <set>.meta.json)."""
    for s in sets:
        m = read_scores_meta(cfg, method, s)
        if m:
            return {"base_method": m.get("method"), "index_dir": m.get("index_dir"), "cfg_overrides": m.get("cfg_overrides") or {}}
    return None


def _index_info(cfg: dict[str, Any], method: str, sets: dict[str, ScoredSet]) -> dict[str, Any]:
    """n_records / memory_bytes / build_seconds / disk_bytes из meta скоров, results/index_stats.json (ключ метода,
    базового метода или варианта '<base>@<каталог>' для индекса из --set paths.indexes), meta.json индекса."""
    info: dict[str, Any] = {"n_records": None, "memory_bytes": None, "build_seconds": None, "disk_bytes": None}
    base, idir = None, None
    for s in sets:
        m = read_scores_meta(cfg, method, s)
        if m:
            base, idir = m.get("method"), m.get("index_dir")
        if m.get("memory_bytes") is not None:
            info["memory_bytes"], info["n_records"] = m.get("memory_bytes"), m.get("n_records")
            break
    stats_path = resolve_path(cfg, "results") / INDEX_STATS_FILE
    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                all_st = json.load(f)
            st = all_st.get(method)
            if st is None and base and base in REGISTRY:
                if idir is None or Path(idir) == index_dir(cfg, base):
                    st = all_st.get(base)   # вариант на базовом индексе (например hybrid_rule)
                else:
                    st = all_st.get(f"{base}@{Path(idir).parent.name}")   # вариантный каталог индексов (04 --set paths.indexes)
                    if st is not None and st.get("path") and Path(st["path"]) != Path(idir):
                        st = None
            st = st or {}
            for k in ("n_records", "memory_bytes", "build_seconds", "disk_bytes"):
                if info.get(k) is None and st.get(k) is not None:
                    info[k] = st[k]
        except json.JSONDecodeError:
            log.warning("corrupt %s", stats_path)
    meta_path = Path(idir) / "meta.json" if idir else (index_dir(cfg, method) / "meta.json" if method in REGISTRY else None)
    if meta_path and meta_path.exists() and any(info[k] is None for k in info):
        with open(meta_path, "r", encoding="utf-8") as f:
            im = json.load(f)
        for k in ("n_records", "memory_bytes", "build_seconds", "disk_bytes"):  # build_seconds/disk_bytes пишет build_index
            if info[k] is None and im.get(k) is not None:
                info[k] = im[k]
    n, mem = info.get("n_records"), info.get("memory_bytes")
    info["bytes_per_record"] = (float(mem) / n) if (n and mem is not None) else None
    return info


def method_metrics(cfg: dict[str, Any], method: str, sets: dict[str, ScoredSet], alphas: Sequence[float],
                   n_boot: int, domain_split: tuple[list[str], list[str]] | None = None,
                   encoder: dict[str, Any] | None = None) -> dict[str, Any]:
    """Все метрики одного метода по загруженным наборам (ключи — имена наборов).
    encoder — encoder_latency(results/latency_encoder.json) для сквозной латентности (None → читается из results/)."""
    bins = [list(map(int, b)) for b in cfg.get("eval", {}).get("length_bins_tokens", [[0, 10 ** 9]])]
    conf = 0.95
    seed = get_rng(cfg, "eval:bootstrap").getrandbits(32)
    if encoder is None:
        encoder = load_encoder_latency(cfg)
    prot, calib, test, hard, win = (sets.get(k) for k in ("protected", "public_calib", "public_test", "hard_neg", WINDOW_SET))
    wcal, wtest = sets.get(WINDOW_CALIB_SET), sets.get(WINDOW_TEST_SET)
    out: dict[str, Any] = {**_index_info(cfg, method, sets), "variant": _variant_info(cfg, method, sets),
                           "incomplete": None, "n_scores": {k: v.n for k, v in sets.items()},
                           "sets": {k: {"n": v.n, "n_queries": v.n_queries, "latency_mode": v.latency_mode} for k, v in sets.items()}}
    boots: dict[str, ClusterBootstrap | None] = {}

    def boot_for(name: str, ss: ScoredSet) -> ClusterBootstrap | None:
        """Кластерный бутстрэп набора (строится один раз на метод, общий для всех α и порогов)."""
        if name not in boots:
            off = _SET_SEED_OFFSET.get(name, 6)
            boots[name] = ClusterBootstrap(ss.source_id, n_boot, np.random.default_rng(seed + off)) if ss.n else None
        return boots[name]

    hard_test: ScoredSet | None = None
    if calib is not None and calib.n and hard is not None and hard.n and domain_split is not None:
        cal_set = set(domain_split[0])
        in_cal = np.fromiter((r in cal_set for r in hard.repo), dtype=bool, count=hard.n)
        hard_test = hard.subset(~in_cal)
    single_mask = one_per_cluster(calib.source_id, np.random.default_rng(seed + 11)) if (calib is not None and calib.n) else None

    out["thresholds"], out["tpr"], out["fpr"], out["confusion"], out["tpr_windows"] = {}, {}, {}, {}, ({} if win is not None else None)
    for a in alphas:
        ak = alpha_key(a)
        taus: dict[str, float | None] = {k: None for k in KINDS}
        n_calib = calib.n if calib is not None else 0
        n_single = int(single_mask.sum()) if single_mask is not None else 0
        n_dom = 0
        tie = None
        cells: dict[str, dict[str, Any]] = {}
        if calib is not None and calib.n:
            taus["conformal"] = conformal_threshold(calib.score, a)
            taus["conformal_single"] = conformal_threshold(calib.score[single_mask], a)
            taus["mondrian"], cells = mondrian_threshold(calib, a)
            tie = tie_mass(calib.score, taus["conformal"])
        if test is not None and test.n:
            taus["oracle"] = threshold_at_fpr(test.score, a)
        if calib is not None and calib.n and hard is not None and hard.n and domain_split is not None:
            dc = domain_calibration(calib.score, hard.score, hard.repo.tolist(), a, calib_repos=domain_split[0])
            taus["domain"], n_dom = dc.tau, dc.n_calib
        # порог окон (сценарий IDE): по негативам формы окна, если набор есть; иначе порог функций с пометкой
        tau_w: float | None = None
        win_calib = None
        if wcal is not None and wcal.n:
            tau_w, win_calib = conformal_threshold(wcal.score, a), WINDOW_CALIB_SET
        elif taus["conformal"] is not None:
            tau_w, win_calib = taus["conformal"], "public_calib (functions; no window negatives)"
        out["thresholds"][ak] = {**{k: (None if v is None else (v if np.isfinite(v) else "inf")) for k, v in taus.items()},
                                 "windows": None if tau_w is None else (tau_w if np.isfinite(tau_w) else "inf"),
                                 "mondrian_cells": cells, "n_calib": n_calib, "n_calib_single": n_single, "n_calib_domain": n_dom,
                                 "n_calib_windows": wcal.n if wcal is not None else 0,
                                 "min_n_calib": min_calibration_size(a), "tie_mass_at_tau": tie}
        if taus["conformal"] is not None and n_calib < min_calibration_size(a):
            log.warning("%s α=%s: калибровочных негативов %d < %d → порог +inf", method, ak, n_calib, min_calibration_size(a))
        if tie:
            log.info("%s α=%s: %.1f%% калибровочных скоров равны τ=%.4g (строгое правило консервативно)", method, ak, 100 * tie, taus["conformal"])
        out["tpr"][ak], out["fpr"][ak], out["confusion"][ak] = {}, {}, {}
        for kind in KINDS:
            tau = taus[kind]
            if tau is None:
                continue
            if prot is not None and prot.n:
                out["tpr"][ak][kind] = tpr_groups(prot, tau, boot_for("protected", prot), bins)
            negs: dict[str, ScoredSet] = {}
            if kind == "domain":
                if test is not None:
                    negs["public_test"] = test
                if hard_test is not None:
                    negs["hard_neg_test"] = hard_test
            else:
                negs = {k: v for k, v in ((n, sets.get(n)) for n in NEG_SETS) if v is not None}
            out["fpr"][ak][kind] = {n: fpr_block(ss, tau, conf, boot_for(n, ss), alpha=a) for n, ss in negs.items() if ss.n}
            conf_m: dict[str, Any] = {}
            if prot is not None and prot.n:
                conf_m["protected"] = _confusion(prot, tau)
            if win is not None and win.n:
                conf_m[WINDOW_SET] = _confusion(win, tau)
            conf_m.update({n: _confusion(ss, tau) for n, ss in negs.items() if ss.n})
            out["confusion"][ak][kind] = conf_m
        if win is not None and win.n and tau_w is not None:
            g = tpr_groups(win, tau_w, boot_for(WINDOW_SET, win), bins)
            block: dict[str, Any] = {k: g[k] for k in ("overall", "by_transform", "by_partial_L", "by_length_bin", "misses_by_reason")}
            block.update({"threshold": tau_w if np.isfinite(tau_w) else "inf", "calibration": win_calib,
                          "n_calib": wcal.n if wcal is not None else 0, "fpr": {}, "fpr_at_function_tau": {},
                          "confusion": {WINDOW_SET: _confusion(win, tau_w)}})
            if wtest is not None and wtest.n:
                block["fpr"][WINDOW_TEST_SET] = fpr_block(wtest, tau_w, conf, boot_for(WINDOW_TEST_SET, wtest), alpha=a)
                block["confusion"][WINDOW_TEST_SET] = _confusion(wtest, tau_w)
                if taus["conformal"] is not None:
                    block["fpr_at_function_tau"][WINDOW_TEST_SET] = fpr_block(wtest, taus["conformal"], conf,
                                                                              boot_for(WINDOW_TEST_SET, wtest), alpha=a)
            if wcal is not None and wcal.n:
                block["fpr"][WINDOW_CALIB_SET] = fpr_block(wcal, tau_w, conf, boot_for(WINDOW_CALIB_SET, wcal), alpha=a)
            out["tpr_windows"][ak] = block

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
    lat = {**_percentiles(lat_all), "mode": ",".join(modes), "by_set": {k: _percentiles(v.latency) for k, v in sets.items()},
           "benchmark": bench}
    lat["end_to_end"] = end_to_end_latency(lat, bench, modes, encoder)
    if lat["end_to_end"]["p95"] is None:
        log.warning("%s: сквозная латентность не определена (%s)", method, lat["end_to_end"]["reason"])
    out["latency_ms"] = lat
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
        queries[s] = {"n": len(meta), "label": label, "window_set": is_window_set(s), "by_transform": dict(sorted(by_t.items())),
                      "by_lang": dict(sorted(by_l.items())),
                      "stats": _read_json(qdir / f"{s}.stats.json"), "llm_stats": _read_json(qdir / f"{s}.llm_stats.json")}
    return {"functions": _read_json(resolve_path(cfg, "functions") / "stats.json"), "queries": queries}


# ----------------------------------------------------------------------------- точка входа


def compute_summary(cfg: dict[str, Any], methods: Iterable[str] | None = None, bootstrap: int | None = None,
                    sets: Iterable[str] | None = None) -> dict[str, Any]:
    """Считает метрики всех методов (по умолчанию — все каталоги results/scores/*) и возвращает summary (см. схему).
    Метод, у которого скоров < 95 % запросов набора (например, отладочный прогон), помечается incomplete + warnings."""
    t0 = time.perf_counter()
    methods = list(methods) if methods else available_methods(cfg)
    alphas = [float(a) for a in cfg.get("calibration", {}).get("alphas", [0.01])]
    n_boot = int(cfg.get("eval", {}).get("bootstrap", 1000) if bootstrap is None else bootstrap)
    set_names = list(sets) if sets else all_sets(cfg)
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
    encoder = load_encoder_latency(cfg)
    summary: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "config_path": cfg.get("_config_path"), "seed": cfg.get("seed"),
        "alphas": alphas, "alpha_keys": [alpha_key(a) for a in alphas], "bootstrap": n_boot,
        "length_bins": [list(map(int, b)) for b in cfg.get("eval", {}).get("length_bins_tokens", [[0, 10 ** 9]])],
        "decision_rule": "score > tau", "notes": dict(NOTES),
        "domain_calibration": {"fraction": frac, "calib_repos": domain_split[0] if domain_split else [],
                               "test_repos": domain_split[1] if domain_split else []},
        "warnings": [],
        "data_stats": data_stats(cfg, qmeta), "methods": {},
    }
    for m in methods:
        loaded = {s: ss for s, ss in ((s, load_scored_set(cfg, m, s, qm)) for s, qm in qmeta.items()) if ss is not None}
        if not loaded:
            log.warning("%s: нет скоров ни для одного набора — пропуск", m)
            continue
        tm = time.perf_counter()
        mm = method_metrics(cfg, m, loaded, alphas, n_boot, domain_split, encoder=encoder)
        incomplete = {s: {"n": ss.n, "n_queries": ss.n_queries} for s, ss in loaded.items()
                      if ss.n_queries and ss.n < INCOMPLETE_FRACTION * ss.n_queries}
        if incomplete:
            mm["incomplete"] = incomplete
            parts = ", ".join(f"{s}: {v['n']}/{v['n_queries']}" for s, v in incomplete.items())
            msg = f"{m}: неполные скоры ({parts}) — прогон --limit? перезапустите scripts/07_eval.py --force"
            log.warning(msg)
            summary["warnings"].append(msg)
        summary["methods"][m] = mm
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
        f.write((doc[start:] if start >= 0 else doc).strip() + "\n\n## Оговорки\n\n" +
                "\n".join(f"- **{k}**: {v}" for k, v in NOTES.items()) + "\n")
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
