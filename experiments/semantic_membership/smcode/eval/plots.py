"""Рисунки F1..F7 (DESIGN.md §10) из results/summary.json → cfg.paths.figures/*.png (300 dpi, подписи на русском).

F1 TPR по преобразованиям; F2 TPR vs длина (бины токенов и L partial); F3 ROC члены vs hard_neg/public_test;
F4 заявленный α vs эмпирический FPR; F5 абляция гибрида; F6 приватность–полезность (results/privacy.json);
F7 латентность vs размер индекса (results/latency.json или точки из summary). Недостающие входы → рисунок
пропускается с сообщением в лог. Цвета методов — фиксированный порядок категориальной палитры плюс маркеры.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402

from smcode.config import resolve_path  # noqa: E402
from smcode.eval.tables import LATENCY_FILE, PRIVACY_FILE, method_label, order_methods, order_transforms, pct, transform_label  # noqa: E402
from smcode.fingerprint.index import REGISTRY  # noqa: E402

log = logging.getLogger(__name__)

DPI = 300
FIGURE_FILES: dict[str, str] = {
    "F1": "F1_tpr_by_transform.png", "F2": "F2_tpr_vs_length.png", "F3": "F3_roc_hardneg.png",
    "F4": "F4_fpr_claimed_vs_empirical.png", "F5": "F5_hybrid_ablation.png", "F6": "F6_privacy_utility.png",
    "F7": "F7_latency_vs_index.png",
}
# Категориальная палитра (фиксированный порядок слотов, светлая поверхность); 9-й и далее — серый.
PALETTE: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
OTHER = "#8a8984"
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


# ----------------------------------------------------------------------------- утилиты


def style_for(methods: Sequence[str]) -> dict[str, tuple[str, str]]:
    """method → (цвет, маркер): известные методы занимают слоты в порядке REGISTRY, остальные — следующие."""
    ordered = order_methods(list(methods))
    out: dict[str, tuple[str, str]] = {}
    for i, m in enumerate(ordered):
        slot = list(REGISTRY).index(m) if m in REGISTRY else len(REGISTRY) + [x for x in ordered if x not in REGISTRY].index(m)
        out[m] = (PALETTE[slot] if slot < len(PALETTE) else OTHER, MARKERS[slot % len(MARKERS)])
    return out


def _get(d: Any, *keys: Any, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _val_err(block: dict[str, Any] | None) -> tuple[float, float, float]:
    """(значение, ошибка вниз, ошибка вверх) из RATE-блока; NaN если нет."""
    if not block or block.get("value") is None:
        return np.nan, 0.0, 0.0
    v = float(block["value"])
    ci = block.get("ci")
    if not ci or ci[0] is None:
        return v, 0.0, 0.0
    return v, max(0.0, v - float(ci[0])), max(0.0, float(ci[1]) - v)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _grouped_bars(ax: plt.Axes, cats: Sequence[str], series: dict[str, list[tuple[float, float, float]]],
                  styles: dict[str, tuple[str, str]], labels: dict[str, str]) -> None:
    n = max(1, len(series))
    width = 0.8 / n
    x = np.arange(len(cats))
    for i, (name, vals) in enumerate(series.items()):
        v = np.array([t[0] for t in vals]); lo = np.array([t[1] for t in vals]); hi = np.array([t[2] for t in vals])
        color = styles.get(name, (OTHER, "o"))[0]
        ax.bar(x + (i - (n - 1) / 2) * width, np.nan_to_num(v), width * 0.92, color=color, label=labels.get(name, name),
               yerr=[lo, hi] if np.any(lo + hi > 0) else None, error_kw={"ecolor": MUTED, "elinewidth": 0.7, "capsize": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="x", visible=False)


# ----------------------------------------------------------------------------- рисунки


def fig_F1(summary: dict[str, Any], path: Path) -> Path | None:
    """F1: TPR при α₀ по классам преобразований; группы столбцов — методы; линия — TPR winnowing на identity."""
    methods = order_methods(list(summary.get("methods", {})))
    a0 = (summary.get("alpha_keys") or ["0.01"])[0]
    transforms = order_transforms(sorted({t for m in methods for t in _get(summary, "methods", m, "tpr", a0, "conformal", "by_transform", default={})}))
    if not methods or not transforms:
        return None
    styles = style_for(methods)
    series = {m: [_val_err(_get(summary, "methods", m, "tpr", a0, "conformal", "by_transform", t)) for t in transforms] for m in methods}
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(transforms) + 2), 3.4))
    _grouped_bars(ax, [transform_label(t) for t in transforms], series, styles, {m: method_label(m) for m in methods})
    ref = _get(summary, "methods", "winnowing", "tpr", a0, "conformal", "by_transform", "identity", "value")
    if ref is not None:
        ax.axhline(float(ref), color=MUTED, linestyle="--", linewidth=1.0, label="M1 winnowing на T0 identity")
    ax.set_xlabel("Класс преобразования")
    ax.set_ylabel(f"TPR при FPR = {pct(a0)}")
    ax.set_title("Полнота по классам преобразований (конформный порог, 95 % ДИ бутстрэпа)")
    ax.legend(ncol=min(4, len(methods) + 1), loc="upper center", bbox_to_anchor=(0.5, -0.32))
    return _save(fig, path)


def fig_F2(summary: dict[str, Any], path: Path, cfg: dict[str, Any] | None = None) -> Path | None:
    """F2: (а) TPR по бинам длины запроса, (б) TPR по L для partial; вертикаль — гарантийная длина winnowing w+k−1."""
    methods = order_methods(list(summary.get("methods", {})))
    a0 = (summary.get("alpha_keys") or ["0.01"])[0]
    bins = summary.get("length_bins") or []
    if not methods or not bins:
        return None
    styles = style_for(methods)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2))
    keys = [f"{lo}-{hi}" for lo, hi in bins]
    xlabels = [f"{lo}–{hi}" if hi < 10 ** 5 else f">{lo}" for lo, hi in bins]
    any_data = False
    for m in methods:
        vals = [_val_err(_get(summary, "methods", m, "tpr", a0, "conformal", "by_length_bin", k)) for k in keys]
        v = np.array([t[0] for t in vals])
        if np.all(np.isnan(v)):
            continue
        any_data = True
        c, mk = styles[m]
        ax1.errorbar(np.arange(len(keys)), v, yerr=[[t[1] for t in vals], [t[2] for t in vals]], color=c, marker=mk, ms=4,
                     lw=1.6, capsize=1.5, elinewidth=0.7, label=method_label(m))
    fp = (cfg or {}).get("fingerprint", {})
    guard = int(fp.get("w", 25)) + int(fp.get("k", 25)) - 1
    for i, (lo, hi) in enumerate(bins):
        if lo <= guard < hi:
            ax1.axvline(i - 0.5 + (guard - lo) / max(1, hi - lo), color=MUTED, linestyle=":", lw=1.0)
            ax1.text(i - 0.5 + (guard - lo) / max(1, hi - lo), 0.02, f" w+k−1 = {guard}", color=MUTED, fontsize=7, va="bottom")
            break
    ax1.set_xticks(np.arange(len(keys)))
    ax1.set_xticklabels(xlabels)
    ax1.set_xlabel("Длина запроса, токенов")
    ax1.set_ylabel(f"TPR при FPR = {pct(a0)}")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("(а) По бинам длины")
    Ls = sorted({int(L) for m in methods for L in _get(summary, "methods", m, "tpr", a0, "conformal", "by_partial_L", default={})})
    for m in methods:
        vals = [_val_err(_get(summary, "methods", m, "tpr", a0, "conformal", "by_partial_L", str(L))) for L in Ls]
        v = np.array([t[0] for t in vals])
        if not Ls or np.all(np.isnan(v)):
            continue
        c, mk = styles[m]
        ax2.errorbar(Ls, v, yerr=[[t[1] for t in vals], [t[2] for t in vals]], color=c, marker=mk, ms=4, lw=1.6, capsize=1.5,
                     elinewidth=0.7, label=method_label(m))
    if Ls:
        ax2.set_xscale("log")
        ax2.xaxis.set_minor_locator(ticker.NullLocator())
        ax2.set_xticks(Ls)
        ax2.set_xticklabels([str(L) for L in Ls])
    ax2.set_xlabel("Окно partial L, строк")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("(б) Класс partial по L")
    if not any_data:
        plt.close(fig)
        return None
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=min(5, len(labels)))
    fig.suptitle("Полнота в зависимости от длины фрагмента", y=1.02)
    return _save(fig, path)


def fig_F3(summary: dict[str, Any], path: Path) -> Path | None:
    """F3: ROC члены vs hard_neg (сплошные) и vs public_test (пунктир), лог-ось FPR; в легенде AUROC [ДИ]."""
    methods = order_methods(list(summary.get("methods", {})))
    styles = style_for(methods)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    drawn = False
    for m in methods:
        c, _ = styles[m]
        for neg, ls in (("hard_neg", "-"), ("public_test", "--")):
            roc = _get(summary, "methods", m, "roc", neg)
            if not roc or not roc.get("fpr"):
                continue
            fpr, tpr = np.asarray(roc["fpr"], float), np.asarray(roc["tpr"], float)
            au = _get(summary, "methods", m, "auroc", neg) or {}
            ci = au.get("ci")
            lab = f"{method_label(m)} vs {neg}: AUROC {au.get('value', float('nan')):.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
            ax.plot(np.maximum(fpr, 1e-6), tpr, color=c, linestyle=ls, lw=1.5, label=lab)
            drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("FPR (лог. шкала)")
    ax.set_ylabel("TPR")
    ax.set_title("ROC: члены против hard_neg (сплошные) и public_test (пунктир)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    return _save(fig, path)


def fig_F4(summary: dict[str, Any], path: Path) -> Path | None:
    """F4: заявленный α vs эмпирический FPR (ДИ Клоппера–Пирсона): круги — public_test, треугольники — hard_neg,
    закрашенные — после доменной калибровки (hard_neg, тестовые репозитории); диагональ — точное выполнение гарантии."""
    methods = order_methods(list(summary.get("methods", {})))
    aks = summary.get("alpha_keys") or []
    if not methods or not aks:
        return None
    styles = style_for(methods)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    drawn = False
    n = len(methods)
    for i, m in enumerate(methods):
        c, _ = styles[m]
        jitter = 10 ** ((i - (n - 1) / 2) * 0.06)
        for neg, mk, kind, filled in (("public_test", "o", "conformal", False), ("hard_neg", "^", "conformal", False), ("hard_neg_test", "^", "domain", True)):
            xs, ys, lo, hi = [], [], [], []
            for ak in aks:
                blk = _get(summary, "methods", m, "fpr", ak, kind, neg)
                if not blk or blk.get("value") is None:
                    continue
                v = max(float(blk["value"]), 1e-5)
                xs.append(float(ak) * jitter); ys.append(v)
                lo.append(max(0.0, v - max(float(blk["ci"][0]), 1e-5))); hi.append(max(0.0, float(blk["ci"][1]) - v))
            if not xs:
                continue
            drawn = True
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=c, mfc=c if filled else "white", mec=c, ms=5.5, capsize=1.5,
                        elinewidth=0.7, lw=0, label=f"{method_label(m)}, {neg}" + (" (доменная калибровка)" if filled else ""))
    if not drawn:
        plt.close(fig)
        return None
    lim = [min(1e-4, min(float(a) for a in aks) / 3), 1.0]
    ax.plot(lim, lim, color=MUTED, linestyle="--", lw=1.0, label="FPR = α")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(1e-5, 1.0)
    ax.set_xlabel("Заявленный уровень α")
    ax.set_ylabel("Эмпирический FPR (ДИ Клоппера–Пирсона)")
    ax.set_title("Конформная гарантия: заявленный α против эмпирического FPR")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=6.5)
    return _save(fig, path)


def fig_F5(summary: dict[str, Any], path: Path) -> Path | None:
    """F5: абляция гибрида — TPR по классам для вариантов hybrid* и лучшей компоненты (max по winnowing/semantic)."""
    all_methods = order_methods(list(summary.get("methods", {})))
    hybrids = [m for m in all_methods if m.startswith("hybrid")]
    if not hybrids:
        return None
    a0 = (summary.get("alpha_keys") or ["0.01"])[0]
    transforms = order_transforms(sorted({t for m in hybrids for t in _get(summary, "methods", m, "tpr", a0, "conformal", "by_transform", default={})}))
    if not transforms:
        return None
    comps = [m for m in ("winnowing", "semantic") if m in all_methods]
    series: dict[str, list[tuple[float, float, float]]] = {m: [_val_err(_get(summary, "methods", m, "tpr", a0, "conformal", "by_transform", t)) for t in transforms] for m in hybrids}
    labels = {m: method_label(m) for m in hybrids}
    if comps:
        best = []
        for t in transforms:
            cand = [(_val_err(_get(summary, "methods", c, "tpr", a0, "conformal", "by_transform", t)), c) for c in comps]
            cand = [x for x in cand if not np.isnan(x[0][0])]
            best.append(max(cand, key=lambda x: x[0][0])[0] if cand else (np.nan, 0.0, 0.0))
        series["best_component"] = best
        labels["best_component"] = "лучшая компонента (M1/M3)"
    styles = style_for(hybrids)
    styles["best_component"] = (OTHER, "o")
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(transforms) + 2), 3.4))
    _grouped_bars(ax, [transform_label(t) for t in transforms], series, styles, labels)
    ax.set_xlabel("Класс преобразования")
    ax.set_ylabel(f"TPR при FPR = {pct(a0)}")
    ax.set_title("Абляция гибрида M4")
    ax.legend(ncol=min(4, len(series)), loc="upper center", bbox_to_anchor=(0.5, -0.32))
    return _save(fig, path)


def _pareto(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Индексы фронта Парето: минимизируем утечку x, максимизируем полезность y."""
    order = np.argsort(xs)
    best, front = -np.inf, []
    for i in order:
        if ys[i] > best:
            front.append(i); best = ys[i]
    return np.asarray(front, dtype=int)


def fig_F6(privacy: dict[str, Any] | None, path: Path) -> Path | None:
    """F6: кривые приватность–полезность из results/privacy.json: x — F1 атаки A2 при n утёкших пар, y — TPR@1 % (M4, иначе M3)."""
    if not privacy:
        return None
    defs = privacy.get("defenses") or privacy.get("results") or []
    if isinstance(defs, dict):
        defs = [{"name": k, **v} for k, v in defs.items()]
    ns = [str(n) for n in (privacy.get("leaked_pairs") or [10000, 1000]) if str(n) != "0"]
    ns = sorted(ns, key=lambda s: -float(s))[:2]
    if not defs or not ns:
        return None

    def _tpr(d: dict[str, Any]) -> float:
        for k in ("hybrid", "semantic"):
            v = _get(d, "tpr", k)
            if isinstance(v, dict):
                v = v.get("value")
            if v is not None:
                return float(v)
        return np.nan

    fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 3.4), squeeze=False)
    drawn = False
    for ax, n in zip(axes[0], ns):
        xs = np.array([float(_get(d, "attacks", "A2", n, "f1", default=np.nan) or np.nan) for d in defs])
        ys = np.array([_tpr(d) for d in defs])
        ok = ~(np.isnan(xs) | np.isnan(ys))
        if not ok.any():
            ax.set_title(f"n = {n}: нет данных")
            continue
        drawn = True
        ax.scatter(xs[ok], ys[ok], s=28, color=PALETTE[0], edgecolor="white", linewidth=0.8, zorder=3)
        for j, (d, x, y, o) in enumerate(zip(defs, xs, ys, ok)):
            if o:  # подписи попеременно сверху/снизу, чтобы близкие точки не сливались
                dy = 4 if j % 2 == 0 else -9
                ax.annotate(str(d.get("name", "")), (x, y), xytext=(3, dy), textcoords="offset points", fontsize=6.5, color=MUTED)
        idx = np.flatnonzero(ok)[_pareto(xs[ok], ys[ok])]
        ax.plot(xs[idx], ys[idx], color=PALETTE[0], lw=1.2, linestyle="--", label="фронт Парето")
        ax.set_xlabel(f"Утечка: F1 атаки A2 при n = {n}")
        ax.set_ylabel("Полезность: TPR при FPR = 1 %")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)
        ax.set_title(f"n = {n} утёкших пар")
        ax.legend(loc="lower right")
    if not drawn:
        plt.close(fig)
        return None
    fig.suptitle("Приватность индекса: утечка против полезности для защит из T6", y=1.02)
    return _save(fig, path)


def fig_F7(summary: dict[str, Any], latency: dict[str, Any] | None, path: Path) -> Path | None:
    """F7: p95 латентности как функция размера индекса N (лог-лог): точки из results/latency.json[method].by_size
    (список {n, p95_ms}) или одна точка на метод из summary; пунктир — линейная экстраполяция; бюджеты чата/IDE."""
    methods = order_methods(list(summary.get("methods", {})))
    latency = latency or {}
    styles = style_for(methods)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    drawn = False
    for m in methods:
        pts = [(float(p["n"]), float(p["p95_ms"])) for p in (_get(latency, m, "by_size") or []) if p.get("n") and p.get("p95_ms") is not None]
        if not pts:
            n = _get(summary, "methods", m, "n_records")
            p95 = _get(summary, "methods", m, "latency_ms", "p95")
            if n and p95 is not None:
                pts = [(float(n), float(p95))]
        if not pts:
            continue
        drawn = True
        c, mk = styles[m]
        xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        ax.plot(xs, ys, marker=mk, color=c, lw=1.4, ms=5, label=method_label(m))
        if len(pts) >= 2:
            k = (ys[-1] - ys[0]) / max(xs[-1] - xs[0], 1e-9)
            xe = np.array([xs[-1], 1e6 if xs[-1] < 1e6 else xs[-1] * 2])
            ax.plot(xe, ys[-1] + k * (xe - xs[-1]), color=c, lw=1.0, linestyle="--")
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xscale("log"); ax.set_yscale("log")
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for y, lab in ((50, "бюджет чата 50 мс"), (20, "бюджет IDE 20 мс")):
        ax.axhline(y, color=MUTED, linestyle=":", lw=1.0)
        ax.text(0.99, y, lab, color=MUTED, fontsize=7, va="bottom", ha="right", transform=tr)
    ax.set_xlabel("Размер индекса N, записей")
    ax.set_ylabel("Латентность p95, мс")
    ax.set_title("Латентность против размера индекса (пунктир — линейная экстраполяция)")
    ax.legend(loc="upper left")
    return _save(fig, path)


# ----------------------------------------------------------------------------- запись


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        log.warning("corrupt json: %s", path)
        return None


def write_figures(cfg: dict[str, Any], summary: dict[str, Any]) -> dict[str, Path]:
    """Строит все рисунки в cfg.paths.figures; возвращает {F#: путь}; пропущенные логируются."""
    fdir = resolve_path(cfg, "figures")
    rdir = resolve_path(cfg, "results")
    privacy, latency = _read_json(rdir / PRIVACY_FILE), _read_json(rdir / LATENCY_FILE)
    builders: dict[str, Callable[[Path], Path | None]] = {
        "F1": lambda p: fig_F1(summary, p), "F2": lambda p: fig_F2(summary, p, cfg), "F3": lambda p: fig_F3(summary, p),
        "F4": lambda p: fig_F4(summary, p), "F5": lambda p: fig_F5(summary, p), "F6": lambda p: fig_F6(privacy, p),
        "F7": lambda p: fig_F7(summary, latency, p),
    }
    out: dict[str, Path] = {}
    for key, build in builders.items():
        path = fdir / FIGURE_FILES[key]
        try:
            res = build(path)
        except Exception as exc:  # noqa: BLE001 — один рисунок не должен ронять отчёт
            log.exception("%s: ошибка построения: %s", key, exc)
            plt.close("all")
            continue
        if res is None:
            log.info("%s: пропуск — нет входных данных (%s)", key, FIGURE_FILES[key])
            continue
        out[key] = res
        log.info("%s → %s", key, res)
    return out


__all__ = ["write_figures", "FIGURE_FILES", "fig_F1", "fig_F2", "fig_F3", "fig_F4", "fig_F5", "fig_F6", "fig_F7"]
