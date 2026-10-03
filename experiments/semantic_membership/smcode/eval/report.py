"""Шаг 09: metrics → results/summary.json → таблицы T1..T7 → рисунки F1..F7 + краткая сводка на русском."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from smcode.config import resolve_path
from smcode.eval.metrics import SUMMARY_FILE, compute_summary, load_summary, write_summary
from smcode.eval.tables import fmt, fmt_ci, method_label, order_methods, pct

log = logging.getLogger(__name__)


def run_report(cfg: dict[str, Any], force: bool = False, bootstrap: int | None = None, methods: Iterable[str] | None = None,
               tables: bool = True, figures: bool = True) -> dict[str, Any]:
    """Полный отчёт. summary.json пересчитывается, если его нет или force; таблицы и рисунки строятся всегда
    (детерминированы по summary и дёшевы). Возвращает {"summary", "summary_path", "tables", "figures"}."""
    rdir = resolve_path(cfg, "results")
    spath = rdir / SUMMARY_FILE
    summary = None if force else load_summary(cfg)
    if summary is not None and not summary.get("methods"):
        log.info("summary.json без методов — пересчёт")
        summary = None
    if summary is None:
        summary = compute_summary(cfg, methods=methods, bootstrap=bootstrap)
        spath = write_summary(cfg, summary)
    else:
        log.info("summary.json найден (%s) — метрики не пересчитываются (используйте --force)", spath)
    out: dict[str, Any] = {"summary": summary, "summary_path": spath, "tables": {}, "figures": {}}
    if tables:
        from smcode.eval.tables import write_tables

        out["tables"] = write_tables(cfg, summary)
    if figures:
        from smcode.eval.plots import write_figures

        out["figures"] = write_figures(cfg, summary)
    return out


def format_summary(summary: dict[str, Any], tables: dict[str, Path] | None = None, figures: dict[str, Path] | None = None) -> str:
    """Краткая сводка результатов на русском для stdout."""
    methods = order_methods(list(summary.get("methods", {})))
    aks = summary.get("alpha_keys") or []
    lines = [f"Сводка оценки ({summary.get('generated_at')}, бутстрэп {summary.get('bootstrap')}, правило: {summary.get('decision_rule')})"]
    if not methods:
        lines.append("  Методов со скорами не найдено (results/scores/<method>/*.jsonl).")
    for m in methods:
        M = summary["methods"][m]
        lines.append(f"- {method_label(m)}: записей в индексе {M.get('n_records') if M.get('n_records') is not None else '—'}, "
                     f"память {fmt((M.get('memory_bytes') or 0) / 1e6, 1)} МБ, латентность p50/p95 "
                     f"{fmt(M.get('latency_ms', {}).get('p50'), 2)}/{fmt(M.get('latency_ms', {}).get('p95'), 2)} мс")
        for ak in aks:
            th = (M.get("thresholds") or {}).get(ak, {})
            tpr = ((M.get("tpr") or {}).get(ak) or {}).get("conformal", {}).get("overall")
            fpr = ((M.get("fpr") or {}).get(ak) or {}).get("conformal", {})
            lines.append(f"    α = {pct(ak)}: τ = {fmt(th.get('conformal'))}, TPR = {fmt_ci(tpr)}, "
                         f"FPR public_test = {fmt_ci(fpr.get('public_test'))}, FPR hard_neg = {fmt_ci(fpr.get('hard_neg'))}")
        au = M.get("auroc") or {}
        if au:
            lines.append(f"    AUROC: hard_neg {fmt_ci(au.get('hard_neg'))}, public_test {fmt_ci(au.get('public_test'))}")
    if tables:
        lines.append("Таблицы: " + ", ".join(f"{k} → {v}" for k, v in tables.items()))
    if figures:
        lines.append("Рисунки: " + ", ".join(f"{k} → {v}" for k, v in figures.items()))
    return "\n".join(lines)
