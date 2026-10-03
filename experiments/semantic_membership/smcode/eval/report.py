"""Шаг 09: metrics → results/summary.json → таблицы T1..T7 → рисунки F1..F7 + краткая сводка на русском.

summary.json пересчитывается, если: его нет / он без методов / force; методы или bootstrap заданы явно;
набор каталогов results/scores/* отличается от summary.methods; любой файл скоров (*.jsonl, *.meta.json) или файл
запросов data/queries/<set>.jsonl новее summary.json. Иначе готовый summary переиспользуется с предупреждением в лог.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from smcode.config import resolve_path
from smcode.eval.build_queries import all_sets
from smcode.eval.metrics import SUMMARY_FILE, available_methods, compute_summary, load_summary, write_summary
from smcode.eval.run_eval import is_limit_file, queries_path, scores_dir
from smcode.eval.tables import fmt, fmt_ci, fmt_ci_cluster, method_label, order_methods, pct, variant_description

log = logging.getLogger(__name__)


def summary_is_stale(cfg: dict[str, Any], summary: dict[str, Any] | None, spath: Path) -> str | None:
    """Причина пересчёта summary.json (None — актуален): нет методов, другой набор каталогов скоров, скоры или запросы
    новее summary."""
    if not summary or not summary.get("methods"):
        return "summary.json отсутствует или без методов"
    if not spath.exists():
        return "summary.json не найден"
    have = set(available_methods(cfg))
    got = set(summary["methods"])
    if have != got:
        return f"набор каталогов results/scores/* ({sorted(have)}) отличается от summary.methods ({sorted(got)})"
    mtime = spath.stat().st_mtime
    for m in have:
        for p in scores_dir(cfg, m).iterdir():
            if (p.suffix in (".jsonl", ".json")) and not is_limit_file(p) and p.stat().st_mtime > mtime:
                return f"файл скоров новее summary.json: {p}"
    for s in all_sets(cfg):
        q = queries_path(cfg, s)
        if q.exists() and q.stat().st_mtime > mtime:
            return f"файл запросов новее summary.json: {q}"
    return None


def privacy_consistency_warning(cfg: dict[str, Any], summary: dict[str, Any]) -> str | None:
    """Строка «none» results/privacy.json обязана воспроизводить TPR M3 основной оценки (тот же индекс и конформный порог);
    при непересекающихся интервалах — предупреждение (разный состав индекса или модель эмбеддингов)."""
    from smcode.eval.tables import PRIVACY_FILE, _read_json

    priv = _read_json(resolve_path(cfg, "results") / PRIVACY_FILE)
    main_m = (summary.get("methods") or {}).get("semantic") or {}
    if not priv or not main_m:
        return None
    none = next((d for d in priv.get("defenses") or [] if d.get("name") == "none"), None)
    if not none:
        return None
    ak = f"{float(priv.get('alpha', 0.01)):g}"
    main = ((main_m.get("tpr") or {}).get(ak) or {}).get("conformal", {}).get("overall")
    p = (none.get("tpr") or {}).get("semantic")
    if not main or not p or main.get("value") is None or p.get("value") is None:
        return None
    lo, hi = main.get("ci") or (main["value"], main["value"])
    plo, phi = p.get("ci_cluster") or p.get("ci") or (p["value"], p["value"])
    if hi < plo or phi < lo:
        return (f"privacy.json: TPR M3 без защиты ({p['value']:.3f}) не совпадает с TPR M3 основной оценки ({main['value']:.3f}, α = {ak}): "
                f"разный состав индекса ({priv.get('index_composition')}) или модель ({priv.get('model')})? Перезапустите "
                "scripts/08_privacy.py --force с теми же эмбеддингами и индексом (with_windows), что у шага 04")
    return None


def run_report(cfg: dict[str, Any], force: bool = False, bootstrap: int | None = None, methods: Iterable[str] | None = None,
               tables: bool = True, figures: bool = True) -> dict[str, Any]:
    """Полный отчёт. summary.json пересчитывается при force, явных methods/bootstrap или устаревании (summary_is_stale);
    таблицы и рисунки строятся всегда (детерминированы по summary и дёшевы).
    Возвращает {"summary", "summary_path", "tables", "figures", "recomputed", "reason"}."""
    rdir = resolve_path(cfg, "results")
    spath = rdir / SUMMARY_FILE
    methods = list(methods) if methods else None
    summary = load_summary(cfg)
    reason: str | None
    if force:
        reason = "--force"
    elif methods is not None or bootstrap is not None:
        reason = "методы/бутстрэп заданы явно"
    else:
        reason = summary_is_stale(cfg, summary, spath)
    recomputed = reason is not None
    if recomputed:
        log.info("пересчёт summary.json: %s", reason)
        summary = compute_summary(cfg, methods=methods, bootstrap=bootstrap)
        pw = privacy_consistency_warning(cfg, summary)
        if pw:
            log.warning(pw)
            summary.setdefault("warnings", []).append(pw)
        spath = write_summary(cfg, summary)
    else:
        log.warning("summary.json найден и актуален (%s) — метрики не пересчитываются (--force для пересчёта)", spath)
    assert summary is not None
    out: dict[str, Any] = {"summary": summary, "summary_path": spath, "tables": {}, "figures": {}, "recomputed": recomputed, "reason": reason}
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
    for w in summary.get("warnings") or []:
        lines.append(f"  ⚠ {w}")
    if not methods:
        lines.append("  Методов со скорами не найдено (results/scores/<method>/*.jsonl).")
    for m in methods:
        M = summary["methods"][m]
        lat = M.get("latency_ms", {}) or {}
        e2e = lat.get("end_to_end") or {}
        e2e_s = (f", сквозная p95 ≈ {fmt(e2e.get('p95'), 2)} мс" if e2e.get("p95") is not None
                 else (f", сквозная латентность: {e2e['reason']}" if e2e.get("reason") else ""))
        desc = variant_description(summary, m)
        lines.append(f"- {method_label(m)}{f' ({desc})' if desc else ''}: записей в индексе {M.get('n_records') if M.get('n_records') is not None else '—'}, "
                     f"память {fmt((M.get('memory_bytes') or 0) / 1e6, 1)} МБ, латентность p50/p95 "
                     f"{fmt(lat.get('p50'), 2)}/{fmt(lat.get('p95'), 2)} мс ({lat.get('mode') or '—'}){e2e_s}")
        for ak in aks:
            th = (M.get("thresholds") or {}).get(ak, {})
            tpr = ((M.get("tpr") or {}).get(ak) or {}).get("conformal", {}).get("overall")
            fpr = ((M.get("fpr") or {}).get(ak) or {}).get("conformal", {})
            lines.append(f"    α = {pct(ak)}: τ = {fmt(th.get('conformal'))}, TPR = {fmt_ci(tpr)}, "
                         f"FPR public_test = {fmt_ci_cluster(fpr.get('public_test'))}, FPR hard_neg = {fmt_ci_cluster(fpr.get('hard_neg'))}")
        au = M.get("auroc") or {}
        if au:
            lines.append(f"    AUROC: hard_neg {fmt_ci(au.get('hard_neg'))}, public_test {fmt_ci(au.get('public_test'))}")
    if tables:
        lines.append("Таблицы: " + ", ".join(f"{k} → {v}" for k, v in tables.items()))
    if figures:
        lines.append("Рисунки: " + ", ".join(f"{k} → {v}" for k, v in figures.items()))
    return "\n".join(lines)
