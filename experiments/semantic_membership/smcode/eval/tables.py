"""Markdown-таблицы T1..T7 (DESIGN.md §12) из results/summary.json → results/tables/*.md.

T1 статистика данных (data_stats), T2 TPR по преобразованию × метод, T3 FPR по наборам негативов с ДИ и
гарантированным α (+ доменная калибровка, оракульный порог), T4 абляции (все методы/варианты из summary),
T5 латентность/память (+ results/latency.json, если есть), T6 приватность (results/privacy.json, если есть),
T7 LLM-преобразования (если в запросах есть paraphrase/translate). Числа — 3 знака, ДИ в квадратных скобках.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

from smcode.config import resolve_path
from smcode.fingerprint.index import REGISTRY

log = logging.getLogger(__name__)

TABLES_DIR = "tables"
PRIVACY_FILE = "privacy.json"
LATENCY_FILE = "latency.json"                   # необязательный сводный файл: {method: {cpu1, cpu4, gpu_b1, gpu_b32, components, by_size}}
LATENCY_ENCODER_FILE = "latency_encoder.json"   # scripts/06_embed.py: {"configs": [{"device","threads","batch","p50_ms","p95_ms","per_item_ms"}], "cpu_threads_all": N}
ENCODER_METHODS: tuple[str, ...] = ("semantic", "hybrid")
DASH = "—"

METHOD_LABELS: dict[str, str] = {
    "exact": "M0 exact", "winnowing": "M1 winnowing", "minhash": "M2 MinHash", "semantic": "M3 semantic", "hybrid": "M4 hybrid",
    "winnowing_nofilter": "M1 без фильтра общности", "semantic_zeroshot": "M3 zero-shot", "semantic_finetuned": "M3 дообученный",
    "semantic_adapt_protected": "M3 adapt_on_protected", "semantic_own_small": "M3 own_small_model",
    "hybrid_rule": "M4 правило двух порогов", "hybrid_no_cos": "M4 без cos", "hybrid_no_overlap": "M4 без ovl_w",
    "hybrid_no_lcs": "M4 без lcr_norm", "hybrid_no_ntokens": "M4 без log(1+m_q)",
}
TRANSFORM_ORDER: tuple[str, ...] = ("identity", "reformat", "strip_comments", "rename_ids", "change_literals", "insert_deadcode",
                                    "reorder_stmts", "partial", "combo", "paraphrase", "translate")
TRANSFORM_LABELS: dict[str, str] = {
    "identity": "T0 identity", "reformat": "T1 reformat", "strip_comments": "T2 strip_comments", "rename_ids": "T3 rename_ids",
    "change_literals": "T4 change_literals", "insert_deadcode": "T5 insert_deadcode", "reorder_stmts": "T6 reorder_stmts",
    "partial": "T7 partial (все L)", "combo": "T8 combo", "paraphrase": "T9 paraphrase", "translate": "T9 translate",
}
SPLIT_LABELS = {"protected": "protected (P)", "hard_neg": "hard_neg", "public_train": "public_train",
                "public_calib": "public_calib", "public_test": "public_test"}


# ----------------------------------------------------------------------------- форматирование


def fmt(v: Any, nd: int = 3) -> str:
    """Число с nd знаками; None/'inf' → '—'/'∞'."""
    if v is None:
        return DASH
    if isinstance(v, str):
        return "∞" if v == "inf" else v
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return DASH
    if f in (float("inf"), float("-inf")):
        return "∞" if f > 0 else "−∞"
    return f"{f:.{nd}f}"


def fmt_ci(block: dict[str, Any] | None, nd: int = 3) -> str:
    """'0.912 [0.901, 0.923]' из блока {'value', 'ci'}; '—' если нет."""
    if not block or block.get("value") is None:
        return DASH
    s = fmt(block["value"], nd)
    ci = block.get("ci")
    if ci and ci[0] is not None:
        s += f" [{fmt(ci[0], nd)}, {fmt(ci[1], nd)}]"
    return s


def pct(a: float | str) -> str:
    """α как проценты: 0.01 → '1 %', 0.001 → '0,1 %'."""
    v = float(a) * 100
    s = f"{v:g}".replace(".", ",")
    return f"{s} %"


def method_label(m: str) -> str:
    return METHOD_LABELS.get(m, m)


def transform_label(t: str) -> str:
    return TRANSFORM_LABELS.get(t, t)


def order_methods(names: Sequence[str]) -> list[str]:
    known = [m for m in REGISTRY if m in names]
    return known + sorted(m for m in names if m not in REGISTRY)


def order_transforms(names: Sequence[str]) -> list[str]:
    known = [t for t in TRANSFORM_ORDER if t in names]
    return known + sorted(t for t in names if t not in TRANSFORM_ORDER)


def md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _get(d: Any, *keys: Any, default: Any = None) -> Any:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# ----------------------------------------------------------------------------- таблицы


def table_T1(summary: dict[str, Any]) -> str:
    """T1: статистика данных после дедупликации (из data/functions/stats.json) + число запросов по наборам."""
    ds = summary.get("data_stats") or {}
    fn = ds.get("functions") or {}
    splits = fn.get("splits") or {}
    bins = [f"{lo}-{hi}" for lo, hi in (fn.get("length_bins_tokens") or summary.get("length_bins") or [])]
    langs = sorted({l for s in splits.values() for l in (s.get("by_lang") or {})})
    header = ["Сплит", "Репозиториев", "Функций", "Окон", "Удалено точных дублей", "Удалено near-dup", *langs, *bins, "Запросов"]
    rows: list[list[Any]] = []
    windows = fn.get("windows") or {}
    queries = ds.get("queries") or {}
    tot = {"repos": 0, "n": 0, "ex": 0, "nd": 0, "q": 0}
    for name in ("protected", "hard_neg", "public_train", "public_calib", "public_test"):
        s = splits.get(name)
        if s is None:
            continue
        n_win = windows.get("n_final") if name == "protected" else None
        q = queries.get(name, {}).get("n")
        rows.append([SPLIT_LABELS.get(name, name), s.get("n_repos", DASH), s.get("n_final", DASH), n_win if n_win is not None else DASH,
                     s.get("removed_exact", DASH), s.get("removed_near_dup", DASH) if name != "protected" else DASH,
                     *[(s.get("by_lang") or {}).get(l, 0) for l in langs], *[(s.get("by_length_bin") or {}).get(b, 0) for b in bins],
                     q if q is not None else DASH])
        tot["repos"] += int(s.get("n_repos") or 0); tot["n"] += int(s.get("n_final") or 0)
        tot["ex"] += int(s.get("removed_exact") or 0); tot["nd"] += int(s.get("removed_near_dup") or 0); tot["q"] += int(q or 0)
    if rows:
        rows.append(["Всего", tot["repos"], tot["n"], windows.get("n_final", DASH), tot["ex"], tot["nd"], *[""] * (len(langs) + len(bins)), tot["q"]])
    else:
        rows = [["(нет data/functions/stats.json)"] + [DASH] * (len(header) - 1)]
    out = "Таблица T1 — Статистика данных стенда после дедупликации (источник: `results/summary.json: data_stats`).\n\n" + md_table(header, rows)
    if queries:
        qrows = [[s, q.get("n", DASH), q.get("label", DASH), ", ".join(f"{t}: {n}" for t, n in (q.get("by_transform") or {}).items())]
                 for s, q in queries.items()]
        out += "\nЗапросы по наборам (data/queries/*.jsonl):\n\n" + md_table(["Набор", "Запросов", "label", "По преобразованиям"], qrows)
    return out


def table_T2(summary: dict[str, Any]) -> str:
    """T2: TPR при конформном пороге α₀ по классам преобразований × метод (+ итоги по всем α)."""
    methods = order_methods(list(summary.get("methods", {})))
    aks = summary.get("alpha_keys") or ["0.01"]
    a0 = aks[0]
    transforms = order_transforms(sorted({t for m in methods for t in _get(summary, "methods", m, "tpr", a0, "conformal", "by_transform", default={})}))
    header = ["Класс", *[method_label(m) for m in methods]]
    rows = [[transform_label(t), *[fmt_ci(_get(summary, "methods", m, "tpr", a0, "conformal", "by_transform", t)) for m in methods]] for t in transforms]
    for ak in aks:
        rows.append([f"Все классы, α = {pct(ak)}", *[fmt_ci(_get(summary, "methods", m, "tpr", ak, "conformal", "overall")) for m in methods]])
    rows.append([f"Все классы, оракульный порог, α = {pct(a0)}", *[fmt_ci(_get(summary, "methods", m, "tpr", a0, "oracle", "overall")) for m in methods]])
    langs = sorted({l for m in methods for l in _get(summary, "methods", m, "tpr", a0, "conformal", "by_lang", default={})})
    for lang in langs:  # блок «по языкам» (статья, табл. 3)
        rows.append([f"По языкам, α = {pct(a0)}: {lang}", *[fmt_ci(_get(summary, "methods", m, "tpr", a0, "conformal", "by_lang", lang)) for m in methods]])
    return (f"Таблица T2 — TPR при FPR = {pct(a0)} (конформный порог τ по public_calib) по классам преобразований и языкам; "
            "в скобках — 95 % ДИ бутстрэпа с кластеризацией по исходной функции.\n\n" + md_table(header, rows))


def table_T3(summary: dict[str, Any]) -> str:
    """T3: FPR при конформном пороге по наборам негативов (ДИ Клоппера–Пирсона), доменная калибровка, оракул."""
    methods = order_methods(list(summary.get("methods", {})))
    aks = summary.get("alpha_keys") or ["0.01"]
    header = ["Метод", "α", "τ_α", "FPR public_test [ДИ]", "FPR hard_neg [ДИ]", "Макс. FPR по подгруппам (язык / класс T)",
              "τ_α при доменной калибровке", "FPR hard_neg (тестовые репозитории) при доменной калибровке [ДИ]",
              "TPR при доменной калибровке", "TPR при оракульном пороге", "FPR public_test > α"]
    rows = []
    for ak in aks:
        for m in methods:
            M = summary["methods"][m]
            th = _get(M, "thresholds", ak, default={}) or {}
            fpr_c = _get(M, "fpr", ak, "conformal", default={}) or {}
            fpr_d = _get(M, "fpr", ak, "domain", default={}) or {}
            ms = _get(fpr_c, "public_test", "max_subgroup") or _get(fpr_c, "hard_neg", "max_subgroup")
            pt = fpr_c.get("public_test")
            if not pt or pt.get("value") is None:
                viol = DASH
            elif pt["value"] <= float(ak):
                viol = "нет"
            else:  # точечная оценка выше α: значимо, только если нижняя граница ДИ выше α
                viol = "да (ДИ выше α)" if pt["ci"][0] > float(ak) else "в пределах ДИ"
            rows.append([method_label(m), pct(ak), fmt(th.get("conformal")), fmt_ci(pt), fmt_ci(fpr_c.get("hard_neg")),
                         f"{fmt(ms['value'])} ({ms['group']}, n={ms['n']})" if ms else DASH, fmt(th.get("domain")),
                         fmt_ci(fpr_d.get("hard_neg_test")), fmt_ci(_get(M, "tpr", ak, "domain", "overall")),
                         fmt_ci(_get(M, "tpr", ak, "oracle", "overall")), viol])
    dc = summary.get("domain_calibration") or {}
    note = (f"Доменная калибровка: {len(dc.get('calib_repos', []))} репозиториев hard_neg в калибровке, "
            f"{len(dc.get('test_repos', []))} — в тесте (доля {dc.get('fraction', DASH)}).")
    return ("Таблица T3 — Эмпирический FPR при конформном пороге τ_α по наборам негативов (95 % ДИ Клоппера–Пирсона), "
            "эффект доменной калибровки и оракульный порог. Гарантия: FPR ≤ α при обменяемости негативов.\n\n"
            + md_table(header, rows) + "\n" + note + "\n")


def table_T4(summary: dict[str, Any]) -> str:
    """T4: абляции — все методы/варианты из summary: TPR при α₀ и α₁, AUROC члены/hard_neg, FPR hard_neg при τ_α₀."""
    methods = order_methods(list(summary.get("methods", {})))
    aks = summary.get("alpha_keys") or ["0.01"]
    a0 = aks[0]
    header = ["Вариант", *[f"TPR@{pct(ak)}" for ak in aks], "AUROC члены/hard_neg", "AUROC члены/public_test", f"FPR hard_neg при τ_{pct(a0)}"]
    rows = [[method_label(m), *[fmt_ci(_get(summary, "methods", m, "tpr", ak, "conformal", "overall")) for ak in aks],
             fmt_ci(_get(summary, "methods", m, "auroc", "hard_neg")), fmt_ci(_get(summary, "methods", m, "auroc", "public_test")),
             fmt_ci(_get(summary, "methods", m, "fpr", a0, "conformal", "hard_neg"))] for m in methods]
    return ("Таблица T4 — Абляции: TPR при конформном пороге, AUROC и FPR на hard_neg. Варианты методов — каталоги "
            "results/scores/<вариант>/ (semantic_zeroshot, hybrid_rule, hybrid_no_* и т. д.).\n\n" + md_table(header, rows))


def _lat_pair(d: Any) -> str:
    if not isinstance(d, dict) or d.get("p50_ms") is None:
        return DASH
    p95 = d.get("p95_ms")
    return f"{fmt(float(d['p50_ms']), 2)} / {fmt(None if p95 is None else float(p95), 2)}"


def encoder_latency(report: dict[str, Any] | None) -> dict[str, Any]:
    """results/latency_encoder.json (scripts/06_embed.py) → {cpu1, cpu4, gpu_b1, gpu_b32: {p50_ms, p95_ms, qps}, threads_all}."""
    if not report or not report.get("configs"):
        return {}
    n_all = report.get("cpu_threads_all")
    out: dict[str, Any] = {"threads_all": n_all}
    for c in report["configs"]:
        dev, thr, bs = c.get("device"), c.get("threads"), int(c.get("batch", 1))
        stats = {"p50_ms": c.get("p50_ms"), "p95_ms": c.get("p95_ms"), "per_item_ms": c.get("per_item_ms"),
                 "qps": (1000.0 * bs / c["p50_ms"]) if c.get("p50_ms") else None}
        if dev == "cpu" and bs == 1:
            out["cpu1" if thr == 1 else "cpu4"] = stats
        elif dev != "cpu" and bs == 1:
            out["gpu_b1"] = stats
        elif dev != "cpu" and bs > 1:
            out["gpu_b32"] = stats
    return out


def table_T5(summary: dict[str, Any], latency: dict[str, Any] | None = None, encoder: dict[str, Any] | None = None) -> str:
    """T5: память индекса, байт/запись, время сборки, латентность p50/p95 (прогон eval; results/latency.json;
    энкодер — results/latency_encoder.json для semantic*/hybrid*)."""
    methods = order_methods(list(summary.get("methods", {})))
    latency = latency or {}
    enc = encoder_latency(encoder)
    n_all = enc.get("threads_all") or 4
    header = ["Метод", "Записей", "Память индекса, МБ", "Байт на запись", "Построение индекса, мин",
              "Прогон eval, p50 / p95, мс (режим)", "CPU, 1 поток, p50 / p95", f"CPU, {n_all} потоков, p50 / p95", "GPU, батч 1, p50 / p95", "GPU, батч 32, запросов/с"]
    rows = []
    for m in methods:
        M = summary["methods"][m]
        lat = M.get("latency_ms") or {}
        ext = dict(latency.get(m) or {})
        if enc and m.startswith(ENCODER_METHODS):
            comps = dict(ext.get("components") or {})
            comps.setdefault("encoder", {k: enc[k] for k in ("cpu1", "cpu4", "gpu_b1", "gpu_b32") if k in enc})
            ext["components"] = comps
        mem = M.get("memory_bytes")
        bs = M.get("build_seconds")
        rows.append([method_label(m), M.get("n_records", DASH), fmt(mem / 1e6 if mem is not None else None, 1),
                     fmt(M.get("bytes_per_record"), 1), fmt(bs / 60 if bs is not None else None, 2),
                     f"{fmt(lat.get('p50'), 2)} / {fmt(lat.get('p95'), 2)} ({lat.get('mode') or DASH})",
                     _lat_pair(ext.get("cpu1")), _lat_pair(ext.get("cpu4")), _lat_pair(ext.get("gpu_b1")), fmt(_get(ext, "gpu_b32", "qps"), 1)])
        comps = ext.get("components") or {}
        for cname, clabel in (("encoder", "— из них инференс энкодера"), ("ann", "— из них поиск ANN"), ("rerank", "— из них переранжирование")):
            c = comps.get(cname)
            if c:
                rows.append([clabel, DASH, DASH, DASH, DASH, DASH, _lat_pair(c.get("cpu1")), _lat_pair(c.get("cpu4")), _lat_pair(c.get("gpu_b1")), fmt(_get(c, "gpu_b32", "qps"), 1)])
        b = lat.get("benchmark")
        if b:
            rows.append([f"— бенчмарк батч 1 ({b.get('repeats')} повторов × {b.get('n_queries')} запросов, {b.get('mode')})", DASH, DASH, DASH, DASH,
                         f"{fmt(b.get('p50_ms'), 2)} / {fmt(b.get('p95_ms'), 2)}", DASH, DASH, DASH, DASH])
    return ("Таблица T5 — Латентность одного запроса (p50 / p95, мс) и память индекса. Колонки CPU/GPU — из results/latency.json "
            "(если файл есть); строка «инференс энкодера» — из results/latency_encoder.json (scripts/06_embed.py); "
            "«прогон eval» — wall-clock в scripts/07_eval.py (для numpy-пути semantic/hybrid — амортизированное время "
            "батча без энкодера, т. е. только ANN/переранжирование).\n\n" + md_table(header, rows))


def table_T6(summary: dict[str, Any], privacy: dict[str, Any] | None) -> str | None:
    """T6: защиты индекса эмбеддингов из results/privacy.json (None, если файла нет).

    Ожидаемая схема: {"defenses": [{"name": str, "memory_ratio": float, "tpr": {"semantic": RATE|float, "hybrid": ...},
    "attacks": {"A1": {"f1": x, "rare_id_recall": y}, "A2": {"100": {"f1": x}, ...}, "A3": {"bleu": x, "id_acc": y}}}]}.
    """
    if not privacy:
        return None
    defs = privacy.get("defenses") or privacy.get("results") or []
    if isinstance(defs, dict):
        defs = [{"name": k, **v} for k, v in defs.items()]
    ns = [str(n) for n in (privacy.get("leaked_pairs") or [100, 1000, 10000]) if str(n) != "0"]
    header = ["Защита", "Память относительно float16", "TPR@1 % M3", "TPR@1 % M4", "A1: F1 по токенам", "A1: доля редких идентификаторов",
              *[f"A2, n = {n}: F1" for n in ns], "A3: BLEU / точность идентификаторов"]

    def _rate(v: Any) -> str:
        return fmt_ci(v) if isinstance(v, dict) else fmt(v)

    rows = []
    for d in defs:
        at = d.get("attacks") or {}
        a1, a2, a3 = at.get("A1") or {}, at.get("A2") or {}, at.get("A3") or {}
        rows.append([d.get("name", DASH), fmt(d.get("memory_ratio"), 3), _rate(_get(d, "tpr", "semantic")), _rate(_get(d, "tpr", "hybrid")),
                     fmt(a1.get("f1")), fmt(a1.get("rare_id_recall")), *[fmt(_get(a2, n, "f1")) for n in ns],
                     f"{fmt(a3.get('bleu'))} / {fmt(a3.get('id_acc'))}" if a3 else DASH])
    return ("Таблица T6 — Защиты индекса эмбеддингов: полезность (TPR при FPR = 1 %, конформный порог под той же защитой) "
            "и утечка (атаки A1–A3). Источник: results/privacy.json.\n\n" + md_table(header, rows))


def _llm_acceptance(llm_stats: dict[str, Any] | None, transform: str) -> float | None:
    """Доля принятых генераций из <set>.llm_stats.json (ключи вида '<t>_ok'/'<t>_failed' или {t: {ok, failed}})."""
    if not llm_stats:
        return None
    v = llm_stats.get(transform)
    if isinstance(v, dict) and "ok" in v:
        ok, bad = float(v.get("ok", 0)), float(v.get("failed", v.get("fail", 0)))
        return ok / (ok + bad) if ok + bad > 0 else None
    ok = bad = None
    for k, val in llm_stats.items():
        if not isinstance(val, (int, float)) or not k.startswith(transform):
            continue
        tail = k[len(transform):].lstrip("_:.")
        if tail in ("ok", "accepted"):
            ok = float(val)
        elif tail in ("failed", "fail", "rejected"):
            bad = float(val)
    if ok is None:
        return None
    return ok / (ok + (bad or 0.0)) if ok + (bad or 0.0) > 0 else None


def table_T7(summary: dict[str, Any]) -> str | None:
    """T7: LLM-преобразования (paraphrase, translate по целевому языку): доля принятых, число запросов, TPR, FPR hard_neg."""
    methods = order_methods(list(summary.get("methods", {})))
    aks = summary.get("alpha_keys") or ["0.01"]
    a0 = aks[0]
    has = any(t in _get(summary, "methods", m, "tpr", a0, "conformal", "by_transform", default={}) for m in methods for t in ("paraphrase", "translate"))
    if not has:
        return None
    qs = _get(summary, "data_stats", "queries", default={}) or {}
    header = ["Преобразование", "Целевой язык", "Принято, %", "Запросов (члены / негативы)", *[f"TPR {method_label(m)}" for m in methods],
              *[f"FPR hard_neg при τ, {method_label(m)}" for m in methods]]
    rows = []

    def _n(set_name: str, t: str) -> int:
        return int(_get(qs, set_name, "by_transform", t, default=0) or 0)

    def _row(t: str, dst: str | None) -> list[Any]:
        acc = _llm_acceptance(_get(qs, "protected", "llm_stats"), t)
        n_pos = _n("protected", t)
        n_neg = sum(_n(s, t) for s in ("public_calib", "public_test", "hard_neg"))
        tprs = [fmt_ci(_get(summary, "methods", m, "tpr", a0, "conformal", "by_translate_dst", dst) if dst else
                       _get(summary, "methods", m, "tpr", a0, "conformal", "by_transform", t)) for m in methods]
        fprs = [fmt_ci(_get(summary, "methods", m, "fpr", a0, "conformal", "hard_neg", "by_transform", t)) for m in methods]
        return [transform_label(t), dst or "= исходный", fmt(acc * 100 if acc is not None else None, 1) if dst is None else DASH,
                f"{n_pos} / {n_neg}" if dst is None else DASH, *tprs, *fprs]

    if any("paraphrase" in _get(summary, "methods", m, "tpr", a0, "conformal", "by_transform", default={}) for m in methods):
        rows.append(_row("paraphrase", None))
    dsts = sorted({d for m in methods for d in _get(summary, "methods", m, "tpr", a0, "conformal", "by_translate_dst", default={})})
    for d in dsts:
        rows.append(_row("translate", d))
    if any("translate" in _get(summary, "methods", m, "tpr", a0, "conformal", "by_transform", default={}) for m in methods):
        r = _row("translate", None)
        r[1] = "все"
        rows.append(r)
    return (f"Таблица T7 — LLM-преобразования: доля принятых генераций, число запросов и TPR при FPR = {pct(a0)} "
            "(конформный порог по public_calib с теми же преобразованиями); перевод — по целевому языку.\n\n" + md_table(header, rows))


# ----------------------------------------------------------------------------- запись


TABLE_FILES: dict[str, str] = {
    "T1": "T1_data_stats.md", "T2": "T2_tpr_by_transform.md", "T3": "T3_fpr_guarantee.md", "T4": "T4_ablation.md",
    "T5": "T5_latency_memory.md", "T6": "T6_privacy.md", "T7": "T7_llm.md",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        log.warning("corrupt json: %s", path)
        return None


def write_tables(cfg: dict[str, Any], summary: dict[str, Any]) -> dict[str, Path]:
    """Пишет все таблицы в results/tables/; возвращает {T#: путь}. T6/T7 пропускаются без входных данных."""
    rdir = resolve_path(cfg, "results")
    tdir = rdir / TABLES_DIR
    tdir.mkdir(parents=True, exist_ok=True)
    privacy = _read_json(rdir / PRIVACY_FILE)
    latency = _read_json(rdir / LATENCY_FILE)
    encoder = _read_json(rdir / LATENCY_ENCODER_FILE)
    builders: dict[str, Callable[[], str | None]] = {
        "T1": lambda: table_T1(summary), "T2": lambda: table_T2(summary), "T3": lambda: table_T3(summary), "T4": lambda: table_T4(summary),
        "T5": lambda: table_T5(summary, latency, encoder), "T6": lambda: table_T6(summary, privacy), "T7": lambda: table_T7(summary),
    }
    out: dict[str, Path] = {}
    for key, build in builders.items():
        try:
            text = build()
        except Exception as exc:  # noqa: BLE001 — одна таблица не должна ронять отчёт
            log.exception("%s: ошибка построения: %s", key, exc)
            continue
        if text is None:
            log.info("%s: пропуск (нет входных данных)", key)
            continue
        path = tdir / TABLE_FILES[key]
        path.write_text(text, encoding="utf-8")
        out[key] = path
        log.info("%s → %s", key, path)
    return out
