"""Markdown-таблицы T1..T7 (DESIGN.md §12) из results/summary.json → results/tables/*.md.

T1 статистика данных (data_stats), T2 TPR по преобразованию × метод, T3 FPR по наборам негативов (кластерный ДИ по
функциям — основной, Клоппер–Пирсон — справочный) с гарантированным α (+ калибровка «1 запрос/функция», доменная
калибровка, оракульный порог), T4 абляции (все методы/варианты из summary, с описанием варианта), T5 латентность/память
(+ сквозная оценка latency_ms.end_to_end; results/latency.json и results/latency_encoder.json, если есть),
T6 приватность (results/privacy.json, схема smcode.privacy.run), T7 LLM-преобразования (если в запросах есть
paraphrase/translate). Числа — 3 знака, ДИ в квадратных скобках.
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
    "winnowing_nofilter": "M1 без фильтра общности", "winnowing_lexical": "M1 lexical (Moss по лексическим токенам)",
    "semantic_zeroshot": "M3 zero-shot", "semantic_finetuned": "M3 дообученный",
    "semantic_adapt_protected": "M3 adapt_on_protected (адаптация на protected)",
    "semantic_noevasion": "M3 без insert_deadcode/reorder_stmts в обучении", "semantic_own_small": "M3 own_small_model",
    "hybrid_rule": "M4 правило двух порогов", "hybrid_no_cos": "M4 без cos", "hybrid_no_overlap": "M4 без ovl_w",
    "hybrid_no_lcs": "M4 без lcr_norm", "hybrid_no_ntokens": "M4 без log(1+m_q)",
}
REASON_LABELS: dict[str, str] = {
    "too_short": "короче k", "all_common": "все отпечатки общие", "unsupported_lang": "язык не поддержан",
    "tokenize_error": "ошибка разбора", "no_hit": "нет совпадений (score 0)", "below_tau": "score ≤ τ",
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


def fmt_ci_cluster(block: dict[str, Any] | None, nd: int = 3) -> str:
    """Как fmt_ci, но с кластерным ДИ (ci_cluster); при его отсутствии — обычный ci."""
    if not block or block.get("value") is None:
        return DASH
    ci = block.get("ci_cluster") or block.get("ci")
    s = fmt(block["value"], nd)
    if ci and ci[0] is not None:
        s += f" [{fmt(ci[0], nd)}, {fmt(ci[1], nd)}]"
    return s


def fmt_interval(ci: Sequence[float] | None, nd: int = 3) -> str:
    """'[lo, hi]' или '—'."""
    if not ci or ci[0] is None:
        return DASH
    return f"[{fmt(ci[0], nd)}, {fmt(ci[1], nd)}]"


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


def variant_description(summary: dict[str, Any], m: str) -> str:
    """Описание варианта: базовый метод и переопределения конфига (из summary.methods[m].variant); '' для базового прогона."""
    v = _get(summary, "methods", m, "variant") or {}
    base, ov = v.get("base_method"), v.get("cfg_overrides") or {}
    parts: list[str] = []
    if base and base != m:
        parts.append(f"база: {method_label(base)}")
    if ov:
        flat: list[str] = []

        def _walk(d: dict[str, Any], prefix: str) -> None:
            for k, val in d.items():
                if isinstance(val, dict):
                    _walk(val, f"{prefix}{k}.")
                else:
                    flat.append(f"{prefix}{k}={val}")

        _walk(ov, "")
        parts.append("; ".join(flat))
    if base and base != m and v.get("index_dir") and not v["index_dir"].rstrip("/").endswith("/" + base):
        parts.append(f"индекс: {v['index_dir']}")
    return "; ".join(parts)


def incomplete_note(summary: dict[str, Any]) -> str:
    """Предупреждение о методах с неполными скорами (summary.methods[m].incomplete); '' если таких нет."""
    bad = [(m, M.get("incomplete")) for m, M in (summary.get("methods") or {}).items() if M.get("incomplete")]
    if not bad:
        return ""
    items = "; ".join(f"{method_label(m)}: " + ", ".join(f"{s} {v['n']}/{v['n_queries']}" for s, v in inc.items()) for m, inc in bad)
    return f"\n⚠ Неполные скоры (прогон --limit?): {items}. Перезапустите scripts/07_eval.py --force.\n"


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
    rows.append([f"Все классы, калибровка «1 запрос/функция», α = {pct(a0)}",
                 *[fmt_ci(_get(summary, "methods", m, "tpr", a0, "conformal_single", "overall")) for m in methods]])
    rows.append([f"Все классы, оракульный порог, α = {pct(a0)}", *[fmt_ci(_get(summary, "methods", m, "tpr", a0, "oracle", "overall")) for m in methods]])
    langs = sorted({l for m in methods for l in _get(summary, "methods", m, "tpr", a0, "conformal", "by_lang", default={})})
    for lang in langs:  # блок «по языкам» (статья, табл. 3)
        rows.append([f"По языкам, α = {pct(a0)}: {lang}", *[fmt_ci(_get(summary, "methods", m, "tpr", a0, "conformal", "by_lang", lang)) for m in methods]])
    rows.append([f"Пропуски членов по причинам, α = {pct(a0)}", *[misses_cell(_get(summary, "methods", m, "tpr", a0, "conformal", "misses_by_reason"))
                                                               for m in methods]])
    note = ("\nПримечание. M1/M2 на abstract(full)-токенах по построению инвариантны к T1–T4 и combo (детекторы клонов "
            "типа 2); информативные классы RQ1 для них — insert_deadcode, reorder_stmts, partial, paraphrase, translate; "
            "вариант «M1 lexical» (fingerprint.token_mode=lexical) показывает, как эти классы ломают классический Moss. "
            "Дообученный M3/M4 на программных классах — результат «в распределении» (тот же генератор преобразований, "
            "что при обучении; шаблоны insert_deadcode в оценке — отдельный набор): см. варианты semantic_zeroshot и "
            "semantic_noevasion. Пропуски по причинам: too_short — запрос короче k токенов, all_common — все отпечатки "
            "отфильтрованы фильтром общности (гарантия winnowing условна: нужен хотя бы один не-общий отпечаток).\n")
    return (f"Таблица T2 — TPR при FPR = {pct(a0)} (конформный порог τ по public_calib) по классам преобразований и языкам; "
            "в скобках — 95 % ДИ бутстрэпа с кластеризацией по исходной функции.\n\n" + md_table(header, rows) + note + incomplete_note(summary))


def misses_cell(block: dict[str, Any] | None) -> str:
    """'n: причина — k, ...' из misses_by_reason; '—' если нет."""
    if not block:
        return DASH
    n = block.get("n_misses")
    parts = [f"{REASON_LABELS.get(k, k)} {v}" for k, v in block.items() if k != "n_misses"]
    return f"{n}" + (f" ({'; '.join(parts)})" if parts else "")


def verdict_cell(pt: dict[str, Any] | None, alpha: float) -> str:
    """Вывод о превышении α: точечная оценка, кластерно-робастный p (test_above_alpha), МДЭ; '—' без данных."""
    if not pt or pt.get("value") is None:
        return DASH
    t = pt.get("test_above_alpha") or {}
    above = pt["value"] > alpha
    s = "выше α" if above else "не выше α"
    if t.get("p_value") is not None:
        s += f"; p = {fmt(t['p_value'], 3)}"
        if above and t["p_value"] < 0.05:
            s += " (значимо)"
    if t.get("mde_80") is not None:
        s += f"; МДЭ ≈ {fmt(t['mde_80'], 4)}"
    return s


def table_T3(summary: dict[str, Any]) -> str:
    """T3: FPR при конформном пороге по наборам негативов — кластерный ДИ (основной) и Клоппер–Пирсон (справочный),
    калибровка «1 запрос/функция», доменная калибровка, оракул. Вывод о нарушении гарантии — по кластерному ДИ."""
    methods = order_methods(list(summary.get("methods", {})))
    aks = summary.get("alpha_keys") or ["0.01"]
    header = ["Метод", "α", "τ_α", "Связи в τ, %", "FPR public_test [кластерный ДИ]", "FPR public_test, ДИ Клоппера–Пирсона",
              "FPR hard_neg [ДИ]", "Макс. FPR по подгруппам (язык / класс T)",
              "τ «1 запрос/функция»", "FPR public_test при τ «1 запрос/функция» [ДИ]",
              "τ мондриан (max по ячейкам)", "FPR public_test при τ мондриан [ДИ]", "TPR при τ мондриан",
              "τ_α при доменной калибровке", "FPR hard_neg (тестовые репозитории) при доменной калибровке [ДИ]",
              "TPR при доменной калибровке", "TPR при оракульном пороге", "FPR public_test vs α (кластерный z-тест; МДЭ при мощности 80 %)"]
    rows = []
    cell_rows: list[list[Any]] = []
    for ak in aks:
        for m in methods:
            M = summary["methods"][m]
            th = _get(M, "thresholds", ak, default={}) or {}
            fpr_c = _get(M, "fpr", ak, "conformal", default={}) or {}
            fpr_s = _get(M, "fpr", ak, "conformal_single", default={}) or {}
            fpr_m = _get(M, "fpr", ak, "mondrian", default={}) or {}
            fpr_d = _get(M, "fpr", ak, "domain", default={}) or {}
            ms = _get(fpr_c, "public_test", "max_subgroup") or _get(fpr_c, "hard_neg", "max_subgroup")
            pt = fpr_c.get("public_test")
            tie = th.get("tie_mass_at_tau")
            rows.append([method_label(m), pct(ak), fmt(th.get("conformal")), fmt(tie * 100 if tie is not None else None, 1),
                         fmt_ci_cluster(pt), fmt_interval(pt.get("ci") if pt else None), fmt_ci_cluster(fpr_c.get("hard_neg")),
                         f"{fmt(ms['value'])} ({ms['group']}, n={ms['n']})" if ms else DASH,
                         fmt(th.get("conformal_single")), fmt_ci_cluster(fpr_s.get("public_test")),
                         fmt(th.get("mondrian")), fmt_ci_cluster(fpr_m.get("public_test")), fmt_ci(_get(M, "tpr", ak, "mondrian", "overall")),
                         fmt(th.get("domain")), fmt_ci_cluster(fpr_d.get("hard_neg_test")), fmt_ci(_get(M, "tpr", ak, "domain", "overall")),
                         fmt_ci(_get(M, "tpr", ak, "oracle", "overall")), verdict_cell(pt, float(ak))])
            cells = th.get("mondrian_cells") or {}
            for cell, c in cells.items():
                cell_rows.append([method_label(m), pct(ak), cell, fmt(c.get("tau")), c.get("n", DASH), "да" if c.get("sufficient") else "нет",
                                  fmt_ci_cluster(_get(fpr_c, "public_test", "by_transform", cell.split("@")[0])),
                                  fmt_ci_cluster(_get(fpr_m, "public_test", "by_transform", cell.split("@")[0]))])
    dc = summary.get("domain_calibration") or {}
    notes = summary.get("notes") or {}
    note = (f"Доменная калибровка: {len(dc.get('calib_repos', []))} репозиториев hard_neg в калибровке, "
            f"{len(dc.get('test_repos', []))} — в тесте (доля {dc.get('fraction', DASH)}). "
            "«Связи в τ» — доля калибровочных скоров, равных τ (при > 0 строгое правило score > τ консервативно). "
            "Вывод «vs α» — не бинарный вердикт по ДИ: точечная оценка, односторонний кластерно-робастный z-тест H0: FPR ≤ α "
            "(p < 0.05 — значимое превышение) и минимально обнаружимый FPR (МДЭ) при мощности 80 % — ниже него превышение "
            "неразличимо на данном объёме теста (при α = 0,1 % и ~5000 функций ≈ 3α). ")
    if notes.get("fpr_ci"):
        note += notes["fpr_ci"] + " "
    if notes.get("exchangeability"):
        note += notes["exchangeability"]
    out = ("Таблица T3 — Эмпирический FPR при конформном пороге τ_α по наборам негативов: 95 % ДИ кластерного бутстрэпа по "
           "исходной функции (основной) и Клоппера–Пирсона (справочный), калибровка «1 запрос/функция» (маргинальная гарантия), "
           "мондриановская калибровка по ячейкам (преобразование, L) с порогом развёртывания max_t τ_t (гарантия по классу), "
           "эффект доменной калибровки и оракульный порог. Гарантия: FPR ≤ α при обменяемости негативов.\n\n"
           + md_table(header, rows) + "\n" + note + "\n")
    if cell_rows:
        out += ("\nПо ячейкам мондриановской калибровки (τ_t; «достаточно» — n функций ≥ ⌈1/α⌉ − 1, иначе τ_t = ∞ и ячейка не входит "
                "в max): FPR public_test класса при объединённом τ_α и при τ мондриан.\n\n"
                + md_table(["Метод", "α", "Ячейка", "τ_t", "n функций", "Достаточно", "FPR класса при τ_α [ДИ]", "FPR класса при τ мондриан [ДИ]"],
                           cell_rows))
    out += table_T3_windows(summary)
    return out + incomplete_note(summary)


def table_T3_windows(summary: dict[str, Any]) -> str:
    """Блок T3 для сценария IDE: порог окон (public_calib_windows), TPR protected_windows (identity — проверка выравнивания
    индекса, partial — результат), FPR public_test_windows при τ окон и при τ функций. '' если окон нет."""
    methods = order_methods(list(summary.get("methods", {})))
    aks = summary.get("alpha_keys") or ["0.01"]
    rows = []
    for ak in aks:
        for m in methods:
            w = _get(summary, "methods", m, "tpr_windows", ak)
            if not w:
                continue
            fpr_w = _get(w, "fpr", "public_test_windows")
            fpr_f = _get(w, "fpr_at_function_tau", "public_test_windows")
            partial = w.get("by_partial_L") or {}
            rows.append([method_label(m), pct(ak), fmt(w.get("threshold")), f"{w.get('calibration', DASH)} (n={w.get('n_calib', DASH)})",
                         fmt_ci(_get(w, "by_transform", "identity")),
                         "; ".join(f"L={L}: {fmt_ci(v)}" for L, v in sorted(partial.items(), key=lambda kv: int(kv[0]))) or DASH,
                         fmt_ci_cluster(fpr_w), verdict_cell(fpr_w, float(ak)), fmt_ci_cluster(fpr_f)])
    if not rows:
        return ""
    return ("\nСценарий IDE (файловые окна): порог τ_w калибруется по негативам формы окна (public_calib_windows), FPR — на "
            "public_test_windows. TPR identity-окон protected_windows — проверка выравнивания индекса (окна индекса совпадают с "
            "окнами запросов; M0 = 1 по построению), результат детекции — частичные окна partial@L.\n\n"
            + md_table(["Метод", "α", "τ_w", "Калибровка окон", "TPR identity-окон (sanity)", "TPR partial@L", "FPR public_test_windows при τ_w [ДИ]",
                        "FPR окон vs α", "FPR public_test_windows при τ функций [ДИ]"], rows))


def table_T4(summary: dict[str, Any]) -> str:
    """T4: абляции — все методы/варианты из summary: описание варианта, TPR при α₀ и α₁, AUROC члены/hard_neg, FPR hard_neg при τ_α₀."""
    methods = order_methods(list(summary.get("methods", {})))
    aks = summary.get("alpha_keys") or ["0.01"]
    a0 = aks[0]
    header = ["Вариант", "Описание (база; переопределения)", *[f"TPR@{pct(ak)}" for ak in aks], "AUROC члены/hard_neg",
              "AUROC члены/public_test", f"FPR hard_neg при τ_{pct(a0)}"]
    rows = [[method_label(m), variant_description(summary, m) or DASH,
             *[fmt_ci(_get(summary, "methods", m, "tpr", ak, "conformal", "overall")) for ak in aks],
             fmt_ci(_get(summary, "methods", m, "auroc", "hard_neg")), fmt_ci(_get(summary, "methods", m, "auroc", "public_test")),
             fmt_ci_cluster(_get(summary, "methods", m, "fpr", a0, "conformal", "hard_neg"))] for m in methods]
    return ("Таблица T4 — Абляции: TPR при конформном пороге, AUROC и FPR на hard_neg (кластерный ДИ). Варианты методов — каталоги "
            "results/scores/<вариант>/ (scripts/07_eval.py --methods hybrid --variant hybrid_rule --set semantic.hybrid_rule=two_threshold; "
            "semantic_zeroshot — --index-dir с индексом zero-shot энкодера и т. д.).\n\n" + md_table(header, rows) + incomplete_note(summary))


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
    """T5: память индекса, байт/запись, время сборки, латентность p50/p95 (прогон eval; сквозная оценка end_to_end;
    results/latency.json; энкодер — results/latency_encoder.json для semantic*/hybrid*)."""
    methods = order_methods(list(summary.get("methods", {})))
    latency = latency or {}
    enc = encoder_latency(encoder)
    n_all = enc.get("threads_all") or 4
    header = ["Метод", "Записей", "Память индекса, МБ", "Байт на запись", "Построение индекса, мин",
              "Прогон eval, p50 / p95, мс (режим)", "Сквозная латентность запроса, p50 / p95, мс (CPU-1, оценка)",
              "CPU, 1 поток, p50 / p95", f"CPU, {n_all} потоков, p50 / p95", "GPU, батч 1, p50 / p95", "GPU, батч 32, запросов/с"]
    rows = []
    sources: dict[str, str] = {}
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
        e2e = lat.get("end_to_end") or {}
        if e2e.get("p95") is not None:
            e2e_s = f"{fmt(e2e.get('p50'), 2)} / {fmt(e2e.get('p95'), 2)}"
            sources[method_label(m)] = str(e2e.get("source"))
        else:
            e2e_s = DASH + (f" ({e2e['reason']})" if e2e.get("reason") else "")
        rows.append([method_label(m), M.get("n_records", DASH), fmt(mem / 1e6 if mem is not None else None, 1),
                     fmt(M.get("bytes_per_record"), 1), fmt(bs / 60 if bs is not None else None, 2),
                     f"{fmt(lat.get('p50'), 2)} / {fmt(lat.get('p95'), 2)} ({lat.get('mode') or DASH})", e2e_s,
                     _lat_pair(ext.get("cpu1")), _lat_pair(ext.get("cpu4")), _lat_pair(ext.get("gpu_b1")), fmt(_get(ext, "gpu_b32", "qps"), 1)])
        comps = ext.get("components") or {}
        for cname, clabel in (("encoder", "— из них инференс энкодера"), ("ann", "— из них поиск ANN"), ("rerank", "— из них переранжирование")):
            c = comps.get(cname)
            if c:
                rows.append([clabel, DASH, DASH, DASH, DASH, DASH, DASH, _lat_pair(c.get("cpu1")), _lat_pair(c.get("cpu4")), _lat_pair(c.get("gpu_b1")), fmt(_get(c, "gpu_b32", "qps"), 1)])
        b = lat.get("benchmark")
        if b:
            rows.append([f"— бенчмарк батч 1 ({b.get('repeats')} повторов × {b.get('n_queries')} запросов, {b.get('mode')})", DASH, DASH, DASH, DASH,
                         f"{fmt(b.get('p50_ms'), 2)} / {fmt(b.get('p95_ms'), 2)}", DASH, DASH, DASH, DASH, DASH])
    src_note = ("\nИсточники сквозной оценки: " + "; ".join(f"{k} — {v}" for k, v in sources.items()) + ".\n") if sources else ""
    return ("Таблица T5 — Латентность одного запроса (p50 / p95, мс) и память индекса. Колонки CPU/GPU — из results/latency.json "
            "(если файл есть); строка «инференс энкодера» — из results/latency_encoder.json (scripts/06_embed.py); "
            "«прогон eval» — wall-clock в scripts/07_eval.py (для numpy-пути semantic/hybrid — амортизированное время "
            "батча без энкодера, т. е. только ANN/переранжирование, и не сравнимо с per_query); «сквозная латентность» — "
            "summary.latency_ms.end_to_end: per_query как есть, для амортизированных режимов — энкодер CPU-1 батч 1 + "
            "бенчмарк батча 1 (--bench).\n\n" + md_table(header, rows) + src_note + incomplete_note(summary))


def table_T6(summary: dict[str, Any], privacy: dict[str, Any] | None) -> str | None:
    """T6: защиты индекса эмбеддингов из results/privacy.json (None, если файла нет).

    Схема (smcode.privacy.run): {"leaked_pairs": [...], "defenses": [{"name", "memory_ratio", "tpr": {"semantic": RATE|float,
    "hybrid": RATE|null}, "fpr": {"public_test": RATE, "hard_neg": RATE}, "attacks": {"A1": {"f1", "rare_id_recall"},
    "A2": {"<n>": {"f1"}}, "A3": {"bleu", "id_acc"} | null}}]}.
    """
    if not privacy:
        return None
    defs = privacy.get("defenses") or privacy.get("results") or []
    if isinstance(defs, dict):
        defs = [{"name": k, **v} for k, v in defs.items()]
    ns = [str(n) for n in (privacy.get("leaked_pairs") or [100, 1000, 10000]) if str(n) != "0"]
    alpha = privacy.get("alpha", 0.01)
    header = ["Защита", "Память относительно float16", f"TPR@{pct(alpha)} M3", f"TPR@{pct(alpha)} M4", "FPR public_test / hard_neg под защитой",
              "A1: F1 по токенам", "A1: полнота по редким идентификаторам, известным в public_train (точность / случайный уровень)",
              *[f"A2, n = {n}: F1" for n in ns], "A3: BLEU / точность идентификаторов"]

    def _rate(v: Any) -> str:
        return fmt_ci(v) if isinstance(v, dict) else fmt(v)

    def _rare(a1: dict[str, Any]) -> str:
        """Полнота по редким идентификаторам; в скобках — точность по ним и случайный уровень, если privacy.json их содержит."""
        rec = fmt(a1.get("rare_id_recall"))
        if a1.get("rare_id_precision") is not None or a1.get("rare_id_recall_chance") is not None:
            rec += f" ({fmt(a1.get('rare_id_precision'))} / {fmt(a1.get('rare_id_recall_chance'))})"
        return rec

    rows = []
    for d in defs:
        at = d.get("attacks") or {}
        a1, a2, a3 = at.get("A1") or {}, at.get("A2") or {}, at.get("A3") or {}
        fp = d.get("fpr") or {}
        rows.append([d.get("name", DASH), fmt(d.get("memory_ratio"), 3), _rate(_get(d, "tpr", "semantic")), _rate(_get(d, "tpr", "hybrid")),
                     f"{_rate(fp.get('public_test'))} / {_rate(fp.get('hard_neg'))}" if fp else DASH,
                     fmt(a1.get("f1")), _rare(a1), *[fmt(_get(a2, n, "f1")) for n in ns],
                     f"{fmt(a3.get('bleu'))} / {fmt(a3.get('id_acc'))}" if a3 else DASH])
    cov = (privacy.get("a1") or {}).get("protected_coverage") or {}
    cov_note = ""
    if cov:
        cov_note = (f" Покрытие словарём A1 вхождений идентификаторов protected: {fmt(cov.get('occurrence_coverage'))}; "
                    f"идентификаторов только protected (df = 0 в public_train): {cov.get('n_protected_only_ids', DASH)} "
                    f"({fmt(cov.get('protected_only_occurrence_fraction'))} вхождений) — A1/A2 их не измеряют (нижняя оценка утечки), "
                    "их покрывает только A3.")
    idx_note = f" Индекс: {privacy.get('index_composition', 'protected')} (как в основной оценке)." if privacy.get("index_composition") else ""
    return (f"Таблица T6 — Защиты индекса эмбеддингов: полезность (TPR при FPR = {pct(alpha)}, конформный порог под той же защитой) "
            "и утечка (атаки A1–A3). Источник: results/privacy.json. Редкие идентификаторы — df ≤ 3 в public_train; "
            "«случайный уровень» — полнота при случайном выборе того же числа токенов." + cov_note + idx_note + "\n\n" + md_table(header, rows))


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
        fprs = [fmt_ci_cluster(_get(summary, "methods", m, "fpr", a0, "conformal", "hard_neg", "by_transform", t)) for m in methods]
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
