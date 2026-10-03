"""Тесты модуля eval: калибровка (DESIGN.md §8), метрики, таблицы, рисунки, скрипты 07/09 (синтетика, CPU, без torch)."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from smcode import calibration as cal
from smcode.config import DEFAULT_CONFIG, load_config
from smcode.eval import metrics as M
from smcode.eval import run_eval as RE
from smcode.eval.plots import FIGURE_FILES, fig_F7, latency_point, privacy_points, write_figures
from smcode.eval.report import format_summary, run_report, summary_is_stale
from smcode.eval.tables import TABLE_FILES, fmt, fmt_ci, fmt_ci_cluster, table_T4, variant_description, write_tables
from smcode.types import QueryResult, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("python", "c", "go")
TRANSFORMS = ("identity", "rename_ids", "combo", "paraphrase", "translate")
PARTIAL_L = (5, 10)
CODE = "def f(x):\n    return x + 1\n"
# параметры Beta-распределений скоров: (члены, не-члены)
METHOD_DISTS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "winnowing": ((3.0, 3.0), (2.0, 5.0)),
    "semantic": ((5.0, 2.0), (2.0, 5.0)),
    "hybrid": ((6.0, 2.0), (2.0, 6.0)),
    "random": ((2.0, 5.0), (2.0, 5.0)),
}


# ----------------------------------------------------------------------------- калибровка


def test_conformal_threshold_fpr_guarantee():
    rng = np.random.default_rng(1)
    alpha = 0.05
    fprs = []
    for _ in range(20):
        calib = rng.beta(2, 5, size=2000)
        tau = cal.conformal_threshold(calib, alpha)
        fresh = rng.beta(2, 5, size=2000)
        fprs.append(cal.rate(fresh, tau))
    assert np.mean(fprs) <= alpha + 0.005  # в среднем FPR ≤ α
    assert max(fprs) <= alpha + 0.02
    # массовые совпадения скоров (нули у отпечатков): строгое правило сохраняет гарантию
    for _ in range(10):
        calib = np.where(rng.random(2000) < 0.97, 0.0, rng.random(2000))
        tau = cal.conformal_threshold(calib, alpha)
        fresh = np.where(rng.random(2000) < 0.97, 0.0, rng.random(2000))
        assert cal.rate(fresh, tau) <= alpha + 0.02
    assert tau == 0.0  # порог упёрся в нулевую массу, но решение score > 0 даёт FPR ≈ 3 % ≤ 5 %


def test_conformal_threshold_small_n_and_level():
    assert cal.conformal_threshold([0.1, 0.2, 0.3], 0.01) == float("inf")
    assert cal.min_calibration_size(0.01) == 99
    assert cal.conformal_level(99, 0.01) == 99
    assert np.isfinite(cal.conformal_threshold(np.linspace(0, 1, 99), 0.01))
    assert cal.conformal_threshold(np.linspace(0, 1, 98), 0.01) == float("inf")
    assert cal.conformal_threshold([], 0.1) == float("inf")
    # k-я порядковая статистика: n=9, α=0.1 → k=⌈10·0.9⌉=9 → максимум
    assert cal.conformal_threshold([0.5, 0.1, 0.9, 0.3, 0.2, 0.4, 0.6, 0.7, 0.8], 0.1) == 0.9
    assert cal.conformal_threshold([0.5, 0.1, 0.9, 0.3, 0.2, 0.4, 0.6, 0.7, 0.8], 0.2) == 0.8
    with pytest.raises(ValueError):
        cal.conformal_threshold([0.1], 1.5)
    assert not cal.decide([0.1, 0.9], float("inf")).any()


def test_clopper_pearson_known_values():
    lo, hi = cal.clopper_pearson(0, 10)
    assert lo == 0.0 and abs(hi - 0.3085) < 1e-3
    lo, hi = cal.clopper_pearson(10, 10)
    assert abs(lo - 0.6915) < 1e-3 and hi == 1.0
    lo, hi = cal.clopper_pearson(5, 10)
    assert abs(lo - 0.1871) < 1e-3 and abs(hi - 0.8129) < 1e-3
    assert cal.clopper_pearson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        cal.clopper_pearson(11, 10)
    blk = cal.rate_with_ci([0.0, 0.2, 0.9, 0.95], 0.5)
    assert blk["k"] == 2 and blk["n"] == 4 and blk["value"] == 0.5 and blk["ci"][0] < 0.5 < blk["ci"][1]


def test_threshold_at_fpr_oracle():
    neg = [0.0, 0.0, 0.0, 0.5, 0.9]
    assert cal.threshold_at_fpr(neg, 0.2) == 0.5
    assert cal.threshold_at_fpr(neg, 0.1) == 0.9
    assert cal.threshold_at_fpr(neg, 0.6) == 0.0
    assert cal.rate(neg, cal.threshold_at_fpr(neg, 0.2)) <= 0.2


def test_domain_calibration_split():
    import random

    repos = [f"r{i}" for i in range(7)]
    a, b = cal.split_repos(repos, 0.5, random.Random(3))
    assert sorted(a + b) == repos and a and b
    assert (a, b) == cal.split_repos(repos, 0.5, random.Random(3))  # детерминизм
    rng = np.random.default_rng(0)
    hard = rng.beta(2, 4, 700)
    hard_repos = [repos[i % 7] for i in range(700)]
    dc = cal.domain_calibration(rng.beta(2, 5, 500), hard, hard_repos, 0.05, calib_repos=a)
    assert dc.n_calib == 500 + sum(r in a for r in hard_repos)
    assert dc.test_mask.sum() == sum(r in b for r in hard_repos)
    assert np.isfinite(dc.tau) and dc.to_dict()["n_test_queries"] == int(dc.test_mask.sum())


# ----------------------------------------------------------------------------- бутстрэп и AUROC


def test_weighted_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(5)
    y = rng.random(400) < 0.4
    s = np.round(np.where(y, rng.beta(4, 2, 400), rng.beta(2, 4, 400)), 2)  # округление → много связей
    est = M.AurocEstimator(y, s)
    assert abs(est.auc() - roc_auc_score(y, s)) < 1e-12
    w = rng.integers(0, 4, 400).astype(float)
    assert abs(est.auc(w) - roc_auc_score(y, s, sample_weight=w)) < 1e-12


def test_cluster_bootstrap_ci():
    rng = np.random.default_rng(2)
    clusters = np.repeat([f"c{i}" for i in range(100)], 7).astype(object)
    hits = rng.random(700) < 0.7
    boot = M.ClusterBootstrap(clusters, 200, rng)
    assert boot.C == 100 and boot.counts.shape == (200, 100) and np.all(boot.counts.sum(1) == 100)
    lo, hi = boot.rate_ci(hits, np.ones(700, bool))
    assert lo <= hits.mean() <= hi and hi - lo < 0.3
    assert boot.rate_ci(hits, np.zeros(700, bool)) is None
    assert M.ClusterBootstrap(clusters, 0, rng).rate_ci(hits) is None


# ----------------------------------------------------------------------------- синтетический стенд


def _make_queries(set_name: str, n_src: int, repos: list[str], rng: np.random.Generator) -> list[dict[str, Any]]:
    label = 1 if set_name.startswith("protected") else 0
    rows = []
    for i in range(n_src):
        lang, repo = LANGS[i % 3], repos[i % len(repos)]
        sid = f"{set_name}/{repo}/f{i}.{lang}:1-20"
        n_tok = int(rng.integers(8, 400))
        for t in TRANSFORMS:
            params: dict[str, Any] = {}
            qlang = lang
            if t == "translate":
                qlang = LANGS[(i + 1) % 3]
                params = {"dst_lang": qlang, "src_lang": lang}
            rows.append({"qid": f"{sid}#{t}#0", "source_id": sid, "label": label, "set": set_name, "transform": t, "params": params,
                         "lang": qlang, "repo": repo, "code": CODE, "n_lines": 2, "n_tokens": n_tok})
        for j, L in enumerate(PARTIAL_L):
            rows.append({"qid": f"{sid}#partial#{j}", "source_id": sid, "label": label, "set": set_name, "transform": "partial",
                         "params": {"L": L}, "lang": lang, "repo": repo, "code": CODE, "n_lines": L, "n_tokens": min(n_tok, 4 * L)})
    return rows


def _make_cfg(tmp_path: Path) -> dict[str, Any]:
    paths = {k: str(tmp_path / v) for k, v in {"raw_repos": "data/raw", "functions": "data/functions", "queries": "data/queries",
                                                "indexes": "data/indexes", "results": "results", "figures": "figures",
                                                "embeddings": "data/embeddings"}.items()}
    return load_config(DEFAULT_CONFIG, overrides={"paths": paths, "calibration": {"alphas": [0.05, 0.01]}, "eval": {"bootstrap": 50}})


@pytest.fixture(scope="module")
def stand(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Синтетический стенд: запросы, скоры четырёх «методов», stats.json, privacy.json."""
    tmp = tmp_path_factory.mktemp("stand")
    cfg = _make_cfg(tmp)
    rng = np.random.default_rng(20261003)
    sets = {"protected": (80, ["p1", "p2", "p3"]), "public_calib": (80, ["c1", "c2", "c3", "c4"]),
            "public_test": (80, ["t1", "t2", "t3", "t4"]), "hard_neg": (60, [f"h{i}" for i in range(6)])}
    queries: dict[str, list[dict[str, Any]]] = {}
    qdir = Path(cfg["paths"]["queries"])
    for s, (n, repos) in sets.items():
        queries[s] = _make_queries(s, n, repos, rng)
        write_jsonl(qdir / f"{s}.jsonl", queries[s])
    with open(qdir / "protected.llm_stats.json", "w") as f:
        json.dump({"set": "protected", "n_sampled": 80, "paraphrase_ok": 70, "paraphrase_failed": 10, "translate_ok": 60, "translate_failed": 20}, f)
    for m, (dpos, dneg) in METHOD_DISTS.items():
        for s, rows in queries.items():
            d = dpos if s == "protected" else dneg
            out = [{"qid": r["qid"], "score": float(rng.beta(*d)), "best_id": "protected/x", "latency_ms": float(rng.uniform(0.2, 3.0))} for r in rows]
            write_jsonl(RE.scores_path(cfg, m, s), out)
            with open(RE.scores_dir(cfg, m) / f"{s}.meta.json", "w") as f:
                json.dump({"memory_bytes": 1_000_000 * (1 + len(m)), "n_records": 1234, "latency_mode": "per_query",
                           "latency_benchmark": {"n_queries": 5, "repeats": 3, "p50_ms": 1.0, "p95_ms": 2.0, "mean_ms": 1.2, "mode": "query"}}, f)
    fdir = Path(cfg["paths"]["functions"])
    fdir.mkdir(parents=True, exist_ok=True)
    stats = {"length_bins_tokens": cfg["eval"]["length_bins_tokens"],
             "splits": {s: {"n_raw": 100, "n_final": 90, "removed_exact": 5, "removed_near_dup": 5, "n_repos": 3,
                            "by_lang": {"python": 50, "c": 40}, "by_length_bin": {f"{lo}-{hi}": 18 for lo, hi in cfg["eval"]["length_bins_tokens"]}}
                        for s in ("protected", "hard_neg", "public_train", "public_calib", "public_test")},
             "windows": {"n_final": 300}, "totals": {"n_raw": 500, "n_final": 450}}
    with open(fdir / "stats.json", "w") as f:
        json.dump(stats, f)
    privacy = {"leaked_pairs": [0, 100, 1000, 10000], "defenses": [
        {"name": "Без защиты", "memory_ratio": 1.0, "tpr": {"semantic": 0.9, "hybrid": 0.93},
         "attacks": {"A1": {"f1": 0.8, "rare_id_recall": 0.5}, "A3": {"bleu": 0.4, "id_acc": 0.6}}},
        {"name": "Проекция Q", "memory_ratio": 1.0, "tpr": {"semantic": {"value": 0.9, "ci": [0.88, 0.92]}, "hybrid": 0.93},
         "attacks": {"A1": {"f1": 0.1, "rare_id_recall": 0.02}, "A2": {"100": {"f1": 0.2}, "1000": {"f1": 0.5}, "10000": {"f1": 0.75}}}},
        {"name": "Q + int8", "memory_ratio": 0.5, "tpr": {"semantic": 0.89, "hybrid": 0.92},
         "attacks": {"A1": {"f1": 0.1, "rare_id_recall": 0.02}, "A2": {"100": {"f1": 0.2}, "1000": {"f1": 0.45}, "10000": {"f1": 0.7}}}},
    ]}
    with open(Path(cfg["paths"]["results"]) / "privacy.json", "w") as f:
        json.dump(privacy, f, ensure_ascii=False)
    with open(Path(cfg["paths"]["results"]) / "latency.json", "w") as f:
        json.dump({"semantic": {"cpu1": {"p50_ms": 30, "p95_ms": 45}, "cpu4": {"p50_ms": 12, "p95_ms": 18}, "gpu_b1": {"p50_ms": 3, "p95_ms": 5},
                                "gpu_b32": {"qps": 900}, "components": {"encoder": {"cpu1": {"p50_ms": 28, "p95_ms": 42}}},
                                "by_size": [{"n": 1000, "p95_ms": 2}, {"n": 10000, "p95_ms": 4}, {"n": 100000, "p95_ms": 20}]}}, f)
    with open(Path(cfg["paths"]["results"]) / "latency_encoder.json", "w") as f:  # схема scripts/06_embed.py
        json.dump({"model": "m", "cpu_threads_all": 8, "configs": [
            {"device": "cpu", "threads": 1, "batch": 1, "p50_ms": 40.0, "p95_ms": 55.0, "per_item_ms": 40.0},
            {"device": "cpu", "threads": 8, "batch": 1, "p50_ms": 15.0, "p95_ms": 21.0, "per_item_ms": 15.0},
            {"device": "cuda", "threads": None, "batch": 1, "p50_ms": 4.0, "p95_ms": 6.0, "per_item_ms": 4.0},
            {"device": "cuda", "threads": None, "batch": 32, "p50_ms": 40.0, "p95_ms": 50.0, "per_item_ms": 1.25}]}, f)
    return {"cfg": cfg, "queries": queries, "tmp": tmp}


# ----------------------------------------------------------------------------- метрики


def test_metrics_summary_end_to_end(stand: dict[str, Any]):
    cfg = stand["cfg"]
    summary = M.compute_summary(cfg)
    assert set(summary["methods"]) == set(METHOD_DISTS)
    assert summary["alpha_keys"] == ["0.05", "0.01"] and summary["decision_rule"] == "score > tau"
    assert summary["domain_calibration"]["calib_repos"] and summary["domain_calibration"]["test_repos"]
    good, rnd = summary["methods"]["semantic"], summary["methods"]["random"]
    for m in (good, rnd):
        assert set(m["thresholds"]) == {"0.05", "0.01"}
        assert {"conformal", "oracle", "domain"} <= set(m["tpr"]["0.05"])
        g = m["tpr"]["0.05"]["conformal"]
        assert set(g["by_transform"]) == set(TRANSFORMS) | {"partial"}
        assert set(g["by_partial_L"]) == {str(L) for L in PARTIAL_L}
        assert set(g["by_lang"]) == set(LANGS)
        assert set(g["by_length_bin"]) == {f"{lo}-{hi}" for lo, hi in cfg["eval"]["length_bins_tokens"]}
        assert set(g["by_translate_dst"]) == set(LANGS)
        blk = g["overall"]
        assert blk["n"] == 80 * (len(TRANSFORMS) + len(PARTIAL_L)) and blk["ci"][0] <= blk["value"] <= blk["ci"][1]
        assert set(m["fpr"]["0.05"]["conformal"]) == {"public_calib", "public_test", "hard_neg"}
        assert set(m["fpr"]["0.05"]["domain"]) == {"public_test", "hard_neg_test"}
        assert m["fpr"]["0.05"]["conformal"]["public_test"]["max_subgroup"]["group"].split(":")[0] in ("lang", "transform")
        pt = m["fpr"]["0.05"]["conformal"]["public_test"]
        assert pt["ci_cluster"] is not None and pt["ci_cluster"][0] <= pt["value"] <= pt["ci_cluster"][1]
        assert "ci_cluster" in pt["max_subgroup"] and all(v["ci_cluster"] is not None for v in pt["by_lang"].values())
        assert "conformal_single" in m["tpr"]["0.05"] and m["thresholds"]["0.05"]["n_calib_single"] == 80
        assert m["thresholds"]["0.05"]["tie_mass_at_tau"] == pytest.approx(1 / (80 * 7))  # непрерывные скоры: связь только сам τ
        assert set(m["fpr"]["0.05"]["conformal_single"]) == {"public_calib", "public_test", "hard_neg"}
        assert m["latency_ms"]["end_to_end"]["p95"] == m["latency_ms"]["p95"] and m["incomplete"] is None
        assert set(m["confusion"]["0.05"]["conformal"]) == {"protected", "public_calib", "public_test", "hard_neg"}
        c = m["confusion"]["0.05"]["conformal"]
        assert c["protected"]["tp"] + c["protected"]["fn"] == blk["n"] and c["public_test"]["fp"] == m["fpr"]["0.05"]["conformal"]["public_test"]["k"]
        assert m["memory_bytes"] > 0 and m["n_records"] == 1234 and m["bytes_per_record"] > 0
        assert m["latency_ms"]["p50"] is not None and m["latency_ms"]["p95"] >= m["latency_ms"]["p50"]
        assert m["latency_ms"]["benchmark"]["p95_ms"] == 2.0
        assert set(m["roc"]) == {"public_test", "hard_neg"} and len(m["roc"]["hard_neg"]["fpr"]) >= 2
    # качество: хороший метод лучше случайного; FPR ≤ α (+ шум) на public_test; калибровочный FPR ≤ α точно
    for ak, a in (("0.05", 0.05), ("0.01", 0.01)):
        assert good["tpr"][ak]["conformal"]["overall"]["value"] > rnd["tpr"][ak]["conformal"]["overall"]["value"] + 0.2
        assert good["fpr"][ak]["conformal"]["public_calib"]["value"] <= a
        assert good["fpr"][ak]["conformal"]["public_test"]["value"] <= a + 0.03
        assert good["fpr"][ak]["oracle"]["public_test"]["value"] <= a
        assert good["tpr"][ak]["oracle"]["overall"]["value"] >= good["tpr"][ak]["conformal"]["overall"]["value"] - 0.1
    assert good["auroc"]["hard_neg"]["value"] > 0.85 and abs(rnd["auroc"]["hard_neg"]["value"] - 0.5) < 0.08
    assert good["auroc"]["hard_neg"]["ci"][0] <= good["auroc"]["hard_neg"]["value"] <= good["auroc"]["hard_neg"]["ci"][1]
    # запись и обратное чтение
    path = M.write_summary(cfg, summary)
    assert path.exists() and (path.parent / M.SCHEMA_FILE).read_text(encoding="utf-8").startswith("# Схема")
    back = json.loads(path.read_text(encoding="utf-8"))
    assert back["methods"]["hybrid"]["tpr"]["0.01"]["conformal"]["overall"]["value"] == summary["methods"]["hybrid"]["tpr"]["0.01"]["conformal"]["overall"]["value"]
    assert back["data_stats"]["functions"]["splits"]["protected"]["n_final"] == 90
    assert back["data_stats"]["queries"]["protected"]["by_transform"]["partial"] == 160


def test_metrics_missing_sets_and_small_calibration(tmp_path: Path):
    """Нет hard_neg и public_test, мало калибровочных негативов для α=0.01 → порог ∞, метрики не падают."""
    cfg = _make_cfg(tmp_path)
    rng = np.random.default_rng(1)
    qdir = Path(cfg["paths"]["queries"])
    q_prot, q_cal = _make_queries("protected", 10, ["p"], rng), _make_queries("public_calib", 5, ["c"], rng)
    write_jsonl(qdir / "protected.jsonl", q_prot)
    write_jsonl(qdir / "public_calib.jsonl", q_cal)
    for s, rows in (("protected", q_prot), ("public_calib", q_cal)):
        write_jsonl(RE.scores_path(cfg, "exact", s), [{"qid": r["qid"], "score": float(rng.random()), "best_id": None, "latency_ms": None} for r in rows])
    summary = M.compute_summary(cfg)
    m = summary["methods"]["exact"]
    assert m["thresholds"]["0.01"]["conformal"] == "inf" and m["thresholds"]["0.01"]["oracle"] is None and m["thresholds"]["0.01"]["domain"] is None
    assert m["tpr"]["0.01"]["conformal"]["overall"]["value"] == 0.0  # порог ∞ → никого не ловим
    assert m["tpr"]["0.05"]["conformal"]["overall"]["n"] == 70 and "oracle" not in m["tpr"]["0.05"]
    assert m["auroc"] == {} and m["latency_ms"]["p50"] is None
    assert summary["domain_calibration"]["calib_repos"] == []
    M.write_summary(cfg, summary)
    assert json.loads((Path(cfg["paths"]["results"]) / "summary.json").read_text())["methods"]["exact"]["thresholds"]["0.01"]["conformal"] == "inf"


# ----------------------------------------------------------------------------- таблицы и рисунки


def test_tables_written(stand: dict[str, Any]):
    cfg = stand["cfg"]
    summary = M.load_summary(cfg) or M.compute_summary(cfg)
    paths = write_tables(cfg, summary)
    assert set(paths) == set(TABLE_FILES)  # T6 (privacy.json) и T7 (LLM) тоже есть
    t2 = paths["T2"].read_text(encoding="utf-8")
    assert "T3 rename_ids" in t2 and "M3 semantic" in t2 and "M4 hybrid" in t2 and "T7 partial (все L)" in t2
    assert re.search(r"\| 0\.\d{3} \[0\.\d{3}, 0\.\d{3}\] \|", t2)  # 3 знака, ДИ в скобках
    assert "Все классы, α = 1 %" in t2 and "Все классы, α = 5 %" in t2
    t3 = paths["T3"].read_text(encoding="utf-8")
    assert "FPR hard_neg [ДИ]" in t3 and "5 %" in t3 and "1 %" in t3 and "Доменная калибровка" in t3
    assert "кластерный ДИ" in t3 and "1 запрос/функция" in t3 and "Обменяемые единицы" in t3
    t5 = paths["T5"].read_text(encoding="utf-8")
    assert "Сквозная латентность" in t5
    t1 = paths["T1"].read_text(encoding="utf-8")
    assert "protected (P)" in t1 and "| 90 |" in t1 and "300" in t1
    t5 = paths["T5"].read_text(encoding="utf-8")
    assert "30.00 / 45.00" in t5 and "инференс энкодера" in t5 and "бенчмарк" in t5 and "CPU, 8 потоков" in t5
    enc_rows = [ln for ln in t5.splitlines() if ln.startswith("| — из них инференс энкодера")]
    assert len(enc_rows) == 2  # semantic (из latency.json components) и hybrid (из latency_encoder.json)
    assert "28.00 / 42.00" in enc_rows[0] and "40.00 / 55.00" in enc_rows[1] and "15.00 / 21.00" in enc_rows[1] and "800.0" in enc_rows[1]
    t6 = paths["T6"].read_text(encoding="utf-8")
    assert "Проекция Q" in t6 and "0.900 [0.880, 0.920]" in t6 and "A2, n = 10000: F1" in t6
    t7 = paths["T7"].read_text(encoding="utf-8")
    assert "T9 paraphrase" in t7 and "87.5" in t7 and "| все |" in t7 and "| go |" in t7
    assert fmt(None) == "—" and fmt("inf") == "∞" and fmt(0.123456) == "0.123" and fmt_ci({"value": 0.5, "ci": None}) == "0.500"


def test_figures_created(stand: dict[str, Any]):
    cfg = stand["cfg"]
    summary = M.load_summary(cfg) or M.compute_summary(cfg)
    paths = write_figures(cfg, summary)
    assert set(paths) == set(FIGURE_FILES)  # F5 (есть hybrid), F6 (privacy.json), F7 (latency.json)
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 10_000 and p.parent == Path(cfg["paths"]["figures"])


def test_figures_skip_gracefully(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    cfg = _make_cfg(tmp_path)
    summary = {"methods": {}, "alpha_keys": ["0.01"], "length_bins": [[0, 32], [32, 100000]]}
    with caplog.at_level("INFO", logger="smcode.eval.plots"):
        paths = write_figures(cfg, summary)
    assert paths == {} and sum("пропуск" in r.message for r in caplog.records) == len(FIGURE_FILES)


# ----------------------------------------------------------------------------- run_eval (07)


class DummyIndex:
    """Индекс-заглушка: score = доля букв 'x' в коде; latency через BaseIndex-подобный query_batch."""

    name = "dummy"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg, self.ids, self.loaded = cfg, ["a", "b"], None

    def load(self, path: Path) -> None:
        self.loaded = Path(path)

    @property
    def n_records(self) -> int:
        return len(self.ids)

    def memory_bytes(self) -> int:
        return 4321

    def query(self, code: str, lang: str) -> QueryResult:
        return QueryResult(score=code.count("x") / max(1, len(code)), best_id="a")

    def query_batch(self, items: list[tuple[str, str]]) -> list[QueryResult]:
        out = []
        for code, lang in items:
            r = self.query(code, lang)
            r.latency_ms = 0.5
            out.append(r)
        return out


class DummySemantic(DummyIndex):
    """numpy-API как у SemanticIndex: score = первая координата эмбеддинга."""

    def score_embeddings(self, q: np.ndarray, top_k: int | None = None):
        s = q[:, 0].astype(np.float64)
        return s, ["a"] * len(s), np.zeros((len(s), 1), int), s[:, None]


class DummyHybrid(DummyIndex):
    """numpy-API как у HybridIndex: (q_emb, q_codes, q_langs, top_k) → (scores, best_ids, details)."""

    def score_embeddings_with_codes(self, q: np.ndarray, q_codes: list[str], q_langs: list[str], top_k: int | None = None):
        assert len(q_codes) == len(q_langs) == q.shape[0] and all(l == "python" or l in LANGS for l in q_langs)
        scores = q[:, 0].astype(np.float32) * 0.5
        return scores, ["b"] * len(scores), [{"n_candidates": 1}] * len(scores)


def _prepare_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, classes: dict[str, type]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = _make_cfg(tmp_path)
    rng = np.random.default_rng(3)
    rows = _make_queries("public_test", 4, ["r"], rng)
    for i, r in enumerate(rows):
        r["code"] = "x" * (i % 5) + "y" * 5
    write_jsonl(Path(cfg["paths"]["queries"]) / "public_test.jsonl", rows)
    for name in classes:
        d = Path(cfg["paths"]["indexes"]) / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text("{}")
    monkeypatch.setattr(RE, "make_index", lambda name, cfg: classes[name](cfg))
    monkeypatch.setitem(RE.REGISTRY, "dummy", "tests:DummyIndex")
    return cfg, rows


def test_run_eval_query_batch_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg, rows = _prepare_eval(tmp_path, monkeypatch, {"exact": DummyIndex})
    res = RE.eval_method(cfg, "exact", sets=["public_test"], bench=3)
    meta = res["public_test"]
    assert meta["n_scored"] == len(rows) and meta["latency_mode"] == "per_query" and meta["memory_bytes"] == 4321 and meta["n_records"] == 2
    assert meta["latency_benchmark"]["n_queries"] == 3 and meta["latency_benchmark"]["repeats"] == cfg["eval"]["latency_repeats"]
    out = list(read_jsonl(RE.scores_path(cfg, "exact", "public_test")))
    assert [r["qid"] for r in out] == [r["qid"] for r in rows] and out[1]["score"] == pytest.approx(1 / 6) and out[0]["latency_ms"] == 0.5
    assert RE.read_scores_meta(cfg, "exact", "public_test")["latency_mode"] == "per_query"
    assert RE.eval_method(cfg, "exact", sets=["public_test"]) == {"public_test": {"skipped": True}}  # идемпотентность
    assert RE.eval_method(cfg, "exact", sets=["public_test"], force=True, limit=3)["public_test"]["n_scored"] == 3
    # --limit пишет отдельный файл и не трогает полный
    assert len(list(read_jsonl(RE.scores_path(cfg, "exact", "public_test")))) == len(rows)
    assert len(list(read_jsonl(RE.scores_path(cfg, "exact", "public_test", limit=3)))) == 3
    assert RE.scores_path(cfg, "exact", "public_test", limit=3).name == "public_test.limit3.jsonl"
    assert RE.read_scores_meta(cfg, "exact", "public_test", limit=3)["limit"] == 3 and RE.read_scores_meta(cfg, "exact", "public_test")["limit"] is None
    assert RE.eval_method(cfg, "exact", sets=["public_test"], limit=3) == {"public_test": {"skipped": True}}
    assert RE.available_methods(cfg) == ["exact"] and RE.available_sets(cfg) == ["public_test"]


def test_run_eval_numpy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg, rows = _prepare_eval(tmp_path, monkeypatch, {"semantic": DummySemantic, "hybrid": DummyHybrid})
    emb = np.random.default_rng(0).random((len(rows), 8)).astype(np.float16)
    p = RE.query_embeddings_path(cfg, "public_test")
    p.parent.mkdir(parents=True, exist_ok=True)
    # один запрос без эмбеддинга (запросы добавлены после 06): полный прогон — ошибка с указанием на 06, а не тихий пропуск
    np.savez(p, qids=np.asarray([r["qid"] for r in rows[:-1]]), emb=emb[:-1])
    with pytest.raises(ValueError, match="06_embed"):
        RE.run_eval(cfg, methods=["semantic"], sets=["public_test"], batch=4)
    assert not RE.scores_path(cfg, "semantic", "public_test").exists()
    sc, meta = RE.score_set(DummySemantic(cfg), rows, emb=([r["qid"] for r in rows[:-1]], emb[:-1].astype(np.float32)), cfg=cfg, batch=4)
    assert meta["n_missing_embeddings"] == 1 and len(sc) == len(rows) - 1  # score_set сам лишь считает пропуски
    np.savez(p, qids=np.asarray([r["qid"] for r in rows]), emb=emb)
    res = RE.run_eval(cfg, methods=["semantic", "hybrid"], sets=["public_test"], batch=4, bench=2)
    for m, k in (("semantic", 1.0), ("hybrid", 0.5)):
        meta = res[m]["public_test"]
        assert meta["latency_mode"] == "batch_amortized(batch=4)" and meta["n_scored"] == len(rows) and meta["n_missing_embeddings"] == 0
        assert meta["latency_benchmark"]["mode"] == "numpy_batch1"
        out = list(read_jsonl(RE.scores_path(cfg, m, "public_test")))
        assert [r["score"] for r in out] == pytest.approx([float(v) * k for v in emb[:, 0].astype(np.float32)], abs=1e-6)
        assert all(r["latency_ms"] >= 0 for r in out)
    assert list(read_jsonl(RE.scores_path(cfg, "hybrid", "public_test")))[0]["best_id"] == "b"


def test_run_eval_import_error_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg, _ = _prepare_eval(tmp_path, monkeypatch, {"exact": DummyIndex})

    def _boom(name: str, cfg: dict[str, Any]):
        raise ImportError("no torch")

    monkeypatch.setattr(RE, "make_index", _boom)
    assert "ImportError" in RE.run_eval(cfg, methods=["exact"], sets=["public_test"])["exact"]["error"]


# ----------------------------------------------------------------------------- отчёт и скрипты


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_report_and_scripts(stand: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    cfg = stand["cfg"]
    res = run_report(cfg, force=True, bootstrap=20, methods=["semantic", "winnowing"])
    assert set(res["summary"]["methods"]) == {"semantic", "winnowing"} and res["summary"]["bootstrap"] == 20
    assert "T2" in res["tables"] and "F1" in res["figures"] and "F5" not in res["figures"]  # без hybrid — F5 пропущен
    text = format_summary(res["summary"], res["tables"], res["figures"])
    assert "M3 semantic" in text and "TPR" in text and "Таблицы" in text
    # повторный запуск без force: набор каталогов скоров (4) ≠ summary.methods (2) → пересчёт
    res2 = run_report(cfg, force=False, figures=False)
    assert res2["recomputed"] and set(res2["summary"]["methods"]) == set(METHOD_DISTS)
    # третий запуск: summary актуален → переиспользование
    res3 = run_report(cfg, force=False, figures=False, tables=False)
    assert not res3["recomputed"] and res3["summary"]["generated_at"] == res2["summary"]["generated_at"]
    # скрипт 09 с YAML-конфигом
    cfg_path = stand["tmp"] / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, allow_unicode=True), encoding="utf-8")
    assert _load_script("09_report").main(["--config", str(cfg_path), "--force", "--bootstrap", "10", "--no-figures"]) == 0
    out = capsys.readouterr().out
    assert "Сводка оценки" in out and "M4 hybrid" in out and "α = 1 %" in out
    assert set(json.loads((Path(cfg["paths"]["results"]) / "summary.json").read_text())["methods"]) == set(METHOD_DISTS)
    # скрипт 07 с индексом-заглушкой
    (Path(cfg["paths"]["indexes"]) / "exact").mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["indexes"]) / "exact" / "meta.json").write_text("{}")
    monkeypatch.setattr(RE, "make_index", lambda name, c: DummyIndex(c))
    assert _load_script("07_eval").main(["--config", str(cfg_path), "--methods", "exact", "--sets", "hard_neg", "--limit", "7"]) == 0
    assert json.loads(capsys.readouterr().out)["exact"]["hard_neg"] == 7
    assert not RE.scores_path(cfg, "exact", "hard_neg").exists()  # --limit не пишет полный файл
    assert len(list(read_jsonl(RE.scores_path(cfg, "exact", "hard_neg", limit=7)))) == 7
    assert "exact" not in M.available_methods(cfg)  # только усечённые файлы → метод не считается готовым


# ----------------------------------------------------------------------------- регрессии после ревью


def _scored(set_name: str, score: np.ndarray, source_id: np.ndarray, lang: str = "c") -> M.ScoredSet:
    n = score.size
    obj = lambda v: np.asarray([v] * n, dtype=object)  # noqa: E731
    return M.ScoredSet(name=set_name, label=0, qid=source_id.copy(), score=score, latency=np.full(n, np.nan), transform=obj("identity"),
                       lang=obj(lang), n_tokens=np.full(n, 10), L=np.full(n, -1), repo=obj("r"), source_id=source_id,
                       dst_lang=obj(""), best_id=obj(""), n_queries=n)


def test_fpr_cluster_ci_accounts_for_clustering():
    """Превышения порога кластеризованы по функции (все 13 запросов функции) → кластерный ДИ заметно шире Клоппера–Пирсона."""
    rng = np.random.default_rng(7)
    n_fun, per = 400, 13
    sid = np.repeat([f"f{i}" for i in range(n_fun)], per).astype(object)
    hot = rng.random(n_fun) < 0.02
    ss = _scored("public_test", np.where(np.repeat(hot, per), 0.9, 0.1), sid)
    boot = M.ClusterBootstrap(ss.source_id, 300, np.random.default_rng(1))
    blk = M.fpr_block(ss, 0.5, boot=boot)
    cp_w, cl_w = blk["ci"][1] - blk["ci"][0], blk["ci_cluster"][1] - blk["ci_cluster"][0]
    assert cl_w > 2 * cp_w and blk["ci_cluster"][0] <= blk["value"] <= blk["ci_cluster"][1]
    assert "ci_cluster" in blk["max_subgroup"] and blk["by_lang"]["c"]["ci_cluster"] is not None
    assert M.fpr_block(ss, 0.5)["ci_cluster"] is None  # без бутстрэпа — только Клоппер–Пирсон
    assert fmt_ci_cluster(blk) != fmt_ci(blk) and fmt_ci_cluster({"value": 0.5, "ci": [0.4, 0.6]}) == "0.500 [0.400, 0.600]"


def test_one_per_cluster_and_tie_mass():
    sid = np.asarray(["a", "b", "a", "c", "b", "a"], dtype=object)
    mask = M.one_per_cluster(sid, np.random.default_rng(0))
    assert mask.sum() == 3 and sorted(sid[mask].tolist()) == ["a", "b", "c"]
    assert np.array_equal(mask, M.one_per_cluster(sid, np.random.default_rng(0)))  # детерминизм
    assert M.tie_mass(np.array([0.0, 0.0, 0.0, 0.5]), 0.0) == 0.75 and M.tie_mass(np.array([0.1]), float("inf")) is None


def test_end_to_end_latency_composition():
    lat = {"p50": 1.0, "p95": 2.0}
    assert M.end_to_end_latency(lat, None, ["per_query"], None)["p95"] == 2.0
    e = M.end_to_end_latency({"p50": 0.001, "p95": 0.0015}, None, ["batch_amortized(batch=256)"], {})
    assert e["p95"] is None and "latency_encoder" in e["reason"]
    enc = {"cpu1": {"p50_ms": 40.0, "p95_ms": 55.0}}
    e = M.end_to_end_latency({"p50": 0.001, "p95": 0.0015}, None, ["batch_amortized(batch=256)"], enc)
    assert e["p95"] is None and "--bench" in e["reason"]
    bench = {"mode": "numpy_batch1", "p50_ms": 3.0, "p95_ms": 5.0}
    e = M.end_to_end_latency({"p50": 0.001, "p95": 0.0015}, bench, ["batch_amortized(batch=256)"], enc)
    assert e["p95"] == 60.0 and e["p50"] == 43.0 and e["components"]["encoder"]["p95"] == 55.0
    e = M.end_to_end_latency(lat, {"mode": "query", "p50_ms": 30.0, "p95_ms": 42.0}, ["amortized_batch(batch=64)"], None)
    assert e["p95"] == 42.0 and "query()" in e["source"]
    assert M.end_to_end_latency(lat, bench, ["amortized_batch(batch=64)"], enc)["p95"] is None  # numpy-бенчмарк не содержит энкодер


def test_f7_skips_amortized_method_without_end_to_end(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """F7 не рисует амортизированное время батча как латентность запроса: без сквозной оценки метод пропускается."""
    base = {"n_records": 1000, "latency_ms": {"p50": 0.001, "p95": 0.0015, "mode": "batch_amortized(batch=256)",
                                              "end_to_end": {"p50": None, "p95": None, "source": None, "components": {}, "reason": "нет бенчмарка"}}}
    summary = {"methods": {"semantic": base}, "alpha_keys": ["0.01"]}
    assert latency_point(summary, "semantic") is None
    with caplog.at_level("WARNING", logger="smcode.eval.plots"):
        assert fig_F7(summary, None, tmp_path / "f7.png") is None
    assert any("F7" in r.message and "semantic" in r.message for r in caplog.records)
    ok = {**base, "latency_ms": {**base["latency_ms"], "end_to_end": {"p50": 43.0, "p95": 60.0, "source": "x", "components": {}, "reason": None}}}
    summary["methods"]["semantic"] = ok
    assert latency_point(summary, "semantic") == (1000.0, 60.0, "энкодер CPU-1 + поиск, батч 1")
    assert fig_F7(summary, None, tmp_path / "f7.png") is not None and (tmp_path / "f7.png").exists()
    pq = {"n_records": 500, "latency_ms": {"p50": 1.0, "p95": 2.0, "mode": "per_query", "end_to_end": {"p50": 1.0, "p95": 2.0, "source": "per_query", "components": {}, "reason": None}}}
    assert latency_point({"methods": {"exact": pq}}, "exact") == (500.0, 2.0, "прогон eval")


def test_f6_zero_f1_point_is_kept():
    defs = [{"name": "none", "tpr": {"semantic": 0.8}, "attacks": {"A2": {"100": {"f1": 0.8}, "10000": {"f1": 0.9}}}},
            {"name": "strong", "tpr": {"semantic": {"value": 0.7}}, "attacks": {"A2": {"100": {"f1": 0.0}, "10000": {"f1": 0.0}}}},
            {"name": "no-a2", "tpr": {"hybrid": 0.75}, "attacks": {"A2": None}}]
    xs, ys = privacy_points(defs, "10000")
    assert xs.tolist()[:2] == [0.9, 0.0] and np.isnan(xs[2]) and ys.tolist() == [0.8, 0.7, 0.75]


class DummyAmortized(DummyIndex):
    """query_batch амортизирует батч, как SemanticIndex/HybridIndex (details["latency_mode"] = "amortized_batch")."""

    query_batch_size = 8

    def query_batch(self, items: list[tuple[str, str]]) -> list[QueryResult]:
        out = []
        for code, lang in items:
            r = self.query(code, lang)
            r.latency_ms, r.details["latency_mode"] = 0.1, "amortized_batch"
            out.append(r)
        return out


def test_amortized_batch_mode_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg, rows = _prepare_eval(tmp_path, monkeypatch, {"semantic": DummyAmortized})
    meta = RE.eval_method(cfg, "semantic", sets=["public_test"], bench=2)["public_test"]
    assert meta["latency_mode"] == "amortized_batch(batch=8)" and meta["latency_benchmark"]["mode"] == "query"
    summary = M.compute_summary(cfg, bootstrap=5)
    lat = summary["methods"]["semantic"]["latency_ms"]
    assert lat["mode"] == "amortized_batch(batch=8)" and lat["end_to_end"]["p95"] == meta["latency_benchmark"]["p95_ms"]


def test_numpy_path_end_to_end_needs_encoder_and_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg, rows = _prepare_eval(tmp_path, monkeypatch, {"semantic": DummySemantic})
    emb = np.random.default_rng(0).random((len(rows), 8)).astype(np.float16)
    p = RE.query_embeddings_path(cfg, "public_test")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, qids=np.asarray([r["qid"] for r in rows]), emb=emb)
    RE.eval_method(cfg, "semantic", sets=["public_test"], batch=4)
    e = M.compute_summary(cfg, bootstrap=5)["methods"]["semantic"]["latency_ms"]["end_to_end"]
    assert e["p95"] is None and e["reason"]
    with open(Path(cfg["paths"]["results"]) / "latency_encoder.json", "w") as f:
        json.dump({"cpu_threads_all": 4, "configs": [{"device": "cpu", "threads": 1, "batch": 1, "p50_ms": 40.0, "p95_ms": 55.0, "per_item_ms": 40.0}]}, f)
    RE.eval_method(cfg, "semantic", sets=["public_test"], batch=4, bench=2, force=True)
    lat = M.compute_summary(cfg, bootstrap=5)["methods"]["semantic"]["latency_ms"]
    assert lat["benchmark"]["mode"] == "numpy_batch1" and lat["end_to_end"]["p95"] == pytest.approx(55.0 + lat["benchmark"]["p95_ms"])
    assert lat["end_to_end"]["components"]["encoder"]["p95"] == 55.0


def test_variant_eval_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Абляция: out_name / index_path / cfg_overrides → отдельный каталог скоров, meta с описанием, подхват в summary/T4."""
    cfg, rows = _prepare_eval(tmp_path, monkeypatch, {"hybrid": DummyHybrid})
    alt = tmp_path / "alt_index"
    alt.mkdir()
    ov = {"semantic": {"hybrid_rule": "two_threshold"}}
    res = RE.run_eval(cfg, methods=["hybrid"], sets=["public_test"], out_name="hybrid_rule", index_path=alt, cfg_overrides=ov)
    meta = res["hybrid"]["public_test"]
    assert meta["variant"] == "hybrid_rule" and meta["method"] == "hybrid" and meta["index_dir"] == str(alt) and meta["cfg_overrides"] == ov
    assert RE.scores_path(cfg, "hybrid_rule", "public_test").exists() and not RE.scores_path(cfg, "hybrid", "public_test").exists()
    assert cfg["semantic"]["hybrid_rule"] == "logistic"  # исходный конфиг не изменён
    assert RE.eval_method(cfg, "hybrid", sets=["public_test"], out_name="hybrid_rule") == {"public_test": {"skipped": True}}
    with pytest.raises(ValueError):
        RE.run_eval(cfg, methods=["hybrid", "exact"], out_name="x")
    summary = M.compute_summary(cfg, bootstrap=5)
    assert list(summary["methods"]) == ["hybrid_rule"]
    v = summary["methods"]["hybrid_rule"]["variant"]
    assert v["base_method"] == "hybrid" and v["cfg_overrides"] == ov and v["index_dir"] == str(alt)
    assert "semantic.hybrid_rule=two_threshold" in variant_description(summary, "hybrid_rule") and "M4 hybrid" in variant_description(summary, "hybrid_rule")
    assert "M4 правило двух порогов" in table_T4(summary) and "semantic.hybrid_rule=two_threshold" in table_T4(summary)
    assert RE.parse_overrides(["a.b=1", "a.c=x y", "d={k: 0.5}", "e="]) == {"a": {"b": 1, "c": "x y"}, "d": {"k": 0.5}, "e": None}
    with pytest.raises(ValueError):
        RE.parse_overrides(["novalue"])


def test_variant_eval_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    cfg, rows = _prepare_eval(tmp_path, monkeypatch, {"exact": DummyIndex})
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, allow_unicode=True), encoding="utf-8")
    alt = tmp_path / "alt"
    alt.mkdir()
    main = _load_script("07_eval").main
    assert main(["--config", str(cfg_path), "--methods", "exact", "--variant", "exact_w3", "--index-dir", str(alt),
                 "--set", "fingerprint.exact_window=3", "--sets", "public_test"]) == 0
    assert json.loads(capsys.readouterr().out) == {"exact_w3": {"public_test": len(rows)}}
    meta = RE.read_scores_meta(cfg, "exact_w3", "public_test")
    assert meta["cfg_overrides"] == {"fingerprint": {"exact_window": 3}} and meta["index_dir"] == str(alt)
    with pytest.raises(SystemExit):  # --variant с двумя методами
        main(["--config", str(cfg_path), "--methods", "exact,winnowing", "--variant", "x"])
    with pytest.raises(SystemExit):  # имя варианта не из REGISTRY
        main(["--config", str(cfg_path), "--methods", "exact_w3"])


def test_report_recomputes_on_stale_summary(tmp_path: Path):
    """Новый каталог скоров или более новый файл скоров → summary.json пересчитывается без --force."""
    cfg = _make_cfg(tmp_path)
    rng = np.random.default_rng(4)
    qdir = Path(cfg["paths"]["queries"])
    queries = {s: _make_queries(s, 12, ["a", "b"], rng) for s in ("protected", "public_calib", "public_test")}
    for s, q in queries.items():
        write_jsonl(qdir / f"{s}.jsonl", q)

    def _scores(m: str) -> None:
        for s, q in queries.items():
            d = (5.0, 2.0) if s == "protected" else (2.0, 5.0)
            write_jsonl(RE.scores_path(cfg, m, s), [{"qid": r["qid"], "score": float(rng.beta(*d)), "best_id": None, "latency_ms": 1.0} for r in q])

    _scores("winnowing")
    res = run_report(cfg, bootstrap=5, figures=False, tables=False)
    assert res["recomputed"]
    res = run_report(cfg, figures=False, tables=False)
    assert not res["recomputed"] and summary_is_stale(cfg, M.load_summary(cfg), res["summary_path"]) is None
    _scores("hybrid_rule")  # новый вариант появился после отчёта
    res = run_report(cfg, figures=False, tables=False)
    assert res["recomputed"] and set(res["summary"]["methods"]) == {"winnowing", "hybrid_rule"}
    import os, time
    sp = RE.scores_path(cfg, "winnowing", "protected")
    os.utime(sp, (time.time() + 5, time.time() + 5))  # скоры «новее» summary.json
    res = run_report(cfg, figures=False, tables=False)
    assert res["recomputed"] and "новее" in res["reason"]
    res = run_report(cfg, methods=["winnowing"], figures=False, tables=False)
    assert res["recomputed"] and list(res["summary"]["methods"]) == ["winnowing"]


def test_incomplete_scores_are_flagged(tmp_path: Path):
    """Полный файл скоров, усечённый старым --limit (meta.limit) → пересчёт в eval, пометка incomplete в summary и таблицах."""
    cfg = _make_cfg(tmp_path)
    rng = np.random.default_rng(5)
    qdir = Path(cfg["paths"]["queries"])
    q = {s: _make_queries(s, 10, ["a"], rng) for s in ("protected", "public_calib")}
    for s, rows in q.items():
        write_jsonl(qdir / f"{s}.jsonl", rows)
        write_jsonl(RE.scores_path(cfg, "exact", s), [{"qid": r["qid"], "score": float(rng.random()), "best_id": None, "latency_ms": None} for r in rows[:7]])
        with open(RE.meta_path(cfg, "exact", s), "w") as f:
            json.dump({"limit": 7, "latency_mode": "per_query"}, f)
    assert not RE._is_done(cfg, "exact", "protected", None)
    summary = M.compute_summary(cfg, bootstrap=5)
    inc = summary["methods"]["exact"]["incomplete"]
    assert inc["protected"] == {"n": 7, "n_queries": 70} and summary["warnings"] and "exact" in summary["warnings"][0]
    assert "Неполные скоры" in write_tables(cfg, summary)["T2"].read_text(encoding="utf-8")
    assert "⚠" in format_summary(summary)
