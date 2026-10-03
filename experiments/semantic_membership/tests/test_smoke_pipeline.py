"""Smoke-тест конвейера в одном процессе (DESIGN.md §11): синтетические репозитории → 01 → 02 → 03 → 04 → 07 → 09.

То же, что scripts/smoke_test.sh, но через main() скриптов; CPU, без сети и без torch."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from smcode.types import read_jsonl
from synthetic_repos import REPOS, SMOKE_CONFIG_NAME, make_source_file, prepare_smoke_workdir, smoke_config

ROOT = Path(__file__).resolve().parents[1]
FP_METHODS = ("exact", "winnowing", "minhash")
SETS = ("protected", "hard_neg", "public_calib", "public_test", "protected_windows")
TABLES = ("T1_data_stats", "T2_tpr_by_transform", "T3_fpr_guarantee", "T4_ablation", "T5_latency_memory")
FIGURES = ("F1_tpr_by_transform", "F2_tpr_vs_length", "F3_roc_hardneg", "F4_fpr_claimed_vs_empirical", "F7_latency_vs_index")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"smoke_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_generated_sources_parse():
    """Каждый сгенерированный файл разбирается tree-sitter без ошибок (иначе transforms отбрасывали бы запросы)."""
    import random

    from smcode.transforms.programmatic import parses_ok

    for lang in ("python", "c", "go", "javascript"):
        for kind in ("secure", "web"):
            text = make_source_file(lang, kind, f"{kind}_{lang[:2]}_core", random.Random(f"t:{lang}:{kind}"), 12)
            assert parses_ok(text, lang), (lang, kind)


def test_smoke_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("SMCODE_INDEX_KEY", "smoke-test-key")
    work = tmp_path / "smoke"
    cfg_path = prepare_smoke_workdir(work)
    assert cfg_path == work / SMOKE_CONFIG_NAME and cfg_path.exists()
    for split, (dirname, _) in REPOS.items():
        assert (work / "raw" / split / dirname / ".cloned").exists()
    cfg = smoke_config(work)
    assert Path(cfg["paths"]["functions"]) == work / "functions"
    args = ["--config", str(cfg_path)]

    # 01 extract -> 02 dedup
    assert _load_script("01_extract").main(args + ["--workers", "1"]) == 0
    for s in ("protected", "hard_neg", "public_train", "public_calib", "public_test"):
        assert (work / "functions" / f"{s}.jsonl").stat().st_size > 0, s
    assert (work / "functions" / "protected_windows.jsonl").stat().st_size > 0
    assert _load_script("02_dedup_split").main(args + ["--workers", "1"]) == 0
    stats = json.loads((work / "functions" / "stats.json").read_text(encoding="utf-8"))
    sp = stats["splits"]
    assert sp["protected"]["n_final"] >= 40 and set(sp["protected"]["by_lang"]) == {"python", "c", "go", "javascript"}
    # синтетические репозитории структурно различны: near-dup фильтр не вычищает негативы
    for s in ("hard_neg", "public_train", "public_calib", "public_test"):
        assert sp[s]["n_final"] >= 0.8 * sp[s]["n_raw"], (s, sp[s])

    # 03 queries
    assert _load_script("03_build_queries").main(args) == 0
    for s in SETS:
        rows = list(read_jsonl(work / "queries" / f"{s}.jsonl"))
        assert rows, s
        assert all(r["label"] == (1 if s.startswith("protected") else 0) for r in rows)
    qstats = json.loads((work / "queries" / "protected.stats.json").read_text(encoding="utf-8"))
    assert qstats["per_transform"]["identity"]["ok"] == qstats["n_sampled"] == cfg["transforms"]["max_per_split"]
    assert qstats["per_transform"]["rename_ids"]["ok"] >= 0.8 * qstats["n_sampled"]

    # 04 indexes (fingerprint methods only; semantic/hybrid need torch)
    assert _load_script("04_build_indexes").main(args + ["--methods", ",".join(FP_METHODS), "--with-windows", "--workers", "1"]) == 0
    for m in FP_METHODS:
        meta = json.loads((work / "indexes" / m / "meta.json").read_text(encoding="utf-8"))
        assert meta["with_windows"] is True and meta["keyed"] is True and meta["n_records"] > sp["protected"]["n_final"]
    istats = json.loads((work / "results" / "index_stats.json").read_text(encoding="utf-8"))
    assert set(istats) == set(FP_METHODS)
    capsys.readouterr()

    # 07 eval (all methods with an index, all sets)
    assert _load_script("07_eval").main(args + ["--bench", "3"]) == 0
    for m in FP_METHODS:
        for s in SETS:
            rows = list(read_jsonl(work / "results" / "scores" / m / f"{s}.jsonl"))
            assert rows and {"qid", "score", "best_id", "latency_ms"} <= set(rows[0]), (m, s)
        meta = json.loads((work / "results" / "scores" / m / "protected.meta.json").read_text(encoding="utf-8"))
        assert meta["latency_mode"] == "per_query" and meta["latency_benchmark"]["n_queries"] == 3
    capsys.readouterr()

    # 09 report
    assert _load_script("09_report").main(args) == 0
    out = capsys.readouterr().out
    assert "Сводка оценки" in out and "M0 exact" in out and "M1 winnowing" in out
    summary = json.loads((work / "results" / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["methods"]) == set(FP_METHODS) and summary["bootstrap"] == cfg["eval"]["bootstrap"]
    assert summary["data_stats"]["functions"]["splits"]["protected"]["n_final"] == sp["protected"]["n_final"]
    for t in TABLES:
        assert (work / "results" / "tables" / f"{t}.md").stat().st_size > 0, t
    for f in FIGURES:
        assert (work / "figures" / f"{f}.png").stat().st_size > 0, f
    ak = summary["alpha_keys"][0]
    exact = summary["methods"]["exact"]
    assert exact["tpr"][ak]["conformal"]["by_transform"]["identity"]["value"] == 1.0
    assert exact["fpr"][ak]["conformal"]["public_test"]["value"] <= 0.1
    assert summary["methods"]["winnowing"]["tpr"][ak]["conformal"]["overall"]["value"] >= 0.8
    assert summary["methods"]["winnowing"]["auroc"]["hard_neg"]["value"] >= 0.9

    # идемпотентность: повторный 07/09 ничего не пересчитывает (summary.json не перезаписывается)
    assert _load_script("07_eval").main(args) == 0
    assert all(v == "skipped" for v in json.loads(capsys.readouterr().out)["exact"].values())
    mtime = (work / "results" / "summary.json").stat().st_mtime_ns
    assert _load_script("09_report").main(args) == 0
    assert (work / "results" / "summary.json").stat().st_mtime_ns == mtime
