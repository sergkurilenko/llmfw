"""Регрессии по итогам финального ревью стенда (CPU, без сети и torch).

Покрывает: отдельные шаблоны insert_deadcode для оценки и обучения; наборы окон негативов (<split>_windows);
fingerprint.token_mode (lexical ломает rename_ids, full — нет); пересборку индекса при обновлении входов и
build_seconds/disk_bytes в meta.json; переиспользование protected_windows.npz в SemanticIndex.build; дозапись
недостающих строк в 06; свежесть файлов скоров/summary; мондриановский порог, кластерный z-тест и пропуски по
причинам; окна в method_metrics/T3; статистику вариантного индекса; состав индекса/проверку модели/полезность M4 в
приватности; маску ложных негативов и параметры обучения train.py.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from smcode.config import load_config
from smcode.eval import metrics as M
from smcode.eval import run_eval as RE
from smcode.eval.build_queries import (QUERY_SETS, WINDOW_SET, all_sets, deadcode_template_set, functions_path, is_window_set,
                                       label_for_set, queries_for_record, source_split, window_sets)
from smcode.eval.report import summary_is_stale
from smcode.eval.tables import misses_cell, verdict_cell, write_tables
from smcode.fingerprint.index import build_from_corpus, index_dir, index_inputs, load_index
from smcode.fingerprint.minhash import MinHashIndex
from smcode.fingerprint.winnowing_index import WinnowingIndex, token_mode_of
from smcode.semantic.ann import l2_normalize
from smcode.semantic.embed import default_query_sets, embed_query_set, load_embeddings, save_embeddings, windows_split
from smcode.semantic.semantic_index import SemanticIndex
from smcode.transforms.evasion import DEADCODE_TEMPLATES, DEADCODE_TEMPLATES_EVAL, TEMPLATE_SETS
from smcode.transforms.programmatic import parses_ok
from smcode.transforms.registry import apply_transform, select_transforms, training_transforms
from smcode.types import FunctionRecord, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SRC = {
    "python": "def f(a, b):\n    x = a + 1\n    y = b * 2\n    if x > y:\n        return x\n    return y\n",
    "c": "int f(int a, int b) {\n    int x = a + 1;\n    int y = b * 2;\n    if (x > y) {\n        return x;\n    }\n    return y;\n}\n",
    "cpp": "int f(int a, int b) {\n    int x = a + 1;\n    int y = b * 2;\n    if (x > y) {\n        return x;\n    }\n    return y;\n}\n",
    "go": "func f(a int, b int) int {\n\tx := a + 1\n\ty := b * 2\n\tif x > y {\n\t\treturn x\n\t}\n\treturn y\n}\n",
    "java": "class A {\n    int f(int a, int b) {\n        int x = a + 1;\n        int y = b * 2;\n        if (x > y) {\n            return x;\n        }\n        return y;\n    }\n}\n",
    "javascript": "function f(a, b) {\n    let x = a + 1;\n    let y = b * 2;\n    if (x > y) {\n        return x;\n    }\n    return y;\n}\n",
}


def _py_fn(i: int, n: int = 14) -> str:
    rng = random.Random(1000 + i)
    ids = [f"value{i}_{j}" for j in range(4)]
    lines = [f"def compute_{i}(alpha, beta, gamma):"]
    for _ in range(n):
        x, y = rng.choice(ids), rng.choice(ids + ["alpha", "beta", "gamma"])
        op = rng.choice(["+", "-", "*", "//", "%"])
        lines.append(f"    {x} = {y} {op} {rng.randint(1, 99)}")
    lines.append(f"    return {rng.choice(ids)}")
    return "\n".join(lines) + "\n"


def _records(n: int, split: str = "protected", kind: str = "function") -> list[FunctionRecord]:
    out = []
    for i in range(n):
        code = _py_fn(i)
        rid = f"{split}/org__repo/f{i}.py:1-{code.count(chr(10))}" if kind == "function" else f"{split}/org__repo/f{i}.py:w{i * 10 + 1}-{i * 10 + 20}"
        out.append(FunctionRecord(id=rid, split=split, repo="org/repo", path=f"f{i // 3}.py", lang="python", code=code, start_line=1,
                                  end_line=code.count("\n"), n_lines=code.count("\n"), n_tokens=len(code.split()), sha=f"sha{kind}{i}", kind=kind))
    return out


def _cfg(tmp_path: Path, **over: Any) -> dict[str, Any]:
    paths = {k: str(tmp_path / v) for k, v in {"raw_repos": "data/raw", "functions": "data/functions", "queries": "data/queries",
                                                "indexes": "data/indexes", "results": "results", "figures": "figures",
                                                "embeddings": "data/embeddings", "runs": "runs"}.items()}
    cfg = load_config(overrides={"paths": paths, "fingerprint": {"k": 5, "w": 4, "workers": 1}, "eval": {"bootstrap": 20},
                                 "calibration": {"alphas": [0.05]}})
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    return cfg


class _StubEncoder:
    """Энкодер-заглушка с именем модели; encode считает вызовы (или падает при fail=True)."""

    def __init__(self, model_name: str, d: int = 8, fail: bool = False, max_length: int = 512):
        self.model_name, self.max_length, self.device, self.kind, self.d, self.fail = model_name, max_length, "cpu", "fake", d, fail
        self.calls: list[list[str]] = []

    def encode(self, codes, langs=None, batch_size=32, show_progress=False):
        if self.fail:
            raise AssertionError(f"encoder called for {len(codes)} texts")
        self.calls.append(list(codes))
        base = np.arange(self.d, dtype=np.float32)[None, :] + np.array([len(c) for c in codes], dtype=np.float32)[:, None] * 0.01
        return l2_normalize(base + 0.5)


# ----------------------------------------------------------------------------- преобразования: шаблоны eval/train, отбор


def test_deadcode_template_sets_are_distinct_and_parse():
    assert set(TEMPLATE_SETS) == {"train", "eval"} and set(DEADCODE_TEMPLATES_EVAL) == set(DEADCODE_TEMPLATES)
    for lang, code in SRC.items():
        assert not set(DEADCODE_TEMPLATES_EVAL[lang]) & set(DEADCODE_TEMPLATES[lang])
        out = {}
        for tset in ("train", "eval"):
            res = apply_transform("insert_deadcode", code, lang, random.Random(3), every=1, template_set=tset)
            assert res.ok and parses_ok(res.code, lang) and res.params["template_set"] == tset, (lang, tset)
            out[tset] = res.code
        assert out["train"] != out["eval"], lang
    assert not apply_transform("insert_deadcode", SRC["python"], "python", random.Random(1), template_set="bogus").ok


def test_queries_use_eval_templates_and_training_uses_train(tmp_path):
    cfg = _cfg(tmp_path)
    assert deadcode_template_set(cfg) == "eval"
    rec = _records(1)[0]
    qs = {q.transform: q for q in queries_for_record(rec, "protected", cfg, transforms=["insert_deadcode", "rename_ids"], partial_lines=[])}
    assert qs["insert_deadcode"].params["template_set"] == "eval"
    qt = {q.transform: q for q in queries_for_record(rec, "combiner1", cfg, transforms=["insert_deadcode"], partial_lines=[], template_set="train")}
    assert qt["insert_deadcode"].params["template_set"] == "train" and qt["insert_deadcode"].code != qs["insert_deadcode"].code
    styled = [q for q in queries_for_record(rec, "x", cfg, transforms=["rename_ids"], partial_lines=[], rename_styles=["vn"])]
    assert styled and styled[0].params["style"] == "vn"


def test_training_transforms_exclusion(tmp_path):
    cfg = _cfg(tmp_path)
    names, partial, styles = training_transforms(cfg)
    assert names == cfg["transforms"]["programmatic"] and partial == cfg["transforms"]["partial_lines"] and styles is None
    cfg["semantic"].update({"exclude_transforms": ["insert_deadcode", "reorder_stmts", "partial"], "rename_styles": ["vn", "camel"]})
    names, partial, styles = training_transforms(cfg)
    assert "insert_deadcode" not in names and "reorder_stmts" not in names and partial == [] and styles == ["vn", "camel"]
    cfg["semantic"]["train_transforms"] = ["identity", "rename_ids", "insert_deadcode"]
    assert training_transforms(cfg)[0] == ["identity", "rename_ids"]
    with pytest.raises(KeyError):
        select_transforms(["identity"], ["no_such_transform"])


# ----------------------------------------------------------------------------- окна негативов


def test_window_sets_from_config(tmp_path):
    from smcode.data import extract

    cfg = _cfg(tmp_path)
    assert window_sets(cfg) == ["protected_windows", "public_calib_windows", "public_test_windows"]
    assert all_sets(cfg)[: len(QUERY_SETS)] == list(QUERY_SETS) and all_sets(cfg)[len(QUERY_SETS)] == WINDOW_SET
    assert is_window_set("public_test_windows") and not is_window_set("public_test")
    assert source_split("public_calib_windows") == "public_calib" and source_split("protected") == "protected"
    assert label_for_set("public_calib_windows") == 0 and label_for_set("protected_windows") == 1
    assert extract.windows_splits(cfg) == ["protected", "public_calib", "public_test"] and extract.windows_file("public_test") == "public_test_windows.jsonl"
    with pytest.raises(ValueError):
        extract.windows_splits({"windows": {"splits": ["protected", "nope"]}})
    assert windows_split("public_calib") == "public_calib_windows" == windows_split("public_calib_windows")
    assert default_query_sets(cfg)[-2:] == ["public_calib_windows", "public_test_windows"]
    fdir = Path(cfg["paths"]["functions"])
    fdir.mkdir(parents=True)
    assert functions_path(cfg, "public_calib_windows") == fdir / "public_calib.jsonl"  # окон нет → откат на функции (kind == window)
    write_jsonl(fdir / "public_calib_windows.jsonl", _records(2, "public_calib", "window"))
    assert functions_path(cfg, "public_calib_windows") == fdir / "public_calib_windows.jsonl"


# ----------------------------------------------------------------------------- token_mode


def test_token_mode_lexical_is_broken_by_rename_but_full_is_not(tmp_path):
    recs = _records(6)
    cfg_full, cfg_lex = _cfg(tmp_path), _cfg(tmp_path, fingerprint={"token_mode": "lexical"})
    assert token_mode_of(cfg_full) == "full" and token_mode_of(cfg_lex) == "lexical"
    with pytest.raises(ValueError):
        token_mode_of({"fingerprint": {"token_mode": "weird"}})
    renamed = apply_transform("rename_ids", recs[0].code, "python", random.Random(5))
    assert renamed.ok and renamed.code != recs[0].code
    scores = {}
    for name, cfg in (("full", cfg_full), ("lexical", cfg_lex)):
        w = WinnowingIndex(cfg)
        w.build(recs, cfg)
        assert w.token_mode == name and w.meta["token_mode"] == name
        scores[name] = w.query(renamed.code, "python").score
        m = MinHashIndex(cfg)
        m.build(recs, cfg)
        assert m.token_mode == name
        scores[name + "_mh"] = m.query(renamed.code, "python").score
        out = tmp_path / f"idx_{name}"
        w.save(out)
        m.save(out / "mh")
        loaded = WinnowingIndex(cfg_full if name == "lexical" else cfg_lex)  # режим индекса важнее конфига
        loaded.load(out)
        assert loaded.token_mode == name and loaded.query(renamed.code, "python").score == pytest.approx(scores[name])
        lm = MinHashIndex(cfg_full if name == "lexical" else cfg_lex)
        lm.load(out / "mh")
        assert lm.token_mode == name
    assert scores["full"] == pytest.approx(1.0) and scores["lexical"] < 0.5, scores
    assert scores["full_mh"] == pytest.approx(1.0) and scores["lexical_mh"] < scores["full_mh"], scores


def test_hybrid_signature_includes_token_mode(tmp_path):
    from smcode.semantic.combiner import Combiner
    from smcode.semantic.hybrid import HybridIndex, combiner_matches

    cfg = _cfg(tmp_path, fingerprint={"token_mode": "lexical"})
    h = HybridIndex(cfg)
    sig = h.combiner_signature()
    assert sig["token_mode"] == "lexical" and h.token_mode == "lexical"
    comb = Combiner(rule="logistic")
    X = np.array([[0.9, 0.9, 0.5, 3.0], [0.1, 0.0, 0.0, 3.0]] * 5, dtype=np.float32)
    comb.fit(X, np.array([1, 0] * 5))
    comb.train_stats.update({**sig, "token_mode": None})  # старый комбинатор без режима — считается обученным на full
    ok, why = combiner_matches(comb, sig)
    assert not ok and "token_mode" in why
    comb.train_stats["token_mode"] = "lexical"
    assert combiner_matches(comb, sig)[0]


# ----------------------------------------------------------------------------- свежесть индекса и meta.json


def test_build_from_corpus_rebuilds_when_inputs_are_newer(tmp_path):
    cfg = _cfg(tmp_path)
    fdir = Path(cfg["paths"]["functions"])
    fdir.mkdir(parents=True)
    write_jsonl(fdir / "protected.jsonl", _records(5))
    st = build_from_corpus("exact", cfg)
    assert st["skipped"] is False
    meta = json.loads((index_dir(cfg, "exact") / "meta.json").read_text(encoding="utf-8"))
    assert meta["build_seconds"] is not None and meta["disk_bytes"] > 0
    st2 = build_from_corpus("exact", cfg)
    assert st2["skipped"] is True and st2["disk_bytes"] == meta["disk_bytes"] and st2["build_seconds"] == meta["build_seconds"]
    assert [p.name for p in index_inputs("exact", cfg)] == ["protected.jsonl"]
    future = time.time() + 10
    os.utime(fdir / "protected.jsonl", (future, future))
    st3 = build_from_corpus("exact", cfg)
    assert st3["skipped"] is False  # входы новее meta.json → пересборка
    assert load_index("exact", cfg).n_records == 5


# ----------------------------------------------------------------------------- SemanticIndex: окна из protected_windows.npz


def test_semantic_index_reuses_window_embeddings_without_encoder(tmp_path):
    cfg = _cfg(tmp_path, semantic={"checkpoint": "model-A"})
    funcs, wins = _records(4), _records(3, kind="window")
    edir = Path(cfg["paths"]["embeddings"])
    rng = np.random.default_rng(0)
    save_embeddings(edir / "protected.npz", [r.id for r in funcs], l2_normalize(rng.random((4, 8)).astype(np.float32)), meta={"model": "model-A"})
    save_embeddings(edir / "protected_windows.npz", [r.id for r in wins], l2_normalize(rng.random((3, 8)).astype(np.float32)), meta={"model": "model-A"})
    idx = SemanticIndex(cfg)
    idx.set_encoder(_StubEncoder("model-A", fail=True))
    idx.build(funcs + wins, cfg)
    assert idx.meta["n_encoded"] == 0 and idx.meta["n_reused"] == 7 and idx.meta["splits"] == ["protected", "protected_windows"]
    assert idx.ids == [r.id for r in funcs + wins]
    (edir / "protected_windows.npz").unlink()  # без файла окон — только окна идут в энкодер
    idx2 = SemanticIndex(cfg)
    enc = _StubEncoder("model-A")
    idx2.set_encoder(enc)
    idx2.build(funcs + wins, cfg)
    assert idx2.meta["n_encoded"] == 3 and idx2.meta["n_reused"] == 4 and len(enc.calls) == 1 and len(enc.calls[0]) == 3


# ----------------------------------------------------------------------------- 06: дозапись недостающих строк


def test_embed_query_set_appends_missing_rows(tmp_path):
    cfg = _cfg(tmp_path, semantic={"checkpoint": "model-A", "max_length": 512})
    qdir = Path(cfg["paths"]["queries"])
    rows = [{"qid": f"q{i}", "code": f"def f{i}(): return {i}\n", "lang": "python"} for i in range(3)]
    write_jsonl(qdir / "protected.jsonl", rows)
    enc = _StubEncoder("model-A")
    first = enc.encode([rows[1]["code"]])
    from smcode.semantic.embed import query_embeddings_path

    save_embeddings(query_embeddings_path(cfg, "protected"), ["q1"], first, key="qids", meta={"model": "model-A", "max_length": 512})
    enc.calls.clear()
    out = embed_query_set(cfg, "protected", enc)
    assert out is not None and enc.calls == [[rows[0]["code"], rows[2]["code"]]]
    ids, emb, meta = load_embeddings(out)
    assert ids == ["q0", "q1", "q2"] and meta["n_encoded"] == 2 and meta["n_reused"] == 1
    assert np.allclose(emb[1], first[0], atol=2e-3)
    assert embed_query_set(cfg, "protected", enc) == out and len(enc.calls) == 1  # всё на месте → без кодирования


# ----------------------------------------------------------------------------- свежесть скоров и summary


def _write_scores(cfg: dict[str, Any], method: str, s: str, qids: list[str], scores: list[float], extra: dict[str, Any] | None = None) -> Path:
    p = RE.scores_path(cfg, method, s)
    write_jsonl(p, [{"qid": q, "score": v, "best_id": None, "latency_ms": 0.1, **(extra or {})} for q, v in zip(qids, scores)])
    with open(RE.meta_path(cfg, method, s), "w", encoding="utf-8") as f:
        json.dump({"method": method, "latency_mode": "per_query"}, f)
    return p


def test_is_done_and_summary_freshness(tmp_path):
    cfg = _cfg(tmp_path)
    qdir = Path(cfg["paths"]["queries"])
    qpath = qdir / "protected.jsonl"
    write_jsonl(qpath, [{"qid": "a#identity#0", "source_id": "a", "label": 1, "set": "protected", "transform": "identity", "params": {},
                         "lang": "python", "repo": "r", "code": "x", "n_lines": 1, "n_tokens": 1}])
    sp = _write_scores(cfg, "exact", "protected", ["a#identity#0"], [1.0])
    assert RE._is_done(cfg, "exact", "protected", None)
    idir = index_dir(cfg, "exact")
    idir.mkdir(parents=True, exist_ok=True)
    (idir / "meta.json").write_text("{}", encoding="utf-8")
    past = sp.stat().st_mtime - 100
    os.utime(idir / "meta.json", (past, past))  # индекс старше скоров → готово
    assert RE._is_done(cfg, "exact", "protected", None, idir)
    future = time.time() + 10
    os.utime(qpath, (future, future))
    assert not RE._is_done(cfg, "exact", "protected", None)
    os.utime(qpath, (future - 100, future - 100))
    os.utime(idir / "meta.json", (future, future))
    assert RE._is_done(cfg, "exact", "protected", None) and not RE._is_done(cfg, "exact", "protected", None, idir)
    # summary.json: запросы новее → пересчёт
    rdir = Path(cfg["paths"]["results"])
    spath = rdir / "summary.json"
    summary = {"methods": {"exact": {}}}
    os.utime(sp, (future - 50, future - 50))
    spath.write_text(json.dumps(summary), encoding="utf-8")
    os.utime(spath, (future + 5, future + 5))
    assert summary_is_stale(cfg, summary, spath) is None
    os.utime(qpath, (future + 10, future + 10))
    assert "запросов" in (summary_is_stale(cfg, summary, spath) or "")


# ----------------------------------------------------------------------------- метрики: мондриан, кластерный тест, пропуски, окна


def _queries(set_name: str, n_src: int, transforms: tuple[str, ...], partial: tuple[int, ...], rng: np.random.Generator) -> list[dict[str, Any]]:
    rows = []
    for i in range(n_src):
        sid = f"{set_name}/r{i % 4}/f{i}.py:1-20"
        n_tok = int(rng.integers(8, 300))
        for t in transforms:
            rows.append({"qid": f"{sid}#{t}#0", "source_id": sid, "label": label_for_set(set_name), "set": set_name, "transform": t,
                         "params": {}, "lang": "python", "repo": f"r{i % 4}", "code": "x", "n_lines": 3, "n_tokens": n_tok})
        for j, L in enumerate(partial):
            rows.append({"qid": f"{sid}#partial#{j}", "source_id": sid, "label": label_for_set(set_name), "set": set_name, "transform": "partial",
                         "params": {"L": L}, "lang": "python", "repo": f"r{i % 4}", "code": "x", "n_lines": L, "n_tokens": min(n_tok, 4 * L)})
    return rows


def _scored(set_name: str, rows: list[dict[str, Any]], scores: list[float], reasons: list[str] | None = None) -> M.ScoredSet:
    qmeta = {r["qid"]: {"label": r["label"], "set": set_name, "transform": r["transform"], "L": int(r["params"].get("L", -1)), "dst_lang": "",
                        "lang": r["lang"], "n_tokens": r["n_tokens"], "repo": r["repo"], "source_id": r["source_id"]} for r in rows}
    srows = [{"qid": r["qid"], "score": s, "best_id": None, "latency_ms": 0.1, **({"reason": reasons[i]} if reasons and reasons[i] else {})}
             for i, (r, s) in enumerate(zip(rows, scores))]
    return M.join_scores(srows, qmeta, set_name)


def test_mondrian_threshold_cells_and_max():
    rng = np.random.default_rng(1)
    rows = _queries("public_calib", 40, ("identity", "rename_ids"), (5, 10), rng)
    scores = [float(rng.beta(2, 5)) if r["transform"] != "partial" else float(rng.beta(1, 2)) for r in rows]
    calib = _scored("public_calib", rows, scores)
    tau, cells = M.mondrian_threshold(calib, 0.05)
    assert set(cells) == {"identity", "rename_ids", "partial@5", "partial@10"}
    assert all(c["n"] == 40 and c["sufficient"] for c in cells.values())
    assert tau == pytest.approx(max(float(c["tau"]) for c in cells.values()))
    tau_small, cells_small = M.mondrian_threshold(calib, 0.01)  # нужно ≥ 99 функций на ячейку → все ячейки недостаточны
    assert tau_small == float("inf") and all(c["tau"] == "inf" and not c["sufficient"] for c in cells_small.values())


def test_cluster_robust_test_properties():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(200), 5)
    zero = M.cluster_robust_test(np.zeros(1000, bool), clusters, 0.01)
    assert zero["p_value"] == 1.0 and zero["z"] is None and zero["mde_80"] > 0.01 and zero["n"] == 1000
    high = M.cluster_robust_test(rng.random(1000) < 0.3, clusters, 0.01)
    assert high["p_value"] < 1e-6 and high["z"] > 5
    near = M.cluster_robust_test(rng.random(1000) < 0.01, clusters, 0.01)
    assert 0.05 < near["p_value"] <= 1.0
    cl_hits = np.repeat(rng.random(200) < 0.05, 5)  # решение постоянно внутри кластера → дизайн-эффект > 1
    corr = M.cluster_robust_test(cl_hits, clusters, 0.01)
    assert corr["design_effect"] > 2 and corr["n_eff"] < 500 and corr["mde_80"] > zero["mde_80"]


def test_misses_by_reason_counts():
    rng = np.random.default_rng(3)
    rows = _queries("protected", 6, ("identity", "rename_ids"), (5,), rng)
    scores = [0.9, 0.0, 0.0] * 6
    reasons = ["", "too_short", ""] * 6
    ss = _scored("protected", rows, scores, reasons)
    assert ss.reason is not None and (ss.reason == "too_short").sum() == 6
    out = M.misses_by_reason(ss, 0.5)
    assert out == {"n_misses": 12, "too_short": 6, "no_hit": 6}
    g = M.tpr_groups(ss, 0.5, None, [[0, 100000]])
    assert g["misses_by_reason"] == out and misses_cell(out).startswith("12 (")


def test_method_metrics_windows_and_tables(tmp_path):
    cfg = _cfg(tmp_path)
    rng = np.random.default_rng(4)
    qdir = Path(cfg["paths"]["queries"])
    specs = {"protected": (30, ("identity", "rename_ids", "insert_deadcode"), (5, 10)), "public_calib": (30, ("identity", "rename_ids", "insert_deadcode"), (5, 10)),
             "public_test": (30, ("identity", "rename_ids", "insert_deadcode"), (5, 10)), "hard_neg": (24, ("identity", "rename_ids", "insert_deadcode"), (5, 10)),
             "protected_windows": (20, ("identity",), (5, 10)), "public_calib_windows": (20, ("identity",), (5, 10)),
             "public_test_windows": (20, ("identity",), (5, 10))}
    for s, (n, tr, pl) in specs.items():
        rows = _queries(s, n, tr, pl, rng)
        write_jsonl(qdir / f"{s}.jsonl", rows)
        member = s.startswith("protected")
        sc = [float(rng.beta(6, 2)) if member else float(rng.beta(1, 6)) for _ in rows]
        reasons = {"reason": "too_short"} if s == "protected" else None
        _write_scores(cfg, "exact", s, [r["qid"] for r in rows], [0.0 if (member and i % 7 == 0) else v for i, v in enumerate(sc)])
        if reasons:
            p = RE.scores_path(cfg, "exact", s)
            data = list(read_jsonl(p))
            for i, r in enumerate(data):
                if r["score"] == 0.0 and i % 2 == 0:
                    r["reason"] = "too_short"
            write_jsonl(p, data)
    summary = M.compute_summary(cfg, methods=["exact"], bootstrap=10)
    ak = summary["alpha_keys"][0]
    E = summary["methods"]["exact"]
    th = E["thresholds"][ak]
    assert th["mondrian"] is not None and set(th["mondrian_cells"]) == {"identity", "rename_ids", "insert_deadcode", "partial@5", "partial@10"}
    assert th["windows"] is not None and th["n_calib_windows"] == 20 * 3
    assert "mondrian" in E["tpr"][ak] and "mondrian" in E["fpr"][ak]
    pt = E["fpr"][ak]["conformal"]["public_test"]
    assert set(pt["test_above_alpha"]) >= {"p_value", "mde_80", "n_eff"} and pt["test_above_alpha"]["mde_80"] > float(ak)
    mr = E["tpr"][ak]["conformal"]["misses_by_reason"]
    assert mr["n_misses"] >= mr.get("too_short", 0) > 0
    w = E["tpr_windows"][ak]
    assert w["calibration"] == "public_calib_windows" and w["fpr"]["public_test_windows"]["n"] == 60
    assert "public_test_windows" in w["fpr_at_function_tau"] and set(w["confusion"]) == {"protected_windows", "public_test_windows"}
    assert summary["data_stats"]["queries"]["public_test_windows"]["window_set"] is True
    assert "windows" in summary["notes"] and "fingerprint_invariance" in summary["notes"]
    tables = write_tables(cfg, summary)
    t3 = tables["T3"].read_text(encoding="utf-8")
    assert "Сценарий IDE" in t3 and "мондриан" in t3 and "public_test_windows" in t3 and "МДЭ" in t3
    t2 = tables["T2"].read_text(encoding="utf-8")
    assert "Пропуски членов по причинам" in t2 and "короче k" in t2
    assert verdict_cell(pt, float(ak)).startswith(("выше α", "не выше α")) and verdict_cell(None, 0.01) == "—"


def test_index_info_variant_tag(tmp_path):
    cfg = _cfg(tmp_path)
    rng = np.random.default_rng(5)
    rows = _queries("protected", 4, ("identity",), (), rng)
    ss = _scored("protected", rows, [0.5] * len(rows))
    vdir = tmp_path / "idx_v" / "exact"
    rdir = Path(cfg["paths"]["results"])
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "index_stats.json", "w", encoding="utf-8") as f:
        json.dump({"exact": {"build_seconds": 1.0, "disk_bytes": 1, "n_records": 1, "memory_bytes": 1},
                   "exact@idx_v": {"build_seconds": 12.5, "disk_bytes": 777, "n_records": 9, "memory_bytes": 90, "path": str(vdir)}}, f)
    RE.scores_dir(cfg, "exact_v")
    with open(RE.meta_path(cfg, "exact_v", "protected"), "w", encoding="utf-8") as f:
        json.dump({"method": "exact", "index_dir": str(vdir)}, f)
    info = M._index_info(cfg, "exact_v", {"protected": ss})
    assert info["build_seconds"] == 12.5 and info["disk_bytes"] == 777 and info["n_records"] == 9 and info["bytes_per_record"] == 10.0


# ----------------------------------------------------------------------------- приватность: состав индекса, модель, полезность M4


def _privacy_files(tmp_path: Path, model: str = "m", hard_neg_model: str | None = None) -> dict[str, Any]:
    cfg = _cfg(tmp_path, semantic={"checkpoint": model, "hybrid_rule": "two_threshold", "ann_top_k": 5},
               privacy={"bow_vocab": 100, "attack_leaked_pairs": [0], "bootstrap": 30, "noise_sigma": [0.0], "quant_bits": []})
    rng = np.random.default_rng(11)
    funcs, wins = _records(30), _records(6, kind="window")
    train = _records(40, split="public_train")
    fdir = Path(cfg["paths"]["functions"])
    fdir.mkdir(parents=True)
    write_jsonl(fdir / "protected.jsonl", funcs)
    write_jsonl(fdir / "protected_windows.jsonl", wins)
    write_jsonl(fdir / "public_train.jsonl", train)
    d = 16
    E_f, E_w, E_t = (l2_normalize(rng.standard_normal((n, d)).astype(np.float32)) for n in (30, 6, 40))
    edir = Path(cfg["paths"]["embeddings"])
    save_embeddings(edir / "protected.npz", [r.id for r in funcs], E_f, meta={"model": model})
    save_embeddings(edir / "protected_windows.npz", [r.id for r in wins], E_w, meta={"model": model})
    save_embeddings(edir / "public_train.npz", [r.id for r in train], E_t, meta={"model": model})
    idir = index_dir(cfg, "semantic")
    idir.mkdir(parents=True, exist_ok=True)
    (idir / "meta.json").write_text(json.dumps({"with_windows": True, "n_records": 36}), encoding="utf-8")
    qdir = Path(cfg["paths"]["queries"])
    for s in ("protected", "public_calib", "public_test", "hard_neg"):
        if s == "protected":
            src = funcs + wins
            emb = l2_normalize(np.concatenate([E_f, E_w]) + 0.05 * rng.standard_normal((36, d)).astype(np.float32))
        else:
            src = _records(25, split=s)
            emb = l2_normalize(rng.standard_normal((25, d)).astype(np.float32))
        qids = [f"{r.id}#identity#0" for r in src]
        save_embeddings(edir / "queries" / f"{s}.npz", qids, emb, key="qids", meta={"model": hard_neg_model if (s == "hard_neg" and hard_neg_model) else model})
        write_jsonl(qdir / f"{s}.jsonl", [{"qid": q, "source_id": r.id, "label": 1 if s == "protected" else 0, "set": s, "transform": "identity",
                                           "params": {}, "lang": "python", "repo": r.repo, "code": r.code, "n_lines": r.n_lines, "n_tokens": r.n_tokens}
                                          for q, r in zip(qids, src)])
    return cfg


def test_privacy_inputs_follow_main_index_and_check_model(tmp_path):
    from smcode.privacy.run import index_with_windows, load_inputs

    cfg = _privacy_files(tmp_path)
    assert index_with_windows(cfg) is True
    inp = load_inputs(cfg)
    assert len(inp.index_ids) == 36 and inp.meta["with_windows"] and inp.meta["n_index_windows"] == 6 and inp.meta["n_index_functions"] == 30
    assert inp.meta["index_composition"].endswith("protected_windows") and inp.model == "m"
    assert inp.query_codes is not None and set(inp.query_codes) == set(inp.queries) and inp.query_codes["protected"][0][0].startswith("def compute_")
    inp2 = load_inputs(cfg, with_windows=False, hybrid=False)
    assert len(inp2.index_ids) == 30 and inp2.query_codes is None and inp2.meta["index_composition"] == "protected (functions)"
    bad = _privacy_files(tmp_path / "bad", hard_neg_model="other")
    with pytest.raises(ValueError, match="different encoders"):
        load_inputs(bad)
    cfg_mismatch = _privacy_files(tmp_path / "mm")
    cfg_mismatch["semantic"]["checkpoint"] = "zzz"
    with pytest.raises(ValueError, match="config expects"):
        load_inputs(cfg_mismatch)


def test_privacy_utility_matches_eval_path_and_has_hybrid(tmp_path):
    from smcode.calibration import conformal_threshold, rate_with_ci
    from smcode.privacy.run import load_inputs, make_defence, run_privacy_inputs

    cfg = _privacy_files(tmp_path)
    inp = load_inputs(cfg)
    res = run_privacy_inputs(cfg, inp, defences=[make_defence(), make_defence(projection=True)], leaked_pairs=[0], a1_backend="ridge", key=b"k")
    assert res["hybrid_utility"]["available"] and res["hybrid_utility"]["rule"] == "two_threshold" and res["index_composition"].endswith("windows")
    none = res["defenses"][0]
    assert none["tpr"]["hybrid"] is not None and none["tpr"]["hybrid"]["n"] == 36 and none["fpr_hybrid"]["public_test"]["n"] == 25
    assert none["tpr_by_alpha"]["0.05"]["threshold_hybrid"] is not None
    assert res["a1"]["protected_coverage"]["occurrence_coverage"] is not None and "n_protected_only_ids" in res["a1"]["protected_coverage"]
    # строка «none» = TPR M3 основной оценки: тот же индекс (функции + окна), тот же порог по public_calib
    idx = SemanticIndex({"semantic": {"use_faiss": False, "ann_top_k": 1}})
    idx.build_from_embeddings(inp.index_ids, inp.index_emb)
    sc = {s: idx.score_embeddings(q, top_k=1)[0] for s, (_, q) in inp.queries.items()}
    tau = conformal_threshold(sc["public_calib"], 0.05)
    assert none["threshold"] == pytest.approx(tau) and none["tpr"]["semantic"]["value"] == pytest.approx(rate_with_ci(sc["protected"], tau)["value"])
    assert res["defenses"][1]["tpr"]["hybrid"]["value"] == pytest.approx(none["tpr"]["hybrid"]["value"], abs=0.1)  # проекция сохраняет cos


def test_privacy_consistency_warning(tmp_path):
    from smcode.eval.report import privacy_consistency_warning

    cfg = _cfg(tmp_path)
    rdir = Path(cfg["paths"]["results"])
    rdir.mkdir(parents=True, exist_ok=True)
    summary = {"methods": {"semantic": {"tpr": {"0.01": {"conformal": {"overall": {"value": 0.80, "ci": [0.76, 0.84], "n": 100, "k": 80}}}}}}}
    assert privacy_consistency_warning(cfg, summary) is None  # нет privacy.json
    priv = {"alpha": 0.01, "index_composition": "protected (functions)", "model": "m",
            "defenses": [{"name": "none", "tpr": {"semantic": {"value": 0.60, "ci": [0.5, 0.7], "ci_cluster": [0.52, 0.68], "n": 100, "k": 60}}}]}
    (rdir / "privacy.json").write_text(json.dumps(priv), encoding="utf-8")
    w = privacy_consistency_warning(cfg, summary)
    assert w and "не совпадает" in w and "protected (functions)" in w
    priv["defenses"][0]["tpr"]["semantic"].update({"value": 0.79, "ci_cluster": [0.74, 0.83]})
    (rdir / "privacy.json").write_text(json.dumps(priv), encoding="utf-8")
    assert privacy_consistency_warning(cfg, summary) is None


# ----------------------------------------------------------------------------- train.py: параметры, маска ложных негативов


def test_train_config_and_false_negative_mask(tmp_path):
    from smcode.semantic.train import ContrastiveDataset, TrainConfig, evaluated_protected_ids, false_negative_mask

    cfg = _cfg(tmp_path)
    tc = TrainConfig().resolved(cfg)
    assert tc.grad_checkpointing is True and tc.grad_accum == 1 and tc.exclude_transforms == [] and tc.rename_styles is None
    cfg["semantic"].update({"grad_checkpointing": False, "grad_accum": 4, "exclude_transforms": ["insert_deadcode"], "rename_styles": ["vn"]})
    tc = TrainConfig().resolved(cfg)
    assert tc.grad_checkpointing is False and tc.grad_accum == 4 and tc.exclude_transforms == ["insert_deadcode"] and tc.rename_styles == ["vn"]
    names, partial, styles = tc.positives_spec(cfg)
    assert "insert_deadcode" not in names and partial == cfg["transforms"]["partial_lines"] and styles == ["vn"]
    tc2 = TrainConfig(exclude_transforms=["partial", "reorder_stmts"], grad_checkpointing=True).resolved(cfg)
    assert tc2.grad_checkpointing is True and tc2.positives_spec(cfg)[1] == [] and "reorder_stmts" not in tc2.positives_spec(cfg)[0]
    recs = _records(6)
    recs[3].sha = recs[0].sha  # вендоренный дубликат
    ds = ContrastiveDataset(recs, cfg, n_hard=2, seed=1, p_llm=0.0, transforms=names, partial_lines=partial, rename_styles=styles)
    assert "insert_deadcode" not in ds.transforms and ds.rename_styles == ["vn"]
    item = ds[0]
    assert item["sha"] == recs[0].sha and len(item["neg_sha"]) == len(item["negatives"]) and recs[3].sha not in item["neg_sha"]
    mask = false_negative_mask([r.sha for r in recs[:4]], [recs[0].sha, "zzz"])
    assert mask is not None and mask.shape == (4, 6) and mask[0, 3] and mask[3, 0] and mask[0, 4] and not mask[0, 0] and not mask[1, 2]
    assert false_negative_mask(["a", "b"], ["c"]) is None and false_negative_mask(None, None) is None
    assert evaluated_protected_ids(cfg) == set()
    qdir = Path(cfg["paths"]["queries"])
    write_jsonl(qdir / "protected.jsonl", [{"qid": "p1#identity#0", "source_id": "p1"}, {"qid": "p2#partial#0", "source_id": "p2"}])
    assert evaluated_protected_ids(cfg) == {"p1", "p2"}
