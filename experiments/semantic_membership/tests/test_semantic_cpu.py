"""CPU-тесты модуля semantic (без torch/faiss): ANN (numpy), SemanticIndex, HybridIndex (numpy-путь),
Combiner, словарь своей модели, хранение эмбеддингов, py_compile GPU-файлов, реестр индексов."""

from __future__ import annotations

import importlib
import os
import py_compile
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from smcode.config import load_config
from smcode.semantic import ann as ann_mod
from smcode.semantic.ann import ANN, cosine_topk, l2_normalize, topk_numpy
from smcode.semantic.combiner import FEATURE_NAMES, N_FEATURES, Combiner, candidate_labels, make_features, two_threshold_score
from smcode.semantic.embed import embeddings_for_records, load_embeddings, save_embeddings
from smcode.semantic.hybrid import HybridIndex, collect_features, combiner_training_queries
from smcode.semantic.model import CodeEncoder, preprocess
from smcode.semantic.own_model import FIRST_WORD_ID, CodeVocab, lexical_pieces, split_identifier
from smcode.semantic.semantic_index import SemanticIndex
from smcode.types import FunctionRecord, QueryResult

ROOT = Path(__file__).resolve().parents[1]
TORCH_FILES = ["smcode/semantic/model.py", "smcode/semantic/own_model.py", "smcode/semantic/train.py", "smcode/semantic/embed.py",
               "scripts/05_train_encoder.py", "scripts/06_embed.py"]


# ----------------------------------------------------------------------------- синтетика


def _py_fn(i: int, n: int = 14) -> str:
    rng = random.Random(1000 + i)
    ids = [f"v{i}_{j}" for j in range(4)]
    lines = [f"def f_{i}(a, b, c):"]
    for _ in range(n):
        x, y = rng.choice(ids), rng.choice(ids + ["a", "b", "c"])
        op = rng.choice(["+", "-", "*", "//", "%"])
        lines.append(f"    {x} = {y} {op} {rng.randint(1, 99)}")
        if rng.random() < 0.3:
            lines.append(f"    if {x} > {rng.randint(1, 50)}:")
            lines.append(f"        {y} = {x} {op} 1")
    lines.append(f"    return {rng.choice(ids)}")
    return "\n".join(lines) + "\n"


def _records(n: int, split: str = "protected", repo: str = "org/repo") -> list[FunctionRecord]:
    out = []
    for i in range(n):
        code = _py_fn(i)
        out.append(FunctionRecord(id=f"{split}/{repo}/f{i}.py:1-{code.count(chr(10))}", split=split, repo=repo if i % 2 == 0 else repo + "2",
                                  path=f"f{i // 3}.py", lang="python", code=code, start_line=1, end_line=code.count("\n"),
                                  n_lines=code.count("\n"), n_tokens=len(code.split()), sha=f"sha{i}"))
    return out


def _cfg(tmp_path: Path, **semantic) -> dict:
    cfg = load_config()
    cfg["paths"] = {"raw_repos": str(tmp_path / "raw"), "functions": str(tmp_path / "functions"), "queries": str(tmp_path / "queries"),
                    "indexes": str(tmp_path / "indexes"), "results": str(tmp_path / "results"), "figures": str(tmp_path / "figures"),
                    "embeddings": str(tmp_path / "embeddings"), "runs": str(tmp_path / "runs")}
    cfg["fingerprint"] = {**cfg["fingerprint"], "k": 5, "w": 4}
    cfg["semantic"] = {**cfg["semantic"], "ann_top_k": 5, "combiner_path": str(tmp_path / "combiner.pkl"), **semantic}
    return cfg


def _unit(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    return l2_normalize(rng.standard_normal((n, d)).astype(np.float32))


# ----------------------------------------------------------------------------- ann


def test_topk_numpy_matches_bruteforce():
    rng = np.random.default_rng(0)
    X = _unit(rng, 300, 16)
    q = _unit(rng, 7, 16)
    sims, idx = topk_numpy(q, X, 5, chunk_rows=64, chunk_queries=3)
    full = q @ X.T
    exp_idx = np.argsort(-full, axis=1)[:, :5]
    assert idx.shape == (7, 5) and sims.shape == (7, 5)
    assert np.array_equal(idx, exp_idx)
    assert np.allclose(sims, np.take_along_axis(full, exp_idx, axis=1), atol=1e-6)
    assert np.all(np.diff(sims, axis=1) <= 1e-6)  # по убыванию


def test_topk_edge_cases():
    rng = np.random.default_rng(1)
    X = _unit(rng, 3, 8)
    sims, idx = topk_numpy(_unit(rng, 2, 8), X, 10)  # k > N
    assert idx.shape == (2, 3)
    sims, idx = topk_numpy(_unit(rng, 2, 8), np.zeros((0, 8), np.float32), 3)
    assert idx.shape == (2, 0) and sims.shape == (2, 0)
    s, i = cosine_topk(np.ones(8) * 3, X * 5, 1)
    assert s.shape == (1, 1) and i.shape == (1, 1)


def test_ann_numpy_backend_without_faiss(monkeypatch):
    monkeypatch.setattr(ann_mod, "_try_faiss", lambda: None)
    rng = np.random.default_rng(2)
    X = _unit(rng, 50, 12)
    a = ANN(X, use_faiss=True)
    assert a.backend == "numpy" and a.n == 50 and a.dim == 12
    sims, idx = a.search(X[[3, 7]], 1)
    assert idx[:, 0].tolist() == [3, 7]
    assert np.allclose(sims[:, 0], 1.0, atol=1e-5)
    assert a.memory_bytes() == X.nbytes


# ----------------------------------------------------------------------------- semantic index


def test_semantic_index_nearest_is_itself(tmp_path):
    cfg = _cfg(tmp_path)
    rng = np.random.default_rng(3)
    n, d = 200, 32
    ids = [f"protected/r/f{i}" for i in range(n)]
    M = _unit(rng, n, d)
    idx = SemanticIndex(cfg)
    idx.build_from_embeddings(ids, M * 2.0, meta={"model": "fake"})  # ненормированный вход нормируется
    assert idx.n_records == n and idx.dim == d
    scores, best, top_idx, top_sims = idx.score_embeddings(M[:10], top_k=3)
    assert scores.shape == (10,) and top_idx.shape == (10, 3) and top_sims.shape == (10, 3)
    assert best == ids[:10]
    assert np.allclose(scores, 1.0, atol=1e-5)
    # возмущённый запрос всё ещё находит исходный вектор
    noisy = l2_normalize(M[5] + 0.1 * rng.standard_normal(d).astype(np.float32))
    s, b, _, _ = idx.score_embeddings(noisy)
    assert b[0] == ids[5] and 0.8 < s[0] <= 1.0
    # score обрезан в [0, 1]: единственный вектор индекса против своей инверсии (cos = −1)
    one = SemanticIndex(cfg)
    one.build_from_embeddings(ids[:1], M[:1])
    s, b, _, sims = one.score_embeddings(-M[0])
    assert s[0] == 0.0 and b[0] == ids[0] and sims[0, 0] < -0.99
    # память и QueryResult
    assert idx.memory_bytes() >= M.nbytes
    r = idx._result(scores[0], best[0], top_idx[0], top_sims[0])
    assert isinstance(r, QueryResult) and r.best_id == ids[0] and "cos" in r.details


def test_semantic_index_save_load_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    rng = np.random.default_rng(4)
    ids = [f"p/f{i}" for i in range(40)]
    M = _unit(rng, 40, 16)
    idx = SemanticIndex(cfg)
    idx.build_from_embeddings(ids, M)
    out = tmp_path / "indexes" / "semantic"
    idx.save(out)
    assert (out / "arrays.npz").exists() and (out / "state.pkl").exists()
    idx2 = SemanticIndex(cfg)
    idx2.load(out)
    assert idx2.ids == ids and idx2.matrix.shape == (40, 16)
    s1, b1, _, _ = idx.score_embeddings(M[:5])
    s2, b2, _, _ = idx2.score_embeddings(M[:5])
    assert b1 == b2 and np.allclose(s1, s2, atol=2e-3)  # float16 на диске
    assert idx2.meta.get("n_records") == 40


def test_semantic_empty_index(tmp_path):
    idx = SemanticIndex(_cfg(tmp_path))
    scores, best, ti, ts = idx.score_embeddings(np.ones((2, 4), np.float32))
    assert scores.tolist() == [0.0, 0.0] and best == [None, None] and ti.shape == (2, 0)


def test_registry_make_index_without_torch(tmp_path):
    from smcode.fingerprint.index import REGISTRY, make_index

    cfg = _cfg(tmp_path)
    assert REGISTRY["semantic"] == "smcode.semantic.semantic_index:SemanticIndex"
    assert REGISTRY["hybrid"] == "smcode.semantic.hybrid:HybridIndex"
    assert isinstance(make_index("semantic", cfg), SemanticIndex)
    assert isinstance(make_index("hybrid", cfg), HybridIndex)
    assert "torch" not in sys.modules


# ----------------------------------------------------------------------------- combiner


def test_combiner_fit_predict_and_pickle(tmp_path):
    rng = np.random.default_rng(5)
    n = 400
    pos = np.column_stack([rng.uniform(0.7, 1.0, n), rng.uniform(0.3, 1.0, n), rng.uniform(0.2, 1.0, n), rng.uniform(2, 6, n)])
    neg = np.column_stack([rng.uniform(0.2, 0.75, n), rng.uniform(0.0, 0.2, n), rng.uniform(0.0, 0.1, n), rng.uniform(2, 6, n)])
    X = np.vstack([pos, neg]).astype(np.float32)
    y = np.array([1] * n + [0] * n)
    c = Combiner(rule="logistic", C=1.0).fit(X, y)
    assert c.fitted and c.n_train == 2 * n
    p = c.predict_proba(X)
    assert p.shape == (2 * n,) and p[:n].mean() > 0.9 and p[n:].mean() < 0.1
    assert set(c.train_stats["coef"]) == set(FEATURE_NAMES)
    path = c.save(tmp_path / "comb.pkl")
    c2 = Combiner.load(path)
    assert np.allclose(c2.predict_proba(X[:10]), p[:10])
    assert c2.describe()["fitted"] is True


def test_two_threshold_rule():
    th = {"cos": 0.8, "overlap": 0.2}
    X = np.array([make_features(0.8, 0.0, 0.0, 10), make_features(0.5, 0.2, 0.1, 10), make_features(0.4, 0.1, 0.0, 10),
                  make_features(0.9, 0.5, 0.5, 10)])
    s = two_threshold_score(X, th)
    assert np.isclose(s[0], 0.5) and np.isclose(s[1], 0.5)
    assert s[2] < 0.5 < s[3]
    c = Combiner(rule="two_threshold", thresholds=th)
    assert np.allclose(c.predict_proba(X), s)
    unfitted = Combiner(rule="logistic")
    assert np.allclose(unfitted.predict_proba(X), s)  # fallback
    with pytest.raises(ValueError):
        Combiner(rule="nope")


def test_candidate_labels():
    y = candidate_labels([1, 0], ["a", "b"], [["a", "c"], ["b", "a"]])
    assert y.tolist() == [[1, 0], [0, 0]]


# ----------------------------------------------------------------------------- hybrid (numpy-путь)


def _hybrid_fixture(tmp_path, rule="logistic"):
    cfg = _cfg(tmp_path, hybrid_rule=rule)
    recs = _records(30)
    rng = np.random.default_rng(6)
    d = 24
    M = _unit(rng, len(recs), d)
    h = HybridIndex(cfg)
    h.build_from_embeddings([r.id for r in recs], M, [r.code for r in recs], [r.lang for r in recs], meta={"model": "fake"})
    return cfg, recs, M, h


def test_hybrid_numpy_path_features_and_scores(tmp_path):
    cfg, recs, M, h = _hybrid_fixture(tmp_path)
    assert h.n_records == 30 and h.rec_hashes.size > 0 and h.rec_offsets.shape == (31,)
    assert all(h.record_fps(i).size > 0 for i in range(30))
    # запрос = сама запись: cos≈1, overlap=1, lcr_norm=1
    feats, idx, sims, info = h.features_for_queries(M[:3], [r.code for r in recs[:3]], ["python"] * 3)
    assert feats.shape == (3, 5, N_FEATURES) and info[0]["n_fps"] > 0
    for i in range(3):
        assert idx[i, 0] == i
        assert np.isclose(feats[i, 0, 0], 1.0, atol=1e-5)
        assert np.isclose(feats[i, 0, 1], 1.0) and np.isclose(feats[i, 0, 2], 1.0)
        assert feats[i, 0, 3] == pytest.approx(np.log1p(info[i]["n_tokens"]))
        assert feats[i, 1:, 1].max() < 1.0  # другие функции совпадают не полностью
    # частичный запрос (середина функции) с «чужим» эмбеддингом: overlap > 0 у исходной записи
    lines = recs[4].code.splitlines()
    part = "\n".join(lines[3:9]) + "\n"
    q_emb = _unit(np.random.default_rng(7), 1, M.shape[1])
    feats, idx, sims, info = h.features_for_queries(q_emb, [part], ["python"])
    if 4 in idx[0]:
        j = list(idx[0]).index(4)
        assert feats[0, j, 1] > 0
    # комбинатор two_threshold → score и best_id
    h.set_combiner(Combiner(rule="two_threshold", thresholds={"cos": 0.8, "overlap": 0.2}))
    scores, best, details = h.score_embeddings_with_codes(M[:3], [r.code for r in recs[:3]], ["python"] * 3)
    assert scores.shape == (3,) and best == [r.id for r in recs[:3]]
    assert np.all(scores > 0.99)
    assert details[0]["rule"] == "two_threshold" and "overlap" in details[0] and details[0]["sem_best_id"] == recs[0].id
    # случайный запрос без отпечатков (короткий код)
    scores, best, details = h.score_embeddings_with_codes(_unit(np.random.default_rng(8), 1, M.shape[1]), ["x = 1\n"], ["python"])
    assert details[0]["n_fps"] == 0 and 0.0 <= scores[0] <= 1.0


def test_hybrid_logistic_combiner_end_to_end_and_roundtrip(tmp_path):
    cfg, recs, M, h = _hybrid_fixture(tmp_path)
    rng = np.random.default_rng(9)
    # члены: код записи + слегка возмущённый эмбеддинг; не-члены: чужой код + случайный эмбеддинг
    q_codes, q_langs, q_emb, labels, srcs = [], [], [], [], []
    for i in range(30):
        q_codes.append(recs[i].code)
        q_langs.append("python")
        q_emb.append(l2_normalize(M[i] + 0.2 * rng.standard_normal(M.shape[1]).astype(np.float32))[0])
        labels.append(1)
        srcs.append(recs[i].id)
    others = _records(30, split="public_test", repo="other/repo")
    for i in range(30):
        q_codes.append(_py_fn(500 + i))
        q_langs.append("python")
        q_emb.append(_unit(rng, 1, M.shape[1])[0])
        labels.append(0)
        srcs.append(others[i].id)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    X, y, groups = collect_features(h, q_emb, q_codes, q_langs, labels, srcs)
    assert X.shape[1] == N_FEATURES and y.sum() == 30 and len(groups) == len(y)
    comb = Combiner(rule="logistic").fit(X, y)
    h.set_combiner(comb)
    scores, best, _ = h.score_embeddings_with_codes(q_emb, q_codes, q_langs)
    assert scores[:30].min() > scores[30:].max()
    assert best[:30] == [r.id for r in recs]
    # save/load сохраняет матрицу, отпечатки и комбинатор
    out = tmp_path / "indexes" / "hybrid"
    h.save(out)
    h2 = HybridIndex(cfg)
    h2.load(out)
    assert h2.ids == h.ids and h2.combiner is not None and h2.combiner.fitted and h2.rule == "logistic"
    assert np.array_equal(h2.rec_hashes, h.rec_hashes) and np.array_equal(h2.rec_offsets, h.rec_offsets)
    s2, b2, _ = h2.score_embeddings_with_codes(q_emb[:5], q_codes[:5], q_langs[:5])
    assert b2 == best[:5] and np.allclose(s2, scores[:5], atol=5e-3)
    assert h2.memory_bytes() > 0 and h2.meta.get("k") == 5


def test_hybrid_two_threshold_rule_from_config(tmp_path):
    cfg, recs, M, h = _hybrid_fixture(tmp_path, rule="two_threshold")
    assert h.combiner is not None and h.combiner.rule == "two_threshold"
    comb = h.ensure_combiner(cfg)  # без обучения и без torch
    assert comb.rule == "two_threshold"
    scores, best, _ = h.score_embeddings_with_codes(M[:2], [r.code for r in recs[:2]], ["python"] * 2)
    assert best == [recs[0].id, recs[1].id]


def test_combiner_training_queries_split_by_repo(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["transforms"] = {**cfg["transforms"], "programmatic": ["identity", "rename_ids"], "partial_lines": [5]}
    recs = _records(12, split="public_train", repo="a/one") + _records(12, split="public_train", repo="b/two")
    for i, r in enumerate(recs):
        r.id = f"public_train/{r.repo}/f{i}"
    index_recs, queries = combiner_training_queries(cfg, recs, n_index=10, n_anchors=4, n_neg=4)
    assert 1 <= len(index_recs) <= 10
    idx_repos = {r.repo for r in index_recs}
    pos = [q for q in queries if q["label"] == 1]
    neg = [q for q in queries if q["label"] == 0]
    assert pos and neg
    assert all(q["source_id"] in {r.id for r in index_recs} for q in pos)
    neg_repos = {q["source_id"].split("/f")[0].split("public_train/")[1] for q in neg}
    assert not (neg_repos & idx_repos)


# ----------------------------------------------------------------------------- embeddings on disk


def test_save_load_embeddings_and_reuse(tmp_path):
    cfg = _cfg(tmp_path)
    rng = np.random.default_rng(10)
    ids = [f"protected/r/f{i}" for i in range(6)]
    E = _unit(rng, 6, 8)
    path = save_embeddings(tmp_path / "embeddings" / "protected.npz", ids, E, key="ids", meta={"model": "fake"})
    ids2, E2, meta = load_embeddings(path)
    assert ids2 == ids and E2.dtype == np.float32 and np.allclose(E2, E, atol=2e-3) and meta["model"] == "fake"
    qp = save_embeddings(tmp_path / "embeddings" / "queries" / "protected.npz", ["q1", "q2"], E[:2], key="qids")
    with np.load(qp) as z:
        assert "qids" in z.files and z["emb"].dtype == np.float16

    class FakeEncoder:
        model_name = "fake"
        calls: list[list[str]] = []

        def encode(self, codes, langs, batch_size=32, show_progress=False):
            self.calls.append(list(codes))
            return np.full((len(codes), 8), 0.5, dtype=np.float32)

    enc = FakeEncoder()
    out = embeddings_for_records(cfg, enc, ids[:3] + ["new/id"], ["c"] * 4, ["python"] * 4, split="protected")
    assert out.shape == (4, 8) and np.allclose(out[:3], E[:3], atol=2e-3) and np.allclose(out[3], 0.5)
    assert enc.calls == [["c"]]  # закодирована только недостающая запись
    with pytest.raises(ValueError):
        save_embeddings(tmp_path / "bad.npz", ids, E[:2])


# ----------------------------------------------------------------------------- own-model vocab, preprocess


def test_split_identifier_and_pieces():
    assert split_identifier("getHTTPResponse_code") == ["get", "http", "response", "code"]
    assert split_identifier("x") == ["x"] and split_identifier("__init__") == ["init"]
    pieces = lexical_pieces('def foo_bar(x):\n    # c\n    return x + "s" + 12345678\n', "python")
    assert "def" in pieces and "foo" in pieces and "bar" in pieces and "<STR>" in pieces and "<NUM>" in pieces
    assert "#" not in " ".join(pieces)
    assert lexical_pieces("a_b = 1", "unknownlang")  # regex-запасной путь


def test_code_vocab_build_encode_roundtrip(tmp_path):
    items = [(_py_fn(i), "python") for i in range(20)]
    v = CodeVocab.build(items, max_size=300, min_count=1)
    assert FIRST_WORD_ID < len(v) <= 300
    ids = v.encode(_py_fn(0), "python", max_length=64)
    assert ids[0] == 2 and ids[-1] == 3 and len(ids) <= 64
    # неизвестное слово → байты (id в диапазоне 4..259)
    ids_unk = v.encode("zzqqxx = 1", "python", max_length=32)
    assert any(4 <= t < FIRST_WORD_ID for t in ids_unk)
    X, mask = v.encode_batch([_py_fn(1), "x = 1"], ["python", "python"], max_length=32)
    assert X.shape == mask.shape and X.shape[1] <= 32 and mask[1].sum() < mask[0].sum()
    p = v.save(tmp_path / "vocab.json")
    v2 = CodeVocab.load(p)
    assert v2.words == v.words and v2.encode(_py_fn(0), "python", 64) == ids


def test_preprocess_strips_comments_and_encoder_is_lazy(tmp_path):
    out = preprocess("def f():\n    # comment\n    return 1  \n", "python")
    assert "comment" not in out and out.endswith("return 1")
    assert preprocess("x", None) == "x" and preprocess("x", "weird") == "x"
    enc = CodeEncoder(_cfg(tmp_path), "some/model")
    assert enc.model is None and enc.model_name == "some/model" and enc.pooling == "mean" and enc.max_length == 512
    assert "torch" not in sys.modules and "transformers" not in sys.modules


# ----------------------------------------------------------------------------- GPU-файлы компилируются


@pytest.mark.parametrize("rel", TORCH_FILES)
def test_torch_files_compile(rel):
    py_compile.compile(str(ROOT / rel), doraise=True)


def test_train_module_imports_without_torch(tmp_path):
    train = importlib.import_module("smcode.semantic.train")
    tc = train.TrainConfig(max_steps=3).resolved(_cfg(tmp_path))
    assert tc.name.endswith("_ft") and tc.epochs == 2 and tc.batch_size == 128 and tc.lr == pytest.approx(2e-5)
    tc_own = train.TrainConfig(own_model=True, adapt_on_protected=True).resolved(_cfg(tmp_path))
    assert tc_own.name == "own_small_protected" and tc_own.lr == pytest.approx(3e-4)
    recs = _records(10, split="public_train")
    cfg = _cfg(tmp_path)
    ds = train.ContrastiveDataset(recs, cfg, n_hard=2, seed=1, p_llm=0.0)
    item = ds[0]
    assert item["anchor"][1] == "python" and len(item["negatives"]) == 2 and item["pos_kind"] != ""
    assert item["positive"][0] != "" and item["negatives"][0][0] != recs[0].code
    ds.set_epoch(1)
    item2 = ds[0]
    assert item2["anchor"] == item["anchor"]
    assert "torch" not in sys.modules


# ----------------------------------------------------------------------------- обучение комбинатора (fake-энкодер)


class _HashEncoder:
    """Детерминированный «энкодер»: мешок лексических частей → случайная проекция (без torch)."""

    model_name = "fake-hash"
    device = "cpu"

    def __init__(self, d: int = 32):
        self.d = d
        self.rng = np.random.default_rng(123)
        self.proj: dict[str, np.ndarray] = {}

    def encode(self, codes, langs=None, batch_size=32, show_progress=False):
        out = np.zeros((len(codes), self.d), dtype=np.float32)
        for i, c in enumerate(codes):
            for piece in lexical_pieces(c, (langs[i] if langs else "python")):
                if piece not in self.proj:
                    self.proj[piece] = self.rng.standard_normal(self.d).astype(np.float32)
                out[i] += self.proj[piece]
        return l2_normalize(out)


def test_train_combiner_from_corpus_and_ensure_combiner(tmp_path):
    from smcode.types import write_jsonl
    from smcode.semantic.hybrid import train_combiner_from_corpus

    cfg = _cfg(tmp_path)
    cfg["transforms"] = {**cfg["transforms"], "programmatic": ["identity", "rename_ids", "reformat"], "partial_lines": [5]}
    cfg["semantic"].update({"combiner_n_index": 20, "combiner_n_anchors": 8, "combiner_n_neg": 8, "reuse_embeddings": False})
    recs = _records(16, split="public_train", repo="a/one") + _records(16, split="public_train", repo="b/two")
    for i, r in enumerate(recs):
        r.id, r.code = f"public_train/{r.repo}/f{i}", _py_fn(300 + i)
    write_jsonl(Path(cfg["paths"]["functions"]) / "public_train.jsonl", recs)
    enc = _HashEncoder()
    comb = train_combiner_from_corpus(cfg, encoder=enc)
    assert comb.fitted and Path(cfg["semantic"]["combiner_path"]).exists()
    assert comb.train_stats["n_pos"] > 0 and comb.train_stats["train_auc"] > 0.9
    # ensure_combiner подхватывает сохранённый файл без обучения
    h = HybridIndex(cfg)
    prot = _records(10)
    h.build_from_embeddings([r.id for r in prot], enc.encode([r.code for r in prot]), [r.code for r in prot], ["python"] * 10)
    got = h.ensure_combiner(cfg, encoder=None)
    assert got.fitted and h.rule == "logistic" and h.meta["combiner"]["fitted"] is True
    scores, best, _ = h.score_embeddings_with_codes(enc.encode([prot[2].code]), [prot[2].code], ["python"])
    assert best[0] == prot[2].id and scores[0] > 0.5
    assert "torch" not in sys.modules
