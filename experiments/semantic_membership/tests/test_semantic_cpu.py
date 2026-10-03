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
    assert comb.train_stats["model"] == "fake-hash" and comb.train_stats["k"] == 5 and comb.train_stats["top_k"] == 5
    # ensure_combiner подхватывает сохранённый файл без обучения (индекс той же модели)
    h = HybridIndex(cfg)
    prot = _records(10)
    h.build_from_embeddings([r.id for r in prot], enc.encode([r.code for r in prot]), [r.code for r in prot], ["python"] * 10,
                            meta={"model": enc.model_name})
    got = h.ensure_combiner(cfg, encoder=None)
    assert got.fitted and h.rule == "logistic" and h.meta["combiner"]["fitted"] is True and h.meta["combiner_fitted"] is True
    scores, best, details = h.score_embeddings_with_codes(enc.encode([prot[2].code]), [prot[2].code], ["python"])
    assert best[0] == prot[2].id and scores[0] > 0.5
    assert details[0]["rule"] == "logistic" and details[0]["combiner_fitted"] is True
    assert "torch" not in sys.modules


# ----------------------------------------------------------------------------- регрессии после ревью


class _NamedEncoder:
    """Фейковый энкодер с заданным именем модели; считает вызовы encode."""

    def __init__(self, model_name: str, d: int = 8, fill: float = 0.5, max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cpu"
        self.kind = "fake"
        self.d = d
        self.fill = fill
        self.calls: list[list[str]] = []

    def encode(self, codes, langs=None, batch_size=32, show_progress=False):
        self.calls.append(list(codes))
        return l2_normalize(np.full((len(codes), self.d), self.fill, dtype=np.float32) + np.arange(self.d, dtype=np.float32) * 0.01)


def test_model_key_and_tag(tmp_path):
    from smcode.config import ROOT as PKG_ROOT
    from smcode.semantic.model import expected_model_key, model_key, model_tag, resolve_model_name

    assert model_key("microsoft/unixcoder-base") == "microsoft/unixcoder-base"
    assert model_key(None) == "" and model_key("") == ""
    assert model_tag("microsoft/unixcoder-base") == "microsoft__unixcoder-base"
    # относительный и абсолютный путь к каталогу внутри стенда дают один ключ
    runs_existed = (PKG_ROOT / "runs").exists()
    run = PKG_ROOT / "runs" / "_test_model_key_tmp" / "best"
    run.mkdir(parents=True, exist_ok=True)
    try:
        assert model_key("runs/_test_model_key_tmp/best") == "runs/_test_model_key_tmp/best"
        assert model_key(str(run)) == "runs/_test_model_key_tmp/best"
        assert model_key("runs/_test_model_key_tmp/best/") == "runs/_test_model_key_tmp/best"
        assert model_tag(str(run)) == "runs___test_model_key_tmp__best"
        cfg = {"semantic": {"checkpoint": "runs/_test_model_key_tmp/best", "base_model": "x/y"}}
        assert resolve_model_name(cfg) == str(run) and expected_model_key(cfg) == "runs/_test_model_key_tmp/best"
    finally:
        run.rmdir()
        run.parent.rmdir()
        if not runs_existed:
            (PKG_ROOT / "runs").rmdir()
    assert expected_model_key({"semantic": {"base_model": "x/y"}}) == "x/y"
    # каталог вне стенда → абсолютный путь
    outside = tmp_path / "ckpt"
    outside.mkdir()
    assert model_key(outside) == str(outside.resolve())


def test_embeddings_reuse_is_keyed_on_model(tmp_path):
    """Блокер: векторы другой модели не переиспользуются и не выдаются за текущую."""
    from smcode.semantic.embed import cached_split_path, embeddings_for_records_ex, read_embeddings_meta

    cfg = _cfg(tmp_path)
    rng = np.random.default_rng(11)
    ids = [f"protected/r/f{i}" for i in range(6)]
    E = _unit(rng, 6, 8)
    save_embeddings(tmp_path / "embeddings" / "protected.npz", ids, E, key="ids", meta={"model": "microsoft/unixcoder-base", "max_length": 512})
    # другой энкодер → ничего не переиспользуется, кодируется всё
    enc_b = _NamedEncoder("runs/finetuned/best")
    out, info = embeddings_for_records_ex(cfg, enc_b, ids[:4], ["c"] * 4, ["python"] * 4, split="protected")
    assert info["n_reused"] == 0 and info["n_encoded"] == 4 and info["model"] == "runs/finetuned/best"
    assert enc_b.calls == [["c"] * 4] and not np.allclose(out[0], E[0], atol=2e-3)
    # тот же энкодер → переиспользование без кодирования
    enc_a = _NamedEncoder("microsoft/unixcoder-base")
    out, info = embeddings_for_records_ex(cfg, enc_a, ids[:4], ["c"] * 4, ["python"] * 4, split="protected")
    assert info["n_reused"] == 4 and info["n_encoded"] == 0 and enc_a.calls == [] and np.allclose(out, E[:4], atol=2e-3)
    # фабрика: ожидаемая модель берётся из cfg (checkpoint/base_model), как в load_encoder
    cfg_b = _cfg(tmp_path, checkpoint="runs/finetuned/best")
    out, info = embeddings_for_records_ex(cfg_b, lambda: enc_b, ids[:2], ["c"] * 2, ["python"] * 2, split="protected")
    assert info["n_reused"] == 0 and info["n_encoded"] == 2 and len(enc_b.calls) == 2
    # энкодер, не совпадающий с запрошенной моделью, — ошибка, а не тихая подмена
    with pytest.raises(ValueError):
        embeddings_for_records_ex(cfg_b, enc_a, ["new/id"], ["c"], ["python"], split="protected", model="runs/finetuned/best")
    with pytest.raises(ValueError):
        embeddings_for_records_ex(cfg_b, lambda: enc_a, ["new/id"], ["c"], ["python"], split="protected")  # cfg ждёт finetuned
    # файл без meta.model не считается проверенным
    save_embeddings(tmp_path / "embeddings" / "hard_neg.npz", ids[:2], E[:2], key="ids", meta={})
    out, info = embeddings_for_records_ex(cfg, enc_a, ids[:2], ["c"] * 2, ["python"] * 2, split="hard_neg")
    assert info["n_reused"] == 0 and info["n_encoded"] == 2
    assert not cached_split_path(cfg, "protected", "runs/finetuned/best").exists()
    assert read_embeddings_meta(tmp_path / "nope.npz") == {}


def test_embed_split_replaces_stale_files_and_keeps_model_cache(tmp_path):
    """Блокер: 06 с другим чекпойнтом перекодирует активные файлы; старые остаются в by_model/<tag>."""
    from smcode.semantic.embed import cached_query_path, cached_split_path, embed_query_set, embed_split, read_embeddings_meta
    from smcode.types import write_jsonl

    cfg = _cfg(tmp_path)
    recs = _records(5, split="protected")
    write_jsonl(Path(cfg["paths"]["functions"]) / "protected.jsonl", recs)
    write_jsonl(Path(cfg["paths"]["queries"]) / "protected.jsonl",
                [{"qid": f"q{i}", "code": r.code, "lang": "python"} for i, r in enumerate(recs)])
    enc_a = _NamedEncoder("model-A", fill=0.1)
    enc_b = _NamedEncoder("model-B", fill=0.9)
    active = embed_split(cfg, "protected", enc_a)
    assert active == tmp_path / "embeddings" / "protected.npz" and read_embeddings_meta(active)["model"] == "model-A"
    assert cached_split_path(cfg, "protected", "model-A").exists()
    assert embed_split(cfg, "protected", enc_a) == active and len(enc_a.calls) == 1  # идемпотентно для той же модели
    # другая модель: активный файл заменяется, A остаётся в кэше
    embed_split(cfg, "protected", enc_b)
    assert read_embeddings_meta(active)["model"] == "model-B" and len(enc_b.calls) == 1
    assert read_embeddings_meta(cached_split_path(cfg, "protected", "model-A"))["model"] == "model-A"
    _, emb_b, _ = load_embeddings(active)
    # возврат к A: восстановление из кэша без кодирования
    embed_split(cfg, "protected", enc_a)
    assert read_embeddings_meta(active)["model"] == "model-A" and len(enc_a.calls) == 1
    _, emb_a, _ = load_embeddings(active)
    assert not np.allclose(emb_a, emb_b)
    # max_length тоже часть ключа
    enc_a_short = _NamedEncoder("model-A", fill=0.1, max_length=128)
    embed_split(cfg, "protected", enc_a_short)
    assert len(enc_a_short.calls) == 1 and read_embeddings_meta(active)["max_length"] == 128
    # наборы запросов: та же логика
    qp = embed_query_set(cfg, "protected", enc_b)
    assert qp is not None and read_embeddings_meta(qp)["model"] == "model-B" and cached_query_path(cfg, "protected", "model-B").exists()
    embed_query_set(cfg, "protected", enc_a)
    assert read_embeddings_meta(qp)["model"] == "model-A"
    assert embed_split(cfg, "public_test", enc_a) is None  # нет файла функций


def test_semantic_index_build_records_true_producer_model(tmp_path):
    """Блокер: meta.model индекса — модель, которой получены векторы, и чужие npz не подхватываются."""
    cfg = _cfg(tmp_path, checkpoint="model-A")
    recs = _records(6, split="protected")
    enc_a = _NamedEncoder("model-A", fill=0.2)
    E = enc_a.encode([r.code for r in recs])
    enc_a.calls.clear()
    save_embeddings(tmp_path / "embeddings" / "protected.npz", [r.id for r in recs], E, key="ids", meta={"model": "model-A"})
    idx = SemanticIndex(cfg)
    idx.set_encoder(enc_a)
    idx.build(recs, cfg)
    assert idx.meta["model"] == "model-A" and idx.meta["n_reused"] == 6 and idx.meta["n_encoded"] == 0 and enc_a.calls == []
    # файл другой модели: кодируется заново своим энкодером, meta.model — честный
    cfg_b = _cfg(tmp_path, checkpoint="model-B")
    enc_b = _NamedEncoder("model-B", fill=0.7)
    idx_b = SemanticIndex(cfg_b)
    idx_b.set_encoder(enc_b)
    idx_b.build(recs, cfg_b)
    assert idx_b.meta["model"] == "model-B" and idx_b.meta["n_reused"] == 0 and idx_b.meta["n_encoded"] == 6 and len(enc_b.calls) == 1
    # без энкодера индекс модели B не может «одолжить» векторы A
    idx_c = SemanticIndex(cfg_b)
    with pytest.raises(Exception):
        idx_c.build(recs, cfg_b)  # фабрика → load_encoder → нет torch (и не было бы реюза A)
    assert "torch" not in sys.modules


def test_combiner_path_respects_paths_indexes(tmp_path):
    """Major: путь комбинатора — из cfg.paths.indexes, с тегом модели; явный combiner_path относителен корню стенда."""
    from smcode.config import ROOT as PKG_ROOT
    from smcode.semantic.hybrid import combiner_path

    cfg = _cfg(tmp_path)
    cfg["semantic"].pop("combiner_path")
    p = combiner_path(cfg)
    assert p == tmp_path / "indexes" / "hybrid" / "combiner.pkl"
    assert combiner_path(cfg, "microsoft/unixcoder-base") == tmp_path / "indexes" / "hybrid" / "combiner_microsoft__unixcoder-base.pkl"
    assert combiner_path(cfg, "runs/x/best").name == "combiner_runs__x__best.pkl"
    assert not (PKG_ROOT / "data" / "indexes" / "hybrid" / "combiner.pkl").exists()
    cfg["semantic"]["combiner_path"] = "data/custom/comb.pkl"
    assert combiner_path(cfg, "any") == PKG_ROOT / "data" / "custom" / "comb.pkl"
    cfg["semantic"]["combiner_path"] = str(tmp_path / "abs.pkl")
    assert combiner_path(cfg) == tmp_path / "abs.pkl"


def test_stale_combiner_is_retrained_on_model_change(tmp_path):
    """Major: комбинатор другой модели/настроек не переиспользуется ни из файла, ни из индекса."""
    from smcode.semantic.hybrid import combiner_matches, train_combiner_from_corpus
    from smcode.types import write_jsonl

    cfg = _cfg(tmp_path)
    cfg["semantic"].pop("combiner_path")
    cfg["transforms"] = {**cfg["transforms"], "programmatic": ["identity", "rename_ids"], "partial_lines": [5]}
    cfg["semantic"].update({"combiner_n_index": 16, "combiner_n_anchors": 6, "combiner_n_neg": 6, "reuse_embeddings": False})
    recs = _records(12, split="public_train", repo="a/one") + _records(12, split="public_train", repo="b/two")
    for i, r in enumerate(recs):
        r.id, r.code = f"public_train/{r.repo}/f{i}", _py_fn(400 + i)
    write_jsonl(Path(cfg["paths"]["functions"]) / "public_train.jsonl", recs)

    class EncA(_HashEncoder):
        model_name = "model-A"

    class EncB(_HashEncoder):
        model_name = "runs/model-B-finetuned/best"

    comb_a = train_combiner_from_corpus(cfg, encoder=EncA())
    path_a = tmp_path / "indexes" / "hybrid" / "combiner_model-A.pkl"
    assert path_a.exists() and comb_a.train_stats["model"] == "model-A"
    # тот же файл, другая модель → переобучение (не возврат A)
    comb_b = train_combiner_from_corpus(cfg, encoder=EncB(), out_path=path_a)
    assert comb_b.train_stats["model"] == "runs/model-B-finetuned/best"
    assert Combiner.load(path_a).train_stats["model"] == "runs/model-B-finetuned/best"
    # сигнатура: k/w/top_k/C/key_id
    sig = {"model": "model-A", "k": 5, "w": 4, "key_id": None, "top_k": 5, "C": 1.0}
    assert combiner_matches(comb_a, sig)[0]
    assert not combiner_matches(comb_a, {**sig, "k": 25})[0] and not combiner_matches(comb_a, {**sig, "C": 0.5})[0]
    assert not combiner_matches(comb_a, {**sig, "key_id": "abcd"})[0] and not combiner_matches(Combiner("logistic"), sig)[0]
    # ensure_combiner у индекса модели B: кэш A игнорируется (другое имя файла и сигнатура), берётся/обучается B
    enc_b = EncB()
    h = HybridIndex(cfg)
    prot = _records(8)
    h.build_from_embeddings([r.id for r in prot], enc_b.encode([r.code for r in prot]), [r.code for r in prot], ["python"] * 8,
                            meta={"model": enc_b.model_name})
    got = h.ensure_combiner(cfg, encoder=enc_b)
    assert got.train_stats["model"] == "runs/model-B-finetuned/best"
    # индекс с устаревшим комбинатором внутри (модель A) переобучает при ensure_combiner
    h.set_combiner(comb_a)
    assert h.combiner is comb_a
    got2 = h.ensure_combiner(cfg, encoder=enc_b)
    assert got2 is not comb_a and got2.train_stats["model"] == "runs/model-B-finetuned/best"
    # а подходящий — оставляет без обучения; force переобучает
    assert h.ensure_combiner(cfg, encoder=enc_b) is got2
    got3 = h.ensure_combiner(cfg, encoder=enc_b, force=True)
    assert got3 is not got2 and got3.fitted
    cfg["semantic"]["combiner_C"] = 0.3
    h2 = HybridIndex(cfg)
    h2.build_from_embeddings([r.id for r in prot], enc_b.encode([r.code for r in prot]), [r.code for r in prot], ["python"] * 8,
                             meta={"model": enc_b.model_name})
    assert h2.ensure_combiner(cfg, encoder=enc_b).train_stats["C"] == pytest.approx(0.3)
    assert "torch" not in sys.modules


def test_hybrid_rule_from_config_honoured_on_load(tmp_path):
    """Major: hybrid_rule из cfg действует при load(); обученный комбинатор сохраняется для обратного переключения."""
    cfg, recs, M, h = _hybrid_fixture(tmp_path)
    rng = np.random.default_rng(12)
    q_emb = np.asarray([l2_normalize(M[i] + 0.2 * rng.standard_normal(M.shape[1]).astype(np.float32))[0] for i in range(30)]
                       + [_unit(rng, 1, M.shape[1])[0] for _ in range(30)], dtype=np.float32)
    q_codes = [r.code for r in recs] + [_py_fn(600 + i) for i in range(30)]
    X, y, _ = collect_features(h, q_emb, q_codes, ["python"] * 60, [1] * 30 + [0] * 30, [r.id for r in recs] + ["x"] * 30)
    h.set_combiner(Combiner(rule="logistic").fit(X, y))
    out = tmp_path / "indexes" / "hybrid"
    h.save(out)
    s_log, _, d_log = h.score_embeddings_with_codes(q_emb[:5], q_codes[:5], ["python"] * 5)
    # абляция по конфигурации, без пересборки
    cfg_tt = _cfg(tmp_path, hybrid_rule="two_threshold", hybrid_thresholds={"cos": 0.7, "overlap": 0.3})
    h_tt = HybridIndex(cfg_tt)
    h_tt.load(out)
    assert h_tt.rule == "two_threshold" and h_tt.combiner is not None and h_tt.combiner.rule == "two_threshold"
    assert h_tt.thresholds == {"cos": 0.7, "overlap": 0.3} and h_tt.meta["rule"] == "two_threshold"
    s_tt, _, d_tt = h_tt.score_embeddings_with_codes(q_emb[:5], q_codes[:5], ["python"] * 5)
    assert d_tt[0]["rule"] == "two_threshold" and d_tt[0]["combiner_fitted"] is False
    exp = two_threshold_score(np.array([make_features(d["cos"], d["overlap"], d["lcr_norm"], 0) for d in d_tt]), h_tt.thresholds)
    assert np.allclose(s_tt, exp, atol=1e-5)
    # обратное переключение: обученный логистический комбинатор сохранён в индексе
    h_tt.set_rule("logistic")
    assert h_tt.combiner is not None and h_tt.combiner.fitted and h_tt.meta["combiner_fitted"] is True
    s_back, _, _ = h_tt.score_embeddings_with_codes(q_emb[:5], q_codes[:5], ["python"] * 5)
    assert np.allclose(s_back, s_log, atol=2e-2)  # float16 на диске → cos ±1e-3 → логит ±(coef·1e-3)
    # загрузка с logistic в cfg → логистический
    h_log = HybridIndex(_cfg(tmp_path, hybrid_rule="logistic"))
    h_log.load(out)
    assert h_log.rule == "logistic" and h_log.combiner is not None and h_log.combiner.fitted
    # индекс, сохранённый с two_threshold, тоже хранит обученный комбинатор
    h_tt.set_rule("two_threshold")
    out2 = tmp_path / "indexes" / "hybrid2"
    h_tt.save(out2)
    h3 = HybridIndex(_cfg(tmp_path, hybrid_rule="logistic"))
    h3.load(out2)
    assert h3.rule == "logistic" and h3.combiner is not None and h3.combiner.fitted
    with pytest.raises(ValueError):
        h3.set_rule("nope")


def test_no_silent_rule_fallback(tmp_path):
    """Minor: без обученного комбинатора правило logistic — ошибка, откат только по allow_rule_fallback."""
    cfg, recs, M, h = _hybrid_fixture(tmp_path, rule="logistic")
    assert h.combiner is None
    with pytest.raises(RuntimeError):
        h.score_embeddings_with_codes(M[:2], [r.code for r in recs[:2]], ["python"] * 2)
    with pytest.raises(RuntimeError):  # нет public_train.jsonl → обучить нельзя
        h.ensure_combiner(cfg, encoder=_HashEncoder())
    cfg2, recs2, M2, h2 = _hybrid_fixture(tmp_path, rule="logistic")
    cfg2["semantic"]["allow_rule_fallback"] = True
    h2._configure(cfg2)
    scores, best, details = h2.score_embeddings_with_codes(M2[:2], [r.code for r in recs2[:2]], ["python"] * 2)
    assert details[0]["rule"] == "two_threshold" and h2.rule == "two_threshold" and "rule_fallback" in h2.meta
    assert best == [recs2[0].id, recs2[1].id]
    h3 = HybridIndex(cfg2)
    h3.build_from_embeddings([r.id for r in recs2[:3]], M2[:3], [r.code for r in recs2[:3]], ["python"] * 3, meta={"model": "fake"})
    assert h3.ensure_combiner(cfg2, encoder=_HashEncoder()).rule == "two_threshold" and h3.meta.get("rule_fallback")


def test_collect_features_drops_ambiguous_duplicates(tmp_path):
    """Minor: дубликат источника в индексе (overlap≈1) не получает метку 0."""
    cfg = _cfg(tmp_path, hybrid_rule="logistic")
    recs = _records(6)
    dup = FunctionRecord(**{**recs[0].to_dict(), "id": "protected/other/dup.py:1-10", "repo": "other/repo"})
    all_recs = recs + [dup]
    enc = _HashEncoder()
    E = enc.encode([r.code for r in all_recs])
    h = HybridIndex(cfg)
    h.build_from_embeddings([r.id for r in all_recs], E, [r.code for r in all_recs], ["python"] * 7, meta={"model": "fake-hash"})
    q = enc.encode([recs[0].code])
    stats: dict = {}
    X, y, g = collect_features(h, q, [recs[0].code], ["python"], [1], [recs[0].id], stats=stats)
    assert stats["n_dropped_ambiguous"] == 1 and y.sum() == 1 and len(y) == stats["n_rows"]
    assert not any((y == 0) & (X[:, 1] >= 0.9))
    X2, y2, _ = collect_features(h, q, [recs[0].code], ["python"], [1], [recs[0].id], drop_ambiguous=False)
    assert len(y2) == len(y) + 1 and ((y2 == 0) & (X2[:, 1] >= 0.9)).sum() == 1


def test_encoder_property_requires_checkpoint_for_own_model(tmp_path):
    """Minor: own_small_model.enabled без checkpoint — ошибка, а не молчаливый HF-базис."""
    cfg = _cfg(tmp_path)
    cfg["semantic"]["own_small_model"] = {**cfg["semantic"]["own_small_model"], "enabled": True}
    cfg["semantic"].pop("checkpoint", None)
    idx = SemanticIndex(cfg)
    with pytest.raises(ValueError):
        _ = idx.encoder
    assert "torch" not in sys.modules


def test_hf_batch_tokenizer_prefix_format():
    """Minor: формат входа UniXcoder [CLS] <encoder-only> [SEP] токены [SEP] и усечение до max_length−4."""
    from smcode.semantic.model import HFBatchTokenizer, default_input_prefix

    class FakeTok:
        cls_token_id, sep_token_id, unk_token_id, pad_token_id = 0, 2, 3, 1
        vocab = {"<encoder-only>": 9}

        def convert_tokens_to_ids(self, t):
            return self.vocab.get(t, self.unk_token_id)

        def __call__(self, texts, add_special_tokens=True, truncation=True, max_length=512, padding=False, return_tensors=None):
            ids = [[100 + i for i in range(len(t.split()))][:max_length] for t in texts]
            if add_special_tokens:
                ids = [[0] + x + [2] for x in ids]
            return {"input_ids": ids}

        def pad(self, enc, padding=True, return_tensors=None):
            L = max(len(x) for x in enc["input_ids"])
            ids = [x + [1] * (L - len(x)) for x in enc["input_ids"]]
            mask = [[1] * len(x) + [0] * (L - len(x)) for x in enc["input_ids"]]
            return {"input_ids": ids, "attention_mask": mask}

    bt = HFBatchTokenizer(FakeTok(), max_length=8, prefix="<encoder-only>")
    out = bt(["a b", "a b c d e f g h i j"])
    assert out["input_ids"][0] == [0, 9, 2, 100, 101, 2, 1, 1]  # [CLS] <encoder-only> [SEP] a b [SEP] + паддинг
    assert out["input_ids"][1] == [0, 9, 2, 100, 101, 102, 103, 2]  # усечение до max_length − 4 токенов тела
    assert out["attention_mask"][0] == [1] * 6 + [0] * 2
    plain = HFBatchTokenizer(FakeTok(), max_length=8, prefix="<unknown-mode>")  # неизвестный префикс → обычный вход
    assert plain.prefix == "" and plain(["a b"])["input_ids"][0] == [0, 100, 101, 2]
    assert default_input_prefix("microsoft/unixcoder-base") == "<encoder-only>" and default_input_prefix("codesage/codesage-small") == ""
    enc = CodeEncoder({"semantic": {"base_model": "microsoft/unixcoder-base"}})
    assert enc.input_prefix == "<encoder-only>" and enc.key == "microsoft/unixcoder-base"
    assert CodeEncoder({"semantic": {"base_model": "microsoft/unixcoder-base", "input_prefix": ""}}).input_prefix == ""
    assert "torch" not in sys.modules and "transformers" not in sys.modules


def test_train_holdout_selection_is_repo_level(tmp_path):
    """Minor: отбор эпохи — по отложенным репозиториям public_train; public_calib только для отчёта."""
    train = importlib.import_module("smcode.semantic.train")
    recs = []
    for repo, n in (("a/one", 10), ("b/two", 10), ("c/three", 10), ("d/four", 30)):
        rs = _records(n, split="public_train", repo=repo)
        for i, r in enumerate(rs):
            r.id, r.repo = f"public_train/{repo}/f{i}", repo
        recs += rs
    rng = random.Random(3)
    tr, held = train.select_holdout(recs, 12, rng)
    assert len(tr) + len(held) == len(recs) and len(held) >= 10
    assert not ({r.repo for r in tr} & {r.repo for r in held})  # репозитории не пересекаются
    cfg = _cfg(tmp_path)
    a, p = train.val_pairs_from_records(cfg, held, 5, random.Random(1))
    assert len(a) == len(p) == min(5, len(held)) and all(code for code, _ in p)
    tc = train.TrainConfig(max_steps=1).resolved(cfg)
    assert tc.val_on == "holdout"
    assert train.TrainConfig(val_on="public_calib").resolved(cfg).val_on == "public_calib"
    with pytest.raises(ValueError):
        train.TrainConfig(val_on="nope").resolved(cfg)
    one_repo = _records(8, split="public_train", repo="z/z")
    for r in one_repo:
        r.repo = "z/z"
    tr1, held1 = train.select_holdout(one_repo, 3, random.Random(0))
    assert len(held1) == 3 and len(tr1) == 5
    assert "torch" not in sys.modules


def test_latency_thread_counts_default_and_config(tmp_path):
    from smcode.semantic.embed import latency_thread_counts

    cfg = _cfg(tmp_path)
    assert latency_thread_counts(cfg) == [1, 4]
    cfg["eval"]["latency_threads"] = [8, 1, 1]
    assert latency_thread_counts(cfg) == [1, 8]
    cfg["eval"]["latency_threads"] = 2
    assert latency_thread_counts(cfg) == [2]


def test_scripts_fail_cleanly_without_torch(tmp_path):
    """06_embed.py: без torch — код возврата 2 и сообщение, а не traceback (в т. ч. --own-model без чекпойнта)."""
    import subprocess

    import yaml

    from smcode.types import write_jsonl

    cfg_path = tmp_path / "cfg.yaml"
    cfg = load_config()
    cfg.pop("_config_path", None)
    cfg["paths"] = {**cfg["paths"], "functions": str(tmp_path / "f"), "queries": str(tmp_path / "q"),
                    "embeddings": str(tmp_path / "e"), "results": str(tmp_path / "r"), "indexes": str(tmp_path / "i")}
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    write_jsonl(tmp_path / "f" / "protected.jsonl", _records(3))  # есть что кодировать → нужен torch
    for extra in ([], ["--own-model"]):
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "06_embed.py"), "--config", str(cfg_path), "--no-latency", *extra],
                              capture_output=True, text=True, cwd=str(ROOT), timeout=120)
        assert proc.returncode == 2, proc.stderr
        assert "Traceback" not in proc.stderr
