"""Тесты модуля приватности (smcode.privacy): проекция, квантование/шум, атаки A1/A2, прогон шага 08.

Синтетика: словарь из 40 частых и 40 редких идентификаторов; каждому идентификатору сопоставлен случайный
вектор, эмбеддинг функции = нормированная сумма векторов её идентификаторов + шум, код функции — короткая
функция Python с этими идентификаторами. Редкий идентификатор встречается ровно в трёх обучающих функциях
(df = 3 ≤ rare_df) и в одной функции protected. Без сети и без torch.
Регрессии после ревью: пороги на токен не стреляют на уровне случайности (per_token_thresholds), df словаря по
полному корпусу, пороги только по обучающим строкам, вложенные выборки утёкших пар, фиксированная подвыборка A3,
кластерный бутстрэп ДИ полезности, сетка σ шума.
"""

from __future__ import annotations

import importlib.util
import json
import py_compile
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import sparse

from smcode.config import load_config
from smcode.privacy import attack_align, attack_bow, attack_gen, projection, quantize
from smcode.privacy.run import (ClusterBoot, DefenceSpec, PrivacyInputs, apply_defence, default_defences, defence_memory, make_cluster_boots,
                                make_defence, parse_defence, query_cluster, rate_block, run_privacy, run_privacy_inputs, utility)

ROOT = Path(__file__).resolve().parents[1]
D = 64
N_COMMON, N_RARE = 40, 40
RARE_TRAIN_DF = 3
COMMON = [f"cm{k}" for k in range(N_COMMON)]
RARE = [f"rr{k}" for k in range(N_RARE)]
RATE_KEYS = {"value", "ci", "n", "k", "ci_cluster", "n_clusters"}


# ----------------------------------------------------------------------------- синтетика


def _norm(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


class Synth:
    """Синтетический корпус: токен-векторы, мешки идентификаторов, эмбеддинги, коды."""

    def __init__(self, seed: int = 0, n_train: int = 1200, n_prot: int = 240, n_neg: int = 300) -> None:
        self.rng = np.random.default_rng(seed)
        self.tok_vec = _norm(self.rng.standard_normal((N_COMMON + N_RARE, D)))
        self.index = {t: i for i, t in enumerate(COMMON + RARE)}
        self.train_bags = [self._bag() for _ in range(n_train)]
        for r in range(N_RARE):  # редкий идентификатор r встречается ровно в RARE_TRAIN_DF обучающих функциях
            for k in range(RARE_TRAIN_DF):
                self.train_bags[RARE_TRAIN_DF * r + k].append(RARE[r])
        self.prot_bags = [self._bag() + [RARE[i % N_RARE]] for i in range(n_prot)]
        self.neg_bags = {s: [self._bag() for _ in range(n_neg)] for s in ("public_calib", "public_test", "hard_neg")}
        self.train_emb = self.embed(self.train_bags)
        self.prot_emb = self.embed(self.prot_bags)
        self.train_codes = [self.code(b, f"tfn{i}") for i, b in enumerate(self.train_bags)]
        self.prot_codes = [self.code(b, f"pfn{i}") for i, b in enumerate(self.prot_bags)]
        q_prot = _norm(self.prot_emb + 0.04 * self.rng.standard_normal(self.prot_emb.shape).astype(np.float32))  # cos ≈ 0.95
        self.queries = {"protected": ([f"q{i}" for i in range(n_prot)], q_prot)}
        for s, bags in self.neg_bags.items():
            self.queries[s] = ([f"{s}{i}" for i in range(n_neg)], self.embed(bags))

    def _bag(self) -> list[str]:
        k = int(self.rng.integers(3, 6))
        return [COMMON[i] for i in self.rng.choice(N_COMMON, size=k, replace=False)]

    def embed(self, bags: list[list[str]]) -> np.ndarray:
        out = np.zeros((len(bags), D), dtype=np.float32)
        for i, b in enumerate(bags):
            out[i] = self.tok_vec[[self.index[t] for t in b]].sum(axis=0)
        return _norm(out + 0.12 * self.rng.standard_normal(out.shape).astype(np.float32))

    @staticmethod
    def code(bag: list[str], name: str) -> str:
        lines = [f"def {name}({bag[0]}):"]
        for a in bag[1:]:
            lines.append(f"    {a} = {bag[0]} + 1")
        lines.append(f"    return {bag[0]}")
        return "\n".join(lines) + "\n"

    def inputs(self) -> PrivacyInputs:
        return PrivacyInputs(
            index_ids=[f"protected/r/p{i}" for i in range(len(self.prot_bags))], index_emb=self.prot_emb, index_codes=self.prot_codes,
            index_langs=["python"] * len(self.prot_bags), train_ids=[f"public_train/r/t{i}" for i in range(len(self.train_bags))],
            train_emb=self.train_emb, train_codes=self.train_codes, train_langs=["python"] * len(self.train_bags),
            queries=self.queries, meta={"limit_eval": None},
        )


@pytest.fixture(scope="module")
def synth() -> Synth:
    return Synth()


@pytest.fixture(scope="module")
def cfg() -> dict:
    c = load_config()
    c["seed"] = 7
    c["privacy"] = {**c["privacy"], "bow_vocab": 200, "a1_backend": "ridge", "attack_leaked_pairs": [0, 100, 1000]}
    c["calibration"] = {"alphas": [0.01], "method": "split_conformal"}
    return c


@pytest.fixture(scope="module")
def a1(synth: Synth, cfg: dict):
    """Обученная A1 на plain-эмбеддингах + цели на protected."""
    train_bags = attack_bow.identifier_bags(synth.train_codes, ["python"] * len(synth.train_codes))
    eval_bags = attack_bow.identifier_bags(synth.prot_codes, ["python"] * len(synth.prot_codes))
    model, vocab, Y_train = attack_bow.train_bow_attack(synth.train_emb, train_bags, cfg, rng=random.Random(1), backend="ridge")
    Y_eval = attack_bow.targets(eval_bags, vocab)
    n_val = Y_train.shape[0] // 10 + 1
    prior = attack_bow.prior_baseline(Y_train[n_val:], Y_train[:n_val], Y_eval, vocab)
    return model, vocab, Y_eval, prior


def _random_targets(n: int, v: int, rng: np.random.Generator, lo: int = 2, hi: int = 3) -> sparse.csr_matrix:
    """Бинарные цели: у каждого токена lo..hi случайных положительных строк."""
    rows, cols = [], []
    for j in range(v):
        for i in rng.choice(n, size=int(rng.integers(lo, hi + 1)), replace=False):
            rows.append(int(i))
            cols.append(j)
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, v))


# ----------------------------------------------------------------------------- проекция


def test_keyed_orthogonal_properties():
    Q = projection.keyed_orthogonal(D, b"secret")
    assert Q.shape == (D, D) and Q.dtype == np.float32
    assert projection.is_orthogonal(Q, tol=1e-4)
    assert np.array_equal(Q, projection.keyed_orthogonal(D, b"secret"))
    Q2 = projection.keyed_orthogonal(D, b"other")
    assert attack_align.relative_error(Q2, Q) > 1.0
    x = np.random.default_rng(0).standard_normal((10, D)).astype(np.float32)
    assert np.allclose(projection.invert(projection.apply(x, Q), Q), x, atol=1e-4)
    with pytest.raises(ValueError):
        projection.apply(x, projection.keyed_orthogonal(8, b"k"))


def test_projection_key_fallbacks(monkeypatch):
    c = {"seed": 3, "privacy": {}, "fingerprint": {"hmac_key_env": "SMCODE_TEST_HMAC_X"}}
    monkeypatch.delenv("SMCODE_PROJECTION_KEY", raising=False)
    monkeypatch.delenv("SMCODE_TEST_HMAC_X", raising=False)
    k0 = projection.projection_key(c)
    assert k0 == projection.projection_key(c)
    monkeypatch.setenv("SMCODE_TEST_HMAC_X", "hmac-key")
    assert projection.projection_key(c) == b"hmac-key"
    monkeypatch.setenv("SMCODE_PROJECTION_KEY", "proj-key")
    assert projection.projection_key(c) == b"proj-key"


def test_projection_preserves_cosines_and_utility(synth: Synth, cfg: dict):
    Q = projection.keyed_orthogonal(D, b"k1")
    A = synth.prot_emb[:50]
    B = synth.queries["public_test"][1][:50]
    plain = A @ B.T
    proj = projection.apply(A, Q) @ projection.apply(B, Q).T
    assert np.allclose(plain, proj, atol=1e-5)
    inp = synth.inputs()
    none, prj = make_defence(), make_defence(projection=True)
    u0 = utility(none, inp, Q, apply_defence(inp.index_emb, none, Q, 0), [0.01])
    u1 = utility(prj, inp, Q, apply_defence(inp.index_emb, prj, Q, 0), [0.01])
    assert u0["tpr"]["semantic"]["value"] == pytest.approx(u1["tpr"]["semantic"]["value"], abs=1e-6)
    assert u0["threshold"] == pytest.approx(u1["threshold"], abs=1e-4)
    assert u0["tpr"]["semantic"]["value"] > 0.9  # члены (слегка возмущённые) находятся при FPR = 1 %
    assert u0["fpr"]["public_test"]["value"] <= 0.03
    assert set(u0["tpr"]["semantic"]) == RATE_KEYS and u0["tpr"]["semantic"]["ci_cluster"] is None  # без boots — нет кластерного ДИ


# ----------------------------------------------------------------------------- квантование и шум


def test_quantize_changes_similarity_slightly(synth: Synth):
    X = synth.prot_emb[:80]
    ref = X @ X.T
    bounds = {8: 0.02, 4: 0.1}
    for bits, tol in bounds.items():
        codes, deq = quantize.quantize(X, bits)
        assert deq.shape == X.shape and deq.dtype == np.float32
        sims = _norm(deq) @ _norm(deq).T
        delta = np.max(np.abs(sims - ref))
        assert 0 < delta < tol, (bits, delta)
    codes, deq = quantize.quantize(X, 1)
    assert codes.dtype == np.uint8 and codes.shape == (80, D // 8)
    assert np.allclose(np.linalg.norm(deq, axis=1), 1.0, atol=1e-5)
    sims = deq @ deq.T
    corr = np.corrcoef(sims[np.triu_indices(80, 1)], ref[np.triu_indices(80, 1)])[0, 1]
    assert corr > 0.6 and np.max(np.abs(sims - ref)) > 0.05  # знаковое квантование в 64-d: грубо, но коррелирует
    q4 = quantize.quantize_full(X, 4)
    assert q4.codes.dtype == np.uint8 and q4.codes.shape == (80, D // 2)
    raw = np.clip(np.rint(X / q4.scale.astype(np.float32)[:, None]), -7, 7).astype(np.int8)
    assert np.array_equal(quantize.unpack_int4(quantize.pack_int4(raw), D), raw)
    with pytest.raises(ValueError):
        quantize.quantize(X, 3)


def test_memory_ratios():
    assert quantize.memory_ratio(768, None) == 1.0
    assert quantize.memory_ratio(768, 8) == pytest.approx((768 + 2) / 1536)
    assert quantize.memory_ratio(768, 4) == pytest.approx((384 + 2) / 1536)
    assert quantize.memory_ratio(768, 1) == pytest.approx(96 / 1536)
    ratio, nbytes = defence_memory(make_defence(projection=True, bits=1), 1000, 768)
    assert ratio == pytest.approx(0.0625) and nbytes == 96000
    assert defence_memory(make_defence(projection=True, sigma=0.1), 10, 768) == (1.0, 10 * 1536)


def test_add_noise_renormalizes_and_matches_expected_cosine(synth: Synth):
    X = synth.prot_emb
    same = quantize.add_noise(X, 0.0, 1)
    assert np.allclose(same, X, atol=1e-6)
    for sigma in (0.5, 1.0):
        Y = quantize.add_noise(X, sigma, random.Random(5))
        assert np.allclose(np.linalg.norm(Y, axis=1), 1.0, atol=1e-5)
        cos = np.sum(X * Y, axis=1).mean()
        assert abs(cos - quantize.expected_cosine(sigma)) < 0.05
    Yabs = quantize.add_noise(X, 0.05, 3, mode="absolute")
    assert abs(np.sum(X * Yabs, axis=1).mean() - quantize.expected_cosine(0.05, "absolute", D)) < 0.05
    assert np.array_equal(quantize.add_noise(X, 0.3, 11), quantize.add_noise(X, 0.3, 11))  # детерминизм по seed
    with pytest.raises(ValueError):
        quantize.add_noise(X, 0.1, 0, mode="weird")


# ----------------------------------------------------------------------------- A1


def test_bow_vocab_and_targets(synth: Synth):
    bags = attack_bow.identifier_bags(synth.train_codes[:4], ["python"] * 4)
    assert bags[0] == set(synth.train_bags[0]) | {"tfn0"}
    all_bags = attack_bow.identifier_bags(synth.train_codes, ["python"] * len(synth.train_codes))
    vocab = attack_bow.build_vocab(all_bags, max_size=200, min_df=2, rare_df=3, rare_fraction=0.25)
    assert set(COMMON) <= set(vocab.tokens)
    assert vocab.n_rare == N_RARE and set(RARE) <= set(vocab.tokens)  # все редкие (df = 3) попали в квоту
    assert not any(t.startswith("tfn") for t in vocab.tokens)  # имена функций (df = 1) исключены
    assert vocab.n_docs == len(all_bags) and vocab.min_df == 2
    Y = attack_bow.targets(all_bags, vocab)
    assert Y.shape == (len(all_bags), len(vocab)) and Y.nnz == sum(len(set(b)) for b in synth.train_bags)
    small = attack_bow.build_vocab(all_bags, max_size=50, rare_fraction=0.2)
    assert len(small) == 50 and small.n_rare == 10
    rt = attack_bow.Vocab.from_dict(vocab.to_dict())
    assert rt.tokens == vocab.tokens and np.array_equal(rt.df, vocab.df) and rt.n_docs == vocab.n_docs
    assert attack_bow.identifier_tokens("def f(:", "nolang") == []
    assert attack_bow.identifier_df(synth.train_codes[:4], ["python"] * 4) == attack_bow.document_frequency(bags)


def test_build_vocab_uses_full_corpus_df(synth: Synth):
    """Регрессия: «редкий» определяется по df полного корпуса public_train, а не подвыборки атакующего;
    токены, отсутствующие в обучающей подвыборке, в словарь не попадают."""
    all_bags = attack_bow.identifier_bags(synth.train_codes, ["python"] * len(synth.train_codes))
    sub = all_bags[:40]  # маленькая подвыборка атакующего: часть частых токенов встречается в ней лишь 2–3 раза
    df_full = dict(attack_bow.document_frequency(all_bags))
    df_full["cm0"] = 10  # частый в полном корпусе
    sub = [set(b) for b in sub]
    sub[0].add("onlysub")  # df = 1 в полном корпусе (и в подвыборке) → исключён
    df_full["onlysub"] = 1
    df_full["absent3"] = 3  # редкий в корпусе, но не встречается в подвыборке → необучаем, исключён
    v_full = attack_bow.build_vocab(sub, max_size=200, df=df_full, n_docs=len(all_bags))
    v_sub = attack_bow.build_vocab(sub, max_size=200)
    assert v_full.n_docs == len(all_bags) and v_sub.n_docs == len(sub)
    assert "onlysub" not in v_full.tokens and "absent3" not in v_full.tokens
    assert not v_full.rare_mask[v_full.index["cm0"]]  # df по корпусу = 10 → не редкий
    assert int(v_full.df[v_full.index["cm0"]]) == 10
    present = attack_bow.document_frequency(sub)
    rare_full = {t for t, r in zip(v_full.tokens, v_full.rare_mask) if r}
    assert rare_full == {t for t, c in df_full.items() if 2 <= c <= 3 and present.get(t, 0) > 0}
    # по подвыборке частый токен может выглядеть редким, по корпусу — нет
    sub_rare = {t for t, r in zip(v_sub.tokens, v_sub.rare_mask) if r}
    assert sub_rare - rare_full, "в подвыборке есть токены, редкие только из-за подвыборки"
    assert all(df_full[t] > 3 for t in sub_rare - rare_full)
    assert rare_full - sub_rare and all(present[t] == 1 for t in rare_full - sub_rare)  # 1 раз в подвыборке: по корпусу редкий и обучаемый


def test_per_token_thresholds_do_not_fire_at_chance():
    """Регрессия (major): при случайных скорах пороги токенов с 2–3 положительными примерами почти никогда не конечны,
    а конечные стреляют на ≤ 1 % строк; старое правило (любой F1 > 0) стреляло на ~30 % строк."""
    rng = np.random.default_rng(0)
    n, v = 3000, 300
    S = rng.standard_normal((n, v)).astype(np.float32)
    Y = _random_targets(n, v, rng)
    thr = attack_bow.per_token_thresholds(lambda lo, hi: S[:, lo:hi], Y)
    finite = np.flatnonzero(np.isfinite(thr))
    assert finite.size <= 0.03 * v, finite.size
    assert all((S[:, j] > thr[j]).mean() <= 0.01 for j in finite)
    thr_old = attack_bow.per_token_thresholds(lambda lo, hi: S[:, lo:hi], Y, min_precision=0.0, min_f1=0.0)
    assert np.isfinite(thr_old).all() and np.median([(S[:, j] > thr_old[j]).mean() for j in range(v)]) > 0.1
    # на новых случайных скорах старое правило «восстанавливает» ~30 % редких токенов ровно на уровне эталона случайности
    S_eval, Y_eval = rng.standard_normal((n, v)).astype(np.float32), _random_targets(n, v, rng)
    vocab = attack_bow.Vocab(tokens=[f"t{j}" for j in range(v)], df=np.full(v, 2))
    m_old = attack_bow.bow_metrics(lambda lo, hi: S_eval[lo:hi] > thr_old[None, :], Y_eval, vocab)
    m_new = attack_bow.bow_metrics(lambda lo, hi: S_eval[lo:hi] > thr[None, :], Y_eval, vocab)
    assert m_old["rare_id_recall"] > 0.2 and abs(m_old["rare_id_recall"] - m_old["rare_id_recall_chance"]) < 0.05, m_old
    assert m_old["rare_id_precision"] < 0.01 and m_old["rare_pred_per_fn"] > 50
    assert m_new["rare_id_recall"] < 0.03 and m_new["rare_fire_rate"] < 0.001 and m_new["rare_pred_per_fn"] < 0.1
    # с сигналом: положительные строки имеют наибольшие скоры → порог конечен и предсказывает ровно их
    S2 = S.copy()
    S2[Y.toarray() > 0] += 10.0
    thr2 = attack_bow.per_token_thresholds(lambda lo, hi: S2[:, lo:hi], Y)
    assert np.isfinite(thr2).all()
    assert np.array_equal(S2 > thr2[None, :], Y.toarray() > 0)


def test_a1_on_plain_embeddings_beats_chance(synth: Synth, a1):
    model, vocab, Y_eval, prior = a1
    m = model.evaluate(synth.prot_emb, Y_eval)
    assert m["backend"] == "ridge" and m["n_eval"] == len(synth.prot_bags)
    assert m["f1"] > prior["f1"] + 0.3, (m["f1"], prior["f1"])
    assert m["rare_id_rule"] == "per_token" and m["n_rare_vocab"] == N_RARE
    assert m["rare_id_recall"] > 0.4 and m["rare_id_precision"] > 0.5, (m["rare_id_recall"], m["rare_id_precision"])
    assert m["rare_id_recall_chance"] < 0.05 and m["rare_id_recall"] > 5 * m["rare_id_recall_chance"]
    assert m["rare_id_recall_global"] == 0.0  # один глобальный порог редкие токены не предсказывает
    assert m["decision"] in ("global", "per_token") and set(model.val_f1_rules) == {"global", "per_token"}
    assert m["f1"] == pytest.approx(m[f"f1_{m['decision']}"]) and max(m["f1_global"], m["f1_per_token"]) > 0.8
    assert m["n_rare_true"] == len(synth.prot_bags)  # по одному редкому идентификатору на функцию
    assert m["thr_min_precision"] == 0.1 and m["thr_min_f1"] == 0.1
    assert prior["rare_id_recall"] == 0.0
    assert 0.0 <= m["mean_jaccard"] <= 1.0 and 0.0 <= m["exact_set_rate"] <= 1.0
    # прямое применение к спроецированным векторам без ключа — уровень случайности и по F1, и по редким токенам
    Q = projection.keyed_orthogonal(D, b"k1")
    blind = model.evaluate(projection.apply(synth.prot_emb, Q), Y_eval)
    assert blind["f1"] < prior["f1"] + 0.1
    assert blind["rare_id_recall"] < 0.1 and (blind["rare_id_precision"] or 0.0) < 0.2, (blind["rare_id_recall"], blind["rare_id_precision"])
    assert blind["rare_id_recall"] < blind["rare_id_recall_chance"] + 0.05 and blind["rare_fire_rate"] < 0.05


def test_thresholds_fitted_on_train_rows_only(synth: Synth):
    """Регрессия: пороги на токен подбираются без строк валидации — токен, встречающийся только в валидации,
    получает порог +inf, а val_f1_rules['per_token'] считается на невиданных строках."""
    bags = attack_bow.identifier_bags(synth.train_codes, ["python"] * len(synth.train_codes))
    vocab = attack_bow.build_vocab(bags, 200)
    model = attack_bow.BowInverter(vocab, backend="ridge").fit(synth.train_emb, attack_bow.targets(bags, vocab), rng=random.Random(0))
    val_rows = np.setdiff1d(np.arange(len(bags)), model._tr_idx)
    assert val_rows.size >= 3
    j = vocab.index[RARE[0]]
    Y2 = attack_bow.targets(bags, vocab).tolil()
    Y2[:, j] = 0
    Y2[val_rows[:3], j] = 1  # положительные только в валидации
    m2 = attack_bow.BowInverter(vocab, backend="ridge").fit(synth.train_emb, Y2.tocsr(), rng=random.Random(0))
    assert np.array_equal(m2._tr_idx, model._tr_idx) and not np.isfinite(m2.thresholds[j])
    assert np.isfinite(model.thresholds[j])  # с положительными в обучении порог конечен
    assert 0.0 <= m2.val_f1_rules["per_token"] <= 1.0


def test_ridge_leverage_matches_reference(synth: Synth):
    """Рычаги h_ii через одно BLAS-умножение совпадают с einsum-эталоном."""
    bags = attack_bow.identifier_bags(synth.train_codes[:300], ["python"] * 300)
    vocab = attack_bow.build_vocab(bags, 100, min_df=1)
    head = attack_bow._RidgeHead(1.0).fit(synth.train_emb[:300], attack_bow.targets(bags, vocab))
    Xc = synth.train_emb[:300].astype(np.float64) - head.mu
    ref = np.clip(1.0 / head.n + np.einsum("ij,jk,ik->i", Xc, head.G_inv, Xc), 0.0, 0.99)
    assert np.allclose(head.leverage(synth.train_emb[:300]), ref, atol=1e-5)


def test_run_attack_bow_end_to_end(synth: Synth, cfg: dict):
    metrics, model, vocab = attack_bow.run_attack_bow(synth.train_emb, synth.train_codes, ["python"] * len(synth.train_codes), synth.prot_emb,
                                                      synth.prot_codes, ["python"] * len(synth.prot_codes), cfg, rng=random.Random(2), backend="ridge")
    assert metrics["vocab_size"] == len(vocab) and metrics["n_rare_vocab"] == N_RARE
    assert metrics["f1"] > metrics["prior_baseline"]["f1"]


# ----------------------------------------------------------------------------- A2


def test_a2_recovers_projection_with_leaked_pairs(synth: Synth, a1):
    model, vocab, Y_eval, prior = a1
    Q = projection.keyed_orthogonal(D, b"k2")
    spec = make_defence(projection=True)
    pool_def = apply_defence(synth.train_emb, spec, Q, 0)
    target = apply_defence(synth.prot_emb, spec, Q, 0)
    curve = attack_align.run_attack_align(model, synth.train_emb, pool_def, target, Y_eval, [0, 1000], random.Random(3), Q_true=Q,
                                          target_plain=synth.prot_emb)
    assert set(curve) == {"0", "1000"}
    assert curve["0"]["q_rel_error"] > 1.0 and curve["0"]["f1"] < prior["f1"] + 0.1
    assert curve["1000"]["n_effective"] == 1000 and curve["1000"]["q_rel_error"] < 1e-2
    assert curve["1000"]["recovered_cos"] > 0.999
    plain_f1 = model.evaluate(synth.prot_emb, Y_eval)["f1"]
    assert abs(curve["1000"]["f1"] - plain_f1) < 0.02
    # МНК при n = 1000 > d тоже восстанавливает Q
    Q_hat = attack_align.estimate_projection(synth.train_emb[:1000], pool_def[:1000], method="lstsq")
    assert attack_align.relative_error(Q_hat, Q) < 1e-2
    with pytest.raises(ValueError):
        attack_align.estimate_projection(synth.train_emb[:10], pool_def[:10], method="magic")


def test_leaked_pairs_are_nested_across_n(synth: Synth):
    """Регрессия: точки кривой утечки используют вложенные выборки пар (одна перестановка пула)."""
    m = synth.train_emb.shape[0]
    perm = attack_align.leak_permutation(m, random.Random(3))
    assert sorted(perm.tolist()) == list(range(m))
    assert np.array_equal(perm, attack_align.leak_permutation(m, random.Random(3)))
    p100, _ = attack_align.sample_leaked_pairs(synth.train_emb, synth.train_emb, 100, perm=perm)
    p1000, d1000 = attack_align.sample_leaked_pairs(synth.train_emb, synth.train_emb, 1000, perm=perm)
    assert p100.shape == (100, D) and p1000.shape == (1000, D)
    assert np.array_equal(p1000, synth.train_emb[np.sort(perm[:1000])]) and np.array_equal(d1000, p1000)
    rows1000 = {r.tobytes() for r in p1000}
    assert all(r.tobytes() in rows1000 for r in p100)  # выборка для n = 100 ⊂ выборки для n = 1000
    with pytest.raises(ValueError):
        attack_align.sample_leaked_pairs(synth.train_emb, synth.train_emb, 10, perm=perm[:-1])
    # run_attack_align: та же перестановка → те же Q̂ (детерминизм), и без perm — из rng
    a1_model = attack_bow.BowInverter(attack_bow.Vocab(tokens=["a"], df=np.array([5])), backend="ridge")
    a1_model.head.W, a1_model.head.b, a1_model.thresholds = np.zeros((D, 1), np.float32), np.zeros(1, np.float32), np.full(1, np.inf, np.float32)
    Y = sparse.csr_matrix((len(synth.prot_emb), 1), dtype=np.float32)
    Q = projection.keyed_orthogonal(D, b"k5")
    pool_def = apply_defence(synth.train_emb, make_defence(projection=True, sigma=1.0), Q, 0)
    c1 = attack_align.run_attack_align(a1_model, synth.train_emb, pool_def, synth.prot_emb, Y, [100, 1000], None, Q_true=Q, perm=perm)
    c2 = attack_align.run_attack_align(a1_model, synth.train_emb, pool_def, synth.prot_emb, Y, [100, 1000], random.Random(3), Q_true=Q)
    assert c1["100"]["q_rel_error"] == pytest.approx(c2["100"]["q_rel_error"]) and c1["1000"]["q_rel_error"] < c1["100"]["q_rel_error"]


def test_a2_under_quantization_and_noise(synth: Synth, a1):
    model, vocab, Y_eval, prior = a1
    Q = projection.keyed_orthogonal(D, b"k3")
    for spec, max_err in ((make_defence(projection=True, bits=8), 0.05), (make_defence(projection=True, sigma=0.5), 0.2)):
        pool_def = apply_defence(synth.train_emb, spec, Q, 1)
        p, dfd = attack_align.sample_leaked_pairs(synth.train_emb, pool_def, 1000, 4)
        Q_hat = attack_align.estimate_projection(p, dfd)
        assert projection.is_orthogonal(Q_hat, tol=1e-3)
        assert attack_align.relative_error(Q_hat, Q) < max_err, spec.name
    p, dfd = attack_align.sample_leaked_pairs(synth.train_emb, synth.train_emb, 5000, 0)
    assert p.shape[0] == synth.train_emb.shape[0]  # обрезка до размера пула
    assert attack_align.sample_leaked_pairs(synth.train_emb, synth.train_emb, 0, 0)[0].shape == (0, D)
    rec = attack_align.recover(synth.prot_emb @ (2.0 * Q).T, 2.0 * Q)  # неортогональная Q̂ → псевдообратная
    assert np.allclose(rec, synth.prot_emb, atol=1e-3)


# ----------------------------------------------------------------------------- полезность: кластерный бутстрэп


def test_cluster_bootstrap_ci(synth: Synth):
    """ДИ полезности: ci — Клоппер–Пирсон по запросам, ci_cluster — бутстрэп по исходным функциям (qid до '#')."""
    assert query_cluster("protected/r/a.c:1-9#rename_ids#0") == "protected/r/a.c:1-9" and query_cluster("q7") == "q7"
    qids = [f"f{i // 13}#t#{i % 13}" for i in range(13 * 200)]
    boot = ClusterBoot([query_cluster(q) for q in qids], 1000, np.random.default_rng(0))
    assert boot.C == 200 and boot.counts.shape == (1000, 200)
    scores = np.array([1.0 if (i // 13) % 5 else 0.0 for i in range(len(qids))])  # решение постоянно внутри кластера
    rb = rate_block(scores, 0.5, boot)
    assert set(rb) == RATE_KEYS and rb["value"] == pytest.approx(0.8) and rb["n_clusters"] == 200
    lo, hi = rb["ci_cluster"]
    assert lo < rb["ci"][0] and hi > rb["ci"][1]  # кластерный ДИ шире биномиального по запросам
    assert (hi - lo) > 2 * (rb["ci"][1] - rb["ci"][0])
    assert rate_block(scores, np.inf, boot)["value"] == 0.0 and rate_block(scores, 0.5, None)["ci_cluster"] is None
    assert ClusterBoot([], 10, np.random.default_rng(0)).rate_ci(np.zeros(0, bool)) is None
    inp = synth.inputs()
    boots = make_cluster_boots(inp, 200, random.Random(1))
    assert set(boots) == set(inp.queries) and boots["protected"].C == len(inp.queries["protected"][0])
    u = utility(make_defence(), inp, None, inp.index_emb, [0.01], boots=boots)
    assert u["tpr"]["semantic"]["ci_cluster"] is not None and u["fpr"]["public_test"]["ci_cluster"] is not None
    assert u["tpr"]["semantic"]["ci_cluster"][0] <= u["tpr"]["semantic"]["value"] <= u["tpr"]["semantic"]["ci_cluster"][1]


# ----------------------------------------------------------------------------- прогон §9 и схема privacy.json


def test_defence_specs(cfg: dict):
    """Сетка σ: неинформативные σ конфига (cos > 0.97) дополняются DEFAULT_NOISE_SIGMA; информативные — нет."""
    names = [d.name for d in default_defences(cfg)]
    assert names == ["none", "proj", "proj+int8", "proj+int4", "proj+bin", "proj+noise0.05", "proj+noise0.1", "proj+noise0.2",
                     "proj+noise0.25", "proj+noise0.5", "proj+noise1", "proj+noise2"]
    assert [d.name for d in default_defences(cfg, extend_noise=False)][5:] == ["proj+noise0.05", "proj+noise0.1", "proj+noise0.2"]
    wide = {**cfg, "privacy": {**cfg["privacy"], "noise_sigma": [0.0, 0.5, 1.0]}}
    assert [d.name for d in default_defences(wide)][5:] == ["proj+noise0.5", "proj+noise1"]
    assert [d.name for d in default_defences({**cfg, "privacy": {**cfg["privacy"], "noise_sigma": [0.0]}})][5:] == []
    absent = {**cfg, "privacy": {k: v for k, v in cfg["privacy"].items() if k != "noise_sigma"}}
    assert [d.name for d in default_defences(absent)][5:] == ["proj+noise0.25", "proj+noise0.5", "proj+noise1", "proj+noise2"]
    absolute = {**cfg, "privacy": {**cfg["privacy"], "noise_mode": "absolute", "noise_sigma": [0.001]}}
    ext = [d.name for d in default_defences(absolute, d=D)][5:]
    assert ext[0] == "proj+noise0.001" and len(ext) == 5 and all(parse_defence(n).sigma < 0.3 for n in ext)  # σ_abs = σ_rel/√d
    assert [d.name for d in default_defences(absolute)][5:] == ["proj+noise0.001"]  # без d проверка невозможна
    assert all(quantize.expected_cosine(parse_defence(n).sigma) < 0.98 for n in names[8:])
    assert quantize.expected_cosine(parse_defence(names[-1]).sigma) < 0.5
    with pytest.raises(ValueError):
        default_defences({**cfg, "privacy": {**cfg["privacy"], "noise_mode": "weird"}})
    assert parse_defence("proj+noise0.1") == DefenceSpec("proj+noise0.1", True, None, 0.1)
    assert parse_defence("int4") == make_defence(bits=4)
    assert parse_defence("none").name == "none"
    with pytest.raises(ValueError):
        parse_defence("proj+int3")
    with pytest.raises(ValueError):
        apply_defence(np.zeros((2, 4), np.float32), make_defence(projection=True), None, 0)


def test_run_privacy_inputs_schema(synth: Synth, cfg: dict):
    specs = [make_defence(), make_defence(projection=True), make_defence(projection=True, bits=1), make_defence(projection=True, sigma=1.0)]
    res = run_privacy_inputs(cfg, synth.inputs(), defences=specs, leaked_pairs=[0, 1000], a1_backend="ridge", key=b"k9")
    assert res["schema"] == "smcode.privacy/1" and res["leaked_pairs"] == [0, 1000] and res["dim"] == D
    assert res["a1"]["backend"] == "ridge" and res["a1"]["n_rare_vocab"] == N_RARE and res["a3"] is None
    assert res["a1"]["rare_df_range"] == [2, 3] and res["a1"]["df_corpus_size"] == len(synth.train_bags) and res["bootstrap"] == 1000
    assert res["a1"]["thr_min_precision"] == 0.1 and 0 < res["a1"]["n_tokens_firing"] <= res["a1"]["vocab_size"]
    by = {d["name"]: d for d in res["defenses"]}
    assert list(by) == ["none", "proj", "proj+bin", "proj+noise1"]
    for d in by.values():
        assert set(d["tpr"]) == {"semantic", "hybrid"} and d["tpr"]["hybrid"] is None
        assert set(d["tpr"]["semantic"]) == RATE_KEYS and d["tpr"]["semantic"]["ci_cluster"] is not None
        assert set(d["fpr"]["hard_neg"]) == RATE_KEYS
        assert {"f1", "precision", "recall", "rare_id_recall", "rare_id_precision", "rare_id_recall_chance", "rare_fire_rate"} <= set(d["attacks"]["A1"])
        assert set(d["attacks"]["A2"]) == {"0", "1000"} and "f1" in d["attacks"]["A2"]["1000"]
        assert d["attacks"]["A3"] is None and "memory_ratio" in d
        assert d["noise_mode"] == ("relative" if d["sigma"] > 0 else None)
    prior_f1 = res["a1"]["prior_baseline"]["f1"]
    assert by["none"]["attacks"]["A1"]["f1"] > prior_f1 + 0.3
    assert by["none"]["attacks"]["A1"]["rare_id_recall"] > 0.4 and by["none"]["attacks"]["A1"]["rare_id_recall_chance"] < 0.05
    assert by["proj"]["attacks"]["A1"]["f1"] < prior_f1 + 0.1  # без ключа — случайность
    assert by["proj"]["attacks"]["A1"]["rare_id_recall"] < 0.1  # … и по редким идентификаторам (регрессия major)
    assert by["proj"]["attacks"]["A2"]["1000"]["f1"] > prior_f1 + 0.3  # с утечкой — восстановлено
    assert by["proj"]["attacks"]["A2"]["1000"]["rare_id_recall"] > 0.4
    assert by["proj"]["tpr"]["semantic"]["value"] == pytest.approx(by["none"]["tpr"]["semantic"]["value"], abs=1e-6)
    assert by["proj+bin"]["memory_ratio"] == pytest.approx(0.0625)
    assert by["proj+noise1"]["expected_cos"] == pytest.approx(1 / np.sqrt(2))
    assert by["proj+noise1"]["attacks"]["A2"]["1000"]["recovered_cos"] < by["proj"]["attacks"]["A2"]["1000"]["recovered_cos"]
    json.dumps(res)  # сериализуемо
    tables = pytest.importorskip("smcode.eval.tables")
    t6 = getattr(tables, "table_T6", None)
    if t6 is not None:
        md = t6({}, res)
        assert isinstance(md, str) and "proj+bin" in md and "T6" in md


def test_a3_eval_subset_fixed_and_skipped_without_torch(synth: Synth, cfg: dict):
    """Регрессия: подвыборка protected для A3 фиксируется один раз (одна и та же для всех защит); без torch A3 → null."""
    c3 = {**cfg, "privacy": {**cfg["privacy"], "a3": {"eval_max": 50}}}
    s1 = attack_gen.eval_subset(240, c3, random.Random(5))
    assert s1.shape == (50,) and np.array_equal(s1, np.sort(s1)) and np.unique(s1).size == 50
    assert np.array_equal(s1, attack_gen.eval_subset(240, c3, random.Random(5)))
    assert np.array_equal(attack_gen.eval_subset(30, c3, random.Random(5)), np.arange(30))
    if attack_bow.torch_available():
        pytest.skip("torch установлен — проверка пропуска A3 не применима")
    res = run_privacy_inputs(c3, synth.inputs(), defences=[make_defence(projection=True)], leaked_pairs=[0, 100], key=b"k", with_a3=True)
    assert "A3 skipped: torch not available" in res["errors"] and res["a3"] is None
    assert res["defenses"][0]["attacks"]["A3"] is None and res["defenses"][0]["attacks"]["A2"]["100"]["n_effective"] == 100


def _write_npz(path: Path, key: str, ids: list[str], emb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: np.asarray(ids, dtype=str), "emb": emb.astype(np.float16)})


def _file_cfg(tmp_path: Path, synth: Synth, cfg: dict) -> dict:
    c = {k: v for k, v in cfg.items() if k != "_config_path"}
    c["paths"] = {"raw_repos": str(tmp_path / "raw"), "functions": str(tmp_path / "functions"), "queries": str(tmp_path / "queries"),
                  "indexes": str(tmp_path / "indexes"), "results": str(tmp_path / "results"), "figures": str(tmp_path / "figures"),
                  "embeddings": str(tmp_path / "embeddings")}
    c["privacy"] = {**cfg["privacy"], "quant_bits": [8], "noise_sigma": [0.0, 0.5], "attack_leaked_pairs": [0, 200], "bootstrap": 100}
    inp = synth.inputs()
    fdir = tmp_path / "functions"
    fdir.mkdir(parents=True)
    for split, ids, codes in (("protected", inp.index_ids, inp.index_codes), ("public_train", inp.train_ids, inp.train_codes)):
        with open(fdir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for i, (rid, code) in enumerate(zip(ids, codes)):
                f.write(json.dumps({"id": rid, "split": split, "repo": "r", "path": f"{i}.py", "lang": "python", "code": code,
                                    "start_line": 1, "end_line": 4, "n_lines": 4, "n_tokens": 10, "sha": str(i)}) + "\n")
    edir = tmp_path / "embeddings"
    _write_npz(edir / "protected.npz", "ids", inp.index_ids, inp.index_emb)
    _write_npz(edir / "public_train.npz", "ids", inp.train_ids, inp.train_emb)
    for s, (qids, emb) in inp.queries.items():
        _write_npz(edir / "queries" / f"{s}.npz", "qids", qids, emb)
    return c


def test_run_privacy_files_and_script_idempotent(tmp_path: Path, synth: Synth, cfg: dict, monkeypatch):
    c = _file_cfg(tmp_path, synth, cfg)
    monkeypatch.setenv("SMCODE_PROJECTION_KEY", "test-key")
    res = run_privacy(c, limit_train=800, limit_eval=100)
    out = tmp_path / "results" / "privacy.json"
    assert out.exists() and res["skipped"] is False
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert [d["name"] for d in saved["defenses"]] == ["none", "proj", "proj+int8", "proj+noise0.5"]  # σ = 0.5 информативна → без расширения
    assert saved["a1"]["n_eval"] == 100 and saved["leaked_pairs"] == [0, 200] and saved["bootstrap"] == 100
    assert saved["a1"]["n_train"] < 800 and saved["a1"]["df_corpus_size"] == len(synth.train_bags)  # df по полному public_train
    assert saved["a1"]["df_corpus"] == "public_train (full)" and saved["a1"]["n_rare_vocab"] > 20
    assert saved["defenses"][1]["attacks"]["A2"]["200"]["q_rel_error"] < 1e-2
    assert saved["defenses"][0]["tpr"]["semantic"]["ci_cluster"] is not None
    assert saved["config"]["seed"] == 7
    again = run_privacy(c)
    assert again["skipped"] is True
    # CLI-обёртка: подмножество защит, --force, --no-full-df
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(c), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("script_08", ROOT / "scripts" / "08_privacy.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.main(["--config", str(cfg_path), "--list-defences"]) == 0
    assert mod.main(["--config", str(cfg_path), "--defences", "none,proj+int4", "--leaked-pairs", "0", "100", "--limit-train", "500",
                     "--limit-eval", "50", "--a1-backend", "ridge", "--no-full-df", "--force"]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert [d["name"] for d in saved["defenses"]] == ["none", "proj+int4"] and saved["leaked_pairs"] == [0, 100]
    assert saved["a1"]["n_eval"] == 50 and saved["a1"]["df_corpus_size"] == 500 and saved["a1"]["df_corpus"] == "public_train (attack subsample)"
    bad = {**c, "paths": {**c["paths"], "embeddings": str(tmp_path / "nope")}}
    (tmp_path / "cfg_bad.yaml").write_text(yaml.safe_dump(bad), encoding="utf-8")
    assert mod.main(["--config", str(tmp_path / "cfg_bad.yaml"), "--force"]) == 1


def test_run_privacy_without_train_embeddings(synth: Synth, cfg: dict):
    inp = synth.inputs()
    inp.train_emb = np.zeros((0, D), np.float32)
    inp.train_codes, inp.train_langs, inp.train_ids = [], [], []
    res = run_privacy_inputs(cfg, inp, defences=[make_defence(projection=True)], leaked_pairs=[0], key=b"k")
    assert res["a1"] is None and res["errors"] and res["defenses"][0]["attacks"]["A1"] is None
    assert res["defenses"][0]["tpr"]["semantic"]["value"] > 0.9


# ----------------------------------------------------------------------------- A3: CPU-части и компиляция GPU-кода


def test_bleu_and_gen_vocab():
    ref = "def f ( a ) : return a + 1".split()
    assert attack_gen.sentence_bleu(ref, ref) == pytest.approx(1.0)
    assert attack_gen.corpus_bleu([ref], [ref]) == pytest.approx(1.0)
    assert attack_gen.sentence_bleu("x y z".split(), ref) == 0.0
    partial = attack_gen.sentence_bleu("def f ( a ) : return a".split(), ref)
    assert 0.0 < partial < 1.0
    assert attack_gen.corpus_bleu([], []) == 0.0 and attack_gen.sentence_bleu([], ref) == 0.0
    with pytest.raises(ValueError):
        attack_gen.corpus_bleu([ref], [])
    toks = [attack_gen.lexical_tokens("def f(a):\n    # c\n    return a + 1\n", "python"), ["x", "=", "1"]]
    assert "#" not in " ".join(toks[0]) and toks[0][:3] == ["def", "f", "("]
    vocab = attack_gen.GenVocab.build(toks, max_size=40)
    assert len(vocab) == 4 + len(set(toks[0]) | set(toks[1])) and vocab.tokens[:4] == list(attack_gen.SPECIAL)
    assert len(attack_gen.GenVocab.build(toks, max_size=8)) == 8
    ids = vocab.encode(["def", "f", "zzz"], max_len=2)
    assert ids[0] == vocab.bos and ids[-1] == vocab.eos and len(ids) == 4
    assert vocab.decode(ids + [vocab.pad]) == ["def", "f"]
    m = attack_gen.identifier_metrics([["a", "b"], ["x"]], [{"a", "b", "c"}, {"x"}, set()])
    assert m["id_acc"] == pytest.approx(3 / 4) and m["id_exact"] == 0.5 and m["n_with_ids"] == 2
    assert attack_gen.identifier_set("def g(q):\n    return q\n", "python") == {"g", "q"}


def test_gpu_parts_compile_and_fail_gracefully_without_torch():
    for f in ("smcode/privacy/attack_gen.py", "smcode/privacy/attack_bow.py", "smcode/privacy/run.py", "scripts/08_privacy.py"):
        py_compile.compile(str(ROOT / f), doraise=True)
    if attack_bow.torch_available():
        pytest.skip("torch установлен — проверка отказа без torch не применима")
    vocab = attack_bow.Vocab(tokens=["a"], df=np.array([5]))
    with pytest.raises(ImportError):
        attack_bow.BowInverter(vocab, backend="torch")
    with pytest.raises(ImportError):
        attack_gen.run_attack_gen(np.zeros((2, 4), np.float32), ["x", "y"], ["python"] * 2, np.zeros((1, 4), np.float32), ["z"], ["python"], {})
    assert "torch" not in sys.modules
