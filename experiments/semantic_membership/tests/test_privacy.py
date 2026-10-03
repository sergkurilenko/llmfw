"""Тесты модуля приватности (smcode.privacy): проекция, квантование/шум, атаки A1/A2, прогон шага 08.

Синтетика: словарь из 40 частых и 40 редких идентификаторов; каждому идентификатору сопоставлен случайный
вектор, эмбеддинг функции = нормированная сумма векторов её идентификаторов + шум, код функции — короткая
функция Python с этими идентификаторами. Без сети и без torch.
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

from smcode.config import load_config
from smcode.privacy import attack_align, attack_bow, attack_gen, projection, quantize
from smcode.privacy.run import (DefenceSpec, PrivacyInputs, apply_defence, default_defences, defence_memory, make_defence,
                                parse_defence, run_privacy, run_privacy_inputs, utility)

ROOT = Path(__file__).resolve().parents[1]
D = 64
N_COMMON, N_RARE = 40, 40
COMMON = [f"cm{k}" for k in range(N_COMMON)]
RARE = [f"rr{k}" for k in range(N_RARE)]


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
        for r in range(N_RARE):  # редкий идентификатор r встречается ровно в двух обучающих функциях (df = 2)
            self.train_bags[2 * r].append(RARE[r])
            self.train_bags[2 * r + 1].append(RARE[r])
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
    assert set(u0["tpr"]["semantic"]) == {"value", "ci", "n", "k"}


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
    assert vocab.n_rare == N_RARE and set(RARE) <= set(vocab.tokens)  # все редкие (df = 2) попали в квоту
    assert not any(t.startswith("tfn") for t in vocab.tokens)  # имена функций (df = 1) исключены
    Y = attack_bow.targets(all_bags, vocab)
    assert Y.shape == (len(all_bags), len(vocab)) and Y.nnz == sum(len(set(b)) for b in synth.train_bags)
    small = attack_bow.build_vocab(all_bags, max_size=50, rare_fraction=0.2)
    assert len(small) == 50 and small.n_rare == 10
    rt = attack_bow.Vocab.from_dict(vocab.to_dict())
    assert rt.tokens == vocab.tokens and np.array_equal(rt.df, vocab.df)
    assert attack_bow.identifier_tokens("def f(:", "nolang") == []


def test_a1_on_plain_embeddings_beats_chance(synth: Synth, a1):
    model, vocab, Y_eval, prior = a1
    m = model.evaluate(synth.prot_emb, Y_eval)
    assert m["backend"] == "ridge" and m["n_eval"] == len(synth.prot_bags)
    assert m["f1"] > prior["f1"] + 0.3, (m["f1"], prior["f1"])
    assert m["rare_id_recall"] is not None and m["rare_id_recall"] > 0.5 and m["rare_id_rule"] == "per_token"
    assert m["rare_id_recall_global"] == 0.0  # один глобальный порог редкие токены не предсказывает
    assert m["decision"] in ("global", "per_token") and set(model.val_f1_rules) == {"global", "per_token"}
    assert m["f1"] == pytest.approx(m[f"f1_{m['decision']}"]) and max(m["f1_global"], m["f1_per_token"]) > 0.8
    assert m["n_rare_true"] == len(synth.prot_bags)  # по одному редкому идентификатору на функцию
    assert prior["rare_id_recall"] == 0.0
    assert 0.0 <= m["mean_jaccard"] <= 1.0 and 0.0 <= m["exact_set_rate"] <= 1.0
    # прямое применение к спроецированным векторам без ключа — уровень случайности
    Q = projection.keyed_orthogonal(D, b"k1")
    blind = model.evaluate(projection.apply(synth.prot_emb, Q), Y_eval)
    assert blind["f1"] < prior["f1"] + 0.1


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


# ----------------------------------------------------------------------------- прогон §9 и схема privacy.json


def test_defence_specs(cfg: dict):
    names = [d.name for d in default_defences(cfg)]
    assert names == ["none", "proj", "proj+int8", "proj+int4", "proj+bin", "proj+noise0.05", "proj+noise0.1", "proj+noise0.2"]
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
    assert res["a1"]["backend"] == "ridge" and res["a1"]["n_rare_vocab"] == N_RARE
    by = {d["name"]: d for d in res["defenses"]}
    assert list(by) == ["none", "proj", "proj+bin", "proj+noise1"]
    for d in by.values():
        assert set(d["tpr"]) == {"semantic", "hybrid"} and d["tpr"]["hybrid"] is None
        assert {"value", "ci", "n", "k"} <= set(d["tpr"]["semantic"])
        assert {"f1", "precision", "recall", "rare_id_recall"} <= set(d["attacks"]["A1"])
        assert set(d["attacks"]["A2"]) == {"0", "1000"} and "f1" in d["attacks"]["A2"]["1000"]
        assert d["attacks"]["A3"] is None and "memory_ratio" in d
    prior_f1 = res["a1"]["prior_baseline"]["f1"]
    assert by["none"]["attacks"]["A1"]["f1"] > prior_f1 + 0.3
    assert by["proj"]["attacks"]["A1"]["f1"] < prior_f1 + 0.1  # без ключа — случайность
    assert by["proj"]["attacks"]["A2"]["1000"]["f1"] > prior_f1 + 0.3  # с утечкой — восстановлено
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


def _write_npz(path: Path, key: str, ids: list[str], emb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: np.asarray(ids, dtype=str), "emb": emb.astype(np.float16)})


def _file_cfg(tmp_path: Path, synth: Synth, cfg: dict) -> dict:
    c = {k: v for k, v in cfg.items() if k != "_config_path"}
    c["paths"] = {"raw_repos": str(tmp_path / "raw"), "functions": str(tmp_path / "functions"), "queries": str(tmp_path / "queries"),
                  "indexes": str(tmp_path / "indexes"), "results": str(tmp_path / "results"), "figures": str(tmp_path / "figures"),
                  "embeddings": str(tmp_path / "embeddings")}
    c["privacy"] = {**cfg["privacy"], "quant_bits": [8], "noise_sigma": [0.0, 0.5], "attack_leaked_pairs": [0, 200]}
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
    assert [d["name"] for d in saved["defenses"]] == ["none", "proj", "proj+int8", "proj+noise0.5"]
    assert saved["a1"]["n_eval"] == 100 and saved["leaked_pairs"] == [0, 200]
    assert saved["defenses"][1]["attacks"]["A2"]["200"]["q_rel_error"] < 1e-2
    assert saved["config"]["seed"] == 7
    again = run_privacy(c)
    assert again["skipped"] is True
    # CLI-обёртка: подмножество защит, --force
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(c), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("script_08", ROOT / "scripts" / "08_privacy.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.main(["--config", str(cfg_path), "--list-defences"]) == 0
    assert mod.main(["--config", str(cfg_path), "--defences", "none,proj+int4", "--leaked-pairs", "0", "100", "--limit-train", "500",
                     "--limit-eval", "50", "--a1-backend", "ridge", "--force"]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert [d["name"] for d in saved["defenses"]] == ["none", "proj+int4"] and saved["leaked_pairs"] == [0, 100]
    assert saved["a1"]["n_eval"] == 50
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
