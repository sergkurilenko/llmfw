"""Тесты модуля fingerprint: реестр, M0 exact, M1 winnowing, M2 minhash, скрипт 04 (синтетика, CPU)."""

from __future__ import annotations

import importlib.util
import json
import logging
import random
import re
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from smcode.config import load_config
from smcode.fingerprint.exact import ExactIndex, normalized_lines, window_hashes
from smcode.fingerprint.index import (
    REGISTRY,
    BaseIndex,
    InvertedIndex,
    build_index,
    check_key,
    deep_sizeof,
    key_id,
    load_index,
    make_index,
    score_hashes,
    sorted_contains,
    tokens_or_reason,
    write_index_stats,
)
from smcode.fingerprint.minhash import (
    SCHEME,
    MinHashIndex,
    band_keys,
    estimate_jaccard,
    jaccard,
    minhash_signature,
    optimal_bands,
    shingle_set,
)
from smcode.fingerprint.winnowing_index import WinnowingIndex
from smcode.types import FunctionRecord, MembershipIndex, QueryResult, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
PY = "python"
KEY_ENV = "SMCODE_INDEX_KEY"

# ----------------------------------------------------------------------------- синтетика

_OPS = ["+", "-", "*", "//", "%", "<<", ">>", "&", "|", "^"]
_FUNCS = ["len", "abs", "min", "max", "sum", "int", "str", "round", "sorted", "list"]


def _expr(rng: random.Random, ids: list[str], depth: int = 0) -> str:
    r = rng.random()
    if depth > 2 or r < 0.3:
        return rng.choice(ids) if rng.random() < 0.7 else str(rng.randint(0, 999))
    if r < 0.55:
        return f"({_expr(rng, ids, depth + 1)} {rng.choice(_OPS)} {_expr(rng, ids, depth + 1)})"
    if r < 0.75:
        return f"{rng.choice(_FUNCS)}({_expr(rng, ids, depth + 1)})"
    return f"{_expr(rng, ids, depth + 1)} {rng.choice(_OPS)} {_expr(rng, ids, depth + 1)}"


def make_fn(rng: random.Random, i: int, n_stmts: int = 12) -> str:
    """Случайная Python-функция с разнообразной структурой выражений (≈ 200–300 токенов)."""
    ids = [f"a{i}_{j}" for j in range(5)]
    lines = [f"def fn_{i}({', '.join(ids[:3])}):"]
    for _ in range(n_stmts):
        t = rng.random()
        v = rng.choice(ids)
        if t < 0.5:
            lines.append(f"    {v} = {_expr(rng, ids)}")
        elif t < 0.65:
            lines.append(f"    if {_expr(rng, ids)} > {_expr(rng, ids)}:")
            lines.append(f"        {v} = {_expr(rng, ids)}")
        elif t < 0.8:
            lines.append(f"    for k{i} in range({_expr(rng, ids)}):")
            lines.append(f"        {v} += {_expr(rng, ids)}")
        else:
            lines.append(f"    {v} = [{_expr(rng, ids)} for q{i} in {rng.choice(ids)}]")
    lines.append(f"    return {_expr(rng, ids)}")
    return "\n".join(lines) + "\n"


def rename(code: str) -> str:
    """Согласованное переименование идентификаторов простым regex."""
    return re.sub(r"\b([a-z]+\d+_\d+|k\d+|q\d+|fn_\d+)\b", lambda m: "zz_" + m.group(1), code)


def partial(code: str, lo: int = 2, hi: int = 8) -> str:
    """Фрагмент из подряд идущих строк тела (≥ 30 токенов, ≥ 5 строк)."""
    return "\n".join(code.splitlines()[lo:hi]) + "\n"


def make_records(n: int = 40, seed: int = 0, split: str = "protected", prefix: str = "r") -> list[FunctionRecord]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        code = make_fn(rng, i + seed * 1000)
        out.append(
            FunctionRecord(
                id=f"{split}/{prefix}/f{i}.py:1-{code.count(chr(10))}", split=split, repo=prefix, path=f"src/f{i // 5}.py",
                lang=PY, code=code, start_line=1, end_line=code.count("\n"), n_lines=code.count("\n"), n_tokens=0, sha="",
            )
        )
    return out


def _median_query_ms(idx, code: str, n: int = 15) -> float:
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        idx.query(code, PY)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


@pytest.fixture()
def cfg(tmp_path: Path) -> dict:
    return load_config(
        overrides={
            "paths": {
                "raw_repos": str(tmp_path / "raw"), "functions": str(tmp_path / "functions"), "queries": str(tmp_path / "queries"),
                "indexes": str(tmp_path / "indexes"), "results": str(tmp_path / "results"), "figures": str(tmp_path / "figures"),
            },
            "fingerprint": {"k": 8, "w": 6, "common_df": 50, "exact_window": 5, "minhash_perm": 128, "minhash_shingle": 5, "workers": 1},
        }
    )


@pytest.fixture()
def records() -> list[FunctionRecord]:
    return make_records(40)


@pytest.fixture()
def nonmember() -> str:
    return make_fn(random.Random(4242), 7777)


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    """Тесты по умолчанию идут без ключа HMAC (ключ задаётся явно в тестах ключа)."""
    monkeypatch.delenv(KEY_ENV, raising=False)


# ----------------------------------------------------------------------------- index.py


def test_registry_and_make_index(cfg):
    assert set(REGISTRY) == {"exact", "winnowing", "minhash", "semantic", "hybrid"}
    for name, cls in [("exact", ExactIndex), ("winnowing", WinnowingIndex), ("minhash", MinHashIndex)]:
        idx = make_index(name, cfg)
        assert isinstance(idx, cls) and idx.name == name
        assert isinstance(idx, MembershipIndex)
    with pytest.raises(KeyError):
        make_index("nope", cfg)


def test_inverted_index_roundtrip_and_without():
    hashes = np.array([5, 3, 5, 9, 3, 5, 2**64 - 1], dtype=np.uint64)
    ids = np.array([1, 0, 1, 2, 2, 0, 3], dtype=np.int32)
    inv = InvertedIndex.from_pairs(hashes, ids)
    assert inv.keys.tolist() == [3, 5, 9, 2**64 - 1]
    assert inv.postings_for(0).tolist() == [0, 2]
    assert inv.postings_for(1).tolist() == [0, 1]  # дубликат пары (5,1) схлопнут
    assert inv.df().tolist() == [2, 2, 1, 1]
    assert inv.find(np.array([5, 4, 9], dtype=np.uint64)).tolist() == [1, -1, 2]
    found, post = inv.gather(np.array([3, 4, 9], dtype=np.uint64))
    assert found.tolist() == [True, False, True] and post.tolist() == [0, 2, 2]
    sub = inv.without(np.array([5, 5, 100], dtype=np.uint64))  # неотсортированный drop с дубликатами и лишним ключом
    assert sub.keys.tolist() == [3, 9, 2**64 - 1] and sub.postings.tolist() == [0, 2, 2, 3]
    back = InvertedIndex.from_arrays(sub.arrays())
    assert np.array_equal(back.keys, sub.keys) and np.array_equal(back.postings, sub.postings)
    score, best, d = score_hashes(inv, np.array([3, 5, 7], dtype=np.uint64))
    assert score == pytest.approx(2 / 3) and best == 0 and d["best_shared"] == 2 and d["n_candidates"] == 3
    assert score_hashes(InvertedIndex(), np.array([1], dtype=np.uint64))[0] == 0.0
    keys = np.array([2, 5, 9], dtype=np.uint64)
    assert sorted_contains(keys, np.array([1, 2, 5, 9, 10], dtype=np.uint64)).tolist() == [False, True, True, True, False]
    assert sorted_contains(np.zeros(0, np.uint64), keys).tolist() == [False] * 3


def test_base_query_batch_latency():
    class Const(BaseIndex):
        name = "const"

        def query(self, code, lang):
            return QueryResult(0.5, "x")

    res = Const({}).query_batch([("a", PY), ("b", PY)])
    assert [r.score for r in res] == [0.5, 0.5]
    assert all(r.latency_ms is not None and r.latency_ms >= 0.0 for r in res)


def test_deep_sizeof():
    assert deep_sizeof(np.zeros(100, dtype=np.uint64)) >= 800
    assert deep_sizeof(["abc"] * 3) > deep_sizeof([])


def test_tokens_or_reason():
    toks, reason = tokens_or_reason("def f(x):\n    return x + 1\n", PY)
    assert toks and reason is None
    toks, reason = tokens_or_reason("fn main() {}", "rust")
    assert toks == [] and reason == "unsupported_lang"


def test_key_id_and_check_key():
    assert key_id(None) is None and key_id(b"k") == key_id(b"k") and key_id(b"k") != key_id(b"j") and len(key_id(b"k")) == 16
    check_key({"keyed": True, "key_id": key_id(b"k")}, b"k")
    check_key({"keyed": False, "key_id": None}, None)
    for stored, cur in [({"key_id": None, "keyed": False}, b"k"), ({"key_id": key_id(b"k"), "keyed": True}, None),
                        ({"key_id": key_id(b"k"), "keyed": True}, b"j"), ({"keyed": True}, None), ({"keyed": False}, b"k")]:
        with pytest.raises(ValueError, match="HMAC key mismatch"):
            check_key(stored, cur)
    check_key({"keyed": True}, b"k")  # старый формат без key_id: только предупреждение


# ----------------------------------------------------------------------------- M0 exact


def test_exact_lines_and_windows():
    code = "def f(x):\n    # comment\n    y = x + 1  # trailing\n\n    return y\n"
    assert normalized_lines(code, PY) == ["def f ( x ) :", "y = x + 1", "return y"]
    assert len(window_hashes(code, PY, window=5)) == 1  # короче окна — одно окно
    assert len(window_hashes("def f():\n" + "    x = 1\n" * 9, PY, window=5)) == 6
    assert window_hashes("", PY) == []
    assert window_hashes(code, PY, key=b"k1") != window_hashes(code, PY, key=b"k2")


def test_exact_scores(cfg, records, nonmember):
    idx = ExactIndex(cfg)
    idx.build(records, cfg)
    q = records[3].code
    assert idx.query(q, PY).score == 1.0 and idx.query(q, PY).best_id == records[3].id
    # переформатирование отступов/пробелов и комментарии не мешают
    reformatted = q.replace("    ", "\t").replace(" = ", "=") + "# tail\n"
    assert idx.query(reformatted, PY).score == 1.0
    r = idx.query(partial(q), PY)
    assert r.score == 1.0 and r.best_id == records[3].id
    assert idx.query(rename(q), PY).score == 0.0  # лексический метод: переименование рушит окна
    assert idx.query(nonmember, PY).score == 0.0
    r = idx.query("", PY)
    assert r.score == 0.0 and r.details["reason"] == "too_short"
    assert idx.memory_bytes() > 0 and idx.meta["n_keys"] > 0
    # правило коротких входов задокументировано в meta: фрагмент из 3 строк хэшируется одним окном и не находится
    assert idx.meta["hash"] == "blake2b-64" and "short_input_rule" in idx.meta
    r = idx.query(partial(q, 2, 5), PY)
    assert r.score == 0.0 and r.details["n_windows"] == 1 and r.details["n_lines"] == 3


# ----------------------------------------------------------------------------- M1 winnowing


def test_winnowing_member_queries(cfg, records, nonmember):
    idx = WinnowingIndex(cfg)
    idx.build(records, cfg)
    q = records[5].code
    for variant in (q, rename(q), partial(q)):
        r = idx.query(variant, PY)
        assert r.score >= 0.95, (variant, r)
        assert r.best_id == records[5].id
        assert r.details["longest_common_run"] >= 3 and r.details["n_fps"] > 0
    r = idx.query(nonmember, PY)
    assert r.score < 0.2, r
    # слишком короткий запрос
    r = idx.query("x = 1\n", PY)
    assert r.score == 0.0 and r.best_id is None and r.details["reason"] == "too_short"
    # пакетный запрос заполняет латентность
    batch = idx.query_batch([(q, PY), (nonmember, PY)])
    assert all(b.latency_ms is not None and b.latency_ms > 0 for b in batch)
    assert batch[0].score >= 0.95 > batch[1].score
    # вспомогательные методы для гибрида
    qh = idx.query_hashes(q, PY)
    assert qh.size > 0 and idx.overlap(qh, idx.idx_of(records[5].id)) == 1.0
    assert idx.overlap(qh, idx.idx_of(records[6].id)) < 0.2


def test_winnowing_common_filter(cfg, records):
    """Отпечатки, частые в public_train, исключаются из индекса и из запроса."""
    shared = make_fn(random.Random(77), 999)  # «общий» код, который есть и в protected, и в 60 public-функциях
    prot = records[:10] + [FunctionRecord(**{**records[0].to_dict(), "id": "protected/r/shared.py:1-5", "path": "src/shared.py", "code": shared})]
    public = [FunctionRecord(**{**r.to_dict(), "id": f"public_train/p/{i}", "split": "public_train", "repo": "p", "code": shared}) for i, r in enumerate(records[:60] * 2)]
    plain = WinnowingIndex(cfg)
    plain.build(prot, cfg)
    assert plain.query(shared, PY).score == 1.0
    filt = WinnowingIndex(cfg)
    n_common = filt.fit_common_filter(public)
    assert n_common > 0
    filt.build(prot, cfg)
    r = filt.query(shared, PY)
    assert r.score == 0.0 and r.details["reason"] == "all_common"
    assert len(filt.inv) < len(plain.inv)
    # членство остальных не страдает
    assert filt.query(records[2].code, PY).score >= 0.95
    # фильтр, подогнанный после build, тоже применяется, и meta обновляется (finding: stale n_common_public)
    late = WinnowingIndex(cfg)
    late.build(prot, cfg)
    assert late.meta["n_common_public"] == 0 and late.meta["n_public_fitted"] == 0
    late.fit_common_filter(public)
    assert late.query(shared, PY).score == 0.0 and len(late.inv) == len(filt.inv)
    assert late.meta["n_common_public"] == n_common and late.meta["n_public_fitted"] == len(public)
    assert late.meta["n_common"] == late.common.size


def test_winnowing_common_filter_latency_independent_of_size(cfg, records):
    """Проверка общности — searchsorted: латентность запроса не растёт с |common| (finding: np.isin)."""
    idx = WinnowingIndex(cfg)
    idx.build(records, cfg)
    q = records[5].code
    base_score = idx.query(q, PY).score
    base_ms = _median_query_ms(idx, q)
    rng = np.random.default_rng(1)
    idx.public_common = np.unique(rng.integers(0, 2**63, size=200_000, dtype=np.int64).astype(np.uint64))
    idx._apply_common()
    assert idx.common.size >= 190_000 and idx.meta["n_common"] == idx.common.size
    big_ms = _median_query_ms(idx, q)
    r = idx.query(q, PY)
    assert r.score == pytest.approx(base_score) and r.best_id == records[5].id
    assert big_ms <= max(2.5 * base_ms, base_ms + 1.0), (base_ms, big_ms)
    hs = idx.query_hashes(q, PY)
    assert hs.size > 0 and not idx.common_mask(hs).any()
    assert idx.filter_common(np.concatenate([hs[:2], idx.common[:3]])).tolist() == hs[:2].tolist()


def test_winnowing_protected_file_filter(cfg):
    """Отпечаток в > 5 % файлов protected (и ≥ common_min_files) — общий."""
    rng = random.Random(3)
    boiler = make_fn(rng, 500)
    recs = []
    for i in range(60):  # 60 файлов; шаблон boiler присутствует в 6 из них (10 %)
        code = boiler if i % 10 == 0 else make_fn(rng, 600 + i)
        recs.append(FunctionRecord(id=f"protected/r/{i}", split="protected", repo="r", path=f"src/{i}.py", lang=PY, code=code,
                                   start_line=1, end_line=10, n_lines=10, n_tokens=0, sha=""))
    idx = WinnowingIndex(cfg)
    idx.build(recs, cfg)
    assert idx.protected_common.size > 0 and idx.meta["n_common_protected"] > 0
    assert idx.query(boiler, PY).score == 0.0
    assert idx.query(recs[1].code, PY).score >= 0.95


def test_winnowing_restore_params(cfg, records, tmp_path):
    """Параметры фильтра (common_file_frac/common_min_files) загружаются из индекса, а не из текущего cfg."""
    idx = WinnowingIndex(cfg)
    idx.build(records, cfg)
    idx.save(tmp_path / "w")
    other = load_config(overrides={**{k: v for k, v in cfg.items() if not k.startswith("_")},
                                   "fingerprint": {**cfg["fingerprint"], "common_file_frac": 0.5, "common_min_files": 99, "common_df": 7}})
    loaded = WinnowingIndex(other)
    assert loaded.common_min_files == 99
    loaded.load(tmp_path / "w")
    assert (loaded.common_file_frac, loaded.common_min_files, loaded.common_df) == (idx.common_file_frac, idx.common_min_files, idx.common_df)
    assert loaded.meta["common_file_frac"] == idx.common_file_frac


def test_winnowing_hmac_key(cfg, records, monkeypatch):
    q = records[7].code
    monkeypatch.setenv(KEY_ENV, "key-one")
    a = WinnowingIndex(cfg)
    a.build(records, cfg)
    monkeypatch.setenv(KEY_ENV, "key-two")
    b = WinnowingIndex(cfg)
    b.build(records, cfg)
    assert a.key == b"key-one" and b.key == b"key-two"
    assert not np.intersect1d(a.inv.keys, b.inv.keys).size  # хэши различны
    for variant in (q, rename(q), partial(q)):
        assert a.query(variant, PY).score == pytest.approx(b.query(variant, PY).score)
    # без ключа индекс с ключом не отвечает
    monkeypatch.delenv(KEY_ENV)
    c = WinnowingIndex(cfg)
    c.build(records, cfg)
    assert c.key is None and c.query(q, PY).score == 1.0
    assert not np.intersect1d(a.inv.keys, c.inv.keys).size
    assert a.meta["keyed"] and a.meta["key_id"] == key_id(b"key-one") and c.meta["key_id"] is None


@pytest.mark.parametrize("name", ["exact", "winnowing", "minhash"])
def test_key_mismatch_on_load_is_an_error(cfg, records, tmp_path, monkeypatch, name):
    """Индекс, построенный без ключа (или с другим ключом), нельзя загрузить с другим ключом (finding: silent TPR=0)."""
    q = records[1].code
    unkeyed = make_index(name, cfg)
    unkeyed.build(records, cfg)
    unkeyed.save(tmp_path / "unkeyed")
    meta = json.loads((tmp_path / "unkeyed" / "meta.json").read_text())
    assert meta["keyed"] is False and meta["key_id"] is None
    monkeypatch.setenv(KEY_ENV, "secret-a")
    with pytest.raises(ValueError, match="HMAC key mismatch"):
        load_index(name, cfg, tmp_path / "unkeyed")
    keyed = make_index(name, cfg)
    keyed.build(records, cfg)
    keyed.save(tmp_path / "keyed")
    assert json.loads((tmp_path / "keyed" / "meta.json").read_text())["key_id"] == key_id(b"secret-a")
    same = load_index(name, cfg, tmp_path / "keyed")  # тот же ключ — загрузка и ответы совпадают
    assert same.query(q, PY).score == pytest.approx(keyed.query(q, PY).score) == 1.0
    monkeypatch.setenv(KEY_ENV, "secret-b")
    with pytest.raises(ValueError, match="HMAC key mismatch"):
        load_index(name, cfg, tmp_path / "keyed")
    monkeypatch.delenv(KEY_ENV)
    with pytest.raises(ValueError, match="HMAC key mismatch"):
        load_index(name, cfg, tmp_path / "keyed")
    assert load_index(name, cfg, tmp_path / "unkeyed").query(q, PY).score == 1.0


@pytest.mark.parametrize("name", ["exact", "winnowing", "minhash"])
def test_unsupported_language_reason(cfg, records, name):
    """Неподдерживаемый язык — details.reason='unsupported_lang', а не 'too_short'; в meta считается n_unsupported."""
    idx = make_index(name, cfg)
    recs = records[:5] + [FunctionRecord(**{**records[0].to_dict(), "id": "protected/r/x.rs", "lang": "rust", "code": "fn main() { let x = 1; }"})]
    idx.build(recs, cfg)
    assert idx.meta["n_unsupported"] == 1 and idx.n_records == 6
    r = idx.query(records[0].code, "rust")
    assert r.score == 0.0 and r.best_id is None and r.details["reason"] == "unsupported_lang"
    short = "" if name == "exact" else "x = 1\n"  # у M0 одна строка — одно окно (правило коротких входов), «коротко» = нет токенов
    assert idx.query(short, PY).details["reason"] == "too_short"


# ----------------------------------------------------------------------------- M2 minhash


def test_shingles_and_jaccard():
    toks = list("abcdefgh")
    sh = shingle_set(toks, 3)
    assert sh.size == 6 and np.all(np.diff(sh.astype(np.float64)) > 0)
    assert shingle_set(toks[:2], 3).size == 0
    assert jaccard(sh, sh) == (1.0, 6)
    j, inter = jaccard(sh, shingle_set(list("abcdxyz"), 3))
    assert inter == 2 and j == pytest.approx(2 / (6 + 5 - 2))
    assert jaccard(sh, np.zeros(0, dtype=np.uint64)) == (0.0, 0)
    assert not np.intersect1d(sh, shingle_set(toks, 3, key=b"k")).size


def test_minhash_signatures_and_bands():
    """Подписи numpy: детерминизм по seed, оценка Жаккара ≈ точной, (b, r) как у datasketch."""
    rng = np.random.default_rng(0)
    a = np.unique(rng.integers(0, 2**63, size=400, dtype=np.int64).astype(np.uint64))
    b = np.unique(np.concatenate([a[:200], rng.integers(0, 2**63, size=200, dtype=np.int64).astype(np.uint64)]))
    sa, sb = minhash_signature(a, 128, 5), minhash_signature(b, 128, 5)
    assert sa.dtype == np.uint32 and sa.shape == (128,) and np.array_equal(sa, minhash_signature(a, 128, 5))
    assert not np.array_equal(sa, minhash_signature(a, 128, 6))
    j, _ = jaccard(a, b)
    assert abs(float(estimate_jaccard(sa, sb)) - j) < 0.15 and float(estimate_jaccard(sa, sa)) == 1.0
    assert estimate_jaccard(sa, np.vstack([sa, sb])).shape == (2,)
    empty = minhash_signature(np.zeros(0, np.uint64), 128, 5)
    assert np.all(empty == 2**31 - 1)
    assert optimal_bands(0.3, 128) == (37, 3) and optimal_bands(0.5, 128) == (25, 5)
    try:
        from datasketch.lsh import _optimal_param
    except ImportError:  # pragma: no cover
        pass
    else:
        assert optimal_bands(0.3, 128) == tuple(_optimal_param(0.3, 128, 0.5, 0.5))
    keys = band_keys(np.vstack([sa, sb]), 37, 3)
    assert keys.shape == (2, 37) and keys.dtype == np.uint64 and np.array_equal(keys[0], band_keys(sa, 37, 3)[0])
    assert np.unique(keys[0]).size == 37  # соль номера полосы: ключи разных полос различны


def test_minhash_scores(cfg, records, nonmember):
    idx = MinHashIndex(cfg)
    idx.build(records, cfg)
    assert (idx.b, idx.r) == (37, 3) and idx.hashvalues.shape == (len(records), 128) and idx.hashvalues.dtype == np.uint32
    assert idx.bands.sorted_keys.shape == (len(records) * 37,) and idx.meta["scheme"] == SCHEME
    q = records[9].code
    for variant in (q, rename(q)):
        r = idx.query(variant, PY)
        assert r.score == 1.0 and r.best_id == records[9].id and r.details["containment"] == 1.0
    # половина функции: Жаккар ≈ доля, кандидат находится
    half = partial(q, 0, 9)
    r = idx.query(half, PY)
    assert r.best_id == records[9].id and 0.3 < r.score < 1.0 and r.details["containment"] == 1.0
    assert r.details["n_candidates"] >= 1 and r.details["n_scored"] <= r.details["n_candidates"]
    r = idx.query(nonmember, PY)
    assert r.score < 0.3, r
    r = idx.query("x = 1\n", PY)
    assert r.score == 0.0 and r.details["reason"] == "too_short"
    assert idx.memory_bytes() > idx.hashvalues.nbytes
    # ограничение числа кандидатов: точный Жаккар — для лучших по оценке подписи
    idx.max_candidates = 1
    r = idx.query(half, PY)
    assert r.best_id == records[9].id and r.details["n_scored"] == 1


def test_minhash_memory_and_load_without_rebuild(cfg, tmp_path, monkeypatch):
    """Память M2 — numpy-массивы (< 4 КБ на запись при 300 записях), полосы грузятся из npz без пересборки."""
    recs = make_records(300, seed=2)
    idx = MinHashIndex(cfg)
    idx.build(recs, cfg)
    per_record = idx.memory_bytes() / len(recs)
    assert per_record < 4096, per_record
    assert not hasattr(idx, "lsh")
    idx.save(tmp_path / "m")
    with np.load(tmp_path / "m" / "arrays.npz") as z:
        assert {"band_keys", "band_order", "hashvalues", "sh_values", "sh_offsets"} <= set(z.files)
    monkeypatch.setattr(MinHashIndex, "_build_bands", lambda self: (_ for _ in ()).throw(AssertionError("rebuilt on load")))
    loaded = MinHashIndex(cfg)
    loaded.load(tmp_path / "m")
    assert (loaded.b, loaded.r) == (idx.b, idx.r) and np.array_equal(loaded.bands.sorted_keys, idx.bands.sorted_keys)
    q = recs[17].code
    assert loaded.query(q, PY).best_id == recs[17].id and loaded.query(partial(q, 0, 9), PY).best_id == recs[17].id
    assert abs(loaded.memory_bytes() - idx.memory_bytes()) < 0.05 * idx.memory_bytes()
    # индекс другой схемы подписей не загружается
    import pickle

    st = pickle.loads((tmp_path / "m" / "state.pkl").read_bytes())
    st["params"]["scheme"] = "datasketch-legacy"
    (tmp_path / "m" / "state.pkl").write_bytes(pickle.dumps(st))
    with pytest.raises(ValueError, match="scheme"):
        MinHashIndex(cfg).load(tmp_path / "m")


# ----------------------------------------------------------------------------- save / load


@pytest.mark.parametrize("name", ["exact", "winnowing", "minhash"])
def test_save_load_roundtrip(cfg, records, nonmember, tmp_path, name):
    idx = make_index(name, cfg)
    idx.build(records, cfg)
    out = tmp_path / "idx" / name
    idx.save(out)
    assert (out / "meta.json").exists() and (out / "state.pkl").exists() and (out / "arrays.npz").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["name"] == name and meta["n_records"] == len(records) and meta["keyed"] is False
    loaded = make_index(name, cfg)
    loaded.load(out)
    assert loaded.ids == idx.ids
    queries = [records[1].code, rename(records[1].code), partial(records[1].code), nonmember]
    for qq in queries:
        a, b = idx.query(qq, PY), loaded.query(qq, PY)
        assert a.score == pytest.approx(b.score) and a.best_id == b.best_id
    assert abs(loaded.memory_bytes() - idx.memory_bytes()) / max(idx.memory_bytes(), 1) < 0.5


def test_save_after_load_writes_fresh_meta(cfg, records, tmp_path, monkeypatch):
    """meta.json после load → изменение → save содержит свежие n_records/memory_bytes/saved_at (finding: stale meta)."""
    import smcode.fingerprint.index as index_mod

    stamps = iter(["2026-01-01T00:00:00", "2026-01-01T00:00:01"])
    monkeypatch.setattr(index_mod.time, "strftime", lambda fmt: next(stamps))
    idx = WinnowingIndex(cfg)
    idx.build(records, cfg)
    idx.save(tmp_path / "a")
    meta_a = json.loads((tmp_path / "a" / "meta.json").read_text())
    loaded = WinnowingIndex(cfg)
    loaded.load(tmp_path / "a")
    assert "saved_at" not in loaded.meta and "memory_bytes" not in loaded.meta and loaded.meta["k"] == cfg["fingerprint"]["k"]
    loaded.ids = loaded.ids[:10]
    loaded.rec_offsets = loaded.rec_offsets[:11]
    loaded.rec_hashes = loaded.rec_hashes[: loaded.rec_offsets[-1]]
    loaded._apply_common()
    loaded.save(tmp_path / "b")
    meta_b = json.loads((tmp_path / "b" / "meta.json").read_text())
    assert meta_b["n_records"] == 10 and meta_a["n_records"] == len(records)
    assert meta_b["memory_bytes"] < meta_a["memory_bytes"] and meta_b["saved_at"] != meta_a["saved_at"]
    assert meta_b["class"] == meta_a["class"] and meta_b["k"] == meta_a["k"]


def test_build_index_and_load_index(cfg, records):
    idx, stats = build_index("exact", cfg, [r.to_dict() for r in records], tmp_dir := cfg["paths"]["indexes"] + "/exact", extra_meta={"with_windows": False})
    assert stats["n_records"] == len(records) and stats["memory_bytes"] > 0 and stats["disk_bytes"] > 0
    assert Path(tmp_dir).exists()
    meta = json.loads((Path(tmp_dir) / "meta.json").read_text())
    assert meta["with_windows"] is False and meta["build_seconds"] >= meta["index_build_seconds"] and meta["fit_seconds"] == 0.0
    loaded = load_index("exact", cfg)
    assert loaded.query(records[0].code, PY).score == 1.0


def test_write_index_stats_merges(cfg):
    p = write_index_stats(cfg, {"exact": {"n_records": 5, "meta": {"a": 1}, "with_windows": True, "skipped": False, "build_seconds": 1.5}})
    write_index_stats(cfg, {"exact": {"n_records": 5, "skipped": True, "build_seconds": None, "meta": {"b": 2}}, "minhash": {"n_records": 3, "disk_bytes": None}})
    data = json.loads(Path(p).read_text())
    assert data["exact"]["meta"] == {"a": 1, "b": 2} and data["exact"]["with_windows"] is True and data["exact"]["skipped"] is True
    assert data["exact"]["build_seconds"] == 1.5  # None при пропуске не затирает известное значение
    assert data["minhash"] == {"n_records": 3, "disk_bytes": None}


# ----------------------------------------------------------------------------- scripts/04_build_indexes.py


def _load_script():
    spec = importlib.util.spec_from_file_location("build_indexes_04", ROOT / "scripts" / "04_build_indexes.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_cfg(cfg: dict, tmp_path: Path, **fp) -> Path:
    cfg_path = tmp_path / "cfg.yaml"
    data = {k: v for k, v in cfg.items() if not k.startswith("_")}
    data["fingerprint"] = {**data["fingerprint"], **fp}
    cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return cfg_path


def test_script_build_indexes(cfg, records, tmp_path, caplog):
    fdir = Path(cfg["paths"]["functions"])
    write_jsonl(fdir / "protected.jsonl", records)
    windows = [FunctionRecord(**{**r.to_dict(), "id": r.id + "#w", "kind": "window", "code": partial(r.code, 0, 9)}) for r in records[:5]]
    write_jsonl(fdir / "protected_windows.jsonl", windows)
    write_jsonl(fdir / "public_train.jsonl", make_records(30, seed=5, split="public_train", prefix="pub"))
    cfg_path = _write_cfg(cfg, tmp_path)
    mod = _load_script()
    with caplog.at_level(logging.WARNING):
        assert mod.main(["--config", str(cfg_path)]) == 0
    assert any("unkeyed" in m for m in caplog.messages)  # предупреждение о сборке без ключа
    stats_path = Path(cfg["paths"]["results"]) / "index_stats.json"
    stats = json.loads(stats_path.read_text())
    assert set(stats) == {"exact", "winnowing", "minhash"}
    for m in ("exact", "winnowing", "minhash"):
        meta = json.loads((Path(cfg["paths"]["indexes"]) / m / "meta.json").read_text())
        assert meta["with_windows"] is False and meta["build_seconds"] >= 0
        assert stats[m]["memory_bytes"] > 0 and stats[m]["build_seconds"] >= 0 and not stats[m]["skipped"]
        assert stats[m]["n_records"] == len(records) and stats[m]["with_windows"] is False
    assert stats["winnowing"]["meta"]["n_public_fitted"] == 30 and stats["winnowing"]["meta"]["public_filter_fitted"]
    # --with-windows без файла окон: индекс без окон сохраняется (предупреждение), а не пересобирается впустую
    (fdir / "protected_windows.jsonl").rename(fdir / "pw.bak")
    assert mod.main(["--config", str(cfg_path), "--methods", "exact", "--with-windows"]) == 0
    assert json.loads(stats_path.read_text())["exact"]["skipped"] and json.loads(stats_path.read_text())["exact"]["n_records"] == len(records)
    (fdir / "pw.bak").rename(fdir / "protected_windows.jsonl")
    # --with-windows: индексы без окон пересобираются (все методы одинаково), окна записываются в meta
    assert mod.main(["--config", str(cfg_path), "--with-windows"]) == 0
    stats2 = json.loads(stats_path.read_text())
    for m in ("exact", "winnowing", "minhash"):
        assert stats2[m]["n_records"] == len(records) + len(windows) and stats2[m]["with_windows"] is True and not stats2[m]["skipped"]
        assert json.loads((Path(cfg["paths"]["indexes"]) / m / "meta.json").read_text())["with_windows"] is True
    # идемпотентность: повторный запуск (с окнами и без) пропускает готовые индексы, статистика сливается (meta сохраняется)
    assert mod.main(["--config", str(cfg_path), "--methods", "exact,winnowing", "--with-windows"]) == 0
    assert mod.main(["--config", str(cfg_path), "--methods", "exact"]) == 0
    stats3 = json.loads(stats_path.read_text())
    assert stats3["exact"]["skipped"] and stats3["winnowing"]["skipped"] and not stats3["minhash"]["skipped"]
    for m in ("exact", "winnowing", "minhash"):
        assert stats3[m]["with_windows"] is True and stats3[m]["n_records"] == len(records) + len(windows)
        assert stats3[m]["meta"]["n_keys" if m != "minhash" else "lsh_bands"] > 0 and stats3[m]["build_seconds"] is not None
    assert stats3["winnowing"]["meta"]["n_public_fitted"] == 30
    # загрузка и запрос; при равных скорах окна и функции побеждает функция (она добавлена раньше)
    for m in ("exact", "winnowing"):
        idx = load_index(m, cfg)
        assert idx.query(records[2].code, PY).best_id == records[2].id
        assert idx.query(windows[0].code, PY).best_id == records[0].id
    # неизвестный метод — ошибка аргументов; GPU-метод без torch — сообщение и код 1 (run_all.sh не продолжает без M3/M4),
    # код 0 только с --allow-missing-gpu
    with pytest.raises(SystemExit):
        mod.main(["--config", str(cfg_path), "--methods", "bogus"])
    assert mod.main(["--config", str(cfg_path), "--methods", "semantic"]) == 1
    assert mod.main(["--config", str(cfg_path), "--methods", "semantic", "--allow-missing-gpu"]) == 0
    assert mod.main(["--config", str(cfg_path), "--methods", "exact,semantic", "--allow-missing-gpu"]) == 0


def test_script_workers_flag(cfg, tmp_path, monkeypatch):
    """--workers переопределяет cfg.fingerprint.workers только когда задан."""
    cfg_path = _write_cfg(cfg, tmp_path, workers=3)
    mod = _load_script()
    seen: list[int] = []

    def fake_build(name, c, with_windows=False, force=False):
        seen.append(c["fingerprint"]["workers"])
        return {"method": name, "n_records": 0, "build_seconds": 0.0, "memory_bytes": 0, "disk_bytes": 0, "path": "", "skipped": False}

    monkeypatch.setattr(mod, "build_from_corpus", fake_build)
    assert mod.main(["--config", str(cfg_path), "--methods", "exact"]) == 0
    assert mod.main(["--config", str(cfg_path), "--methods", "exact", "--workers", "2"]) == 0
    assert seen == [3, 2]
